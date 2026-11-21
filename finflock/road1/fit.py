"""Fitting the road-1 methods on dev data only (spec 8.2).

Labels arrive as ``EventGraph.y`` (set by the caller); this module never imports the labels store.
Every method gets the same folds, the same dial standardisation and the same budget. Event losses are
weighted ``1 / |F_e|`` so each event counts once, whatever its size.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import torch

from finflock.common import ckpt
from finflock.common.config import Config
from finflock.common.cv import time_blocked_folds
from finflock.common.rng import derive_seed, make_rng
from finflock.common.timing import TradingCalendar
from finflock.road1.arms import (
    Batch,
    GraphModel,
    MatrixModel,
    SheafModel,
    collate,
    matrix_design,
)
from finflock.road1.graph import EventGraph

Model = GraphModel | SheafModel
SHARD_ENV = "FINFLOCK_FIT_SHARD"  # "k/N": this process runs only the cross-validation fits whose position in the job list is k modulo N
SHARD_PATTERN = re.compile(r"^(\d+)/(\d+)$")
BYTES_PER_VALUE = 8  # float64
MEMORY_CAP_KEY = "cpu.resources.max_batch_bytes"  # dense Laplacian blocks of one micro-batch; absent = no cap


def dial_scale(graphs: Sequence[EventGraph], floor: float) -> np.ndarray:
    """Standard deviation of each dial label over all free nodes (floored)."""
    labels = np.concatenate([g.y for g in graphs if g.y is not None and g.n_free])
    return np.maximum(np.nanstd(labels, axis=0), floor)


def standardise(graphs: Sequence[EventGraph], scale: np.ndarray) -> list[EventGraph]:
    """Divide pinned values and labels by the dial scale."""
    out = []
    for g in graphs:
        scaled = EventGraph(
            g.event_id, g.nodes, g.n_pinned, g.src, g.dst, g.link_type, g.weight,
            g.s / scale, g.group_probs, g.truncated, None if g.y is None else g.y / scale, g.meta,
        )
        out.append(scaled)
    return out


def usable(graphs: Sequence[EventGraph]) -> list[EventGraph]:
    """Graphs with at least one labelled free node."""
    return [g for g in graphs if g.y is not None and g.n_free > 0 and np.isfinite(g.y).all(axis=1).any()]


def batch_loss(pred: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean over events of ``(1 / |F_e|) sum_j sum_dial (pred - y)^2`` in standardised units."""
    squared = ((pred - batch.y) ** 2).sum(dim=-1) * batch.mask
    return (squared.sum(dim=1) / batch.n_real_free).mean()


# --- A2: closed-form ridge --------------------------------------------------------------------


def fit_matrix(graphs: Sequence[EventGraph], rung: int, ridge: float, n_types: int) -> MatrixModel:
    """Ridge least squares for the one-hop table; rows weighted ``1 / |F_e|``."""
    data = usable(graphs)
    d = data[0].s.shape[1]
    if rung == 1:
        rows, targets, weights = [], [], []
        for g in data:
            phi = matrix_design(g, 1, n_types)  # (n_free, T, d)
            ok = np.isfinite(g.y).all(axis=1)
            for a in range(d):
                rows.append(phi[ok][:, :, a])
                targets.append(g.y[ok, a])
                weights.append(np.full(int(ok.sum()), 1.0 / ok.sum()))
        X, y, w = np.vstack(rows), np.concatenate(targets), np.concatenate(weights)
        return MatrixModel(1, _ridge(X, y, w, ridge), n_types)
    finite = [np.isfinite(g.y).all(axis=1) for g in data]
    X = np.vstack([matrix_design(g, rung, n_types)[ok] for g, ok in zip(data, finite, strict=True)])
    Y = np.vstack([g.y[ok] for g, ok in zip(data, finite, strict=True)])
    w = np.concatenate([np.full(int(ok.sum()), 1.0 / ok.sum()) for ok in finite])
    theta = np.stack([_ridge(X, Y[:, a], w, ridge) for a in range(d)])
    return MatrixModel(rung, theta, n_types)


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    weighted = X * w[:, None]
    gram = X.T @ weighted + ridge * np.eye(X.shape[1])
    return np.linalg.solve(gram, weighted.T @ y)


# --- A1 and A3: gradient training ---------------------------------------------------------------


@dataclass
class TrainResult:
    model: Model
    best_step: int
    best_val: float | None
    history: list[tuple[int, float, float | None]] = field(default_factory=list)
    stopped_early: bool = False


def size_buckets(graphs: Sequence[EventGraph]) -> list[list[EventGraph]]:
    """Graphs grouped by power-of-two size of the free set, so padding never more than doubles a graph (the solve is cubic in size)."""
    groups: dict[int, list[EventGraph]] = {}
    for g in graphs:
        groups.setdefault(max(g.n_free, 1).bit_length(), []).append(g)
    return list(groups.values())


def micro_batches(graphs: Sequence[EventGraph], max_bytes: int | None) -> list[list[EventGraph]]:
    """Size buckets (padding never more than doubles a graph), further cut so that the dense Laplacian blocks of each part
    ``events x nodes x nodes x dials x dials`` stay below ``max_bytes``. The events and their losses are unchanged; only how many go
    through one solve at a time."""
    parts = size_buckets(graphs)
    if max_bytes is None:
        return parts
    out = []
    for part in parts:
        d = part[0].s.shape[1]
        chunk, pinned, free = [], 0, 0
        for g in part:
            wider_pinned, wider_free = max(pinned, g.n_pinned), max(free, g.n_free)
            if chunk and (len(chunk) + 1) * (wider_pinned + wider_free) ** 2 * d * d * BYTES_PER_VALUE > max_bytes:
                out.append(chunk)
                chunk, wider_pinned, wider_free = [], g.n_pinned, g.n_free
            chunk.append(g)
            pinned, free = wider_pinned, wider_free
        out.append(chunk)
    return out


def mean_loss(model: Model, graphs: Sequence[EventGraph], grounding: float, min_prob: float, max_bytes: int | None = None) -> torch.Tensor:
    """Mean event loss; identical to one padded batch, computed per size bucket (and memory-bounded micro-batch) to avoid padding small graphs to the largest."""
    total = sum(len(part) * batch_loss(model.predict_batch(batch, grounding, min_prob), batch)
                for part in micro_batches(graphs, max_bytes) for batch in [collate(part)])
    return total / len(graphs)


def accumulate_gradients(model: Model, graphs: Sequence[EventGraph], grounding: float, min_prob: float, max_bytes: int | None) -> float:
    """Back-propagate the mean event loss of ``graphs`` micro-batch by micro-batch (each weighted by its share of the events) and return the
    loss value. The summed gradients equal those of one pass over all the events; peak memory follows the micro-batch, not the batch."""
    total = 0.0
    for part in micro_batches(graphs, max_bytes):
        tensors = collate(part)
        part_loss = batch_loss(model.predict_batch(tensors, grounding, min_prob), tensors) * (len(part) / len(graphs))
        part_loss.backward()
        total += float(part_loss.detach())
    return total


def evaluate_loss(
    model: Model, graphs: Sequence[EventGraph], grounding: float, min_prob: float, chunk: int, max_bytes: int | None = None
) -> float:
    """Mean event loss over many graphs without gradients, in chunks to bound padding."""
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(graphs), chunk):
            part = graphs[start : start + chunk]
            total += float(mean_loss(model, part, grounding, min_prob, max_bytes)) * len(part)
            count += len(part)
    return total / count


def train_model(
    model: Model,
    train: Sequence[EventGraph],
    val: Sequence[EventGraph] | None,
    *,
    grounding: float,
    penalty_mu: float,
    cfg: Config,
    seed_parts: tuple,
    fixed_steps: int | None = None,
) -> TrainResult:
    """Adam with early stopping on ``val`` (or a fixed number of steps when ``val`` is ``None``)."""
    opt_cfg = cfg.get("road1.optimiser")
    min_prob = float(cfg.get("road1.min_group_prob"))
    cap = cfg.get(MEMORY_CAP_KEY, None)
    cap = None if cap is None else int(cap)
    train, val = usable(train), usable(val) if val is not None else None
    torch.manual_seed(derive_seed(*seed_parts))
    rng = make_rng(*seed_parts, "batches")
    optimiser = torch.optim.Adam(model.parameters(), lr=float(opt_cfg["lr"]))
    steps = fixed_steps if fixed_steps is not None else int(opt_cfg["max_steps"])
    batch = min(int(opt_cfg["batch_events"]), len(train))
    best, best_step, best_state = np.inf, 0, copy.deepcopy(model.state_dict())
    history: list[tuple[int, float, float | None]] = []
    stopped = False
    for step in range(steps):
        picks = rng.choice(len(train), size=batch, replace=False)
        step_graphs = [train[i] for i in picks]
        optimiser.zero_grad()
        loss_value = accumulate_gradients(model, step_graphs, grounding, min_prob, cap)
        penalty = penalty_mu / len(train) * model.penalty()
        if penalty.requires_grad:
            penalty.backward()
        loss_value += float(penalty.detach())
        optimiser.step()
        if val is not None and step % int(opt_cfg["eval_every"]) == 0:
            current = evaluate_loss(model, val, grounding, min_prob, int(opt_cfg["batch_events"]), cap)
            history.append((step, loss_value, current))
            if current < best:
                best, best_step, best_state = current, step, copy.deepcopy(model.state_dict())
            elif step - best_step > int(opt_cfg["patience"]):
                stopped = True
                break
    if val is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, best_step, None if val is None else best, history, stopped)


def new_model(arm: str, cfg: Config) -> Model:
    n_types = len(cfg.get("road1.link_types"))
    d = len(cfg.get("road1.dial_names"))
    if arm == "wgraph":
        return GraphModel(n_types, d)
    n_groups = len(cfg.get("causes.groups"))
    symmetric = [t in set(cfg.get("road1.symmetric_types")) for t in cfg.get("road1.link_types")]
    return SheafModel(n_types, n_groups, d, symmetric)


@dataclass
class CVResult:
    arm: str
    grounding: float
    penalty_mu: float
    val_loss: float
    best_steps: list[int]


def cross_validate(
    arm: str,
    graphs: Sequence[EventGraph],
    dates,
    calendar: TradingCalendar,
    cfg: Config,
    *,
    grounding: float,
    penalty_mu: float,
    seed_parts: tuple = (),
) -> CVResult:
    """Mean validation loss over the time-blocked folds for one configuration."""
    cv = cfg.get("road1.cv")
    folds = time_blocked_folds(dates, int(cv["folds"]), int(cv["embargo_sessions"]), calendar)
    losses, steps = [], []
    for fold, (train_idx, val_idx) in enumerate(folds):
        if not usable([graphs[i] for i in train_idx]) or not usable([graphs[i] for i in val_idx]):
            continue  # a fold with no labelled graphs on either side carries no information
        result = train_model(
            new_model(arm, cfg), [graphs[i] for i in train_idx], [graphs[i] for i in val_idx],
            grounding=grounding, penalty_mu=penalty_mu, cfg=cfg, seed_parts=(*seed_parts, arm, fold),
        )
        losses.append(result.best_val)
        steps.append(result.best_step)
    if not losses:
        raise ValueError("no cross-validation fold has labelled graphs on both sides")
    return CVResult(arm, grounding, penalty_mu, float(np.mean(losses)), steps)


class ShardPending(BaseException):
    """Raised by ``select_and_fit`` in shard mode once this shard's own fits are stored and the rest belong to other shards.

    A BaseException on purpose: the generic ``except Exception`` handlers on the way up must not swallow it. The scripts that start a fit catch it
    and end the run cleanly (the stored fits are the shard's result)."""


def shard_of_env() -> tuple[int, int] | None:
    """(k, N) from FINFLOCK_FIT_SHARD, or None when the variable is not set. A malformed value is an error, never a silent full run."""
    value = os.environ.get(SHARD_ENV, "").strip()
    if not value:
        return None
    match = SHARD_PATTERN.match(value)
    if not match or not 0 <= int(match.group(1)) < int(match.group(2)):
        raise ValueError(f"{SHARD_ENV}={value!r} is not 'k/N' with 0 <= k < N")
    return int(match.group(1)), int(match.group(2))


def shard_owns(index: int, shard: tuple[int, int] | None) -> bool:
    """Whether the fit at ``index`` of the full job list belongs to this shard (always, without a shard). The index is the position among all jobs,
    stored or not, so every shard cuts the same list the same way."""
    return shard is None or index % shard[1] == shard[0]


_SHARED: dict = {}


def _init_worker(graphs, cfg, threads: int | None = 1) -> None:
    """Share the graphs and config with a fitting process; one torch thread each, since the processes fill the cores."""
    if threads:
        torch.set_num_threads(threads)
    _SHARED.update(graphs=graphs, cfg=cfg)


def _fit_fold(job) -> tuple[float, int]:
    """One (configuration, fold) fit: validation loss at the best step and that step."""
    arm, grounding, penalty, _fold, train_idx, val_idx, seed_parts = job
    graphs, cfg = _SHARED["graphs"], _SHARED["cfg"]
    result = train_model(new_model(arm, cfg), [graphs[i] for i in train_idx], [graphs[i] for i in val_idx],
                         grounding=grounding, penalty_mu=penalty, cfg=cfg, seed_parts=seed_parts)
    return result.best_val, result.best_step


JOB_HEAD = 4  # (arm, grounding, penalty, fold) at the start of a fit job
JOB_TRAIN_AT = 4
JOB_VALID_AT = 5


def _job_key(job: tuple) -> tuple:
    return job[:JOB_HEAD], np.asarray(job[JOB_TRAIN_AT]).tolist(), np.asarray(job[JOB_VALID_AT]).tolist()


def select_and_fit(
    arm: str,
    graphs: Sequence[EventGraph],
    dates,
    calendar: TradingCalendar,
    cfg: Config,
    *,
    seed_parts: tuple = (),
    on_config: Callable[[CVResult], None] | None = None,
    workers: int = 1,
    grid: Sequence[tuple[float, float]] | None = None,
) -> tuple[Model, CVResult, list[CVResult]]:
    """Grid over grounding (and sheaf penalty), pick the best by CV loss, refit on all dev data.

    ``grid``: the (grounding, penalty) points to evaluate instead of the configured grids (an ablation run that keeps the point Full chose evaluates one point,
    which still gives the number of steps by time-blocked CV).

    Every evaluated configuration is reported through ``on_config`` so the configuration count that
    is printed beside each headline can be kept exactly. With ``workers`` above 1 the (configuration, fold)
    fits run in separate processes; each fit has its own seed, so the numbers equal the sequential ones.
    """
    groundings = [float(x) for x in cfg.get("road1.grounding_lambda_grid")]
    penalties = [float(x) for x in cfg.get("road1.sheaf_penalty_grid")] if arm == "sheaf" else [0.0]
    cv = cfg.get("road1.cv")
    folds = [(fold, tr, va) for fold, (tr, va) in enumerate(time_blocked_folds(dates, int(cv["folds"]), int(cv["embargo_sessions"]), calendar))
             if usable([graphs[i] for i in tr]) and usable([graphs[i] for i in va])]   # a fold with no labelled graphs carries no information
    if not folds:
        raise ValueError("no cross-validation fold has labelled graphs on both sides")
    grid = [(float(g), float(p)) for g, p in grid] if grid is not None else [(grounding, penalty) for grounding in groundings for penalty in penalties]
    jobs = [(arm, grounding, penalty, fold, tr, va, (*seed_parts, arm, fold)) for grounding, penalty in grid for fold, tr, va in folds]
    scope = (cfg.hash, arm, tuple(seed_parts), hashlib.sha256("|".join(g.event_id for g in graphs).encode("utf-8")).hexdigest())
    fitted: list = [None] * len(jobs)
    todo = []
    shard = shard_of_env()
    for index, job in enumerate(jobs):
        hit = ckpt.get("cvjob", scope, *_job_key(job))
        if hit is None:
            if shard_owns(index, shard):
                todo.append(index)
        else:
            fitted[index] = hit
    if shard is not None:
        owned = sum(shard_owns(i, shard) for i in range(len(jobs)))
        print(f"[fit] SHARD {shard[0]}/{shard[1]}: this shard owns {owned} of the {len(jobs)} fits of {arm}", flush=True)
    print(f"[fit] {arm}: {len(grid)} configuration(s) x {len(folds)} fold(s) = {len(jobs)} fits; {sum(v is not None for v in fitted)} already stored in the checkpoint, "
          f"{len(todo)} to run (workers {workers})", flush=True)
    started = time.time()

    def finished(index: int, value: tuple, how: str) -> None:
        fitted[index] = value
        ckpt.put("cvjob", value, scope, *_job_key(jobs[index])) if how == "run" else None
        done = sum(v is not None for v in fitted)
        _, grounding, penalty, fold = jobs[index][:JOB_HEAD]
        print(f"[fit] {arm} grounding={grounding} penalty={penalty} fold={fold}: val_loss={value[0]:.6f} best_steps={value[1]} "
              f"| {done}/{len(jobs)} fits done, {time.time() - started:.0f} s in this session", flush=True)

    for index in range(len(jobs)):
        if fitted[index] is not None:
            _, grounding, penalty, fold = jobs[index][:JOB_HEAD]
            print(f"[fit] {arm} grounding={grounding} penalty={penalty} fold={fold}: val_loss={fitted[index][0]:.6f} (from checkpoint)", flush=True)
    if workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(list(graphs), cfg)) as pool:
            futures = {pool.submit(_fit_fold, jobs[index]): index for index in todo}
            for future in as_completed(futures):
                finished(futures[future], future.result(), "run")
    elif todo:
        _init_worker(list(graphs), cfg, threads=None)
        for index in todo:
            finished(index, _fit_fold(jobs[index]), "run")
    if shard is not None and any(v is None for v in fitted):  # the other shards' fits are not here: this shard's work is done, the assemble run does the rest
        missing = sum(v is None for v in fitted)
        print(f"[fit] SHARD {shard[0]}/{shard[1]} DONE for {arm}: {len(jobs) - missing}/{len(jobs)} fits are stored; {missing} belong to other shards. "
              f"Attach every shard's checkpoint bundle to the assemble run.", flush=True)
        raise ShardPending(arm)
    results = []
    for index, (grounding, penalty) in enumerate(grid):
        part = fitted[index * len(folds) : (index + 1) * len(folds)]
        results.append(CVResult(arm, grounding, penalty, float(np.mean([v for v, _ in part])), [s for _, s in part]))
        print(f"[fit] {arm} CONFIG grounding={grounding} penalty={penalty}: cv_loss={results[-1].val_loss:.6f} (mean over {len(part)} folds)", flush=True)
        if on_config:
            on_config(results[-1])
    best = min(results, key=lambda r: r.val_loss)
    steps = max(int(np.mean(best.best_steps)), 1)
    print(f"[fit] {arm} BEST grounding={best.grounding} penalty={best.penalty_mu} cv_loss={best.val_loss:.6f}; final refit on all dev data for {steps} steps", flush=True)
    stored = ckpt.get("final", scope, best.grounding, best.penalty_mu, steps)
    if stored is not None:
        print(f"[fit] {arm} final model taken from the checkpoint", flush=True)
        return stored, best, results
    began = time.time()
    final = train_model(
        new_model(arm, cfg), graphs, None, grounding=best.grounding, penalty_mu=best.penalty_mu,
        cfg=cfg, seed_parts=(*seed_parts, arm, "final"), fixed_steps=steps,
    )
    ckpt.put("final", final.model, scope, best.grounding, best.penalty_mu, steps)
    print(f"[fit] {arm} final refit done in {time.time() - began:.0f} s", flush=True)
    return final.model, best, results


def predict_graph(model: Model, graph: EventGraph, grounding: float, scale: np.ndarray, cfg: Config) -> np.ndarray:
    """Predicted free-node dials in original units."""
    with torch.no_grad():
        standardised = standardise([graph], scale)[0]
        raw = model.predict(standardised, grounding, float(cfg.get("road1.min_group_prob")))
    return raw.numpy() * scale


def oos_r2(y: np.ndarray, pred: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    """Out-of-sample R2 per dial against the training-mean forecast (Gu-Kelly-Xiu style)."""
    sse = ((y - pred) ** 2).sum(axis=0)
    sst = ((y - train_mean) ** 2).sum(axis=0)
    return 1.0 - sse / sst

