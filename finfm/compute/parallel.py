"""Data-parallel engine: one replica per GPU, each in its own process (spec 2.2, Gemma's production path).

Each worker process sets ``CUDA_VISIBLE_DEVICES`` to its index before building the engine, so a replica
sees exactly one GPU. Requests are split into contiguous chunks, one per replica, and the results come
back in request order. ``factory`` must be a picklable top-level callable taking the replica index.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import tempfile
from collections.abc import Callable, Sequence
from typing import Any

from finfm.compute.engine import Engine, GenParams, GenRequest, GenResult

JOIN_TIMEOUT_SECONDS = 30  # how long a replica gets to exit before it is terminated
STOP = "__stop__"
READY = "__ready__"
LOG_TAIL_LINES = 60  # lines of a replica's own output sent back with its error: the engine-core process writes its real error there
REPLICA_LOG = "finfm_replica_{index}.log"


def _tail(path: str, lines: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return ""


def _worker(index: int, factory: Callable[[int], Engine], inbox, outbox, set_devices: bool) -> None:
    if set_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    # a replica's output (and its engine-core child's) never reaches the notebook log, so it is kept in a file and its tail travels with any error
    log_path = os.path.join(tempfile.gettempdir(), REPLICA_LOG.format(index=index))
    log = open(log_path, "w", buffering=1, encoding="utf-8")  # noqa: SIM115 (lives as long as the process)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    try:
        engine = factory(index)
        outbox.put((READY, index, None))
        while True:
            message = inbox.get()
            if message == STOP:
                break
            chunk_id, requests, params = message
            outbox.put((chunk_id, index, engine.generate(requests, params)))
        engine.close()
    except BaseException as exc:  # report instead of dying silently
        outbox.put(("error", index, f"{exc!r}\n--- last {LOG_TAIL_LINES} lines of replica {index}'s own output ---\n{_tail(log_path, LOG_TAIL_LINES)}"))


class DataParallelEngine(Engine):
    name = "data_parallel"

    def __init__(self, factory: Callable[[int], Engine], replicas: int, *, set_devices: bool = True) -> None:
        ctx = mp.get_context("spawn")
        self.replicas = replicas
        self.closed = False
        self.outbox = ctx.Queue()
        self.inboxes = [ctx.Queue() for _ in range(replicas)]
        # not daemonic: vLLM's V1 engine starts its own engine-core subprocess, and a daemonic process may not have children
        self.procs = [ctx.Process(target=_worker, args=(i, factory, self.inboxes[i], self.outbox, set_devices), daemon=False)
                      for i in range(replicas)]
        atexit.register(self.close)  # without the daemon flag, a parent that dies early must still stop its workers
        for proc in self.procs:  # one replica at a time: two vLLM engines starting together can clash on ports and cache files
            proc.start()
            tag, index, payload = self.outbox.get()
            if tag == "error":
                self.close()
                raise RuntimeError(f"replica {index} failed to start: {payload}")

    def generate(self, requests: Sequence[GenRequest], params: GenParams) -> list[GenResult]:
        size = -(-len(requests) // self.replicas) if requests else 0
        chunks = [list(requests[i * size : (i + 1) * size]) for i in range(self.replicas)]
        active = [i for i, c in enumerate(chunks) if c]
        for i in active:
            self.inboxes[i].put((i, chunks[i], params))
        results: dict[int, list[GenResult]] = {}
        for _ in active:
            tag, index, payload = self.outbox.get()
            if tag == "error":
                raise RuntimeError(f"replica {index} failed: {payload}")
            results[tag] = payload
        return [r for i in active for r in results[i]]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for inbox in self.inboxes:
            inbox.put(STOP)
        for proc in self.procs:
            if proc.pid is None:  # never started (an earlier replica failed first)
                continue
            proc.join(timeout=JOIN_TIMEOUT_SECONDS)
            if proc.is_alive():
                proc.terminate()


def describe(engine: Any) -> dict[str, Any]:
    """Engine facts for metadata (versions, detected API choices); empty for engines that have none."""
    return engine.describe() if hasattr(engine, "describe") else {"engine": getattr(engine, "name", "unknown")}
