"""Shard runs: N shard processes store disjoint cross-validation fits; an assemble run over their union equals one unsplit run."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
import json

import numpy as np
import pandas as pd
import pytest

from finfm.common import ckpt
from finfm.common.resources import ResourceLog
from finfm.road1 import fit as fit_module
from finfm.stage.road1_run import fit_and_score
from scripts import cpu_session as cs
from tests.test_road1_run import setup  # noqa: F401  (fixture)


def test_every_job_belongs_to_exactly_one_shard():
    for total, shards in ((36, 5), (16, 5), (7, 3), (1, 5)):
        owners = [[k for k in range(shards) if fit_module.shard_owns(i, (k, shards))] for i in range(total)]
        assert all(len(o) == 1 for o in owners)
    assert fit_module.shard_owns(3, None)


def test_a_malformed_shard_is_an_error_not_a_full_run(monkeypatch):
    for bad in ("5/5", "x", "1/0", "-1/3", "2"):
        monkeypatch.setenv(fit_module.SHARD_ENV, bad)
        with pytest.raises(ValueError):
            fit_module.shard_of_env()
    monkeypatch.delenv(fit_module.SHARD_ENV)
    assert fit_module.shard_of_env() is None


def _run(setup, ckpt_dir, monkeypatch, shard=None):  # noqa: F811
    cfg, calendar, dev, test, day0, types = setup
    monkeypatch.setenv(ckpt.ENV, str(ckpt_dir))
    if shard is None:
        monkeypatch.delenv(fit_module.SHARD_ENV, raising=False)
    else:
        monkeypatch.setenv(fit_module.SHARD_ENV, shard)
    return fit_and_score(dev, test, day0, types, calendar, cfg, ("t",), 1, ResourceLog().stage, with_r_report=False)


def test_the_union_of_shard_checkpoints_gives_the_unsplit_result_without_refitting(setup, monkeypatch, tmp_path):  # noqa: F811
    whole = _run(setup, tmp_path / "whole", monkeypatch)
    n_jobs = ckpt.counts()["cvjob"]
    union = tmp_path / "union"
    union.mkdir()
    fitted_by_shard = []
    for k in range(2):
        shard_dir = tmp_path / f"s{k}"
        with pytest.raises(fit_module.ShardPending):
            _run(setup, shard_dir, monkeypatch, shard=f"{k}/2")
        pieces = sorted(shard_dir.glob("cvjob-*.pkl"))
        fitted_by_shard.append({p.name for p in pieces})
        assert not list(shard_dir.glob("final-*.pkl"))  # a shard never refits the final model
        for piece in pieces:
            shutil.copy(piece, union / piece.name)
    assert all(fitted_by_shard) and sum(len(s) for s in fitted_by_shard) == len(set().union(*fitted_by_shard))  # every shard fitted something, no two shards the same piece
    assert sum(len(s) for s in fitted_by_shard) < n_jobs  # a shard stops at its first arm (wgraph): the sheaf's pieces are not there yet
    calls = {"n": 0}
    original = fit_module._fit_fold

    def counting(job):
        calls["n"] += 1
        return original(job)

    monkeypatch.setattr(fit_module, "_fit_fold", counting)
    # the union holds the first arm only; the second arm (sheaf) is completed shard by shard in the same union folder
    for k in range(2):
        monkeypatch.setenv(ckpt.ENV, str(union))
        monkeypatch.setenv(fit_module.SHARD_ENV, f"{k}/2")
        cfg, calendar, dev, test, day0, types = setup
        try:  # the last shard to run finds every fit stored and simply completes; the earlier ones stop with ShardPending
            fit_and_score(dev, test, day0, types, calendar, cfg, ("t",), 1, ResourceLog().stage, with_r_report=False)
        except fit_module.ShardPending:
            pass
    calls["n"] = 0
    result = _run(setup, union, monkeypatch)
    assert calls["n"] == 0  # every fit came from the shards' pieces
    for name in ("fits", "road1_scores"):
        pd.testing.assert_frame_equal(whole.tables[name].reset_index(drop=True), result.tables[name].reset_index(drop=True))
    assert np.allclose(whole.scores["sheaf"].to_numpy(), result.scores["sheaf"].to_numpy(), atol=1e-6)


# ---- the supervisor side --------------------------------------------------------------------------------------------------------------------------

def _bundle(path, label, pieces):
    files = {}
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in pieces.items():
            archive.writestr(f"ckpt/{name}", data)
            import hashlib
            files[f"ckpt/{name}"] = hashlib.sha256(data).hexdigest()
        archive.writestr("MANIFEST.json", json.dumps({"label": label, "stage": "sheaf", "session": 1, "pieces": {"cvjob": len(pieces)}, "files": files,
                                                       "created_utc": "2026-01-01T00:00:00Z"}))


def test_the_assemble_run_unpacks_every_shard_bundle_and_ignores_other_runs(tmp_path, capsys):
    inputs = tmp_path / "in"
    inputs.mkdir()
    _bundle(inputs / "a.ckpt.bundle", "r-sheaf-s0of3", {"cvjob-a.pkl": b"A", "cvjob-b.pkl": b"B"})
    _bundle(inputs / "b.ckpt.bundle", "r-sheaf-s1of3", {"cvjob-c.pkl": b"C"})
    _bundle(inputs / "c.ckpt.bundle", "other-sheaf-s2of3", {"cvjob-z.pkl": b"Z"})
    _bundle(inputs / "d.ckpt.bundle", "r-sheaf-s2of4", {"cvjob-y.pkl": b"Y"})  # a different split of the same run: not mine
    target = tmp_path / "ckpt"
    manifest = cs.open_checkpoint(inputs, "r-sheaf", target, merge_shards=3)
    out = capsys.readouterr().out
    assert sorted(p.name for p in target.glob("*.pkl")) == ["cvjob-a.pkl", "cvjob-b.pkl", "cvjob-c.pkl"]
    assert manifest["pieces"] == {"cvjob": 3} and manifest["session"] == 0
    assert "REJECTED: it belongs to run 'other-sheaf-s2of3'" in out and "REJECTED: it belongs to run 'r-sheaf-s2of4'" in out
    assert "WARNING: shard bundle(s) [2] of 3 are not attached" in out


def test_all_shards_attached_is_reported_and_a_conflicting_piece_is_refused(tmp_path, capsys):
    inputs = tmp_path / "in"
    inputs.mkdir()
    for k in range(2):
        _bundle(inputs / f"{k}.ckpt.bundle", f"r-sheaf-s{k}of2", {f"cvjob-{k}.pkl": b"x"})
    cs.open_checkpoint(inputs, "r-sheaf", tmp_path / "ckpt", merge_shards=2)
    assert "all 2 shard bundles are attached" in capsys.readouterr().out
    _bundle(inputs / "bad.ckpt.bundle", "r-sheaf-s0of2", {"cvjob-0.pkl": b"DIFFERENT"})
    with pytest.raises(SystemExit):
        cs.open_checkpoint(inputs, "r-sheaf", tmp_path / "ckpt2", merge_shards=2)


CHILD_SHARD = """
import os, pickle
d = os.environ['FINFM_CKPT_DIR']; os.makedirs(d, exist_ok=True)
print('shard env seen by the child:', os.environ.get('FINFM_FIT_SHARD'), flush=True)
pickle.dump({'i': 1}, open(f'{d}/cvjob-p.pkl', 'wb'))
"""


def _args(tmp, label, **extra):
    base = dict(label=label, stage="sheaf", bucket="tessera", llm="llm3", modes="m", input_root=str(tmp / "in"), work=str(tmp / "work"), pack=None, runner="unused",
                cutoff_seconds=60.0, tick_seconds=1.0, no_sheaf=False, fetch_vxf=False)
    return argparse.Namespace(**{**base, **extra})


def test_a_finished_shard_run_passes_its_shard_to_the_child_and_still_writes_the_bundle(tmp_path, capsys, monkeypatch):
    (tmp_path / "in").mkdir()
    monkeypatch.setenv("FINFM_FIT_SHARD", "9/9")  # a stale value from outside must not leak into a run without --shard
    status = cs.run_session(_args(tmp_path, "r-sheaf-s1of5", shard="1/5"), [sys.executable, "-c", CHILD_SHARD])
    out = capsys.readouterr().out
    assert status == cs.EXIT_FINISHED
    assert "shard env seen by the child: 1/5" in out and "SHARD CHECKPOINT BUNDLE written" in out
    bundle = tmp_path / "work" / "r-sheaf-s1of5.ckpt.bundle"
    assert json.loads(zipfile.ZipFile(bundle).read("MANIFEST.json"))["pieces"] == {"cvjob": 1}


def test_a_run_without_shard_does_not_inherit_a_stale_shard(tmp_path, capsys, monkeypatch):
    (tmp_path / "in").mkdir()
    monkeypatch.setenv("FINFM_FIT_SHARD", "9/9")
    cs.run_session(_args(tmp_path, "r-sheaf"), [sys.executable, "-c", CHILD_SHARD])
    assert "shard env seen by the child: None" in capsys.readouterr().out


def test_shard_and_merge_cannot_be_combined_and_a_bad_shard_is_refused(tmp_path):
    (tmp_path / "in").mkdir()
    with pytest.raises(SystemExit):
        cs.run_session(_args(tmp_path, "r", shard="0/2", merge_shards=2), [sys.executable, "-c", "pass"])
    with pytest.raises(SystemExit):
        cs.run_session(_args(tmp_path, "r", shard="2/2"), [sys.executable, "-c", "pass"])
