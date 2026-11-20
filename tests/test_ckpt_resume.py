"""Stored pieces of work make a restarted fit skip what is finished and give exactly the same numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfm.common import ckpt
from finfm.common.resources import ResourceLog
from finfm.road1 import fit as fit_module
from finfm.stage.road1_run import fit_and_score
from tests.test_road1_run import setup  # noqa: F401  (fixture)


def test_a_restarted_fit_reuses_stored_fits_and_gives_identical_tables(setup, monkeypatch, tmp_path, capfd):  # noqa: F811
    cfg, calendar, dev, test, day0, types = setup
    monkeypatch.setenv(ckpt.ENV, str(tmp_path / "ckpt"))
    calls = {"n": 0}
    original = fit_module._fit_fold

    def counting(job):
        calls["n"] += 1
        return original(job)

    monkeypatch.setattr(fit_module, "_fit_fold", counting)
    first = fit_and_score(dev, test, day0, types, cfg, ("t",), 1, ResourceLog().stage, with_r_report=False) if False else \
        fit_and_score(dev, test, day0, types, calendar, cfg, ("t",), 1, ResourceLog().stage, with_r_report=False)
    ran_first = calls["n"]
    assert ran_first > 0 and ckpt.counts().get("cvjob") == ran_first and ckpt.counts().get("final", 0) >= 2
    calls["n"] = 0
    second = fit_and_score(dev, test, day0, types, calendar, cfg, ("t",), 1, ResourceLog().stage, with_r_report=False)
    assert calls["n"] == 0  # every fit came from the checkpoint
    for name in ("fits", "road1_scores"):
        pd.testing.assert_frame_equal(first.tables[name].reset_index(drop=True), second.tables[name].reset_index(drop=True))
    assert np.allclose(first.scores["sheaf"].to_numpy(), second.scores["sheaf"].to_numpy(), atol=1e-6)
    out = capfd.readouterr().out
    assert "already stored in the checkpoint" in out and "(from checkpoint)" in out and "final model taken from the checkpoint" in out


def test_without_the_environment_variable_nothing_is_stored(setup, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.delenv(ckpt.ENV, raising=False)
    ckpt.put("x", 1, "a")
    assert ckpt.get("x", "a") is None and ckpt.counts() == {}
