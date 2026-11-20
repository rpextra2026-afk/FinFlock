"""The framework's part 1 result (dial maps and the fitted sheaf) survives being written to a file and read back."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch

from oryxis.framework.road1_stage import Road1Predictor
from oryxis.real.framework_fit import DevFit
from oryxis.road1.arms import SheafModel


def test_dev_fit_round_trips_through_a_pickle(cfg):
    torch.manual_seed(0)
    model = SheafModel(2, 1, 3, [True, False])
    predictor = Road1Predictor("sheaf", model, 0.1, np.ones(3), pd.DataFrame({"a": [1]}), cfg)
    dev = DevFit({}, predictor, 5, 4, {"grounding": 0.1})
    dials = {"e1": pd.DataFrame({"ret": [0.1]})}
    restored = pickle.loads(pickle.dumps({"meta": {"arm": "sheaf"}, "dev": dev, "dials": dials}, protocol=pickle.HIGHEST_PROTOCOL))
    got = restored["dev"].road1
    assert got.arm == "sheaf" and got.grounding == 0.1 and restored["dev"].n_graphs == 4
    for (name, a), (_, b) in zip(model.state_dict().items(), got.model.state_dict().items(), strict=True):
        assert torch.equal(a, b), name
    assert restored["dials"]["e1"].equals(dials["e1"])


def test_a_saved_dev_fit_is_checked_on_the_fit_settings_only():
    from scripts.run_cpu import dev_fit_problems

    now = {"bucket": "db3", "llm": "llm2", "reader_mode": "m", "arm": "sheaf", "config_hash": "new-full", "fit_config_hash": "fit-A"}
    same = {**now, "config_hash": "old-full"}  # only the reporting settings changed: allowed
    assert dev_fit_problems(same, now) == ([], [])
    assert dev_fit_problems({**same, "fit_config_hash": "fit-B"}, now)[0]  # the fit settings changed: refused
    assert dev_fit_problems({**same, "reader_mode": "other"}, now)[0]
    old = {k: v for k, v in same.items() if k != "fit_config_hash"}  # a file from before the fit hash existed
    problems, warnings = dev_fit_problems(old, now)
    assert problems == [] and "cannot be verified" in warnings[0]
