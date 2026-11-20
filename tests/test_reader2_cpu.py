"""The CPU stage reads the final reader's tape as it is (no reparse)."""
import json

import pandas as pd

from oryxis.common.config import load_config
from oryxis.reader.salvage import config_for_mode, prepare_reader_tape
from oryxis.real.tapes import is_reader_v2


def test_reader2_tape_is_used_as_stored_and_not_reparsed():
    cfg = load_config("config")
    run_cfg, mode = config_for_mode(cfg, "salvaged_no_certainty")
    prefix = str(cfg.get("reader.ref_prefix"))
    offered = json.dumps([{"node_id": "n1", "priority": 1.0}])
    pin = {"ref": f"{prefix}1", "node_id": "n1", "direction": "up", "value_effect_pct": 1.0, "magnitude": "small", "role": "direct"}
    tape = pd.DataFrame({"unit_id": ["u1", "u2"], "status": ["ok", "skipped"], "raw_output": ["{not what the old parser expects}", None],
                         "parse_ok": [True, None], "n_pins_given": [1, 0], "reading_pins_json": [json.dumps([pin]), None], "event_type_final": ["asset_event", None]})
    assert is_reader_v2(tape)
    out, report, fill = prepare_reader_tape(tape, {"u1": offered, "u2": "[]"}, run_cfg, mode)
    assert fill is None and out["parse_ok"].tolist() == [True, False]
    assert out.loc[0, "reading_pins_json"] == tape.loc[0, "reading_pins_json"] and "reader2" in report["reader"]
