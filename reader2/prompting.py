"""The prompt of the winning POC arm (gate + worked examples), built from the same functions the POC used."""
from __future__ import annotations

import hashlib

from pilot.reader_v3_prompts import system_prompt
from pilot.runner import TokenizerAdapter, asset_names, name_assets, v2_prompt  # noqa: F401  (re-exported on purpose)

# From pilot_reader_v3_poc_gate_shots.yaml (tests compare it with that file).
GLOSSARY = ("Link types: industry = companies in the same industry; rival = direct competitors; supplier = firms that sell inputs "
            "to the named company; customer = firms that buy from it; country = the countries named; commodity = the commodities "
            "named; etf_member = funds that hold the stock.")


def prompt_settings() -> dict:
    """The ``settings`` mapping ``pilot.runner.v2_prompt`` expects."""
    return {"reader_v2": {"system": system_prompt(gate=True, shots=True), "glossary": GLOSSARY, "gate": True}}


def prompt_sha256() -> str:
    """Identity of the prompt text, stored in every bundle: a bundle made with another prompt is never resumed."""
    return hashlib.sha256(prompt_settings()["reader_v2"]["system"].encode("utf-8")).hexdigest()
