"""The CPU-stage section ``cpu`` is readable but never part of the config hash the GPU notebooks carry."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from oryxis.common.config import UNHASHED_SECTIONS, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def folder(tmp_path):
    """A copy of the real config folder without any cpu.yaml (bytes of every other file are kept)."""
    target = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, target)
    for name in UNHASHED_SECTIONS:
        (target / f"{name}.yaml").unlink(missing_ok=True)
    return target


def test_adding_or_changing_cpu_yaml_leaves_the_hash_alone(folder):
    base = load_config(folder)
    (folder / "cpu.yaml").write_text("setting: 1\n", encoding="utf-8")
    with_cpu = load_config(folder)
    assert with_cpu.hash == base.hash
    assert with_cpu.hash_without(("kaggle",)) == base.hash_without(("kaggle",))
    assert with_cpu.get("cpu.setting") == 1  # still readable like any section
    (folder / "cpu.yaml").write_text("setting: 2\n", encoding="utf-8")
    changed = load_config(folder)
    assert changed.hash == base.hash
    assert changed.unhashed_hash != with_cpu.unhashed_hash  # the settings are still on record, separately


def test_other_sections_still_change_the_hash(folder):
    base = load_config(folder)
    assert base.with_overrides({"runtime.base_seed": 1}).hash != base.hash


def test_the_real_config_hash_does_not_depend_on_its_cpu_yaml():
    real = load_config(CONFIG_DIR)
    assert real.hash == real.hash_without(())
    assert real.get("cpu.score_basic.job")
