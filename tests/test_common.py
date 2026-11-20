"""Config, hashing, seeds and atomic IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finfm.common.config import ConfigError, load_config
from finfm.common.hashing import canonical_json, sha256_file, sha256_hex, stable_hash
from finfm.common.io import atomic_write_bytes, atomic_write_parquet, parquet_bytes_to_frame
from finfm.common.rng import SeedFactory, derive_seed, make_rng

import pandas as pd

from tests.conftest import CONFIG_DIR


def test_config_hash_changes_with_any_value(cfg):
    changed = cfg.with_overrides({"runtime.min_parse_rate": 0.5})
    assert changed.hash != cfg.hash
    assert changed.get("runtime.min_parse_rate") == 0.5
    assert cfg.get("runtime.min_parse_rate") != 0.5  # original untouched


def test_config_hash_is_stable_and_order_independent(cfg):
    assert load_config(CONFIG_DIR).hash == cfg.hash


def test_config_hash_covers_csv_files(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\n")
    (tmp_path / "table.csv").write_text("a,b\n1,2\n")
    first = load_config(tmp_path).hash
    (tmp_path / "table.csv").write_text("a,b\n1,3\n")
    assert load_config(tmp_path).hash != first


def test_unknown_override_and_missing_key_fail(cfg):
    with pytest.raises(ConfigError):
        cfg.with_overrides({"kaggle.no_such_key": 1})
    with pytest.raises(ConfigError):
        cfg.get("kaggle.no_such_key")
    assert cfg.get("kaggle.no_such_key", default=7) == 7


def test_measured_yaml_overrides_but_cannot_add_keys(tmp_path):
    (tmp_path / "kaggle.yaml").write_text("model_load_minutes: 15\n")
    (tmp_path / "measured.yaml").write_text("kaggle:\n  model_load_minutes: 11\n")
    assert load_config(tmp_path).get("kaggle.model_load_minutes") == 11
    (tmp_path / "measured.yaml").write_text("kaggle.typo_minutes: 11\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_canonical_json_and_hashes(tmp_path):
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == sha256_hex(b"abc")
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def test_seeds_are_deterministic_and_unit_specific():
    assert derive_seed("db1", "llm2", "e1", 0) == derive_seed("db1", "llm2", "e1", 0)
    assert derive_seed("db1", "llm2", "e1", 0) != derive_seed("db1", "llm2", "e1", 1)
    a = make_rng("x", 1).random(5)
    b = make_rng("x", 1).random(5)
    assert np.array_equal(a, b)
    with pytest.raises(TypeError):
        derive_seed(1.5)


def test_seed_factory_depends_on_base_seed():
    assert SeedFactory(1).seed("a") != SeedFactory(2).seed("a")
    assert SeedFactory(1).seed("a") == SeedFactory(1).seed("a")


def test_atomic_write_leaves_no_temp_file_on_failure(tmp_path):
    target = tmp_path / "out.bin"
    atomic_write_bytes(target, b"first")

    class Boom(Exception):
        pass

    from finfm.common.io import atomic_path

    with pytest.raises(Boom):
        with atomic_path(target) as tmp:
            tmp.write_bytes(b"half written")
            raise Boom()
    assert target.read_bytes() == b"first"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]


def test_parquet_roundtrip_is_byte_stable(tmp_path):
    frame = pd.DataFrame({"unit_id": ["a", "b"], "v": [1.5, 2.5]})
    atomic_write_parquet(tmp_path / "1.parquet", frame, "zstd")
    atomic_write_parquet(tmp_path / "2.parquet", frame, "zstd")
    assert (tmp_path / "1.parquet").read_bytes() == (tmp_path / "2.parquet").read_bytes()
    back = parquet_bytes_to_frame((tmp_path / "1.parquet").read_bytes())
    assert back.equals(frame)
    assert isinstance(tmp_path, Path)
