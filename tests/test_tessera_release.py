"""Checks of the shipped dataset, its integrity, and the held-out split guard."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "tessera"


def test_dataset_parts_match_manifest_and_have_no_embedded_schema_metadata():
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    actual = {p.relative_to(DATA).as_posix() for p in DATA.rglob("*.parquet") if p.is_file()}
    assert actual == {entry["path"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        path = DATA / entry["path"]
        assert path.stat().st_size == entry["bytes"] < 50_000_000
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert pq.ParquetFile(path).metadata.num_rows == entry["rows"]
        assert pq.read_schema(path).metadata is None
    for entry in manifest["tables"]:
        table = pq.ParquetDataset(DATA / entry["path"])
        assert sum(fragment.count_rows() for fragment in table.fragments) == entry["rows"]
    assert sum(entry["rows"] for entry in manifest["tables"] if entry["path"] in {
        "financial/items.parquet", "geopolitical/items.parquet",
        "social_reddit/items.parquet", "geo_supplement_records/records.parquet",
    }) == 431817


def test_shipped_loader_separates_training_from_heldout_records():
    import importlib.util

    spec = importlib.util.spec_from_file_location("tessera_loader", DATA / "loader" / "tessera_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    train = loader.load_items(DATA, categories=("geopolitical",), with_text=False)
    assert set(train["partition"]) == {"train"} and set(train["split"]) == {"dev"}
    heldout = loader.load_heldout(DATA, categories=("geopolitical",), split="test")
    assert set(heldout["partition"]) == {"heldout_test"}
    assert not set(train["item_id"]) & set(heldout["item_id"])
    with pytest.raises(loader.SplitGuardError):
        loader.guard_training_batch(heldout)


def test_reader_adapter_includes_both_supplement_text_sources(monkeypatch):
    from finflock.data.buckets.package import _supplement_events

    records = pd.DataFrame([
        {"record_id": key, "event_id": "event-" + key, "available_utc": pd.Timestamp("2020-01-02", tz="UTC"),
         "headline": "", "source_name": "source", "exposure_ciks": []}
        for key in ["approved", "attributed"]
    ])
    frames = {
        "records.parquet": records,
        "text.parquet": pd.DataFrame({"record_id": ["approved"], "text": ["approved text"]}),
        "public_text.parquet": pd.DataFrame({"record_id": ["attributed"], "text": ["attributed text"]}),
        "pins.parquet": pd.DataFrame(columns=["record_id", "pin_type", "pin_id", "confidence"]),
    }
    monkeypatch.setattr(pd, "read_parquet", lambda path: frames[Path(path).name].copy())
    result = _supplement_events(Path("local-input"))
    assert dict(zip(result["event_id"], result["text"])) == {
        "event-approved": "approved text", "event-attributed": "attributed text",
    }
