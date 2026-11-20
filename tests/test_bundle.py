"""Bundle format, verification, selection among several attached bundles, and merging."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd

from finfm.common.schema import ATTEMPTS, BUNDLE_SUFFIX, PARTS_DIR, STATUS, UNIT_ID, Status
from finfm.compute.bundle import (
    BundleWriter,
    consolidate_rows,
    make_job_id,
    merge_bundles,
    part_sources,
    read_consolidated,
    scan_bundles,
    unit_state,
    verify_bundle,
)

JOB = make_job_id("reader", "db1", "llm2")
COMPRESSION = "zstd"
LEVEL = 6


def rows(ids, status=Status.OK.value, attempts=1, value=1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {UNIT_ID: list(ids), STATUS: status, ATTEMPTS: attempts, "value": value}
    )


def write_one(out_dir: Path, frame: pd.DataFrame, *, job: str = JOB, seq: int = 1, carried=()):
    writer = BundleWriter(
        out_dir,
        job,
        "s0000",
        carried=carried,
        parents=[],
        seq=seq,
        meta={"note": "test"},
        compression=COMPRESSION,
        compress_level=LEVEL,
    )
    writer.add_rows(frame)
    return writer.checkpoint(state={"k": 1})


def test_roundtrip_and_verify(tmp_path):
    path = write_one(tmp_path, rows(["a", "b", "c"]))
    result = verify_bundle(path)
    assert result.ok, result.problems
    assert result.manifest["job_id"] == JOB
    assert result.manifest["parts"][0]["rows"] == 3
    scan = scan_bundles([tmp_path], JOB)
    assert len(scan.valid) == 1
    assert list(read_consolidated(scan.valid)[UNIT_ID]) == ["a", "b", "c"]


def test_truncated_bundle_is_rejected(tmp_path):
    path = write_one(tmp_path, rows(["a", "b"]))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    assert not verify_bundle(path).ok


def test_tampered_member_is_rejected_even_with_a_consistent_zip(tmp_path):
    path = write_one(tmp_path, rows(["a", "b"]))
    tampered = tmp_path / "tampered" / path.name
    tampered.parent.mkdir()
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename.startswith(PARTS_DIR):
                payload = payload[:-1] + bytes([payload[-1] ^ 1])
            dst.writestr(info.filename, payload)
    result = verify_bundle(tampered)
    assert not result.ok
    assert any("mismatch" in p for p in result.problems)


def test_selection_among_old_new_corrupt_foreign_and_duplicate(tmp_path):
    old_dir, new_dir, junk_dir = tmp_path / "old", tmp_path / "new", tmp_path / "junk"
    old = write_one(old_dir, rows(["a", "b"]), seq=1)
    old_scan = scan_bundles([old_dir], JOB)
    new = write_one(
        new_dir, rows(["c"]), seq=2, carried=part_sources(old_scan.valid)
    )  # superset of old
    # duplicate of the new bundle under another name and directory
    junk_dir.mkdir()
    shutil.copy(new, junk_dir / ("copy" + BUNDLE_SUFFIX))
    # corrupt bundle
    broken = junk_dir / ("broken" + BUNDLE_SUFFIX)
    broken.write_bytes(new.read_bytes()[:100])
    # bundle of another job
    write_one(junk_dir, rows(["zzz"]), job=make_job_id("reader", "db3", "llm1"))

    scan = scan_bundles([old_dir, new_dir, junk_dir], JOB)
    assert [r.seq for r in scan.valid] == [1, 2]
    assert len(scan.duplicates) == 1
    reasons = {p.name: why for p, why in scan.rejected}
    assert "broken.bundle" in reasons
    assert any("other job" in why for why in reasons.values())
    assert scan.next_seq == 3
    state = unit_state(scan.valid)
    assert list(state[UNIT_ID]) == ["a", "b", "c"]
    assert old.exists() and new.exists()  # inputs are never deleted


def test_selection_ignores_file_names_and_modified_times(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = write_one(a, rows(["a"]), seq=1)
    second = write_one(b, rows(["b"]), seq=2)
    first.rename(first.with_name("zzz_newest_looking" + BUNDLE_SUFFIX))
    second.rename(second.with_name("aaa_oldest_looking" + BUNDLE_SUFFIX))
    scan = scan_bundles([a, b], JOB)
    assert [r.seq for r in scan.valid] == [1, 2]
    assert set(unit_state(scan.valid)[UNIT_ID]) == {"a", "b"}


def test_consolidation_prefers_ok_over_error_and_more_attempts():
    err = rows(["x", "y"], status=Status.ERROR.value, attempts=1)
    err_more = rows(["y"], status=Status.ERROR.value, attempts=2)
    ok = rows(["x"], status=Status.OK.value, attempts=2)
    result = consolidate_rows([err, err_more, ok]).set_index(UNIT_ID)
    assert result.loc["x", STATUS] == Status.OK.value
    assert result.loc["y", STATUS] == Status.ERROR.value
    assert result.loc["y", ATTEMPTS] == 2


def test_identical_parts_are_stored_once(tmp_path):
    first = write_one(tmp_path / "one", rows(["a", "b"]))
    again = write_one(tmp_path / "two", rows(["a", "b"]))
    scan = scan_bundles([tmp_path], JOB)
    assert len(scan.valid) == 1  # same content, same bundle id
    assert len(scan.duplicates) == 1
    assert first.exists() and again.exists()


def test_rolling_checkpoints_leave_exactly_one_bundle(tmp_path):
    writer = BundleWriter(
        tmp_path, JOB, "s0000", carried=[], parents=[], seq=1, meta={}, compression=COMPRESSION,
        compress_level=LEVEL,
    )
    for i in range(4):
        writer.add_rows(rows([f"u{i}"]))
        writer.checkpoint(state={"i": i})
    files = list(tmp_path.glob(f"*{BUNDLE_SUFFIX}"))
    assert len(files) == 1
    assert not list(tmp_path.glob("*.tmp"))
    scan = scan_bundles([tmp_path], JOB)
    assert scan.valid[0].seq == 4
    assert list(read_consolidated(scan.valid)[UNIT_ID]) == ["u0", "u1", "u2", "u3"]


def test_merge_combines_shards_from_different_accounts(tmp_path):
    write_one(tmp_path / "acct1", rows(["a", "b"]), seq=1)
    write_one(tmp_path / "acct2", rows(["c", "d"]), seq=1)
    merged = merge_bundles(
        [tmp_path / "acct1", tmp_path / "acct2"],
        tmp_path / "merged",
        job_id=JOB,
        meta={},
        compression=COMPRESSION,
        compress_level=LEVEL,
    )
    scan = scan_bundles([merged.parent], JOB)
    assert list(read_consolidated(scan.valid)[UNIT_ID]) == ["a", "b", "c", "d"]
