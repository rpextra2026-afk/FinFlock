"""The bundle: ONE file that carries a job's outputs, checkpoint state and diagnostics.

A Kaggle session leaves exactly one ``.bundle`` file. Upload it as a private dataset, attach it to the
next session and the worker continues from it (experiment configuration).

Format: a zip archive with fixed timestamps. ``MANIFEST.json`` lists every other member with its
sha256 and size. Parquet parts are content addressed (``part-<sha256[:16]>.parquet``), so the same
rows attached twice are stored once. Selection never trusts filenames or modified times: every
member hash is verified, the union of completed units of all valid bundles of the job is used.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from finfm.common.hashing import HASH_CHUNK_BYTES, canonical_json, sha256_hex, stable_hash
from finfm.common.io import atomic_path, frame_to_parquet_bytes, parquet_bytes_to_frame
from finfm.common.schema import (
    ATTEMPTS,
    BUNDLE_FORMAT_VERSION,
    BUNDLE_PREFIX,
    BUNDLE_SUFFIX,
    DIAGNOSTICS_DIR,
    JOB_ID_SEPARATOR,
    LOGS_DIR,
    MANIFEST_NAME,
    PART_NAME_HASH_CHARS,
    PART_PREFIX,
    PART_SUFFIX,
    PARTS_DIR,
    SHARD_ID_ALL,
    STATE_NAME,
    STATUS,
    STATUS_RANK,
    UNIT_ID,
)

ZIP_FIXED_DATE = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE = 0o644
ZIP_MODE_SHIFT = 16
SEQ_WIDTH = 5
FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


class BundleError(Exception):
    """Raised when a bundle cannot be written or fails verification."""


def make_job_id(job: str, bucket: str, llm: str) -> str:
    """Job identity, e.g. ``reader__db1__llm2``. Rivals use job names like ``rival_basic``."""
    return JOB_ID_SEPARATOR.join([job, bucket, llm])


def bundle_filename(job_id: str, shard_id: str, seq: int) -> str:
    """File name of a bundle: ``finfm_<job>_<shard>_<seq>.bundle`` with unsafe characters removed."""
    job = FILENAME_UNSAFE.sub("_", job_id)
    shard = FILENAME_UNSAFE.sub("_", shard_id)
    return f"{BUNDLE_PREFIX}_{job}_{shard}_{seq:0{SEQ_WIDTH}d}{BUNDLE_SUFFIX}"


# --- parts -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PartSource:
    """A parquet part and a way to read its bytes (from memory or from another bundle)."""

    name: str
    sha256: str
    rows: int
    size: int
    open: Callable[[], BinaryIO] = field(repr=False, compare=False)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, compression: str) -> PartSource:
        data = frame_to_parquet_bytes(frame, compression)
        sha = sha256_hex(data)
        name = f"{PART_PREFIX}{sha[:PART_NAME_HASH_CHARS]}{PART_SUFFIX}"
        return cls(name, sha, len(frame), len(data), lambda: io.BytesIO(data))

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        with self.open() as handle:
            return parquet_bytes_to_frame(handle.read(), columns)


def _member_opener(path: Path, member: str) -> Callable[[], BinaryIO]:
    def opener() -> BinaryIO:
        with zipfile.ZipFile(path) as archive:
            return io.BytesIO(archive.read(member))

    return opener


# --- writing ---------------------------------------------------------------------------------


def _copy_member(
    archive: zipfile.ZipFile, name: str, source: BinaryIO, compress_type: int
) -> tuple[str, int]:
    info = zipfile.ZipInfo(name, date_time=ZIP_FIXED_DATE)
    info.compress_type = compress_type
    info.external_attr = ZIP_FILE_MODE << ZIP_MODE_SHIFT
    digest = hashlib.sha256()
    size = 0
    with source, archive.open(info, "w", force_zip64=True) as dest:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
            dest.write(chunk)
    return digest.hexdigest(), size


def _bundle_id(job_id: str, members: Mapping[str, Mapping[str, Any]]) -> str:
    return stable_hash({"job_id": job_id, "members": {k: v["sha256"] for k, v in members.items()}})


def write_bundle(
    path: str | Path,
    *,
    job_id: str,
    shard_id: str,
    seq: int,
    parents: Sequence[str],
    parts: Iterable[PartSource],
    files: Mapping[str, bytes],
    meta: Mapping[str, Any],
    compress_level: int,
) -> dict[str, Any]:
    """Write a bundle atomically, verify it, and return its manifest."""
    target = Path(path)
    members: dict[str, dict[str, Any]] = {}
    part_rows: list[dict[str, Any]] = []
    with atomic_path(target) as tmp:
        with zipfile.ZipFile(tmp, "w", allowZip64=True, compresslevel=compress_level) as archive:
            for part in sorted({p.name: p for p in parts}.values(), key=lambda p: p.name):
                member = PARTS_DIR + part.name
                sha, size = _copy_member(archive, member, part.open(), zipfile.ZIP_STORED)
                if sha != part.sha256:
                    raise BundleError(f"part {part.name} changed while being copied")
                members[member] = {"sha256": sha, "bytes": size}
                part_rows.append(
                    {"name": part.name, "sha256": sha, "rows": part.rows, "bytes": size}
                )
            for name in sorted(files):
                sha, size = _copy_member(
                    archive, name, io.BytesIO(files[name]), zipfile.ZIP_DEFLATED
                )
                members[name] = {"sha256": sha, "bytes": size}
            manifest = {
                "format_version": BUNDLE_FORMAT_VERSION,
                "bundle_id": _bundle_id(job_id, members),
                "job_id": job_id,
                "shard_id": shard_id,
                "seq": seq,
                "parents": sorted(parents),
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "parts": part_rows,
                "members": members,
                "meta": dict(meta),
            }
            _copy_member(
                archive,
                MANIFEST_NAME,
                io.BytesIO(canonical_json(manifest).encode("utf-8")),
                zipfile.ZIP_DEFLATED,
            )
    result = verify_bundle(target)
    if not result.ok:
        target.unlink(missing_ok=True)
        raise BundleError(f"bundle failed verification after writing: {result.problems}")
    return manifest


# --- verification ----------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    problems: tuple[str, ...]
    manifest: dict[str, Any] | None


def verify_bundle(path: str | Path) -> VerifyResult:
    """Check the manifest and the sha256 and size of every member."""
    problems: list[str] = []
    manifest: dict[str, Any] | None = None
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if MANIFEST_NAME not in names:
                return VerifyResult(False, ("missing manifest",), None)
            manifest = json.loads(archive.read(MANIFEST_NAME))
            if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
                problems.append(f"unsupported format_version {manifest.get('format_version')}")
            members = manifest["members"]
            for name, meta in members.items():
                if name not in names:
                    problems.append(f"missing member {name}")
                    continue
                digest = hashlib.sha256()
                size = 0
                with archive.open(name) as source:
                    for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
                        digest.update(chunk)
                        size += len(chunk)
                if digest.hexdigest() != meta["sha256"] or size != meta["bytes"]:
                    problems.append(f"hash or size mismatch in {name}")
            extra = names - set(members) - {MANIFEST_NAME}
            if extra:
                problems.append(f"unlisted members {sorted(extra)}")
            if manifest.get("bundle_id") != _bundle_id(manifest["job_id"], members):
                problems.append("bundle_id does not match members")
    except (zipfile.BadZipFile, OSError, KeyError, ValueError, TypeError) as exc:
        return VerifyResult(False, (f"unreadable: {exc}",), None)
    return VerifyResult(not problems, tuple(problems), manifest)


# --- scanning and selection -----------------------------------------------------------------


@dataclass(frozen=True)
class BundleRef:
    path: Path
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    @property
    def seq(self) -> int:
        return int(self.manifest["seq"])

    @property
    def job_id(self) -> str:
        return self.manifest["job_id"]


@dataclass
class ScanResult:
    valid: list[BundleRef]
    rejected: list[tuple[Path, str]]
    duplicates: list[Path]

    @property
    def next_seq(self) -> int:
        return max((ref.seq for ref in self.valid), default=0) + 1


def scan_bundles(dirs: Iterable[str | Path], job_id: str) -> ScanResult:
    """Find, verify and select the bundles of ``job_id`` in the given directories.

    Corrupt or foreign bundles are rejected with a reason and never abort the scan. Selection does
    not use file names or modified times.
    """
    found: set[Path] = set()
    for directory in dirs:
        root = Path(directory)
        if root.is_dir():
            found.update(p for p in root.rglob(f"*{BUNDLE_SUFFIX}") if p.is_file())
    valid: dict[str, BundleRef] = {}
    rejected: list[tuple[Path, str]] = []
    duplicates: list[Path] = []
    for path in sorted(found):
        result = verify_bundle(path)
        if not result.ok or result.manifest is None:
            rejected.append((path, "; ".join(result.problems)))
            continue
        if result.manifest["job_id"] != job_id:
            rejected.append((path, f"other job {result.manifest['job_id']}"))
            continue
        ref = BundleRef(path, result.manifest)
        if ref.bundle_id in valid:
            duplicates.append(path)
            continue
        valid[ref.bundle_id] = ref
    ordered = sorted(valid.values(), key=lambda r: (r.seq, r.bundle_id))
    return ScanResult(ordered, rejected, duplicates)


def part_sources(refs: Iterable[BundleRef]) -> list[PartSource]:
    """Parts of all bundles, each distinct content (sha256) once, in a deterministic order."""
    unique: dict[str, PartSource] = {}
    for ref in refs:
        for row in ref.manifest["parts"]:
            if row["sha256"] in unique:
                continue
            member = PARTS_DIR + row["name"]
            unique[row["sha256"]] = PartSource(
                row["name"],
                row["sha256"],
                int(row["rows"]),
                int(row["bytes"]),
                _member_opener(ref.path, member),
            )
    return sorted(unique.values(), key=lambda p: p.name)


def consolidate_rows(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """One row per unit: ok beats skipped beats failed beats error; among equals a parsed answer beats an unparsed one
    (a rerun that parses replaces a first try that did not); more attempts win the remaining ties."""
    pieces = [f for f in frames if len(f)]
    if not pieces:
        return pd.DataFrame({UNIT_ID: [], STATUS: [], ATTEMPTS: []})
    frame = pd.concat(pieces, ignore_index=True)
    frame["_rank"] = frame[STATUS].map(STATUS_RANK)
    frame["_unparsed"] = frame["parse_ok"].map(lambda v: 0 if pd.isna(v) or bool(v) else 1) if "parse_ok" in frame.columns else 0
    frame = frame.sort_values(
        [UNIT_ID, "_rank", "_unparsed", ATTEMPTS], ascending=[True, True, True, False], kind="mergesort"
    )
    frame = frame.drop_duplicates(UNIT_ID, keep="first").drop(columns=["_rank", "_unparsed"])
    return frame.reset_index(drop=True)


def unit_state(refs: Iterable[BundleRef]) -> pd.DataFrame:
    """Consolidated (unit_id, status, attempts) over every part of the given bundles."""
    columns = [UNIT_ID, STATUS, ATTEMPTS]
    return consolidate_rows(part.read(columns) for part in part_sources(refs))


def read_consolidated(refs: Iterable[BundleRef]) -> pd.DataFrame:
    """All output rows of the given bundles, one row per unit."""
    return consolidate_rows(part.read() for part in part_sources(refs))


# --- rolling writer --------------------------------------------------------------------------


class BundleWriter:
    """Accumulates new rows and rewrites the single output bundle at each checkpoint.

    The previous output file is removed only after the new one has been written and verified, so a
    crash during a checkpoint always leaves at least one valid bundle in the output directory.
    """

    def __init__(
        self,
        out_dir: str | Path,
        job_id: str,
        shard_id: str,
        *,
        carried: Sequence[PartSource],
        parents: Sequence[str],
        seq: int,
        meta: Mapping[str, Any],
        compression: str,
        compress_level: int,
    ):
        self.out_dir = Path(out_dir)
        self.job_id = job_id
        self.shard_id = shard_id
        self._parts: list[PartSource] = list(carried)
        self._parents = list(parents)
        self._seq = seq - 1
        self._meta = dict(meta)
        self._compression = compression
        self._compress_level = compress_level
        self._buffer: list[pd.DataFrame] = []
        self._last_path: Path | None = None

    @property
    def latest_path(self) -> Path | None:
        return self._last_path

    def add_rows(self, frame: pd.DataFrame) -> None:
        if len(frame):
            self._buffer.append(frame)

    def checkpoint(
        self,
        *,
        state: Mapping[str, Any],
        diagnostics: Mapping[str, Any] | None = None,
        logs: Mapping[str, str] | None = None,
    ) -> Path:
        """Flush buffered rows into a new part and rewrite the bundle."""
        if self._buffer:
            frame = pd.concat(self._buffer, ignore_index=True)
            self._parts.append(PartSource.from_frame(frame, self._compression))
            self._buffer = []
        self._seq += 1
        files: dict[str, bytes] = {STATE_NAME: canonical_json(dict(state)).encode("utf-8")}
        for name, payload in (diagnostics or {}).items():
            files[f"{DIAGNOSTICS_DIR}{name}"] = canonical_json(payload).encode("utf-8")
        for name, text in (logs or {}).items():
            files[f"{LOGS_DIR}{name}"] = text.encode("utf-8")
        path = self.out_dir / bundle_filename(self.job_id, self.shard_id, self._seq)
        write_bundle(
            path,
            job_id=self.job_id,
            shard_id=self.shard_id,
            seq=self._seq,
            parents=self._parents,
            parts=self._parts,
            files=files,
            meta=self._meta,
            compress_level=self._compress_level,
        )
        if self._last_path is not None and self._last_path != path:
            self._last_path.unlink(missing_ok=True)
        self._last_path = path
        return path


def merge_bundles(
    inputs: Iterable[str | Path],
    out_dir: str | Path,
    *,
    job_id: str,
    meta: Mapping[str, Any],
    compression: str,
    compress_level: int,
) -> Path:
    """Combine all valid bundles of a job (e.g. from different accounts) into one job-level bundle."""
    scan = scan_bundles(inputs, job_id)
    if not scan.valid:
        raise BundleError(f"no valid bundle for job {job_id}; rejected: {scan.rejected}")
    writer = BundleWriter(
        out_dir,
        job_id,
        SHARD_ID_ALL,
        carried=part_sources(scan.valid),
        parents=[ref.bundle_id for ref in scan.valid],
        seq=scan.next_seq,
        meta=meta,
        compression=compression,
        compress_level=compress_level,
    )
    return writer.checkpoint(state={"merged_from": [ref.bundle_id for ref in scan.valid]})


def iter_bundle_paths(directory: str | Path) -> Iterator[Path]:
    """All bundle files directly or indirectly under ``directory``, sorted."""
    yield from sorted(Path(directory).rglob(f"*{BUNDLE_SUFFIX}"))
