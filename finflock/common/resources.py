"""Memory and CPU planning for the heavy CPU stage, and a resource log for every stage.

Why: the road-1 fits run one process per (configuration, fold). Each process holds the graphs and builds dense ``(events, nodes, nodes, d, d)``
tensors, so the number of processes and the size of a micro-batch decide whether a machine runs out of memory. Both are derived here from the
memory that is actually free, never from the core count alone. Nothing here changes what is fitted: the sample, the grid and the batch of events
per optimiser step stay as configured (``road1.optimiser.batch_events``); a batch is only cut into micro-batches whose gradients are summed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from finflock.common.config import Config

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CGROUP_MEMORY_FILES = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
BYTES_PER_GB = 2**30
SAMPLE_SECONDS = 1.0
UNLIMITED_BYTES = 2**60  # cgroup files hold a huge number or 'max' when no limit is set
GB_DECIMALS = 2
RSS_DECIMALS = 3
SECOND_DECIMALS = 1


def limit_threads(n: int) -> None:
    """Pin BLAS / OpenMP threads for this process and every child it starts. Call before numpy / torch are imported (or use ``set_num_threads``)."""
    for name in THREAD_VARIABLES:
        os.environ[name] = str(n)


def usable_cpus() -> int:
    """Cores this process may use (the scheduler's affinity where it exists)."""
    getter = getattr(os, "sched_getaffinity", None)
    return len(getter(0)) if getter else (os.cpu_count() or 1)


def memory_limit_bytes() -> int | None:
    """The container's memory limit, if one is set."""
    for path in CGROUP_MEMORY_FILES:
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < UNLIMITED_BYTES:
            return int(text)
    return None


def available_memory_bytes() -> int:
    """Memory a new process can still get: the smaller of what the OS reports free and the container limit minus what this tree uses."""
    import psutil

    free = int(psutil.virtual_memory().available)
    limit = memory_limit_bytes()
    if limit is None:
        return free
    return min(free, max(limit - tree_rss_bytes(), 0))


def tree_rss_bytes() -> int:
    """Resident memory of this process and all its children."""
    import psutil

    me = psutil.Process()
    total = 0
    for proc in [me, *me.children(recursive=True)]:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            continue
    return total


def plan_workers(cfg: Config, requested: int | None = None) -> dict[str, Any]:
    """How many fitting processes fit in memory. ``requested`` (from the command line) can only lower the number."""
    spec = cfg.get("cpu.resources")
    cpus = usable_cpus()
    available = available_memory_bytes()
    per_worker = int(float(spec["worker_base_gb"]) * BYTES_PER_GB) + int(float(spec["peak_factor"]) * int(spec["max_batch_bytes"]))
    by_memory = max(int(available * float(spec["memory_fraction"]) // per_worker), 1)
    workers = max(min(cpus, by_memory, int(spec["max_workers"]), requested or cpus), 1)
    return {"cpus": cpus, "available_gb": round(available / BYTES_PER_GB, GB_DECIMALS), "per_worker_gb": round(per_worker / BYTES_PER_GB, GB_DECIMALS),
            "workers_by_memory": by_memory, "workers": workers, "blas_threads": int(spec["blas_threads"]), "memory_fraction": float(spec["memory_fraction"])}


class ResourceLog:
    """Wall time and peak resident memory (process tree, sampled every second) of each named stage; written to JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self.path, self.stages = path, []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        peak, stop = [tree_rss_bytes()], threading.Event()

        def sample() -> None:
            while not stop.wait(SAMPLE_SECONDS):
                peak[0] = max(peak[0], tree_rss_bytes())

        thread = threading.Thread(target=sample, daemon=True)
        started = time.time()
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            peak[0] = max(peak[0], tree_rss_bytes())
            self.stages.append({"stage": name, "seconds": round(time.time() - started, SECOND_DECIMALS), "peak_rss_gb": round(peak[0] / BYTES_PER_GB, RSS_DECIMALS),
                                "available_gb_after": round(available_memory_bytes() / BYTES_PER_GB, GB_DECIMALS)})
            self.flush()

    def flush(self) -> None:
        if self.path is not None:
            self.path.write_text(json.dumps({"stages": self.stages}, indent=1), encoding="utf-8")
