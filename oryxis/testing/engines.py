"""Picklable factories for testing the data-parallel engine without a GPU."""

from __future__ import annotations

import multiprocessing as mp

from oryxis.compute.engine import Engine, MockEngine


def mock_factory(index: int) -> Engine:
    return MockEngine()


def _noop() -> None:
    pass


def child_process_factory(index: int) -> Engine:
    """Starts a child process while building, as vLLM's V1 engine does (fails inside a daemonic worker)."""
    child = mp.get_context("spawn").Process(target=_noop)
    child.start()
    child.join()
    return MockEngine()
