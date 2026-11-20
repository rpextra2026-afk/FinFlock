"""The final reader's engine layer against a fake vLLM: compact decoding, graph-capture fallback, restart, supervisor, picklable replicas."""
from __future__ import annotations

import pickle
import types

import pytest

from finfm.common.config import load_config
from finfm.compute.engine import GenParams, GenRequest
from reader2 import settings as S
from reader2.engine import EngineSupervisor, ReaderV2VLLM, _ReplicaFactory
from tests.conftest import CONFIG_DIR
from tests.test_engines import FakeLLM, FakeSampling

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}
PARAMS = GenParams(temperature=0.0, top_p=1.0, top_k=None, max_tokens=50, json_schema=SCHEMA, seed=7)


def fake_vllm(accept=("disable_any_whitespace", "whitespace_pattern", "")):
    def structured(json, **option):
        key = next(iter(option), "")
        if key not in accept:
            raise TypeError(f"unexpected {key}")
        return ("structured", json, option)
    module = types.SimpleNamespace(__name__="fakevllm", __version__="9.9", SamplingParams=FakeSampling)
    module.sampling_params = types.SimpleNamespace(StructuredOutputsParams=structured)
    return module


def build(cfg, *, accept=("disable_any_whitespace", "whitespace_pattern", ""), fail_graph_once=False, patch=False):
    holder = {"builds": []}

    def factory(**kwargs):
        holder["builds"].append(dict(kwargs))
        if fail_graph_once and len(holder["builds"]) == 1:
            raise RuntimeError("CUDA graph capture failed")
        holder["llm"] = FakeLLM(**kwargs)
        return holder["llm"]

    engine = ReaderV2VLLM("some/model", cfg, tensor_parallel_size=2, llm_factory=factory, vllm_module=fake_vllm(accept), patch_attention=patch)
    return engine, holder


def test_reader_task_uses_compact_json_decoding(cfg):
    engine, holder = build(cfg)
    engine.generate([GenRequest("u", "hello", task="reader_v2")], PARAMS)
    sampling = holder["llm"].calls[0][1][0].kwargs
    assert sampling["structured_outputs"][2] == {"disable_any_whitespace": True} and engine.whitespace_option == "disable_any_whitespace"
    assert sampling["seed"] == 7 and sampling["temperature"] == 0.0


def test_retry_request_keeps_its_own_seed(cfg):
    engine, holder = build(cfg)
    engine.generate([GenRequest("u", "hello", task="reader_v2", seed=123)], PARAMS)
    assert holder["llm"].calls[0][1][0].kwargs["seed"] == 123


def test_other_tasks_use_the_ordinary_path(cfg):
    engine, holder = build(cfg)
    engine.generate([GenRequest("u", "hello", task="generic")], PARAMS)
    assert holder["llm"].calls[0][1][0].kwargs["structured_outputs"] == ("structured", SCHEMA, ) or "structured_outputs" in holder["llm"].calls[0][1][0].kwargs


def test_falls_back_when_the_whitespace_option_is_unknown(cfg):
    engine, holder = build(cfg, accept=("",))
    engine.generate([GenRequest("u", "hello", task="reader_v2")], PARAMS)
    assert engine.whitespace_option == "backend_default"


def test_cuda_graph_capture_failure_rebuilds_without_graphs(cfg):
    engine, holder = build(cfg, fail_graph_once=True)
    assert len(holder["builds"]) == 2 and holder["builds"][1]["enforce_eager"] is True and engine.describe()["enforce_eager"] is True


def test_engine_death_restarts_the_llm_and_repeats_the_batch(cfg):
    engine, holder = build(cfg)
    first = holder["llm"]
    calls = {"n": 0}
    real = first.generate

    def dying(prompts, sampling, use_tqdm=False):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("EngineCore died")
        return real(prompts, sampling, use_tqdm)

    first.generate = dying
    results = engine.generate([GenRequest("u", "hello", task="reader_v2")], PARAMS)
    assert len(results) == 1 and len(holder["builds"]) == 2 and engine.restarts == 1 and holder["llm"] is not first


def test_describe_lists_what_a_bundle_should_remember(cfg):
    engine, _ = build(cfg)
    engine.generate([GenRequest("u", "hello", task="reader_v2")], PARAMS)
    d = engine.describe()
    assert d["vllm"] == "9.9" and d["tensor_parallel_size"] == 2 and d["dtype"] == "float16" and d["whitespace_option"]


def test_supervisor_rebuilds_on_fatal_errors_only():
    class Flaky:
        def __init__(self, fail):
            self.fail, self.closed = fail, False

        def generate(self, requests, params):
            if self.fail:
                raise RuntimeError(self.fail)
            return ["ok"]

        def close(self):
            self.closed = True

    made = []

    def factory():
        made.append(Flaky("replica 0 failed: CUDA error" if not made else None))
        return made[-1]

    sup = EngineSupervisor(factory, log=lambda *_: None)
    assert sup.generate([], PARAMS) == ["ok"] and len(made) == 2 and made[0].closed and sup.rebuilds == 1
    ordinary = EngineSupervisor(lambda: Flaky("bad input"), log=lambda *_: None)
    with pytest.raises(RuntimeError, match="bad input"):
        ordinary.generate([], PARAMS)
    assert ordinary.rebuilds == 0


def test_supervisor_gives_up_after_repeated_fatal_errors():
    class Dead:
        def generate(self, requests, params):
            raise RuntimeError("EngineCore died")

        def close(self):
            pass

    sup = EngineSupervisor(Dead, log=lambda *_: None)
    with pytest.raises(RuntimeError):
        sup.generate([], PARAMS)
    assert sup.rebuilds == S.ENGINE_REBUILDS


def test_replica_factory_can_be_pickled_for_the_spawned_processes():
    factory = _ReplicaFactory(str(CONFIG_DIR), "google/gemma-4-12B-it-qat-w4a16-ct", 1, {}, True)
    again = pickle.loads(pickle.dumps(factory))
    assert again.args == factory.args


def test_engines_follow_the_configured_gpu_layout(cfg):
    from reader2.engine import build_engine_v2, WIDE_HEAD_LLMS
    assert WIDE_HEAD_LLMS == ("llm1",)
    assert cfg.get("engine.models.llm1.replicas") == 2 and cfg.get("engine.models.llm2.tensor_parallel") == 2 and cfg.get("engine.models.llm3.tensor_parallel") == 2
