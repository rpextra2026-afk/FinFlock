"""Real-engine plumbing tested with fakes: vLLM version differences, system-role fallback, data parallel."""

from __future__ import annotations

import types

import pytest

from finfm.common.config import load_config
from finfm.compute.engine import GenParams, GenRequest, MockEngine
from finfm.compute.factory import build_engine, resolve_model_path
from finfm.compute.parallel import DataParallelEngine, describe
from finfm.compute.vllm_engine import VLLMEngine, structured_kwargs
from finfm.testing.engines import child_process_factory, mock_factory
from tests.conftest import CONFIG_DIR

PARAMS = GenParams(temperature=0.7, top_p=0.9, top_k=64, max_tokens=50, seed=3)


class FakeSampling:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOutput:
    def __init__(self, text, n_tokens=3, finish="stop"):
        self.text, self.token_ids, self.finish_reason = text, list(range(n_tokens)), finish


class FakeRequestOutput:
    def __init__(self, outputs, prompt_tokens):
        self.outputs, self.prompt_token_ids = outputs, list(range(prompt_tokens))


class FakeTokenizer:
    def __init__(self, reject_system=False):
        self.reject_system, self.seen = reject_system, []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if self.reject_system and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        self.seen.append(messages)
        return "||".join(f"{m['role']}:{m['content']}" for m in messages)


class FakeLLM:
    def __init__(self, reject_system=False, empty=False, **kwargs):
        self.kwargs, self.tokenizer, self.empty, self.calls = kwargs, FakeTokenizer(reject_system), empty, []

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling, use_tqdm=False):
        self.calls.append((prompts, sampling))
        return [FakeRequestOutput([FakeOutput("" if self.empty else f"out{i}-{k}") for k in range(s.kwargs["n"])], 5)
                for i, s in enumerate(sampling)]


def fake_vllm(structured="new"):
    module = types.SimpleNamespace(__name__="fakevllm", __version__="9.9", SamplingParams=FakeSampling)
    classes = {}
    if structured == "new":
        classes["StructuredOutputsParams"] = lambda json: ("structured", json)
    if structured == "old":
        classes["GuidedDecodingParams"] = lambda json: ("guided", json)
    module.sampling_params = types.SimpleNamespace(**classes)
    return module


def make_engine(cfg, reject_system=False, empty=False, structured="new"):
    holder = {}

    def factory(**kwargs):
        holder["llm"] = FakeLLM(reject_system=reject_system, empty=empty, **kwargs)
        return holder["llm"]

    engine = VLLMEngine("some/model", cfg, tensor_parallel_size=2, llm_factory=factory, vllm_module=fake_vllm(structured))
    return engine, holder


def test_engine_passes_the_configured_settings_to_vllm(cfg):
    _, holder = make_engine(cfg)
    kwargs = holder["llm"].kwargs
    assert kwargs["dtype"] == "float16" and kwargs["tensor_parallel_size"] == 2 and kwargs["max_model_len"] == int(cfg.get("engine.max_model_len"))
    assert kwargs["enable_prefix_caching"] is True and kwargs["gpu_memory_utilization"] == 0.9 and kwargs["max_num_seqs"] == 64


def test_generation_returns_results_in_order_with_seeds_draws_and_token_counts(cfg):
    engine, holder = make_engine(cfg)
    requests = [GenRequest("a", "hello", n=3, seed=11, system="be brief"), GenRequest("b", "world", n=1)]
    results = engine.generate(requests, PARAMS)
    assert [r.unit_id for r in results] == ["a", "b"] and len(results[0].texts) == 3 and len(results[1].texts) == 1
    assert results[0].tokens_in == 5 and results[0].tokens_out == 9
    sampling = holder["llm"].calls[0][1]
    assert sampling[0].kwargs["seed"] == 11 and sampling[1].kwargs["seed"] == 3  # per-request seed, else the default
    assert sampling[0].kwargs["n"] == 3 and sampling[0].kwargs["top_k"] == 64 and sampling[0].kwargs["temperature"] == 0.7
    assert holder["llm"].calls[0][0][0] == "system:be brief||user:hello"


def test_system_role_is_merged_into_the_user_turn_when_the_template_refuses_it(cfg):
    engine, holder = make_engine(cfg, reject_system=True)
    results = engine.generate([GenRequest("a", "hello", system="rules")], PARAMS)
    assert holder["llm"].calls[0][0][0] == "user:rules\n\nhello" and engine.system_ok is False
    assert len(results) == 1
    again = engine.generate([GenRequest("b", "x", system="rules")], PARAMS)  # the decision is remembered
    assert holder["llm"].calls[1][0][0] == "user:rules\n\nx" and len(again) == 1


def test_structured_output_parameter_is_detected_and_unsupported_versions_fail_loudly(cfg):
    schema = {"type": "object"}
    new_kwargs, new_name = structured_kwargs(fake_vllm("new"), schema)
    old_kwargs, old_name = structured_kwargs(fake_vllm("old"), schema)
    assert (new_name, old_name) == ("structured_outputs", "guided_decoding")
    assert new_kwargs["structured_outputs"] == ("structured", schema) and old_kwargs["guided_decoding"] == ("guided", schema)
    with pytest.raises(RuntimeError):
        structured_kwargs(fake_vllm("none"), schema)
    engine, holder = make_engine(cfg, structured="old")
    engine.generate([GenRequest("a", "x")], GenParams(0.0, 1.0, None, 20, json_schema=schema))
    assert "guided_decoding" in holder["llm"].calls[0][1][0].kwargs and engine.describe()["structured_keyword"] == "guided_decoding"
    assert engine.describe()["vllm"] == "9.9"


def test_empty_output_for_a_non_empty_prompt_is_flagged_as_nan_suspect(cfg):
    engine, _ = make_engine(cfg, empty=True)
    assert engine.generate([GenRequest("a", "hello")], PARAMS)[0].nan_suspect is True
    ok, _ = make_engine(cfg)
    assert ok.generate([GenRequest("a", "hello")], PARAMS)[0].nan_suspect is False


def test_attention_backend_environment_variable_is_set_before_import(cfg, monkeypatch):
    monkeypatch.delenv("FAKE_BACKEND", raising=False)
    configured = cfg.with_overrides({"engine.attention_backend_env": {"name": "FAKE_BACKEND", "value": "xformers"}})
    make_engine(configured)
    import os
    assert os.environ["FAKE_BACKEND"] == "xformers"


def test_data_parallel_engine_keeps_request_order_across_processes():
    engine = DataParallelEngine(mock_factory, replicas=2, set_devices=False)
    try:
        requests = [GenRequest(f"u{i}", f"prompt number {i}", n=2) for i in range(7)]
        results = engine.generate(requests, PARAMS)
        reference = MockEngine().generate(requests, PARAMS)
        assert [r.unit_id for r in results] == [r.unit_id for r in requests]
        assert [r.texts for r in results] == [r.texts for r in reference]  # same answers as one engine
        assert engine.generate([], PARAMS) == []
    finally:
        engine.close()


def test_replica_workers_may_start_their_own_child_processes():
    engine = DataParallelEngine(child_process_factory, replicas=2, set_devices=False)  # vLLM V1 does this while loading
    try:
        assert engine.generate([GenRequest("a", "hello")], PARAMS)[0].unit_id == "a"
    finally:
        engine.close()
    engine.close()  # closing twice is harmless (atexit closes it again)


def test_model_paths_prefer_local_weights_and_the_mock_engine_is_selectable(cfg, tmp_path):
    local = tmp_path / "org--model"
    local.mkdir()
    rooted = cfg.with_overrides({"engine.local_weights_root": str(tmp_path)})
    assert resolve_model_path(rooted, "org/model") == str(local) and resolve_model_path(rooted, "org/other") == "org/other"
    mock = cfg.with_overrides({"engine.engine": "mock"})
    assert isinstance(build_engine(mock, "llm2", "config"), MockEngine)
    assert describe(MockEngine())["engine"] == "mock"


def test_a_prompt_that_cannot_fit_is_returned_empty_and_does_not_stop_the_rest_of_the_batch():
    from finfm.compute.vllm_engine import PROMPT_TOO_LONG

    cfg = load_config(CONFIG_DIR).with_overrides({"engine.max_model_len": 60})
    factory = lambda **kw: FakeLLM(**kw)  # noqa: E731
    engine = VLLMEngine("some/model", cfg, tensor_parallel_size=1, llm_factory=factory, vllm_module=fake_vllm(None))
    engine.tokenizer.encode = lambda text, add_special_tokens=False: text.split()  # one token per word
    params = GenParams(temperature=0.0, top_p=1.0, top_k=None, max_tokens=50, seed=1)
    out = engine.generate([GenRequest("short", "two words"), GenRequest("long", "word " * 40), GenRequest("short2", "ok")], params)
    assert [r.unit_id for r in out] == ["short", "long", "short2"]
    assert out[1].finish == PROMPT_TOO_LONG and out[1].texts == ("",) and out[0].texts[0] != "" and out[2].texts[0] != ""


def test_a_64_bit_seed_is_reduced_to_fit_a_c_long():
    big = (1 << 64) - 1
    assert VLLMEngine._seed(big) == (1 << 63) - 1 and VLLMEngine._seed(5) == 5 and VLLMEngine._seed(None) is None
    assert VLLMEngine._seed(big) < 2**63


def test_a_dead_engine_is_restarted_and_the_batch_is_redone():
    class EngineDeadError(RuntimeError):
        pass

    cfg = load_config(CONFIG_DIR)
    built = []

    class Flaky(FakeLLM):
        def generate(self, prompts, sampling, use_tqdm=False):
            if not built:  # the first engine dies on its first batch
                built.append("dead")
                raise EngineDeadError("EngineCore encountered an issue")
            return super().generate(prompts, sampling, use_tqdm)

    engine = VLLMEngine("some/model", cfg, tensor_parallel_size=1, llm_factory=lambda **kw: Flaky(**kw), vllm_module=fake_vllm(None))
    out = engine.generate([GenRequest("a", "hello")], GenParams(temperature=0.0, top_p=1.0, top_k=None, max_tokens=10, seed=1))
    assert out[0].texts[0] != "" and engine.restarts == 1
    engine2 = VLLMEngine("some/model", cfg.with_overrides({"engine.max_restarts": 0}), tensor_parallel_size=1,
                         llm_factory=lambda **kw: Flaky(**kw), vllm_module=fake_vllm(None))
    built.clear()
    with pytest.raises(EngineDeadError):
        engine2.generate([GenRequest("a", "hello")], GenParams(temperature=0.0, top_p=1.0, top_k=None, max_tokens=10, seed=1))


def failing_factory(index):
    import sys

    print("the real engine-core error: CUDA out of memory", file=sys.stderr, flush=True)
    raise RuntimeError("Engine core initialization failed")


def test_a_replica_that_fails_to_start_reports_its_own_output_and_the_others_are_not_started():
    with pytest.raises(RuntimeError, match="the real engine-core error: CUDA out of memory"):
        DataParallelEngine(failing_factory, replicas=2, set_devices=False)
