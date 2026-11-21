"""Pilot-only vLLM options and continuation scoring; production engine remains untouched."""
from __future__ import annotations
import importlib
from finflock.compute.vllm_engine import VLLMEngine, STRUCTURED_CANDIDATES
from finflock.reader.cause_scores import continuation_tokens, sum_label_logprobs


class ScoringVLLM(VLLMEngine):
    def __init__(self, model_path, cfg, *, tensor_parallel_size=2, vllm_module=None):
        # Olmo needs no Gemma-specific installed-file patch; leave the installed runtime untouched.
        self.cfg = cfg
        self.vllm = vllm_module or importlib.import_module('vllm')
        self._factory = self.vllm.LLM
        self._kwargs = dict(model=model_path, dtype=cfg.get('engine.dtype'),
            tensor_parallel_size=tensor_parallel_size, max_model_len=cfg.get('engine.max_model_len'),
            gpu_memory_utilization=cfg.get('engine.gpu_memory_utilization'),
            enable_prefix_caching=True, enforce_eager=cfg.get('engine.enforce_eager'),
            max_num_seqs=cfg.get('engine.max_num_seqs'))
        self.llm = self._factory(**self._kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_path, self.system_ok, self.structured_keyword = model_path, cfg.get('engine.system_role_supported'), None
        self.whitespace_option = None
        self.compact_enabled = True

    def _sampling(self, params, request):
        if request.task != 'reader_v2' or not self.compact_enabled:
            return super()._sampling(params, request)
        sampling = getattr(self.vllm, 'sampling_params', None) or importlib.import_module('vllm.sampling_params')
        for keyword, class_name in STRUCTURED_CANDIDATES:
            cls = getattr(sampling, class_name, None)
            if cls is None:
                continue
            for option in ({'disable_any_whitespace': True}, {'whitespace_pattern': ''}, {}):
                try:
                    structured = cls(json=dict(params.json_schema), **option)
                    result = self.vllm.SamplingParams(n=request.n, temperature=params.temperature,
                        top_p=params.top_p, max_tokens=params.max_tokens,
                        seed=self._seed(request.seed if request.seed is not None else params.seed),
                        **{keyword: structured})
                except (TypeError, ValueError):
                    continue
                self.structured_keyword, self.whitespace_option = keyword, (next(iter(option)) if option else 'backend_default')
                return result
        raise RuntimeError('no_supported_structured_output')

    def generate(self, requests, params):
        try:
            results = super().generate(requests, params)
            self.last_finishes = {r.unit_id: r.finish for r in results}
            return results
        except (TypeError, ValueError) as exc:
            if any(r.task == 'reader_v2' for r in requests) and self.compact_enabled and 'whitespace' in str(exc).lower():
                self.compact_enabled = False
                self.whitespace_option = 'backend_default_after_rejection'
                results = super().generate(requests, params)
                self.last_finishes = {r.unit_id: r.finish for r in results}
                return results
            raise

    def score_continuations(self, prompts, continuations):
        if len(prompts) != len(continuations):
            raise ValueError('prompt_continuation_length_mismatch')
        tokens = [continuation_tokens(self.tokenizer, p, c) for p, c in zip(prompts, continuations)]
        results = [{'logp': None, 'boundary_merge': t.boundary_merge, 'label_tokens': t.label_tokens,
                    'error': 'prompt_too_long'} for t in tokens]
        run = [i for i, t in enumerate(tokens) if len(t.token_ids) < int(self.cfg.get('engine.max_model_len'))]
        if run:
            params = [self.vllm.SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0) for _ in run]
            outputs = self._generate_with_restart([{'prompt_token_ids': tokens[i].token_ids} for i in run], params)
            for i, output in zip(run, outputs, strict=True):
                try:
                    results[i].update(logp=sum_label_logprobs(tokens[i], output.prompt_logprobs), error=None)
                except (ValueError, TypeError, AttributeError) as exc:
                    results[i]['error'] = str(exc)
        return results

    def describe(self):
        return {**super().describe(), 'whitespace_option': self.whitespace_option,
                'dtype': self._kwargs['dtype'], 'tensor_parallel_size': self._kwargs['tensor_parallel_size']}
