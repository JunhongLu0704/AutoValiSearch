from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set

from openai import OpenAI

from outer.schema import SearchSpace
from outer.validator_role import build_builtin_validator_role_output


class AgentClientError(RuntimeError):
    pass


class JSONExtractionError(ValueError):
    pass


class BaseProposalClient:
    def describe_backend(self) -> Dict[str, Any]:
        raise NotImplementedError

    def health_check(self) -> Dict[str, Any]:
        return {'backend': self.describe_backend(), 'healthcheck': {'status': 'not_required'}}

    def propose_trial(
        self,
        proposal_id: str,
        prompt_bundle: Mapping[str, str],
        search_space: SearchSpace,
        candidate_hash_fn: Callable[[Mapping[str, Any]], str],
        forbidden_hashes: Set[str],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def propose_validator(
        self,
        validator_role_id: str,
        prompt_bundle: Mapping[str, str],
        *,
        validator_protocol: str,
        validator_preset: Optional[str],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class HeuristicAgentClient(BaseProposalClient):
    def __init__(self, seed: int = 0, max_attempts: int = 64):
        self.rng = random.Random(seed)
        self.max_attempts = max_attempts

    def _choose(self, values: Sequence[Any]) -> Any:
        return values[self.rng.randrange(len(values))]

    def _default_config(self, search_space: SearchSpace) -> Dict[str, Any]:
        return {field: self._choose(values) for field, values in search_space.tunables.items()}

    def _mutate_from_best(self, best_trial: Mapping[str, Any], search_space: SearchSpace) -> Dict[str, Any]:
        config = dict(best_trial.get('config') or self._default_config(search_space))
        fields = list(search_space.tunables.keys())
        num_mutations = 1 if self.rng.random() < 0.65 else min(2, len(fields))
        for field in self.rng.sample(fields, k=min(num_mutations, len(fields))):
            allowed = list(search_space.tunables[field])
            current = config.get(field)
            choices = [value for value in allowed if value != current]
            if choices:
                config[field] = self._choose(choices)
        return config

    def describe_backend(self) -> Dict[str, Any]:
        return {
            'backend': 'heuristic',
            'transport': 'none',
            'model_name': None,
            'server_url': None,
            'api_key_env': None,
            'timeout_sec': None,
            'max_retries': self.max_attempts,
            'experimental': False,
            'supports_validator_role': True,
            'metadata': {},
        }

    def propose_trial(
        self,
        proposal_id: str,
        prompt_bundle: Mapping[str, str],
        search_space: SearchSpace,
        candidate_hash_fn: Callable[[Mapping[str, Any]], str],
        forbidden_hashes: Set[str],
    ) -> Dict[str, Any]:
        del prompt_bundle
        best_trials = []
        try:
            payload = json.loads(prompt_bundle.get('user_prompt') or '{}')
            best_trials = list(payload.get('best_trials') or [])
        except json.JSONDecodeError:
            best_trials = []
        last_candidate = None
        for _ in range(self.max_attempts):
            use_best = bool(best_trials) and self.rng.random() < 0.7
            config = self._mutate_from_best(self._choose(best_trials), search_space) if use_best else self._default_config(search_space)
            config_hash = candidate_hash_fn(config)
            last_candidate = config
            if config_hash not in forbidden_hashes:
                hypothesis = (
                    'Refine around the best J_trial region.' if use_best else 'Explore a new J_trial region.'
                )
                return {'proposal_id': proposal_id, 'hypothesis': hypothesis, 'config': config}
        if last_candidate is None:
            raise AgentClientError('Failed to construct any proposal candidate')
        return {'proposal_id': proposal_id, 'hypothesis': 'Fallback heuristic candidate.', 'config': last_candidate}

    def propose_validator(
        self,
        validator_role_id: str,
        prompt_bundle: Mapping[str, str],
        *,
        validator_protocol: str,
        validator_preset: Optional[str],
    ) -> Dict[str, Any]:
        del prompt_bundle
        return build_builtin_validator_role_output(
            validator_role_id=validator_role_id,
            validator_protocol=validator_protocol,
            validator_preset=validator_preset,
        )


@dataclass
class OpenAIClientConfig:
    backend: str = 'api_openai'
    model_name: str = 'gpt-5.4'
    server_url: str = 'https://www.autodl.art/api/v1'
    api_key_env: Optional[str] = 'AUTODL_API_KEY'
    timeout_sec: float = 180.0
    max_retries: int = 4
    requires_api_key: bool = True
    supports_validator_role: bool = True
    experimental: bool = False
    metadata: Optional[Dict[str, Any]] = None


def _normalize_openai_base_url(server_url: str) -> str:
    normalized = str(server_url).strip().rstrip('/')
    if normalized.endswith('/v1'):
        return normalized
    return normalized + '/v1'


def _normalize_server_root(server_url: str) -> str:
    normalized = str(server_url).strip().rstrip('/')
    if normalized.endswith('/v1'):
        return normalized[:-3]
    return normalized


def _fetch_json(url: str, timeout_sec: float) -> tuple[int, Any]:
    request = urllib.request.Request(url, headers={'User-Agent': 'AutoValiSearch-AgentClient/1.0'})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        status = int(getattr(response, 'status', 200))
        raw = response.read().decode('utf-8', errors='replace')
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw
    return status, payload


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith('```') and stripped.endswith('```'):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return '\n'.join(lines[1:-1]).strip()
    return stripped


def _extract_first_json_object(text: str) -> str:
    candidate = _strip_code_fence(text)
    if candidate.startswith('{') and candidate.endswith('}'):
        return candidate
    start = candidate.find('{')
    if start < 0:
        raise JSONExtractionError('No JSON object found in model output')
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(candidate)):
        char = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return candidate[start:idx + 1]
    raise JSONExtractionError('Model output contained an unterminated JSON object')


def extract_json_payload(text: str) -> Dict[str, Any]:
    json_text = _extract_first_json_object(text)
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise JSONExtractionError('Top-level model output must decode to a JSON object')
    return payload


class OpenAIProposalClient(BaseProposalClient):
    def __init__(self, config: OpenAIClientConfig):
        api_key = None
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
        if config.requires_api_key and not api_key:
            raise AgentClientError(f'Missing API key env var: {config.api_key_env}')
        if not api_key:
            api_key = 'local-vllm'
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=_normalize_openai_base_url(config.server_url),
            timeout=config.timeout_sec,
        )
        self._server_root = _normalize_server_root(config.server_url)

    def describe_backend(self) -> Dict[str, Any]:
        return {
            'backend': self.config.backend,
            'transport': 'openai_compatible_http',
            'model_name': self.config.model_name,
            'server_url': self.config.server_url,
            'api_key_env': self.config.api_key_env,
            'timeout_sec': self.config.timeout_sec,
            'max_retries': self.config.max_retries,
            'experimental': self.config.experimental,
            'supports_validator_role': self.config.supports_validator_role,
            'metadata': dict(self.config.metadata or {}),
        }

    def health_check(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            'backend': self.config.backend,
            'model_name': self.config.model_name,
            'server_url': self.config.server_url,
            'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
        }
        health_url = self._server_root + '/health'
        models_url = _normalize_openai_base_url(self.config.server_url) + '/models'
        try:
            status, health_payload = _fetch_json(health_url, self.config.timeout_sec)
            payload['health_status_code'] = status
            payload['health_payload'] = health_payload
            payload['health_ok'] = 200 <= status < 300
        except Exception as exc:
            payload['health_ok'] = False
            payload['health_error'] = str(exc)
        try:
            status, models_payload = _fetch_json(models_url, self.config.timeout_sec)
            payload['models_status_code'] = status
            payload['models_payload'] = models_payload
            payload['models_ok'] = 200 <= status < 300
        except Exception as exc:
            payload['models_ok'] = False
            payload['models_error'] = str(exc)
        if not payload.get('health_ok') and not payload.get('models_ok'):
            raise AgentClientError(
                f'Backend health check failed for {self.config.backend} at {self.config.server_url}: '
                f"{payload.get('health_error') or payload.get('models_error')}"
            )
        return payload

    def _request_once(self, prompt_bundle: Mapping[str, str]) -> Dict[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {'role': 'system', 'content': prompt_bundle['system_prompt']},
                {'role': 'user', 'content': prompt_bundle['user_prompt']},
            ],
            stream=False,
        )
        if not completion.choices:
            raise AgentClientError('OpenAI client returned no choices')
        content = completion.choices[0].message.content
        if content is None:
            raise AgentClientError('OpenAI client returned empty message content')
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    parts.append(item.get('text', ''))
            content_text = ''.join(parts)
        else:
            content_text = str(content)
        return extract_json_payload(content_text)

    def propose_trial(
        self,
        proposal_id: str,
        prompt_bundle: Mapping[str, str],
        search_space: SearchSpace,
        candidate_hash_fn: Callable[[Mapping[str, Any]], str],
        forbidden_hashes: Set[str],
    ) -> Dict[str, Any]:
        del search_space, candidate_hash_fn, forbidden_hashes
        last_error: Optional[Exception] = None
        for _ in range(self.config.max_retries):
            try:
                proposal = self._request_once(prompt_bundle)
                if 'proposal_id' not in proposal or not str(proposal['proposal_id']).strip():
                    proposal['proposal_id'] = proposal_id
                return proposal
            except Exception as exc:
                last_error = exc
        raise AgentClientError(f'OpenAI trial proposal generation failed after {self.config.max_retries} attempts: {last_error}')

    def propose_validator(
        self,
        validator_role_id: str,
        prompt_bundle: Mapping[str, str],
        *,
        validator_protocol: str,
        validator_preset: Optional[str],
    ) -> Dict[str, Any]:
        del validator_protocol, validator_preset
        last_error: Optional[Exception] = None
        for _ in range(self.config.max_retries):
            try:
                proposal = self._request_once(prompt_bundle)
                if 'validator_role_id' not in proposal or not str(proposal['validator_role_id']).strip():
                    proposal['validator_role_id'] = validator_role_id
                return proposal
            except Exception as exc:
                last_error = exc
        raise AgentClientError(f'OpenAI validator proposal generation failed after {self.config.max_retries} attempts: {last_error}')


def build_agent_client(
    agent_mode: str,
    seed: int,
    *,
    agent_backend: Optional[str] = None,
    agent_server_url: Optional[str] = None,
    agent_model_name: Optional[str] = None,
    agent_model: str = 'gpt-5.4',
    agent_base_url: str = 'https://www.autodl.art/api/v1',
    agent_api_key_env: str = 'AUTODL_API_KEY',
    agent_timeout_sec: float = 180.0,
    agent_max_attempts: int = 4,
    agent_max_retries: Optional[int] = None,
    agent_backend_metadata: Optional[Mapping[str, Any]] = None,
) -> BaseProposalClient:
    mode = str(agent_mode).lower()
    backend = str(agent_backend).lower() if agent_backend else None
    max_retries = int(agent_max_retries if agent_max_retries is not None else agent_max_attempts)
    if backend is not None:
        if backend == 'heuristic':
            return HeuristicAgentClient(seed=seed, max_attempts=max_retries)
        if backend == 'api_openai':
            return OpenAIProposalClient(
                OpenAIClientConfig(
                    backend='api_openai',
                    model_name=agent_model_name or agent_model,
                    server_url=agent_server_url or agent_base_url,
                    api_key_env=agent_api_key_env,
                    timeout_sec=agent_timeout_sec,
                    max_retries=max_retries,
                    requires_api_key=True,
                    supports_validator_role=True,
                    experimental=False,
                    metadata=dict(agent_backend_metadata or {}),
                )
            )
        if backend == 'local_vllm_gemma4_26b':
            metadata = {
                'experimental': True,
                'runtime': 'vllm',
                'text_only': True,
                'kv_cache_dtype': 'fp8',
                'quantization_route': 'gguf_or_fallback_quantized_checkpoint',
            }
            metadata.update(dict(agent_backend_metadata or {}))
            return OpenAIProposalClient(
                OpenAIClientConfig(
                    backend='local_vllm_gemma4_26b',
                    model_name=agent_model_name or 'gemma-4-26b-a4b-it',
                    server_url=agent_server_url or 'http://127.0.0.1:8000',
                    api_key_env=agent_api_key_env,
                    timeout_sec=agent_timeout_sec,
                    max_retries=max_retries,
                    requires_api_key=False,
                    supports_validator_role=True,
                    experimental=True,
                    metadata=metadata,
                )
            )
        raise AgentClientError(f'Unsupported agent_backend: {agent_backend}')
    if mode == 'heuristic':
        return HeuristicAgentClient(seed=seed, max_attempts=max_retries)
    if mode in {'openai', 'autodl'}:
        return OpenAIProposalClient(
            OpenAIClientConfig(
                backend='api_openai',
                model_name=agent_model,
                server_url=agent_base_url,
                api_key_env=agent_api_key_env,
                timeout_sec=agent_timeout_sec,
                max_retries=max_retries,
                requires_api_key=True,
                supports_validator_role=True,
                experimental=False,
                metadata=dict(agent_backend_metadata or {}),
            )
        )
    raise AgentClientError(f'Unsupported agent_mode: {agent_mode}')

