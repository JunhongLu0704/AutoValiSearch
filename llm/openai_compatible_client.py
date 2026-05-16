from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from utils.io import write_jsonl
from utils.json_utils import redact_secrets
from .schemas import BackendSpec, LLMMessage


@dataclass
class OpenAICompatibleClient:
    backend: BackendSpec
    trace_dir: Path | None = None
    trace_rows: list[dict[str, Any]] = field(default_factory=list)

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get(self.backend.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(dict(self.backend.extra_headers))
        return headers

    def build_payload(self, messages: Sequence[LLMMessage], **kwargs: Any) -> dict[str, Any]:
        payload = {
            "model": self.backend.model,
            "messages": [message.__dict__ for message in messages],
        }
        payload.update(kwargs)
        return payload

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> dict[str, Any]:
        payload = self.build_payload(messages, **kwargs)
        request = urllib.request.Request(
            url=self.backend.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        start = time.time()
        last_error = None
        for attempt in range(max(self.backend.max_retries, 1)):
            try:
                with urllib.request.urlopen(request, timeout=self.backend.timeout_sec) as response:
                    body = response.read().decode("utf-8")
                result = json.loads(body)
                self._record_trace(payload, result, time.time() - start, attempt, ok=True)
                return result
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                self._record_trace(payload, {"error": str(exc)}, time.time() - start, attempt, ok=False)
        raise RuntimeError(f"OpenAI-compatible request failed: {last_error}") from last_error

    def _record_trace(self, payload: Mapping[str, Any], response: Mapping[str, Any], elapsed_sec: float, attempt: int, *, ok: bool) -> None:
        finish_reason = None
        if isinstance(response.get("choices"), list) and response["choices"]:
            first_choice = response["choices"][0]
            if isinstance(first_choice, Mapping):
                finish_reason = first_choice.get("finish_reason")
        row = {
            "backend": self.backend.public_dict(),
            "ok": ok,
            "attempt": attempt,
            "elapsed_sec": round(elapsed_sec, 4),
            "max_tokens": payload.get("max_tokens"),
            "finish_reason": finish_reason,
            "llm_response_truncated": finish_reason == "length",
            "payload": redact_secrets(dict(payload)),
            "response": redact_secrets(dict(response)),
        }
        self.trace_rows.append(row)
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(self.trace_dir / "llm_trace.jsonl", self.trace_rows)


