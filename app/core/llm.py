"""Provider-agnostic LLM client (OpenAI-compatible chat completions over httpx).

Usage:
    client = get_llm_client()
    if client.available:
        resp = client.chat([{"role": "user", "content": "..."}])
        data = client.chat_json(messages)   # -> parsed dict, or None on failure
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import LLMConfig


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None, transport: Optional[httpx.BaseTransport] = None):
        self.config = config or LLMConfig.from_env()
        self._transport = transport  # injectable for tests

    @property
    def available(self) -> bool:
        return self.config.usable

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not self.available:
            raise LLMError("LLM not configured (missing API key or disabled)")
        cfg = self.config
        payload: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        url = f"{cfg.base_url}/chat/completions"

        last_err: Optional[Exception] = None
        t0 = time.perf_counter()
        with httpx.Client(timeout=cfg.timeout_s, transport=self._transport) as http:
            for attempt in range(cfg.max_retries + 1):
                try:
                    r = http.post(url, json=payload, headers=headers)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise LLMError(f"retryable HTTP {r.status_code}")
                    if r.status_code >= 400:
                        raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                    body = r.json()
                    usage = body.get("usage") or {}
                    return LLMResponse(
                        content=body["choices"][0]["message"]["content"] or "",
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                        model=body.get("model", cfg.model),
                    )
                except (httpx.HTTPError, LLMError, KeyError, ValueError) as e:
                    last_err = e
                    non_retryable = isinstance(e, LLMError) and "retryable" not in str(e)
                    if non_retryable or attempt >= cfg.max_retries:
                        break
                    time.sleep(min(2**attempt * 0.5, 4))
        raise LLMError(f"LLM call failed: {last_err}")

    def chat_json(self, messages: List[Dict[str, str]], **kw: Any) -> Optional[Dict[str, Any]]:
        """Returns the parsed JSON object, or None on any failure (callers fall back to rules)."""
        data, _ = self.chat_json_with_usage(messages, **kw)
        return data

    def chat_json_with_usage(self, messages: List[Dict[str, str]], **kw: Any):
        try:
            resp = self.chat(messages, json_mode=True, **kw)
        except LLMError:
            return None, None
        text = resp.content.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            return json.loads(m.group(0) if m else text), resp
        except (json.JSONDecodeError, TypeError):
            return None, resp


_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm_client(client: Optional[LLMClient] = None) -> None:
    """Replace/clear the process-wide client (tests, hot config reload)."""
    global _client
    _client = client
