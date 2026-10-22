"""LLM provider configuration (env-driven, DeepSeek by default).

Environment variables (all optional):
    LLM_PROVIDER     deepseek (default) | openai | qwen | moonshot | zhipu | gemini | custom
    LLM_API_KEY      generic key; falls back to the provider-specific key (e.g. DEEPSEEK_API_KEY)
    LLM_MODEL        overrides the provider's default model
    LLM_BASE_URL     overrides the provider's default OpenAI-compatible base URL
    LLM_TIMEOUT_S    request timeout in seconds (default 60)
    LLM_MAX_RETRIES  retries on 429/5xx/network errors (default 2)
    LLM_ENABLED      set to 0/false to force the rule-based fallback

A `.env` file in the project root is loaded automatically (without overriding real env vars).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    model: str
    key_env: str


PROVIDER_PRESETS: Dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset("https://api.deepseek.com", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "openai": ProviderPreset("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "qwen": ProviderPreset(
        "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", "DASHSCOPE_API_KEY"
    ),
    "moonshot": ProviderPreset("https://api.moonshot.cn/v1", "moonshot-v1-8k", "MOONSHOT_API_KEY"),
    "zhipu": ProviderPreset("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", "ZHIPU_API_KEY"),
    "gemini": ProviderPreset(
        "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash", "GEMINI_API_KEY"
    ),
    # Any other OpenAI-compatible endpoint (vLLM, Ollama, one-api...): set LLM_BASE_URL + LLM_MODEL.
    "custom": ProviderPreset("", "", "LLM_API_KEY"),
}

_ENV_LOADED = False


def _load_dotenv() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        elif v.startswith("#"):
            v = ""
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        os.environ.setdefault(k.strip(), v)


_load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: Optional[str]
    timeout_s: float
    max_retries: int
    enabled: bool

    @property
    def usable(self) -> bool:
        return self.enabled and bool(self.api_key) and bool(self.base_url) and bool(self.model)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        _load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        if provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. Supported: {sorted(PROVIDER_PRESETS)}"
            )
        preset = PROVIDER_PRESETS[provider]
        api_key = os.getenv("LLM_API_KEY") or os.getenv(preset.key_env) or None
        enabled = os.getenv("LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
        return cls(
            provider=provider,
            base_url=(os.getenv("LLM_BASE_URL") or preset.base_url).rstrip("/"),
            model=os.getenv("LLM_MODEL") or preset.model,
            api_key=api_key,
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            enabled=enabled,
        )
