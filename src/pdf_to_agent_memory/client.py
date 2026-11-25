"""Minimal OpenAI-compatible JSON client, replaceable for agent integration."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dotenv import dotenv_values


PROVIDERS = ("openai", "qwen", "compatible")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _settings(env_file: str | Path | None) -> dict[str, str]:
    path = Path(env_file) if env_file is not None else Path(".env")
    if env_file is not None and not path.is_file():
        raise ValueError(f"Environment file not found: {path}")
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    return {**{k: v for k, v in values.items() if v is not None}, **os.environ}


class ChatClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: int = 120,
                 *, provider: str | None = None, model: str | None = None,
                 env_file: str | Path | None = None):
        settings = _settings(env_file)
        self.provider = (provider or settings.get("PDF2MEM_PROVIDER") or "openai").strip().lower()
        if self.provider not in PROVIDERS:
            raise ValueError("Unknown provider. Choose openai, qwen or compatible.")
        prefix = self.provider.upper()
        key_name = "DASHSCOPE_API_KEY" if self.provider == "qwen" else f"{prefix}_API_KEY"
        self.api_key = api_key or settings.get(key_name)
        if not self.api_key and self.provider != "compatible":
            raise ValueError(f"Set {key_name} in the deployment environment or .env.")
        defaults = {"openai": DEFAULT_OPENAI_MODEL, "qwen": "qwen-plus"}
        self.model = model or settings.get(f"{prefix}_MODEL") or defaults.get(self.provider)
        if not self.model:
            raise ValueError(f"Set {prefix}_MODEL or pass --model.")
        url = base_url or settings.get(f"{prefix}_BASE_URL")
        if not url and self.provider == "openai":
            url = "https://api.openai.com/v1"
        if not url:
            raise ValueError(f"Set {prefix}_BASE_URL to the service's chat API base URL.")
        self.base_url = url.rstrip("/")
        parsed = urlsplit(self.base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                or parsed.query or parsed.fragment or parsed.path.endswith("/chat/completions")):
            raise ValueError(f"{prefix}_BASE_URL must be an HTTP(S) base URL without credentials, query, or /chat/completions.")
        self.timeout = timeout

    def complete(self, *, model: str, system: str, text: str) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "response_format": {"type": "json_object"},
        }
        if self.provider == "qwen":
            payload["enable_thinking"] = False  # JSON mode on Qwen-Plus
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError("Model response must be a JSON object")
                return result
            except HTTPError as exc:
                if exc.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"Model API returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError("Model request failed")
