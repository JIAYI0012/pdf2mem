"""Minimal OpenAI-compatible JSON client, replaceable for agent integration."""

from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ChatClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: int = 120):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY before extracting memories.")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

    def complete(self, *, model: str, system: str, text: str) -> dict:
        data = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        request = Request(self.base_url + "/chat/completions", data=data, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"
        }, method="POST")
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
