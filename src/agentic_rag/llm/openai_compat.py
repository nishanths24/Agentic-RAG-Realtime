from __future__ import annotations

import json
from typing import Any

import httpx

from agentic_rag.errors import LLMAuthError, LLMRateLimitError, LLMResponseError, LLMTimeoutError


class OpenAICompatClient:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: float = 30.0,
    ) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMResponseError("LLM request failed") from exc
        if response.status_code in (401, 403):
            raise LLMAuthError("LLM authentication failed")
        if response.status_code == 429:
            raise LLMRateLimitError("LLM rate limited")
        if response.status_code >= 400:
            raise LLMResponseError(f"LLM HTTP {response.status_code}")
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMResponseError("LLM returned an unexpected payload") from exc

    def generate_json(self, system: str, user: str, schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        text = self.generate(system, user + "\nRespond with JSON only.", **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("LLM did not return JSON") from exc
