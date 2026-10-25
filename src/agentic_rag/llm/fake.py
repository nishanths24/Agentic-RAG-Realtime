from __future__ import annotations

from typing import Any


class FakeLLMClient:
    def __init__(self, responses: list[str] | None = None, json_responses: list[dict] | None = None) -> None:
        self.responses = list(responses or [])
        self.json_responses = list(json_responses or [])
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: float = 30.0,
    ) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if not self.responses:
            return "Insufficient evidence."
        return self.responses.pop(0)

    def generate_json(self, system: str, user: str, schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if not self.json_responses:
            return {}
        return self.json_responses.pop(0)
