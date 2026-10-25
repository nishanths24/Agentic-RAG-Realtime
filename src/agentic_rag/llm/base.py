from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: float = 30.0,
    ) -> str: ...

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]: ...
