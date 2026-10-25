from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agentic_rag.models import ToolResult


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    args_schema: dict[str, Any]
    requires_credentials: bool

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def run(self, arguments: dict[str, Any], *, timeout_s: float) -> ToolResult: ...
