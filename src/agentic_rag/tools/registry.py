from __future__ import annotations

from typing import Any

from agentic_rag.errors import ToolUnavailableError, ToolValidationError
from agentic_rag.models import ToolResult
from agentic_rag.tools.calculator import CalculatorTool
from agentic_rag.tools.currency import CurrencyConvertTool
from agentic_rag.tools.current_time import CurrentTimeTool
from agentic_rag.tools.weather import WeatherTool


class ToolRegistry:
    def __init__(self, tools: list[Any] | None = None) -> None:
        self._tools = {tool.name: tool for tool in (tools or default_tools())}

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Any:
        if name not in self._tools:
            raise ToolUnavailableError(f"Unknown or disallowed tool: {name}")
        return self._tools[name]

    def schemas(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "description": tool.description,
                "args_schema": tool.args_schema,
                "requires_credentials": tool.requires_credentials,
                "available": True,
            }
            for name, tool in self._tools.items()
        }

    def run(self, name: str, arguments: dict[str, Any], *, timeout_s: float) -> ToolResult:
        tool = self.get(name)
        validated = tool.validate(arguments)
        return tool.run(validated, timeout_s=timeout_s)


def default_tools() -> list[Any]:
    return [CurrentTimeTool(), CalculatorTool(), CurrencyConvertTool(), WeatherTool()]
