from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agentic_rag.errors import ToolValidationError
from agentic_rag.models import ToolResult


class CurrentTimeTool:
    name = "current_time"
    description = "Return the current date and time in an IANA timezone."
    args_schema = {
        "type": "object",
        "required": ["timezone"],
        "properties": {
            "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Tokyo"},
        },
    }
    requires_credentials = False

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        tz = str(arguments.get("timezone", "")).strip()
        if not tz or len(tz) > 64:
            raise ToolValidationError("timezone is required and must be a valid IANA name")
        try:
            ZoneInfo(tz)
        except ZoneInfoNotFoundError as exc:
            raise ToolValidationError(f"Unknown timezone: {tz}") from exc
        return {"timezone": tz}

    def run(self, arguments: dict[str, Any], *, timeout_s: float = 5.0) -> ToolResult:
        args = self.validate(arguments)
        now = datetime.now(ZoneInfo(args["timezone"]))
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data={
                "timezone": args["timezone"],
                "iso": now.isoformat(),
                "display": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
            },
            error=None,
            retrieved_at=now,
            source="zoneinfo",
        )
