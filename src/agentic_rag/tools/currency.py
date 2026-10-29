import os
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

import httpx

from agentic_rag.errors import ToolTimeoutError, ToolUnavailableError, ToolValidationError
from agentic_rag.models import ToolResult

_CODE = re.compile(r"^[A-Z]{3}$")
_DEFAULT_FRANKFURTER = "https://api.frankfurter.dev/v1/latest"


class CurrencyConvertTool:
    name = "currency_convert"
    description = "Convert an amount between ISO 4217 currencies using live ECB rates."
    args_schema = {
        "type": "object",
        "required": ["amount", "from_currency", "to_currency"],
        "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        },
    }
    requires_credentials = False

    def __init__(self, client: httpx.Client | None = None, endpoint: str | None = None) -> None:
        self._client = client
        self._endpoint = (
            endpoint
            or os.getenv("FRANKFURTER_URL")
            or os.getenv("FRANKFURTER_BASE_URL")
            or os.getenv("CURRENCY_API_URL")
            or _DEFAULT_FRANKFURTER
        )

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            amount = Decimal(str(arguments.get("amount")))
        except (InvalidOperation, TypeError) as exc:
            raise ToolValidationError("amount must be a number") from exc
        if amount <= 0 or amount > Decimal("1000000000"):
            raise ToolValidationError("amount must be positive and within a reasonable range")
        src = str(arguments.get("from_currency", "")).strip().upper()
        dst = str(arguments.get("to_currency", "")).strip().upper()
        if not _CODE.match(src) or not _CODE.match(dst):
            raise ToolValidationError("currency codes must be 3-letter ISO 4217 values")
        if src == dst:
            raise ToolValidationError("from_currency and to_currency must differ")
        return {"amount": amount, "from_currency": src, "to_currency": dst}

    def run(self, arguments: dict[str, Any], *, timeout_s: float = 15.0) -> ToolResult:
        args = self.validate(arguments)
        params = {
            "amount": str(args["amount"]),
            "from": args["from_currency"],
            "to": args["to_currency"],
        }
        try:
            if self._client is None:
                response = httpx.get(self._endpoint, params=params, timeout=timeout_s, follow_redirects=True)
            else:
                response = self._client.get(self._endpoint, params=params, timeout=timeout_s, follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError("Currency API timed out") from exc
        except httpx.HTTPError as exc:
            raise ToolUnavailableError("Currency API is unavailable") from exc

        if response.status_code != 200:
            raise ToolUnavailableError(f"Currency API HTTP {response.status_code}")
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ToolUnavailableError("Currency API returned an unexpected payload")
            rates = payload.get("rates")
            if not isinstance(rates, dict) or args["to_currency"] not in rates:
                raise ToolUnavailableError("Currency API returned an unexpected payload")
            converted = Decimal(str(rates[args["to_currency"]]))
            if converted.is_nan() or converted.is_infinite() or converted <= 0:
                raise ToolUnavailableError("Currency API returned an invalid exchange rate")
            rate = converted / Decimal(str(args["amount"]))
            if rate.is_nan() or rate.is_infinite() or rate <= 0:
                raise ToolUnavailableError("Currency API returned an invalid exchange rate")
        except ToolUnavailableError:
            raise
        except (KeyError, TypeError, ValueError, InvalidOperation, ZeroDivisionError) as exc:
            raise ToolUnavailableError("Currency API returned an unexpected payload") from exc

        converted = converted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        rate = rate.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        now = datetime.now(timezone.utc)
        host = getattr(getattr(response, "url", None), "host", None) or "api.frankfurter.dev"
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data={
                "amount": str(args["amount"]),
                "from": args["from_currency"],
                "to": args["to_currency"],
                "rate": str(rate),
                "converted": str(converted),
                "as_of": payload.get("date"),
            },
            error=None,
            retrieved_at=now,
            source=host,
        )
