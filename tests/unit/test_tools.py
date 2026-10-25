from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from agentic_rag.errors import ToolValidationError
from agentic_rag.models import ToolResult
from agentic_rag.tools.calculator import CalculatorTool
from agentic_rag.tools.currency import CurrencyConvertTool
from agentic_rag.tools.current_time import CurrentTimeTool
from agentic_rag.tools.registry import ToolRegistry
from agentic_rag.tools.weather import WeatherTool


def test_time_valid_and_invalid():
    tool = CurrentTimeTool()
    result = tool.run({"timezone": "Asia/Kolkata"}, timeout_s=2)
    assert result.ok and "iso" in result.data
    with pytest.raises(ToolValidationError):
        tool.validate({"timezone": "Not/AZone"})


def test_calculator_safe_and_hostile():
    tool = CalculatorTool()
    assert tool.run({"expression": "(12.5 * 4) / 2"}, timeout_s=1).data["value"] == 25
    with pytest.raises(ToolValidationError):
        tool.validate({"expression": "__import__('os').system('x')"})
    with pytest.raises(ToolValidationError):
        tool.run({"expression": "2 ** 99"}, timeout_s=1)
    failed = tool.run({"expression": "1/0"}, timeout_s=1)
    assert not failed.ok


def test_currency_validation_and_mock():
    tool = CurrencyConvertTool()
    with pytest.raises(ToolValidationError):
        tool.validate({"amount": -1, "from_currency": "USD", "to_currency": "INR"})
    with pytest.raises(ToolValidationError):
        tool.validate({"amount": 10, "from_currency": "US", "to_currency": "INR"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"amount": 50, "base": "USD", "date": "2026-10-25", "rates": {"INR": 4200.0}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = CurrencyConvertTool(client=client).run(
        {"amount": 50, "from_currency": "USD", "to_currency": "INR"}, timeout_s=2
    )
    assert result.ok
    assert result.data["converted"] == "4200.00"


def test_weather_mock_and_bad_city():
    with pytest.raises(ToolValidationError):
        WeatherTool().validate({"city": ""})
    with pytest.raises(ToolValidationError):
        WeatherTool().validate({"city": "http://evil.example"})

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocoding" in url:
            return httpx.Response(
                200,
                json={"results": [{"name": "Paris", "latitude": 48.8, "longitude": 2.3, "country": "France"}]},
            )
        return httpx.Response(200, json={"current": {"temperature_2m": 18.5, "weather_code": 0, "wind_speed_10m": 3.2, "relative_humidity_2m": 50}})

    result = WeatherTool(client=httpx.Client(transport=httpx.MockTransport(handler))).run({"city": "Paris"}, timeout_s=2)
    assert result.ok
    assert result.data["temperature_c"] == 18.5


def test_registry_rejects_unknown():
    registry = ToolRegistry()
    with pytest.raises(Exception):
        registry.get("explode")
