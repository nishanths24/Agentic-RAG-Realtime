from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from agentic_rag.errors import ToolTimeoutError, ToolUnavailableError, ToolValidationError
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
    with pytest.raises(ToolValidationError):
        tool.validate({"amount": 10, "from_currency": "USD", "to_currency": "USD"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"amount": 50, "base": "USD", "date": "2026-10-25", "rates": {"INR": 4200.0}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = CurrencyConvertTool(client=client).run(
        {"amount": 50, "from_currency": "USD", "to_currency": "INR"}, timeout_s=2
    )
    assert result.ok
    assert result.data["converted"] == "4200.00"
    assert result.data["rate"] == "84.0000"


def test_currency_api_error_handling():
    # 1. HTTP 500 error
    client_500 = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500, text="Internal Error")))
    with pytest.raises(ToolUnavailableError, match="HTTP 500"):
        CurrencyConvertTool(client=client_500).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})

    # 2. Timeout
    def timeout_handler(req: httpx.Request):
        raise httpx.ReadTimeout("Timed out")

    client_timeout = httpx.Client(transport=httpx.MockTransport(timeout_handler))
    with pytest.raises(ToolTimeoutError, match="timed out"):
        CurrencyConvertTool(client=client_timeout).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})

    # 3. Invalid JSON / HTML payload
    client_html = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, text="<html>Redirect</html>")))
    with pytest.raises(ToolUnavailableError, match="unexpected payload"):
        CurrencyConvertTool(client=client_html).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})

    # 4. Missing rates key
    client_no_rates = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"amount": 50, "base": "USD"})))
    with pytest.raises(ToolUnavailableError, match="unexpected payload"):
        CurrencyConvertTool(client=client_no_rates).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})

    # 5. Missing target currency in rates
    client_wrong_currency = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"amount": 50, "rates": {"EUR": 45.0}}))
    )
    with pytest.raises(ToolUnavailableError, match="unexpected payload"):
        CurrencyConvertTool(client=client_wrong_currency).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})

    # 6. Zero rate
    client_zero = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"amount": 50, "rates": {"INR": 0}}))
    )
    with pytest.raises(ToolUnavailableError, match="invalid exchange rate"):
        CurrencyConvertTool(client=client_zero).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})

    # 7. Negative rate
    client_neg = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"amount": 50, "rates": {"INR": -4200.0}}))
    )
    with pytest.raises(ToolUnavailableError, match="invalid exchange rate"):
        CurrencyConvertTool(client=client_neg).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})

    # 8. NaN rate
    client_nan = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"amount": 50, "rates": {"INR": "NaN"}}))
    )
    with pytest.raises(ToolUnavailableError, match="invalid exchange rate"):
        CurrencyConvertTool(client=client_nan).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})

    # 9. Infinite rate
    client_inf = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"amount": 50, "rates": {"INR": "Infinity"}}))
    )
    with pytest.raises(ToolUnavailableError, match="invalid exchange rate"):
        CurrencyConvertTool(client=client_inf).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})


def test_currency_redirect_follow():
    def redirect_handler(req: httpx.Request) -> httpx.Response:
        if "app" in str(req.url):
            return httpx.Response(301, headers={"Location": "https://api.frankfurter.dev/v1/latest?amount=50&from=USD&to=INR"})
        return httpx.Response(200, json={"amount": 50, "base": "USD", "date": "2026-10-25", "rates": {"INR": 4200.0}})

    client = httpx.Client(transport=httpx.MockTransport(redirect_handler))
    result = CurrencyConvertTool(client=client).run({"amount": 50, "from_currency": "USD", "to_currency": "INR"})
    assert result.ok
    assert result.data["converted"] == "4200.00"


def test_currency_configured_endpoint():
    def custom_handler(req: httpx.Request) -> httpx.Response:
        assert "custom-rates.example" in str(req.url)
        return httpx.Response(200, json={"amount": 100, "base": "EUR", "date": "2026-10-25", "rates": {"USD": 108.5}})

    client = httpx.Client(transport=httpx.MockTransport(custom_handler))
    tool = CurrencyConvertTool(client=client, endpoint="https://custom-rates.example/v1/latest")
    result = tool.run({"amount": 100, "from_currency": "EUR", "to_currency": "USD"})
    assert result.ok
    assert result.data["converted"] == "108.50"


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
