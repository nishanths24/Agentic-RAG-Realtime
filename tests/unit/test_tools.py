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


def test_knowledge_tool_wikipedia_and_various_phrasings():
    from agentic_rag.tools.knowledge import KnowledgeTool

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "summary/Mysore" in url_str or "summary/Mysuru" in url_str:
            return httpx.Response(
                200,
                json={
                    "extract": "Mysore, officially Mysuru, is a city in the southern Indian state of Karnataka.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Mysore"}},
                },
            )
        if "summary/Chennai" in url_str:
            return httpx.Response(
                200,
                json={
                    "extract": "Chennai is the capital and largest city of Tamil Nadu, India.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Chennai"}},
                },
            )
        if "summary/Hyderabad" in url_str:
            return httpx.Response(
                200,
                json={
                    "extract": "Hyderabad is the capital and largest city of the Indian state of Telangana.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Hyderabad"}},
                },
            )
        if "summary/Bengaluru" in url_str:
            return httpx.Response(
                200,
                json={
                    "extract": "Bengaluru is the capital of the Indian state of Karnataka.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Bengaluru"}},
                },
            )
        if "summary/Kyoto" in url_str:
            return httpx.Response(
                200,
                json={
                    "extract": "Kyoto is the capital city of Kyoto Prefecture in Japan.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Kyoto"}},
                },
            )
        return httpx.Response(404, json={"detail": "Not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = KnowledgeTool(client=client)

    # 1. Mysore / Mysuru
    r1 = tool.run({"query": "where Mysore is located"})
    assert r1.ok
    assert "Karnataka" in r1.data["answer"]
    assert "https://en.wikipedia.org/wiki/Mysore" in r1.data["answer"]

    r2 = tool.run({"query": "Where is Mysuru situated?"})
    assert r2.ok
    assert "Karnataka" in r2.data["answer"]

    # 2. Chennai
    r3 = tool.run({"query": "Which state is Chennai in?"})
    assert r3.ok
    assert "Tamil Nadu" in r3.data["answer"]

    # 3. Hyderabad
    r4 = tool.run({"query": "Tell me about Hyderabad's location."})
    assert r4.ok
    assert "Telangana" in r4.data["answer"]

    # 4. Bengaluru
    r5 = tool.run({"query": "Where is Bengaluru located?"})
    assert r5.ok
    assert "Karnataka" in r5.data["answer"]

    # 5. Unseen city (Kyoto)
    r6 = tool.run({"query": "Where is Kyoto located?"})
    assert r6.ok
    assert "Japan" in r6.data["answer"]


def test_knowledge_tool_web_search_api():
    from agentic_rag.tools.knowledge import KnowledgeTool
    import json

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["api_key"] == "tvly-test-key"
        assert body["query"] == "Where is Mysore located?"
        return httpx.Response(
            200,
            json={
                "answer": "Mysore is a city located in Karnataka, India.",
                "results": [{"url": "https://example.com/mysore", "content": "Mysore is in Karnataka."}],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = KnowledgeTool(
        search_api_key="tvly-test-key",
        search_provider="tavily",
        client=client,
    )
    result = tool.run({"query": "Where is Mysore located?"})
    assert result.ok
    assert "Karnataka" in result.data["answer"]
    assert "https://example.com/mysore" in result.data["answer"]


def test_knowledge_tool_geocoding_fallback():
    from agentic_rag.tools.knowledge import KnowledgeTool

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "summary" in url_str:
            return httpx.Response(404, json={"detail": "Not found"})
        if "geocoding" in url_str:
            return httpx.Response(
                200,
                json={"results": [{"name": "Pune", "admin1": "Maharashtra", "country": "India"}]},
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = KnowledgeTool(client=client)
    res = tool.run({"query": "where is Pune located"})
    assert res.ok
    assert "Maharashtra" in res.data["answer"]
    assert "India" in res.data["answer"]


def test_knowledge_tool_error_handling():
    from agentic_rag.tools.knowledge import KnowledgeTool

    # 1. Empty query
    with pytest.raises(ToolValidationError):
        KnowledgeTool().validate({"query": ""})

    # 2. Unavailable
    with pytest.raises(ToolUnavailableError, match="unavailable"):
        KnowledgeTool(available=False).run({"query": "Where is Paris?"})

    # 3. Network timeout
    def timeout_handler(req: httpx.Request):
        raise httpx.ReadTimeout("Timed out")

    client_timeout = httpx.Client(transport=httpx.MockTransport(timeout_handler))
    with pytest.raises(ToolTimeoutError):
        KnowledgeTool(client=client_timeout).run({"query": "Where is Paris?"})

    # 4. Search API 401 Auth error
    def auth_err_handler(req: httpx.Request):
        return httpx.Response(401, json={"detail": "Unauthorized"})

    client_auth = httpx.Client(transport=httpx.MockTransport(auth_err_handler))
    with pytest.raises(ToolUnavailableError, match="authentication failed"):
        KnowledgeTool(search_api_key="bad-key", search_provider="tavily", client=client_auth).run({"query": "Where is Paris?"})
