from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from agentic_rag.errors import ToolTimeoutError, ToolUnavailableError, ToolValidationError
from agentic_rag.models import ToolResult

_CITY = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,63}$")
_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST = "https://api.open-meteo.com/v1/forecast"


class WeatherTool:
    name = "weather"
    description = "Current weather for a city using Open-Meteo."
    args_schema = {
        "type": "object",
        "required": ["city"],
        "properties": {"city": {"type": "string"}},
    }
    requires_credentials = False

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        city = str(arguments.get("city", "")).strip()
        if not city or not _CITY.match(city):
            raise ToolValidationError("city must be a short alphabetic place name")
        return {"city": city}

    def _get(self, url: str, params: dict[str, Any], timeout_s: float) -> httpx.Response:
        try:
            if self._client is None:
                return httpx.get(url, params=params, timeout=timeout_s)
            return self._client.get(url, params=params, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError("Weather API timed out") from exc
        except httpx.HTTPError as exc:
            raise ToolUnavailableError("Weather API is unavailable") from exc

    def run(self, arguments: dict[str, Any], *, timeout_s: float = 15.0) -> ToolResult:
        args = self.validate(arguments)
        geo = self._get(_GEOCODE, {"name": args["city"], "count": 1}, timeout_s)
        if geo.status_code >= 400:
            raise ToolUnavailableError(f"Geocoding HTTP {geo.status_code}")
        try:
            results = geo.json().get("results") or []
            place = results[0]
            lat = place["latitude"]
            lon = place["longitude"]
            label = ", ".join(
                part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ToolUnavailableError(f"Could not geocode city {args['city']!r}") from exc

        forecast = self._get(
            _FORECAST,
            {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            },
            timeout_s,
        )
        if forecast.status_code >= 400:
            raise ToolUnavailableError(f"Forecast HTTP {forecast.status_code}")
        try:
            current = forecast.json()["current"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolUnavailableError("Forecast payload was malformed") from exc

        return ToolResult(
            tool_name=self.name,
            ok=True,
            data={
                "city": args["city"],
                "location": label,
                "temperature_c": current.get("temperature_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "humidity": current.get("relative_humidity_2m"),
                "weather_code": current.get("weather_code"),
                "summary": _code_to_text(current.get("weather_code")),
            },
            error=None,
            retrieved_at=datetime.now(timezone.utc),
            source="api.open-meteo.com",
        )


def _code_to_text(code: Any) -> str:
    mapping = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        61: "rain",
        71: "snow",
        80: "rain showers",
        95: "thunderstorm",
    }
    try:
        return mapping.get(int(code), f"weather code {code}")
    except (TypeError, ValueError):
        return "unknown conditions"
