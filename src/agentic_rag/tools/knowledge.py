from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from agentic_rag.errors import ToolTimeoutError, ToolUnavailableError, ToolValidationError
from agentic_rag.models import ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = "AgenticRAG/1.0 (agentic-rag@example.com)"
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WIKIPEDIA_REST_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

_ENTITY_PATTERNS = [
    re.compile(r"\bwhere\s+(?:\b(?:is|are)\b\s+)?([a-zA-Z\s'-]+?)(?:\s+\b(?:is|are)\b)?(?:\s+\b(?:located|situated)\b)?$", re.I),
    re.compile(r"\b(?:which|what)\s+(?:state|country|nation|continent|region|province)\s+(?:\b(?:is|are)\b\s+)?([a-zA-Z\s'-]+?)(?:\s+\b(?:is|are)\b)?(?:\s+\b(?:in|located\s+in|part\s+of)\b)?$", re.I),
    re.compile(r"\b(?:tell\s+me\s+about|information\s+about)\s+([a-zA-Z\s'-]+?)(?:'s\s+location|\s+location)?$", re.I),
    re.compile(r"\b(?:what\s+is\s+the\s+capital\s+of|capital\s+city\s+of)\s+([a-zA-Z\s'-]+?)$", re.I),
]


def extract_entity(query: str) -> str:
    q = query.strip().rstrip("?.!")
    for pattern in _ENTITY_PATTERNS:
        match = pattern.search(q)
        if match:
            cand = match.group(1).strip()
            if cand and len(cand.split()) <= 4:
                return cand
    return q


class KnowledgeTool:
    name = "knowledge"
    description = "Provides factual answers to general-knowledge questions via dynamic web-search and knowledge APIs."
    args_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
    }
    requires_credentials = False

    def __init__(
        self,
        *,
        search_api_key: str = "",
        search_provider: str = "wikipedia",
        search_endpoint: str = "https://api.tavily.com/search",
        client: httpx.Client | None = None,
        available: bool = True,
    ) -> None:
        self.search_api_key = search_api_key
        self.search_provider = search_provider
        self.search_endpoint = search_endpoint
        self._client = client
        self.available = available

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolValidationError("query must not be empty")
        return {"query": query}

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(headers={"User-Agent": _DEFAULT_USER_AGENT})

    def run(self, arguments: dict[str, Any], *, timeout_s: float = 15.0) -> ToolResult:
        if not self.available:
            raise ToolUnavailableError("General knowledge provider is currently unavailable or unconfigured.")

        args = self.validate(arguments)
        query = args["query"]

        # 1. External search API integration (Tavily/Brave) when SEARCH_API_KEY is configured
        if self.search_api_key and self.search_provider != "wikipedia":
            return self._search_via_api(query, timeout_s=timeout_s)

        # 2. Dynamic knowledge API integration (Wikipedia REST API + Open-Meteo Geocoding)
        return self._search_via_knowledge_api(query, timeout_s=timeout_s)

    def _search_via_api(self, query: str, timeout_s: float) -> ToolResult:
        client = self._get_client()
        payload = {
            "api_key": self.search_api_key,
            "query": query,
            "include_answer": True,
            "max_results": 3,
        }
        try:
            resp = client.post(self.search_endpoint, json=payload, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError(f"Web search API timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ToolUnavailableError(f"Web search API request failed: {exc}") from exc

        if resp.status_code in (401, 403):
            raise ToolUnavailableError(
                f"Web search authentication failed (HTTP {resp.status_code}). Check SEARCH_API_KEY."
            )
        if resp.status_code >= 400:
            raise ToolUnavailableError(f"Web search API error (HTTP {resp.status_code}).")

        try:
            data = resp.json()
            answer = data.get("answer")
            results = data.get("results") or []
            source_url = results[0].get("url") if results else ""
            if not answer and results:
                content = str(results[0].get("content") or "").strip()
                if content:
                    answer = re.split(r"(?<=[.!?])\s+", content)[0]

            if not answer:
                raise ToolUnavailableError(f"No verified information found from search provider for {query!r}.")

            clean_answer = str(answer).strip().rstrip(".")
            final_text = f"{clean_answer} (Source: {source_url})" if source_url else clean_answer
            return ToolResult(
                tool_name=self.name,
                ok=True,
                data={"query": query, "answer": final_text, "source_url": source_url},
                error=None,
                retrieved_at=datetime.now(timezone.utc),
                source=f"web-search:{self.search_provider}",
            )
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ToolUnavailableError(f"Unexpected search API payload: {exc}") from exc

    def _search_via_knowledge_api(self, query: str, timeout_s: float) -> ToolResult:
        client = self._get_client()
        entity = extract_entity(query)
        encoded_entity = quote(entity.replace(" ", "_"))

        # Primary: Wikipedia REST API
        wiki_url = f"{_WIKIPEDIA_REST_URL}/{encoded_entity}"
        try:
            wiki_resp = client.get(wiki_url, timeout=timeout_s)
            if wiki_resp.status_code == 200:
                wiki_data = wiki_resp.json()
                extract = str(wiki_data.get("extract", "")).strip()
                page_url = (
                    wiki_data.get("content_urls", {}).get("desktop", {}).get("page")
                    or f"https://en.wikipedia.org/wiki/{encoded_entity}"
                )
                if extract:
                    first_sent = re.split(r"(?<=[.!?])\s+", extract)[0].strip().rstrip(".")
                    ans = f"{first_sent} (Source: {page_url})"
                    return ToolResult(
                        tool_name=self.name,
                        ok=True,
                        data={"query": query, "answer": ans, "source_url": page_url},
                        error=None,
                        retrieved_at=datetime.now(timezone.utc),
                        source="wikipedia",
                    )
        except httpx.TimeoutException as exc:
            logger.warning("Wikipedia summary API timed out for %r: %s", entity, exc)
        except httpx.HTTPError as exc:
            logger.warning("Wikipedia summary API HTTP error for %r: %s", entity, exc)

        # Fallback: Open-Meteo Geocoding
        try:
            geo_resp = client.get(_GEOCODE_URL, params={"name": entity, "count": 1}, timeout=timeout_s)
            if geo_resp.status_code == 200:
                geo_data = geo_resp.json()
                results = geo_data.get("results") or []
                if results:
                    place = results[0]
                    name = place.get("name")
                    admin1 = place.get("admin1")
                    country = place.get("country")
                    loc_parts = [p for p in (admin1, country) if p]
                    loc_str = ", ".join(loc_parts)
                    ans = f"{name} is located in {loc_str}. (Source: {_GEOCODE_URL})"
                    return ToolResult(
                        tool_name=self.name,
                        ok=True,
                        data={"query": query, "answer": ans, "source_url": _GEOCODE_URL},
                        error=None,
                        retrieved_at=datetime.now(timezone.utc),
                        source="geocoding",
                    )
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError(f"Knowledge geocoding timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ToolUnavailableError(f"Knowledge geocoding unavailable: {exc}") from exc

        raise ToolUnavailableError(f"No verified information found from knowledge provider for {query!r}.")
