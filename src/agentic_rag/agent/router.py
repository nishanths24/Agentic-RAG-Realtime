from __future__ import annotations

import re
from typing import Any

from agentic_rag.models import RoutingDecision, ToolRequest

_TIME = re.compile(r"\b(what time|current time|time in|timezone|clock)\b", re.I)
_WEATHER = re.compile(r"\b(weather|forecast|temperature|raining|rain in)\b", re.I)
_CURRENCY = re.compile(r"\b(convert|exchange rate|inr|usd|eur|gbp|jpy|currency|what's that in|what is that in)\b", re.I)
_CALC = re.compile(r"\b(calculate|compute)\b|\d+\s*[\+\-\*/]\s*\d+", re.I)
_DOCS = re.compile(
    r"\b(policy|reimburs|hotel|meal|allowance|receipt|rebook|travel|guideline|handbook|document|limit|per night|per day)\b",
    re.I,
)
_LIVE = re.compile(r"\b(today|right now|current|now|live|latest)\b", re.I)
_CITY = re.compile(
    r"\b(?:in|for|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)|\b(Tokyo|Paris|London|Delhi|Mumbai|Kolkata|New York|Berlin|Sydney)\b",
)
_TZ_MAP = {
    "tokyo": "Asia/Tokyo",
    "paris": "Europe/Paris",
    "london": "Europe/London",
    "delhi": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "new york": "America/New_York",
    "berlin": "Europe/Berlin",
    "sydney": "Australia/Sydney",
}
_ISO = re.compile(r"\b([A-Z]{3})\b")


class RuleRouter:
    def route(self, question: str) -> RoutingDecision:
        tools: list[ToolRequest] = []
        use_docs = bool(_DOCS.search(question))
        q = question.strip()

        if _TIME.search(q):
            city = _extract_city(q) or "Kolkata"
            timezone = _TZ_MAP.get(city.lower(), "UTC")
            tools.append(ToolRequest("current_time", {"timezone": timezone}))
        if _WEATHER.search(q):
            city = _extract_city(q) or "Paris"
            tools.append(ToolRequest("weather", {"city": city.split(",")[0]}))
        if _CURRENCY.search(q):
            src, dst = _extract_currencies(q)
            amount = _extract_amount(q)
            tools.append(
                ToolRequest(
                    "currency_convert",
                    {"amount": amount or 50, "from_currency": src, "to_currency": dst},
                )
            )
        if _CALC.search(q) and not tools:
            expr = _extract_expression(q)
            if expr:
                tools.append(ToolRequest("calculator", {"expression": expr}))

        if _LIVE.search(q) and not use_docs and not tools:
            use_docs = True

        if not use_docs and not tools:
            # Ambiguous policy-like wording still might need docs; otherwise abstain.
            if any(word in q.lower() for word in ("who won", "world cup", "secret", "password")):
                use_docs = False
            elif len(q.split()) >= 6:
                use_docs = True

        reasoning = _reason(use_docs, tools)
        confidence = 0.86 if tools or use_docs else 0.7
        return RoutingDecision(
            use_documents=use_docs,
            tools=tuple(tools),
            reasoning=reasoning,
            confidence=confidence,
            strategy="rules",
        )


def maybe_followup_currency(question: str, passage_text: str) -> ToolRequest | None:
    if not _CURRENCY.search(question) and "inr" not in question.lower() and "convert" not in question.lower():
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(USD|INR|EUR|GBP)", passage_text)
    if not match:
        return None
    amount = float(match.group(1))
    src = match.group(2)
    dst = "INR" if src != "INR" else "USD"
    q_codes = _ISO.findall(question.upper())
    if "INR" in q_codes:
        dst = "INR"
    if "USD" in q_codes and src != "USD":
        dst = "USD"
    if src == dst:
        return None
    return ToolRequest("currency_convert", {"amount": amount, "from_currency": src, "to_currency": dst})


def _extract_city(question: str) -> str | None:
    match = _CITY.search(question)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip() or None


def _extract_currencies(question: str) -> tuple[str, str]:
    codes = _ISO.findall(question.upper())
    codes = [c for c in codes if c in {"USD", "INR", "EUR", "GBP", "JPY", "AUD", "CAD"}]
    if len(codes) >= 2:
        return codes[0], codes[1]
    if len(codes) == 1:
        other = "INR" if codes[0] != "INR" else "USD"
        if "inr" in question.lower():
            return ("USD", "INR") if codes[0] == "USD" else (codes[0], "INR")
        return codes[0], other
    if "inr" in question.lower():
        return "USD", "INR"
    return "USD", "INR"


def _extract_amount(question: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(USD|INR|EUR|GBP|dollars)?", question, re.I)
    if match:
        return float(match.group(1))
    return None


def _extract_expression(question: str) -> str | None:
    match = re.search(r"(\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\d+(?:\.\d+)?)*)", question)
    return match.group(1).strip() if match else None


def _reason(use_docs: bool, tools: list[ToolRequest]) -> str:
    names = [t.tool_name for t in tools]
    if use_docs and names:
        return f"Mixed intent: retrieve documents and call {', '.join(names)}."
    if names:
        return f"Live question: call {', '.join(names)}."
    if use_docs:
        return "Policy/document question: retrieve local sources only."
    return "No reliable source matched; abstain unless retrieval finds evidence."
