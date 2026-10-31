from __future__ import annotations

import re
from typing import Any

from agentic_rag.models import RoutingDecision, ToolRequest

_TIME = re.compile(r"\b(what time|current time|time in|timezone|clock)\b", re.I)
_WEATHER = re.compile(r"\b(weather|forecast|temperature|raining|rain in)\b", re.I)
_CURRENCY = re.compile(r"\b(convert|exchange rate|inr|usd|eur|gbp|jpy|currency|what's that in|what is that in)\b", re.I)
_CONVERT_INTENT = re.compile(
    r"\b(convert|converted|conversion|exchange|equivalent|what's that in|what is that in|how much is .* in)\b",
    re.I,
)
_CALC = re.compile(r"\b(calculate|compute|solve|math|evaluate)\b|\d+\s*[\+\-\*/\^×xX÷]\s*\d+", re.I)
_DOCS = re.compile(
    r"\b(policy|reimburs|hotel|meal|allowance|receipt|rebook|travel|guideline|handbook|document|documents|limit|per night|per day|project|brief|summary|summarize|overview|topic|server|confidential|company|corporate|internal|employee|expenses?|ceo|vp|hr|salary)\b",
    re.I,
)
_LIVE = re.compile(r"\b(today|right now|current|now|live|latest)\b", re.I)
_CITY = re.compile(
    r"\b(?:in|for|at)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)?)|"
    r"\b(Tokyo|Paris|London|Delhi|Mumbai|Kolkata|New York|Berlin|Sydney)\b",
    re.I,
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

_GENERAL_KNOWLEDGE_PATTERNS = [
    re.compile(r"\bwhere\s+(?:\b(?:is|are)\b\s+)?([a-zA-Z\s'-]+?)(?:\s+\b(?:is|are)\b)?(?:\s+\b(?:located|situated)\b)?\??$", re.I),
    re.compile(r"\b(?:which|what)\s+(?:state|country|nation|continent|region|province)\s+(?:\b(?:is|are)\b\s+)?([a-zA-Z\s'-]+?)(?:\s+\b(?:is|are)\b)?(?:\s+\b(?:in|located\s+in|part\s+of)\b)?\??$", re.I),
    re.compile(r"\b(?:what\s+is\s+the\s+capital\s+of|capital\s+city\s+of)\s+([a-zA-Z\s'-]+?)\??$", re.I),
    re.compile(r"\b(?:tell\s+me\s+about|information\s+about)\s+([a-zA-Z\s'-]+?)(?:'s\s+location|\s+location)?\??$", re.I),
]


def _extract_knowledge_query(question: str) -> str | None:
    q = question.strip().rstrip("?.!")
    if _DOCS.search(q) or _TIME.search(q) or _WEATHER.search(q) or _CURRENCY.search(q) or _CALC.search(q):
        return None
    if re.search(r"\b(confidential|secret|password|server|internal|company|corporate|ceo|employee|salary)\b", q, re.I):
        return None
    for pattern in _GENERAL_KNOWLEDGE_PATTERNS:
        match = pattern.search(q)
        if match:
            place = match.group(1).strip()
            words = [w.lower() for w in re.findall(r"[a-zA-Z]+", place)]
            if 1 <= len(words) <= 4 and not any(w in {"policy", "reimbursement", "project", "code", "document", "hotel", "meal"} for w in words):
                return question.strip()
    return None


def _has_conversion_intent(question: str) -> bool:
    if _CONVERT_INTENT.search(question):
        return True
    if _LIVE.search(question) and re.search(r"\b(rate|rates|currency|inr|usd|eur|gbp|jpy)\b", question, re.I):
        return True
    codes = [c for c in _ISO.findall(question.upper()) if c in {"USD", "INR", "EUR", "GBP", "JPY", "AUD", "CAD"}]
    if len(set(codes)) >= 2:
        return True
    return False


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
            should_convert = True
            if use_docs:
                should_convert = _has_conversion_intent(q)
            if should_convert:
                src, dst = _extract_currencies(q)
                amount = _extract_amount(q)
                tools.append(
                    ToolRequest(
                        "currency_convert",
                        {"amount": amount or 50, "from_currency": src, "to_currency": dst},
                    )
                )
        if _CALC.search(q) and not any(t.tool_name == "calculator" for t in tools):
            expr = _extract_expression(q)
            if expr:
                tools.append(ToolRequest("calculator", {"expression": expr}))
                if not re.search(r"\b(policy|reimburs|hotel|meal|allowance|travel|receipt)\b", q, re.I):
                    use_docs = False

        if not tools and not use_docs:
            gk_q = _extract_knowledge_query(q)
            if gk_q:
                tools.append(ToolRequest("knowledge", {"query": q}))
                use_docs = False

        if _LIVE.search(q) and not use_docs and not tools:
            use_docs = True

        if not use_docs and not tools:
            # Questions not matching live tools should search documents unless explicitly abstaining on trivia
            if any(word in q.lower() for word in ("who won", "world cup", "secret", "password")):
                use_docs = False
            elif len(q.split()) >= 2:
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
    if (
        not _CURRENCY.search(question)
        and "inr" not in question.lower()
        and "convert" not in question.lower()
        and "equivalent" not in question.lower()
    ):
        return None

    q_src, q_dst = _extract_currencies(question)

    is_meal = bool(re.search(r"\b(meal|allowance|food|dinner|lunch|breakfast)\b", question, re.I))
    is_hotel = bool(re.search(r"\b(hotel|lodging|room)\b", question, re.I))

    # Split into sentences or clauses (periods, newlines, semicolons)
    segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+|;\s*", passage_text) if s.strip()]

    # Filter segments relevant to the topic
    if is_meal:
        topic_segments = [s for s in segments if re.search(r"\b(meal|allowance|food|dinner|lunch|breakfast)\b", s, re.I)]
    elif is_hotel:
        topic_segments = [s for s in segments if re.search(r"\b(hotel|lodging|room)\b", s, re.I)]
    else:
        topic_segments = segments

    candidate_segments = topic_segments if topic_segments else segments

    matches_found: list[dict[str, Any]] = []
    amt_regex = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Z]{3})\b")

    codes_in_q = {c for c in _ISO.findall(question.upper()) if c in {"USD", "INR", "EUR", "GBP", "JPY", "AUD", "CAD"}}
    src_explicit = (q_src in codes_in_q) or bool(re.search(r"\b(dollars?|cents?|rupees?|euros?|pounds?)\b", question, re.I))
    topic_words = r"\b(meal|allowance|food|dinner|lunch|breakfast)\b" if is_meal else (r"\b(hotel|lodging|room)\b" if is_hotel else None)
    has_target_in_topic = False
    if topic_words:
        for block in re.split(r"\n\s*\n+", passage_text):
            if re.search(topic_words, block, re.I):
                for m in amt_regex.finditer(block):
                    if m.group(2).upper() == q_dst:
                        has_target_in_topic = True
                        break
            if has_target_in_topic:
                break
    else:
        has_target_in_topic = any(
            m.group(2).upper() == q_dst
            for seg in candidate_segments
            for m in amt_regex.finditer(seg)
        )

    if has_target_in_topic and not src_explicit and not _has_conversion_intent(question):
        return None

    for seg in candidate_segments:
        for m in amt_regex.finditer(seg):
            amt = float(m.group(1))
            curr = m.group(2).upper()
            if curr == q_dst:
                continue

            # Check context around the amount within 35 characters
            pre_context = seg[max(0, m.start() - 35) : m.start()].lower()
            post_context = seg[m.end() : min(len(seg), m.end() + 35)].lower()
            context = f"{pre_context} {post_context}"

            is_hotel_specific = is_meal and bool(re.search(r"\b(hotel|lodging|room)\b", context))
            is_meal_specific = is_meal and bool(re.search(r"\b(meal|allowance|food|dinner|lunch)\b", context))

            dist = 999
            if is_meal:
                t_matches = list(re.finditer(r"\b(meal|allowance)\b", seg, re.I))
                if t_matches:
                    dist = min(abs(m.start() - tm.start()) for tm in t_matches)
            elif is_hotel:
                t_matches = list(re.finditer(r"\b(hotel|lodging)\b", seg, re.I))
                if t_matches:
                    dist = min(abs(m.start() - tm.start()) for tm in t_matches)
            else:
                dist = 0

            matches_found.append({
                "amt": amt,
                "curr": curr,
                "is_hotel_specific": is_hotel_specific,
                "is_meal_specific": is_meal_specific,
                "matches_src": (curr == q_src) if q_src else False,
                "dist": dist,
            })

    if not matches_found:
        return None

    matches_found.sort(
        key=lambda x: (
            1 if x["is_hotel_specific"] else 0,
            0 if x["is_meal_specific"] else 1,
            0 if x["matches_src"] else 1,
            x["dist"],
        )
    )

    best = matches_found[0]
    amount = best["amt"]
    src = best["curr"]

    dst = q_dst if src != q_dst else ("USD" if src == "INR" else "INR")
    if src == dst:
        return None

    amount_val: float | int = int(amount) if int(amount) == amount else amount
    return ToolRequest("currency_convert", {"amount": amount_val, "from_currency": src, "to_currency": dst})


def _extract_city(question: str) -> str | None:
    lowered = question.lower()
    for name in sorted(_TZ_MAP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return "New York" if name == "new york" else name.title()
    match = _CITY.search(question)
    if not match:
        return None
    raw = (match.group(1) or match.group(2) or "").strip()
    if not raw:
        return None
    return raw.split()[0]


def _extract_currencies(question: str) -> tuple[str, str]:
    # 1. '[TARGET] equivalent of [SOURCE]' -> (SOURCE, TARGET)
    m_equiv = re.search(r"\b([A-Z]{3})\s+equivalent\s+(?:of|for)\b.*?\b([A-Z]{3})\b", question, re.I)
    if m_equiv:
        src, dst = m_equiv.group(2).upper(), m_equiv.group(1).upper()
        if src != dst:
            return src, dst

    # 2. 'equivalent of [SOURCE] in/into/to [TARGET]' -> (SOURCE, TARGET)
    m_equiv2 = re.search(r"\bequivalent\s+of\b.*?\b([A-Z]{3})\b.*?\b(?:in|into|to)\s+([A-Z]{3})\b", question, re.I)
    if m_equiv2:
        src, dst = m_equiv2.group(1).upper(), m_equiv2.group(2).upper()
        if src != dst:
            return src, dst

    # 3. '[SOURCE] (to|into|in) [TARGET]' -> (SOURCE, TARGET)
    m_to = re.search(r"\b([A-Z]{3})\s+(?:to|into|in)\s+([A-Z]{3})\b", question, re.I)
    if m_to:
        src, dst = m_to.group(1).upper(), m_to.group(2).upper()
        if src != dst:
            return src, dst

    # 4. Check if amount is explicitly attached to one currency (e.g. '... 50 USD ... INR')
    amt_match = re.search(r"(\d+(?:\.\d+)?)\s*([A-Z]{3})\b", question, re.I)
    codes = _ISO.findall(question.upper())
    codes = [c for c in codes if c in {"USD", "INR", "EUR", "GBP", "JPY", "AUD", "CAD"}]
    if amt_match and len(codes) >= 2:
        src_cand = amt_match.group(2).upper()
        if src_cand in codes:
            dst_cand = next((c for c in codes if c != src_cand), None)
            if dst_cand:
                return src_cand, dst_cand

    if len(codes) >= 2:
        distinct = [c for c in codes if c != codes[0]]
        if distinct:
            return codes[0], distinct[0]

    if len(codes) >= 1:
        other = "USD" if codes[0] == "INR" else "INR"
        if re.search(r"\b(?:to|into|in)\s+" + re.escape(codes[0]), question, re.I):
            return other, codes[0]
        return codes[0], other
    if "inr" in question.lower():
        return "USD", "INR"
    return "USD", "INR"


def _extract_amount(question: str) -> float | int | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(USD|INR|EUR|GBP|dollars)?", question, re.I)
    if match:
        val = float(match.group(1))
        return int(val) if int(val) == val else val
    return None


def _extract_expression(question: str) -> str | None:
    normalized = question.replace("×", "*").replace("÷", "/")
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)", r"\1 * \2", normalized)
    match = re.search(
        r"(\d+(?:\.\d+)?\s*(?:[\+\-\*/\^%]|(?:\*\*))\s*\d+(?:\.\d+)?(?:\s*(?:[\+\-\*/\^%]|(?:\*\*))\s*\d+(?:\.\d+)?)*)",
        normalized,
    )
    if match:
        expr = match.group(1).strip()
        if re.search(r"[\+\-\*/\^%]", expr):
            return expr
    return None


def _reason(use_docs: bool, tools: list[ToolRequest]) -> str:
    names = [t.tool_name for t in tools]
    if use_docs and names:
        return f"Mixed intent: retrieve documents and call {', '.join(names)}."
    if names:
        if "knowledge" in names:
            return "General knowledge question: route to knowledge provider."
        return f"Live question: call {', '.join(names)}."
    if use_docs:
        return "Policy/document question: retrieve local sources only."
    return "No reliable source matched; abstain unless retrieval finds evidence."
