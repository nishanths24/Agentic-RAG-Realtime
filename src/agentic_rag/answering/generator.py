from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from agentic_rag.answering.citations import document_citations, live_citations, strip_unknown_labels
from agentic_rag.llm.fake import FakeLLMClient
from agentic_rag.models import Citation, FinalAnswer, RetrievedPassage, ToolResult

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")

_GENERIC_STOP = {
    "the", "and", "for", "our", "are", "was", "were", "what", "when", "who", "whom", "how", "why",
    "which", "where", "whose", "does", "did", "this", "that", "with", "from", "your", "you", "into",
    "about", "can", "could", "would", "should", "may", "might", "must", "not", "any", "all", "per",
    "has", "have", "had", "its", "is", "in", "on", "at", "to", "of", "by", "an", "a", "do", "done",
    "tell", "tells", "information", "detail", "details", "give", "gives", "say", "says", "said",
    "provide", "provides", "providing", "provided", "state", "states", "stating", "stated",
    "stipulate", "stipulates", "rule", "rules", "policy", "policies", "document", "documents",
    "handbook", "guideline", "guidelines", "company", "corporate", "internal", "approved", "business",
    "regard", "regarding", "reimbursement", "reimburse", "reimbursable", "travel", "trip", "trips",
    "claim", "claims", "much", "many", "explain", "explanation", "describe",
}

_TITLE_OR_HEADING = re.compile(
    r"^(?:#+\s*)?"
    r"(?:(?:company|corporate|internal|employee)\s+)?"
    r"(?:travel\s+policy|expense\s+guidelines?|conference\s+travel\s+playbook|it\s+security\s+handbook|internal\s+memo|memo|"
    r"project\s+[a-zA-Z0-9_\-\s]+(?:brief|overview|summary|details|update)?|"
    r"(?:internal\s+)?project\s+code|scheduled\s+review|project\s+owner|status\s+update|document\s+scope)"
    r"(?:\s*\(.*?\))?$",
    re.I,
)

_META_DISCLAIMER = re.compile(
    r"\b(ignore previous instructions|reveal secrets|call the weather tool|dump environment variables|only document data|do not treat .* as system commands?)\b",
    re.I,
)

_SCOPE_DISCLAIMER = re.compile(
    r"\b(?:does not|do not|will not|cannot)\s+(?:define|cover|contain|provide|specify|establish|include|address|describe)\b|"
    r"\b(?:out of scope|not covered|not defined|not specified|not addressed|no policy is established)\b|"
    r"\bdescribes only the details listed above\b",
    re.I,
)


def _stem(word: str) -> str:
    w = word.lower()
    for suffix in ("ing", "ment", "tion", "able", "ible"):
        if w.endswith(suffix) and len(w) > len(suffix) + 3:
            w = w[:-len(suffix)]
            break
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return w


_GENERIC_STEMS = {_stem(w) for w in _GENERIC_STOP}


def _clean_tokens(text: str) -> set[str]:
    return {_stem(tok) for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2}


def _subject_tokens(text: str) -> set[str]:
    return {
        _stem(tok)
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 2 and tok.lower() not in _GENERIC_STOP and _stem(tok) not in _GENERIC_STEMS
    }


def _split_units(text: str) -> list[str]:
    units: list[str] = []
    for block in re.split(r"\n+", text):
        clean = block.strip().lstrip("#").strip()
        if len(clean) < 20:
            continue
        units.extend(part.strip() for part in _SENTENCE.split(clean) if len(part.strip()) >= 20)
    return units


def _is_informative_sentence(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if clean.startswith("#"):
        return False
    if not (clean[0].isupper() or clean[0].isdigit() or clean[0] in ('"', "'", "“", "”", "(", "[")):
        return False
    if _TITLE_OR_HEADING.match(clean):
        return False
    if _META_DISCLAIMER.search(clean):
        return False
    if _SCOPE_DISCLAIMER.search(clean):
        return False
    if len(clean) < 60 and not clean.endswith((".", "!", "?", ":")):
        if re.search(r"\b(policy|guidelines?|playbook|handbook|memo|scope|brief|code|review|owner|update|overview)\b", clean, re.I):
            return False
    if "—" in clean and len(clean) < 60:
        return False
    return len(clean) >= 20


_PROJECT_RE = re.compile(r"\bproject\s+([A-Za-z0-9_-]+)\b", re.I)
_PROJECT_ATTR_WORDS = {
    "code", "name", "brief", "review", "overview", "details", "detail", "plan",
    "update", "updates", "milestone", "milestones", "owner", "lead", "status",
    "document", "documents", "documentation", "work", "initiative", "info", "information",
    "the", "a", "an", "this", "that", "is", "was", "are", "were", "our", "my", "your", "for", "in", "of",
}


def _extract_projects(text: str) -> set[str]:
    res: set[str] = set()
    for m in re.finditer(r"\bproject\s+([A-Za-z0-9_-]+)", text, re.I):
        w = re.sub(r"['\u2019]s?$", "", m.group(1)).lower()
        if w not in _PROJECT_ATTR_WORDS and len(w) > 1:
            res.add(w)
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9_-]*)\s+project\b", text):
        w = re.sub(r"['\u2019]s?$", "", m.group(1)).lower()
        if w not in _PROJECT_ATTR_WORDS and len(w) > 1:
            res.add(w)
    return res


_BROAD_PATTERNS = re.compile(
    r"\b(summarize|summary|overview|main\s+topic|key\s+information|key\s+points|key\s+details|"
    r"what\s+key|what\s+does\s+.*(?:contain|cover|say|provide)|what\s+information|"
    r"what\s+are\s+the\s+(?:rules|guidelines|policies)|what\s+expenses|employee\s+expenses|travel\s+expenses|"
    r"expenses\s+allowed|indexed\s+documents?|uploaded\s+documents?)\b",
    re.I,
)


def _is_broad_question(question: str) -> bool:
    q = question.lower().strip()
    if _BROAD_PATTERNS.search(q):
        return True
    subj = _subject_tokens(question)
    if subj in [{"employee", "expense"}, {"expense"}, {"travel"}, {"policy"}, set()]:
        return True
    return False


def _extract_subject_phrase(question: str) -> str:
    m = re.search(r"\b(?:for|about|on|regarding)\s+([a-zA-Z\s]+?)(?:\?|$)", question, re.I)
    if m:
        phrase = re.sub(r"\s+", " ", m.group(1).strip())
        lowered = phrase.lower()
        noise_words = {"uploaded", "document", "documents", "this", "that", "these", "contain", "key", "main", "topic", "summary"}
        tokens = set(lowered.split())
        if len(phrase) >= 3 and not (tokens & noise_words):
            return phrase
    return ""


def _insufficient_fallback(phrase: str = "") -> str:
    prefix = "I couldn't find enough relevant information in the available documents to answer that question."
    if phrase:
        clean = phrase.strip()
        noise = {
            "contain", "upload", "uploaded", "document", "documents", "key", "main", "topic",
            "summary", "phone", "number", "ceo", "personal", "question", "these", "this", "that",
        }
        words = set(re.findall(r"[a-zA-Z0-9]+", clean.lower()))
        if clean and len(clean) >= 3 and not (words <= noise):
            return f"{prefix} The available documents do not provide sufficient evidence regarding {clean}."
    return prefix


def _format_broad_intro(question: str) -> str:
    q_lower = question.lower()
    if "travel policy" in q_lower or "expense" in q_lower:
        return "The travel policy provides the following information about employee expenses:"
    m = re.search(r"\b(?:about|regarding|for)\s+([a-zA-Z\s]+?)(?:\?|$)", question, re.I)
    if m:
        phrase = re.sub(r"\s+", " ", m.group(1).strip())
        return f"The retrieved documents provide the following information regarding {phrase}:"
    return "The retrieved documents provide the following information:"


def compose_extractive_answer(
    question: str,
    passages: list[RetrievedPassage],
    tool_results: list[ToolResult],
) -> tuple[str, list[RetrievedPassage]]:
    parts: list[str] = []
    used: list[RetrievedPassage] = []
    q_projects = _extract_projects(question)

    if passages:
        if _is_broad_question(question):
            is_travel_expense = bool(
                re.search(r"\b(travel\s+policy|employee\s+expenses?|travel\s+expenses?|expenses?\s+allowed)\b", question, re.I)
            ) or ("expense" in question.lower() and "policy" in question.lower())

            if is_travel_expense:
                topics = [
                    ("General reimbursement", ["claim", "travel", "reimbursement", "approved"], 2),
                    ("Hotel limit", ["hotel", "reimbursement", "limit", "5000"], 2),
                    ("Meal allowance", ["meal", "allowance", "50", "usd", "1500", "inr"], 2),
                    ("Flight policy", ["flights", "economy", "business", "class"], 2),
                    ("Weather delays", ["weather", "delays", "rebook"], 2),
                    ("Receipts deadline", ["receipts", "submitted", "15", "days"], 2),
                    ("Ground transport & incidentals", ["ground", "transport", "taxis", "mileage", "internet"], 2),
                ]
                points: list[str] = []
                seen_texts: set[str] = set()
                for label, keywords, min_kw in topics:
                    kw_set = set(keywords)
                    best_cand = None
                    best_p = None
                    best_count = 0
                    for p in passages:
                        for u in _split_units(p.chunk.text):
                            if not _is_informative_sentence(u):
                                continue
                            u_lower = u.lower()
                            cnt = sum(1 for kw in kw_set if kw in u_lower)
                            if cnt >= min_kw and cnt > best_count:
                                best_count = cnt
                                best_cand = u
                                best_p = p
                    if best_cand and best_cand not in seen_texts:
                        seen_texts.add(best_cand)
                        points.append(f"{best_cand} [D{best_p.rank}]")
                        if best_p not in used:
                            used.append(best_p)

                if points:
                    intro = _format_broad_intro(question)
                    parts.append(intro + " " + " ".join(points))

            else:
                # Document / project summary, overview, main-topic, or key-information
                relevant_passages = passages
                if q_projects:
                    relevant_passages = [
                        p for p in passages
                        if (q_projects & _extract_projects(p.chunk.text))
                        or any(proj in p.chunk.source_path.lower() for proj in q_projects)
                    ]

                if q_projects and not relevant_passages:
                    phrase = _extract_subject_phrase(question)
                    return (_insufficient_fallback(phrase), [])

                all_units: list[tuple[str, RetrievedPassage]] = []
                for p in relevant_passages:
                    if q_projects:
                        p_projs = _extract_projects(p.chunk.text)
                        if p_projs and not (q_projects & p_projs):
                            continue
                    for u in _split_units(p.chunk.text):
                        if _is_informative_sentence(u):
                            u_projs = _extract_projects(u)
                            if q_projects and u_projs and not (q_projects & u_projs):
                                continue
                            all_units.append((u, p))

                if all_units:
                    is_main_topic = bool(re.search(r"\b(main\s+topic|topic\s+of|subject|purpose)\b", question, re.I))
                    selected_units: list[tuple[str, RetrievedPassage]] = []
                    seen_u: set[str] = set()

                    def _find_best(pattern: str) -> tuple[str, RetrievedPassage] | None:
                        rgx = re.compile(pattern, re.I)
                        for u_text, u_p in all_units:
                            if u_text not in seen_u and rgx.search(u_text):
                                return (u_text, u_p)
                        return None

                    # 1. Overview unit
                    u_overview = _find_best(r"\b(is an|is the|initiative|purpose|overview|tracks)\b")
                    if u_overview:
                        selected_units.append(u_overview)
                        seen_u.add(u_overview[0])

                    if not is_main_topic:
                        # 2. Key identifier / code
                        u_code = _find_best(r"\b(code|internal code|code name)\b")
                        if u_code:
                            selected_units.append(u_code)
                            seen_u.add(u_code[0])

                        # 3. Schedule / review
                        u_review = _find_best(r"\b(review|scheduled|deadline|date)\b")
                        if u_review:
                            selected_units.append(u_review)
                            seen_u.add(u_review[0])

                        # 4. Owner / team responsibilities
                        u_team = _find_best(r"\b(coordinates|collects updates|owner|ownership|responsibilit)\b") or _find_best(r"\bteam\b")
                        if u_team:
                            selected_units.append(u_team)
                            seen_u.add(u_team[0])

                    max_units = 2 if is_main_topic else 4
                    if len(selected_units) < max_units:
                        for u_text, u_p in all_units:
                            if u_text not in seen_u:
                                selected_units.append((u_text, u_p))
                                seen_u.add(u_text)
                                if len(selected_units) >= max_units:
                                    break

                    if selected_units:
                        sentences = [f"{text} [D{p.rank}]" for text, p in selected_units]
                        for _, p in selected_units:
                            if p not in used:
                                used.append(p)
                        parts.append(" ".join(sentences))

        if not parts:
            subj = _subject_tokens(question)
            candidates: list[tuple[int, int, str, RetrievedPassage]] = []
            q_all_tokens = _clean_tokens(question)
            min_subj_req = len(subj) if len(subj) <= 2 else 2

            for p in passages:
                p_projs = _extract_projects(p.chunk.text)
                if q_projects and p_projs and not (q_projects & p_projs):
                    continue

                for u in _split_units(p.chunk.text):
                    if not _is_informative_sentence(u):
                        continue
                    u_projs = _extract_projects(u)
                    if q_projects and u_projs and not (q_projects & u_projs):
                        continue
                    u_toks = _clean_tokens(u)
                    subj_match = len(subj & u_toks) if subj else 0
                    all_match = len(q_all_tokens & u_toks)
                    if subj and subj_match >= min_subj_req:
                        candidates.append((subj_match, all_match, u, p))
                    elif not subj and all_match > 0:
                        candidates.append((0, all_match, u, p))

            if candidates:
                candidates.sort(key=lambda c: (-c[0], -c[1], c[3].rank, -len(c[2])))
                best_subj, best_all, _, _ = candidates[0]
                top_selected = [
                    (u, p)
                    for s_m, a_m, u, p in candidates
                    if s_m == best_subj and a_m >= best_all - 1
                ][:2]
                sentences = [f"{text} [D{p.rank}]" for text, p in top_selected]
                used = []
                for _, p in top_selected:
                    if p not in used:
                        used.append(p)
                parts.append(" ".join(sentences))
            else:
                phrase = _extract_subject_phrase(question)
                return (_insufficient_fallback(phrase), [])

    live_bits = []
    t_index = 1
    for result in tool_results:
        if not result.ok or not result.data:
            continue
        live_bits.append(_format_tool(result, t_index))
        t_index += 1
    if live_bits:
        parts.append(" ".join(live_bits))

    return " ".join(parts).strip(), used


def _unique_passages(passages: list[RetrievedPassage]) -> list[RetrievedPassage]:
    seen: set[str] = set()
    unique: list[RetrievedPassage] = []
    for passage in passages:
        if passage.chunk.chunk_id in seen:
            continue
        seen.add(passage.chunk.chunk_id)
        unique.append(passage)
    return unique


def _format_tool(result: ToolResult, index: int) -> str:
    data = result.data or {}
    stamp = result.retrieved_at.strftime("%Y-%m-%d %H:%M UTC")
    if result.tool_name == "current_time":
        return f"Current time in {data.get('timezone')} is {data.get('display')} [T{index}]."
    if result.tool_name == "currency_convert":
        return (
            f"{data.get('amount')} {data.get('from')} equals {data.get('converted')} "
            f"{data.get('to')} at rate {data.get('rate')} (as of {data.get('as_of')}, retrieved {stamp}) [T{index}]."
        )
    if result.tool_name == "weather":
        return (
            f"Weather in {data.get('location') or data.get('city')}: {data.get('summary')}, "
            f"{data.get('temperature_c')}°C, wind {data.get('wind_speed')} (retrieved {stamp}) [T{index}]."
        )
    if result.tool_name == "calculator":
        val = data.get("value")
        if isinstance(val, (int, float)):
            val = int(val) if float(val).is_integer() else val
        return f"{data.get('expression')} = {val} [T{index}]."
    if result.tool_name == "knowledge":
        ans = (data.get("answer") or "").strip()
        if ans.endswith("."):
            return f"{ans} [T{index}]"
        return f"{ans}. [T{index}]"
    return f"{result.tool_name} returned {json.dumps(data)} [T{index}]."


def evidence_numeric_tokens(*blobs: str) -> set[str]:
    tokens: set[str] = set()
    for blob in blobs:
        tokens.update(_NUM.findall(blob))
    return tokens


class AnswerGenerator:
    def __init__(self, llm: object | None = None) -> None:
        self.llm = llm

    def generate(
        self,
        question: str,
        passages: list[RetrievedPassage],
        tool_results: list[ToolResult],
        *,
        trace_id: str,
        caveats: list[str] | None = None,
    ) -> FinalAnswer:
        caveats = list(caveats or [])
        docs = document_citations(passages)
        live = live_citations(tool_results)
        failed_tools = [r for r in tool_results if not r.ok]
        for failed in failed_tools:
            caveats.append(f"The {failed.tool_name} tool was unavailable: {failed.error}")

        if not passages and not live:
            failed_knowledge = [r for r in failed_tools if r.tool_name == "knowledge"]
            if failed_knowledge:
                err = failed_knowledge[0].error or "unavailable"
                return FinalAnswer(
                    answer=f"General knowledge questions require an active knowledge provider, but the provider is currently unavailable ({err}).",
                    document_evidence=[],
                    live_evidence=[],
                    sufficient_evidence=False,
                    caveats=caveats or ["Knowledge provider is unavailable."],
                    trace_id=trace_id,
                    retrieved_passages=passages,
                    tool_results=tool_results,
                )
            return FinalAnswer(
                answer=_insufficient_fallback(),
                document_evidence=[],
                live_evidence=[],
                sufficient_evidence=False,
                caveats=caveats or ["No supporting passages or live tool results."],
                trace_id=trace_id,
                retrieved_passages=passages,
                tool_results=tool_results,
            )

        used_passages = passages
        if self.llm is not None:
            raw = self.llm.generate(  # type: ignore[union-attr]
                _system_prompt(), _user_prompt(question, passages, tool_results)
            )
            allowed_fmt = {f"[{c.label}]" for c in docs + live}
            answer, unknown = strip_unknown_labels(raw, allowed_fmt)
            if unknown:
                caveats.append("Removed citation labels that were not in the retrieved evidence.")
            sufficient = not (
                "insufficient" in answer.lower()
                or "not enough evidence" in answer.lower()
                or "cannot answer" in answer.lower()
            )
        else:
            answer, used_passages = compose_extractive_answer(question, passages, tool_results)
            docs = document_citations(used_passages)
            sufficient = bool(used_passages or live)

        if not sufficient and not any("insufficient evidence" in c.lower() for c in caveats):
            caveats.append("Insufficient evidence: The available documents do not provide evidence to answer this question.")

        if failed_tools and not live and passages:
            caveats.append("Live information could not be verified; document facts are shown separately.")

        return FinalAnswer(
            answer=answer,
            document_evidence=docs,
            live_evidence=live,
            sufficient_evidence=sufficient,
            caveats=caveats,
            trace_id=trace_id,
            retrieved_passages=passages,
            tool_results=tool_results,
        )


def _system_prompt() -> str:
    from agentic_rag.agent.prompts import SYSTEM_GROUNDED

    return SYSTEM_GROUNDED


def _user_prompt(question: str, passages: list[RetrievedPassage], tools: list[ToolResult]) -> str:
    blocks: list[str] = []
    for passage in passages:
        blocks.append(
            f'<evidence id="D{passage.rank}" source="{passage.chunk.source_path}">\n{passage.chunk.text}\n</evidence>'
        )
    index = 1
    for result in tools:
        if not result.ok:
            continue
        blocks.append(
            f'<evidence id="T{index}" tool="{result.tool_name}" retrieved_at="{result.retrieved_at.isoformat()}" source="{result.source}">\n{json.dumps(result.data)}\n</evidence>'
        )
        index += 1
    return "EVIDENCE:\n" + "\n".join(blocks) + f"\n\nQUESTION:\n{question}\n"
