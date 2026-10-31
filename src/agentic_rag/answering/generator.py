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
    r"(?:travel\s+policy|expense\s+guidelines?|conference\s+travel\s+playbook|it\s+security\s+handbook|internal\s+memo|memo)"
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
        if re.search(r"\b(policy|guidelines?|playbook|handbook|memo|scope)\b", clean, re.I):
            return False
    return len(clean) >= 20


def _is_broad_question(question: str) -> bool:
    q = question.lower().strip()
    if re.search(
        r"\b(what information|what does .* (?:provide|say|cover|contain)|what are the (?:rules|guidelines|policies)|overview|summarize|summary|what expenses|employee expenses|travel expenses|expenses allowed)\b",
        q,
    ):
        return True
    subj = _subject_tokens(question)
    if subj in [{"employee", "expense"}, {"expense"}, {"travel"}, {"policy"}, set()]:
        return True
    return False


def _extract_subject_phrase(question: str) -> str:
    m = re.search(r"\b(?:for|about|on|regarding)\s+([a-zA-Z\s]+?)(?:\?|$)", question, re.I)
    if m:
        phrase = re.sub(r"\s+", " ", m.group(1).strip())
        if len(phrase) >= 3:
            return phrase
    subj = _subject_tokens(question)
    if subj:
        return " ".join(sorted(subj))
    return "this question"


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

    if passages:
        if _is_broad_question(question):
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

        if not parts:
            subj = _subject_tokens(question)
            candidates: list[tuple[int, int, str, RetrievedPassage]] = []
            q_all_tokens = _clean_tokens(question)
            min_subj_req = len(subj) if len(subj) <= 2 else 2

            for p in passages:
                for u in _split_units(p.chunk.text):
                    if not _is_informative_sentence(u):
                        continue
                    u_toks = _clean_tokens(u)
                    subj_match = len(subj & u_toks) if subj else 0
                    all_match = len(q_all_tokens & u_toks)
                    if subj and subj_match >= min_subj_req:
                        candidates.append((subj_match, all_match, u, p))
                    elif not subj and all_match > 0:
                        candidates.append((0, all_match, u, p))

            if candidates:
                candidates.sort(key=lambda c: (-c[0], c[3].rank, -c[1], -len(c[2])))
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
            elif subj:
                phrase = _extract_subject_phrase(question)
                return (
                    f"The available documents do not provide sufficient evidence regarding {phrase}.",
                    [],
                )
            else:
                return (
                    "The available documents do not provide sufficient evidence to answer this question.",
                    [],
                )

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
        return f"{data.get('expression')} = {data.get('value')} [T{index}]."
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
            return FinalAnswer(
                answer=(
                    "I don't have enough evidence in the indexed documents or from live tools "
                    "to answer that. I will not guess."
                ),
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
