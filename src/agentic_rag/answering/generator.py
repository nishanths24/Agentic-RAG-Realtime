from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from agentic_rag.answering.citations import document_citations, live_citations, strip_unknown_labels
from agentic_rag.llm.fake import FakeLLMClient
from agentic_rag.models import Citation, FinalAnswer, RetrievedPassage, ToolResult

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def compose_extractive_answer(
    question: str,
    passages: list[RetrievedPassage],
    tool_results: list[ToolResult],
) -> str:
    parts: list[str] = []
    if passages:
        sentences = _pick_sentences(question, passages)
        if sentences:
            parts.append(" ".join(sentences))
        else:
            parts.append(passages[0].chunk.text.strip())
        parts[-1] = parts[-1] + " " + " ".join(f"[D{p.rank}]" for p in passages[: min(3, len(passages))])
    live_bits = []
    t_index = 1
    for result in tool_results:
        if not result.ok or not result.data:
            continue
        live_bits.append(_format_tool(result, t_index))
        t_index += 1
    if live_bits:
        parts.append(" ".join(live_bits))
    return " ".join(parts).strip()


def _pick_sentences(question: str, passages: list[RetrievedPassage]) -> list[str]:
    q_tokens = {tok for tok in re.findall(r"[a-z0-9]+", question.lower()) if len(tok) > 2}
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for passage in passages:
        for sentence in _SENTENCE.split(passage.chunk.text):
            clean = sentence.strip()
            if len(clean) < 20 or clean in seen:
                continue
            tokens = set(re.findall(r"[a-z0-9]+", clean.lower()))
            score = len(q_tokens & tokens)
            if score:
                scored.append((score, clean))
                seen.add(clean)
    scored.sort(key=lambda item: (-item[0], -len(item[1])))
    return [text for _, text in scored[:4]]


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

        if self.llm is not None:
            raw = self.llm.generate(  # type: ignore[union-attr]
                _system_prompt(), _user_prompt(question, passages, tool_results)
            )
            allowed = {c.label for c in docs + live}
            allowed_fmt = {f"[{label}]" if not label.startswith("[") else label for label in allowed}
            # labels stored as D1, allow [D1]
            allowed_fmt = {f"[{c.label}]" for c in docs + live}
            answer, unknown = strip_unknown_labels(raw, allowed_fmt)
            if unknown:
                caveats.append("Removed citation labels that were not in the retrieved evidence.")
        else:
            answer = compose_extractive_answer(question, passages, tool_results)

        sufficient = bool(passages or live)
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
