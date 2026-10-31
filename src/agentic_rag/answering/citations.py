from __future__ import annotations

import re

from agentic_rag.models import Citation, RetrievedPassage, ToolResult

_LABEL = re.compile(r"\[(D|T)(\d+)\]")


def document_citations(passages: list[RetrievedPassage]) -> list[Citation]:
    citations: list[Citation] = []
    for passage in passages:
        citations.append(
            Citation(
                kind="document",
                reference=passage.chunk.source_path,
                label=f"D{passage.rank}",
                preview=passage.chunk.text[:240],
            )
        )
    return citations


def live_citations(results: list[ToolResult]) -> list[Citation]:
    citations: list[Citation] = []
    index = 1
    for result in results:
        if not result.ok:
            continue
        citations.append(
            Citation(
                kind="tool",
                reference=f"{result.tool_name}:{result.source}",
                label=f"T{index}",
                preview=str(result.data)[:240],
            )
        )
        index += 1
    return citations


def strip_unknown_labels(answer: str, allowed: set[str]) -> tuple[str, list[str]]:
    unknown: list[str] = []

    def repl(match: re.Match[str]) -> str:
        label = match.group(0)
        if label not in allowed:
            unknown.append(label)
            return ""
        return label

    cleaned = _LABEL.sub(repl, answer)
    cleaned = re.sub(r" +", " ", cleaned).strip()
    return cleaned, unknown
