from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    source_path: str
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class ToolRequest:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_name: str
    ok: bool
    data: dict[str, Any] | None
    error: str | None
    retrieved_at: datetime
    source: str


@dataclass(frozen=True, slots=True)
class Citation:
    kind: Literal["document", "tool"]
    reference: str
    label: str
    preview: str = ""


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    use_documents: bool
    tools: tuple[ToolRequest, ...]
    reasoning: str
    confidence: float
    strategy: str = "rules"


@dataclass(frozen=True, slots=True)
class TraceStep:
    name: str
    detail: str
    duration_ms: float
    ok: bool = True


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    answer: str
    document_evidence: list[Citation]
    live_evidence: list[Citation]
    sufficient_evidence: bool
    caveats: list[str]
    trace_id: str
    routing: RoutingDecision | None = None
    trace: list[TraceStep] = field(default_factory=list)
    retrieved_passages: list[RetrievedPassage] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SkipRecord:
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class IngestionReport:
    processed: tuple[str, ...]
    skipped: tuple[SkipRecord, ...]
    failed: tuple[SkipRecord, ...]
    chunk_count: int
