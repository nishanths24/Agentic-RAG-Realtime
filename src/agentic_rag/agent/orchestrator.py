from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from agentic_rag.agent.router import RuleRouter, maybe_followup_currency
from agentic_rag.answering.generator import AnswerGenerator
from agentic_rag.config import Settings
from agentic_rag.errors import (
    RetrievalError,
    ToolError,
    ToolTimeoutError,
    ToolUnavailableError,
    ToolValidationError,
)
from agentic_rag.models import FinalAnswer, RetrievedPassage, ToolRequest, ToolResult, TraceStep
from agentic_rag.retrieval.retriever import Retriever
from agentic_rag.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        tools: ToolRegistry,
        router: RuleRouter | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.tools = tools
        self.router = router or RuleRouter()
        self.generator = generator or AnswerGenerator()

    def ask(self, question: str) -> FinalAnswer:
        trace_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        extra = {"trace_id": trace_id}
        logger.info("ask.start question_length=%s", len(question), extra=extra)
        traces: list[TraceStep] = []
        caveats: list[str] = []
        deadline = started + max(self.settings.request_timeout_s * 3, 20)

        t0 = time.perf_counter()
        decision = self.router.route(question)
        traces.append(
            TraceStep(
                "route",
                decision.reasoning,
                (time.perf_counter() - t0) * 1000,
            )
        )

        passages: list[RetrievedPassage] = []
        if decision.use_documents:
            t0 = time.perf_counter()
            try:
                passages = self.retriever.search(
                    question,
                    top_k=self.settings.top_k,
                    min_score=self.settings.min_score,
                    max_chars=self.settings.max_question_chars,
                )
                traces.append(
                    TraceStep(
                        "retrieve_docs",
                        f"{len(passages)} passages, top score "
                        f"{passages[0].score:.3f}" if passages else "0 passages",
                        (time.perf_counter() - t0) * 1000,
                    )
                )
            except RetrievalError as exc:
                caveats.append(str(exc))
                traces.append(TraceStep("retrieve_docs", str(exc), (time.perf_counter() - t0) * 1000, ok=False))

        tool_results: list[ToolResult] = []
        seen_calls: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        pending = list(decision.tools)
        steps = 0
        max_steps = self.settings.max_agent_steps

        while pending and steps < max_steps:
            if time.perf_counter() > deadline:
                caveats.append("Stopped because the request deadline was reached.")
                break
            request = pending.pop(0)
            key = (request.tool_name, tuple(sorted((k, str(v)) for k, v in request.arguments.items())))
            if key in seen_calls:
                traces.append(TraceStep("skip_duplicate", f"{request.tool_name} already called", 0, ok=True))
                continue
            seen_calls.add(key)
            steps += 1
            t0 = time.perf_counter()
            result = self._run_tool(request)
            tool_results.append(result)
            traces.append(
                TraceStep(
                    f"tool:{request.tool_name}",
                    "ok" if result.ok else (result.error or "failed"),
                    (time.perf_counter() - t0) * 1000,
                    ok=result.ok,
                )
            )

        passage_blob = " ".join(p.chunk.text for p in passages)
        follow = maybe_followup_currency(question, passage_blob)
        if follow and steps < max_steps:
            key = (follow.tool_name, tuple(sorted((k, str(v)) for k, v in follow.arguments.items())))
            if key not in seen_calls:
                t0 = time.perf_counter()
                result = self._run_tool(follow)
                tool_results.append(result)
                traces.append(
                    TraceStep(
                        "followup_currency",
                        f"{follow.arguments}",
                        (time.perf_counter() - t0) * 1000,
                        ok=result.ok,
                    )
                )

        t0 = time.perf_counter()
        answer = self.generator.generate(
            question,
            passages,
            tool_results,
            trace_id=trace_id,
            caveats=caveats,
        )
        traces.append(TraceStep("compose", "grounded answer assembled", (time.perf_counter() - t0) * 1000))
        logger.info(
            "ask.done sufficient=%s tools=%s passages=%s duration_ms=%.0f",
            answer.sufficient_evidence,
            len(tool_results),
            len(passages),
            (time.perf_counter() - started) * 1000,
            extra=extra,
        )
        return FinalAnswer(
            answer=answer.answer,
            document_evidence=answer.document_evidence,
            live_evidence=answer.live_evidence,
            sufficient_evidence=answer.sufficient_evidence,
            caveats=answer.caveats,
            trace_id=trace_id,
            routing=decision,
            trace=traces,
            retrieved_passages=passages,
            tool_results=tool_results,
        )

    def _run_tool(self, request: ToolRequest) -> ToolResult:
        try:
            return self.tools.run(
                request.tool_name,
                request.arguments,
                timeout_s=self.settings.request_timeout_s,
            )
        except ToolValidationError as exc:
            return ToolResult(
                tool_name=request.tool_name,
                ok=False,
                data=None,
                error=str(exc),
                retrieved_at=datetime.now(timezone.utc),
                source="validation",
            )
        except (ToolTimeoutError, ToolUnavailableError, ToolError) as exc:
            logger.warning("tool.fail name=%s error=%s", request.tool_name, exc)
            return ToolResult(
                tool_name=request.tool_name,
                ok=False,
                data=None,
                error=str(exc),
                retrieved_at=datetime.now(timezone.utc),
                source="tool-error",
            )
