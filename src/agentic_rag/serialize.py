from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from agentic_rag.models import FinalAnswer


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def answer_payload(answer: FinalAnswer) -> dict[str, Any]:
    payload = to_jsonable(answer)
    payload["sections"] = {
        "answer": answer.answer,
        "from_your_documents": [
            {"label": c.label, "reference": c.reference, "preview": c.preview}
            for c in answer.document_evidence
        ],
        "live_information": [
            {"label": c.label, "reference": c.reference, "preview": c.preview}
            for c in answer.live_evidence
        ],
        "caveats": answer.caveats,
    }
    return payload
