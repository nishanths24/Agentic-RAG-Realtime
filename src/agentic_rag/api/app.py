from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentic_rag.config import Settings
from agentic_rag.errors import RetrievalError
from agentic_rag.runtime import get_settings, get_stack as load_stack
from agentic_rag.serialize import answer_payload

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    trace: bool = True


def get_stack():
    settings = get_settings()
    orchestrator, indexer, store = load_stack()
    return settings, orchestrator, indexer, store


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic RAG Realtime", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            raise HTTPException(500, "UI is missing")
        return FileResponse(index_file)

    @app.get("/health")
    def health(stack=Depends(get_stack)) -> dict[str, Any]:
        settings, _, _, store = stack
        return {
            "status": "ok",
            "chunks": store.count(),
            "embedding_model": settings.embedding_model,
            "llm_provider": settings.llm_provider,
        }

    @app.get("/sources")
    def sources(stack=Depends(get_stack)) -> dict[str, Any]:
        _, _, _, store = stack
        hashes = store.document_hashes()
        return {
            "documents": [{"path": path, "content_hash": content[:12]} for path, content in sorted(hashes.items())],
            "chunks": store.count(),
        }

    @app.post("/ask")
    def ask(body: AskRequest, stack=Depends(get_stack)) -> dict[str, Any]:
        settings, orchestrator, _, _ = stack
        if len(body.question) > settings.max_question_chars:
            raise HTTPException(422, "Question is too long")
        try:
            answer = orchestrator.ask(body.question)
        except RetrievalError as exc:
            raise HTTPException(400, str(exc)) from exc
        payload = answer_payload(answer)
        if not body.trace:
            payload.pop("trace", None)
        return payload

    @app.post("/index")
    def index_docs(
        rebuild: bool = False,
        x_api_key: str | None = Header(default=None),
        stack=Depends(get_stack),
    ) -> dict[str, Any]:
        settings, _, indexer, _ = stack
        _protect_writes(settings, x_api_key)
        return indexer.build(rebuild=rebuild)

    return app


def _protect_writes(settings: Settings, header_key: str | None) -> None:
    if settings.api_key and header_key != settings.api_key:
        raise HTTPException(401, "Invalid API key")


app = create_app()
