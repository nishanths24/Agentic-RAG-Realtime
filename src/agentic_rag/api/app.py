from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentic_rag.config import Settings
from agentic_rag.errors import RetrievalError
from agentic_rag.ingestion.cleaning import normalize_document_text
from agentic_rag.ingestion.loaders import SUPPORTED_EXTENSIONS, _assert_inside_root
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
        settings, _, _, store = stack
        hashes = store.document_hashes()
        docs = []
        for path, content in sorted(hashes.items()):
            file_path = settings.documents_dir / path
            size = file_path.stat().st_size if file_path.exists() else None
            ext = Path(path).suffix.lower()
            docs.append({
                "path": path,
                "filename": Path(path).name,
                "content_hash": content[:12],
                "extension": ext,
                "size_bytes": size,
                "status": "indexed",
            })
        return {
            "documents": docs,
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

    @app.post("/documents/upload")
    @app.post("/documents")
    def upload_document(
        file: UploadFile = File(...),
        overwrite: bool = Form(default=False),
        x_api_key: str | None = Header(default=None),
        stack=Depends(get_stack),
    ) -> dict[str, Any]:
        settings, _, indexer, store = stack
        _protect_writes(settings, x_api_key)

        import re
        import urllib.parse

        raw_filename = file.filename or ""
        if not raw_filename.strip():
            raise HTTPException(400, "Filename cannot be empty")

        decoded = urllib.parse.unquote(raw_filename)
        if any(c in raw_filename or c in decoded for c in ["\0", "..", "/", "\\"]):
            raise HTTPException(
                400, "Invalid filename: path traversal characters are not permitted"
            )

        filename = Path(raw_filename).name.strip()
        if not filename or filename.startswith(".") or not re.match(r"^[a-zA-Z0-9_\-. ()]+$", filename):
            raise HTTPException(
                400, "Invalid filename: unsupported filename characters or hidden file"
            )

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(SUPPORTED_EXTENSIONS)
            raise HTTPException(
                400,
                f"Unsupported file type '{suffix}'. Supported formats are: {allowed}",
            )

        target_path = settings.documents_dir / filename
        try:
            _assert_inside_root(target_path, settings.documents_dir)
        except Exception as exc:
            raise HTTPException(400, "Invalid file destination") from exc

        if target_path.exists() and not overwrite:
            raise HTTPException(
                409,
                f"A document named '{filename}' already exists. Please rename the file or choose a different document.",
            )

        content = file.file.read(settings.max_file_bytes + 1)
        if not content:
            raise HTTPException(
                400, "File is empty. Uploaded documents must contain text content."
            )
        if len(content) > settings.max_file_bytes:
            max_kb = settings.max_file_bytes // 1024
            raise HTTPException(
                413,
                f"File exceeds maximum allowed size of {settings.max_file_bytes} bytes ({max_kb} KB).",
            )

        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(
                400, "File is not a valid UTF-8 text file. Only text-encoded files are supported."
            )

        normalized = normalize_document_text(text)
        if not normalized:
            raise HTTPException(
                400, "Document contains no readable text content after normalization."
            )

        settings.documents_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(content)

        try:
            index_report = indexer.build(rebuild=False)
        except Exception as exc:
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
            raise HTTPException(500, f"Indexing failed: {exc}") from exc

        return {
            "status": "ok",
            "message": f"Successfully uploaded and indexed '{filename}'.",
            "filename": filename,
            "size_bytes": len(content),
            "extension": suffix,
            "chunks_total": store.count(),
            "index_report": index_report,
        }

    @app.delete("/documents/{filename}")
    def delete_document(
        filename: str,
        x_api_key: str | None = Header(default=None),
        stack=Depends(get_stack),
    ) -> dict[str, Any]:
        settings, _, indexer, store = stack
        _protect_writes(settings, x_api_key)

        import re
        import urllib.parse

        raw_filename = filename or ""
        if not raw_filename.strip():
            raise HTTPException(400, "Filename cannot be empty")

        decoded = urllib.parse.unquote(raw_filename)
        if any(c in raw_filename or c in decoded for c in ["\0", "..", "/", "\\"]):
            raise HTTPException(
                400, "Invalid filename: path traversal characters are not permitted"
            )

        clean_filename = Path(raw_filename).name.strip()
        if not clean_filename or clean_filename.startswith(".") or not re.match(r"^[a-zA-Z0-9_\-. ()]+$", clean_filename):
            raise HTTPException(
                400, "Invalid filename: unsupported filename characters or hidden file"
            )

        suffix = Path(clean_filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(SUPPORTED_EXTENSIONS)
            raise HTTPException(
                400,
                f"Unsupported file type '{suffix}'. Supported formats are: {allowed}",
            )

        target_path = settings.documents_dir / clean_filename
        try:
            _assert_inside_root(target_path, settings.documents_dir)
        except Exception as exc:
            raise HTTPException(400, "Invalid file destination") from exc

        if not target_path.exists():
            raise HTTPException(404, f"Document '{clean_filename}' not found.")

        try:
            target_path.unlink()
        except OSError as exc:
            raise HTTPException(500, f"Failed to delete document file: {exc}") from exc

        try:
            removed_chunks = indexer.remove_document(clean_filename)
        except Exception as exc:
            raise HTTPException(500, f"Failed to update index after file deletion: {exc}") from exc

        return {
            "status": "ok",
            "message": f"Successfully deleted '{clean_filename}'.",
            "filename": clean_filename,
            "removed_chunks": removed_chunks,
            "chunks_removed": removed_chunks,
            "chunks_total": store.count(),
        }

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
