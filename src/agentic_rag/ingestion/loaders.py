from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from agentic_rag.errors import (
    DocumentNotFoundError,
    EmptyDocumentError,
    PathEscapeError,
    UnsupportedDocumentError,
)
from agentic_rag.ingestion.chunking import chunks_from_document
from agentic_rag.ingestion.cleaning import normalize_document_text
from agentic_rag.models import Chunk, Document, IngestionReport, SkipRecord

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".md")


def relative_source_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def stable_doc_id(source_path: str) -> str:
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_documents(
    root: Path,
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
) -> list[Path]:
    root = root.resolve()
    if not root.exists():
        raise DocumentNotFoundError(f"Documents root does not exist: {root}")
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda p: relative_source_path(p, root))


def _assert_inside_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PathEscapeError(f"Path escapes documents root: {path}") from exc
    return resolved


def read_text_file(file_path: Path) -> str:
    """Read a UTF-8 text file (legacy helper used by run_ingestion.py)."""
    return Path(file_path).read_text(encoding="utf-8")


def load_document(
    path: Path,
    root: Path,
    *,
    max_file_bytes: int = 1_048_576,
) -> Document:
    resolved = _assert_inside_root(Path(path), root)
    if not resolved.exists():
        raise DocumentNotFoundError(f"Document not found: {path}")
    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(f"Unsupported type: {suffix}")
    size = resolved.stat().st_size
    if size > max_file_bytes:
        raise UnsupportedDocumentError(
            f"File exceeds max size ({size} > {max_file_bytes} bytes): {path}"
        )
    raw = resolved.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    if "\ufffd" in text:
        logger.warning("Replaced invalid UTF-8 bytes in %s", resolved)
    normalized = normalize_document_text(text)
    if not normalized:
        raise EmptyDocumentError(f"Empty document: {path}")
    source = relative_source_path(resolved, root)
    return Document(
        doc_id=stable_doc_id(source),
        source_path=source,
        text=normalized,
        content_hash=content_hash(normalized),
    )


def ingest_directory(
    root: Path,
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
    max_file_bytes: int = 1_048_576,
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
) -> tuple[list[Chunk], list[Document], IngestionReport]:
    processed: list[str] = []
    skipped: list[SkipRecord] = []
    failed: list[SkipRecord] = []
    chunks: list[Chunk] = []
    documents: list[Document] = []

    if not root.exists():
        raise DocumentNotFoundError(f"Documents root does not exist: {root}")

    all_files = sorted(
        (p for p in root.resolve().rglob("*") if p.is_file()),
        key=lambda p: relative_source_path(p, root),
    )
    for path in all_files:
        rel = relative_source_path(path, root)
        if path.suffix.lower() not in extensions:
            logger.warning("Skipping unsupported file %s", rel)
            skipped.append(SkipRecord(rel, f"unsupported type {path.suffix or '(none)'}"))
            continue
        try:
            document = load_document(path, root, max_file_bytes=max_file_bytes)
        except EmptyDocumentError as exc:
            skipped.append(SkipRecord(rel, str(exc)))
            continue
        except (UnsupportedDocumentError, PathEscapeError, OSError) as exc:
            failed.append(SkipRecord(rel, str(exc)))
            continue
        doc_chunks = chunks_from_document(
            doc_id=document.doc_id,
            source_path=document.source_path,
            text=document.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        documents.append(document)
        chunks.extend(doc_chunks)
        processed.append(rel)

    report = IngestionReport(
        processed=tuple(processed),
        skipped=tuple(skipped),
        failed=tuple(failed),
        chunk_count=len(chunks),
    )
    return chunks, documents, report
