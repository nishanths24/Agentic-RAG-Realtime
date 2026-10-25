from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.errors import EmptyDocumentError, PathEscapeError, UnsupportedDocumentError
from agentic_rag.ingestion.loaders import ingest_directory, load_document


def test_ingest_directory_skips_unsupported_and_empty(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("Hotel limit is 5000 INR per night.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "note.md").write_text("Markdown policy note about receipts.", encoding="utf-8")
    chunks, documents, report = ingest_directory(tmp_path, chunk_size=80, overlap=10)
    assert len(documents) == 2
    assert report.chunk_count == len(chunks)
    assert any("empty" in s.path for s in report.skipped)
    assert any("skip.bin" in s.path for s in report.skipped)
    assert all("/" in d.source_path or d.source_path.endswith(".txt") or d.source_path.endswith(".md") for d in documents)
    assert documents[0].source_path <= documents[1].source_path


def test_load_rejects_escape(tmp_path: Path):
    root = tmp_path / "docs"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(PathEscapeError):
        load_document(outside, root)


def test_non_utf8_replaced(tmp_path: Path):
    path = tmp_path / "latin.txt"
    path.write_bytes("café".encode("latin-1"))
    doc = load_document(path, tmp_path)
    assert doc.text


def test_empty_raises(tmp_path: Path):
    path = tmp_path / "blank.txt"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(path, tmp_path)


def test_pdf_unsupported(tmp_path: Path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF")
    with pytest.raises(UnsupportedDocumentError):
        load_document(path, tmp_path)


def test_chunk_metadata_slice(tmp_path: Path):
    (tmp_path / "a.txt").write_text("Alpha beta gamma delta epsilon zeta eta theta.", encoding="utf-8")
    chunks, documents, _ = ingest_directory(tmp_path, chunk_size=20, overlap=5)
    doc = documents[0]
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.source_path == "a.txt"
        assert chunk.chunk_id.startswith(chunk.doc_id)
