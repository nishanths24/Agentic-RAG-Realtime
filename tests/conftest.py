from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.config import load_settings
from agentic_rag.embeddings.hashing import FakeEmbeddingProvider
from agentic_rag.vectorstore.numpy_store import NumpyVectorStore


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "travel_policy.txt").write_text(
        "The standard hotel reimbursement limit is 5000 INR per night.\n"
        "The daily meal allowance is 50 USD per day for international travel.\n"
        "Employees may rebook flights when a weather delay is documented.\n",
        encoding="utf-8",
    )
    (docs / "other.txt").write_text("Laptops must be encrypted. VPN credentials stay private.\n", encoding="utf-8")
    monkeypatch.setenv("DOCUMENTS_DIR", str(docs))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("CHUNK_SIZE", "120")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "64")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-hash-v1")
    return load_settings()


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingProvider(dimension=64)


@pytest.fixture
def store(tmp_path: Path, fake_embedder):
    return NumpyVectorStore(
        tmp_path / "index",
        model_name=fake_embedder.model_name,
        dimension=fake_embedder.dimension,
        chunk_size=120,
        chunk_overlap=20,
    )
