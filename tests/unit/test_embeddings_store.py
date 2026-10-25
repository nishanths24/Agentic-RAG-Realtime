from __future__ import annotations

import pytest

from agentic_rag.embeddings.hashing import FakeEmbeddingProvider
from agentic_rag.errors import EmbeddingError, IndexMismatchError
from agentic_rag.ingestion.chunking import chunks_from_document
from agentic_rag.vectorstore.numpy_store import NumpyVectorStore


def test_fake_embedder_determinism_and_similarity():
    embedder = FakeEmbeddingProvider(dimension=64)
    a = embedder.embed_query("hotel reimbursement limit")
    b = embedder.embed_query("hotel reimbursement limit")
    c = embedder.embed_query("encrypted laptops and vpn")
    assert a == b
    sim_ab = sum(x * y for x, y in zip(a, a))
    sim_ac = sum(x * y for x, y in zip(a, c))
    assert sim_ab > sim_ac


def test_empty_embed_rejected():
    embedder = FakeEmbeddingProvider(dimension=32)
    with pytest.raises(EmbeddingError):
        embedder.embed_documents([])
    with pytest.raises(EmbeddingError):
        embedder.embed_query("  ")


def test_store_persistence_and_idempotent_upsert(tmp_path, fake_embedder):
    store = NumpyVectorStore(tmp_path / "idx", model_name=fake_embedder.model_name, dimension=64, chunk_size=80, chunk_overlap=10)
    chunks = chunks_from_document(doc_id="d1", source_path="a.txt", text="Hotel limit is 5000 INR per night for approved trips.", chunk_size=80, overlap=10)
    vectors = fake_embedder.embed_documents([c.text for c in chunks])
    store.upsert(chunks, vectors)
    store.set_document_hash("a.txt", "hash1")
    store.persist()
    store.upsert(chunks, vectors)
    assert store.count() == len(chunks)
    store2 = NumpyVectorStore(tmp_path / "idx", model_name=fake_embedder.model_name, dimension=64, chunk_size=80, chunk_overlap=10)
    store2.load()
    assert store2.count() == store.count()
    hits = store2.search(fake_embedder.embed_query("hotel limit"), top_k=3)
    assert hits and hits[0].chunk.source_path == "a.txt"


def test_delete_and_mismatch(tmp_path, fake_embedder):
    store = NumpyVectorStore(tmp_path / "idx", model_name=fake_embedder.model_name, dimension=64, chunk_size=80, chunk_overlap=10)
    chunks = chunks_from_document(doc_id="d1", source_path="a.txt", text="alpha beta gamma", chunk_size=80, overlap=10)
    store.upsert(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    assert store.delete_by_source("a.txt") >= 1
    assert store.count() == 0
    store.persist()
    other = NumpyVectorStore(tmp_path / "idx", model_name="other", dimension=64, chunk_size=80, chunk_overlap=10)
    with pytest.raises(IndexMismatchError):
        other.load()
    assert store.search([0.0] * 64, top_k=3) == []
