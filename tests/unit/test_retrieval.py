from __future__ import annotations

import pytest

from agentic_rag.errors import RetrievalError
from agentic_rag.indexing import Indexer
from agentic_rag.retrieval.retriever import Retriever, normalize_query


def test_normalize_query_rejects_blank():
    with pytest.raises(RetrievalError):
        normalize_query("   ")


def test_hotel_query_hits_policy(tmp_settings, fake_embedder, store):
    Indexer(tmp_settings, fake_embedder, store).build(rebuild=True)
    retriever = Retriever(store, fake_embedder)
    hits = retriever.search("hotel reimbursement limit", top_k=3, min_score=0.0)
    assert hits
    assert hits[0].chunk.source_path == "travel_policy.txt"
    assert hits == sorted(hits, key=lambda h: (-h.score, h.rank))
