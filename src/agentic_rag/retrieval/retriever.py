from __future__ import annotations

import re

from agentic_rag.embeddings.hashing import HashingEmbeddingProvider
from agentic_rag.errors import RetrievalError
from agentic_rag.models import RetrievedPassage
from agentic_rag.vectorstore.numpy_store import NumpyVectorStore

_WS = re.compile(r"\s+")


def normalize_query(query: str, max_chars: int = 2000) -> str:
    cleaned = _WS.sub(" ", query).strip()
    if not cleaned:
        raise RetrievalError("Query must be non-empty")
    if len(cleaned) > max_chars:
        raise RetrievalError(f"Query exceeds {max_chars} characters")
    return cleaned


class Retriever:
    def __init__(self, store: NumpyVectorStore, embedder: HashingEmbeddingProvider) -> None:
        self.store = store
        self.embedder = embedder

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float = 0.0,
        max_chars: int = 2000,
    ) -> list[RetrievedPassage]:
        cleaned = normalize_query(query, max_chars=max_chars)
        vector = self.embedder.embed_query(cleaned)
        results = self.store.search(vector, top_k=top_k)
        return [item for item in results if item.score >= min_score]
