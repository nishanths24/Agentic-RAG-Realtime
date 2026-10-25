from __future__ import annotations

import re

from agentic_rag.embeddings.hashing import HashingEmbeddingProvider
from agentic_rag.errors import RetrievalError
from agentic_rag.models import RetrievedPassage
from agentic_rag.vectorstore.numpy_store import NumpyVectorStore

_WS = re.compile(r"\s+")
_STOP = {
    "the",
    "and",
    "for",
    "our",
    "are",
    "was",
    "were",
    "what",
    "when",
    "who",
    "how",
    "why",
    "does",
    "did",
    "this",
    "that",
    "with",
    "from",
    "your",
    "you",
    "into",
    "about",
    "can",
    "may",
    "not",
    "any",
    "all",
    "per",
}


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
        pool = self.store.search(vector, top_k=max(top_k * 3, top_k))
        q_tokens = _content_tokens(cleaned)
        ranked: list[RetrievedPassage] = []
        for item in pool:
            lexical = _lexical_overlap(q_tokens, item.chunk.text)
            combined = 0.65 * item.score + 0.35 * lexical
            if combined < min_score:
                continue
            if q_tokens and lexical == 0.0 and item.score < 0.35:
                continue
            ranked.append(
                RetrievedPassage(chunk=item.chunk, score=combined, rank=item.rank)
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        selected = ranked[:top_k]
        return [
            RetrievedPassage(chunk=item.chunk, score=item.score, rank=index)
            for index, item in enumerate(selected, start=1)
        ]


def _content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 2 and tok not in _STOP
    }


def _lexical_overlap(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _content_tokens(text)
    return len(query_tokens & doc_tokens) / len(query_tokens)
