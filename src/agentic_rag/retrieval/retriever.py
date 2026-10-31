from __future__ import annotations

import re

from agentic_rag.embeddings.hashing import HashingEmbeddingProvider
from agentic_rag.errors import RetrievalError
from agentic_rag.models import RetrievedPassage
from agentic_rag.vectorstore.numpy_store import NumpyVectorStore

_WS = re.compile(r"\s+")
_PROJECT_RE = re.compile(r"\bproject\s+([A-Za-z0-9_-]+)\b", re.I)
_META_BROAD_RE = re.compile(
    r"\b(summarize|summary|overview|main\s+topic|key\s+information|key\s+points|key\s+details|"
    r"uploaded\s+documents?|indexed\s+documents?|all\s+documents|what\s+does\s+.*(?:contain|cover|say))\b",
    re.I,
)


_PROJECT_ATTR_WORDS = {
    "code", "name", "brief", "review", "overview", "details", "detail", "plan",
    "update", "updates", "milestone", "milestones", "owner", "lead", "status",
    "document", "documents", "documentation", "work", "initiative", "info", "information",
    "the", "a", "an", "this", "that", "is", "was", "are", "were", "our", "my", "your", "for", "in", "of",
}


def _extract_projects(text: str) -> set[str]:
    res: set[str] = set()
    for m in re.finditer(r"\bproject\s+([A-Za-z0-9_-]+)", text, re.I):
        w = re.sub(r"['\u2019]s?$", "", m.group(1)).lower()
        if w not in _PROJECT_ATTR_WORDS and len(w) > 1:
            res.add(w)
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9_-]*)\s+project\b", text):
        w = re.sub(r"['\u2019]s?$", "", m.group(1)).lower()
        if w not in _PROJECT_ATTR_WORDS and len(w) > 1:
            res.add(w)
    return res
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
        q_projects = _extract_projects(cleaned)
        is_broad = bool(_META_BROAD_RE.search(cleaned))

        pool_size = max(top_k * 5, 20) if (is_broad or q_projects) else max(top_k * 3, top_k)
        pool = self.store.search(vector, top_k=pool_size)
        q_tokens = _content_tokens(cleaned)
        ranked: list[RetrievedPassage] = []

        for item in pool:
            chunk_projects = _extract_projects(item.chunk.text)
            # If query explicitly targets a named project, reject chunks from a conflicting project
            if q_projects and chunk_projects and not (q_projects & chunk_projects):
                continue

            lexical = _lexical_overlap(q_tokens, item.chunk.text)
            combined = 0.65 * item.score + 0.35 * lexical

            is_project_match = bool(
                q_projects
                and (
                    (q_projects & chunk_projects)
                    or any(proj in item.chunk.source_path.lower() for proj in q_projects)
                )
            )
            if is_project_match:
                combined += 0.4
            elif is_broad and item.chunk.chunk_index == 0:
                combined += 0.2

            if combined < min_score:
                continue
            if not is_broad and q_tokens and lexical == 0.0 and item.score < 0.35:
                continue

            ranked.append(
                RetrievedPassage(chunk=item.chunk, score=combined, rank=item.rank)
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        effective_top_k = max(top_k, 6) if (is_broad and q_projects) else top_k
        selected = ranked[:effective_top_k]
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
