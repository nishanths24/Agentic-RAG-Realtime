from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from agentic_rag.embeddings.base import validate_vectors
from agentic_rag.errors import EmbeddingError

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _add(vec: list[float], token: str, weight: float) -> None:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    dim = len(vec)
    for offset in range(0, 12, 4):
        idx = int.from_bytes(digest[offset : offset + 4], "big") % dim
        sign = 1.0 if digest[offset + 4] % 2 == 0 else -1.0
        vec[idx] += sign * weight


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


class HashingEmbeddingProvider:
    """Deterministic bag-of-words + character n-gram hashing embedder.

    Offline, no model download. Shared tokens produce similar vectors, which is
    enough for a small labeled corpus and fully reproducible tests.
    """

    def __init__(self, dimension: int = 384, model_name: str = "hashing-bow-v1") -> None:
        if dimension <= 0:
            raise EmbeddingError("dimension must be positive")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = _tokenize(text)
        for token in tokens:
            _add(vec, f"w:{token}", 1.0)
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    _add(vec, f"g:{token[i:i+3]}", 0.35)
        lowered = text.lower()
        for i in range(max(0, len(lowered) - 4)):
            gram = lowered[i : i + 5]
            if gram.strip():
                _add(vec, f"c:{gram}", 0.15)
        return _normalize(vec)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            raise EmbeddingError("texts must be non-empty")
        vectors = [self._embed_one(text) for text in texts]
        validate_vectors(vectors, self._dimension)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("query must be non-empty")
        vector = self._embed_one(text)
        validate_vectors([vector], self._dimension)
        return vector


class FakeEmbeddingProvider(HashingEmbeddingProvider):
    """Alias used in tests; identical to the hashing provider."""

    def __init__(self, dimension: int = 32, model_name: str = "fake-hash-v1") -> None:
        super().__init__(dimension=dimension, model_name=model_name)
