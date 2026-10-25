from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from agentic_rag.errors import EmbeddingDimensionError, EmbeddingError


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def validate_vectors(vectors: Sequence[Sequence[float]], dimension: int) -> None:
    if not vectors:
        raise EmbeddingError("No vectors returned")
    for vector in vectors:
        if len(vector) != dimension:
            raise EmbeddingDimensionError(
                f"Expected dimension {dimension}, got {len(vector)}"
            )
