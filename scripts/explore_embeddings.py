from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.embeddings.hashing import HashingEmbeddingProvider


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


def main() -> None:
    embedder = HashingEmbeddingProvider()
    pairs = [
        ("hotel reimbursement limit", "nightly hotel cap for travel"),
        ("hotel reimbursement limit", "encrypted laptop policy"),
        ("meal allowance USD", "daily food stipend in dollars"),
    ]
    for left, right in pairs:
        score = cosine(embedder.embed_query(left), embedder.embed_query(right))
        print(f"{score:.3f}  {left!r} vs {right!r}")


if __name__ == "__main__":
    main()
