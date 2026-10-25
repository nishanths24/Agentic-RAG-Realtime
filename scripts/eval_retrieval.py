from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.runtime import build_orchestrator, get_settings


def main() -> None:
    settings = get_settings()
    orch, indexer, store = build_orchestrator(settings)
    if store.count() == 0:
        indexer.build(rebuild=True)
    path = ROOT / "eval" / "retrieval_questions.jsonl"
    hits1 = hits3 = total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("unanswerable"):
            continue
        total += 1
        results = orch.retriever.search(row["question"], top_k=5, min_score=0.0)
        paths = [r.chunk.source_path for r in results]
        if paths[:1] == [row["source_path"]]:
            hits1 += 1
        if row["source_path"] in paths[:3]:
            hits3 += 1
        print(f"{row['question'][:60]:60}  top={paths[:3]}")
    print(f"hit@1={hits1/total:.2f} hit@3={hits3/total:.2f} n={total}")


if __name__ == "__main__":
    main()
