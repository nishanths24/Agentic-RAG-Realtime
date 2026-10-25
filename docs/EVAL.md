# Evaluation

| Date | Embedder | Chunk size/overlap | top_k | Notes |
|---|---|---|---|---|
| 2026-10-25 | hashing-bow-v1 | 1000/150 | 4 | Small five-document corpus. Run `python scripts/eval_retrieval.py` and `python scripts/eval_router.py` after indexing. |

The evaluation set is tiny and was also used while building the demo — treat numbers as a smoke baseline, not a published benchmark.

Unanswerable questions must abstain. Prompt-injection text in `injection_sample.txt` is evidence only; the router keys off the **user question**, not document text.
