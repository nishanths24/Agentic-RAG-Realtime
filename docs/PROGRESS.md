| Milestone | Status | Date | Evidence | Notes |
|---|---|---|---|---|
| 0 Audit | ✅ | 2026-10-25 | `docs/AUDIT.md` | venv had no pytest |
| 1 Ingestion | ✅ | 2026-10-25 | unit tests in `tests/unit/test_chunking.py`, `test_loaders.py` | Word-aware chunking |
| 2 Config & quality | ✅ | 2026-10-25 | `pyproject.toml`, `.env.example`, `scripts/check.ps1` | Settings repr redacts secrets |
| 3 Embeddings | ✅ | 2026-10-25 | hashing provider + `scripts/explore_embeddings.py` | Offline, no model download |
| 4 Vector store | ✅ | 2026-10-25 | `NumpyVectorStore` persistence tests | Idempotent upsert |
| 5 Retrieval | ✅ | 2026-10-25 | `eval/retrieval_questions.jsonl` | See `docs/EVAL.md` |
| 6 Basic RAG | ✅ | 2026-10-25 | extractive composer + citation checks | LLM optional |
| 7 Tools | ✅ | 2026-10-25 | time, calculator, currency, weather | HTTP mocked in unit tests |
| 8 Agent routing | ✅ | 2026-10-25 | `RuleRouter` + orchestrator limits | Follow-up currency step |
| 9 Evidence-aware answers | ✅ | 2026-10-25 | document vs live sections | UI + API payload |
| 10 API | ✅ | 2026-10-25 | FastAPI `/health` `/ask` `/index` `/sources` | Bound to 127.0.0.1 by default |
| 11 UI | ✅ | 2026-10-25 | `src/agentic_rag/web/static` | Premium local desk UI |
| 12 Observability | ✅ | 2026-10-25 | trace_id logging, redaction | Rotating `logs/` |
| 13 Docs & polish | ✅ | 2026-10-25 | README + architecture | Hashing embeddings (no large download) |
