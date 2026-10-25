# Architecture

```
question
  → RuleRouter (documents / tools / both / neither)
  → Retriever (hashing embeddings + NumPy cosine search)
  → ToolRegistry (time, calculator, currency, weather)
  → optional follow-up (e.g. policy USD amount → INR)
  → AnswerGenerator (extractive or optional LLM)
  → FinalAnswer {answer, document_evidence, live_evidence, caveats, trace}
```

Ingestion: discover `.txt`/`.md` → normalize → word-aware chunks with metadata → embed → persist `data/index/`.

Interfaces: CLI (`python main.py ...`) and FastAPI (`/ask`, `/health`, `/index`, `/sources`) plus the Aether UI at `/`.
