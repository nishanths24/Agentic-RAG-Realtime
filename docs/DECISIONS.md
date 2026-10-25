### Decision: Embedding provider   (2026-10-25)

Options considered: sentence-transformers local model, hosted embeddings API, hashing bag-of-words embedder
Criteria: learning value, reliability, cost, Windows/OneDrive, Python 3.12, no huge first-run download
Verification: hashing embedder implemented and tested locally; sentence-transformers not installed to avoid large model files on OneDrive
Choice and reason: start with a deterministic hashing embedder behind `EmbeddingProvider`. It is offline, free, and good enough for a small labeled corpus.
Revisit if: corpus grows or paraphrase recall is too weak — then add sentence-transformers through the same interface.

### Decision: Vector store   (2026-10-25)

Options considered: NumPy cosine + JSON/NPY, Chroma, FAISS
Choice: NumPy local store for learning and zero setup.
Revisit if: document count exceeds a few thousand.

### Decision: LLM   (2026-10-25)

Options considered: required OpenAI, local llama, extractive composer
Choice: extractive grounded composer by default; optional OpenAI-compatible client when `LLM_PROVIDER` and `LLM_API_KEY` are set.
Reason: the demo must run with an empty `.env`.

### Decision: Live APIs   (2026-10-25)

- Time: `zoneinfo` + `tzdata` on Windows
- Currency: Frankfurter (`https://api.frankfurter.app`) ECB rates, no key
- Weather: Open-Meteo geocoding + forecast, no key
- Calculator: AST, never `eval`
Verification: unit tests mock HTTP; live calls happen only from the running app/CLI.

### Decision: UI   (2026-10-25)

Options: Streamlit vs static page on FastAPI
Choice: static premium page served by FastAPI so one `serve` command runs API + UI.
