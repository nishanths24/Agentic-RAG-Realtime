# Agentic RAG Realtime

Grounded question answering over local documents, with optional live tools for time, weather, currency, and calculator. Answers cite document chunks and tool results separately, and abstain when there is not enough evidence.

## Quickstart (Windows / PowerShell)

```powershell
cd "C:\Users\nisha\OneDrive\Desktop\Agentic RAG Realtime"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
copy .env.example .env
python main.py index --rebuild
python -m pytest
python main.py serve
```

Open http://127.0.0.1:8000

If you do not create `.env`, built-in defaults still work: hashing embeddings and the extractive answer composer (`LLM_PROVIDER=none`). Do not commit `.env`.

## Example questions

| Question | Expected behavior |
|---|---|
| What is the hotel reimbursement limit in our travel policy? | Retrieve `travel_policy.txt` and cite the 5000 INR hotel limit. |
| Who won the 2030 World Cup? | Insufficient evidence; no guess. |
| What time is it in Tokyo right now? | `current_time` tool with `Asia/Tokyo`. |

CLI:

```powershell
python main.py ask "What is the hotel reimbursement limit in our travel policy?"
python main.py ask "Who won the 2030 World Cup?"
python main.py ask "What time is it in Tokyo right now?" --trace
```

## API

- `GET /health`
- `GET /sources`
- `POST /ask` body: `{"question": "...", "trace": true}`
- `POST /index?rebuild=true`

## Layout

- `src/agentic_rag/` — ingestion, embeddings, retrieval, tools, agent, API, UI
- `data/documents/` — source files
- `data/index/` — generated vectors (git-ignored)
- `tests/` — unit tests

This project lives in OneDrive. Keep `data/index/`, `logs/`, `.venv/`, and `.env` out of Git; OneDrive file locks can interrupt index writes.
