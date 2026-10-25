# Milestone 0 audit (verified 2026-10-25)

## 1. File tree (at audit time, before the full build)

Existing: `main.py`, `run_ingestion.py`, `src/agentic_rag/ingestion.py`, `tests/test_ingestion.py`, `data/documents/travel_policy.txt`, `.gitignore`, `.github/workflows/build.yml`, `README.md`, `AgenticRAG.md` (untracked).

## 2. Source summary

- `ingestion.py` had `clean_text`, `read_text_file`, character-window `chunk_text`.
- Six unittest cases; overlap test used a spaceless string.
- `main.py` printed a hello message. No embeddings, tools, API, or UI.

## 3. Environment

- Python 3.12.10, Git 2.51.0, Windows, PowerShell.
- `.venv` existed but only `pip` was installed (`pytest` missing).
- Interpreter inside `.venv`: `.venv\Scripts\python.exe`.

## 4. Git

Repo on `main`, tracking `origin/main`, clean except untracked `AgenticRAG.md`.

## 5. Test baseline

`python -m pytest` failed: no pytest module. `python -m unittest discover tests` was the documented CI path.

## 6. Scripts

`python main.py` printed a startup hello. `python run_ingestion.py` printed cleaned travel policy text and character chunks.

## 7. Defects

Word-splitting chunks, no metadata, no directory ingestion, `chunk_size <= overlap` raised ValueError (good), no type-safe settings.

## 8–10. Follow-up

The application in this repository now implements ingestion through API/UI. OneDrive still applies: keep `data/index/`, `.venv/`, and `.env` ignored; do not commit secrets.
