$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest
python -m pytest --cov=agentic_rag --cov-report=term-missing
