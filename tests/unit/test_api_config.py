from __future__ import annotations

from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app
from agentic_rag.config import load_settings
from agentic_rag.indexing import Indexer
from agentic_rag.retrieval.retriever import Retriever
from agentic_rag.agent.orchestrator import Orchestrator
from agentic_rag.tools.registry import ToolRegistry


def test_settings_repr_hides_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value-should-hide")
    monkeypatch.setenv("DOCUMENTS_DIR", str(tmp_path))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "i"))
    settings = load_settings()
    text = repr(settings)
    assert "sk-secret" not in text
    assert "***" in text


def test_api_health_and_ask(tmp_settings, fake_embedder, store, monkeypatch):
    indexer = Indexer(tmp_settings, fake_embedder, store)
    indexer.build(rebuild=True)
    orch = Orchestrator(tmp_settings, Retriever(store, fake_embedder), ToolRegistry())

    def fake_stack():
        return tmp_settings, orch, indexer, store

    monkeypatch.setattr("agentic_rag.api.app.load_stack", lambda: (orch, indexer, store))
    monkeypatch.setattr("agentic_rag.api.app.get_settings", lambda: tmp_settings)

    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["chunks"] >= 1
    bad = client.post("/ask", json={"question": ""})
    assert bad.status_code == 422
    ok = client.post("/ask", json={"question": "What is the hotel reimbursement limit?"})
    assert ok.status_code == 200
    body = ok.json()
    assert "answer" in body
    assert "document_evidence" in body
    assert "sk-" not in str(body)
