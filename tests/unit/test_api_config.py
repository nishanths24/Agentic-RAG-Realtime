from __future__ import annotations

import sys
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

    api_module = sys.modules["agentic_rag.api.app"]
    monkeypatch.setattr(api_module, "load_stack", lambda: (orch, indexer, store))
    monkeypatch.setattr(api_module, "get_settings", lambda: tmp_settings)

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
    assert body["sufficient_evidence"] is True
    assert "sk-" not in str(body)

    broad = client.post("/ask", json={"question": "What information does the travel policy provide about employee expenses?"})
    assert broad.status_code == 200
    broad_body = broad.json()
    assert broad_body["sufficient_evidence"] is True
    assert len(broad_body["document_evidence"]) >= 1
    assert "Company Travel Policy (effective January 2024) [D1]" != broad_body["answer"].strip()
    assert "50 USD" in broad_body["answer"] or "travel" in broad_body["answer"].lower()

    unsupported = client.post("/ask", json={"question": "What is the company's reimbursement policy for private jet travel?"})
    assert unsupported.status_code == 200
    unsupp_body = unsupported.json()
    assert unsupp_body["sufficient_evidence"] is False
    assert unsupp_body["document_evidence"] == []
    assert "available documents do not provide sufficient evidence" in unsupp_body["answer"].lower()
    assert "private jet travel" in unsupp_body["answer"].lower()


def test_static_ui_served_without_obsolete_pills():
    client = TestClient(create_app())
    html = client.get("/")
    assert html.status_code == 200
    assert "sources-fold" in html.text
    assert "rebuild" in html.text
    assert "new-chat" in html.text
    assert "sidebar" in html.text
    assert "sources-search" in html.text
    assert "open-upload-btn" not in html.text
    assert "topbar-add-doc" not in html.text
    assert "sidebar-add-doc" in html.text
    assert "sidebar-toggle" in html.text
    assert "robot-logo.png" in html.text
    assert "upload-modal" in html.text
    assert "pills" not in html.text

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "pills" not in js.text
    assert "EXAMPLES" not in js.text
    assert "showEmpty" in js.text
    assert "refreshMeta" in js.text
    assert "openModal" in js.text
    assert "sidebarAddDocBtn" in js.text
    assert "robot-logo.png" in js.text
    assert "stopSpeech" in js.text
    assert "answer-actions" in js.text
    assert "welcome-heading" in js.text
    assert "suggestion-chip" in js.text
    assert "Hello, welcome to Agentic RAG" in js.text
    assert "What is the hotel reimbursement limit?" in js.text
    assert "What is the daily meal allowance in INR?" in js.text
    assert "What time is it in Tokyo right now?" in js.text

    css = client.get("/static/styles.css")
    assert css.status_code == 200
    assert ".app" in css.text
    assert ".rail" in css.text
    assert ".stage" in css.text
    assert ".sidebar-collapsed" in css.text
    assert ".brand-robot-logo" in css.text
    assert ".welcome-screen" in css.text
    assert ".composer" in css.text
    assert ".action-btn" in css.text
    assert ".answer-actions" in css.text
    assert ".feature-card" in css.text
    assert ".modal-overlay" in css.text
    assert ".upload-dropzone" in css.text

    logo = client.get("/static/robot-logo.png")
    assert logo.status_code == 200
    assert "image" in logo.headers.get("content-type", "")


def test_ui_robot_logo_and_sidebar_toggle():
    client = TestClient(create_app())
    html = client.get("/").text
    js = client.get("/static/app.js").text
    css = client.get("/static/styles.css").text

    # Robot logo placements
    assert "/static/robot-logo.png" in html
    assert 'class="brand-robot-logo"' in html
    assert 'class="topbar-robot-logo"' in html
    assert "welcome-robot-logo" in js
    assert "agent-robot-icon" in js

    # Sidebar toggle hamburger
    assert 'id="sidebar-toggle"' in html
    assert 'aria-expanded="true"' in html
    assert "toggleSidebar" in js
    assert "sidebar-collapsed" in js
    assert ".app.sidebar-collapsed .rail" in css

    # Add Document in sidebar only (with single plus icon)
    assert 'id="sidebar-add-doc"' in html
    assert 'id="topbar-add-doc"' not in html
    assert "sidebarAddDocBtn" in js
    assert "openModal" in js

    # Topbar New Chat
    assert 'id="topbar-new-chat"' in html

    # Read Aloud preserved
    assert "cleanSpokenText" in js
    assert "handleTTSClick" in js
    assert "SpeechSynthesisUtterance" in js
    assert "stopSpeech" in js
    assert "tts-btn" in js

    # Full width workspace layout (no blank right-hand space)
    assert "flex: 1" in css or "flex: 1 1 0%" in css
    assert "min-width: 0" in css
    assert "width: 100%" in css

    # Header branding state visibility
    assert 'id="topbar-brand"' in html
    assert ".topbar-brand.visible" in css
    assert "setHeaderBrandingVisible" in js
    assert "topbarBrand" in js
