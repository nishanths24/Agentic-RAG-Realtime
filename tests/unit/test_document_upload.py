from __future__ import annotations

import io
import sys
from pathlib import Path
from fastapi.testclient import TestClient

from agentic_rag.api.app import create_app
from agentic_rag.indexing import Indexer
from agentic_rag.retrieval.retriever import Retriever
from agentic_rag.agent.orchestrator import Orchestrator
from agentic_rag.tools.registry import ToolRegistry


def _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch):
    indexer = Indexer(tmp_settings, fake_embedder, store)
    indexer.build(rebuild=True)
    retriever = Retriever(store, fake_embedder)
    orch = Orchestrator(tmp_settings, retriever, ToolRegistry())

    api_module = sys.modules["agentic_rag.api.app"]
    monkeypatch.setattr(api_module, "load_stack", lambda: (orch, indexer, store))
    monkeypatch.setattr(api_module, "get_settings", lambda: tmp_settings)

    client = TestClient(create_app())
    return client, orch, indexer, store


def test_upload_and_index_success(tmp_settings, fake_embedder, store, monkeypatch):
    client, orch, indexer, store = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    doc_content = "Remote Work Policy: Full-time employees are eligible for up to two remote work days per week with manager approval."
    files = {"file": ("remote_work_policy.txt", io.BytesIO(doc_content.encode("utf-8")), "text/plain")}

    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["filename"] == "remote_work_policy.txt"
    assert data["size_bytes"] == len(doc_content.encode("utf-8"))
    assert data["extension"] == ".txt"
    assert data["chunks_total"] >= 1

    # Verify file saved on disk
    saved_file = tmp_settings.documents_dir / "remote_work_policy.txt"
    assert saved_file.exists()
    assert "Remote Work Policy" in saved_file.read_text(encoding="utf-8")

    # Verify that the newly uploaded document can be retrieved by /ask immediately
    ask_resp = client.post("/ask", json={"question": "What is the remote work policy?"})
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert ask_data["sufficient_evidence"] is True
    assert any("remote work" in c["preview"].lower() for c in ask_data["document_evidence"])


def test_upload_project_falcon_workflow(tmp_settings, fake_embedder, store, monkeypatch):
    client, orch, indexer, store = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    content = "Project Falcon uses the internal code ORCHID-742. Its scheduled review date is 18 September 2027."
    files = {"file": ("rag_upload_test.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}

    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Query 1: Internal code
    q1 = client.post("/ask", json={"question": "What is the internal code for Project Falcon?"})
    assert q1.status_code == 200
    data1 = q1.json()
    assert data1["sufficient_evidence"] is True
    assert "ORCHID-742" in data1["answer"]
    assert any(c["reference"] == "rag_upload_test.txt" for c in data1["document_evidence"])
    assert any("rag_upload_test.txt" in c["preview"] or "ORCHID-742" in c["preview"] for c in data1["document_evidence"])
    assert "[D1]" in data1["answer"]

    # Query 2: Review date
    q2 = client.post("/ask", json={"question": "When is Project Falcon's scheduled review date?"})
    assert q2.status_code == 200
    data2 = q2.json()
    assert data2["sufficient_evidence"] is True
    assert "18 September 2027" in data2["answer"]
    assert "[D1]" in data2["answer"]

    # Preservation check: previously existing document remains searchable
    q_existing = client.post("/ask", json={"question": "What is the hotel reimbursement limit?"})
    assert q_existing.status_code == 200
    data_existing = q_existing.json()
    assert data_existing["sufficient_evidence"] is True
    assert "5000 INR" in data_existing["answer"]

    # Unrelated check: insufficient evidence handling remains intact
    q_unrelated = client.post("/ask", json={"question": "What is the company's reimbursement policy for private jet travel?"})
    assert q_unrelated.status_code == 200
    data_unrelated = q_unrelated.json()
    assert data_unrelated["sufficient_evidence"] is False
    assert data_unrelated["document_evidence"] == []


def test_upload_markdown_success(tmp_settings, fake_embedder, store, monkeypatch):
    client, orch, indexer, store = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    doc_content = "# Equipment Stipend\n\nAll engineers receive a standard hardware stipend of 2000 USD every two years."
    files = {"file": ("hardware_policy.md", io.BytesIO(doc_content.encode("utf-8")), "text/markdown")}

    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["filename"] == "hardware_policy.md"
    assert data["extension"] == ".md"

    # Verify in /sources
    sources_resp = client.get("/sources")
    assert sources_resp.status_code == 200
    sources_data = sources_resp.json()
    found = next((d for d in sources_data["documents"] if d["filename"] == "hardware_policy.md"), None)
    assert found is not None
    assert found["extension"] == ".md"
    assert found["status"] == "indexed"
    assert found["size_bytes"] == len(doc_content.encode("utf-8"))


def test_upload_unsupported_file_type(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    files = {"file": ("unsupported.pdf", io.BytesIO(b"%PDF-1.4 fake binary content"), "application/pdf")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "unsupported file type" in detail
    assert ".txt" in detail and ".md" in detail

    # Verify file was not written to disk
    assert not (tmp_settings.documents_dir / "unsupported.pdf").exists()


def test_upload_empty_file(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()

    # Whitespace-only file
    files_spaces = {"file": ("spaces.txt", io.BytesIO(b"   \n\t   "), "text/plain")}
    resp_spaces = client.post("/documents/upload", files=files_spaces)
    assert resp_spaces.status_code == 400
    assert "no readable text" in resp_spaces.json()["detail"].lower()


def test_upload_oversized_file(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    # Content exceeding max_file_bytes (tmp_settings fixture defaults or 1MB)
    oversized_bytes = b"A" * (tmp_settings.max_file_bytes + 50)
    files = {"file": ("huge.txt", io.BytesIO(oversized_bytes), "text/plain")}

    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 413
    assert "exceeds maximum allowed size" in resp.json()["detail"].lower()
    assert not (tmp_settings.documents_dir / "huge.txt").exists()


def test_upload_duplicate_filename_handling(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    # Upload first time
    original_text = "Version 1 of the policy statement."
    files = {"file": ("version_test.txt", io.BytesIO(original_text.encode("utf-8")), "text/plain")}
    resp1 = client.post("/documents/upload", files=files)
    assert resp1.status_code == 200

    # Upload duplicate without overwrite -> should return 409 Conflict
    files2 = {"file": ("version_test.txt", io.BytesIO(b"New version should be rejected without overwrite"), "text/plain")}
    resp2 = client.post("/documents/upload", files=files2)
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"].lower()

    # Verify original file content was preserved
    saved_file = tmp_settings.documents_dir / "version_test.txt"
    assert saved_file.read_text(encoding="utf-8") == original_text

    # Upload duplicate with overwrite=true -> should succeed and update
    files3 = {"file": ("version_test.txt", io.BytesIO(b"Version 2 updated text with more information."), "text/plain")}
    resp3 = client.post("/documents/upload", files=files3, data={"overwrite": "true"})
    assert resp3.status_code == 200
    assert "Version 2 updated" in saved_file.read_text(encoding="utf-8")


def test_upload_path_traversal_rejection(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    for bad_name in ["../../evil.txt", "subdir/evil.txt", "..\\evil.txt", "test\0.txt", ".hidden.txt"]:
        files = {"file": (bad_name, io.BytesIO(b"malicious file attempt"), "text/plain")}
        resp = client.post("/documents/upload", files=files)
        assert resp.status_code == 400
        assert not (tmp_settings.documents_dir / "evil.txt").exists()


def test_upload_invalid_utf8(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    # Invalid raw bytes that fail UTF-8 decoding
    bad_bytes = b"\x80\x81\xff\xfe"
    files = {"file": ("bad_utf8.txt", io.BytesIO(bad_bytes), "text/plain")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 400
    assert "utf-8" in resp.json()["detail"].lower()


def test_upload_indexing_failure_cleanup(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, indexer, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    # Force indexer.build to raise an unexpected error
    def broken_build(*args, **kwargs):
        raise RuntimeError("Simulated vector storage disk error")

    monkeypatch.setattr(indexer, "build", broken_build)

    files = {"file": ("will_fail.txt", io.BytesIO(b"Some valid policy text to index"), "text/plain")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 500
    assert "indexing failed" in resp.json()["detail"].lower()

    # Crucial security & data integrity: uploaded file should be removed if indexing fails
    assert not (tmp_settings.documents_dir / "will_fail.txt").exists()


def test_sources_endpoint_enriched_metadata(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    resp = client.get("/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "chunks" in data
    assert len(data["documents"]) >= 1

    doc = data["documents"][0]
    assert "path" in doc
    assert "filename" in doc
    assert "extension" in doc
    assert "size_bytes" in doc
    assert "status" in doc
    assert doc["status"] == "indexed"


def test_upload_write_protection_with_api_key(tmp_settings, fake_embedder, store, monkeypatch):
    # Set an API_KEY in settings
    from dataclasses import replace
    protected_settings = replace(tmp_settings, api_key="secret-token-xyz")

    indexer = Indexer(protected_settings, fake_embedder, store)
    indexer.build(rebuild=True)
    orch = Orchestrator(protected_settings, Retriever(store, fake_embedder), ToolRegistry())

    api_module = sys.modules["agentic_rag.api.app"]
    monkeypatch.setattr(api_module, "load_stack", lambda: (orch, indexer, store))
    monkeypatch.setattr(api_module, "get_settings", lambda: protected_settings)

    client = TestClient(create_app())

    files = {"file": ("auth_test.txt", io.BytesIO(b"Testing write protection with API key"), "text/plain")}

    # Missing API key header -> 401
    resp_no_auth = client.post("/documents/upload", files=files)
    assert resp_no_auth.status_code == 401

    # Wrong API key header -> 401
    resp_bad_auth = client.post("/documents/upload", files=files, headers={"x-api-key": "wrong-key"})
    assert resp_bad_auth.status_code == 401

    # Valid API key header -> 200
    files["file"][1].seek(0)
    resp_auth = client.post("/documents/upload", files=files, headers={"x-api-key": "secret-token-xyz"})
    assert resp_auth.status_code == 200
    assert resp_auth.json()["status"] == "ok"


def test_upload_persistence_and_restart(tmp_settings, fake_embedder, store, monkeypatch):
    """Verify uploaded document remains searchable after application restart."""
    client, orch, indexer, store = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    content = "Project Falcon uses the internal code ORCHID-742. Its scheduled review date is 18 September 2027."
    files = {"file": ("project_falcon_workflow.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}

    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # 1. Immediate retrieval before restart
    q1 = client.post("/ask", json={"question": "What is the internal code for Project Falcon?"})
    assert q1.status_code == 200
    data1 = q1.json()
    assert data1["sufficient_evidence"] is True
    assert "ORCHID-742" in data1["answer"]
    # Citation must display the actual document filename
    assert any(c["reference"] == "project_falcon_workflow.txt" for c in data1["document_evidence"])
    assert any("ORCHID-742" in c["preview"] for c in data1["document_evidence"])

    # 2. Simulate complete restart: re-instantiate store from index directory on disk
    from agentic_rag.vectorstore.numpy_store import NumpyVectorStore
    restarted_store = NumpyVectorStore(
        tmp_settings.index_dir,
        model_name=fake_embedder.model_name,
        dimension=fake_embedder.dimension,
        chunk_size=tmp_settings.chunk_size,
        chunk_overlap=tmp_settings.chunk_overlap,
    )
    restarted_store.load()
    assert restarted_store.count() >= 2

    restarted_retriever = Retriever(restarted_store, fake_embedder)
    restarted_indexer = Indexer(tmp_settings, fake_embedder, restarted_store)
    restarted_orch = Orchestrator(tmp_settings, restarted_retriever, ToolRegistry())

    api_module = sys.modules["agentic_rag.api.app"]
    monkeypatch.setattr(api_module, "load_stack", lambda: (restarted_orch, restarted_indexer, restarted_store))
    restarted_client = TestClient(create_app())

    # Query after restart: Project Falcon remains searchable
    q_after = restarted_client.post("/ask", json={"question": "What is the internal code for Project Falcon?"})
    assert q_after.status_code == 200
    data_after = q_after.json()
    assert data_after["sufficient_evidence"] is True
    assert "ORCHID-742" in data_after["answer"]
    assert any(c["reference"] == "project_falcon_workflow.txt" for c in data_after["document_evidence"])

    # Existing document preservation check
    q_existing = restarted_client.post("/ask", json={"question": "What is the hotel reimbursement limit?"})
    assert q_existing.status_code == 200
    assert q_existing.json()["sufficient_evidence"] is True
    assert "5000 INR" in q_existing.json()["answer"]

    # Unsupported question abstention check (no fabrication)
    q_unsupported = restarted_client.post("/ask", json={"question": "What is the reimbursement policy for deep-sea submarine travel?"})
    assert q_unsupported.status_code == 200
    data_unsupported = q_unsupported.json()
    assert data_unsupported["sufficient_evidence"] is False
    assert data_unsupported["document_evidence"] == []


def test_offline_document_addition_indexed_on_restart(tmp_settings, fake_embedder, store, monkeypatch):
    """Verify that documents added while offline are automatically indexed on restart."""
    indexer = Indexer(tmp_settings, fake_embedder, store)
    indexer.build(rebuild=True)
    assert store.count() >= 1

    # Add a document directly to the directory while offline
    offline_file = tmp_settings.documents_dir / "project_phoenix.txt"
    offline_file.write_text("Project Phoenix review deadline is 22 December 2028.", encoding="utf-8")

    from agentic_rag.runtime import build_orchestrator
    orch, new_indexer, new_store = build_orchestrator(tmp_settings)
    assert new_store.count() > store.count()

    ans = orch.ask("When is the Project Phoenix review deadline?")
    assert ans.sufficient_evidence is True
    assert "22 December 2028" in ans.answer
    assert any(c.reference == "project_phoenix.txt" for c in ans.document_evidence)


def test_delete_document_success(tmp_settings, fake_embedder, store, monkeypatch):
    client, orch, indexer, store = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    content = "Project Vega launch is set for 01 January 2030."
    files = {"file": ("project_vega.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 200

    # Ensure searchable
    q1 = client.post("/ask", json={"question": "When is the Project Vega launch?"})
    assert q1.status_code == 200
    assert q1.json()["sufficient_evidence"] is True
    assert "01 January 2030" in q1.json()["answer"]

    # Delete the document
    del_resp = client.delete("/documents/project_vega.txt")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["status"] == "ok"
    assert del_data["filename"] == "project_vega.txt"
    assert del_data["chunks_removed"] >= 1

    # Verify file is deleted from disk
    assert not (tmp_settings.documents_dir / "project_vega.txt").exists()

    # Verify content is no longer retrievable
    q2 = client.post("/ask", json={"question": "When is the Project Vega launch?"})
    assert q2.status_code == 200
    assert q2.json()["sufficient_evidence"] is False
    assert q2.json()["document_evidence"] == []

    # Verify remaining documents remain searchable
    q_existing = client.post("/ask", json={"question": "What is the hotel reimbursement limit?"})
    assert q_existing.status_code == 200
    assert q_existing.json()["sufficient_evidence"] is True
    assert "5000 INR" in q_existing.json()["answer"]

    # Verify /sources no longer lists deleted document
    sources_resp = client.get("/sources")
    assert sources_resp.status_code == 200
    sources_list = sources_resp.json()["documents"]
    assert not any(d["filename"] == "project_vega.txt" for d in sources_list)


def test_delete_document_not_found_404(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    resp = client.delete("/documents/nonexistent_file.txt")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_delete_document_security_validations(tmp_settings, fake_embedder, store, monkeypatch):
    client, _, _, _ = _setup_api_client(tmp_settings, fake_embedder, store, monkeypatch)

    # Path traversal attempt
    resp_traversal = client.delete("/documents/..%2fevil.txt")
    assert resp_traversal.status_code in (400, 404)

    # Hidden file attempt
    resp_hidden = client.delete("/documents/.hidden.txt")
    assert resp_hidden.status_code == 400

    # Unsupported extension attempt
    resp_ext = client.delete("/documents/script.py")
    assert resp_ext.status_code == 400
    assert "unsupported file type" in resp_ext.json()["detail"].lower()


def test_delete_document_write_protection_with_api_key(tmp_settings, fake_embedder, store, monkeypatch):
    from dataclasses import replace
    protected_settings = replace(tmp_settings, api_key="secure-delete-key")

    indexer = Indexer(protected_settings, fake_embedder, store)
    indexer.build(rebuild=True)
    orch = Orchestrator(protected_settings, Retriever(store, fake_embedder), ToolRegistry())

    api_module = sys.modules["agentic_rag.api.app"]
    monkeypatch.setattr(api_module, "load_stack", lambda: (orch, indexer, store))
    monkeypatch.setattr(api_module, "get_settings", lambda: protected_settings)

    client = TestClient(create_app())

    # Create dummy doc to delete
    dummy = tmp_settings.documents_dir / "protected_delete.txt"
    dummy.write_text("Secret content to delete.", encoding="utf-8")
    indexer.build(rebuild=True)

    # Missing API key -> 401
    r_no_key = client.delete("/documents/protected_delete.txt")
    assert r_no_key.status_code == 401
    assert dummy.exists()

    # Wrong API key -> 401
    r_bad_key = client.delete("/documents/protected_delete.txt", headers={"x-api-key": "wrong"})
    assert r_bad_key.status_code == 401
    assert dummy.exists()

    # Valid API key -> 200
    r_ok = client.delete("/documents/protected_delete.txt", headers={"x-api-key": "secure-delete-key"})
    assert r_ok.status_code == 200
    assert not dummy.exists()


