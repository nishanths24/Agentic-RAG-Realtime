from __future__ import annotations

import logging
from pathlib import Path

from agentic_rag.config import Settings
from agentic_rag.embeddings.hashing import HashingEmbeddingProvider
from agentic_rag.ingestion.loaders import ingest_directory
from agentic_rag.models import Chunk, Document, IngestionReport
from agentic_rag.vectorstore.numpy_store import NumpyVectorStore

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, settings: Settings, embedder: HashingEmbeddingProvider, store: NumpyVectorStore) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def build(self, *, rebuild: bool = False) -> dict[str, int | str]:
        if rebuild:
            self.store.clear()
        else:
            self.store.load()

        chunks, documents, report = ingest_directory(
            self.settings.documents_dir,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            max_file_bytes=self.settings.max_file_bytes,
        )
        current_paths = {doc.source_path for doc in documents}
        existing = self.store.document_hashes()
        removed = 0
        for source in list(existing):
            if source not in current_paths:
                removed += self.store.delete_by_source(source)

        changed_docs = [
            doc
            for doc in documents
            if rebuild or existing.get(doc.source_path) != doc.content_hash
        ]
        if changed_docs:
            for doc in changed_docs:
                self.store.delete_by_source(doc.source_path)
            changed_chunks = [chunk for chunk in chunks if chunk.source_path in {d.source_path for d in changed_docs}]
            vectors = self.embedder.embed_documents([chunk.text for chunk in changed_chunks])
            self.store.upsert(changed_chunks, vectors)
            for doc in changed_docs:
                self.store.set_document_hash(doc.source_path, doc.content_hash)

        self.store.persist()
        logger.info(
            "Indexed %s documents (%s changed, %s removed, %s chunks)",
            len(documents),
            len(changed_docs),
            removed,
            self.store.count(),
        )
        return {
            "documents": len(documents),
            "changed": len(changed_docs),
            "removed": removed,
            "chunks": self.store.count(),
            "skipped": len(report.skipped),
            "failed": len(report.failed),
        }

    def remove_document(self, source_path: str) -> int:
        removed = self.store.delete_by_source(source_path)
        self.store.persist()
        logger.info(
            "Removed document %s (%s chunks removed, %s remaining)",
            source_path,
            removed,
            self.store.count(),
        )
        return removed


def default_stack(settings: Settings) -> tuple[HashingEmbeddingProvider, NumpyVectorStore, Indexer]:
    embedder = HashingEmbeddingProvider(
        dimension=settings.embedding_dimension,
        model_name=settings.embedding_model,
    )
    store = NumpyVectorStore(
        settings.index_dir,
        model_name=embedder.model_name,
        dimension=embedder.dimension,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    indexer = Indexer(settings, embedder, store)
    return embedder, store, indexer
