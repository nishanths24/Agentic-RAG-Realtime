from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agentic_rag.errors import IndexMismatchError, VectorStoreError
from agentic_rag.models import Chunk, RetrievedPassage

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NumpyVectorStore:
    def __init__(
        self,
        directory: Path,
        *,
        model_name: str,
        dimension: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.directory = Path(directory)
        self.model_name = model_name
        self.dimension = dimension
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._ids: list[str] = []
        self._chunks: dict[str, dict[str, Any]] = {}
        self._matrix = np.zeros((0, dimension), dtype=np.float32)
        self._doc_hashes: dict[str, str] = {}

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def meta_path(self) -> Path:
        return self.directory / "chunks.json"

    @property
    def vectors_path(self) -> Path:
        return self.directory / "vectors.npy"

    def count(self) -> int:
        return len(self._ids)

    def document_hashes(self) -> dict[str, str]:
        return dict(self._doc_hashes)

    def clear(self) -> None:
        self._ids = []
        self._chunks = {}
        self._matrix = np.zeros((0, self.dimension), dtype=np.float32)
        self._doc_hashes = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        id_to_row = {chunk_id: i for i, chunk_id in enumerate(self._ids)}
        rows = self._matrix.tolist()
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dimension:
                raise VectorStoreError(
                    f"Vector dimension {len(vector)} != store dimension {self.dimension}"
                )
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_path": chunk.source_path,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            if chunk.chunk_id in id_to_row:
                rows[id_to_row[chunk.chunk_id]] = list(vector)
                self._chunks[chunk.chunk_id] = payload
            else:
                id_to_row[chunk.chunk_id] = len(self._ids)
                self._ids.append(chunk.chunk_id)
                self._chunks[chunk.chunk_id] = payload
                rows.append(list(vector))
        self._matrix = np.asarray(rows, dtype=np.float32)

    def delete_by_source(self, source_path: str) -> int:
        keep_ids = [
            cid
            for cid in self._ids
            if self._chunks[cid]["source_path"] != source_path
        ]
        removed = len(self._ids) - len(keep_ids)
        if removed == 0:
            self._doc_hashes.pop(source_path, None)
            return 0
        keep_index = [self._ids.index(cid) for cid in keep_ids]
        self._matrix = self._matrix[keep_index] if keep_index else np.zeros((0, self.dimension), np.float32)
        self._ids = keep_ids
        self._chunks = {cid: self._chunks[cid] for cid in keep_ids}
        self._doc_hashes.pop(source_path, None)
        return removed

    def set_document_hash(self, source_path: str, content_hash: str) -> None:
        self._doc_hashes[source_path] = content_hash

    def search(self, query_vector: Sequence[float], top_k: int) -> list[RetrievedPassage]:
        if top_k <= 0 or self.count() == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != self.dimension:
            raise VectorStoreError("Query dimension does not match the index")
        qn = np.linalg.norm(query) or 1.0
        mn = np.linalg.norm(self._matrix, axis=1)
        mn[mn == 0] = 1.0
        scores = (self._matrix @ query) / (mn * qn)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results: list[RetrievedPassage] = []
        for rank, row in enumerate(top, start=1):
            meta = self._chunks[self._ids[int(row)]]
            chunk = Chunk(
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                source_path=meta["source_path"],
                chunk_index=meta["chunk_index"],
                text=meta["text"],
                char_start=meta["char_start"],
                char_end=meta["char_end"],
            )
            results.append(RetrievedPassage(chunk=chunk, score=float(scores[int(row)]), rank=rank))
        return results

    def persist(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp_vectors = self.directory / "vectors.npy.tmp"
        np.save(tmp_vectors, self._matrix)
        tmp_vectors.replace(self.vectors_path)
        meta = {
            "ids": self._ids,
            "chunks": self._chunks,
            "doc_hashes": self._doc_hashes,
        }
        tmp_meta = self.directory / "chunks.json.tmp"
        tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp_meta.replace(self.meta_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "count": self.count(),
            "updated_at": _utc_now(),
        }
        tmp_manifest = self.directory / "manifest.json.tmp"
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp_manifest.replace(self.manifest_path)

    def load(self) -> None:
        if not self.manifest_path.exists():
            self.clear()
            return
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_name") != self.model_name or int(manifest.get("dimension", -1)) != self.dimension:
            raise IndexMismatchError(
                "Index was built with a different embedding model/dimension. "
                "Run `python main.py index --rebuild`."
            )
        if not self.vectors_path.exists() or not self.meta_path.exists():
            raise VectorStoreError("Index files are incomplete; rebuild the index.")
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        matrix = np.load(self.vectors_path)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise VectorStoreError("Corrupt vectors file")
        self._ids = list(meta.get("ids", []))
        self._chunks = dict(meta.get("chunks", {}))
        self._doc_hashes = dict(meta.get("doc_hashes", {}))
        self._matrix = matrix.astype(np.float32)
        if len(self._ids) != self._matrix.shape[0]:
            raise VectorStoreError("Manifest/vector count mismatch; rebuild the index.")
