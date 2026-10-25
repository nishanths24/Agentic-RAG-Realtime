from __future__ import annotations

from agentic_rag.errors import InvalidChunkParametersError
from agentic_rag.models import Chunk


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise InvalidChunkParametersError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise InvalidChunkParametersError("overlap must be >= 0 and < chunk_size")


def _is_ws(char: str) -> bool:
    return char.isspace()


def chunk_spans(text: str, chunk_size: int = 100, overlap: int = 20) -> list[tuple[int, int, str]]:
    """Word-aware overlapping windows over already-normalized text.

    Offsets are relative to the input string after stripping leading/trailing
    whitespace (the same string returned as chunk text slices).
    """
    _validate(chunk_size, overlap)
    text = text.strip()
    if not text:
        return []

    n = len(text)
    spans: list[tuple[int, int, str]] = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        if end < n and not _is_ws(text[end - 1]) and not _is_ws(text[end]):
            cut = None
            for i in range(end - 1, start, -1):
                if _is_ws(text[i]):
                    cut = i
                    break
            if cut is not None:
                while cut > start and _is_ws(text[cut]):
                    cut -= 1
                if cut > start:
                    end = cut + 1
        if end <= start:
            end = min(start + chunk_size, n)

        spans.append((start, end, text[start:end]))
        if end >= n:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1

        if 0 < next_start < n and not _is_ws(text[next_start]) and not _is_ws(text[next_start - 1]):
            i = next_start
            while i < end and i < n and not _is_ws(text[i]):
                i += 1
            if i < end and i < n and _is_ws(text[i]):
                while i < n and _is_ws(text[i]):
                    i += 1
                if start < i < n:
                    next_start = i

        if next_start <= start:
            next_start = start + 1
        start = next_start

    return spans


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    return [span[2] for span in chunk_spans(text, chunk_size, overlap)]


def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    import hashlib

    short = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}:{chunk_index}:{short}"


def chunks_from_document(
    *,
    doc_id: str,
    source_path: str,
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    result: list[Chunk] = []
    for index, (start, end, piece) in enumerate(chunk_spans(text, chunk_size, overlap)):
        result.append(
            Chunk(
                chunk_id=make_chunk_id(doc_id, index, piece),
                doc_id=doc_id,
                source_path=source_path,
                chunk_index=index,
                text=piece,
                char_start=start,
                char_end=end,
            )
        )
    return result
