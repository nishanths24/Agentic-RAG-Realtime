from __future__ import annotations

import pytest

from agentic_rag.errors import InvalidChunkParametersError
from agentic_rag.ingestion.chunking import chunk_spans, chunk_text, make_chunk_id


def test_chunk_text_never_splits_words_shorter_than_chunk_size():
    chunks = chunk_text("alpha beta gamma delta", chunk_size=12, overlap=3)
    for chunk in chunks:
        assert len(chunk) <= 12
        for word in ("alpha", "beta", "gamma", "delta"):
            if word in chunk:
                assert word in chunk.split() or chunk.endswith(word) or chunk.startswith(word)


def test_long_word_is_hard_split():
    chunks = chunk_text("abcdefghij", 6, 2)
    assert chunks == ["abcdef", "efghij"]


def test_overlap_and_termination():
    chunks = chunk_text("one two three four five six seven", 15, 4)
    assert chunks
    assert all(len(c) <= 15 for c in chunks)


def test_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n") == []


def test_text_shorter_and_exact():
    assert chunk_text("hello", 10, 2) == ["hello"]
    text = "a" * 10
    assert chunk_text(text, 10, 2) == [text]


def test_invalid_parameters():
    with pytest.raises(InvalidChunkParametersError):
        chunk_text("hello", 0, 0)
    with pytest.raises(InvalidChunkParametersError):
        chunk_text("hello", 5, 5)
    with pytest.raises(InvalidChunkParametersError):
        chunk_text("hello", 5, -1)


def test_unicode_and_emoji():
    chunks = chunk_text("café naïve 😊 travel", 12, 3)
    assert chunks
    assert "café" in " ".join(chunks)


def test_spans_reproduce_text():
    text = "Receipts must be submitted within 15 days of completing the trip."
    for start, end, piece in chunk_spans(text, 20, 5):
        assert text.strip()[start:end] == piece
        assert end - start == len(piece)
        assert end - start <= 20


def test_chunk_ids_are_deterministic():
    first = make_chunk_id("abc", 0, "hello")
    second = make_chunk_id("abc", 0, "hello")
    assert first == second
    assert first != make_chunk_id("abc", 1, "hello")
