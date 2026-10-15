from pathlib import Path


def clean_text(text: str) -> str:
    """Remove unnecessary whitespace from text."""
    return " ".join(text.split())


def read_text_file(file_path: Path) -> str:
    """Read a text file using UTF-8."""
    return file_path.read_text(encoding="utf-8")


def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks
