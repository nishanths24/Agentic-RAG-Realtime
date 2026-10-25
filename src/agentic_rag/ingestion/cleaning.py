from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Collapse all whitespace to single spaces (legacy helper)."""
    return " ".join(text.split())


def normalize_document_text(text: str) -> str:
    """Normalize whitespace while keeping paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    joined = "\n".join(lines)
    joined = _MULTI_NL.sub("\n\n", joined)
    return joined.strip()
