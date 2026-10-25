from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_REDACT = re.compile(r"(sk-[A-Za-z0-9]{8,}|api[_-]?key\s*[:=]\s*\S+)", re.I)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return _REDACT.sub("[REDACTED]", message)


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    root = logging.getLogger()
    if getattr(root, "_agentic_rag_configured", False):
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    formatter = RedactingFormatter(
        "%(asctime)s %(levelname)s %(name)s [trace=%(trace_id)s] %(message)s"
    )
    old_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        return record

    logging.setLogRecordFactory(factory)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "agentic_rag.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root._agentic_rag_configured = True  # type: ignore[attr-defined]
