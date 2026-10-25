from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from agentic_rag.errors import ConfigError

_SECRET_KEYS = ("LLM_API_KEY", "SEARCH_API_KEY", "API_KEY")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    documents_dir: Path
    index_dir: Path
    log_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_score: float
    log_level: str
    request_timeout_s: float
    max_agent_steps: int
    max_file_bytes: int
    max_question_chars: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    llm_provider: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str
    search_api_key: str
    api_key: str
    host: str
    port: int

    def __repr__(self) -> str:
        return (
            "Settings("
            f"documents_dir={self.documents_dir!s}, "
            f"index_dir={self.index_dir!s}, "
            f"chunk_size={self.chunk_size}, "
            f"embedding_provider={self.embedding_provider!r}, "
            f"llm_provider={self.llm_provider!r}, "
            "llm_api_key=***, search_api_key=***, api_key=***"
            ")"
        )


def _as_bool_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path


def load_settings(env_file: Path | None = None) -> Settings:
    root = _project_root()
    dotenv_path = env_file or (root / ".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)

    def getenv(name: str, default: str) -> str:
        value = os.getenv(name, default)
        return default if value is None or value.strip() == "" else value

    try:
        chunk_size = int(getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(getenv("CHUNK_OVERLAP", "150"))
        top_k = int(getenv("TOP_K", "4"))
        min_score = float(getenv("MIN_SCORE", "0.08"))
        timeout = float(getenv("REQUEST_TIMEOUT_S", "15"))
        max_steps = int(getenv("MAX_AGENT_STEPS", "4"))
        max_file = int(getenv("MAX_FILE_BYTES", "1048576"))
        max_q = int(getenv("MAX_QUESTION_CHARS", "2000"))
        dim = int(getenv("EMBEDDING_DIMENSION", "384"))
        port = int(getenv("PORT", "8000"))
    except ValueError as exc:
        raise ConfigError(f"Invalid numeric setting: {exc}") from exc

    if chunk_size <= 0 or not (0 <= chunk_overlap < chunk_size):
        raise ConfigError("CHUNK_SIZE must be > 0 and CHUNK_OVERLAP must satisfy 0 <= overlap < size")

    return Settings(
        project_root=root,
        documents_dir=_as_bool_path(getenv("DOCUMENTS_DIR", "data/documents"), root),
        index_dir=_as_bool_path(getenv("INDEX_DIR", "data/index"), root),
        log_dir=_as_bool_path(getenv("LOG_DIR", "logs"), root),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        min_score=min_score,
        log_level=getenv("LOG_LEVEL", "INFO").upper(),
        request_timeout_s=timeout,
        max_agent_steps=max_steps,
        max_file_bytes=max_file,
        max_question_chars=max_q,
        embedding_provider=getenv("EMBEDDING_PROVIDER", "hashing").lower(),
        embedding_model=getenv("EMBEDDING_MODEL", "hashing-bow-v1"),
        embedding_dimension=dim,
        llm_provider=getenv("LLM_PROVIDER", "none").lower(),
        llm_model=getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        search_api_key=os.getenv("SEARCH_API_KEY", ""),
        api_key=os.getenv("API_KEY", ""),
        host=getenv("HOST", "127.0.0.1"),
        port=port,
    )


def secret_env_names() -> tuple[str, ...]:
    return _SECRET_KEYS
