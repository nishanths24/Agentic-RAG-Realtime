from __future__ import annotations

from functools import lru_cache

from agentic_rag.agent.orchestrator import Orchestrator
from agentic_rag.answering.generator import AnswerGenerator
from agentic_rag.config import Settings, load_settings
from agentic_rag.indexing import default_stack
from agentic_rag.llm.openai_compat import OpenAICompatClient
from agentic_rag.logging_setup import setup_logging
from agentic_rag.retrieval.retriever import Retriever
from agentic_rag.tools.registry import ToolRegistry, default_tools


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    return settings


@lru_cache(maxsize=1)
def get_stack():
    return build_orchestrator(get_settings())


def build_orchestrator(settings: Settings | None = None) -> tuple[Orchestrator, object, object]:
    settings = settings or get_settings()
    embedder, store, indexer = default_stack(settings)
    try:
        store.load()
    except Exception:
        indexer.build(rebuild=True)
        store.load()
    if store.count() == 0:
        indexer.build(rebuild=True)
        store.load()
    else:
        indexer.build(rebuild=False)
    retriever = Retriever(store, embedder)
    generator = AnswerGenerator()
    if settings.llm_provider not in {"", "none"} and settings.llm_api_key:
        generator = AnswerGenerator(
            OpenAICompatClient(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
            )
        )
    orchestrator = Orchestrator(settings, retriever, ToolRegistry(default_tools(settings)), generator=generator)
    return orchestrator, indexer, store
