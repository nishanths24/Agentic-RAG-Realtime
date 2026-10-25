"""Typed errors used across the package."""


class AgenticRAGError(Exception):
    """Base error for this application."""


class DocumentNotFoundError(AgenticRAGError, FileNotFoundError):
    pass


class UnsupportedDocumentError(AgenticRAGError):
    pass


class EmptyDocumentError(AgenticRAGError):
    pass


class PathEscapeError(AgenticRAGError):
    pass


class InvalidChunkParametersError(AgenticRAGError, ValueError):
    pass


class EmbeddingError(AgenticRAGError):
    pass


class EmbeddingDimensionError(EmbeddingError):
    pass


class VectorStoreError(AgenticRAGError):
    pass


class IndexMismatchError(VectorStoreError):
    pass


class RetrievalError(AgenticRAGError):
    pass


class LLMError(AgenticRAGError):
    pass


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


class ToolError(AgenticRAGError):
    pass


class ToolValidationError(ToolError, ValueError):
    pass


class ToolUnavailableError(ToolError):
    pass


class ToolTimeoutError(ToolError):
    pass


class ConfigError(AgenticRAGError):
    pass
