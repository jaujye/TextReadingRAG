"""Custom exceptions for the TextReadingRAG application."""

from typing import Any, Dict, Optional


class RAGException(Exception):
    """Base exception for all RAG application errors."""

    def __init__(
        self,
        message: str,
        error_type: str,
        **details: Any,
    ):
        """
        Initialize exception.

        Args:
            message: Error message
            error_type: Type of error (e.g., 'CONFIG_ERROR', 'FILE_PROCESSING_ERROR')
            **details: Additional context as keyword arguments
        """
        self.message = message
        self.error_type = error_type
        self.details = details
        self.error_code = error_type  # For backward compatibility
        super().__init__(self.message)


# Legacy aliases for backward compatibility
TextReadingRAGException = RAGException
ConfigurationError = lambda msg, **kw: RAGException(msg, "CONFIG_ERROR", **kw)
FileProcessingError = lambda msg, **kw: RAGException(msg, "FILE_PROCESSING_ERROR", **kw)
DocumentIngestionError = lambda msg, **kw: RAGException(msg, "DOCUMENT_INGESTION_ERROR", **kw)
VectorStoreError = lambda msg, **kw: RAGException(msg, "VECTOR_STORE_ERROR", **kw)
RetrievalError = lambda msg, **kw: RAGException(msg, "RETRIEVAL_ERROR", **kw)
RerankingError = lambda msg, **kw: RAGException(msg, "RERANKING_ERROR", **kw)
QueryExpansionError = lambda msg, **kw: RAGException(msg, "QUERY_EXPANSION_ERROR", **kw)
LLMError = lambda msg, **kw: RAGException(msg, "LLM_ERROR", **kw)
EmbeddingError = lambda msg, **kw: RAGException(msg, "EMBEDDING_ERROR", **kw)
ValidationError = lambda msg, **kw: RAGException(msg, "VALIDATION_ERROR", **kw)
AuthenticationError = lambda msg="Authentication failed", **kw: RAGException(msg, "AUTHENTICATION_ERROR", **kw)
AuthorizationError = lambda msg="Access denied", **kw: RAGException(msg, "AUTHORIZATION_ERROR", **kw)
RateLimitError = lambda msg, **kw: RAGException(msg, "RATE_LIMIT_ERROR", **kw)
ResourceNotFoundError = lambda msg, **kw: RAGException(msg, "RESOURCE_NOT_FOUND", **kw)
DatabaseError = lambda msg, **kw: RAGException(msg, "DATABASE_ERROR", **kw)
CacheError = lambda msg, **kw: RAGException(msg, "CACHE_ERROR", **kw)