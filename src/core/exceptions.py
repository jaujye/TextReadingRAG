"""Custom exceptions for the TextReadingRAG application."""

from typing import Any, Dict, Optional


class TextReadingRAGException(Exception):
    """Base exception for TextReadingRAG application."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)


class ConfigurationError(TextReadingRAGException):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str, missing_config: Optional[str] = None):
        super().__init__(
            message=message,
            details={"missing_config": missing_config} if missing_config else {},
            error_code="CONFIG_ERROR",
        )


class FileProcessingError(TextReadingRAGException):
    """Exception raised during file processing operations."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"file_path": file_path, "file_type": file_type},
            error_code="FILE_PROCESSING_ERROR",
        )


class DocumentIngestionError(TextReadingRAGException):
    """Exception raised during document ingestion."""

    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"document_id": document_id, "stage": stage},
            error_code="DOCUMENT_INGESTION_ERROR",
        )


class VectorStoreError(TextReadingRAGException):
    """Exception raised for vector store operations."""

    def __init__(
        self,
        message: str,
        collection_name: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"collection_name": collection_name, "operation": operation},
            error_code="VECTOR_STORE_ERROR",
        )


class RetrievalError(TextReadingRAGException):
    """Exception raised during retrieval operations."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        retrieval_type: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"query": query, "retrieval_type": retrieval_type},
            error_code="RETRIEVAL_ERROR",
        )


class RerankingError(TextReadingRAGException):
    """Exception raised during reranking operations."""

    def __init__(
        self,
        message: str,
        reranker_type: Optional[str] = None,
        num_documents: Optional[int] = None,
    ):
        super().__init__(
            message=message,
            details={"reranker_type": reranker_type, "num_documents": num_documents},
            error_code="RERANKING_ERROR",
        )


class QueryExpansionError(TextReadingRAGException):
    """Exception raised during query expansion."""

    def __init__(
        self,
        message: str,
        original_query: Optional[str] = None,
        expansion_method: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={
                "original_query": original_query,
                "expansion_method": expansion_method,
            },
            error_code="QUERY_EXPANSION_ERROR",
        )


class LLMError(TextReadingRAGException):
    """Exception raised for LLM-related operations."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        api_error: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"model_name": model_name, "api_error": api_error},
            error_code="LLM_ERROR",
        )


class EmbeddingError(TextReadingRAGException):
    """Exception raised for embedding-related operations."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        text_length: Optional[int] = None,
    ):
        super().__init__(
            message=message,
            details={"model_name": model_name, "text_length": text_length},
            error_code="EMBEDDING_ERROR",
        )


class ValidationError(TextReadingRAGException):
    """Exception raised for input validation errors."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
    ):
        super().__init__(
            message=message,
            details={"field_name": field_name, "field_value": field_value},
            error_code="VALIDATION_ERROR",
        )


class AuthenticationError(TextReadingRAGException):
    """Exception raised for authentication errors."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
        )


class AuthorizationError(TextReadingRAGException):
    """Exception raised for authorization errors."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
        )


class RateLimitError(TextReadingRAGException):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"retry_after": retry_after, "limit_type": limit_type},
            error_code="RATE_LIMIT_ERROR",
        )


class ResourceNotFoundError(TextReadingRAGException):
    """Exception raised when a requested resource is not found."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"resource_type": resource_type, "resource_id": resource_id},
            error_code="RESOURCE_NOT_FOUND",
        )


class DatabaseError(TextReadingRAGException):
    """Exception raised for database-related errors."""

    def __init__(
        self,
        message: str,
        database_type: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"database_type": database_type, "operation": operation},
            error_code="DATABASE_ERROR",
        )


class CacheError(TextReadingRAGException):
    """Exception raised for cache-related errors."""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            details={"cache_key": cache_key, "operation": operation},
            error_code="CACHE_ERROR",
        )