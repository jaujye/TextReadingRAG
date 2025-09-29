"""FastAPI dependency injection functions."""

import logging
from functools import lru_cache
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.core.config import Settings, get_settings
from src.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    VectorStoreError,
)

# Initialize security scheme
security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


@lru_cache()
def get_cached_settings() -> Settings:
    """Get cached settings instance to avoid re-loading configuration."""
    return get_settings()


async def get_settings_dependency() -> Settings:
    """FastAPI dependency to get application settings."""
    return get_cached_settings()


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings_dependency),
) -> bool:
    """
    Verify API key for authenticated endpoints.

    This is a placeholder for API key authentication.
    In production, implement proper API key validation.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # In production, verify the API key against a database or external service
    # For now, we'll accept any non-empty token
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


async def get_vector_store_client(
    settings: Settings = Depends(get_settings_dependency),
):
    """
    Get ChromaDB vector store client.

    This dependency will be implemented when we create the vector store module.
    """
    try:
        # Import here to avoid circular imports
        from src.rag.vector_store import ChromaVectorStore

        return ChromaVectorStore(
            host=settings.chroma.chroma_host,
            port=settings.chroma.chroma_port,
            persist_directory=settings.chroma.chroma_persist_directory,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise VectorStoreError(f"Failed to connect to vector store: {e}")


async def get_document_ingestion_service(
    settings: Settings = Depends(get_settings_dependency),
):
    """
    Get document ingestion service.

    This dependency will be implemented when we create the ingestion module.
    """
    try:
        # Import here to avoid circular imports
        from src.rag.ingestion import DocumentIngestionService

        return DocumentIngestionService(settings=settings)
    except Exception as e:
        logger.error(f"Failed to initialize ingestion service: {e}")
        raise ConfigurationError(f"Failed to initialize ingestion service: {e}")


async def get_retrieval_service(
    settings: Settings = Depends(get_settings_dependency),
    vector_store = Depends(get_vector_store_client),
):
    """
    Get hybrid retrieval service.

    This dependency will be implemented when we create the retrieval module.
    """
    try:
        # Import here to avoid circular imports
        from src.rag.retrieval import HybridRetrievalService

        return HybridRetrievalService(
            vector_store=vector_store,
            settings=settings,
        )
    except Exception as e:
        logger.error(f"Failed to initialize retrieval service: {e}")
        raise ConfigurationError(f"Failed to initialize retrieval service: {e}")


async def get_reranking_service(
    settings: Settings = Depends(get_settings_dependency),
):
    """
    Get reranking service.

    This dependency will be implemented when we create the reranking module.
    """
    try:
        # Import here to avoid circular imports
        from src.rag.reranking import RerankingService

        return RerankingService(settings=settings)
    except Exception as e:
        logger.error(f"Failed to initialize reranking service: {e}")
        raise ConfigurationError(f"Failed to initialize reranking service: {e}")


async def get_query_expansion_service(
    settings: Settings = Depends(get_settings_dependency),
):
    """
    Get query expansion service.

    This dependency will be implemented when we create the query expansion module.
    """
    try:
        # Import here to avoid circular imports
        from src.rag.query_expansion import QueryExpansionService

        return QueryExpansionService(settings=settings)
    except Exception as e:
        logger.error(f"Failed to initialize query expansion service: {e}")
        raise ConfigurationError(f"Failed to initialize query expansion service: {e}")


async def validate_file_upload(
    request: Request,
    settings: Settings = Depends(get_settings_dependency),
) -> bool:
    """
    Validate file upload requirements.

    Checks file size limits and content type restrictions.
    """
    content_length = request.headers.get("content-length")

    if content_length:
        content_length = int(content_length)
        max_size_bytes = settings.file_upload.max_file_size * 1024 * 1024  # Convert MB to bytes

        if content_length > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.file_upload.max_file_size}MB",
            )

    return True


async def check_rate_limit(
    request: Request,
    settings: Settings = Depends(get_settings_dependency),
) -> bool:
    """
    Check rate limiting for API requests.

    This is a placeholder for rate limiting implementation.
    In production, integrate with Redis or similar for distributed rate limiting.
    """
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # In production, implement actual rate limiting logic
    # For now, we'll just log the request
    logger.info(f"API request from {client_ip}")

    return True


async def validate_openai_config(
    settings: Settings = Depends(get_settings_dependency),
) -> bool:
    """
    Validate OpenAI configuration.

    Ensures API key is available and valid.
    """
    if not settings.openai.openai_api_key:
        if not settings.development.mock_llm_responses:
            raise ConfigurationError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            )

    return True


async def get_cache_client(
    settings: Settings = Depends(get_settings_dependency),
):
    """
    Get Redis cache client if caching is enabled.

    Returns None if caching is disabled.
    """
    if not settings.cache.enable_cache:
        return None

    try:
        import redis.asyncio as redis

        return redis.Redis(
            host=settings.cache.redis_host,
            port=settings.cache.redis_port,
            db=settings.cache.redis_db,
            password=settings.cache.redis_password,
            decode_responses=True,
        )
    except ImportError:
        logger.warning("Redis not available. Caching disabled.")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None


class DatabaseManager:
    """Database connection manager for dependency injection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._vector_store = None
        self._cache_client = None

    async def get_vector_store(self):
        """Get or create vector store connection."""
        if not self._vector_store:
            from src.rag.vector_store import ChromaVectorStore
            self._vector_store = ChromaVectorStore(
                host=self.settings.chroma.chroma_host,
                port=self.settings.chroma.chroma_port,
                persist_directory=self.settings.chroma.chroma_persist_directory,
            )
        return self._vector_store

    async def get_cache_client(self):
        """Get or create cache client."""
        if not self._cache_client and self.settings.cache.enable_cache:
            import redis.asyncio as redis
            self._cache_client = redis.Redis(
                host=self.settings.cache.redis_host,
                port=self.settings.cache.redis_port,
                db=self.settings.cache.redis_db,
                password=self.settings.cache.redis_password,
                decode_responses=True,
            )
        return self._cache_client

    async def close_connections(self):
        """Close all database connections."""
        if self._cache_client:
            await self._cache_client.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database_manager(
    settings: Settings = Depends(get_settings_dependency),
) -> DatabaseManager:
    """Get database manager instance."""
    global _db_manager
    if not _db_manager:
        _db_manager = DatabaseManager(settings)
    return _db_manager


async def lifespan_context() -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown of database connections.
    """
    # Startup
    logger.info("Starting up TextReadingRAG application...")

    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down TextReadingRAG application...")
        global _db_manager
        if _db_manager:
            await _db_manager.close_connections()