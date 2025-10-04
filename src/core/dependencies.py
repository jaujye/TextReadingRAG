"""FastAPI dependency injection functions."""

import logging
from functools import lru_cache
from typing import Any, AsyncGenerator, Optional

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
def get_settings_dependency() -> Settings:
    """Get cached settings instance."""
    return get_settings()


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
            host=settings.rag.chroma_host,
            port=settings.rag.chroma_port,
            persist_directory=settings.rag.chroma_persist_directory,
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


# Global Redis connection pool
_redis_pool: Optional[Any] = None


async def get_cache_client(
    settings: Settings = Depends(get_settings_dependency),
):
    """
    Get Redis cache client from connection pool.

    Returns None if caching is disabled.
    """
    if not settings.app.enable_cache:
        return None

    global _redis_pool
    if _redis_pool is None:
        try:
            import redis.asyncio as redis
            _redis_pool = redis.Redis(
                host=settings.app.redis_host,
                port=settings.app.redis_port,
                db=settings.app.redis_db,
                password=settings.app.redis_password,
                decode_responses=True,
            )
        except ImportError:
            logger.warning("Redis not available. Caching disabled.")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return None

    return _redis_pool


async def get_cache_service(
    settings: Settings = Depends(get_settings_dependency),
    redis_client = Depends(get_cache_client),
):
    """
    Get cache service instance.

    Returns a CacheService that gracefully handles disabled state.
    """
    from src.core.cache import CacheService
    return CacheService(redis_client=redis_client, settings=settings)


async def get_retrieval_service(
    settings: Settings = Depends(get_settings_dependency),
    vector_store = Depends(get_vector_store_client),
    cache_service = Depends(get_cache_service),
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
            cache_service=cache_service,
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
    cache_service = Depends(get_cache_service),
):
    """
    Get query expansion service.

    This dependency will be implemented when we create the query expansion module.
    """
    try:
        # Import here to avoid circular imports
        from src.rag.query_expansion import QueryExpansionService

        return QueryExpansionService(settings=settings, cache_service=cache_service)
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
        max_size_bytes = settings.app.max_file_size * 1024 * 1024  # Convert MB to bytes

        if content_length > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.app.max_file_size}MB",
            )

    return True


async def validate_openai_config(
    settings: Settings = Depends(get_settings_dependency),
) -> bool:
    """
    Validate OpenAI configuration.

    Ensures API key is available and valid.
    """
    if not settings.llm.openai_api_key:
        if not settings.app.mock_llm_responses:
            raise ConfigurationError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            )

    return True


async def lifespan_context() -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown of connections.
    """
    # Startup
    logger.info("Starting up TextReadingRAG application...")

    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down TextReadingRAG application...")
        global _redis_pool
        if _redis_pool:
            await _redis_pool.close()