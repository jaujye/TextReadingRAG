"""
TextReadingRAG - Advanced PDF RAG System

FastAPI application for processing and querying PDF documents using
hybrid retrieval strategies, query expansion, and reranking.
"""

import logging
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.config import get_settings
from src.core.dependencies import lifespan_context
from src.core.exceptions import (
    TextReadingRAGException,
    ConfigurationError,
    FileProcessingError,
    DocumentIngestionError,
    VectorStoreError,
    RetrievalError,
    RerankingError,
    QueryExpansionError,
    LLMError,
    EmbeddingError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ResourceNotFoundError,
    DatabaseError,
    CacheError,
)

# Initialize settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.app.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info(f"Starting {settings.app.app_name} v{settings.app.app_version}")
    logger.info("Initializing application components...")

    try:
        # Validate critical configuration
        if not settings.openai.openai_api_key and not settings.development.mock_llm_responses:
            raise ConfigurationError("OpenAI API key is required")

        # Initialize any global resources here
        # (Vector store connections, cache, etc. will be handled by dependencies)

        logger.info("Application startup completed successfully")
        yield

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        # Cleanup will be handled by lifespan_context
        logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title=settings.app.app_name,
    version=settings.app.app_version,
    description="Advanced PDF RAG System with hybrid retrieval, query expansion, and reranking",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=settings.security.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(TextReadingRAGException)
async def text_reading_rag_exception_handler(
    request: Request, exc: TextReadingRAGException
) -> JSONResponse:
    """Handle custom TextReadingRAG exceptions."""
    logger.error(f"TextReadingRAG exception: {exc.message}", extra=exc.details)

    status_code_map = {
        "CONFIG_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "FILE_PROCESSING_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "DOCUMENT_INGESTION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "VECTOR_STORE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "RETRIEVAL_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "RERANKING_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "QUERY_EXPANSION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "LLM_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "EMBEDDING_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "AUTHENTICATION_ERROR": status.HTTP_401_UNAUTHORIZED,
        "AUTHORIZATION_ERROR": status.HTTP_403_FORBIDDEN,
        "RATE_LIMIT_ERROR": status.HTTP_429_TOO_MANY_REQUESTS,
        "RESOURCE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "DATABASE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "CACHE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
    }

    status_code = status_code_map.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": exc.message,
                "error_code": exc.error_code,
                "details": exc.details,
                "type": type(exc).__name__,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation error",
                "error_code": "VALIDATION_ERROR",
                "details": {"validation_errors": exc.errors()},
                "type": "RequestValidationError",
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "error_code": f"HTTP_{exc.status_code}",
                "details": {},
                "type": "HTTPException",
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected exception: {exc}", exc_info=True)

    if settings.app.debug:
        error_details = {
            "traceback": traceback.format_exc(),
            "exception_type": type(exc).__name__,
        }
    else:
        error_details = {}

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal server error",
                "error_code": "INTERNAL_SERVER_ERROR",
                "details": error_details,
                "type": "InternalServerError",
            }
        },
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app.app_name,
        "version": settings.app.app_version,
        "debug": settings.app.debug,
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with component status."""
    health_status = {
        "status": "healthy",
        "service": settings.app.app_name,
        "version": settings.app.app_version,
        "timestamp": None,  # Will be set by the frontend
        "components": {},
    }

    # Check OpenAI configuration
    health_status["components"]["openai"] = {
        "status": "healthy" if settings.openai.openai_api_key or settings.development.mock_llm_responses else "unhealthy",
        "configured": bool(settings.openai.openai_api_key),
        "mock_mode": settings.development.mock_llm_responses,
    }

    # Check ChromaDB configuration
    health_status["components"]["chromadb"] = {
        "status": "configured",
        "host": settings.chroma.chroma_host,
        "port": settings.chroma.chroma_port,
        "persist_directory": settings.chroma.chroma_persist_directory,
    }

    # Check cache configuration
    health_status["components"]["cache"] = {
        "status": "enabled" if settings.cache.enable_cache else "disabled",
        "enabled": settings.cache.enable_cache,
        "redis_host": settings.cache.redis_host if settings.cache.enable_cache else None,
    }

    # Check file upload configuration
    health_status["components"]["file_upload"] = {
        "status": "configured",
        "max_file_size_mb": settings.file_upload.max_file_size,
        "allowed_extensions": settings.file_upload.allowed_extensions,
        "upload_dir": settings.file_upload.upload_dir,
    }

    # Overall status based on critical components
    critical_unhealthy = any(
        component.get("status") == "unhealthy"
        for component in health_status["components"].values()
    )
    if critical_unhealthy:
        health_status["status"] = "unhealthy"

    return health_status


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with basic service information."""
    return {
        "service": settings.app.app_name,
        "version": settings.app.app_version,
        "description": "Advanced PDF RAG System with hybrid retrieval",
        "docs": "/docs",
        "health": "/health",
    }


# Include API routers
# Note: These will be imported and included once the endpoint modules are created

# from src.api.endpoints import documents, query, health
# app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
# app.include_router(query.router, prefix="/api/query", tags=["Query"])
# app.include_router(health.router, prefix="/api/health", tags=["Health"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload and settings.app.debug,
        log_level=settings.app.log_level.lower(),
    )