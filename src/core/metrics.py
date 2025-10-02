"""
Prometheus metrics for TextReadingRAG monitoring.

This module defines custom metrics for monitoring RAG system performance,
including retrieval, reranking, query expansion, and document processing.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any
import time
from functools import wraps


# ============================================================================
# HTTP Metrics (handled by prometheus-fastapi-instrumentator)
# ============================================================================

# ============================================================================
# RAG System Metrics
# ============================================================================

# Document Processing Metrics
document_processing_duration = Histogram(
    "rag_document_processing_duration_seconds",
    "Time spent processing documents",
    ["document_type", "use_llamaparse"],
    buckets=(1, 5, 10, 15, 30, 60, 120, 300)
)

document_processing_errors = Counter(
    "rag_document_processing_errors_total",
    "Total number of document processing errors",
    ["error_type", "document_type"]
)

documents_indexed = Counter(
    "rag_documents_indexed_total",
    "Total number of documents indexed",
    ["collection_name"]
)

documents_pending = Gauge(
    "rag_documents_pending",
    "Number of documents pending processing"
)

documents_processing = Gauge(
    "rag_documents_processing",
    "Number of documents currently being processed"
)

documents_completed = Counter(
    "rag_documents_completed",
    "Total number of documents completed processing"
)

# Retrieval Metrics
retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "Time spent on document retrieval",
    ["retrieval_strategy", "collection_name"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0)
)

retrieval_errors = Counter(
    "rag_retrieval_errors_total",
    "Total number of retrieval errors",
    ["error_type", "retrieval_strategy"]
)

retrieval_quality_score = Histogram(
    "rag_retrieval_quality_score",
    "Quality score of retrieved documents",
    ["retrieval_strategy"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

documents_retrieved = Histogram(
    "rag_documents_retrieved",
    "Number of documents retrieved per query",
    ["retrieval_strategy"],
    buckets=(1, 3, 5, 10, 15, 20, 30, 50)
)

# Query Expansion Metrics
query_expansion_duration = Histogram(
    "rag_query_expansion_duration_seconds",
    "Time spent on query expansion",
    ["expansion_method"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
)

query_expansion_errors = Counter(
    "rag_query_expansion_errors_total",
    "Total number of query expansion errors",
    ["error_type", "expansion_method"]
)

expanded_queries = Histogram(
    "rag_expanded_queries_count",
    "Number of expanded queries generated",
    ["expansion_method"],
    buckets=(1, 2, 3, 5, 7, 10)
)

# Reranking Metrics
reranking_duration = Histogram(
    "rag_reranking_duration_seconds",
    "Time spent on document reranking",
    ["reranking_model"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
)

reranking_errors = Counter(
    "rag_reranking_errors_total",
    "Total number of reranking errors",
    ["error_type", "reranking_model"]
)

reranking_score_improvement = Histogram(
    "rag_reranking_score_improvement",
    "Score improvement after reranking",
    ["reranking_model"],
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)
)

documents_reranked = Histogram(
    "rag_documents_reranked",
    "Number of documents reranked",
    ["reranking_model"],
    buckets=(1, 3, 5, 10, 15, 20, 30)
)

# Response Generation Metrics
generation_duration = Histogram(
    "rag_generation_duration_seconds",
    "Time spent on response generation",
    ["model_name", "streaming"],
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0)
)

generation_errors = Counter(
    "rag_generation_errors_total",
    "Total number of generation errors",
    ["error_type", "model_name"]
)

tokens_generated = Histogram(
    "rag_tokens_generated",
    "Number of tokens generated in response",
    ["model_name"],
    buckets=(50, 100, 250, 500, 750, 1000, 1500, 2000, 3000)
)

# Query Metrics
queries_total = Counter(
    "rag_queries_total",
    "Total number of queries processed",
    ["query_type", "collection_name"]
)

query_processing_duration = Histogram(
    "rag_query_processing_duration_seconds",
    "Total time spent processing a query (end-to-end)",
    ["query_type"],
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0, 60.0)
)

# Cache Metrics
cache_hits = Counter(
    "rag_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"]
)

cache_misses = Counter(
    "rag_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"]
)

cache_errors = Counter(
    "rag_cache_errors_total",
    "Total number of cache errors",
    ["error_type"]
)

# Vector Store Metrics
vector_store_operations = Counter(
    "rag_vector_store_operations_total",
    "Total number of vector store operations",
    ["operation_type", "collection_name"]
)

vector_store_errors = Counter(
    "rag_vector_store_errors_total",
    "Total number of vector store errors",
    ["error_type", "operation_type"]
)

# Embedding Metrics
embedding_duration = Histogram(
    "rag_embedding_duration_seconds",
    "Time spent generating embeddings",
    ["model_name"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
)

embedding_errors = Counter(
    "rag_embedding_errors_total",
    "Total number of embedding errors",
    ["error_type", "model_name"]
)

embeddings_generated = Histogram(
    "rag_embeddings_generated",
    "Number of embeddings generated",
    ["model_name"],
    buckets=(1, 5, 10, 20, 50, 100, 200, 500)
)

# System Info
system_info = Info(
    "rag_system_info",
    "System information"
)


# ============================================================================
# Metric Decorators
# ============================================================================

def track_retrieval_time(retrieval_strategy: str = "hybrid", collection_name: str = "default"):
    """Decorator to track retrieval duration."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                retrieval_duration.labels(
                    retrieval_strategy=retrieval_strategy,
                    collection_name=collection_name
                ).observe(duration)
                return result
            except Exception as e:
                retrieval_errors.labels(
                    error_type=type(e).__name__,
                    retrieval_strategy=retrieval_strategy
                ).inc()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                retrieval_duration.labels(
                    retrieval_strategy=retrieval_strategy,
                    collection_name=collection_name
                ).observe(duration)
                return result
            except Exception as e:
                retrieval_errors.labels(
                    error_type=type(e).__name__,
                    retrieval_strategy=retrieval_strategy
                ).inc()
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_reranking_time(reranking_model: str = "default"):
    """Decorator to track reranking duration."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                reranking_duration.labels(reranking_model=reranking_model).observe(duration)
                return result
            except Exception as e:
                reranking_errors.labels(
                    error_type=type(e).__name__,
                    reranking_model=reranking_model
                ).inc()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                reranking_duration.labels(reranking_model=reranking_model).observe(duration)
                return result
            except Exception as e:
                reranking_errors.labels(
                    error_type=type(e).__name__,
                    reranking_model=reranking_model
                ).inc()
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_query_expansion_time(expansion_method: str = "llm"):
    """Decorator to track query expansion duration."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                query_expansion_duration.labels(expansion_method=expansion_method).observe(duration)
                return result
            except Exception as e:
                query_expansion_errors.labels(
                    error_type=type(e).__name__,
                    expansion_method=expansion_method
                ).inc()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                query_expansion_duration.labels(expansion_method=expansion_method).observe(duration)
                return result
            except Exception as e:
                query_expansion_errors.labels(
                    error_type=type(e).__name__,
                    expansion_method=expansion_method
                ).inc()
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# ============================================================================
# Helper Functions
# ============================================================================

import asyncio


def record_document_processing(duration: float, document_type: str, use_llamaparse: bool, success: bool = True):
    """Record document processing metrics."""
    document_processing_duration.labels(
        document_type=document_type,
        use_llamaparse=str(use_llamaparse).lower()
    ).observe(duration)

    if success:
        documents_completed.inc()


def record_retrieval_quality(score: float, retrieval_strategy: str):
    """Record retrieval quality score."""
    retrieval_quality_score.labels(retrieval_strategy=retrieval_strategy).observe(score)


def record_query(query_type: str, collection_name: str, duration: float):
    """Record query metrics."""
    queries_total.labels(query_type=query_type, collection_name=collection_name).inc()
    query_processing_duration.labels(query_type=query_type).observe(duration)


def record_cache_access(hit: bool, cache_type: str = "query"):
    """Record cache access metrics."""
    if hit:
        cache_hits.labels(cache_type=cache_type).inc()
    else:
        cache_misses.labels(cache_type=cache_type).inc()


def set_system_info(info: Dict[str, Any]):
    """Set system information."""
    system_info.info(info)
