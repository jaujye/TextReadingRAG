"""API request models for TextReadingRAG application."""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


class QueryType(str, Enum):
    """Types of queries supported by the RAG system."""

    SEARCH = "search"
    SUMMARIZE = "summarize"
    QUESTION_ANSWER = "question_answer"
    EXTRACT = "extract"
    COMPARE = "compare"


class RetrievalStrategy(str, Enum):
    """Retrieval strategies for the RAG system."""

    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    SEMANTIC_HYBRID = "semantic_hybrid"


class RerankingModel(str, Enum):
    """Available reranking models."""

    BGE_RERANKER_LARGE = "BAAI/bge-reranker-large"
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    LLM_RERANK = "llm-rerank"
    COLBERT = "colbert"


class DocumentUploadRequest(BaseModel):
    """Request model for document upload metadata."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the file")
    size: int = Field(..., description="File size in bytes")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the document"
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Target collection name (uses default if not specified)"
    )

    @validator("size")
    def validate_file_size(cls, v):
        max_size = 50 * 1024 * 1024  # 50 MB
        if v > max_size:
            raise ValueError(f"File size {v} exceeds maximum allowed size of {max_size} bytes")
        return v


class DocumentProcessingOptions(BaseModel):
    """Options for document processing."""

    chunk_size: Optional[int] = Field(
        default=512,
        ge=100,
        le=2000,
        description="Text chunk size for document splitting"
    )
    chunk_overlap: Optional[int] = Field(
        default=128,
        ge=0,
        le=500,
        description="Overlap between chunks"
    )
    use_llamaparse: Optional[bool] = Field(
        default=False,
        description="Use LlamaParse for advanced PDF processing"
    )
    extract_tables: Optional[bool] = Field(
        default=True,
        description="Extract tables from documents"
    )
    extract_images: Optional[bool] = Field(
        default=False,
        description="Extract and process images from documents"
    )


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., min_length=1, max_length=1000, description="The query text")
    query_type: QueryType = Field(
        default=QueryType.QUESTION_ANSWER,
        description="Type of query to perform"
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Target collection name (searches all if not specified)"
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific document IDs to search (searches all if not specified)"
    )

    # Retrieval parameters
    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Retrieval strategy to use"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to return"
    )
    dense_top_k: Optional[int] = Field(
        default=10,
        ge=1,
        le=100,
        description="Top-k for dense vector search"
    )
    sparse_top_k: Optional[int] = Field(
        default=10,
        ge=1,
        le=100,
        description="Top-k for sparse BM25 search"
    )
    alpha: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for dense vs sparse search (0.0=sparse only, 1.0=dense only)"
    )

    # Query expansion options
    enable_query_expansion: Optional[bool] = Field(
        default=True,
        description="Enable query expansion"
    )
    expansion_methods: Optional[List[str]] = Field(
        default=["llm"],
        description="Query expansion methods to use"
    )

    # Reranking options
    enable_reranking: Optional[bool] = Field(
        default=True,
        description="Enable reranking of results"
    )
    reranking_model: Optional[RerankingModel] = Field(
        default=RerankingModel.BGE_RERANKER_LARGE,
        description="Reranking model to use"
    )
    rerank_top_n: Optional[int] = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of documents to rerank"
    )

    # Response generation options
    include_sources: Optional[bool] = Field(
        default=True,
        description="Include source documents in response"
    )
    include_scores: Optional[bool] = Field(
        default=True,
        description="Include relevance scores in response"
    )
    max_response_length: Optional[int] = Field(
        default=2048,
        ge=100,
        le=4096,
        description="Maximum length of generated response"
    )
    temperature: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Temperature for response generation"
    )


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""

    queries: List[QueryRequest] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of queries to process"
    )
    parallel_processing: Optional[bool] = Field(
        default=True,
        description="Process queries in parallel"
    )


class DocumentUpdateRequest(BaseModel):
    """Request model for updating document metadata."""

    metadata: Dict[str, Any] = Field(..., description="Updated metadata")
    reprocess: Optional[bool] = Field(
        default=False,
        description="Reprocess the document with new settings"
    )
    processing_options: Optional[DocumentProcessingOptions] = Field(
        default=None,
        description="New processing options (if reprocessing)"
    )


class CollectionCreateRequest(BaseModel):
    """Request model for creating a new collection."""

    name: str = Field(..., min_length=1, max_length=100, description="Collection name")
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Collection description"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Collection metadata"
    )

    @validator("name")
    def validate_collection_name(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Collection name must contain only alphanumeric characters, hyphens, and underscores")
        return v


class SearchFilters(BaseModel):
    """Filters for search operations."""

    date_range: Optional[Dict[str, datetime]] = Field(
        default=None,
        description="Date range filter with 'start' and 'end' keys"
    )
    document_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by document types"
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata-based filters"
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold"
    )


class AdvancedQueryRequest(QueryRequest):
    """Extended query request with advanced options."""

    filters: Optional[SearchFilters] = Field(
        default=None,
        description="Search filters"
    )
    context_window: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of surrounding chunks to include for context"
    )
    enable_citations: Optional[bool] = Field(
        default=True,
        description="Include citations in the response"
    )
    response_format: Optional[str] = Field(
        default="text",
        description="Response format (text, json, markdown)"
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Custom prompt template for response generation"
    )


class DocumentComparisonRequest(BaseModel):
    """Request model for comparing documents."""

    document_ids: List[str] = Field(
        ...,
        min_items=2,
        max_items=5,
        description="Document IDs to compare"
    )
    comparison_aspects: Optional[List[str]] = Field(
        default=["content", "themes", "conclusions"],
        description="Aspects to compare"
    )
    output_format: Optional[str] = Field(
        default="structured",
        description="Output format for comparison results"
    )


class SummarizationRequest(BaseModel):
    """Request model for document summarization."""

    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific documents to summarize (all if not specified)"
    )
    summary_type: Optional[str] = Field(
        default="extractive",
        description="Type of summary (extractive, abstractive, key_points)"
    )
    max_length: Optional[int] = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum length of summary"
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas to focus on in summary"
    )