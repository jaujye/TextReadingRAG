"""API response models for TextReadingRAG application."""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Status of document processing operations."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetrievalSource(BaseModel):
    """Information about a retrieved document source."""

    document_id: str = Field(..., description="Unique document identifier")
    chunk_id: Optional[str] = Field(default=None, description="Chunk identifier within document")
    filename: str = Field(..., description="Original filename")
    page_number: Optional[int] = Field(default=None, description="Page number (if applicable)")
    chunk_index: Optional[int] = Field(default=None, description="Index of chunk within document")
    content: str = Field(..., description="Retrieved content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class RetrievalResult(BaseModel):
    """Result from retrieval operation."""

    source: RetrievalSource = Field(..., description="Source information")
    score: float = Field(..., description="Relevance score")
    retrieval_method: str = Field(..., description="Method used for retrieval (vector, bm25, hybrid)")
    rank: int = Field(..., description="Rank in retrieval results")
    rerank_score: Optional[float] = Field(default=None, description="Score after reranking")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    query_type: str = Field(..., description="Type of query performed")

    # Retrieval information
    retrieved_documents: List[RetrievalResult] = Field(
        default=[],
        description="Documents retrieved for the query"
    )
    total_retrieved: int = Field(..., description="Total number of documents retrieved")

    # Processing information
    retrieval_strategy: str = Field(..., description="Retrieval strategy used")
    query_expansion_enabled: bool = Field(..., description="Whether query expansion was used")
    expanded_queries: Optional[List[str]] = Field(
        default=None,
        description="Expanded queries (if query expansion was used)"
    )
    reranking_enabled: bool = Field(..., description="Whether reranking was applied")
    reranking_model: Optional[str] = Field(default=None, description="Reranking model used")

    # Timing and performance
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    retrieval_time_ms: float = Field(..., description="Time spent on retrieval")
    reranking_time_ms: Optional[float] = Field(
        default=None,
        description="Time spent on reranking"
    )
    generation_time_ms: float = Field(..., description="Time spent on answer generation")

    # Response metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score for the answer"
    )
    citations: Optional[List[str]] = Field(
        default=None,
        description="Citations for the answer"
    )


class DocumentInfo(BaseModel):
    """Information about a document."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    size: int = Field(..., description="File size in bytes")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    processing_status: ProcessingStatus = Field(..., description="Processing status")
    chunk_count: Optional[int] = Field(default=None, description="Number of chunks created")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    collection_name: str = Field(..., description="Collection the document belongs to")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    collection_name: str = Field(..., description="Target collection")
    estimated_processing_time: Optional[int] = Field(
        default=None,
        description="Estimated processing time in seconds"
    )


class BatchUploadResponse(BaseModel):
    """Response model for batch document upload."""

    total_files: int = Field(..., description="Total number of files in batch")
    successful_uploads: List[DocumentUploadResponse] = Field(
        default=[],
        description="Successfully uploaded files"
    )
    failed_uploads: List[Dict[str, str]] = Field(
        default=[],
        description="Failed uploads with error messages"
    )
    batch_id: str = Field(..., description="Batch identifier for tracking")


class ProcessingProgress(BaseModel):
    """Progress information for document processing."""

    document_id: str = Field(..., description="Document identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress_percentage: float = Field(..., description="Progress percentage (0-100)")
    current_stage: str = Field(..., description="Current processing stage")
    stages_completed: List[str] = Field(default=[], description="Completed processing stages")
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time"
    )
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error details if processing failed"
    )


class CollectionInfo(BaseModel):
    """Information about a document collection."""

    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(default=None, description="Collection description")
    document_count: int = Field(..., description="Number of documents in collection")
    total_chunks: int = Field(..., description="Total number of chunks across all documents")
    created_timestamp: datetime = Field(..., description="Collection creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Collection metadata")


class SearchStatistics(BaseModel):
    """Statistics about search operations."""

    total_queries: int = Field(..., description="Total number of queries processed")
    average_response_time: float = Field(..., description="Average response time in milliseconds")
    average_retrieval_count: float = Field(..., description="Average number of documents retrieved")
    most_common_query_types: List[Dict[str, Union[str, int]]] = Field(
        default=[],
        description="Most common query types and their counts"
    )
    performance_metrics: Dict[str, float] = Field(
        default={},
        description="Various performance metrics"
    )


class HealthStatus(BaseModel):
    """Health status response."""

    status: str = Field(..., description="Overall health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Component-specific health information"
    )


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: Dict[str, Any] = Field(..., description="Error details")


class BatchQueryResponse(BaseModel):
    """Response model for batch queries."""

    batch_id: str = Field(..., description="Batch identifier")
    total_queries: int = Field(..., description="Total number of queries in batch")
    completed_queries: int = Field(..., description="Number of completed queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    results: List[QueryResponse] = Field(default=[], description="Query results")
    batch_processing_time_ms: float = Field(..., description="Total batch processing time")
    errors: List[Dict[str, str]] = Field(default=[], description="Errors encountered during processing")


class DocumentComparisonResponse(BaseModel):
    """Response model for document comparison."""

    comparison_id: str = Field(..., description="Comparison identifier")
    document_ids: List[str] = Field(..., description="Documents that were compared")
    comparison_aspects: List[str] = Field(..., description="Aspects that were compared")

    similarities: Dict[str, Any] = Field(..., description="Identified similarities")
    differences: Dict[str, Any] = Field(..., description="Identified differences")

    summary: str = Field(..., description="Summary of the comparison")
    detailed_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed comparison analysis"
    )

    confidence_scores: Dict[str, float] = Field(
        default={},
        description="Confidence scores for different aspects"
    )
    processing_time_ms: float = Field(..., description="Time taken for comparison")


class SummarizationResponse(BaseModel):
    """Response model for document summarization."""

    summary_id: str = Field(..., description="Summary identifier")
    document_ids: List[str] = Field(..., description="Documents that were summarized")
    summary_type: str = Field(..., description="Type of summary generated")

    summary: str = Field(..., description="Generated summary")
    key_points: List[str] = Field(default=[], description="Key points extracted")

    source_references: List[RetrievalSource] = Field(
        default=[],
        description="References to source content"
    )

    summary_length: int = Field(..., description="Length of summary in characters")
    compression_ratio: float = Field(..., description="Ratio of summary to original content")
    processing_time_ms: float = Field(..., description="Time taken for summarization")


class AnalyticsResponse(BaseModel):
    """Response model for analytics and insights."""

    time_period: str = Field(..., description="Time period for analytics")
    query_statistics: SearchStatistics = Field(..., description="Query-related statistics")
    document_statistics: Dict[str, Any] = Field(..., description="Document-related statistics")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    usage_patterns: Dict[str, Any] = Field(..., description="Usage pattern insights")
    generated_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConfigurationResponse(BaseModel):
    """Response model for configuration information."""

    retrieval_config: Dict[str, Any] = Field(..., description="Retrieval configuration")
    processing_config: Dict[str, Any] = Field(..., description="Processing configuration")
    llm_config: Dict[str, Any] = Field(..., description="LLM model configuration")
    limits_config: Dict[str, Any] = Field(..., description="Limits and constraints")
    features_enabled: List[str] = Field(..., description="Enabled features")


class StreamingResponse(BaseModel):
    """Response model for streaming operations."""

    event_type: str = Field(..., description="Type of streaming event")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sequence_number: Optional[int] = Field(default=None, description="Sequence number for ordering")