import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """Application configuration settings."""

    app_name: str = Field(default="TextReadingRAG", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # FastAPI settings
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    reload: bool = Field(default=False, description="Enable auto-reload in development")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class OpenAISettings(BaseSettings):
    """OpenAI API configuration."""

    openai_api_key: str = Field(description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    openai_temperature: float = Field(default=0.1, description="LLM temperature")
    openai_max_tokens: int = Field(default=2048, description="Maximum tokens for responses")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @validator("openai_temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v


class LlamaParseSettings(BaseSettings):
    """LlamaParse configuration for advanced PDF processing."""

    llama_cloud_api_key: Optional[str] = Field(
        default=None,
        description="LlamaCloud API key for LlamaParse"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class ChromaDBSettings(BaseSettings):
    """ChromaDB vector store configuration."""

    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, description="ChromaDB port")
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        default="pdf_documents",
        description="Default ChromaDB collection name"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @validator("chroma_persist_directory")
    def create_persist_directory(cls, v):
        """Ensure the persistence directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class VectorStoreSettings(BaseSettings):
    """Vector store and embedding configuration."""

    embedding_dimension: int = Field(default=1536, description="Embedding dimension")
    chunk_size: int = Field(default=512, description="Text chunk size")
    chunk_overlap: int = Field(default=128, description="Chunk overlap size")
    max_chunks_per_doc: int = Field(
        default=1000,
        description="Maximum chunks per document"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v < 100 or v > 2000:
            raise ValueError("Chunk size must be between 100 and 2000")
        return v

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class HybridSearchSettings(BaseSettings):
    """Hybrid search configuration."""

    dense_top_k: int = Field(default=10, description="Top-k for dense vector search")
    sparse_top_k: int = Field(default=10, description="Top-k for sparse BM25 search")
    hybrid_top_k: int = Field(default=5, description="Final top-k after hybrid fusion")
    alpha: float = Field(
        default=0.5,
        description="Weight for dense vs sparse search (0.0=sparse only, 1.0=dense only)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @validator("alpha")
    def validate_alpha(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        return v


class RerankingSettings(BaseSettings):
    """Reranking configuration."""

    rerank_top_n: int = Field(default=3, description="Number of documents to rerank")
    rerank_model: str = Field(
        default="BAAI/bge-reranker-large",
        description="Reranking model"
    )
    use_llm_rerank: bool = Field(
        default=True,
        description="Enable LLM-based reranking"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class QueryExpansionSettings(BaseSettings):
    """Query expansion configuration."""

    enable_query_expansion: bool = Field(
        default=True,
        description="Enable query expansion"
    )
    query_expansion_methods: List[str] = Field(
        default=["llm", "synonym"],
        description="Query expansion methods"
    )
    max_expanded_queries: int = Field(
        default=3,
        description="Maximum number of expanded queries"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @validator("query_expansion_methods")
    def validate_expansion_methods(cls, v):
        allowed_methods = {"llm", "synonym", "hyde"}
        if not all(method in allowed_methods for method in v):
            raise ValueError(f"Invalid expansion methods. Allowed: {allowed_methods}")
        return v


class FileUploadSettings(BaseSettings):
    """File upload configuration."""

    max_file_size: int = Field(default=50, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt"],
        description="Allowed file extensions"
    )
    upload_dir: str = Field(
        default="./data/uploads",
        description="Upload directory"
    )
    processed_dir: str = Field(
        default="./data/processed",
        description="Processed files directory"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @validator("upload_dir", "processed_dir")
    def create_directories(cls, v):
        """Ensure directories exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class CacheSettings(BaseSettings):
    """Caching configuration."""

    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    enable_cache: bool = Field(default=False, description="Enable caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class SecuritySettings(BaseSettings):
    """Security configuration."""

    secret_key: str = Field(description="Secret key for JWT and session management")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="CORS allowed origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class PerformanceSettings(BaseSettings):
    """Performance configuration."""

    max_concurrent_uploads: int = Field(
        default=5,
        description="Maximum concurrent file uploads"
    )
    enable_async_processing: bool = Field(
        default=True,
        description="Enable asynchronous processing"
    )
    batch_size: int = Field(default=10, description="Batch size for processing")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class DevelopmentSettings(BaseSettings):
    """Development and testing configuration."""

    mock_embeddings: bool = Field(
        default=False,
        description="Use mock embeddings for development"
    )
    mock_llm_responses: bool = Field(
        default=False,
        description="Use mock LLM responses for development"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class Settings(BaseSettings):
    """Main settings class combining all configuration sections."""

    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    llama_parse: LlamaParseSettings = Field(default_factory=LlamaParseSettings)
    chroma: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    hybrid_search: HybridSearchSettings = Field(default_factory=HybridSearchSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    query_expansion: QueryExpansionSettings = Field(default_factory=QueryExpansionSettings)
    file_upload: FileUploadSettings = Field(default_factory=FileUploadSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    development: DevelopmentSettings = Field(default_factory=DevelopmentSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()