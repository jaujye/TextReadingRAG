"""PDF document ingestion module for TextReadingRAG."""

import asyncio
import hashlib
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    from llama_parse import LlamaParse
    LLAMA_PARSE_AVAILABLE = True
except ImportError:
    LLAMA_PARSE_AVAILABLE = False
    LlamaParse = None

from src.core.config import Settings
from src.core.exceptions import (
    DocumentIngestionError,
    FileProcessingError,
    VectorStoreError,
)
from src.rag.vector_store import ChromaVectorStore
from src.rag.language_utils import detect_language, is_chinese, split_chinese_text

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing and metadata extraction."""

    def __init__(self, settings: Settings):
        """
        Initialize document processor.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.embedding_model = self._initialize_embedding_model()

    def _initialize_embedding_model(self) -> OpenAIEmbedding:
        """Initialize the embedding model."""
        try:
            return OpenAIEmbedding(
                api_key=self.settings.llm.openai_api_key,
                model=self.settings.llm.openai_embedding_model,
            )
        except Exception as e:
            if not self.settings.app.mock_embeddings:
                raise DocumentIngestionError(f"Failed to initialize embedding model: {e}")
            logger.warning("Using mock embeddings for development")
            return None

    def create_text_splitter(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        language: Optional[str] = None,
    ) -> SentenceSplitter:
        """
        Create a text splitter with specified parameters.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            language: Language code for language-specific splitting

        Returns:
            Configured text splitter
        """
        # Use language-specific chunk sizes for Chinese
        if language == 'zh':
            chunk_size = chunk_size or self.settings.rag.chinese_chunk_size
            chunk_overlap = chunk_overlap or self.settings.rag.chinese_chunk_overlap
        else:
            chunk_size = chunk_size or self.settings.rag.chunk_size
            chunk_overlap = chunk_overlap or self.settings.rag.chunk_overlap

        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;]+[,.;]?",
        )

    def split_text_with_language_detection(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Tuple[List[str], str]:
        """
        Split text using language-aware splitting.

        Args:
            text: Text to split
            chunk_size: Target chunk size
            chunk_overlap: Overlap size

        Returns:
            Tuple of (chunks, detected_language)
        """
        # Detect language if enabled
        language = detect_language(text) if self.settings.rag.enable_language_detection else self.settings.rag.default_language

        # Use Chinese-specific splitting for Chinese text
        if language == 'zh':
            chunk_size = chunk_size or self.settings.rag.chinese_chunk_size
            chunk_overlap = chunk_overlap or self.settings.rag.chinese_chunk_overlap
            chunks = split_chinese_text(text, chunk_size, chunk_overlap)
        else:
            # Use LlamaIndex SentenceSplitter for English
            text_splitter = self.create_text_splitter(chunk_size, chunk_overlap, language)
            from llama_index.core import Document
            doc = Document(text=text)
            nodes = text_splitter.get_nodes_from_documents([doc])
            chunks = [node.text for node in nodes]

        return chunks, language

    def extract_document_metadata(
        self,
        file_path: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract metadata from a document file.

        Args:
            file_path: Path to the document file
            additional_metadata: Additional metadata to include

        Returns:
            Document metadata
        """
        try:
            file_path = Path(file_path)
            file_stats = file_path.stat()

            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size": file_stats.st_size,
                "file_extension": file_path.suffix.lower(),
                "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "processed_at": datetime.utcnow().isoformat(),
            }

            # Add MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                metadata["mime_type"] = mime_type

            # Calculate file hash for deduplication
            metadata["file_hash"] = self._calculate_file_hash(file_path)

            # Add additional metadata
            if additional_metadata:
                metadata.update(additional_metadata)

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            raise FileProcessingError(f"Failed to extract metadata: {e}", file_path=str(file_path))

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""


class DocumentLoader:
    """Handles loading documents from various sources."""

    def __init__(self, settings: Settings):
        """
        Initialize document loader.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.processor = DocumentProcessor(settings)

    def load_from_file(
        self,
        file_path: str,
        use_llamaparse: bool = False,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load documents from a single file.

        Args:
            file_path: Path to the file
            use_llamaparse: Whether to use LlamaParse for advanced parsing
            additional_metadata: Additional metadata to include

        Returns:
            List of loaded documents
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileProcessingError(f"File not found: {file_path}")

            # Extract base metadata
            metadata = self.loader.processor.extract_document_metadata(
                str(file_path),
                additional_metadata,
            )

            # Choose loading strategy
            if use_llamaparse and LLAMA_PARSE_AVAILABLE and self.settings.llm.llama_cloud_api_key:
                documents = self._load_with_llamaparse(file_path, metadata)
            else:
                documents = self._load_with_simple_reader(file_path, metadata)

            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise FileProcessingError(f"Failed to load file: {e}", file_path=str(file_path))

    def _load_with_llamaparse(
        self,
        file_path: Path,
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Load document using LlamaParse."""
        if not LLAMA_PARSE_AVAILABLE:
            raise DocumentIngestionError("LlamaParse is not available")

        try:
            parser = LlamaParse(
                api_key=self.settings.llm.llama_cloud_api_key,
                result_type="markdown",
                verbose=True,
            )

            documents = parser.load_data(str(file_path))

            # Add metadata to documents
            for doc in documents:
                doc.metadata.update(base_metadata)
                doc.metadata["parsing_method"] = "llamaparse"
                # Detect and add language
                if self.settings.rag.enable_language_detection and doc.text:
                    doc.metadata["language"] = detect_language(doc.text)

            return documents

        except Exception as e:
            logger.error(f"LlamaParse failed for {file_path}: {e}")
            # Fallback to simple reader
            logger.info(f"Falling back to SimpleDirectoryReader for {file_path}")
            return self._load_with_simple_reader(file_path, base_metadata)

    def _load_with_simple_reader(
        self,
        file_path: Path,
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Load document using SimpleDirectoryReader."""
        try:
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
                filename_as_id=True,
            )

            documents = reader.load_data()

            # Add metadata to documents
            for doc in documents:
                doc.metadata.update(base_metadata)
                doc.metadata["parsing_method"] = "simple_reader"
                # Detect and add language
                if self.settings.rag.enable_language_detection and doc.text:
                    doc.metadata["language"] = detect_language(doc.text)

            return documents

        except Exception as e:
            logger.error(f"SimpleDirectoryReader failed for {file_path}: {e}")
            raise FileProcessingError(f"Failed to read file with SimpleDirectoryReader: {e}")

    def load_from_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        use_llamaparse: bool = False,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load documents from a directory.

        Args:
            directory_path: Path to the directory
            file_extensions: Allowed file extensions
            use_llamaparse: Whether to use LlamaParse
            recursive: Whether to search recursively

        Returns:
            List of loaded documents
        """
        try:
            directory_path = Path(directory_path)

            if not directory_path.exists() or not directory_path.is_dir():
                raise FileProcessingError(f"Directory not found: {directory_path}")

            # Get file extensions from settings if not provided
            if file_extensions is None:
                file_extensions = self.settings.app.allowed_extensions

            # Find all matching files
            pattern = "**/*" if recursive else "*"
            all_files = []

            for ext in file_extensions:
                ext = ext.lower()
                if not ext.startswith("."):
                    ext = f".{ext}"
                files = list(directory_path.glob(f"{pattern}{ext}"))
                all_files.extend(files)

            logger.info(f"Found {len(all_files)} files in {directory_path}")

            # Load all files
            all_documents = []
            for file_path in all_files:
                try:
                    documents = self.load_from_file(
                        str(file_path),
                        use_llamaparse=use_llamaparse,
                    )
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to load file {file_path}: {e}")
                    # Continue with other files

            return all_documents

        except Exception as e:
            logger.error(f"Failed to load directory {directory_path}: {e}")
            raise FileProcessingError(f"Failed to load directory: {e}")


class DocumentIngestionService:
    """Main service for document ingestion and processing."""

    def __init__(self, settings: Settings, vector_store: Optional[ChromaVectorStore] = None):
        """
        Initialize document ingestion service.

        Args:
            settings: Application settings
            vector_store: Vector store instance
        """
        self.settings = settings
        self.vector_store = vector_store or ChromaVectorStore(
            host=settings.rag.chroma_host,
            port=settings.rag.chroma_port,
            persist_directory=settings.rag.chroma_persist_directory,
            settings=settings,
        )
        self.loader = DocumentLoader(settings)

    async def ingest_file(
        self,
        file_path: str,
        collection_name: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a single file into the vector store.

        Args:
            file_path: Path to the file
            collection_name: Target collection name
            processing_options: Processing options

        Returns:
            Ingestion results
        """
        try:
            start_time = datetime.utcnow()

            # Parse processing options
            options = processing_options or {}
            chunk_size = options.get("chunk_size")
            chunk_overlap = options.get("chunk_overlap")
            use_llamaparse = options.get("use_llamaparse", False)
            additional_metadata = options.get("metadata", {})

            # Load documents
            documents = self.loader.load_from_file(
                file_path,
                use_llamaparse=use_llamaparse,
                additional_metadata=additional_metadata,
            )

            if not documents:
                raise DocumentIngestionError("No documents loaded from file")

            # Process documents into nodes
            nodes = await self._process_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Generate embeddings if not using mock mode
            if not self.settings.app.mock_embeddings:
                await self._generate_embeddings(nodes)

            # Store in vector store
            collection_name = collection_name or self.settings.rag.chroma_collection_name
            node_ids = self.vector_store.add(nodes, collection_name=collection_name)

            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            # Prepare results
            results = {
                "document_id": str(uuid.uuid4()),
                "file_path": file_path,
                "filename": Path(file_path).name,
                "collection_name": collection_name,
                "nodes_created": len(nodes),
                "node_ids": node_ids,
                "processing_time_seconds": processing_time,
                "processed_at": end_time.isoformat(),
                "processing_options": options,
            }

            logger.info(f"Successfully ingested {file_path}: {len(nodes)} nodes created")
            return results

        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            raise DocumentIngestionError(f"Failed to ingest file: {e}", file_path=file_path)

    async def ingest_batch(
        self,
        file_paths: List[str],
        collection_name: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest multiple files in batch.

        Args:
            file_paths: List of file paths
            collection_name: Target collection name
            processing_options: Processing options
            parallel: Whether to process files in parallel

        Returns:
            Batch ingestion results
        """
        try:
            start_time = datetime.utcnow()

            if parallel and self.settings.app.enable_async_processing:
                # Process files in parallel
                tasks = [
                    self.ingest_file(file_path, collection_name, processing_options)
                    for file_path in file_paths
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Process files sequentially
                results = []
                for file_path in file_paths:
                    try:
                        result = await self.ingest_file(file_path, collection_name, processing_options)
                        results.append(result)
                    except Exception as e:
                        results.append(e)

            # Separate successful and failed results
            successful_results = []
            failed_results = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_results.append({
                        "file_path": file_paths[i],
                        "error": str(result),
                        "error_type": type(result).__name__,
                    })
                else:
                    successful_results.append(result)

            end_time = datetime.utcnow()
            total_processing_time = (end_time - start_time).total_seconds()

            # Compile batch results
            batch_results = {
                "batch_id": str(uuid.uuid4()),
                "total_files": len(file_paths),
                "successful_files": len(successful_results),
                "failed_files": len(failed_results),
                "successful_results": successful_results,
                "failed_results": failed_results,
                "total_processing_time_seconds": total_processing_time,
                "processed_at": end_time.isoformat(),
                "collection_name": collection_name,
                "processing_options": processing_options,
            }

            logger.info(
                f"Batch ingestion completed: {len(successful_results)}/{len(file_paths)} files successful"
            )
            return batch_results

        except Exception as e:
            logger.error(f"Failed to ingest batch: {e}")
            raise DocumentIngestionError(f"Failed to ingest batch: {e}")

    async def _process_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[BaseNode]:
        """
        Process documents into nodes with language-aware chunking.

        Args:
            documents: List of documents to process
            chunk_size: Chunk size for text splitting
            chunk_overlap: Chunk overlap size

        Returns:
            List of processed nodes
        """
        try:
            all_nodes = []

            for doc in documents:
                # Get language from metadata or detect it
                language = doc.metadata.get("language")
                if not language and self.settings.rag.enable_language_detection:
                    language = detect_language(doc.text)
                    doc.metadata["language"] = language

                # Use language-specific splitting for Chinese
                if language == 'zh':
                    cs = chunk_size or self.settings.rag.chinese_chunk_size
                    co = chunk_overlap or self.settings.rag.chinese_chunk_overlap
                    chunks = split_chinese_text(doc.text, cs, co)

                    # Create nodes from chunks
                    for i, chunk in enumerate(chunks):
                        node = TextNode(
                            text=chunk,
                            metadata={
                                **doc.metadata,
                                "chunk_index": i,
                                "chunk_size": cs,
                                "chunk_overlap": co,
                                "language": language,
                            },
                            id_=f"{doc.doc_id}_{i}" if hasattr(doc, 'doc_id') else str(uuid.uuid4()),
                        )
                        all_nodes.append(node)
                else:
                    # Use LlamaIndex pipeline for English
                    text_splitter = self.loader.processor.create_text_splitter(
                        chunk_size, chunk_overlap, language
                    )

                    pipeline = IngestionPipeline(
                        transformations=[text_splitter],
                        docstore=None,
                    )

                    nodes = pipeline.run(documents=[doc], show_progress=False)

                    # Add language metadata
                    for node in nodes:
                        node.metadata["language"] = language or self.settings.rag.default_language
                        node.metadata["chunk_size"] = chunk_size or self.settings.rag.chunk_size
                        node.metadata["chunk_overlap"] = chunk_overlap or self.settings.rag.chunk_overlap

                    all_nodes.extend(nodes)

            # Add node IDs if missing
            for node in all_nodes:
                if not node.node_id:
                    node.node_id = str(uuid.uuid4())
                node.metadata["node_id"] = node.node_id

            logger.info(f"Processed {len(documents)} documents into {len(all_nodes)} nodes")
            return all_nodes

        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            raise DocumentIngestionError(f"Failed to process documents: {e}")

    async def _generate_embeddings(self, nodes: List[BaseNode]) -> None:
        """
        Generate embeddings for nodes.

        Args:
            nodes: List of nodes to generate embeddings for
        """
        try:
            if self.settings.app.mock_embeddings:
                # Generate mock embeddings for development
                import numpy as np
                for node in nodes:
                    node.embedding = np.random.rand(self.settings.rag.embedding_dimension).tolist()
                return

            embedding_model = self.loader.processor.embedding_model
            if not embedding_model:
                logger.warning("No embedding model available, skipping embedding generation")
                return

            # Generate embeddings in batches
            batch_size = self.settings.app.batch_size
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                texts = [node.get_content() for node in batch]

                try:
                    embeddings = embedding_model._get_text_embeddings(texts)
                    for node, embedding in zip(batch, embeddings):
                        node.embedding = embedding
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                    # Continue with next batch

            logger.info(f"Generated embeddings for {len(nodes)} nodes")

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise DocumentIngestionError(f"Failed to generate embeddings: {e}")

    def check_duplicate(
        self,
        file_hash: str,
        collection_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a document with the same hash already exists.

        Args:
            file_hash: File hash to check
            collection_name: Target collection name

        Returns:
            Existing document info if found, None otherwise
        """
        try:
            collection_name = collection_name or self.settings.rag.chroma_collection_name

            # Search for documents with the same hash
            results = self.vector_store.search_by_metadata(
                metadata_filter={"file_hash": file_hash},
                collection_name=collection_name,
                limit=1,
            )

            if results:
                return results[0]

            return None

        except Exception as e:
            logger.error(f"Failed to check for duplicates: {e}")
            return None

    def get_ingestion_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get ingestion statistics for a collection.

        Args:
            collection_name: Target collection name

        Returns:
            Ingestion statistics
        """
        try:
            collection_name = collection_name or self.settings.rag.chroma_collection_name
            return self.vector_store.get_collection_stats(collection_name)

        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            raise DocumentIngestionError(f"Failed to get ingestion stats: {e}")