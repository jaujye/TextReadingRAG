"""ChromaDB vector store integration for TextReadingRAG."""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from llama_index.core.schema import NodeWithScore, BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

from src.core.config import Settings
from src.core.exceptions import VectorStoreError, DatabaseError

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation with hybrid search support."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        persist_directory: Optional[str] = None,
        collection_name: str = "pdf_documents",
        embedding_function: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            host: ChromaDB host
            port: ChromaDB port
            persist_directory: Directory for persistent storage
            collection_name: Default collection name
            embedding_function: Custom embedding function
            settings: Application settings
        """
        self.host = host
        self.port = port
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.settings = settings

        # Initialize ChromaDB client
        try:
            # Use PersistentClient only for localhost with persist_directory
            # Use HttpClient for remote connections
            is_local = host in ["localhost", "127.0.0.1", "0.0.0.0"]

            if persist_directory and is_local:
                self._client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )
                logger.info(f"Using PersistentClient at {persist_directory}")
            else:
                self._client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                    ),
                )
                logger.info(f"Using HttpClient to connect to {host}:{port}")

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise VectorStoreError(f"Failed to connect to ChromaDB: {e}")

        # Set up embedding function
        if embedding_function:
            self.embedding_function = embedding_function
        else:
            # Use default OpenAI embedding function
            try:
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.llm.openai_api_key if settings else None,
                    model_name=settings.llm.openai_embedding_model if settings else "text-embedding-3-small",
                )
            except Exception as e:
                logger.warning(f"Failed to initialize embedding function: {e}")
                self.embedding_function = None

        # Collection cache
        self._collections: Dict[str, Any] = {}

    @property
    def client(self):
        """Get ChromaDB client."""
        return self._client

    def get_collection(
        self,
        collection_name: Optional[str] = None,
        create_if_not_exists: bool = True,
    ) -> Any:
        """
        Get or create a ChromaDB collection.

        Args:
            collection_name: Name of the collection
            create_if_not_exists: Whether to create the collection if it doesn't exist

        Returns:
            ChromaDB collection object
        """
        name = collection_name or self.collection_name

        if name in self._collections:
            return self._collections[name]

        try:
            if create_if_not_exists:
                collection = self._client.get_or_create_collection(
                    name=name,
                    embedding_function=self.embedding_function,
                    metadata={"created_at": datetime.utcnow().isoformat()},
                )
            else:
                collection = self._client.get_collection(
                    name=name,
                    embedding_function=self.embedding_function,
                )

            self._collections[name] = collection
            logger.info(f"Retrieved collection: {name}")
            return collection

        except Exception as e:
            logger.error(f"Failed to get collection {name}: {e}")
            raise VectorStoreError(f"Failed to get collection {name}: {e}")

    def add(
        self,
        nodes: List[BaseNode],
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Add nodes to the vector store.

        Args:
            nodes: List of nodes to add
            collection_name: Target collection name

        Returns:
            List of node IDs
        """
        if not nodes:
            return []

        collection = self.get_collection(collection_name)

        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for node in nodes:
                # Generate ID if not present
                node_id = node.node_id or str(uuid.uuid4())
                ids.append(node_id)

                # Get embedding
                if node.embedding:
                    embeddings.append(node.embedding)
                else:
                    # If no embedding, let ChromaDB generate it
                    embeddings = None
                    break

                # Get text content
                documents.append(node.get_content())

                # Prepare metadata
                metadata = node.metadata or {}
                metadata.update({
                    "node_id": node_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "text_length": len(node.get_content()),
                })

                # Ensure metadata values are JSON serializable
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)

                metadatas.append(cleaned_metadata)

            # Add to collection
            if embeddings:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )

            logger.info(f"Added {len(nodes)} nodes to collection {collection.name}")
            return ids

        except Exception as e:
            logger.error(f"Failed to add nodes to vector store: {e}")
            raise VectorStoreError(f"Failed to add nodes: {e}")

    def delete(
        self,
        ref_doc_id: str,
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Delete nodes by reference document ID.

        Args:
            ref_doc_id: Reference document ID
            collection_name: Target collection name
        """
        collection = self.get_collection(collection_name, create_if_not_exists=False)

        try:
            # Query for nodes with this ref_doc_id
            results = collection.get(
                where={"ref_doc_id": ref_doc_id},
                include=["metadatas"],
            )

            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} nodes for document {ref_doc_id}")

        except Exception as e:
            logger.error(f"Failed to delete nodes for document {ref_doc_id}: {e}")
            raise VectorStoreError(f"Failed to delete nodes: {e}")

    def query(
        self,
        query: VectorStoreQuery,
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> VectorStoreQueryResult:
        """
        Query the vector store.

        Args:
            query: Vector store query object
            collection_name: Target collection name

        Returns:
            Query results
        """
        collection = self.get_collection(collection_name, create_if_not_exists=False)

        try:
            # Build query parameters
            query_params = {
                "n_results": query.similarity_top_k,
                "include": ["documents", "metadatas", "distances"],
            }

            # Add query embedding or text
            if query.query_embedding:
                query_params["query_embeddings"] = [query.query_embedding]
            elif query.query_str:
                query_params["query_texts"] = [query.query_str]
            else:
                raise VectorStoreError("Either query_embedding or query_str must be provided")

            # Add metadata filters
            if query.filters:
                where_clause = self._build_where_clause(query.filters)
                if where_clause:
                    query_params["where"] = where_clause

            # Execute query
            results = collection.query(**query_params)

            # Convert results to VectorStoreQueryResult
            nodes = []
            similarities = []
            ids = []

            if results["ids"] and results["ids"][0]:
                for i, node_id in enumerate(results["ids"][0]):
                    # Create TextNode from results
                    document = results["documents"][0][i] if results["documents"] else ""
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0.0

                    # Convert distance to similarity (assuming cosine distance)
                    similarity = 1.0 - distance

                    node = TextNode(
                        id_=node_id,
                        text=document,
                        metadata=metadata,
                    )

                    nodes.append(node)
                    similarities.append(similarity)
                    ids.append(node_id)

            return VectorStoreQueryResult(
                nodes=nodes,
                similarities=similarities,
                ids=ids,
            )

        except Exception as e:
            logger.error(f"Failed to query vector store: {e}")
            raise VectorStoreError(f"Failed to query vector store: {e}")

    def _build_where_clause(self, filters: MetadataFilters) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from metadata filters.

        Args:
            filters: Metadata filters

        Returns:
            ChromaDB where clause
        """
        if not filters.filters:
            return {}

        where_conditions = []

        for filter_item in filters.filters:
            condition = self._build_filter_condition(filter_item)
            if condition:
                where_conditions.append(condition)

        if not where_conditions:
            return {}

        # Handle multiple conditions based on condition type
        if len(where_conditions) == 1:
            return where_conditions[0]

        # For multiple conditions, use AND by default
        # ChromaDB uses different syntax for complex queries
        if filters.condition and filters.condition.value == "OR":
            return {"$or": where_conditions}
        else:
            return {"$and": where_conditions}

    def _build_filter_condition(self, filter_item: MetadataFilter) -> Dict[str, Any]:
        """
        Build a single filter condition for ChromaDB.

        Args:
            filter_item: Single metadata filter

        Returns:
            ChromaDB filter condition
        """
        key = filter_item.key
        value = filter_item.value
        operator = filter_item.operator

        # Map LlamaIndex operators to ChromaDB operators
        operator_map = {
            FilterOperator.EQ: "$eq",
            FilterOperator.NE: "$ne",
            FilterOperator.GT: "$gt",
            FilterOperator.GTE: "$gte",
            FilterOperator.LT: "$lt",
            FilterOperator.LTE: "$lte",
            FilterOperator.IN: "$in",
            FilterOperator.NIN: "$nin",
            FilterOperator.CONTAINS: "$contains",
        }

        chroma_operator = operator_map.get(operator)
        if not chroma_operator:
            logger.warning(f"Unsupported operator {operator}, skipping filter")
            return {}

        return {key: {chroma_operator: value}}

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about a collection.

        Args:
            collection_name: Target collection name

        Returns:
            Collection statistics
        """
        collection = self.get_collection(collection_name, create_if_not_exists=False)

        try:
            # Get collection info
            count = collection.count()
            peek_results = collection.peek(limit=1)

            stats = {
                "name": collection.name,
                "count": count,
                "metadata": collection.metadata,
            }

            # Add sample metadata if available
            if peek_results["metadatas"]:
                sample_metadata = peek_results["metadatas"][0]
                stats["sample_metadata_keys"] = list(sample_metadata.keys())

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise VectorStoreError(f"Failed to get collection stats: {e}")

    def list_collections(self) -> List[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise VectorStoreError(f"Failed to list collections: {e}")

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create a new collection.

        Args:
            name: Collection name
            metadata: Collection metadata

        Returns:
            True if successful
        """
        try:
            collection_metadata = metadata or {}
            collection_metadata["created_at"] = datetime.utcnow().isoformat()

            collection = self._client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
                metadata=collection_metadata,
            )

            self._collections[name] = collection
            logger.info(f"Created collection: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            raise VectorStoreError(f"Failed to create collection {name}: {e}")

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if successful
        """
        try:
            self._client.delete_collection(name=name)

            # Remove from cache
            if name in self._collections:
                del self._collections[name]

            logger.info(f"Deleted collection: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            raise VectorStoreError(f"Failed to delete collection {name}: {e}")

    def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata only.

        Args:
            metadata_filter: Metadata filter conditions
            collection_name: Target collection name
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        collection = self.get_collection(collection_name, create_if_not_exists=False)

        try:
            results = collection.get(
                where=metadata_filter,
                limit=limit,
                include=["documents", "metadatas"],
            )

            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    documents.append({
                        "id": doc_id,
                        "document": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    })

            return documents

        except Exception as e:
            logger.error(f"Failed to search by metadata: {e}")
            raise VectorStoreError(f"Failed to search by metadata: {e}")

    def close(self) -> None:
        """Close the vector store connection."""
        try:
            # Clear collection cache
            self._collections.clear()

            # ChromaDB client doesn't need explicit closing
            logger.info("Vector store connection closed")

        except Exception as e:
            logger.error(f"Error closing vector store: {e}")

    @property
    def client_info(self) -> Dict[str, Any]:
        """Get client information."""
        return {
            "host": self.host,
            "port": self.port,
            "persist_directory": self.persist_directory,
            "default_collection": self.collection_name,
            "embedding_function": type(self.embedding_function).__name__ if self.embedding_function else None,
        }