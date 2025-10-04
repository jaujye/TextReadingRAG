"""Hybrid retrieval strategies for TextReadingRAG."""

import asyncio
import logging
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from datetime import datetime

from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, MetadataFilters
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from src.core.config import Settings
from src.core.exceptions import RetrievalError, ConfigurationError
from src.rag.vector_store import ChromaVectorStore
from src.rag.language_utils import detect_language, tokenize_chinese

logger = logging.getLogger(__name__)


def multilingual_tokenizer(text: str) -> List[str]:
    """
    Tokenize text based on language detection.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    try:
        language = detect_language(text)
        if language == 'zh':
            # Use jieba for Chinese
            return tokenize_chinese(text)
        else:
            # Use simple whitespace tokenization for English
            return text.lower().split()
    except Exception as e:
        logger.warning(f"Tokenization failed, using default: {e}")
        return text.lower().split()


class RetrievalMode(str, Enum):
    """Different retrieval modes available."""

    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    SEMANTIC_HYBRID = "semantic_hybrid"
    AUTO = "auto"


class FusionMethod(str, Enum):
    """Methods for fusing multiple retrieval results."""

    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion
    WEIGHTED_SCORE = "weighted_score"
    LINEAR_COMBINATION = "linear_combination"
    MAX_SCORE = "max_score"


class DenseRetriever:
    """Dense vector retrieval using semantic embeddings."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_model: OpenAIEmbedding,
        settings: Settings,
        cache_service=None,
    ):
        """
        Initialize dense retriever.

        Args:
            vector_store: Vector store instance
            embedding_model: Embedding model for query encoding
            settings: Application settings
            cache_service: Optional cache service for embeddings
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.settings = settings
        self.cache = cache_service

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents using dense vector search.

        Args:
            query: Query string
            top_k: Number of results to retrieve
            collection_name: Target collection
            filters: Metadata filters

        Returns:
            List of retrieved nodes with scores
        """
        try:
            # Try to get cached embedding
            query_embedding = None
            if self.cache and not self.settings.app.mock_embeddings:
                model_name = self.settings.llm.openai_embedding_model
                query_embedding = await self.cache.get_embedding(query, model_name)

            # Generate query embedding if not cached
            if query_embedding is None:
                if self.settings.app.mock_embeddings:
                    import numpy as np
                    query_embedding = np.random.rand(self.settings.rag.embedding_dimension).tolist()
                else:
                    query_embedding = self.embedding_model.get_query_embedding(query)

                    # Cache the embedding
                    if self.cache:
                        model_name = self.settings.llm.openai_embedding_model
                        await self.cache.set_embedding(query, model_name, query_embedding)

            # Create vector store query
            vector_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=filters,
            )

            # Execute query
            result = self.vector_store.query(vector_query, collection_name=collection_name)

            # Convert to NodeWithScore objects
            nodes_with_scores = []
            for node, similarity, node_id in zip(result.nodes, result.similarities, result.ids):
                node_with_score = NodeWithScore(
                    node=node,
                    score=similarity,
                )
                # Add retrieval metadata
                node.metadata["retrieval_method"] = "dense_vector"
                node.metadata["similarity_score"] = similarity
                node.metadata["query"] = query

                nodes_with_scores.append(node_with_score)

            logger.debug(f"Dense retrieval returned {len(nodes_with_scores)} nodes for query: {query}")
            return nodes_with_scores

        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            raise RetrievalError(f"Dense retrieval failed: {e}", query=query, retrieval_type="dense")


class SparseRetriever:
    """Sparse retrieval using BM25 algorithm."""

    def __init__(self, settings: Settings):
        """
        Initialize sparse retriever.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._bm25_retrievers: Dict[str, BM25Retriever] = {}

    def _get_or_create_bm25_retriever(
        self,
        nodes: List[BaseNode],
        collection_name: str,
    ) -> BM25Retriever:
        """Get or create BM25 retriever with multilingual tokenization support."""
        if collection_name not in self._bm25_retrievers:
            try:
                # Adjust top_k to not exceed corpus size
                corpus_size = len(nodes)
                top_k = min(self.settings.rag.sparse_top_k, corpus_size) if corpus_size > 0 else 1

                retriever = BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=top_k,
                    tokenizer=multilingual_tokenizer,
                )
                self._bm25_retrievers[collection_name] = retriever
                logger.info(f"Created BM25 retriever with multilingual tokenizer for collection: {collection_name} (corpus_size={corpus_size}, top_k={top_k})")
            except Exception as e:
                logger.error(f"Failed to create BM25 retriever: {e}")
                raise RetrievalError(f"Failed to create BM25 retriever: {e}")

        return self._bm25_retrievers[collection_name]

    async def retrieve(
        self,
        query: str,
        nodes: List[BaseNode],
        top_k: int = 10,
        collection_name: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents using BM25 sparse retrieval.

        Args:
            query: Query string
            nodes: Corpus nodes for BM25
            top_k: Number of results to retrieve
            collection_name: Collection name for caching

        Returns:
            List of retrieved nodes with scores
        """
        try:
            collection_name = collection_name or "default"

            # Get or create BM25 retriever
            bm25_retriever = self._get_or_create_bm25_retriever(nodes, collection_name)

            # Create query bundle
            query_bundle = QueryBundle(query_str=query)

            # Retrieve using BM25
            retrieved_nodes = bm25_retriever.retrieve(query_bundle)

            # Limit results to top_k
            retrieved_nodes = retrieved_nodes[:top_k]

            # Add retrieval metadata
            for node_with_score in retrieved_nodes:
                node_with_score.node.metadata["retrieval_method"] = "bm25_sparse"
                node_with_score.node.metadata["bm25_score"] = node_with_score.score
                node_with_score.node.metadata["query"] = query

            logger.debug(f"BM25 retrieval returned {len(retrieved_nodes)} nodes for query: {query}")
            return retrieved_nodes

        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            raise RetrievalError(f"BM25 retrieval failed: {e}", query=query, retrieval_type="bm25")


class HybridFusion:
    """Fusion algorithms for combining multiple retrieval results."""

    @staticmethod
    def reciprocal_rank_fusion(
        results_list: List[List[NodeWithScore]],
        k: int = 60,
        weights: Optional[List[float]] = None,
    ) -> List[NodeWithScore]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        Args:
            results_list: List of retrieval results from different methods
            k: RRF parameter (typically 60)
            weights: Weights for different retrieval methods

        Returns:
            Fused results
        """
        if not results_list:
            return []

        # Default equal weights
        if weights is None:
            weights = [1.0] * len(results_list)

        # Collect all unique nodes
        node_scores: Dict[str, float] = {}
        node_objects: Dict[str, NodeWithScore] = {}

        for method_idx, results in enumerate(results_list):
            weight = weights[method_idx]

            for rank, node_with_score in enumerate(results):
                node_id = node_with_score.node.node_id

                # Calculate RRF score
                rrf_score = weight / (k + rank + 1)

                if node_id in node_scores:
                    node_scores[node_id] += rrf_score
                else:
                    node_scores[node_id] = rrf_score
                    node_objects[node_id] = node_with_score

        # Sort by combined score
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Create result list
        fused_results = []
        for node_id, score in sorted_nodes:
            node_with_score = node_objects[node_id]
            # Update score with RRF score
            node_with_score.score = score
            node_with_score.node.metadata["fusion_method"] = "rrf"
            node_with_score.node.metadata["rrf_score"] = score
            fused_results.append(node_with_score)

        return fused_results

    @staticmethod
    def weighted_score_fusion(
        results_list: List[List[NodeWithScore]],
        weights: Optional[List[float]] = None,
        normalize_scores: bool = True,
    ) -> List[NodeWithScore]:
        """
        Combine results using weighted score fusion.

        Args:
            results_list: List of retrieval results from different methods
            weights: Weights for different retrieval methods
            normalize_scores: Whether to normalize scores before fusion

        Returns:
            Fused results
        """
        if not results_list:
            return []

        # Default equal weights
        if weights is None:
            weights = [1.0] * len(results_list)

        # Normalize scores if requested
        if normalize_scores:
            for results in results_list:
                if results:
                    max_score = max(node.score for node in results)
                    min_score = min(node.score for node in results)
                    score_range = max_score - min_score

                    if score_range > 0:
                        for node in results:
                            node.score = (node.score - min_score) / score_range

        # Collect all unique nodes
        node_scores: Dict[str, float] = {}
        node_counts: Dict[str, int] = {}
        node_objects: Dict[str, NodeWithScore] = {}

        for method_idx, results in enumerate(results_list):
            weight = weights[method_idx]

            for node_with_score in results:
                node_id = node_with_score.node.node_id
                weighted_score = node_with_score.score * weight

                if node_id in node_scores:
                    node_scores[node_id] += weighted_score
                    node_counts[node_id] += 1
                else:
                    node_scores[node_id] = weighted_score
                    node_counts[node_id] = 1
                    node_objects[node_id] = node_with_score

        # Average scores across methods
        for node_id in node_scores:
            node_scores[node_id] /= node_counts[node_id]

        # Sort by combined score
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Create result list
        fused_results = []
        for node_id, score in sorted_nodes:
            node_with_score = node_objects[node_id]
            node_with_score.score = score
            node_with_score.node.metadata["fusion_method"] = "weighted_score"
            node_with_score.node.metadata["fused_score"] = score
            fused_results.append(node_with_score)

        return fused_results


class HybridRetrievalService:
    """Main service for hybrid retrieval combining multiple strategies."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        settings: Settings,
        embedding_model: Optional[OpenAIEmbedding] = None,
        cache_service=None,
    ):
        """
        Initialize hybrid retrieval service.

        Args:
            vector_store: Vector store instance
            settings: Application settings
            embedding_model: Embedding model
            cache_service: Optional cache service
        """
        self.vector_store = vector_store
        self.settings = settings
        self.cache = cache_service

        # Initialize embedding model
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            try:
                self.embedding_model = OpenAIEmbedding(
                    api_key=settings.llm.openai_api_key,
                    model=settings.llm.openai_embedding_model,
                )
            except Exception as e:
                if not settings.app.mock_embeddings:
                    raise ConfigurationError(f"Failed to initialize embedding model: {e}")
                logger.warning("Using mock embeddings for development")
                self.embedding_model = None

        # Initialize retrievers with cache
        self.dense_retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_model=self.embedding_model,
            settings=settings,
            cache_service=cache_service,
        )
        self.sparse_retriever = SparseRetriever(settings=settings)

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        collection_name: Optional[str] = None,
        filters: Optional[MetadataFilters] = None,
        fusion_method: FusionMethod = FusionMethod.RRF,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents using specified mode and parameters.

        Args:
            query: Query string
            mode: Retrieval mode
            top_k: Final number of results to return
            dense_top_k: Number of dense results to retrieve
            sparse_top_k: Number of sparse results to retrieve
            alpha: Weight for dense vs sparse (0.0=sparse only, 1.0=dense only)
            collection_name: Target collection
            filters: Metadata filters
            fusion_method: Method for fusing results

        Returns:
            Retrieved nodes with scores
        """
        try:
            start_time = datetime.utcnow()

            # Set default parameters
            top_k = top_k or self.settings.rag.hybrid_top_k
            dense_top_k = dense_top_k or self.settings.rag.dense_top_k
            sparse_top_k = sparse_top_k or self.settings.rag.sparse_top_k
            alpha = alpha if alpha is not None else self.settings.rag.alpha

            logger.info(f"Starting {mode.value} retrieval for query: {query[:50]}...")

            # Determine retrieval strategy
            if mode == RetrievalMode.AUTO:
                mode = self._determine_auto_mode(query)

            # Execute retrieval based on mode
            if mode == RetrievalMode.VECTOR_ONLY:
                results = await self.dense_retriever.retrieve(
                    query=query,
                    top_k=top_k,
                    collection_name=collection_name,
                    filters=filters,
                )

            elif mode == RetrievalMode.BM25_ONLY:
                # Get corpus nodes for BM25
                corpus_nodes = await self._get_corpus_nodes(collection_name, filters)
                results = await self.sparse_retriever.retrieve(
                    query=query,
                    nodes=corpus_nodes,
                    top_k=top_k,
                    collection_name=collection_name,
                )

            elif mode in [RetrievalMode.HYBRID, RetrievalMode.SEMANTIC_HYBRID]:
                results = await self._hybrid_retrieve(
                    query=query,
                    dense_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                    alpha=alpha,
                    collection_name=collection_name,
                    filters=filters,
                    fusion_method=fusion_method,
                )

            else:
                raise RetrievalError(f"Unsupported retrieval mode: {mode}")

            # Limit final results
            results = results[:top_k]

            # Add timing metadata
            end_time = datetime.utcnow()
            retrieval_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms

            for node_with_score in results:
                node_with_score.node.metadata["retrieval_mode"] = mode.value
                node_with_score.node.metadata["retrieval_time_ms"] = retrieval_time
                node_with_score.node.metadata["alpha"] = alpha

            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.2f}ms")
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Retrieval failed: {e}", query=query, retrieval_type=mode.value)

    async def _hybrid_retrieve(
        self,
        query: str,
        dense_top_k: int,
        sparse_top_k: int,
        alpha: float,
        collection_name: Optional[str],
        filters: Optional[MetadataFilters],
        fusion_method: FusionMethod,
    ) -> List[NodeWithScore]:
        """Execute hybrid retrieval combining dense and sparse methods."""
        try:
            logger.info(f"Starting hybrid retrieval for query: '{query}' in collection: '{collection_name}'")

            # Execute both retrievals in parallel
            dense_task = self.dense_retriever.retrieve(
                query=query,
                top_k=dense_top_k,
                collection_name=collection_name,
                filters=filters,
            )

            # Get corpus nodes for BM25
            corpus_nodes = await self._get_corpus_nodes(collection_name, filters)
            sparse_task = self.sparse_retriever.retrieve(
                query=query,
                nodes=corpus_nodes,
                top_k=sparse_top_k,
                collection_name=collection_name,
            )

            # Wait for both retrievals to complete
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

            logger.debug(f"Dense retrieval: {len(dense_results)} results")
            logger.debug(f"Sparse retrieval: {len(sparse_results)} results")

            # Fuse results
            if fusion_method == FusionMethod.RRF:
                fused_results = HybridFusion.reciprocal_rank_fusion(
                    results_list=[dense_results, sparse_results],
                    weights=[alpha, 1.0 - alpha],
                )
            elif fusion_method == FusionMethod.WEIGHTED_SCORE:
                fused_results = HybridFusion.weighted_score_fusion(
                    results_list=[dense_results, sparse_results],
                    weights=[alpha, 1.0 - alpha],
                )
            else:
                raise RetrievalError(f"Unsupported fusion method: {fusion_method}")

            return fused_results

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise RetrievalError(f"Hybrid retrieval failed: {e}")

    async def _get_corpus_nodes(
        self,
        collection_name: Optional[str],
        filters: Optional[MetadataFilters],
    ) -> List[BaseNode]:
        """Get all nodes from the corpus for BM25 retrieval."""
        try:
            # Use ChromaDB's get() method to retrieve all documents efficiently
            collection = self.vector_store.get_collection(collection_name, create_if_not_exists=False)

            # Build get parameters
            get_params = {
                "limit": 10000,
                "include": ["documents", "metadatas"],
            }

            # Add metadata filters if provided
            if filters:
                where_clause = self.vector_store._build_where_clause(filters)
                if where_clause:
                    get_params["where"] = where_clause

            # Get all documents
            results = collection.get(**get_params)

            # Convert to BaseNode objects
            nodes = []
            if results and "ids" in results and results["ids"]:
                for i, node_id in enumerate(results["ids"]):
                    document = results["documents"][i] if results.get("documents") and i < len(results["documents"]) else ""
                    metadata = results["metadatas"][i] if results.get("metadatas") and i < len(results["metadatas"]) else {}

                    node = TextNode(
                        id_=node_id,
                        text=document,
                        metadata=metadata,
                    )
                    nodes.append(node)

            logger.info(f"Retrieved {len(nodes)} corpus nodes for BM25 indexing from collection '{collection_name}'")
            return nodes

        except Exception as e:
            logger.error(f"Failed to get corpus nodes: {e}", exc_info=True)
            raise RetrievalError(f"Failed to get corpus nodes: {e}")

    def _determine_auto_mode(self, query: str) -> RetrievalMode:
        """
        Automatically determine the best retrieval mode based on query characteristics.

        Args:
            query: Query string

        Returns:
            Recommended retrieval mode
        """
        # Simple heuristics for mode selection
        query_length = len(query.split())

        # Short queries often benefit from BM25
        if query_length <= 3:
            return RetrievalMode.BM25_ONLY

        # Long, complex queries benefit from semantic search
        if query_length >= 15:
            return RetrievalMode.VECTOR_ONLY

        # Medium queries benefit from hybrid approach
        return RetrievalMode.HYBRID

    def get_retrieval_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get retrieval statistics.

        Args:
            collection_name: Target collection

        Returns:
            Retrieval statistics
        """
        try:
            stats = self.vector_store.get_collection_stats(collection_name)

            # Add retrieval-specific stats
            stats.update({
                "dense_retriever_available": self.embedding_model is not None,
                "sparse_retrievers_cached": len(self.sparse_retriever._bm25_retrievers),
                "default_dense_top_k": self.settings.rag.dense_top_k,
                "default_sparse_top_k": self.settings.rag.sparse_top_k,
                "default_alpha": self.settings.rag.alpha,
            })

            return stats

        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {e}")
            raise RetrievalError(f"Failed to get retrieval stats: {e}")

    def clear_cache(self) -> None:
        """Clear cached retrievers and models."""
        try:
            self.sparse_retriever._bm25_retrievers.clear()
            logger.info("Cleared retrieval cache")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")