"""Reranking strategies for improving retrieval relevance."""

import asyncio
import logging
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from datetime import datetime

from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI

try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    FlagEmbeddingReranker = None

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None

from src.core.config import Settings
from src.core.exceptions import RerankingError, LLMError

logger = logging.getLogger(__name__)


class RerankingModel(str, Enum):
    """Available reranking models."""

    BGE_RERANKER_LARGE = "BAAI/bge-reranker-large"
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    BGE_RERANKER_SMALL = "BAAI/bge-reranker-small"
    LLM_RERANK = "llm-rerank"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    COLBERT = "colbert"
    MULTI_STAGE = "multi-stage"


class RerankingStrategy(str, Enum):
    """Reranking strategies."""

    SINGLE_MODEL = "single_model"
    ENSEMBLE = "ensemble"
    CASCADE = "cascade"
    ADAPTIVE = "adaptive"


class CrossEncoderReranker:
    """Cross-encoder based reranking using BGE and similar models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        top_n: int = 3,
        device: Optional[str] = None,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            top_n: Number of top results to return
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name
        self.top_n = top_n
        self.device = device

        # Initialize reranker based on available libraries
        if FLAG_EMBEDDING_AVAILABLE and "bge-reranker" in model_name.lower():
            try:
                self.reranker = FlagEmbeddingReranker(
                    model=model_name,
                    top_n=top_n,
                )
                self.reranker_type = "flag_embedding"
                logger.info(f"Initialized FlagEmbedding reranker: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize FlagEmbedding reranker: {e}")
                self.reranker = None
                self.reranker_type = None
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(model_name, device=device)
                self.reranker_type = "cross_encoder"
                logger.info(f"Initialized CrossEncoder reranker: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize CrossEncoder reranker: {e}")
                self.cross_encoder = None
                self.reranker_type = None
        else:
            logger.warning("No reranking libraries available")
            self.reranker = None
            self.reranker_type = None

    async def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """
        Rerank nodes using cross-encoder model.

        Args:
            query: Query string
            nodes: List of nodes to rerank

        Returns:
            Reranked nodes
        """
        try:
            if not nodes:
                return []

            if self.reranker_type == "flag_embedding":
                return await self._rerank_with_flag_embedding(query, nodes)
            elif self.reranker_type == "cross_encoder":
                return await self._rerank_with_cross_encoder(query, nodes)
            else:
                logger.warning("No reranker available, returning original order")
                return nodes[:self.top_n]

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            raise RerankingError(
                f"Cross-encoder reranking failed: {e}",
                reranker_type="cross_encoder",
                num_documents=len(nodes),
            )

    async def _rerank_with_flag_embedding(
        self,
        query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """Rerank using FlagEmbedding reranker."""
        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)

            # Apply reranking
            reranked_nodes = self.reranker.postprocess_nodes(nodes, query_bundle)

            # Add reranking metadata
            for i, node in enumerate(reranked_nodes):
                node.node.metadata["rerank_position"] = i + 1
                node.node.metadata["rerank_model"] = self.model_name
                node.node.metadata["original_score"] = getattr(node, "original_score", node.score)

            logger.debug(f"FlagEmbedding reranked {len(nodes)} -> {len(reranked_nodes)} nodes")
            return reranked_nodes

        except Exception as e:
            logger.error(f"FlagEmbedding reranking failed: {e}")
            raise RerankingError(f"FlagEmbedding reranking failed: {e}")

    async def _rerank_with_cross_encoder(
        self,
        query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """Rerank using sentence-transformers CrossEncoder."""
        try:
            # Prepare query-document pairs
            pairs = [(query, node.node.get_content()) for node in nodes]

            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)

            # Create new NodeWithScore objects with updated scores
            reranked_nodes = []
            for node, score in zip(nodes, scores):
                new_node = NodeWithScore(
                    node=node.node,
                    score=float(score),
                )
                new_node.node.metadata["original_score"] = node.score
                new_node.node.metadata["rerank_score"] = float(score)
                new_node.node.metadata["rerank_model"] = self.model_name
                reranked_nodes.append(new_node)

            # Sort by rerank score
            reranked_nodes.sort(key=lambda x: x.score, reverse=True)

            # Add position metadata
            for i, node in enumerate(reranked_nodes):
                node.node.metadata["rerank_position"] = i + 1

            # Return top N
            result = reranked_nodes[:self.top_n]
            logger.debug(f"CrossEncoder reranked {len(nodes)} -> {len(result)} nodes")
            return result

        except Exception as e:
            logger.error(f"CrossEncoder reranking failed: {e}")
            raise RerankingError(f"CrossEncoder reranking failed: {e}")


class LLMReranker:
    """LLM-based reranking using language models for relevance scoring."""

    def __init__(
        self,
        llm: Optional[OpenAI] = None,
        top_n: int = 3,
        choice_batch_size: int = 5,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize LLM reranker.

        Args:
            llm: Language model instance
            top_n: Number of top results to return
            choice_batch_size: Batch size for LLM processing
            settings: Application settings
        """
        self.top_n = top_n
        self.choice_batch_size = choice_batch_size
        self.settings = settings

        if llm:
            self.llm = llm
        elif settings:
            try:
                self.llm = OpenAI(
                    api_key=settings.llm.openai_api_key,
                    model=settings.llm.openai_model,
                    temperature=0.0,  # Use deterministic scoring
                )
            except Exception as e:
                if not settings.app.mock_llm_responses:
                    raise RerankingError(f"Failed to initialize LLM: {e}")
                logger.warning("Using mock LLM for development")
                self.llm = None
        else:
            self.llm = None

        # Initialize LlamaIndex LLM reranker if LLM is available
        if self.llm:
            self.llm_reranker = LLMRerank(
                llm=self.llm,
                top_n=top_n,
                choice_batch_size=choice_batch_size,
            )

    async def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """
        Rerank nodes using LLM.

        Args:
            query: Query string
            nodes: List of nodes to rerank

        Returns:
            Reranked nodes
        """
        try:
            if not self.llm:
                logger.warning("No LLM available for reranking")
                return nodes[:self.top_n]

            if not nodes:
                return []

            # Create query bundle
            query_bundle = QueryBundle(query_str=query)

            # Apply LLM reranking
            reranked_nodes = self.llm_reranker.postprocess_nodes(nodes, query_bundle)

            # Add reranking metadata
            for i, node in enumerate(reranked_nodes):
                node.node.metadata["rerank_position"] = i + 1
                node.node.metadata["rerank_model"] = "llm_rerank"
                node.node.metadata["llm_model"] = self.settings.llm.openai_model if self.settings else "unknown"
                if not hasattr(node.node.metadata, "original_score"):
                    node.node.metadata["original_score"] = getattr(node, "original_score", node.score)

            logger.debug(f"LLM reranked {len(nodes)} -> {len(reranked_nodes)} nodes")
            return reranked_nodes

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            raise RerankingError(
                f"LLM reranking failed: {e}",
                reranker_type="llm",
                num_documents=len(nodes),
            )


class RerankingService:
    """Main service for managing reranking strategies."""

    def __init__(self, settings: Settings):
        """
        Initialize reranking service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._rerankers: Dict[str, Any] = {}

        # Initialize available rerankers
        self._initialize_rerankers()

    def _initialize_rerankers(self):
        """Initialize available reranking models."""
        try:
            # Cross-encoder reranker
            if FLAG_EMBEDDING_AVAILABLE or SENTENCE_TRANSFORMERS_AVAILABLE:
                self._rerankers["cross_encoder"] = CrossEncoderReranker(
                    model_name=self.settings.rag.rerank_model,
                    top_n=self.settings.rag.rerank_top_n,
                )

            # LLM reranker
            if self.settings.rag.use_llm_rerank:
                self._rerankers["llm"] = LLMReranker(
                    top_n=self.settings.rag.rerank_top_n,
                    settings=self.settings,
                )

            logger.info(f"Initialized {len(self._rerankers)} rerankers")

        except Exception as e:
            logger.error(f"Failed to initialize rerankers: {e}")

    async def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        model: Optional[RerankingModel] = None,
        top_n: Optional[int] = None,
        strategy: RerankingStrategy = RerankingStrategy.SINGLE_MODEL,
    ) -> List[NodeWithScore]:
        """
        Rerank nodes using specified model and strategy.

        Args:
            query: Query string
            nodes: List of nodes to rerank
            model: Reranking model to use
            top_n: Number of top results to return
            strategy: Reranking strategy

        Returns:
            Reranked nodes
        """
        try:
            if not nodes:
                return []

            start_time = datetime.utcnow()
            top_n = top_n or self.settings.rag.rerank_top_n

            # Determine model
            if model is None:
                model = RerankingModel(self.settings.rag.rerank_model)

            logger.info(f"Reranking {len(nodes)} nodes with {model.value}")

            # Apply reranking strategy
            if strategy == RerankingStrategy.SINGLE_MODEL:
                reranked_nodes = await self._single_model_rerank(query, nodes, model, top_n)
            elif strategy == RerankingStrategy.ENSEMBLE:
                reranked_nodes = await self._ensemble_rerank(query, nodes, top_n)
            else:
                logger.warning(f"Strategy {strategy} not implemented, using single model")
                reranked_nodes = await self._single_model_rerank(query, nodes, model, top_n)

            # Add timing metadata
            end_time = datetime.utcnow()
            rerank_time = (end_time - start_time).total_seconds() * 1000

            for node in reranked_nodes:
                node.node.metadata["rerank_time_ms"] = rerank_time
                node.node.metadata["rerank_strategy"] = strategy.value

            logger.info(f"Reranked {len(nodes)} -> {len(reranked_nodes)} nodes in {rerank_time:.2f}ms")
            return reranked_nodes

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise RerankingError(f"Reranking failed: {e}")

    async def _single_model_rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        model: RerankingModel,
        top_n: int,
    ) -> List[NodeWithScore]:
        """Apply single model reranking."""
        if model == RerankingModel.LLM_RERANK and "llm" in self._rerankers:
            return await self._rerankers["llm"].rerank(query, nodes)
        elif "cross_encoder" in self._rerankers:
            return await self._rerankers["cross_encoder"].rerank(query, nodes)
        else:
            logger.warning("No suitable reranker available")
            return nodes[:top_n]

    async def _ensemble_rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: int,
    ) -> List[NodeWithScore]:
        """Apply ensemble reranking."""
        # Collect available rerankers with weights
        rerankers = []
        if "cross_encoder" in self._rerankers:
            rerankers.append(("cross_encoder", self._rerankers["cross_encoder"], 0.7))
        if "llm" in self._rerankers:
            rerankers.append(("llm", self._rerankers["llm"], 0.3))

        if not rerankers:
            logger.warning("No rerankers available for ensemble")
            return nodes[:top_n]

        # Create ensemble reranker
        ensemble = EnsembleReranker(
            rerankers=rerankers,
            top_n=top_n,
            fusion_method="weighted_average",
        )

        return await ensemble.rerank(query, nodes)

    def get_available_models(self) -> List[str]:
        """Get list of available reranking models."""
        models = []
        if "cross_encoder" in self._rerankers:
            models.extend([
                RerankingModel.BGE_RERANKER_LARGE.value,
                RerankingModel.BGE_RERANKER_BASE.value,
                RerankingModel.SENTENCE_TRANSFORMER.value,
            ])
        if "llm" in self._rerankers:
            models.append(RerankingModel.LLM_RERANK.value)

        return models

    def get_reranking_stats(self) -> Dict[str, Any]:
        """Get reranking service statistics."""
        return {
            "available_rerankers": list(self._rerankers.keys()),
            "default_model": self.settings.rag.rerank_model,
            "default_top_n": self.settings.rag.rerank_top_n,
            "llm_rerank_enabled": self.settings.rag.use_llm_rerank,
            "cross_encoder_available": FLAG_EMBEDDING_AVAILABLE or SENTENCE_TRANSFORMERS_AVAILABLE,
            "flag_embedding_available": FLAG_EMBEDDING_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
        }