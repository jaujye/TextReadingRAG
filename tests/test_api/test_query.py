"""
Test script for query API endpoints.

This module tests the query functionality including:
- Basic RAG queries
- Query expansion
- Reranking
- Batch queries
- Streaming queries
- Document comparison
- Document summarization
"""

import json
import pytest
from typing import Generator
from unittest.mock import Mock, patch, AsyncMock

from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

from src.api.main import app
from src.api.models.requests import QueryType, RetrievalStrategy, RerankingModel
from src.core.config import get_settings
from src.rag.retrieval import RetrievalMode


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_retrieval_nodes() -> list[NodeWithScore]:
    """Create mock retrieval nodes for testing."""
    nodes = []
    for i in range(5):
        text_node = TextNode(
            text=f"This is test document content chunk {i}. It contains relevant information about the query topic.",
            id_=f"node-{i}",
            metadata={
                "document_id": f"doc-{i // 2}",
                "filename": f"test_document_{i // 2}.pdf",
                "page_number": i + 1,
                "chunk_index": i,
                "retrieval_method": "hybrid"
            }
        )
        node_with_score = NodeWithScore(node=text_node, score=0.9 - (i * 0.1))
        nodes.append(node_with_score)

    return nodes


class TestQueryEndpoints:
    """Test cases for query endpoints."""

    def test_basic_query_success(self, client: TestClient, mock_retrieval_nodes: list[NodeWithScore]):
        """
        Test successful basic RAG query.

        Verifies:
        - HTTP 200 response
        - Answer is generated
        - Retrieved documents are returned
        - Processing times are included
        """
        print("\n=== Testing Basic Query ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.get_query_expansion_service") as mock_expansion, \
             patch("src.api.endpoints.query.get_reranking_service") as mock_reranking, \
             patch("src.api.endpoints.query.OpenAI") as mock_llm:

            # Mock retrieval service
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve = AsyncMock(return_value=mock_retrieval_nodes)
            mock_retrieval.return_value = mock_retrieval_instance

            # Mock query expansion service
            mock_expansion_instance = Mock()
            mock_expansion_instance.expand_query = AsyncMock(return_value={
                "original": ["test query"],
                "llm": ["expanded query 1", "expanded query 2"]
            })
            mock_expansion.return_value = mock_expansion_instance

            # Mock reranking service
            mock_reranking_instance = Mock()
            mock_reranking_instance.rerank = AsyncMock(return_value=mock_retrieval_nodes[:3])
            mock_reranking.return_value = mock_reranking_instance

            # Mock LLM
            mock_llm_instance = Mock()
            mock_completion = Mock()
            mock_completion.text = "This is a generated answer based on the retrieved documents. The information shows that the query topic is well covered in the available content."
            mock_llm_instance.acomplete = AsyncMock(return_value=mock_completion)
            mock_llm.return_value = mock_llm_instance

            # Prepare query request
            query_data = {
                "query": "What is the main topic of the documents?",
                "query_type": QueryType.QUESTION_ANSWER.value,
                "retrieval_strategy": RetrievalStrategy.HYBRID.value,
                "top_k": 5,
                "enable_query_expansion": True,
                "enable_reranking": True,
                "reranking_model": RerankingModel.BGE_RERANKER_LARGE.value
            }

            response = client.post("/api/query/", json=query_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()

            # Verify response structure
            assert "query" in response_data
            assert "answer" in response_data
            assert "retrieved_documents" in response_data
            assert "processing_time_ms" in response_data
            assert "retrieval_time_ms" in response_data
            assert "generation_time_ms" in response_data

            # Verify content
            assert response_data["query"] == query_data["query"]
            assert len(response_data["answer"]) > 0
            assert len(response_data["retrieved_documents"]) > 0
            assert response_data["query_expansion_enabled"] is True
            assert response_data["reranking_enabled"] is True

            print(f"✓ Query processed successfully")
            print(f"  Answer length: {len(response_data['answer'])} characters")
            print(f"  Documents retrieved: {response_data['total_retrieved']}")
            print(f"  Processing time: {response_data['processing_time_ms']:.2f}ms")


    def test_query_without_expansion(self, client: TestClient, mock_retrieval_nodes: list[NodeWithScore]):
        """
        Test query without query expansion.

        Verifies:
        - Query works without expansion
        - expansion_enabled is False
        - No expanded queries in response
        """
        print("\n=== Testing Query Without Expansion ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.get_query_expansion_service") as mock_expansion, \
             patch("src.api.endpoints.query.get_reranking_service") as mock_reranking, \
             patch("src.api.endpoints.query.OpenAI") as mock_llm:

            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve = AsyncMock(return_value=mock_retrieval_nodes)
            mock_retrieval.return_value = mock_retrieval_instance

            mock_expansion_instance = Mock()
            mock_expansion.return_value = mock_expansion_instance

            mock_reranking_instance = Mock()
            mock_reranking_instance.rerank = AsyncMock(return_value=mock_retrieval_nodes[:3])
            mock_reranking.return_value = mock_reranking_instance

            mock_llm_instance = Mock()
            mock_completion = Mock()
            mock_completion.text = "Generated answer without query expansion."
            mock_llm_instance.acomplete = AsyncMock(return_value=mock_completion)
            mock_llm.return_value = mock_llm_instance

            query_data = {
                "query": "Test query without expansion",
                "enable_query_expansion": False,
                "enable_reranking": True
            }

            response = client.post("/api/query/", json=query_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["query_expansion_enabled"] is False
            assert response_data["expanded_queries"] is None

            # Verify expansion service was not called
            mock_expansion_instance.expand_query.assert_not_called()

            print("✓ Query without expansion processed successfully")


    def test_query_without_reranking(self, client: TestClient, mock_retrieval_nodes: list[NodeWithScore]):
        """
        Test query without reranking.

        Verifies:
        - Query works without reranking
        - reranking_enabled is False
        - Results are still returned
        """
        print("\n=== Testing Query Without Reranking ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.get_query_expansion_service") as mock_expansion, \
             patch("src.api.endpoints.query.get_reranking_service") as mock_reranking, \
             patch("src.api.endpoints.query.OpenAI") as mock_llm:

            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve = AsyncMock(return_value=mock_retrieval_nodes)
            mock_retrieval.return_value = mock_retrieval_instance

            mock_expansion_instance = Mock()
            mock_expansion.return_value = mock_expansion_instance

            mock_reranking_instance = Mock()
            mock_reranking.return_value = mock_reranking_instance

            mock_llm_instance = Mock()
            mock_completion = Mock()
            mock_completion.text = "Generated answer without reranking."
            mock_llm_instance.acomplete = AsyncMock(return_value=mock_completion)
            mock_llm.return_value = mock_llm_instance

            query_data = {
                "query": "Test query without reranking",
                "enable_query_expansion": False,
                "enable_reranking": False
            }

            response = client.post("/api/query/", json=query_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["reranking_enabled"] is False
            assert response_data["reranking_time_ms"] is None

            # Verify reranking service was not called
            mock_reranking_instance.rerank.assert_not_called()

            print("✓ Query without reranking processed successfully")


    def test_query_with_different_retrieval_strategies(self, client: TestClient, mock_retrieval_nodes: list[NodeWithScore]):
        """
        Test different retrieval strategies.

        Verifies:
        - VECTOR_ONLY strategy works
        - BM25_ONLY strategy works
        - HYBRID strategy works
        """
        print("\n=== Testing Different Retrieval Strategies ===")

        strategies = [
            RetrievalStrategy.VECTOR_ONLY,
            RetrievalStrategy.BM25_ONLY,
            RetrievalStrategy.HYBRID
        ]

        for strategy in strategies:
            print(f"\nTesting strategy: {strategy.value}")

            with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
                 patch("src.api.endpoints.query.get_query_expansion_service") as mock_expansion, \
                 patch("src.api.endpoints.query.get_reranking_service") as mock_reranking, \
                 patch("src.api.endpoints.query.OpenAI") as mock_llm:

                mock_retrieval_instance = Mock()
                mock_retrieval_instance.retrieve = AsyncMock(return_value=mock_retrieval_nodes)
                mock_retrieval.return_value = mock_retrieval_instance

                mock_expansion_instance = Mock()
                mock_expansion.return_value = mock_expansion_instance

                mock_reranking_instance = Mock()
                mock_reranking.return_value = mock_reranking_instance

                mock_llm_instance = Mock()
                mock_completion = Mock()
                mock_completion.text = f"Answer using {strategy.value} strategy."
                mock_llm_instance.acomplete = AsyncMock(return_value=mock_completion)
                mock_llm.return_value = mock_llm_instance

                query_data = {
                    "query": f"Test query with {strategy.value}",
                    "retrieval_strategy": strategy.value,
                    "enable_query_expansion": False,
                    "enable_reranking": False
                }

                response = client.post("/api/query/", json=query_data)

                print(f"  Status Code: {response.status_code}")

                assert response.status_code == 200
                response_data = response.json()
                assert response_data["retrieval_strategy"] == strategy.value

                print(f"  ✓ Strategy {strategy.value} works correctly")


    def test_batch_query(self, client: TestClient, mock_retrieval_nodes: list[NodeWithScore]):
        """
        Test batch query processing.

        Verifies:
        - Multiple queries are processed
        - Batch ID is returned
        - All queries succeed
        """
        print("\n=== Testing Batch Query ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.get_query_expansion_service") as mock_expansion, \
             patch("src.api.endpoints.query.get_reranking_service") as mock_reranking, \
             patch("src.api.endpoints.query.OpenAI") as mock_llm:

            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve = AsyncMock(return_value=mock_retrieval_nodes)
            mock_retrieval.return_value = mock_retrieval_instance

            mock_expansion_instance = Mock()
            mock_expansion.return_value = mock_expansion_instance

            mock_reranking_instance = Mock()
            mock_reranking.return_value = mock_reranking_instance

            mock_llm_instance = Mock()
            mock_completion = Mock()
            mock_completion.text = "Batch query answer."
            mock_llm_instance.acomplete = AsyncMock(return_value=mock_completion)
            mock_llm.return_value = mock_llm_instance

            batch_data = {
                "queries": [
                    {
                        "query": "First query",
                        "enable_query_expansion": False,
                        "enable_reranking": False
                    },
                    {
                        "query": "Second query",
                        "enable_query_expansion": False,
                        "enable_reranking": False
                    },
                    {
                        "query": "Third query",
                        "enable_query_expansion": False,
                        "enable_reranking": False
                    }
                ],
                "parallel_processing": True
            }

            response = client.post("/api/query/batch", json=batch_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()

            assert "batch_id" in response_data
            assert response_data["total_queries"] == 3
            assert response_data["completed_queries"] == 3
            assert response_data["failed_queries"] == 0
            assert len(response_data["results"]) == 3

            print(f"✓ Batch query processed successfully")
            print(f"  Batch ID: {response_data['batch_id']}")
            print(f"  Completed: {response_data['completed_queries']}/{response_data['total_queries']}")


    def test_query_no_results(self, client: TestClient):
        """
        Test query with no retrieval results.

        Verifies:
        - Query still succeeds with empty results
        - Appropriate message is returned
        """
        print("\n=== Testing Query with No Results ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.get_query_expansion_service") as mock_expansion, \
             patch("src.api.endpoints.query.get_reranking_service") as mock_reranking, \
             patch("src.api.endpoints.query.OpenAI") as mock_llm:

            # Return empty list for retrieval
            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve = AsyncMock(return_value=[])
            mock_retrieval.return_value = mock_retrieval_instance

            mock_expansion_instance = Mock()
            mock_expansion.return_value = mock_expansion_instance

            mock_reranking_instance = Mock()
            mock_reranking.return_value = mock_reranking_instance

            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance

            query_data = {
                "query": "Query with no results",
                "enable_query_expansion": False,
                "enable_reranking": False
            }

            response = client.post("/api/query/", json=query_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["total_retrieved"] == 0
            assert "couldn't find" in response_data["answer"].lower() or "no relevant" in response_data["answer"].lower()

            print("✓ Query with no results handled correctly")


    def test_document_comparison(self, client: TestClient, mock_retrieval_nodes: list[NodeWithScore]):
        """
        Test document comparison endpoint.

        Verifies:
        - Comparison is performed
        - Similarities and differences are returned
        """
        print("\n=== Testing Document Comparison ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.OpenAI") as mock_llm:

            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve = AsyncMock(return_value=mock_retrieval_nodes)
            mock_retrieval.return_value = mock_retrieval_instance

            mock_llm_instance = Mock()
            mock_completion = Mock()
            comparison_result = {
                "similarities": {
                    "content": "Both documents discuss similar topics",
                    "structure": "Similar document structure"
                },
                "differences": {
                    "focus": "Different focus areas",
                    "details": "Different level of detail"
                },
                "summary": "Documents are related but have distinct perspectives"
            }
            mock_completion.text = json.dumps(comparison_result)
            mock_llm_instance.acomplete = AsyncMock(return_value=mock_completion)
            mock_llm.return_value = mock_llm_instance

            comparison_data = {
                "document_ids": ["doc-1", "doc-2"],
                "comparison_aspects": ["content", "themes", "conclusions"]
            }

            response = client.post("/api/query/compare", json=comparison_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()

            assert "comparison_id" in response_data
            assert "similarities" in response_data
            assert "differences" in response_data
            assert "summary" in response_data
            assert len(response_data["document_ids"]) == 2

            print("✓ Document comparison completed successfully")


    def test_document_summarization(self, client: TestClient, mock_retrieval_nodes: list[NodeWithScore]):
        """
        Test document summarization endpoint.

        Verifies:
        - Summary is generated
        - Key points are extracted
        - Source references are included
        """
        print("\n=== Testing Document Summarization ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.OpenAI") as mock_llm:

            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve = AsyncMock(return_value=mock_retrieval_nodes)
            mock_retrieval.return_value = mock_retrieval_instance

            mock_llm_instance = Mock()
            mock_completion = Mock()
            mock_completion.text = """
This is a comprehensive summary of the documents.

Key Points:
• First important point about the content
• Second critical insight from the analysis
• Third significant finding
• Fourth notable observation

The documents cover multiple aspects of the topic with detailed information.
"""
            mock_llm_instance.acomplete = AsyncMock(return_value=mock_completion)
            mock_llm.return_value = mock_llm_instance

            summarization_data = {
                "document_ids": ["doc-1", "doc-2"],
                "summary_type": "extractive",
                "max_length": 500,
                "focus_areas": ["main_topics", "key_findings"]
            }

            response = client.post("/api/query/summarize", json=summarization_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()

            assert "summary_id" in response_data
            assert "summary" in response_data
            assert "key_points" in response_data
            assert len(response_data["summary"]) > 0
            assert len(response_data["key_points"]) > 0

            print("✓ Document summarization completed successfully")
            print(f"  Summary length: {response_data['summary_length']} characters")
            print(f"  Key points: {len(response_data['key_points'])}")


    def test_query_stats(self, client: TestClient):
        """
        Test query statistics endpoint.

        Verifies:
        - Statistics are returned
        - Required fields are present
        """
        print("\n=== Testing Query Statistics ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.get_query_expansion_service") as mock_expansion, \
             patch("src.api.endpoints.query.get_reranking_service") as mock_reranking:

            mock_retrieval_instance = Mock()
            mock_retrieval_instance.get_retrieval_stats = Mock(return_value={
                "total_queries": 100,
                "average_retrieval_time": 150.5
            })
            mock_retrieval.return_value = mock_retrieval_instance

            mock_expansion_instance = Mock()
            mock_expansion_instance.get_expansion_stats = Mock(return_value={
                "total_expansions": 50
            })
            mock_expansion.return_value = mock_expansion_instance

            mock_reranking_instance = Mock()
            mock_reranking_instance.get_reranking_stats = Mock(return_value={
                "total_reranks": 75
            })
            mock_reranking.return_value = mock_reranking_instance

            response = client.get("/api/query/stats")

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()

            assert "retrieval_stats" in response_data
            assert "expansion_stats" in response_data
            assert "reranking_stats" in response_data
            assert "system_config" in response_data

            print("✓ Query statistics retrieved successfully")


    def test_query_validation_errors(self, client: TestClient):
        """
        Test query validation with invalid parameters.

        Verifies:
        - Validation errors are caught
        - Appropriate error messages returned
        """
        print("\n=== Testing Query Validation Errors ===")

        # Test empty query
        print("\n  Testing empty query...")
        response = client.post("/api/query/", json={"query": ""})
        print(f"  Status Code: {response.status_code}")
        assert response.status_code == 422
        print("  ✓ Empty query rejected")

        # Test invalid top_k
        print("\n  Testing invalid top_k...")
        response = client.post("/api/query/", json={
            "query": "test",
            "top_k": 0
        })
        print(f"  Status Code: {response.status_code}")
        assert response.status_code == 422
        print("  ✓ Invalid top_k rejected")

        # Test invalid alpha
        print("\n  Testing invalid alpha...")
        response = client.post("/api/query/", json={
            "query": "test",
            "alpha": 1.5
        })
        print(f"  Status Code: {response.status_code}")
        assert response.status_code == 422
        print("  ✓ Invalid alpha rejected")

        print("\n✓ All validation errors handled correctly")


    def test_query_with_custom_parameters(self, client: TestClient, mock_retrieval_nodes: list[NodeWithScore]):
        """
        Test query with custom parameters.

        Verifies:
        - Custom parameters are accepted
        - Parameters affect query processing
        """
        print("\n=== Testing Query with Custom Parameters ===")

        with patch("src.api.endpoints.query.get_retrieval_service") as mock_retrieval, \
             patch("src.api.endpoints.query.get_query_expansion_service") as mock_expansion, \
             patch("src.api.endpoints.query.get_reranking_service") as mock_reranking, \
             patch("src.api.endpoints.query.OpenAI") as mock_llm:

            mock_retrieval_instance = Mock()
            mock_retrieval_instance.retrieve = AsyncMock(return_value=mock_retrieval_nodes)
            mock_retrieval.return_value = mock_retrieval_instance

            mock_expansion_instance = Mock()
            mock_expansion.return_value = mock_expansion_instance

            mock_reranking_instance = Mock()
            mock_reranking_instance.rerank = AsyncMock(return_value=mock_retrieval_nodes[:3])
            mock_reranking.return_value = mock_reranking_instance

            mock_llm_instance = Mock()
            mock_completion = Mock()
            mock_completion.text = "Custom answer."
            mock_llm_instance.acomplete = AsyncMock(return_value=mock_completion)
            mock_llm.return_value = mock_llm_instance

            query_data = {
                "query": "Test with custom params",
                "top_k": 3,
                "dense_top_k": 15,
                "sparse_top_k": 15,
                "alpha": 0.7,
                "temperature": 0.3,
                "max_response_length": 1000,
                "enable_query_expansion": False,
                "enable_reranking": True,
                "rerank_top_n": 5
            }

            response = client.post("/api/query/", json=query_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

            assert response.status_code == 200
            response_data = response.json()

            # Verify custom parameters were used
            assert response_data["retrieval_strategy"] == "hybrid"  # default
            assert len(response_data["retrieved_documents"]) <= 3  # top_k limit

            print("✓ Custom parameters processed successfully")


if __name__ == "__main__":
    """
    Run tests with verbose output.

    Usage:
        python -m pytest tests/test_api/test_query.py -v -s
    """
    pytest.main([__file__, "-v", "-s"])
