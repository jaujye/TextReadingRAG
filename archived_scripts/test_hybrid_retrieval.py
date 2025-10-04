#!/usr/bin/env python3
"""Test hybrid retrieval directly."""

import asyncio
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.rag.retrieval import HybridRetrievalService, RetrievalMode
from src.rag.vector_store import ChromaVectorStore
from src.core.config import Settings


async def test_hybrid_retrieval():
    """Test hybrid retrieval."""
    print("\n=== Testing Hybrid Retrieval ===\n")

    settings = Settings()
    print(f"ChromaDB host: {settings.rag.chroma_host}:{settings.rag.chroma_port}")
    print(f"Collection: test_collection\n")

    vector_store = ChromaVectorStore(
        host=settings.rag.chroma_host,
        port=settings.rag.chroma_port,
        persist_directory=settings.rag.chroma_persist_directory,
        settings=settings
    )

    service = HybridRetrievalService(vector_store=vector_store, settings=settings)

    query = "test document"
    print(f"Query: '{query}'")
    print(f"Mode: HYBRID\n")

    try:
        results = await service.retrieve(
            query=query,
            mode=RetrievalMode.HYBRID,
            collection_name='test_collection'
        )

        print(f"\n✓ Retrieved {len(results)} results\n")

        for i, r in enumerate(results):
            print(f"{i+1}. Score: {r.score:.4f}")
            print(f"   Text: {r.node.text[:150]}...")
            print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_hybrid_retrieval())
