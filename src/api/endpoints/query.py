"""RAG query API endpoints."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    BackgroundTasks,
    Query as QueryParam,
)
from fastapi.responses import StreamingResponse
from llama_index.llms.openai import OpenAI
from llama_index.core.llms.llm import LLM

from src.api.models.requests import (
    QueryRequest,
    BatchQueryRequest,
    AdvancedQueryRequest,
    DocumentComparisonRequest,
    SummarizationRequest,
)
from src.api.models.responses import (
    QueryResponse,
    BatchQueryResponse,
    DocumentComparisonResponse,
    SummarizationResponse,
    RetrievalResult,
    RetrievalSource,
)
from src.core.config import Settings
from src.core.dependencies import (
    get_settings_dependency,
    get_retrieval_service,
    get_query_expansion_service,
    get_reranking_service,
)
from src.core.exceptions import (
    RetrievalError,
    QueryExpansionError,
    RerankingError,
    LLMError,
)
from src.rag.retrieval import HybridRetrievalService, RetrievalMode
from src.rag.query_expansion import QueryExpansionService
from src.rag.reranking import RerankingService, RerankingModel

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Query documents",
    description="Perform a RAG query against indexed documents",
)
async def query_documents(
    request: QueryRequest,
    settings: Settings = Depends(get_settings_dependency),
    retrieval_service: HybridRetrievalService = Depends(get_retrieval_service),
    query_expansion_service: QueryExpansionService = Depends(get_query_expansion_service),
    reranking_service: RerankingService = Depends(get_reranking_service),
) -> QueryResponse:
    """Perform a RAG query with hybrid retrieval, query expansion, and reranking."""
    try:
        start_time = time.time()

        # Initialize LLM for response generation
        llm = OpenAI(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.openai_model,
            temperature=request.temperature or settings.llm.openai_temperature,
            max_tokens=request.max_response_length or settings.llm.openai_max_tokens,
        )

        # Step 1: Query Expansion (if enabled)
        expanded_queries = {}
        expansion_time = 0.0

        if request.enable_query_expansion:
            expansion_start = time.time()
            try:
                expansion_methods = []
                if request.expansion_methods:
                    from src.rag.query_expansion import ExpansionMethod
                    expansion_methods = [ExpansionMethod(method) for method in request.expansion_methods]

                expanded_queries = await query_expansion_service.expand_query(
                    query=request.query,
                    methods=expansion_methods,
                    max_expansions=3,
                )
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
                expanded_queries = {"original": [request.query]}

            expansion_time = (time.time() - expansion_start) * 1000

        # Prepare queries for retrieval
        queries_to_search = [request.query]
        if expanded_queries:
            # Add expanded queries
            for method_queries in expanded_queries.values():
                if method_queries and method_queries != [request.query]:
                    queries_to_search.extend(method_queries[:2])  # Limit expanded queries

        # Step 2: Retrieval
        retrieval_start = time.time()

        # Convert retrieval strategy
        retrieval_mode = RetrievalMode(request.retrieval_strategy.value)

        # Perform retrieval for each query
        all_retrieved_nodes = []
        for query in queries_to_search:
            try:
                nodes = await retrieval_service.retrieve(
                    query=query,
                    mode=retrieval_mode,
                    top_k=request.top_k,
                    dense_top_k=request.dense_top_k,
                    sparse_top_k=request.sparse_top_k,
                    alpha=request.alpha,
                    collection_name=request.collection_name,
                )
                all_retrieved_nodes.extend(nodes)
            except Exception as e:
                logger.error(f"Retrieval failed for query '{query}': {e}")

        # Remove duplicates based on node ID
        seen_ids = set()
        unique_nodes = []
        for node in all_retrieved_nodes:
            if node.node.node_id not in seen_ids:
                unique_nodes.append(node)
                seen_ids.add(node.node.node_id)

        # Limit to top results
        retrieval_nodes = unique_nodes[:request.dense_top_k or 20]
        retrieval_time = (time.time() - retrieval_start) * 1000

        # Step 3: Reranking (if enabled)
        reranking_time = 0.0
        final_nodes = retrieval_nodes

        if request.enable_reranking and retrieval_nodes:
            reranking_start = time.time()
            try:
                reranking_model = RerankingModel(request.reranking_model.value)
                final_nodes = await reranking_service.rerank(
                    query=request.query,
                    nodes=retrieval_nodes,
                    model=reranking_model,
                    top_n=request.rerank_top_n,
                )
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                final_nodes = retrieval_nodes[:request.rerank_top_n or 5]

            reranking_time = (time.time() - reranking_start) * 1000

        # Limit final results
        final_nodes = final_nodes[:request.top_k]

        # Step 4: Generate Response
        generation_start = time.time()

        if final_nodes:
            # Prepare context from retrieved documents
            context_pieces = []
            for i, node_with_score in enumerate(final_nodes):
                context_pieces.append(f"[{i+1}] {node_with_score.node.get_content()}")

            context = "\n\n".join(context_pieces)

            # Generate response
            prompt = f"""Based on the following context, please answer the query.

Query: {request.query}

Context:
{context}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the query, please indicate that clearly."""

            response_text = await llm.acomplete(prompt)
            answer = response_text.text

        else:
            answer = "I couldn't find any relevant information to answer your query. Please try rephrasing your question or check if documents have been uploaded."

        generation_time = (time.time() - generation_start) * 1000

        # Step 5: Prepare Response
        retrieved_documents = []
        for i, node_with_score in enumerate(final_nodes):
            node = node_with_score.node
            source = RetrievalSource(
                document_id=node.metadata.get("document_id", node.node_id),
                chunk_id=node.node_id,
                filename=node.metadata.get("filename", "unknown"),
                page_number=node.metadata.get("page_number"),
                chunk_index=node.metadata.get("chunk_index"),
                content=node.get_content(),
                metadata=node.metadata,
            )

            retrieval_result = RetrievalResult(
                source=source,
                score=node_with_score.score,
                retrieval_method=node.metadata.get("retrieval_method", "unknown"),
                rank=i + 1,
                rerank_score=node.metadata.get("rerank_score"),
            )

            retrieved_documents.append(retrieval_result)

        # Calculate total processing time
        total_time = (time.time() - start_time) * 1000

        # Prepare expanded queries list
        expanded_query_list = []
        if expanded_queries:
            for method, queries in expanded_queries.items():
                if method != "original" and queries:
                    expanded_query_list.extend(queries)

        # Create response
        response = QueryResponse(
            query=request.query,
            answer=answer,
            query_type=request.query_type.value,
            retrieved_documents=retrieved_documents,
            total_retrieved=len(retrieved_documents),
            retrieval_strategy=request.retrieval_strategy.value,
            query_expansion_enabled=request.enable_query_expansion,
            expanded_queries=expanded_query_list if expanded_query_list else None,
            reranking_enabled=request.enable_reranking,
            reranking_model=request.reranking_model.value if request.enable_reranking else None,
            processing_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            reranking_time_ms=reranking_time if reranking_time > 0 else None,
            generation_time_ms=generation_time,
        )

        logger.info(f"Query processed in {total_time:.2f}ms: '{request.query[:50]}...'")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchQueryResponse,
    summary="Batch query documents",
    description="Process multiple queries in parallel",
)
async def batch_query_documents(
    request: BatchQueryRequest,
    settings: Settings = Depends(get_settings_dependency),
    retrieval_service: HybridRetrievalService = Depends(get_retrieval_service),
    query_expansion_service: QueryExpansionService = Depends(get_query_expansion_service),
    reranking_service: RerankingService = Depends(get_reranking_service),
) -> BatchQueryResponse:
    """Process multiple queries in batch."""
    try:
        start_time = time.time()
        batch_id = str(uuid.uuid4())

        # Process queries
        if request.parallel_processing:
            # Process in parallel
            tasks = [
                query_documents(
                    query_request,
                    settings,
                    retrieval_service,
                    query_expansion_service,
                    reranking_service,
                )
                for query_request in request.queries
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            results = []
            for query_request in request.queries:
                try:
                    result = await query_documents(
                        query_request,
                        settings,
                        retrieval_service,
                        query_expansion_service,
                        reranking_service,
                    )
                    results.append(result)
                except Exception as e:
                    results.append(e)

        # Separate successful and failed results
        successful_results = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "query_index": i,
                    "query": request.queries[i].query,
                    "error": str(result),
                    "error_type": type(result).__name__,
                })
            else:
                successful_results.append(result)

        batch_processing_time = (time.time() - start_time) * 1000

        response = BatchQueryResponse(
            batch_id=batch_id,
            total_queries=len(request.queries),
            completed_queries=len(successful_results),
            failed_queries=len(errors),
            results=successful_results,
            batch_processing_time_ms=batch_processing_time,
            errors=errors,
        )

        logger.info(f"Batch query completed: {len(successful_results)}/{len(request.queries)} successful")
        return response

    except Exception as e:
        logger.error(f"Batch query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query processing failed: {str(e)}",
        )


@router.post(
    "/stream",
    summary="Stream query response",
    description="Get streaming response for a query",
)
async def stream_query_documents(
    request: QueryRequest,
    settings: Settings = Depends(get_settings_dependency),
    retrieval_service: HybridRetrievalService = Depends(get_retrieval_service),
    query_expansion_service: QueryExpansionService = Depends(get_query_expansion_service),
    reranking_service: RerankingService = Depends(get_reranking_service),
):
    """Stream query response in real-time."""
    try:
        async def generate_streaming_response():
            # Initialize LLM for streaming
            llm = OpenAI(
                api_key=settings.llm.openai_api_key,
                model=settings.llm.openai_model,
                temperature=request.temperature or settings.llm.openai_temperature,
                max_tokens=request.max_response_length or settings.llm.openai_max_tokens,
            )

            # Step 1: Send initial status
            yield f"data: {{'event': 'status', 'data': {{'stage': 'starting', 'query': '{request.query}'}}}}\n\n"

            # Step 2: Query expansion
            if request.enable_query_expansion:
                yield f"data: {{'event': 'status', 'data': {{'stage': 'expanding_query'}}}}\n\n"

            # Step 3: Retrieval
            yield f"data: {{'event': 'status', 'data': {{'stage': 'retrieving_documents'}}}}\n\n"

            retrieval_mode = RetrievalMode(request.retrieval_strategy.value)
            retrieved_nodes = await retrieval_service.retrieve(
                query=request.query,
                mode=retrieval_mode,
                top_k=request.top_k,
                collection_name=request.collection_name,
            )

            yield f"data: {{'event': 'retrieved', 'data': {{'count': {len(retrieved_nodes)}}}}}\n\n"

            # Step 4: Reranking
            if request.enable_reranking and retrieved_nodes:
                yield f"data: {{'event': 'status', 'data': {{'stage': 'reranking_documents'}}}}\n\n"

                reranking_model = RerankingModel(request.reranking_model.value)
                final_nodes = await reranking_service.rerank(
                    query=request.query,
                    nodes=retrieved_nodes,
                    model=reranking_model,
                    top_n=request.rerank_top_n,
                )
            else:
                final_nodes = retrieved_nodes[:request.top_k]

            # Step 5: Generate streaming response
            yield f"data: {{'event': 'status', 'data': {{'stage': 'generating_answer'}}}}\n\n"

            if final_nodes:
                # Prepare context
                context_pieces = []
                for i, node_with_score in enumerate(final_nodes):
                    context_pieces.append(f"[{i+1}] {node_with_score.node.get_content()}")

                context = "\n\n".join(context_pieces)

                prompt = f"""Based on the following context, please answer the query.

Query: {request.query}

Context:
{context}

Please provide a comprehensive answer based on the context provided."""

                # Stream the response
                response_stream = await llm.astream_complete(prompt)
                async for chunk in response_stream:
                    if chunk.delta:
                        yield f"data: {{'event': 'token', 'data': {{'token': '{chunk.delta}'}}}}\n\n"

            else:
                yield f"data: {{'event': 'token', 'data': {{'token': 'No relevant information found.'}}}}\n\n"

            # Send completion event
            yield f"data: {{'event': 'completed', 'data': {{'sources': {len(final_nodes)}}}}}\n\n"

        return StreamingResponse(
            generate_streaming_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    except Exception as e:
        logger.error(f"Streaming query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming query failed: {str(e)}",
        )


@router.post(
    "/compare",
    response_model=DocumentComparisonResponse,
    summary="Compare documents",
    description="Compare multiple documents and analyze similarities/differences",
)
async def compare_documents(
    request: DocumentComparisonRequest,
    settings: Settings = Depends(get_settings_dependency),
    retrieval_service: HybridRetrievalService = Depends(get_retrieval_service),
) -> DocumentComparisonResponse:
    """Compare multiple documents."""
    try:
        start_time = time.time()
        comparison_id = str(uuid.uuid4())

        # Initialize LLM
        llm = OpenAI(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.openai_model,
            temperature=0.1,  # Lower temperature for more consistent comparisons
        )

        # Retrieve content for each document
        document_contents = {}
        for doc_id in request.document_ids:
            # Search for document content
            nodes = await retrieval_service.retrieve(
                query="",  # Empty query to get document content
                mode=RetrievalMode.VECTOR_ONLY,
                top_k=10,  # Get multiple chunks per document
            )

            # Filter nodes for this document
            doc_nodes = [
                node for node in nodes
                if node.node.metadata.get("document_id") == doc_id
            ]

            if doc_nodes:
                content = "\n\n".join([node.node.get_content() for node in doc_nodes])
                document_contents[doc_id] = content
            else:
                document_contents[doc_id] = ""

        # Generate comparison
        comparison_prompt = f"""Compare the following documents across these aspects: {', '.join(request.comparison_aspects)}.

Documents to compare:
"""

        for i, (doc_id, content) in enumerate(document_contents.items()):
            comparison_prompt += f"\nDocument {i+1} (ID: {doc_id}):\n{content[:2000]}...\n"

        comparison_prompt += f"""
Please provide a detailed comparison focusing on:
{chr(10).join([f"- {aspect}" for aspect in request.comparison_aspects])}

Format your response as JSON with the following structure:
{{
    "similarities": {{"aspect1": "description", "aspect2": "description"}},
    "differences": {{"aspect1": "description", "aspect2": "description"}},
    "summary": "Overall comparison summary"
}}
"""

        response = await llm.acomplete(comparison_prompt)

        # Parse the response (simplified - in production, use more robust parsing)
        try:
            import json
            comparison_data = json.loads(response.text)
            similarities = comparison_data.get("similarities", {})
            differences = comparison_data.get("differences", {})
            summary = comparison_data.get("summary", "Comparison completed.")
        except:
            # Fallback if JSON parsing fails
            similarities = {"content": "Documents show some similarities in structure and topics."}
            differences = {"content": "Documents differ in specific details and focus areas."}
            summary = "Comparison completed with basic analysis."

        processing_time = (time.time() - start_time) * 1000

        response = DocumentComparisonResponse(
            comparison_id=comparison_id,
            document_ids=request.document_ids,
            comparison_aspects=request.comparison_aspects,
            similarities=similarities,
            differences=differences,
            summary=summary,
            processing_time_ms=processing_time,
        )

        logger.info(f"Document comparison completed in {processing_time:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"Document comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document comparison failed: {str(e)}",
        )


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    summary="Summarize documents",
    description="Generate summaries of documents or document collections",
)
async def summarize_documents(
    request: SummarizationRequest,
    settings: Settings = Depends(get_settings_dependency),
    retrieval_service: HybridRetrievalService = Depends(get_retrieval_service),
) -> SummarizationResponse:
    """Generate document summaries."""
    try:
        start_time = time.time()
        summary_id = str(uuid.uuid4())

        # Initialize LLM
        llm = OpenAI(
            api_key=settings.llm.openai_api_key,
            model=settings.llm.openai_model,
            temperature=0.2,
        )

        # Retrieve document content
        if request.document_ids:
            # Summarize specific documents
            all_content = []
            source_references = []

            for doc_id in request.document_ids:
                nodes = await retrieval_service.retrieve(
                    query="",
                    mode=RetrievalMode.VECTOR_ONLY,
                    top_k=20,
                )

                doc_nodes = [
                    node for node in nodes
                    if node.node.metadata.get("document_id") == doc_id
                ]

                for node in doc_nodes:
                    all_content.append(node.node.get_content())
                    source_references.append(RetrievalSource(
                        document_id=doc_id,
                        chunk_id=node.node.node_id,
                        filename=node.node.metadata.get("filename", "unknown"),
                        content=node.node.get_content()[:200] + "...",
                        metadata=node.node.metadata,
                    ))

        else:
            # Summarize all documents (simplified approach)
            nodes = await retrieval_service.retrieve(
                query="overview summary",
                mode=RetrievalMode.HYBRID,
                top_k=50,
            )

            all_content = [node.node.get_content() for node in nodes]
            source_references = [
                RetrievalSource(
                    document_id=node.node.metadata.get("document_id", "unknown"),
                    chunk_id=node.node.node_id,
                    filename=node.node.metadata.get("filename", "unknown"),
                    content=node.node.get_content()[:200] + "...",
                    metadata=node.node.metadata,
                )
                for node in nodes
            ]

        # Generate summary
        combined_content = "\n\n".join(all_content)
        content_length = len(combined_content)

        summary_prompt = f"""Please provide a {request.summary_type} summary of the following content.

Content to summarize:
{combined_content[:8000]}...

Requirements:
- Maximum length: {request.max_length} words
- Summary type: {request.summary_type}
- Focus areas: {', '.join(request.focus_areas) if request.focus_areas else 'general overview'}

Please provide:
1. A comprehensive summary
2. Key points (as a bulleted list)
"""

        response = await llm.acomplete(summary_prompt)

        # Extract key points (simplified)
        summary_text = response.text
        key_points = []

        # Try to extract bullet points
        lines = summary_text.split('\n')
        for line in lines:
            if line.strip().startswith(('•', '-', '*', '1.', '2.', '3.')):
                key_points.append(line.strip())

        if not key_points:
            # Generate some basic key points
            key_points = ["Summary covers main topics and themes", "Analysis based on available content"]

        processing_time = (time.time() - start_time) * 1000
        compression_ratio = len(summary_text) / max(content_length, 1)

        response = SummarizationResponse(
            summary_id=summary_id,
            document_ids=request.document_ids or ["all"],
            summary_type=request.summary_type,
            summary=summary_text,
            key_points=key_points,
            source_references=source_references,
            summary_length=len(summary_text),
            compression_ratio=compression_ratio,
            processing_time_ms=processing_time,
        )

        logger.info(f"Document summarization completed in {processing_time:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"Document summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document summarization failed: {str(e)}",
        )


@router.get(
    "/stats",
    summary="Get query statistics",
    description="Get statistics about query performance and usage",
)
async def get_query_stats(
    collection_name: Optional[str] = QueryParam(None, description="Collection to get stats for"),
    retrieval_service: HybridRetrievalService = Depends(get_retrieval_service),
    query_expansion_service: QueryExpansionService = Depends(get_query_expansion_service),
    reranking_service: RerankingService = Depends(get_reranking_service),
    settings: Settings = Depends(get_settings_dependency),
) -> Dict[str, Any]:
    """Get comprehensive query and system statistics."""
    try:
        stats = {
            "retrieval_stats": retrieval_service.get_retrieval_stats(collection_name),
            "expansion_stats": query_expansion_service.get_expansion_stats(),
            "reranking_stats": reranking_service.get_reranking_stats(),
            "system_config": {
                "default_retrieval_mode": settings.rag.alpha,
                "default_top_k": settings.rag.hybrid_top_k,
                "query_expansion_enabled": settings.rag.enable_query_expansion,
                "reranking_enabled": settings.rag.use_llm_rerank,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return stats

    except Exception as e:
        logger.error(f"Failed to get query stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get query stats: {str(e)}",
        )