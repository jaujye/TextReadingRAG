"""Document management API endpoints."""

import asyncio
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
    BackgroundTasks,
    Query,
)
from fastapi.responses import JSONResponse

from src.api.models.requests import (
    DocumentProcessingOptions,
    DocumentUpdateRequest,
    CollectionCreateRequest,
)
from src.api.models.responses import (
    DocumentInfo,
    DocumentUploadResponse,
    BatchUploadResponse,
    ProcessingProgress,
    CollectionInfo,
    ProcessingStatus,
)
from src.core.config import Settings
from src.core.dependencies import (
    get_settings_dependency,
    get_document_ingestion_service,
    get_vector_store_client,
    get_cache_service,
    validate_file_upload,
)
from src.core.exceptions import (
    DocumentIngestionError,
    FileProcessingError,
    VectorStoreError,
    ValidationError,
    ResourceNotFoundError,
)
from src.rag.ingestion import DocumentIngestionService
from src.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Track processing jobs
processing_jobs: Dict[str, Dict[str, Any]] = {}


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a single document",
    description="Upload and process a single PDF document",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    collection_name: Optional[str] = Form(None, description="Target collection name"),
    chunk_size: Optional[int] = Form(None, description="Text chunk size"),
    chunk_overlap: Optional[int] = Form(None, description="Chunk overlap size"),
    use_llamaparse: bool = Form(False, description="Use LlamaParse for advanced processing"),
    metadata: Optional[str] = Form(None, description="JSON metadata string"),
    settings: Settings = Depends(get_settings_dependency),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
    cache_service = Depends(get_cache_service),
    _: bool = Depends(validate_file_upload),
) -> DocumentUploadResponse:
    """Upload and process a single document."""
    try:
        # Validate file type
        if not file.filename.lower().endswith(tuple(settings.app.allowed_extensions)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File type not supported. Allowed: {settings.app.allowed_extensions}",
            )

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Save uploaded file
        upload_path = Path(settings.app.upload_dir) / f"{document_id}_{file.filename}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)

        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            import json
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid JSON in metadata field",
                )

        # Prepare processing options
        processing_options = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "use_llamaparse": use_llamaparse,
            "metadata": doc_metadata,
        }

        # Create response
        response = DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            size=len(content),
            content_type=file.content_type or "application/pdf",
            collection_name=collection_name or settings.rag.chroma_collection_name,
            processing_status=ProcessingStatus.PENDING,
        )

        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document_id=document_id,
            file_path=str(upload_path),
            collection_name=collection_name,
            processing_options=processing_options,
            ingestion_service=ingestion_service,
            cache_service=cache_service,
        )

        logger.info(f"Document upload initiated: {document_id} - {file.filename}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )


@router.post(
    "/upload/batch",
    response_model=BatchUploadResponse,
    summary="Upload multiple documents",
    description="Upload and process multiple PDF documents in batch",
)
async def upload_documents_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF files to upload"),
    collection_name: Optional[str] = Form(None, description="Target collection name"),
    chunk_size: Optional[int] = Form(None, description="Text chunk size"),
    chunk_overlap: Optional[int] = Form(None, description="Chunk overlap size"),
    use_llamaparse: bool = Form(False, description="Use LlamaParse for advanced processing"),
    parallel_processing: bool = Form(True, description="Process files in parallel"),
    settings: Settings = Depends(get_settings_dependency),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
    cache_service = Depends(get_cache_service),
    _: bool = Depends(validate_file_upload),
) -> BatchUploadResponse:
    """Upload and process multiple documents in batch."""
    try:
        # Validate batch size
        max_files = settings.app.max_concurrent_uploads
        if len(files) > max_files:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Too many files. Maximum: {max_files}",
            )

        batch_id = str(uuid.uuid4())
        successful_uploads = []
        failed_uploads = []

        # Process each file
        for file in files:
            try:
                # Validate file type
                if not file.filename.lower().endswith(tuple(settings.app.allowed_extensions)):
                    failed_uploads.append({
                        "filename": file.filename,
                        "error": f"File type not supported. Allowed: {settings.app.allowed_extensions}",
                    })
                    continue

                # Generate document ID
                document_id = str(uuid.uuid4())

                # Save uploaded file
                upload_path = Path(settings.app.upload_dir) / f"{document_id}_{file.filename}"
                upload_path.parent.mkdir(parents=True, exist_ok=True)

                with open(upload_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)

                # Create upload response
                upload_response = DocumentUploadResponse(
                    document_id=document_id,
                    filename=file.filename,
                    size=len(content),
                    content_type=file.content_type or "application/pdf",
                    collection_name=collection_name or settings.rag.chroma_collection_name,
                    processing_status=ProcessingStatus.PENDING,
                )

                successful_uploads.append(upload_response)

                # Prepare processing options
                processing_options = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "use_llamaparse": use_llamaparse,
                    "metadata": {"batch_id": batch_id},
                }

                # Start background processing
                background_tasks.add_task(
                    process_document_background,
                    document_id=document_id,
                    file_path=str(upload_path),
                    collection_name=collection_name,
                    processing_options=processing_options,
                    ingestion_service=ingestion_service,
                    cache_service=cache_service,
                )

            except Exception as e:
                failed_uploads.append({
                    "filename": file.filename,
                    "error": str(e),
                })

        response = BatchUploadResponse(
            total_files=len(files),
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            batch_id=batch_id,
        )

        logger.info(f"Batch upload initiated: {batch_id} - {len(successful_uploads)}/{len(files)} files successful")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}",
        )


@router.get(
    "/progress/{document_id}",
    response_model=ProcessingProgress,
    summary="Get document processing progress",
    description="Check the processing status and progress of a document",
)
async def get_processing_progress(
    document_id: str,
    settings: Settings = Depends(get_settings_dependency),
) -> ProcessingProgress:
    """Get processing progress for a document."""
    try:
        if document_id not in processing_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or not being processed",
            )

        job_info = processing_jobs[document_id]

        return ProcessingProgress(
            document_id=document_id,
            status=ProcessingStatus(job_info["status"]),
            progress_percentage=job_info.get("progress", 0.0),
            current_stage=job_info.get("current_stage", "unknown"),
            stages_completed=job_info.get("stages_completed", []),
            estimated_completion=job_info.get("estimated_completion"),
            error_details=job_info.get("error_details"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get progress: {str(e)}",
        )


@router.get(
    "/list",
    response_model=List[DocumentInfo],
    summary="List documents",
    description="List all documents in a collection with optional filtering",
)
async def list_documents(
    collection_name: Optional[str] = Query(None, description="Collection name to filter by"),
    limit: int = Query(50, description="Maximum number of documents to return"),
    offset: int = Query(0, description="Number of documents to skip"),
    status_filter: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    vector_store: ChromaVectorStore = Depends(get_vector_store_client),
    settings: Settings = Depends(get_settings_dependency),
) -> List[DocumentInfo]:
    """List documents with optional filtering."""
    try:
        collection_name = collection_name or settings.rag.chroma_collection_name

        # Get documents from vector store
        # This is a simplified implementation - in production you might want a proper document index
        search_results = vector_store.search_by_metadata(
            metadata_filter={},  # Get all documents
            collection_name=collection_name,
            limit=limit + offset,
        )

        # Apply offset and limit
        paginated_results = search_results[offset:offset + limit]

        # Convert to DocumentInfo objects
        documents = []
        for result in paginated_results:
            metadata = result.get("metadata", {})

            doc_info = DocumentInfo(
                document_id=result["id"],
                filename=metadata.get("filename", "unknown"),
                content_type=metadata.get("mime_type", "application/pdf"),
                size=metadata.get("file_size", 0),
                upload_timestamp=datetime.fromisoformat(metadata.get("processed_at", datetime.utcnow().isoformat())),
                processing_status=ProcessingStatus.COMPLETED,  # Assume completed if in vector store
                chunk_count=metadata.get("chunk_count"),
                metadata=metadata,
                collection_name=collection_name,
            )

            # Apply status filter
            if status_filter is None or doc_info.processing_status == status_filter:
                documents.append(doc_info)

        logger.info(f"Listed {len(documents)} documents from collection {collection_name}")
        return documents

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )


@router.get(
    "/{document_id}",
    response_model=DocumentInfo,
    summary="Get document details",
    description="Get detailed information about a specific document",
)
async def get_document(
    document_id: str,
    collection_name: Optional[str] = Query(None, description="Collection name"),
    vector_store: ChromaVectorStore = Depends(get_vector_store_client),
    settings: Settings = Depends(get_settings_dependency),
) -> DocumentInfo:
    """Get detailed information about a document."""
    try:
        collection_name = collection_name or settings.rag.chroma_collection_name

        # Search for document by ID
        search_results = vector_store.search_by_metadata(
            metadata_filter={"document_id": document_id},
            collection_name=collection_name,
            limit=1,
        )

        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        result = search_results[0]
        metadata = result.get("metadata", {})

        doc_info = DocumentInfo(
            document_id=document_id,
            filename=metadata.get("filename", "unknown"),
            content_type=metadata.get("mime_type", "application/pdf"),
            size=metadata.get("file_size", 0),
            upload_timestamp=datetime.fromisoformat(metadata.get("processed_at", datetime.utcnow().isoformat())),
            processing_status=ProcessingStatus.COMPLETED,
            chunk_count=metadata.get("chunk_count"),
            metadata=metadata,
            collection_name=collection_name,
        )

        return doc_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}",
        )


@router.put(
    "/{document_id}",
    response_model=DocumentInfo,
    summary="Update document",
    description="Update document metadata or reprocess with new settings",
)
async def update_document(
    document_id: str,
    update_request: DocumentUpdateRequest,
    background_tasks: BackgroundTasks,
    collection_name: Optional[str] = Query(None, description="Collection name"),
    vector_store: ChromaVectorStore = Depends(get_vector_store_client),
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
    settings: Settings = Depends(get_settings_dependency),
) -> DocumentInfo:
    """Update document metadata or reprocess."""
    try:
        collection_name = collection_name or settings.rag.chroma_collection_name

        # Check if document exists
        search_results = vector_store.search_by_metadata(
            metadata_filter={"document_id": document_id},
            collection_name=collection_name,
            limit=1,
        )

        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        # If reprocessing is requested
        if update_request.reprocess:
            # Find original file path
            metadata = search_results[0].get("metadata", {})
            file_path = metadata.get("file_path")

            if not file_path or not os.path.exists(file_path):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Original file not found, cannot reprocess",
                )

            # Delete existing document from vector store
            vector_store.delete(document_id, collection_name=collection_name)

            # Start reprocessing
            processing_options = {}
            if update_request.processing_options:
                processing_options.update({
                    "chunk_size": update_request.processing_options.chunk_size,
                    "chunk_overlap": update_request.processing_options.chunk_overlap,
                    "use_llamaparse": update_request.processing_options.use_llamaparse,
                })

            # Merge metadata
            processing_options["metadata"] = {**metadata, **update_request.metadata}

            background_tasks.add_task(
                process_document_background,
                document_id=document_id,
                file_path=file_path,
                collection_name=collection_name,
                processing_options=processing_options,
                ingestion_service=ingestion_service,
            )

            # Return updated info
            return DocumentInfo(
                document_id=document_id,
                filename=metadata.get("filename", "unknown"),
                content_type=metadata.get("mime_type", "application/pdf"),
                size=metadata.get("file_size", 0),
                upload_timestamp=datetime.fromisoformat(metadata.get("processed_at", datetime.utcnow().isoformat())),
                processing_status=ProcessingStatus.PROCESSING,
                metadata={**metadata, **update_request.metadata},
                collection_name=collection_name,
            )

        else:
            # Just update metadata (not implemented in this simplified version)
            logger.info(f"Metadata update requested for {document_id} (not implemented)")
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Metadata-only updates not yet implemented",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}",
        )


@router.delete(
    "/{document_id}",
    summary="Delete document",
    description="Delete a document and all its associated data",
)
async def delete_document(
    document_id: str,
    collection_name: Optional[str] = Query(None, description="Collection name"),
    delete_file: bool = Query(True, description="Whether to delete the original file"),
    vector_store: ChromaVectorStore = Depends(get_vector_store_client),
    settings: Settings = Depends(get_settings_dependency),
) -> JSONResponse:
    """Delete a document and its associated data."""
    try:
        collection_name = collection_name or settings.rag.chroma_collection_name

        # Check if document exists
        search_results = vector_store.search_by_metadata(
            metadata_filter={"document_id": document_id},
            collection_name=collection_name,
            limit=1,
        )

        if not search_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        # Delete from vector store
        vector_store.delete(document_id, collection_name=collection_name)

        # Delete original file if requested
        if delete_file:
            metadata = search_results[0].get("metadata", {})
            file_path = metadata.get("file_path")
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")

        # Remove from processing jobs if present
        if document_id in processing_jobs:
            del processing_jobs[document_id]

        logger.info(f"Deleted document: {document_id}")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Document deleted successfully"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )


@router.post(
    "/collections",
    response_model=CollectionInfo,
    summary="Create collection",
    description="Create a new document collection",
)
async def create_collection(
    request: CollectionCreateRequest,
    vector_store: ChromaVectorStore = Depends(get_vector_store_client),
) -> CollectionInfo:
    """Create a new document collection."""
    try:
        # Create collection in vector store
        success = vector_store.create_collection(
            name=request.name,
            metadata=request.metadata,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create collection",
            )

        # Return collection info
        return CollectionInfo(
            name=request.name,
            description=request.description,
            document_count=0,
            total_chunks=0,
            created_timestamp=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            metadata=request.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create collection {request.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create collection: {str(e)}",
        )


@router.get(
    "/collections",
    response_model=List[CollectionInfo],
    summary="List collections",
    description="List all available document collections",
)
async def list_collections(
    vector_store: ChromaVectorStore = Depends(get_vector_store_client),
) -> List[CollectionInfo]:
    """List all available collections."""
    try:
        collection_names = vector_store.list_collections()

        collections = []
        for name in collection_names:
            try:
                stats = vector_store.get_collection_stats(name)

                collection_info = CollectionInfo(
                    name=name,
                    description=stats.get("metadata", {}).get("description"),
                    document_count=stats.get("count", 0),
                    total_chunks=stats.get("count", 0),  # Simplified - each node is a chunk
                    created_timestamp=datetime.utcnow(),  # Would need to be stored in metadata
                    last_updated=datetime.utcnow(),
                    metadata=stats.get("metadata", {}),
                )

                collections.append(collection_info)

            except Exception as e:
                logger.warning(f"Failed to get stats for collection {name}: {e}")
                continue

        logger.info(f"Listed {len(collections)} collections")
        return collections

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}",
        )


@router.delete(
    "/collections/{collection_name}",
    summary="Delete collection",
    description="Delete a collection and all its documents",
)
async def delete_collection(
    collection_name: str,
    vector_store: ChromaVectorStore = Depends(get_vector_store_client),
) -> JSONResponse:
    """Delete a collection and all its documents."""
    try:
        success = vector_store.delete_collection(collection_name)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found",
            )

        logger.info(f"Deleted collection: {collection_name}")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Collection '{collection_name}' deleted successfully"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {str(e)}",
        )


async def process_document_background(
    document_id: str,
    file_path: str,
    collection_name: Optional[str],
    processing_options: Dict[str, Any],
    ingestion_service: DocumentIngestionService,
    cache_service=None,
) -> None:
    """Background task for document processing."""
    try:
        # Update processing status
        processing_jobs[document_id] = {
            "status": ProcessingStatus.PROCESSING.value,
            "progress": 0.0,
            "current_stage": "starting",
            "stages_completed": [],
        }

        # Stage 1: Loading
        processing_jobs[document_id].update({
            "progress": 20.0,
            "current_stage": "loading_document",
        })

        # Stage 2: Processing
        processing_jobs[document_id].update({
            "progress": 40.0,
            "current_stage": "processing_document",
            "stages_completed": ["loading_document"],
        })

        # Process the document
        result = await ingestion_service.ingest_file(
            file_path=file_path,
            collection_name=collection_name,
            processing_options=processing_options,
        )

        # Stage 3: Indexing
        processing_jobs[document_id].update({
            "progress": 80.0,
            "current_stage": "indexing",
            "stages_completed": ["loading_document", "processing_document"],
        })

        # Invalidate cache for this collection
        if cache_service and collection_name:
            try:
                deleted = await cache_service.invalidate_collection(collection_name)
                logger.info(f"Cache invalidated for collection '{collection_name}': {deleted} keys")
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")

        # Complete
        processing_jobs[document_id].update({
            "status": ProcessingStatus.COMPLETED.value,
            "progress": 100.0,
            "current_stage": "completed",
            "stages_completed": ["loading_document", "processing_document", "indexing"],
        })

        logger.info(f"Document processing completed: {document_id}")

    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}")
        processing_jobs[document_id] = {
            "status": ProcessingStatus.FAILED.value,
            "progress": 0.0,
            "current_stage": "failed",
            "error_details": {"error": str(e), "error_type": type(e).__name__},
        }

    # Clean up file after a delay (whether successful or failed)
    await asyncio.sleep(300)  # 5 minutes
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up file {file_path}: {e}")

    # Remove job from tracking after a longer delay
    await asyncio.sleep(3600)  # 1 hour
    if document_id in processing_jobs:
        del processing_jobs[document_id]