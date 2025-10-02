"""
Test script for document upload API endpoints.

This module tests the document upload functionality including:
- Single document upload
- Batch document upload
- Upload validation
- Processing progress tracking
- Document listing and retrieval
- Document deletion
"""

import io
import os
import time
import pytest
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch, AsyncMock

from fastapi import UploadFile
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models.responses import ProcessingStatus
from src.core.config import get_settings


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_file() -> Generator[io.BytesIO, None, None]:
    """Create a sample PDF file for testing."""
    # Simple PDF content (minimal valid PDF)
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF
"""
    file_obj = io.BytesIO(pdf_content)
    file_obj.name = "test_document.pdf"
    yield file_obj
    file_obj.close()


@pytest.fixture
def multiple_pdf_files() -> Generator[list[io.BytesIO], None, None]:
    """Create multiple sample PDF files for batch testing."""
    files = []
    for i in range(3):
        pdf_content = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF {i}) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF
""".encode()
        file_obj = io.BytesIO(pdf_content)
        file_obj.name = f"test_document_{i}.pdf"
        files.append(file_obj)

    yield files

    for file_obj in files:
        file_obj.close()


class TestDocumentUpload:
    """Test cases for document upload endpoints."""

    def test_upload_single_document_success(self, client: TestClient, sample_pdf_file: io.BytesIO):
        """
        Test successful upload of a single document.

        Verifies:
        - HTTP 200 response
        - Valid document ID returned
        - Correct filename and content type
        - Processing status is PENDING
        """
        print("\n=== Testing Single Document Upload ===")

        # Mock the ingestion service and dependencies
        with patch("src.api.endpoints.documents.get_document_ingestion_service") as mock_ingestion, \
             patch("src.api.endpoints.documents.validate_file_upload", return_value=True):

            mock_ingestion_instance = Mock()
            mock_ingestion_instance.ingest_file = AsyncMock(return_value={"status": "success"})
            mock_ingestion.return_value = mock_ingestion_instance

            # Prepare file for upload
            files = {
                "file": ("test_document.pdf", sample_pdf_file, "application/pdf")
            }
            data = {
                "collection_name": "test_collection",
                "chunk_size": 512,
                "chunk_overlap": 128,
                "use_llamaparse": False
            }

            # Make request
            response = client.post("/api/documents/upload", files=files, data=data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            # Assertions
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

            response_data = response.json()
            assert "document_id" in response_data
            assert response_data["filename"] == "test_document.pdf"
            assert response_data["content_type"] == "application/pdf"
            assert response_data["processing_status"] == ProcessingStatus.PENDING.value
            assert response_data["collection_name"] == "test_collection"
            assert response_data["size"] > 0

            print(f"✓ Document uploaded successfully with ID: {response_data['document_id']}")


    def test_upload_document_invalid_file_type(self, client: TestClient):
        """
        Test upload rejection for invalid file types.

        Verifies:
        - HTTP 422 response for non-PDF files
        - Appropriate error message
        """
        print("\n=== Testing Invalid File Type Upload ===")

        with patch("src.api.endpoints.documents.validate_file_upload", return_value=True):
            # Create a text file instead of PDF
            text_file = io.BytesIO(b"This is not a PDF file")
            files = {
                "file": ("test_document.txt", text_file, "text/plain")
            }

            response = client.post("/api/documents/upload", files=files)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 422
            response_data = response.json()
            assert "not supported" in response_data["detail"].lower() or "error" in response_data

            print("✓ Invalid file type correctly rejected")


    def test_upload_document_with_metadata(self, client: TestClient, sample_pdf_file: io.BytesIO):
        """
        Test document upload with custom metadata.

        Verifies:
        - Metadata is accepted and stored
        - Document upload succeeds with metadata
        """
        print("\n=== Testing Document Upload with Metadata ===")

        with patch("src.api.endpoints.documents.get_document_ingestion_service") as mock_ingestion, \
             patch("src.api.endpoints.documents.validate_file_upload", return_value=True):

            mock_ingestion_instance = Mock()
            mock_ingestion_instance.ingest_file = AsyncMock(return_value={"status": "success"})
            mock_ingestion.return_value = mock_ingestion_instance

            import json
            metadata = {
                "author": "Test Author",
                "department": "Engineering",
                "tags": ["test", "sample"]
            }

            files = {
                "file": ("test_document.pdf", sample_pdf_file, "application/pdf")
            }
            data = {
                "metadata": json.dumps(metadata)
            }

            response = client.post("/api/documents/upload", files=files, data=data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            response_data = response.json()
            assert "document_id" in response_data

            print(f"✓ Document uploaded with metadata successfully")


    def test_batch_upload_documents(self, client: TestClient, multiple_pdf_files: list[io.BytesIO]):
        """
        Test batch upload of multiple documents.

        Verifies:
        - All documents are accepted
        - Batch ID is returned
        - Correct count of successful uploads
        """
        print("\n=== Testing Batch Document Upload ===")

        with patch("src.api.endpoints.documents.get_document_ingestion_service") as mock_ingestion, \
             patch("src.api.endpoints.documents.validate_file_upload", return_value=True):

            mock_ingestion_instance = Mock()
            mock_ingestion_instance.ingest_file = AsyncMock(return_value={"status": "success"})
            mock_ingestion.return_value = mock_ingestion_instance

            files = [
                ("files", (f"test_document_{i}.pdf", file, "application/pdf"))
                for i, file in enumerate(multiple_pdf_files)
            ]
            data = {
                "collection_name": "test_batch_collection",
                "parallel_processing": True
            }

            response = client.post("/api/documents/upload/batch", files=files, data=data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            response_data = response.json()
            assert "batch_id" in response_data
            assert response_data["total_files"] == len(multiple_pdf_files)
            assert len(response_data["successful_uploads"]) == len(multiple_pdf_files)
            assert len(response_data["failed_uploads"]) == 0

            print(f"✓ Batch upload successful: {len(multiple_pdf_files)} files uploaded")
            print(f"  Batch ID: {response_data['batch_id']}")


    def test_batch_upload_exceeds_limit(self, client: TestClient):
        """
        Test batch upload rejection when exceeding maximum file limit.

        Verifies:
        - HTTP 422 response when too many files
        - Appropriate error message
        """
        print("\n=== Testing Batch Upload Limit Exceeded ===")

        settings = get_settings()
        max_files = settings.app.max_concurrent_uploads

        with patch("src.api.endpoints.documents.validate_file_upload", return_value=True):
            # Create more files than allowed
            files = [
                ("files", (f"test_{i}.pdf", io.BytesIO(b"fake pdf"), "application/pdf"))
                for i in range(max_files + 1)
            ]

            response = client.post("/api/documents/upload/batch", files=files)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 422
            response_data = response.json()
            assert "maximum" in response_data["detail"].lower() or "too many" in response_data["detail"].lower()

            print(f"✓ Batch upload correctly rejected (limit: {max_files} files)")


    def test_get_processing_progress(self, client: TestClient):
        """
        Test retrieval of document processing progress.

        Verifies:
        - Progress endpoint returns status
        - Progress information is valid
        """
        print("\n=== Testing Processing Progress Retrieval ===")

        # First, we need to mock a document in the processing_jobs dict
        from src.api.endpoints import documents as doc_module

        test_doc_id = "test-doc-id-12345"
        doc_module.processing_jobs[test_doc_id] = {
            "status": ProcessingStatus.PROCESSING.value,
            "progress": 50.0,
            "current_stage": "processing_document",
            "stages_completed": ["loading_document"],
        }

        try:
            response = client.get(f"/api/documents/progress/{test_doc_id}")

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["document_id"] == test_doc_id
            assert response_data["status"] == ProcessingStatus.PROCESSING.value
            assert response_data["progress_percentage"] == 50.0
            assert response_data["current_stage"] == "processing_document"

            print(f"✓ Processing progress retrieved successfully: {response_data['progress_percentage']}%")

        finally:
            # Cleanup
            if test_doc_id in doc_module.processing_jobs:
                del doc_module.processing_jobs[test_doc_id]


    def test_get_processing_progress_not_found(self, client: TestClient):
        """
        Test processing progress retrieval for non-existent document.

        Verifies:
        - HTTP 404 response
        - Appropriate error message
        """
        print("\n=== Testing Processing Progress Not Found ===")

        non_existent_id = "non-existent-document-id"
        response = client.get(f"/api/documents/progress/{non_existent_id}")

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        assert response.status_code == 404
        response_data = response.json()
        assert "not found" in response_data["detail"].lower()

        print("✓ Non-existent document correctly returns 404")


    def test_list_documents(self, client: TestClient):
        """
        Test listing documents in a collection.

        Verifies:
        - Documents list is returned
        - Pagination parameters work
        """
        print("\n=== Testing Document Listing ===")

        with patch("src.api.endpoints.documents.get_vector_store_client") as mock_vector_store:
            # Mock vector store response
            mock_store_instance = Mock()
            mock_store_instance.search_by_metadata = Mock(return_value=[
                {
                    "id": "doc-1",
                    "metadata": {
                        "filename": "test1.pdf",
                        "mime_type": "application/pdf",
                        "file_size": 1024,
                        "processed_at": "2025-01-01T00:00:00",
                    }
                },
                {
                    "id": "doc-2",
                    "metadata": {
                        "filename": "test2.pdf",
                        "mime_type": "application/pdf",
                        "file_size": 2048,
                        "processed_at": "2025-01-01T01:00:00",
                    }
                }
            ])
            mock_vector_store.return_value = mock_store_instance

            response = client.get("/api/documents/list?limit=10&offset=0")

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            response_data = response.json()
            assert isinstance(response_data, list)
            assert len(response_data) > 0

            print(f"✓ Document list retrieved: {len(response_data)} documents")


    def test_get_document_by_id(self, client: TestClient):
        """
        Test retrieval of a specific document by ID.

        Verifies:
        - Document details are returned
        - Information is complete
        """
        print("\n=== Testing Document Retrieval by ID ===")

        with patch("src.api.endpoints.documents.get_vector_store_client") as mock_vector_store:
            test_doc_id = "test-document-12345"

            mock_store_instance = Mock()
            mock_store_instance.search_by_metadata = Mock(return_value=[
                {
                    "id": test_doc_id,
                    "metadata": {
                        "filename": "test.pdf",
                        "mime_type": "application/pdf",
                        "file_size": 1024,
                        "processed_at": "2025-01-01T00:00:00",
                        "chunk_count": 10,
                    }
                }
            ])
            mock_vector_store.return_value = mock_store_instance

            response = client.get(f"/api/documents/{test_doc_id}")

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["document_id"] == test_doc_id
            assert response_data["filename"] == "test.pdf"
            assert response_data["processing_status"] == ProcessingStatus.COMPLETED.value

            print(f"✓ Document retrieved successfully: {response_data['filename']}")


    def test_delete_document(self, client: TestClient):
        """
        Test document deletion.

        Verifies:
        - Document is deleted from vector store
        - HTTP 200 response
        - Success message returned
        """
        print("\n=== Testing Document Deletion ===")

        with patch("src.api.endpoints.documents.get_vector_store_client") as mock_vector_store:
            test_doc_id = "test-document-to-delete"

            mock_store_instance = Mock()
            mock_store_instance.search_by_metadata = Mock(return_value=[
                {
                    "id": test_doc_id,
                    "metadata": {
                        "filename": "test.pdf",
                        "file_path": "/tmp/test.pdf",
                    }
                }
            ])
            mock_store_instance.delete = Mock(return_value=True)
            mock_vector_store.return_value = mock_store_instance

            response = client.delete(f"/api/documents/{test_doc_id}?delete_file=false")

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            response_data = response.json()
            assert "deleted successfully" in response_data["message"].lower()

            # Verify delete was called
            mock_store_instance.delete.assert_called_once()

            print(f"✓ Document deleted successfully")


    def test_create_collection(self, client: TestClient):
        """
        Test creation of a new document collection.

        Verifies:
        - Collection is created
        - Collection info is returned
        """
        print("\n=== Testing Collection Creation ===")

        with patch("src.api.endpoints.documents.get_vector_store_client") as mock_vector_store:
            mock_store_instance = Mock()
            mock_store_instance.create_collection = Mock(return_value=True)
            mock_vector_store.return_value = mock_store_instance

            collection_data = {
                "name": "test_new_collection",
                "description": "Test collection for unit tests",
                "metadata": {
                    "created_by": "test_user"
                }
            }

            response = client.post("/api/documents/collections", json=collection_data)

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["name"] == "test_new_collection"
            assert response_data["description"] == "Test collection for unit tests"
            assert response_data["document_count"] == 0

            print(f"✓ Collection created successfully: {response_data['name']}")


    def test_list_collections(self, client: TestClient):
        """
        Test listing all available collections.

        Verifies:
        - Collections list is returned
        - Collection information is complete
        """
        print("\n=== Testing Collection Listing ===")

        with patch("src.api.endpoints.documents.get_vector_store_client") as mock_vector_store:
            mock_store_instance = Mock()
            mock_store_instance.list_collections = Mock(return_value=[
                "collection_1",
                "collection_2",
                "test_collection"
            ])
            mock_store_instance.get_collection_stats = Mock(return_value={
                "count": 10,
                "metadata": {"description": "Test collection"}
            })
            mock_vector_store.return_value = mock_store_instance

            response = client.get("/api/documents/collections")

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")

            assert response.status_code == 200
            response_data = response.json()
            assert isinstance(response_data, list)
            assert len(response_data) > 0

            print(f"✓ Collections list retrieved: {len(response_data)} collections")


if __name__ == "__main__":
    """
    Run tests with verbose output.

    Usage:
        python -m pytest tests/test_api/test_document_upload.py -v -s
    """
    pytest.main([__file__, "-v", "-s"])
