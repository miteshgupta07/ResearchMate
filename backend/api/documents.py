"""
Documents API Router

Provides the /documents/upload endpoint for PDF document ingestion.
This is a thin transport layer that delegates to core RAG logic.
"""

from io import BytesIO
from fastapi import APIRouter, HTTPException, UploadFile, File

from ..schemas.document import DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a PDF document",
    description="Upload a PDF document for RAG processing. The document will be chunked, embedded, and indexed."
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload")
) -> DocumentUploadResponse:
    """
    Handle PDF document upload and ingestion.
    
    This endpoint:
    1. Validates the uploaded file is a PDF
    2. Reads the file content
    3. Calls the core ingest_document function
    4. Returns the document ID and status
    
    Args:
        file: The uploaded PDF file
    
    Returns:
        DocumentUploadResponse with document_id, filename, and status
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Validate content type
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Expected application/pdf"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Create a file-like object for the core function
        file_obj = BytesIO(content)
        
        # Ingest the document using core logic
        from backend.core.rag import ingest_document
        document_id, _ = ingest_document(
            uploaded_file=file_obj,
            original_filename=file.filename
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="processed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )
