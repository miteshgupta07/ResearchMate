"""
Document Schemas Module

Pydantic models for document upload request/response validation.
"""

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response model for the document upload endpoint."""
    
    document_id: str = Field(
        ...,
        description="Unique identifier for the uploaded document"
    )
    filename: str = Field(
        ...,
        description="Original filename of the uploaded document"
    )
    status: str = Field(
        default="processed",
        description="Processing status of the document"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_id": "7c3421af-5fa1-47c7-8555-6b763e06d666",
                    "filename": "research_paper.pdf",
                    "status": "processed"
                }
            ]
        }
    }
