"""
RAG Schemas Module

Pydantic models for RAG query request/response validation.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Request model for the RAG query endpoint."""
    
    session_id: str = Field(
        ...,
        description="Unique session identifier for chat history isolation",
        min_length=1
    )
    document_id: str = Field(
        ...,
        description="The ID of the document to query against",
        min_length=1
    )
    message: str = Field(
        ...,
        description="The user's query about the document",
        min_length=1
    )
    language: Optional[str] = Field(
        default="English",
        description="Target language for the response"
    )
    model_type: Optional[str] = Field(
        default=None,
        description="Frontend model name (e.g., 'DeepSeek r1', 'LLaMA 3.1-8B')"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for response generation (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in generated response",
        gt=0
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "user-123-session-1",
                    "document_id": "7c3421af-5fa1-47c7-8555-6b763e06d666",
                    "message": "What are the main findings of this paper?",
                    "language": "English",
                    "model_type": "LLaMA 3.1-8B",
                    "temperature": 0.7,
                    "max_tokens": 512
                }
            ]
        }
    }


class RAGQueryResponse(BaseModel):
    """Response model for the RAG query endpoint."""
    
    role: str = Field(
        default="assistant",
        description="The role of the responder"
    )
    content: str = Field(
        ...,
        description="The assistant's response based on document context"
    )
    sources: List[Any] = Field(
        default_factory=list,
        description="Source references from the document (reserved for future use)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "role": "assistant",
                    "content": "Based on the document, the main findings are...",
                    "sources": []
                }
            ]
        }
    }
