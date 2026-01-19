"""
Chat Schemas Module

Pydantic models for chat request/response validation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    
    session_id: str = Field(
        ...,
        description="Unique session identifier for chat history isolation",
        min_length=1
    )
    message: str = Field(
        ...,
        description="The user's chat message",
        min_length=1
    )
    language: Optional[str] = Field(
        default="English",
        description="Target language for the response"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "user-123-session-1",
                    "message": "What is machine learning?",
                    "language": "English"
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    
    role: str = Field(
        default="assistant",
        description="The role of the responder"
    )
    content: str = Field(
        ...,
        description="The assistant's response content"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence..."
                }
            ]
        }
    }
