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
                    "message": "What is machine learning?",
                    "language": "English",
                    "model_type": "LLaMA 3.1-8B",
                    "temperature": 0.7,
                    "max_tokens": 512
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
