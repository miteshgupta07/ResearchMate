"""
Agent Schemas Module

Pydantic models for the deterministic agent request/response validation.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Request model for the agent route endpoint."""
    
    session_id: str = Field(
        ...,
        description="Unique session identifier for chat history isolation",
        min_length=1
    )
    message: str = Field(
        ...,
        description="The user's message to the agent",
        min_length=1
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID for RAG-based queries"
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
                    "document_id": None,
                    "language": "English"
                }
            ]
        }
    }


class AgentResponse(BaseModel):
    """Response model for the agent route endpoint."""
    
    role: str = Field(
        default="assistant",
        description="The role of the responder"
    )
    content: str = Field(
        ...,
        description="The assistant's response content"
    )
    intent: str = Field(
        ...,
        description="The classified intent that was routed"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for the intent classification (0.0-1.0)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (sources, papers, etc.)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence...",
                    "intent": "INTENT_CHAT",
                    "confidence": 1.0,
                    "metadata": {}
                }
            ]
        }
    }
