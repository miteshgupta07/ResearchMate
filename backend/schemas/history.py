"""
History Schemas Module

Pydantic models for chat history request/response validation.
"""

from typing import List
from pydantic import BaseModel, Field


class MessageItem(BaseModel):
    """Model for a single chat message."""
    
    role: str = Field(
        ...,
        description="The role of the message sender (user or assistant)"
    )
    content: str = Field(
        ...,
        description="The text content of the message"
    )


class HistoryResponse(BaseModel):
    """Response model for the get history endpoint."""
    
    session_id: str = Field(
        ...,
        description="The session identifier"
    )
    messages: List[MessageItem] = Field(
        default_factory=list,
        description="List of chat messages in the session"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "user-123-session-1",
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Hi there! How can I help you today?"}
                    ]
                }
            ]
        }
    }


class HistoryClearResponse(BaseModel):
    """Response model for the clear history endpoint."""
    
    status: str = Field(
        default="cleared",
        description="Status of the clear operation"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "cleared"
                }
            ]
        }
    }
