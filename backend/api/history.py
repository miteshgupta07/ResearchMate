"""
History API Router

Provides endpoints for managing chat history.
This is a thin transport layer that uses the InMemoryChatHistoryStore.
"""

from fastapi import APIRouter, Depends, Query

from ..schemas.history import (
    HistoryResponse,
    HistoryClearResponse,
    MessageItem
)
from ..core.deps import get_chat_history_store
from ..core.chat_history import PostgresChatHistoryStore

router = APIRouter(prefix="/history", tags=["History"])


@router.get(
    "",
    response_model=HistoryResponse,
    summary="Get chat history",
    description="Retrieve the chat history for a specific session."
)
def get_history(
    session_id: str = Query(..., description="The session identifier", min_length=1),
    history_store: PostgresChatHistoryStore = Depends(get_chat_history_store)
) -> HistoryResponse:
    """
    Get chat history for a specific session.
    
    Args:
        session_id: The session identifier
        history_store: Injected chat history store
    
    Returns:
        HistoryResponse with session_id and list of messages
    """
    messages = history_store.get_session_messages(session_id)
    
    return HistoryResponse(
        session_id=session_id,
        messages=[
            MessageItem(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
    )


@router.delete(
    "",
    response_model=HistoryClearResponse,
    summary="Clear chat history",
    description="Clear the chat history for a specific session."
)
def clear_history(
    session_id: str = Query(..., description="The session identifier", min_length=1),
    history_store: PostgresChatHistoryStore = Depends(get_chat_history_store)
) -> HistoryClearResponse:
    """
    Clear chat history for a specific session.
    
    Args:
        session_id: The session identifier
        history_store: Injected chat history store
    
    Returns:
        HistoryClearResponse with status
    """
    history_store.clear_session(session_id)
    
    return HistoryClearResponse(status="cleared")
