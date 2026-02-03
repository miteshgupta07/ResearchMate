"""
Chat API Router

Provides the /chat endpoint for normal (non-RAG) chat interactions.
This is a thin transport layer that delegates to core logic.
"""

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.chat import ChatRequest, ChatResponse
from ..core.deps import get_chat_history_store, get_llm
from ..core.chat_history import PostgresChatHistoryStore

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    description="Send a message and receive an AI response. Uses session-based chat history for context."
)
def chat(
    request: ChatRequest,
    history_store: PostgresChatHistoryStore = Depends(get_chat_history_store)
) -> ChatResponse:
    """
    Handle a chat message and generate a response.
    
    This endpoint:
    1. Sets the session context for chat history
    2. Adds the user message to history
    3. Generates a response using the LLM
    4. Adds the assistant response to history
    5. Returns the response
    
    Args:
        request: Chat request with session_id, message, and optional language/LLM config
        history_store: Injected chat history store
    
    Returns:
        ChatResponse with the assistant's message
    """

    try:
        # Set session context
        history_store.set_session(request.session_id)
        
        # Add user message to history
        history_store.add_message("user", request.message)
        
        # Get chat history in LangChain format
        chat_history_langchain = history_store.get_langchain_messages()

        # Get LLM instance with dynamic configuration
        llm = get_llm(
            model_type=request.model_type,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        # Generate response using core logic (no RAG)
        from backend.core.rag import answer_without_rag 
        response_content = answer_without_rag(
            llm=llm,
            chat_history_langchain=chat_history_langchain,
            language=request.language or "English"
        )
        # Add assistant response to history
        history_store.add_message("assistant", response_content)
        return ChatResponse(
            role="assistant",
            content=response_content
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate chat response: {str(e)}"
        )
