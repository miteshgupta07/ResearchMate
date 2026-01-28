"""
Dependency Wiring Module

This module provides dependency injection for FastAPI endpoints.
It provides the PostgresChatHistoryStore that persists chat history
to PostgreSQL database.
"""

from .chat_history import PostgresChatHistoryStore


# Global singleton instance of the chat history store
# This maintains state across requests within the same server process
_chat_history_store: PostgresChatHistoryStore = None


def get_chat_history_store() -> PostgresChatHistoryStore:
    """
    Get the global PostgresChatHistoryStore instance.
    
    This is a dependency injection function for FastAPI.
    
    Returns:
        The singleton PostgresChatHistoryStore instance
    """
    global _chat_history_store
    if _chat_history_store is None:
        _chat_history_store = PostgresChatHistoryStore()
    return _chat_history_store


def get_llm():
    """
    Get a configured LLM instance for chat and RAG operations.
    
    Returns:
        Configured ChatGroq model instance
    """
    from services.llm import create_llm
    
    # Default model configuration for API usage
    return create_llm(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=512
    )
