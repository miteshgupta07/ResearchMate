"""
Dependency Wiring Module

This module provides dependency injection for FastAPI endpoints.
It provides the PostgresChatHistoryStore that persists chat history
to PostgreSQL database.
"""

from typing import Optional
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


def get_llm(
    model_type: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    """
    Get a configured LLM instance for chat and RAG operations.
    
    This function uses the pre-initialized LLM registry to get a model instance.
    The registry is initialized at FastAPI startup, so no cold-start delay occurs.
    
    Temperature and max_tokens are applied per-request, not at initialization.
    Defaults are applied inside the LLM service layer:
    - temperature: 0.7
    - max_tokens: 512
    - model: llama-3.1-8b-instant
    
    Args:
        model_type: Frontend model name (e.g., "DeepSeek r1", "LLaMA 3.1-8B")
        temperature: Controls randomness in responses (0.0-1.0)
        max_tokens: Maximum tokens in generated response
    
    Returns:
        Configured ChatGroq model instance with per-request parameters
    """
    from backend.core.llm import get_llm_registry
    
    registry = get_llm_registry()
    return registry.get_llm_with_params(
        model_type=model_type,
        temperature=temperature,
        max_tokens=max_tokens
    )
