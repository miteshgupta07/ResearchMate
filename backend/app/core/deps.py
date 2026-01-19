"""
Dependency Wiring Module

This module provides dependency injection for FastAPI endpoints.
It includes the InMemoryChatHistoryStore implementation that follows
the ChatHistoryStore interface from core.chat_history.base.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path for importing core modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import directly from submodules to avoid importing Streamlit
# (the __init__.py imports StreamlitSessionChatHistory which requires streamlit)
from core.chat_history.base import ChatHistoryStore
from core.chat_history.models import ChatMessage


class InMemoryChatHistoryStore(ChatHistoryStore):
    """
    In-memory implementation of ChatHistoryStore for FastAPI backend.
    
    This implementation stores chat messages in a simple dictionary,
    providing session isolation based on session_id.
    
    Note: This is a non-persistent store. Messages are lost when the
    server restarts. For production use, replace with a database-backed
    implementation.
    """
    
    def __init__(self):
        """Initialize the in-memory store with an empty messages dictionary."""
        self._messages: Dict[str, List[dict]] = {}
        self._current_session_id: str = ""
    
    def set_session(self, session_id: str) -> None:
        """
        Set the current session context.
        
        Args:
            session_id: The session identifier to use for subsequent operations
        """
        self._current_session_id = session_id
        if session_id not in self._messages:
            self._messages[session_id] = []
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the current session's chat history.
        
        Args:
            role: The role of the message sender (e.g., 'user', 'assistant')
            content: The text content of the message
        """
        if not self._current_session_id:
            raise ValueError("No session set. Call set_session() first.")
        
        message = ChatMessage(role, content)
        self._messages[self._current_session_id].append(message.to_dict())
    
    def get_messages(self) -> List[dict]:
        """
        Get all messages for the current session in dictionary format.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if not self._current_session_id:
            return []
        return self._messages.get(self._current_session_id, [])
    
    def get_langchain_messages(self) -> List[Tuple[str, str]]:
        """
        Convert current session messages to LangChain format for prompt templates.
        
        Returns:
            List of (role, content) tuples in LangChain format
        """
        messages = self.get_messages()
        return [
            ChatMessage.from_dict(msg).to_langchain_tuple()
            for msg in messages
        ]
    
    def clear(self) -> None:
        """
        Clear all messages from the current session's history.
        """
        if self._current_session_id:
            self._messages[self._current_session_id] = []
    
    def get_session_messages(self, session_id: str) -> List[dict]:
        """
        Get all messages for a specific session.
        
        Args:
            session_id: The session identifier
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self._messages.get(session_id, [])
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all messages for a specific session.
        
        Args:
            session_id: The session identifier to clear
        """
        if session_id in self._messages:
            self._messages[session_id] = []


# Global singleton instance of the chat history store
# This maintains state across requests within the same server process
_chat_history_store: InMemoryChatHistoryStore = None


def get_chat_history_store() -> InMemoryChatHistoryStore:
    """
    Get the global InMemoryChatHistoryStore instance.
    
    This is a dependency injection function for FastAPI.
    
    Returns:
        The singleton InMemoryChatHistoryStore instance
    """
    global _chat_history_store
    if _chat_history_store is None:
        _chat_history_store = InMemoryChatHistoryStore()
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
