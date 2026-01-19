"""
Chat History Store Base Module

This module defines the abstract interface for chat history storage.
It is framework-agnostic and does not depend on Streamlit.
"""

from abc import ABC, abstractmethod
from typing import List


class ChatHistoryStore(ABC):
    """
    Abstract base class for chat history storage.
    
    This interface defines the contract for storing and retrieving chat messages,
    allowing for different backend implementations (session state, database, etc.).
    """
    
    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the chat history.
        
        Args:
            role: The role of the message sender (e.g., 'user', 'assistant')
            content: The text content of the message
        """
        pass
    
    @abstractmethod
    def get_messages(self) -> List[dict]:
        """
        Get all messages in dictionary format.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all messages from history.
        """
        pass
