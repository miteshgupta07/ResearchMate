"""
Chat Message Models Module

This module contains the core data models for chat messages.
It provides a single source of truth for message representation.
"""

from typing import Tuple


class ChatMessage:
    """
    Represents a single chat message with role and content.
    
    This is the single source of truth for all chat messages across the application.
    """
    
    def __init__(self, role: str, content: str):
        """
        Initialize a chat message.
        
        Args:
            role: The role of the message sender (e.g., 'user', 'assistant')
            content: The text content of the message
        """
        self.role = role
        self.content = content
    
    def to_dict(self) -> dict:
        """
        Convert message to dictionary format for storage and display.
        
        Returns:
            Dictionary with 'role' and 'content' keys
        """
        return {"role": self.role, "content": self.content}
    
    def to_langchain_tuple(self) -> Tuple[str, str]:
        """
        Convert to LangChain message format (role, content) tuple.
        
        Returns:
            Tuple in format ('human'|'ai', content)
        """
        if self.role == "user":
            return ("human", self.content)
        elif self.role == "assistant":
            return ("ai", self.content)
        else:
            return (self.role, self.content)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        """
        Create a ChatMessage instance from a dictionary.
        
        Args:
            data: Dictionary with 'role' and 'content' keys
        
        Returns:
            ChatMessage instance
        """
        return cls(role=data["role"], content=data["content"])
