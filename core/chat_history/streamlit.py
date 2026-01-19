"""
Streamlit Chat History Implementation Module

This module provides a Streamlit session_state backed implementation
of the ChatHistoryStore interface. This is the ONLY place in the project
that directly interacts with st.session_state for chat history.
"""

import streamlit as st
from typing import List, Tuple

from .base import ChatHistoryStore
from .models import ChatMessage


class StreamlitSessionChatHistory(ChatHistoryStore):
    """
    Chat history store backed by Streamlit session state.
    
    This implementation uses st.session_state to persist chat messages
    across Streamlit reruns within a single browser session.
    """
    
    def __init__(self, session_key: str = "messages"):
        """
        Initialize the Streamlit-backed chat history store.
        
        Args:
            session_key: The key to use in st.session_state for storing messages
                        Different session keys allow for separate chat histories
                        (e.g., 'rag_messages', 'agent_messages')
        """
        self.session_key = session_key
        
        # Initialize session state if not already present
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the chat history.
        
        Args:
            role: The role of the message sender (e.g., 'user', 'assistant')
            content: The text content of the message
        """
        message = ChatMessage(role, content)
        st.session_state[self.session_key].append(message.to_dict())
    
    def get_messages(self) -> List[dict]:
        """
        Get all messages in dictionary format.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return st.session_state[self.session_key]
    
    def get_langchain_messages(self) -> List[Tuple[str, str]]:
        """
        Convert messages to LangChain format for prompt templates.
        
        This method provides LangChain-compatible message tuples
        for use with ChatPromptTemplate and MessagesPlaceholder.
        
        Returns:
            List of (role, content) tuples in LangChain format
        """
        return [
            ChatMessage.from_dict(msg).to_langchain_tuple()
            for msg in st.session_state[self.session_key]
        ]
    
    def clear(self) -> None:
        """
        Clear all messages from history.
        """
        st.session_state[self.session_key] = []
