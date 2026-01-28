"""
PostgreSQL Chat History Store Module

Implements ChatHistoryStore interface using PostgreSQL for persistence.
Uses SQLAlchemy Core for database operations.
"""

from typing import List, Tuple

from fastapi import HTTPException
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, Text, DateTime, text
from sqlalchemy.engine import Engine

from .base import ChatHistoryStore
from ..config import DatabaseConfig


# Define table metadata for SQLAlchemy Core
metadata = MetaData()

chat_messages = Table(
    "chat_messages",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("session_id", String(128), nullable=False),
    Column("role", String(20), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
)


class PostgresChatHistoryStore(ChatHistoryStore):
    """
    PostgreSQL-backed implementation of ChatHistoryStore.
    
    This implementation persists chat messages to a PostgreSQL database,
    providing session isolation based on session_id.
    """
    
    _engine: Engine = None
    
    def __init__(self):
        """Initialize the PostgreSQL store with database connection."""
        self._current_session_id: str = ""
        self._ensure_engine()
    
    @classmethod
    def _ensure_engine(cls) -> None:
        """Ensure a single database engine instance exists."""
        if cls._engine is None:
            cls._engine = create_engine(DatabaseConfig.get_connection_url())
    
    def set_session(self, session_id: str) -> None:
        """
        Set the current session context.
        
        Args:
            session_id: The session identifier to use for subsequent operations
        """
        self._current_session_id = session_id
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the current session's chat history.
        
        Args:
            role: The role of the message sender (e.g., 'user', 'assistant')
            content: The text content of the message
        
        Raises:
            HTTPException: If database operation fails
        """
        if not self._current_session_id:
            raise ValueError("No session set. Call set_session() first.")
        
        try:
            with self._engine.connect() as conn:
                stmt = chat_messages.insert().values(
                    session_id=self._current_session_id,
                    role=role,
                    content=content
                )
                conn.execute(stmt)
                conn.commit()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to add message to database: {str(e)}"
            )
    
    def get_messages(self) -> List[dict]:
        """
        Get all messages for the current session in dictionary format.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys,
            ordered by created_at ASC
        """
        if not self._current_session_id:
            return []
        return self.get_session_messages(self._current_session_id)
    
    def get_langchain_messages(self) -> List[Tuple[str, str]]:
        """
        Convert current session messages to LangChain format for prompt templates.
        
        Returns:
            List of (role, content) tuples in LangChain format
        """
        messages = self.get_messages()
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                result.append(("human", content))
            elif role == "assistant":
                result.append(("ai", content))
            else:
                result.append((role, content))
        return result
    
    def clear(self) -> None:
        """
        Clear all messages from the current session's history.
        
        Raises:
            HTTPException: If database operation fails
        """
        if self._current_session_id:
            self.clear_session(self._current_session_id)
    
    def get_session_messages(self, session_id: str) -> List[dict]:
        """
        Get all messages for a specific session.
        
        Args:
            session_id: The session identifier
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys,
            ordered by created_at ASC
        
        Raises:
            HTTPException: If database operation fails
        """
        try:
            with self._engine.connect() as conn:
                stmt = (
                    chat_messages.select()
                    .where(chat_messages.c.session_id == session_id)
                    .order_by(chat_messages.c.created_at.asc())
                )
                result = conn.execute(stmt)
                return [
                    {"role": row.role, "content": row.content}
                    for row in result
                ]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve messages from database: {str(e)}"
            )
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all messages for a specific session.
        
        Args:
            session_id: The session identifier to clear
        
        Raises:
            HTTPException: If database operation fails
        """
        try:
            with self._engine.connect() as conn:
                stmt = chat_messages.delete().where(
                    chat_messages.c.session_id == session_id
                )
                conn.execute(stmt)
                conn.commit()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear messages from database: {str(e)}"
            )
