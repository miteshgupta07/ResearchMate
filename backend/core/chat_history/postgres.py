"""
PostgreSQL Chat History Store Module

Implements ChatHistoryStore interface using PostgreSQL for persistence.
Uses SQLAlchemy Core for database operations.
"""

from typing import List, Tuple
from contextlib import contextmanager, nullcontext
from hashlib import blake2b

from fastapi import HTTPException
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, Text, DateTime, text
from sqlalchemy.engine import Engine

from .base import ChatHistoryStore
from ..config import Config as DatabaseConfig


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
    
    def __init__(self, connection=None):
        """Initialize the PostgreSQL store with database connection."""
        self._ensure_engine()
        self._connection = connection

    def _connect(self):
        if self._connection is not None:
            if self._connection.invalidated or self._connection.closed:
                raise RuntimeError("The conversation database connection was lost.")
            return nullcontext(self._connection)
        return self._engine.connect()

    @contextmanager
    def session_lock(self, session_id: str, *, wait: bool = False):
        """Own one session on a checked-out connection, including all history writes.

        Session locks survive commits and disappear when the owning process/DB
        connection terminates. Always unlock before returning a connection to the pool.
        """
        key = int.from_bytes(blake2b(session_id.strip().encode(), digest_size=8).digest(), "big", signed=True)
        connection = None
        acquired = False
        try:
            try:
                connection = self._engine.connect().execution_options(isolation_level="AUTOCOMMIT")
                if wait:
                    connection.execute(text("SET statement_timeout = '60s'"))
                    connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
                    acquired = True
                else:
                    acquired = connection.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": key}).scalar_one()
                connection.commit()
            except Exception:
                if connection is not None:
                    connection.invalidate()
                raise HTTPException(status_code=503, detail="Conversation database is unavailable. Try again later.") from None
            if not acquired:
                raise HTTPException(status_code=409, detail="This conversation is busy. End Voice Mode or wait for the current response before trying again.")
            yield PostgresChatHistoryStore(connection)
        finally:
            if connection is not None:
                try:
                    if not connection.invalidated:
                        if acquired:
                            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
                        if wait:
                            connection.execute(text("RESET statement_timeout"))
                        connection.commit()
                except Exception:
                    connection.invalidate()
                finally:
                    connection.close()
    
    @classmethod
    def _ensure_engine(cls) -> None:
        """Ensure a single database engine instance exists."""
        if cls._engine is None:
            cls._engine = create_engine(DatabaseConfig.get_connection_url())
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a new message to the specified session's chat history.
        
        Args:
            session_id: The session identifier
            role: The role of the message sender (e.g., 'user', 'assistant')
            content: The text content of the message
        
        Raises:
            HTTPException: If database operation fails
        """
        if not session_id:
            raise ValueError("session_id must not be empty.")
        
        try:
            with self._connect() as conn:
                stmt = chat_messages.insert().values(
                    session_id=session_id,
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
    
    def get_messages(self, session_id: str) -> List[dict]:
        """
        Get all messages for the specified session in dictionary format.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys,
            ordered by created_at ASC
        """
        if not session_id:
            return []
        return self.get_session_messages(session_id)
    
    def get_langchain_messages(self, session_id: str) -> List[Tuple[str, str]]:
        """
        Convert the specified session's messages to LangChain format for prompt templates.
        
        Returns:
            List of (role, content) tuples in LangChain format
        """
        messages = self.get_messages(session_id)
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
    
    def clear(self, session_id: str) -> None:
        """
        Clear all messages from the specified session's history.
        
        Raises:
            HTTPException: If database operation fails
        """
        if session_id:
            self.clear_session(session_id)
    
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
            with self._connect() as conn:
                stmt = (
                    chat_messages.select()
                    .where(chat_messages.c.session_id == session_id)
                    .order_by(chat_messages.c.created_at.asc(), chat_messages.c.id.asc())
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
            with self._connect() as conn:
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
