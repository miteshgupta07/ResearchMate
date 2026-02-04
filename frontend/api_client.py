"""
API Client Module

Provides helper functions for communicating with the FastAPI backend.
All Streamlit views should use this module instead of calling core logic directly.
"""

import requests
from typing import Optional, List, Dict, Any
from backend.core.config import Config

# Base URL for the FastAPI backend
API_BASE_URL = Config.BASE_URL


class APIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def send_chat_message(
    session_id: str,
    message: str,
    language: str = "English",
    model_type: str = None,
    temperature: float = None,
    max_tokens: int = None
) -> Dict[str, Any]:
    """
    Send a chat message to the backend and get a response.
    
    Args:
        session_id: Unique session identifier
        message: The user's message
        language: Target language for the response
        model_type: Frontend model name (e.g. 'LLaMA 3.1-8B')
        temperature: Temperature for response generation (0.0-1.0)
        max_tokens: Maximum tokens in generated response
    
    Returns:
        Dict with 'role' and 'content' keys
    
    Raises:
        APIError: If the request fails
    """
    url = f"{API_BASE_URL}/chat"
    payload = {
        "session_id": session_id,
        "message": message,
        "language": language,
        "model_type": model_type,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise APIError(f"Failed to send chat message: {str(e)}")


def send_rag_query(
    session_id: str,
    document_id: str,
    message: str,
    language: str = "English",
    model_type: str = None,
    temperature: float = None,
    max_tokens: int = None
) -> Dict[str, Any]:
    """
    Send a RAG query to the backend and get a response based on document context.
    
    Args:
        session_id: Unique session identifier
        document_id: The ID of the document to query against
        message: The user's query
        language: Target language for the response
        model_type: Frontend model name (e.g. 'LLaMA 3.1-8B')
        temperature: Temperature for response generation (0.0-1.0)
        max_tokens: Maximum tokens in generated response
    
    Returns:
        Dict with 'role', 'content', and 'sources' keys
    
    Raises:
        APIError: If the request fails
    """
    url = f"{API_BASE_URL}/rag/query"
    payload = {
        "session_id": session_id,
        "document_id": document_id,
        "message": message,
        "language": language,
        "model_type": model_type,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise APIError(f"Failed to send RAG query: {str(e)}")


def upload_document(file) -> Dict[str, Any]:
    """
    Upload a PDF document to the backend for processing.
    
    Args:
        file: File-like object (e.g., from Streamlit file_uploader)
    
    Returns:
        Dict with 'document_id', 'filename', and 'status' keys
    
    Raises:
        APIError: If the upload fails
    """
    url = f"{API_BASE_URL}/documents/upload"
    
    try:
        # Reset file pointer to beginning
        file.seek(0)
        
        files = {
            "file": (file.name, file, "application/pdf")
        }
        
        response = requests.post(url, files=files, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise APIError(f"Failed to upload document: {str(e)}")


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Fetch chat history for a specific session.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        List of message dicts with 'role' and 'content' keys
    
    Raises:
        APIError: If the request fails
    """
    url = f"{API_BASE_URL}/history"
    params = {"session_id": session_id}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("messages", [])
    except requests.exceptions.RequestException as e:
        raise APIError(f"Failed to fetch chat history: {str(e)}")


def clear_chat_history(session_id: str) -> Dict[str, str]:
    """
    Clear chat history for a specific session.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Dict with 'status' key
    
    Raises:
        APIError: If the request fails
    """
    url = f"{API_BASE_URL}/history"
    params = {"session_id": session_id}
    
    try:
        response = requests.delete(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise APIError(f"Failed to clear chat history: {str(e)}")


def send_agent_message(
    session_id: str,
    message: str,
    document_id: str = None,
    language: str = "English",
    model_type: str = None,
    temperature: float = None,
    max_tokens: int = None
) -> Dict[str, Any]:
    """
    Send a message to the agent backend and return the response.
    
    Args:
        session_id: Unique session identifier
        message: The user's message
        document_id: Optional document ID for RAG context
        language: Target language for the response
        model_type: Frontend model name (e.g. 'LLaMA 3.1-8B')
        temperature: Temperature for response generation (0.0-1.0)
        max_tokens: Maximum tokens in generated response
    
    Returns:
        Dict with agent response data
    
    Raises:
        APIError: If the request fails
    """
    url = f"{API_BASE_URL}/agent/route"
    payload = {
        "session_id": session_id,
        "message": message,
        "document_id": document_id,
        "language": language,
        "model_type": model_type,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise APIError(f"Failed to send agent message: {str(e)}")


def check_backend_health() -> bool:
    """
    Check if the backend is available.
    
    Returns:
        True if backend is reachable, False otherwise
    """
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
