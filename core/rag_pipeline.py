"""
RAG Pipeline Module

This module contains all RAG-related logic including:
- PDF processing and document loading
- Text chunking and splitting
- Embedding generation
- Vector store creation and persistence
- Retrieval and answer generation
- Document metadata management
"""

import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from services.llm import create_rag_chat_prompt


# Directory structure for persistent storage
RAG_DATA_DIR = Path("data/rag")
DOCUMENTS_DIR = RAG_DATA_DIR / "documents"
INDEXES_DIR = RAG_DATA_DIR / "indexes"
METADATA_DIR = RAG_DATA_DIR / "metadata"

# Session-level cache to avoid reprocessing within the same session
_retriever_cache = {}


def _ensure_directories():
    """Ensure all required directories exist."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_embeddings():
    """
    Get the embeddings model instance.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _save_metadata(document_id: str, filename: str, chunk_count: int):
    """
    Save document metadata to disk.
    
    Args:
        document_id: Unique document identifier
        filename: Original filename
        chunk_count: Number of chunks created
    """
    metadata = {
        "document_id": document_id,
        "filename": filename,
        "chunk_count": chunk_count,
        "timestamp": datetime.now().isoformat(),
    }
    
    metadata_path = METADATA_DIR / f"{document_id}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _load_metadata(document_id: str) -> Optional[dict]:
    """
    Load document metadata from disk.
    
    Args:
        document_id: Unique document identifier
    
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = METADATA_DIR / f"{document_id}.json"
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest_document(uploaded_file, original_filename: str = None) -> Tuple[str, any]:
    """
    Ingest a PDF document with persistent storage.
    
    This function:
    1. Generates a unique document ID
    2. Saves the PDF to persistent storage
    3. Loads and parses the PDF
    4. Splits text into chunks
    5. Generates embeddings
    6. Creates and persists FAISS vector store
    7. Saves metadata
    8. Caches retriever for session reuse
    
    Args:
        uploaded_file: Streamlit UploadedFile object or file-like object
        original_filename: Original filename (optional, extracted from uploaded_file if available)
    
    Returns:
        Tuple of (document_id, retriever)
    """
    _ensure_directories()
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    # Determine original filename
    if original_filename is None:
        original_filename = getattr(uploaded_file, 'name', 'document.pdf')
    
    # Check session cache first
    cache_key = f"file_{id(uploaded_file)}"
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]
    
    # Save PDF to persistent storage
    pdf_path = DOCUMENTS_DIR / f"{document_id}.pdf"
    with open(pdf_path, "wb") as f:
        # Reset file pointer if it's seekable
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        f.write(uploaded_file.read())
    
    # Load the document using PyMuPDF
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    
    # Split text into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = _get_embeddings()
    
    # Build FAISS vector store from document chunks
    db = FAISS.from_documents(docs, embeddings)
    
    # Persist FAISS index to disk
    index_path = INDEXES_DIR / document_id
    index_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_path))
    
    # Save metadata
    _save_metadata(document_id, original_filename, len(docs))
    
    # Get retriever
    retriever = db.as_retriever()
    
    # Cache for session reuse
    _retriever_cache[cache_key] = (document_id, retriever)
    
    return document_id, retriever


def load_retriever(document_id: str) -> Optional[any]:
    """
    Load a retriever from a persisted FAISS index.
    
    Args:
        document_id: Unique document identifier
    
    Returns:
        FAISS retriever instance or None if not found
    """
    _ensure_directories()
    
    # Check session cache first
    cache_key = f"doc_{document_id}"
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]
    
    # Check if index exists
    index_path = INDEXES_DIR / document_id
    if not index_path.exists():
        return None
    
    # Load embeddings
    embeddings = _get_embeddings()
    
    # Load FAISS index from disk
    db = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Get retriever
    retriever = db.as_retriever()
    
    # Cache for session reuse
    _retriever_cache[cache_key] = retriever
    
    return retriever


def answer_query(llm, retriever, chat_history_langchain: list, user_input: str, language: str) -> str:
    """
    Generate an answer using RAG (Retrieval-Augmented Generation).
    
    This function:
    1. Retrieves relevant context from the vector store
    2. Constructs a prompt with context and chat history
    3. Generates a response using the LLM
    
    Args:
        llm: The language model instance
        retriever: FAISS retriever for document search
        chat_history_langchain: List of (role, content) tuples in LangChain format
        user_input: The user's question
        language: Target language for the response
    
    Returns:
        Generated answer text incorporating retrieved context
    """
    # Retrieve relevant context from the vector store
    context = retriever.invoke(user_input)
    
    # Create RAG-specific prompt and chain
    rag_prompt = create_rag_chat_prompt()
    chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)
    
    # Generate response with context and chat history
    response = chain.invoke({
        "context": context,
        "language": language,
        "rag_messages": chat_history_langchain
    })
    
    return response


# Legacy functions for backward compatibility


def process_uploaded_pdf(uploaded_file, temp_file_path: str = "uploaded_document.pdf"):
    """
    Legacy function for backward compatibility.
    
    Process an uploaded PDF file and create a retriever.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        temp_file_path: Path where to save the temporary PDF file (ignored in new implementation)
    
    Returns:
        FAISS retriever instance for document search
    """
    document_id, retriever = ingest_document(uploaded_file)
    return retriever


def answer_with_rag(llm, retriever, user_input: str, chat_history_langchain: list, language: str):
    """
    Legacy function for backward compatibility.
    
    Generate an answer using RAG (Retrieval-Augmented Generation).
    
    Args:
        llm: The language model instance
        retriever: FAISS retriever for document search
        user_input: The user's question
        chat_history_langchain: List of (role, content) tuples in LangChain format
        language: Target language for the response
    
    Returns:
        Generated answer text incorporating retrieved context
    """
    # Use new API with matching parameter order
    return answer_query(llm, retriever, chat_history_langchain, user_input, language)


def answer_without_rag(llm, chat_history_langchain: list, language: str):
    """
    Generate an answer without RAG (normal chat mode).
    
    This is a convenience wrapper that delegates to the LLM service.
    
    Args:
        llm: The language model instance
        chat_history_langchain: List of (role, content) tuples in LangChain format
        language: Target language for the response
    
    Returns:
        Generated answer text
    """
    from services.llm import generate_chat_response
    return generate_chat_response(llm, chat_history_langchain, language)

