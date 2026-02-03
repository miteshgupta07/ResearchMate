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

from .pipeline import (
    ingest_document,
    load_retriever,
    answer_query,
    process_uploaded_pdf,
    answer_with_rag,
    answer_without_rag,
    RAG_DATA_DIR,
    DOCUMENTS_DIR,
    INDEXES_DIR,
    METADATA_DIR,
)
from .utils import (
    list_all_documents,
    get_document_metadata,
    delete_document,
    get_storage_stats,
)

__all__ = [
    "ingest_document",
    "load_retriever",
    "answer_query",
    "process_uploaded_pdf",
    "answer_with_rag",
    "answer_without_rag",
    "RAG_DATA_DIR",
    "DOCUMENTS_DIR",
    "INDEXES_DIR",
    "METADATA_DIR",
    "list_all_documents",
    "get_document_metadata",
    "delete_document",
    "get_storage_stats",
]
