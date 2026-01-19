"""
RAG Utility Functions

Helper functions for managing RAG documents and indexes.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json

from core.rag_pipeline import METADATA_DIR, DOCUMENTS_DIR, INDEXES_DIR


def list_all_documents() -> List[Dict]:
    """
    List all ingested documents with their metadata.
    
    Returns:
        List of metadata dictionaries for all documents
    """
    documents = []
    
    if not METADATA_DIR.exists():
        return documents
    
    for metadata_file in METADATA_DIR.glob("*.json"):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            documents.append(metadata)
    
    return sorted(documents, key=lambda x: x.get("timestamp", ""), reverse=True)


def get_document_metadata(document_id: str) -> Optional[Dict]:
    """
    Get metadata for a specific document.
    
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


def delete_document(document_id: str) -> bool:
    """
    Delete a document and all associated data.
    
    This removes:
    - The PDF file
    - The FAISS index
    - The metadata file
    
    Args:
        document_id: Unique document identifier
    
    Returns:
        True if deletion was successful, False otherwise
    """
    import shutil
    
    success = True
    
    # Delete PDF
    pdf_path = DOCUMENTS_DIR / f"{document_id}.pdf"
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except Exception:
            success = False
    
    # Delete FAISS index directory
    index_path = INDEXES_DIR / document_id
    if index_path.exists():
        try:
            shutil.rmtree(index_path)
        except Exception:
            success = False
    
    # Delete metadata
    metadata_path = METADATA_DIR / f"{document_id}.json"
    if metadata_path.exists():
        try:
            metadata_path.unlink()
        except Exception:
            success = False
    
    return success


def get_storage_stats() -> Dict:
    """
    Get statistics about RAG storage usage.
    
    Returns:
        Dictionary with storage statistics
    """
    stats = {
        "total_documents": 0,
        "total_chunks": 0,
        "storage_size_mb": 0.0,
    }
    
    # Count documents and chunks
    documents = list_all_documents()
    stats["total_documents"] = len(documents)
    stats["total_chunks"] = sum(doc.get("chunk_count", 0) for doc in documents)
    
    # Calculate storage size
    total_bytes = 0
    
    for directory in [DOCUMENTS_DIR, INDEXES_DIR, METADATA_DIR]:
        if directory.exists():
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_bytes += file_path.stat().st_size
    
    stats["storage_size_mb"] = round(total_bytes / (1024 * 1024), 2)
    
    return stats
