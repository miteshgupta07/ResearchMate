"""
FastAPI Application Entry Point

This is the main entry point for the ResearchMate FastAPI backend.
It serves as a thin orchestration layer that exposes existing functionality via HTTP APIs.

Usage:
    uvicorn backend.app.main:app --reload --port 8000
    
    Or from the backend directory:
    uvicorn app.main:app --reload --port 8000
"""

import sys
from pathlib import Path

# Add project root to path for importing core modules
# This allows the backend to import from core/ and services/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, rag, documents, history

# Create FastAPI application
app = FastAPI(
    title="ResearchMate API",
    description="""
ResearchMate FastAPI Backend - A thin orchestration layer for research assistance.

## Features

* **Chat**: Normal conversational AI interactions
* **RAG**: Document-based question answering with retrieval-augmented generation
* **Documents**: PDF upload and processing for RAG
* **History**: Session-based chat history management

## Design Principles

This backend acts as a **transport layer** that:
- Exposes existing functionality via HTTP APIs
- Uses Pydantic schemas for validation
- Calls existing core logic directly (no duplication)
- Maintains session isolation for chat history
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS middleware
# This allows the frontend (Streamlit or other) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat.router)
app.include_router(rag.router)
app.include_router(documents.router)
app.include_router(history.router)


@app.get(
    "/",
    tags=["Health"],
    summary="Root endpoint",
    description="Returns basic API information and health status."
)
def root():
    """
    Root endpoint for API health check and information.
    
    Returns:
        Basic API information and status
    """
    return {
        "name": "ResearchMate API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Simple health check endpoint for monitoring."
)
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}
