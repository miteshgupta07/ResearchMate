"""
RAG API Router

Provides the /rag/query endpoint for document-based Q&A.
This is a thin transport layer that delegates to core RAG logic.
"""

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.rag import RAGQueryRequest, RAGQueryResponse
from ..core.deps import (
    get_chat_history_store,
    get_llm,
    InMemoryChatHistoryStore
)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query a document using RAG",
    description="Ask a question about an uploaded document. Uses retrieval-augmented generation for accurate answers."
)
def rag_query(
    request: RAGQueryRequest,
    history_store: InMemoryChatHistoryStore = Depends(get_chat_history_store)
) -> RAGQueryResponse:
    """
    Handle a RAG query and generate a response based on document context.
    
    This endpoint:
    1. Loads the retriever for the specified document
    2. Sets the session context for chat history
    3. Adds the user message to history
    4. Generates a response using RAG
    5. Adds the assistant response to history
    6. Returns the response with sources
    
    Args:
        request: RAG query request with session_id, document_id, message, and optional language
        history_store: Injected chat history store
    
    Returns:
        RAGQueryResponse with the assistant's message and sources
    """
    try:
        # Load retriever for the document
        from core.rag_pipeline import load_retriever, answer_query
        retriever = load_retriever(request.document_id)
        
        if retriever is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {request.document_id}"
            )
        
        # Set session context
        history_store.set_session(request.session_id)
        
        # Add user message to history
        history_store.add_message("user", request.message)
        
        # Get chat history in LangChain format
        chat_history_langchain = history_store.get_langchain_messages()
        
        # Get LLM instance
        llm = get_llm()
        
        # Generate response using RAG (core logic)
        response_content = answer_query(
            llm=llm,
            retriever=retriever,
            chat_history_langchain=chat_history_langchain,
            user_input=request.message,
            language=request.language or "English"
        )
        
        # Add assistant response to history
        history_store.add_message("assistant", response_content)
        
        return RAGQueryResponse(
            role="assistant",
            content=response_content,
            sources=[]  # Reserved for future source extraction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate RAG response: {str(e)}"
        )
