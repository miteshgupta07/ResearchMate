"""
RAG Pipeline Module

This module contains all RAG-related logic including:
- PDF processing and document loading
- Text chunking and splitting
- Embedding generation
- Vector store creation
- Retrieval and answer generation
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from services.llm import create_rag_chat_prompt


def process_uploaded_pdf(uploaded_file, temp_file_path: str = "uploaded_document.pdf"):
    """
    Process an uploaded PDF file and create a retriever.
    
    This function:
    1. Saves the uploaded file temporarily
    2. Loads and parses the PDF
    3. Splits text into chunks
    4. Generates embeddings
    5. Creates a FAISS vector store
    6. Returns a retriever interface
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        temp_file_path: Path where to save the temporary PDF file
    
    Returns:
        FAISS retriever instance for document search
    """
    # Save the uploaded file temporarily
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Load the document using PyMuPDF
    loader = PyMuPDFLoader(temp_file_path)
    documents = loader.load()
    
    # Split text into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    # Create embeddings using HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build FAISS vector store from document chunks
    db = FAISS.from_documents(docs, embeddings)
    
    # Return retriever interface
    return db.as_retriever()


def answer_with_rag(llm, retriever, user_input: str, chat_history_langchain: list, language: str):
    """
    Generate an answer using RAG (Retrieval-Augmented Generation).
    
    This function:
    1. Retrieves relevant context from the vector store
    2. Constructs a prompt with context and chat history
    3. Generates a response using the LLM
    
    Args:
        llm: The language model instance
        retriever: FAISS retriever for document search
        user_input: The user's question
        chat_history_langchain: List of (role, content) tuples in LangChain format
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
