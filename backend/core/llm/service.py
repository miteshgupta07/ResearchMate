"""
LLM Service Module

This module handles all interactions with language models.
It provides clean interfaces for chat and RAG-based responses.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# ============================================================================
# LLM CONFIGURATION DEFAULTS (Single Source of Truth)
# ============================================================================

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512

# Frontend model name → Backend model identifier mapping
MODEL_MAPPING = {
    "DeepSeek": "deepseek-r1-distill-llama-70b",
    "DeepSeek r1": "deepseek-r1-distill-llama-70b",
    "LLaMA 3.1-8B": "llama-3.1-8b-instant",
    "Gemma2 9B": "gemma2-9b-it",
    "Mixtral": "mixtral-8x7b-32768",
}


def resolve_model_name(model_type: Optional[str]) -> str:
    """
    Resolve frontend model name to backend model identifier.
    
    Args:
        model_type: Frontend model name (e.g., "DeepSeek r1", "LLaMA 3.1-8B")
    
    Returns:
        Backend model identifier for the Groq API
    """
    if model_type is None:
        return DEFAULT_MODEL
    
    # Look up in mapping, fall back to default if not found
    return MODEL_MAPPING.get(model_type, DEFAULT_MODEL)


def create_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_type: Optional[str] = None
):
    """
    Create and configure a ChatGroq language model instance.
    
    Args:
        model_name: The model identifier for the Groq API
        temperature: Controls randomness in responses (0.0-1.0)
        max_tokens: Maximum tokens in generated response
    
    Returns:
        Configured ChatGroq model instance
    """
    model_name=resolve_model_name(model_name)
    return ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        verbose=False
    )


def create_normal_chat_prompt():
    """
    Create a prompt template for normal (non-RAG) chat interactions.
    
    Returns:
        ChatPromptTemplate configured for research-focused chat
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are a research-focused assistant. Provide detailed, evidence-based responses and reference credible sources when possible."),
        MessagesPlaceholder(variable_name="rag_messages"),
    ])


def create_rag_chat_prompt():
    """
    Create a prompt template for RAG-based chat interactions.
    
    Returns:
        ChatPromptTemplate configured for RAG with context
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Utilize the provided context to deliver accurate, well-researched, and evidence-backed responses. Ensure responses are aligned with academic and research standards."),
        ("human", "Context: {context}"),
        MessagesPlaceholder(variable_name="rag_messages"),
    ])


def generate_chat_response(llm, chat_history_langchain: list, language: str):
    """
    Generate a response using normal chat (without RAG context).
    
    Args:
        llm: The language model instance
        chat_history_langchain: List of (role, content) tuples in LangChain format
        language: Target language for the response
    
    Returns:
        Generated response text
    """
    try:
        prompt = create_normal_chat_prompt()
        chain = prompt | llm
        response = chain.invoke({
            "language": language,
            "rag_messages": chat_history_langchain
        })
    except Exception as e:
        import traceback
        print("ERROR MESSAGE:", str(e))
        print("FULL TRACEBACK:")
        traceback.print_exc()
        raise

    return response.content
