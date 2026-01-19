"""
LLM Service Module

This module handles all interactions with language models.
It provides clean interfaces for chat and RAG-based responses.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


def create_llm(model_name: str, temperature: float = 0.7, max_tokens: int = 512):
    """
    Create and configure a ChatGroq language model instance.
    
    Args:
        model_name: The model identifier for the Groq API
        temperature: Controls randomness in responses (0.0-1.0)
        max_tokens: Maximum tokens in generated response
    
    Returns:
        Configured ChatGroq model instance
    """
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
    prompt = create_normal_chat_prompt()
    chain = prompt | llm
    
    response = chain.invoke({
        "language": language,
        "rag_messages": chat_history_langchain
    })
    
    return response.content
