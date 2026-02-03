"""
LLM Service Module

This module handles all interactions with language models.
Provides clean interfaces for chat and RAG-based responses.
"""

from .service import (
    create_llm,
    create_normal_chat_prompt,
    create_rag_chat_prompt,
    generate_chat_response,
    resolve_model_name,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    MODEL_MAPPING,
    LLMRegistry,
    get_llm_registry,
    initialize_llm_registry,
)

__all__ = [
    "create_llm",
    "create_normal_chat_prompt",
    "create_rag_chat_prompt",
    "generate_chat_response",
    "resolve_model_name",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "MODEL_MAPPING",
    "LLMRegistry",
    "get_llm_registry",
    "initialize_llm_registry",
]
