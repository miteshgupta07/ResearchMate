"""
LLM Service Module

This module handles all interactions with language models.
It provides clean interfaces for chat and RAG-based responses.
"""

import os
from typing import Optional, Dict
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
    "LLaMA 3.1-8B": "llama-3.1-8b-instant",
    "Gemma2 9B": "gemma2-9b-it",
    "Mixtral": "mixtral-8x7b-32768",
}


# ============================================================================
# LLM REGISTRY (Pre-initialized Model Cache)
# ============================================================================

class LLMRegistry:
    """
    Central registry for pre-initialized LLM instances.
    
    This registry holds initialized ChatGroq instances keyed by model identifier.
    Models are initialized once at FastAPI startup and reused across all requests.
    
    Note: temperature and max_tokens are NOT baked into the registry.
    They are applied per-request via the get_llm_with_params() method.
    """
    
    def __init__(self):
        self._models: Dict[str, ChatGroq] = {}
        self._initialized: bool = False
    
    def initialize_all_models(self) -> None:
        """
        Initialize all supported models at startup.
        
        This method creates ChatGroq instances for all models in MODEL_MAPPING.
        Models are initialized with streaming enabled but WITHOUT temperature/max_tokens
        since those are applied per-request.
        """
        if self._initialized:
            return
        
        api_key = os.getenv("GROQ_API_KEY")
        
        # Get all unique model identifiers from the mapping
        model_identifiers = set(MODEL_MAPPING.values())
        
        # Also add the default model if not already in mapping values
        model_identifiers.add(DEFAULT_MODEL)
        
        for model_id in model_identifiers:
            self._models[model_id] = ChatGroq(
                model=model_id,
                api_key=api_key,
                streaming=True,
                verbose=False
            )
        
        self._initialized = True
    
    def get_model(self, model_identifier: str) -> Optional[ChatGroq]:
        """
        Get a pre-initialized model by its identifier.
        
        Args:
            model_identifier: The backend model identifier (e.g., "llama-3.1-8b-instant")
        
        Returns:
            The pre-initialized ChatGroq instance, or None if not found
        """
        return self._models.get(model_identifier)
    
    def get_llm_with_params(
        self,
        model_type: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> ChatGroq:
        """
        Get a configured LLM instance with per-request parameters.
        
        This method:
        1. Resolves the frontend model name to backend identifier
        2. Gets the pre-initialized model from the registry
        3. Returns a configured instance with temperature and max_tokens applied
        
        Args:
            model_type: Frontend model name (e.g., "LLaMA 3.1-8B")
            temperature: Controls randomness in responses (0.0-1.0), defaults to 0.7
            max_tokens: Maximum tokens in generated response, defaults to 512
        
        Returns:
            Configured ChatGroq model instance with per-request parameters
        """
        # Resolve frontend model name to backend identifier
        model_identifier = resolve_model_name(model_type)
        
        # Apply defaults for temperature and max_tokens
        effective_temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        effective_max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        
        # Get pre-initialized model
        base_model = self._models.get(model_identifier)
        
        if base_model is None:
            # Fallback: if model not in registry, use default
            base_model = self._models.get(DEFAULT_MODEL)
        
        # Apply per-request parameters by binding them
        # ChatGroq supports .bind() to create a runnable with bound kwargs
        return base_model.bind(
            temperature=effective_temperature,
            max_tokens=effective_max_tokens
        )
    
    @property
    def is_initialized(self) -> bool:
        """Check if the registry has been initialized."""
        return self._initialized
    
    @property
    def available_models(self) -> list:
        """Get list of available model identifiers."""
        return list(self._models.keys())


# Global singleton instance of the LLM registry
_llm_registry: Optional[LLMRegistry] = None


def get_llm_registry() -> LLMRegistry:
    """
    Get the global LLMRegistry singleton instance.
    
    Returns:
        The singleton LLMRegistry instance
    """
    global _llm_registry
    if _llm_registry is None:
        _llm_registry = LLMRegistry()
    return _llm_registry


def initialize_llm_registry() -> None:
    """
    Initialize the global LLM registry.
    
    This should be called once at FastAPI startup to pre-initialize all models.
    """
    registry = get_llm_registry()
    registry.initialize_all_models()


def resolve_model_name(model_type: Optional[str]) -> str:
    """
    Resolve frontend model name to backend model identifier.
    
    Args:
        model_type: Frontend model name (e.g. "LLaMA 3.1-8B")
    
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
