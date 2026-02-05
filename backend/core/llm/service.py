"""
LLM Service Module

This module handles all interactions with language models.
It provides clean interfaces for chat and RAG-based responses.
"""

import os
from typing import Optional, Dict, Any, List, Iterator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

# ============================================================================
# LLM CONFIGURATION DEFAULTS (Single Source of Truth)
# ============================================================================

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512

# Auto-continuation configuration
MAX_CONTINUATION_ATTEMPTS = 2
CONTINUATION_MAX_TOKENS = 512
CONTINUATION_PROMPT = "Continue from where you left off. Do not repeat what was already said."


# ============================================================================
# AUTO-CONTINUATION WRAPPER
# ============================================================================

class AutoContinueLLM(Runnable):
    """
    Wrapper that handles auto-continuation for truncated LLM responses.
    
    This class wraps a ChatGroq instance (or bound runnable) and intercepts
    invoke calls to automatically continue generation when responses are
    truncated due to token limits.
    
    The wrapper is transparent to callers - they receive a complete response
    without knowing continuation occurred.
    
    Inherits from Runnable to be compatible with LangChain Expression Language (LCEL).
    """
    
    def __init__(
        self,
        base_llm: ChatGroq,
        max_tokens: int,
        temperature: float
    ):
        """
        Initialize the auto-continuation wrapper.
        
        Args:
            base_llm: The underlying ChatGroq instance
            max_tokens: The max_tokens setting for initial generation
            temperature: The temperature setting for generation
        """
        self._base_llm = base_llm
        self._max_tokens = max_tokens
        self._temperature = temperature
    
    @property
    def InputType(self):
        """Return input type for LCEL compatibility."""
        return Any
    
    @property
    def OutputType(self):
        """Return output type for LCEL compatibility."""
        return Any
    
    def _is_truncated(self, response) -> bool:
        """
        Detect if a response was truncated due to token limits.
        
        Uses two detection strategies:
        1. Provider finish_reason == "length" (preferred)
        2. Heuristic: response doesn't end with . ? !
        
        Args:
            response: The LLM response object
            
        Returns:
            True if the response appears truncated, False otherwise
        """
        content = getattr(response, 'content', '')
        if not content or not content.strip():
            return False
        
        # Strategy 1: Check finish_reason from response metadata
        if hasattr(response, 'response_metadata'):
            finish_reason = response.response_metadata.get('finish_reason', '')
            if finish_reason == 'length':
                return True
        
        # Strategy 2: Heuristic - check if response ends cleanly
        clean_content = content.strip()
        if clean_content and clean_content[-1] not in '.?!':
            return True
        
        return False
    
    def _request_continuation(self, partial_response: str) -> str:
        """
        Request a continuation of a truncated response.
        
        Sends a follow-up request with the partial response as context
        and asks the model to continue from where it left off.
        
        Args:
            partial_response: The truncated response text
            
        Returns:
            The continuation text (may be empty if no meaningful content added)
        """
        # Create continuation messages with context
        continuation_messages = [
            AIMessage(content=partial_response),
            HumanMessage(content=CONTINUATION_PROMPT)
        ]
        
        # Use lower token limit for continuation
        continuation_llm = self._base_llm.bind(
            max_tokens=CONTINUATION_MAX_TOKENS,
            temperature=self._temperature
        )
        
        try:
            continuation_response = continuation_llm.invoke(continuation_messages)
            return getattr(continuation_response, 'content', '')
        except Exception:
            # On any error during continuation, return empty string
            return ''
    
    def _handle_continuation(self, initial_response) -> Any:
        """
        Handle auto-continuation for a potentially truncated response.
        
        If the initial response is truncated, this method will:
        1. Request continuations (up to MAX_CONTINUATION_ATTEMPTS)
        2. Concatenate all parts into a complete response
        3. Stop early if response ends cleanly or no new tokens added
        
        Args:
            initial_response: The initial LLM response object
            
        Returns:
            The response object with potentially extended content
        """
        if not self._is_truncated(initial_response):
            return initial_response
        
        full_content = initial_response.content
        
        for _ in range(MAX_CONTINUATION_ATTEMPTS):
            continuation_text = self._request_continuation(full_content)
            
            # Stop if no meaningful tokens added
            if not continuation_text or not continuation_text.strip():
                break
            
            # Concatenate continuation
            full_content += continuation_text
            
            # Stop if response now ends cleanly
            clean_content = full_content.strip()
            if clean_content and clean_content[-1] in '.?!':
                break
        
        # Update response content with full text
        initial_response.content = full_content
        return initial_response
    
    def invoke(
        self, 
        input: Any, 
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Any:
        """
        Invoke the LLM with auto-continuation support.
        
        This method is compatible with LangChain Expression Language (LCEL).
        
        Args:
            input: The input to the LLM (messages, prompt, etc.)
            config: Optional RunnableConfig for LCEL compatibility
            **kwargs: Additional arguments passed to underlying LLM
            
        Returns:
            LLM response with auto-continuation applied if needed
        """
        # Get bound LLM with parameters
        configured_llm = self._base_llm.bind(
            max_tokens=self._max_tokens,
            temperature=self._temperature
        )
        
        # Invoke underlying LLM
        response = configured_llm.invoke(input, config, **kwargs)
        
        # Handle potential continuation
        return self._handle_continuation(response)
    
    def bind(self, **kwargs) -> "AutoContinueLLM":
        """
        Return a new AutoContinueLLM with updated parameters.
        
        This allows callers to further customize the LLM while
        maintaining auto-continuation support.
        
        Args:
            **kwargs: Parameters to bind (temperature, max_tokens, etc.)
            
        Returns:
            New AutoContinueLLM instance with updated parameters
        """
        new_max_tokens = kwargs.pop('max_tokens', self._max_tokens)
        new_temperature = kwargs.pop('temperature', self._temperature)
        
        # If other kwargs remain, bind them to base LLM
        if kwargs:
            new_base = self._base_llm.bind(**kwargs)
        else:
            new_base = self._base_llm
        
        return AutoContinueLLM(new_base, new_max_tokens, new_temperature)
    
    def __getattr__(self, name: str) -> Any:
        """
        Proxy attribute access to the underlying LLM.
        
        This ensures compatibility with code that accesses
        LLM attributes directly.
        """
        return getattr(self._base_llm, name)


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
    ) -> AutoContinueLLM:
        """
        Get a configured LLM instance with per-request parameters.
        
        This method:
        1. Resolves the frontend model name to backend identifier
        2. Gets the pre-initialized model from the registry
        3. Returns an AutoContinueLLM wrapper with temperature and max_tokens applied
        
        The returned wrapper handles auto-continuation transparently when
        responses are truncated due to token limits.
        
        Args:
            model_type: Frontend model name (e.g., "LLaMA 3.1-8B")
            temperature: Controls randomness in responses (0.0-1.0), defaults to 0.7
            max_tokens: Maximum tokens in generated response, defaults to 512
        
        Returns:
            AutoContinueLLM wrapper with auto-continuation support
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
        
        # Return AutoContinueLLM wrapper for transparent continuation handling
        return AutoContinueLLM(
            base_llm=base_model,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature
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
