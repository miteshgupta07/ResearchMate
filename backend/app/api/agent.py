"""
Agent API Router

Provides the /agent/route endpoint for deterministic intent-based routing.
This is a production-grade single-pass agent that classifies intent and routes
to exactly one capability with no loops, retries, or recursion.

Pipeline: PreChecks → IntentClassifier → CapabilityRouter → ResponseAssembler
"""

import re
import arxiv
from typing import Optional, Tuple, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException

from ..schemas.agent import AgentRequest, AgentResponse
from ..core.deps import (
    get_chat_history_store,
    get_llm,
    PostgresChatHistoryStore
)

router = APIRouter(prefix="/agent", tags=["Agent"])

# ============================================================================
# INTENT CONSTANTS
# ============================================================================

INTENT_CHAT = "INTENT_CHAT"
INTENT_RAG_QA = "INTENT_RAG_QA"
INTENT_RAG_SUMMARY = "INTENT_RAG_SUMMARY"
INTENT_ARXIV_SEARCH = "INTENT_ARXIV_SEARCH"
INTENT_ARXIV_RECOMMEND = "INTENT_ARXIV_RECOMMEND"
INTENT_FALLBACK = "INTENT_FALLBACK"

ALLOWED_INTENTS = {
    INTENT_CHAT,
    INTENT_RAG_QA,
    INTENT_RAG_SUMMARY,
    INTENT_ARXIV_SEARCH,
    INTENT_ARXIV_RECOMMEND,
    INTENT_FALLBACK,
}

# Greeting patterns for PreChecks
GREETING_PATTERNS = [
    r"^hi\b",
    r"^hello\b",
    r"^hey\b",
    r"^greetings\b",
    r"^good\s*(morning|afternoon|evening|day)\b",
    r"^howdy\b",
    r"^hola\b",
    r"^bonjour\b",
    r"^namaste\b",
    r"^hallo\b",
]

# ============================================================================
# STEP 1: PRECHECKS (No LLM, pure rule-based)
# ============================================================================

def prechecks(message: str, document_id: Optional[str]) -> Optional[Tuple[str, float]]:
    """
    Run pre-checks to determine intent without LLM.
    
    Rules:
    - If message is greeting or very short → INTENT_CHAT
    - If document_id is null → block RAG intents (let classifier decide between CHAT/ARXIV)
    - If message is generic and document exists → INTENT_CHAT
    
    Args:
        message: The user's message
        document_id: Optional document ID
    
    Returns:
        Tuple of (intent, confidence) if decided, None otherwise
    """
    message_lower = message.lower().strip()
    
    # Rule 1: Greeting patterns → INTENT_CHAT
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, message_lower, re.IGNORECASE):
            return (INTENT_CHAT, 1.0)
    
    # Rule 2: Very short messages (less than 5 characters) → INTENT_CHAT
    if len(message_lower) < 5:
        return (INTENT_CHAT, 1.0)
    
    # Rule 3: Single word messages that are generic → INTENT_CHAT
    if len(message_lower.split()) == 1 and message_lower in {
        "yes", "no", "ok", "okay", "thanks", "thank", "bye", "goodbye", "sure", "cool"
    }:
        return (INTENT_CHAT, 1.0)
    
    # No pre-check decision; proceed to classifier
    return None


# ============================================================================
# STEP 2: INTENT CLASSIFIER (Single LLM call, no tools, no retries)
# ============================================================================

CLASSIFIER_PROMPT = """You are an intent classifier. Classify the user's message into exactly ONE of the following intents.

ALLOWED INTENTS:
- INTENT_CHAT: General conversation, greetings, or questions not requiring document/paper search
- INTENT_RAG_QA: Questions about a specific uploaded document (requires document context)
- INTENT_RAG_SUMMARY: Request to summarize an uploaded document (requires document context)
- INTENT_ARXIV_SEARCH: Request to search for academic papers on arXiv
- INTENT_ARXIV_RECOMMEND: Request for paper recommendations based on a topic

CONTEXT:
- Document available: {has_document}

RULES:
1. If document is NOT available, do NOT choose INTENT_RAG_QA or INTENT_RAG_SUMMARY
2. For paper search requests (e.g., "find papers about X", "search for research on Y"), choose INTENT_ARXIV_SEARCH
3. For paper recommendations (e.g., "recommend papers on X", "suggest research about Y"), choose INTENT_ARXIV_RECOMMEND
4. For questions about an uploaded document, choose INTENT_RAG_QA
5. For summary requests about an uploaded document, choose INTENT_RAG_SUMMARY
6. For general conversation or questions, choose INTENT_CHAT

USER MESSAGE: {message}

Respond with ONLY the intent label (e.g., INTENT_CHAT). No explanation."""


def classify_intent(llm, message: str, document_id: Optional[str]) -> Tuple[str, float]:
    """
    Classify user intent using a single LLM call.
    
    Args:
        llm: The language model instance
        message: The user's message
        document_id: Optional document ID
    
    Returns:
        Tuple of (intent, confidence)
    """
    has_document = "YES" if document_id else "NO"
    
    prompt_text = CLASSIFIER_PROMPT.format(
        has_document=has_document,
        message=message
    )
    
    # Single LLM call - no retries
    response = llm.invoke(prompt_text)
    raw_intent = response.content.strip().upper()
    
    # Extract intent from response (handle potential extra text)
    for intent in ALLOWED_INTENTS:
        if intent in raw_intent:
            # Block RAG intents if no document
            if document_id is None and intent in {INTENT_RAG_QA, INTENT_RAG_SUMMARY}:
                return (INTENT_CHAT, 0.8)
            return (intent, 0.9)
    
    # Fallback if no valid intent found
    return (INTENT_FALLBACK, 0.5)


# ============================================================================
# STEP 3: CAPABILITY ROUTER (No reasoning, direct dispatch)
# ============================================================================

def route_to_chat(
    llm,
    history_store: PostgresChatHistoryStore,
    session_id: str,
    message: str,
    language: str
) -> Tuple[str, Dict[str, Any]]:
    """Route to chat capability."""
    # Set session context
    history_store.set_session(session_id)
    
    # Add user message to history
    history_store.add_message("user", message)
    
    # Get chat history in LangChain format
    chat_history_langchain = history_store.get_langchain_messages()
    
    # Generate response using core logic (no RAG)
    from core.rag_pipeline import answer_without_rag
    response_content = answer_without_rag(
        llm=llm,
        chat_history_langchain=chat_history_langchain,
        language=language
    )
    
    # Add assistant response to history
    history_store.add_message("assistant", response_content)
    
    return (response_content, {})


def route_to_rag_qa(
    llm,
    history_store: PostgresChatHistoryStore,
    session_id: str,
    document_id: str,
    message: str,
    language: str
) -> Tuple[str, Dict[str, Any]]:
    """Route to RAG Q&A capability."""
    from core.rag_pipeline import load_retriever, answer_query
    
    retriever = load_retriever(document_id)
    if retriever is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}"
        )
    
    # Set session context
    history_store.set_session(session_id)
    
    # Add user message to history
    history_store.add_message("user", message)
    
    # Get chat history in LangChain format
    chat_history_langchain = history_store.get_langchain_messages()
    
    # Generate response using RAG
    response_content = answer_query(
        llm=llm,
        retriever=retriever,
        chat_history_langchain=chat_history_langchain,
        user_input=message,
        language=language
    )
    
    # Add assistant response to history
    history_store.add_message("assistant", response_content)
    
    return (response_content, {"document_id": document_id})


def route_to_rag_summary(
    llm,
    history_store: PostgresChatHistoryStore,
    session_id: str,
    document_id: str,
    language: str
) -> Tuple[str, Dict[str, Any]]:
    """Route to RAG summary capability."""
    from core.rag_pipeline import load_retriever, answer_query
    
    retriever = load_retriever(document_id)
    if retriever is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}"
        )
    
    # Set session context
    history_store.set_session(session_id)
    
    # Construct summary request
    summary_prompt = "Please provide a comprehensive summary of this document, highlighting the key points, main arguments, and conclusions."
    
    # Add user message to history
    history_store.add_message("user", summary_prompt)
    
    # Get chat history in LangChain format
    chat_history_langchain = history_store.get_langchain_messages()
    
    # Generate summary using RAG
    response_content = answer_query(
        llm=llm,
        retriever=retriever,
        chat_history_langchain=chat_history_langchain,
        user_input=summary_prompt,
        language=language
    )
    
    # Add assistant response to history
    history_store.add_message("assistant", response_content)
    
    return (response_content, {"document_id": document_id, "type": "summary"})


def route_to_arxiv_search(
    message: str,
    language: str
) -> Tuple[str, Dict[str, Any]]:
    """Route to arXiv search capability."""
    # Extract search query from message
    search_query = _extract_search_query(message)
    
    # Search arXiv
    papers = _search_arxiv(search_query, max_results=5)
    
    if not papers:
        return (
            f"No papers found on arXiv for query: '{search_query}'. Try refining your search terms.",
            {"query": search_query, "papers": []}
        )
    
    # Format results
    response_content = _format_arxiv_results(papers, search_query)
    
    return (response_content, {"query": search_query, "papers": papers})


def route_to_arxiv_recommend(
    message: str,
    language: str
) -> Tuple[str, Dict[str, Any]]:
    """Route to arXiv recommend capability."""
    # Extract topic from message
    topic = _extract_recommendation_topic(message)
    
    # Search arXiv with relevance sorting
    papers = _search_arxiv(topic, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
    
    if not papers:
        return (
            f"No paper recommendations found for topic: '{topic}'. Try a different topic.",
            {"topic": topic, "papers": []}
        )
    
    # Format recommendations
    response_content = _format_arxiv_recommendations(papers, topic)
    
    return (response_content, {"topic": topic, "papers": papers})


def _extract_search_query(message: str) -> str:
    """Extract search query from user message."""
    # Remove common prefixes
    prefixes = [
        "search for papers",
        "search papers",
        "find papers",
        "look for papers",
        "search for research",
        "search research",
        "find research",
        "search for",
        "find",
        "look for",
        "search",
    ]
    
    query = message.lower().strip()
    for prefix in prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):].strip()
            break
    
    # Remove common suffixes
    suffixes = ["on arxiv", "from arxiv", "in arxiv"]
    for suffix in suffixes:
        if query.endswith(suffix):
            query = query[:-len(suffix)].strip()
            break
    
    # Clean up common words
    query = query.strip(".,!?\"'")
    
    return query if query else message


def _extract_recommendation_topic(message: str) -> str:
    """Extract recommendation topic from user message."""
    prefixes = [
        "recommend papers",
        "suggest papers",
        "recommend research",
        "suggest research",
        "recommend",
        "suggest",
        "papers on",
        "research on",
    ]
    
    topic = message.lower().strip()
    for prefix in prefixes:
        if topic.startswith(prefix):
            topic = topic[len(prefix):].strip()
            break
    
    # Remove common suffixes
    suffixes = ["on arxiv", "from arxiv", "please", "for me"]
    for suffix in suffixes:
        if topic.endswith(suffix):
            topic = topic[:-len(suffix)].strip()
            break
    
    topic = topic.strip(".,!?\"'")
    
    return topic if topic else message


def _search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate
) -> List[Dict[str, Any]]:
    """Search arXiv for papers."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary[:500] + "..." if len(result.summary) > 500 else result.summary,
                "published": result.published.strftime("%Y-%m-%d") if result.published else None,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
            })
        
        return papers
    except Exception:
        return []


def _format_arxiv_results(papers: List[Dict[str, Any]], query: str) -> str:
    """Format arXiv search results as readable text."""
    lines = [f"**Found {len(papers)} papers for '{query}':**\n"]
    
    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_str += f" et al."
        
        lines.append(f"**{i}. {paper['title']}**")
        lines.append(f"   Authors: {authors_str}")
        lines.append(f"   Published: {paper['published']}")
        lines.append(f"   Link: {paper['url']}")
        lines.append(f"   Summary: {paper['summary'][:200]}...")
        lines.append("")
    
    return "\n".join(lines)


def _format_arxiv_recommendations(papers: List[Dict[str, Any]], topic: str) -> str:
    """Format arXiv recommendations as readable text."""
    lines = [f"**Recommended papers on '{topic}':**\n"]
    
    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_str += f" et al."
        
        lines.append(f"**{i}. {paper['title']}**")
        lines.append(f"   Authors: {authors_str}")
        lines.append(f"   Published: {paper['published']}")
        lines.append(f"   Link: {paper['url']}")
        lines.append(f"   Summary: {paper['summary'][:200]}...")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# STEP 4: RESPONSE ASSEMBLER
# ============================================================================

def assemble_response(
    content: str,
    intent: str,
    confidence: float,
    metadata: Dict[str, Any]
) -> AgentResponse:
    """
    Assemble the final response.
    
    Args:
        content: The response content
        intent: The classified intent
        confidence: Confidence score
        metadata: Additional metadata
    
    Returns:
        AgentResponse object
    """
    return AgentResponse(
        role="assistant",
        content=content,
        intent=intent,
        confidence=confidence,
        metadata=metadata
    )


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@router.post(
    "/route",
    response_model=AgentResponse,
    summary="Route a message through the deterministic agent",
    description="""
    Single-pass deterministic agent that:
    1. Classifies user intent
    2. Routes to exactly ONE capability
    3. Executes once
    4. Returns response
    
    No loops, no retries, no recursion.
    """
)
def agent_route(
    request: AgentRequest,
    history_store: PostgresChatHistoryStore = Depends(get_chat_history_store)
) -> AgentResponse:
    """
    Handle agent routing with deterministic single-pass execution.
    
    Pipeline: PreChecks → IntentClassifier → CapabilityRouter → ResponseAssembler
    
    Args:
        request: Agent request with session_id, message, optional document_id, language, and LLM config
        history_store: Injected chat history store
    
    Returns:
        AgentResponse with intent, confidence, content, and metadata
    """
    try:
        # Get LLM instance with dynamic configuration
        llm = get_llm(
            model_type=request.model_type,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Extract request parameters
        session_id = request.session_id
        message = request.message
        document_id = request.document_id
        language = request.language or "English"
        
        # ================================================================
        # STEP 1: PRECHECKS (No LLM)
        # ================================================================
        precheck_result = prechecks(message, document_id)
        
        if precheck_result is not None:
            intent, confidence = precheck_result
        else:
            # ================================================================
            # STEP 2: INTENT CLASSIFIER (Single LLM call)
            # ================================================================
            intent, confidence = classify_intent(llm, message, document_id)
        
        # ================================================================
        # STEP 3: CAPABILITY ROUTER (Direct dispatch)
        # ================================================================
        if intent == INTENT_CHAT:
            content, metadata = route_to_chat(
                llm, history_store, session_id, message, language
            )
        
        elif intent == INTENT_RAG_QA:
            if document_id is None:
                # Safety check: fallback to chat if no document
                intent = INTENT_CHAT
                confidence = 0.7
                content, metadata = route_to_chat(
                    llm, history_store, session_id, message, language
                )
            else:
                content, metadata = route_to_rag_qa(
                    llm, history_store, session_id, document_id, message, language
                )
        
        elif intent == INTENT_RAG_SUMMARY:
            if document_id is None:
                # Safety check: fallback to chat if no document
                intent = INTENT_CHAT
                confidence = 0.7
                content, metadata = route_to_chat(
                    llm, history_store, session_id, message, language
                )
            else:
                content, metadata = route_to_rag_summary(
                    llm, history_store, session_id, document_id, language
                )
        
        elif intent == INTENT_ARXIV_SEARCH:
            content, metadata = route_to_arxiv_search(message, language)
        
        elif intent == INTENT_ARXIV_RECOMMEND:
            content, metadata = route_to_arxiv_recommend(message, language)
        
        elif intent == INTENT_FALLBACK:
            # Fallback routes to chat
            content, metadata = route_to_chat(
                llm, history_store, session_id, message, language
            )
        
        else:
            # Unknown intent: fallback to chat
            intent = INTENT_FALLBACK
            confidence = 0.5
            content, metadata = route_to_chat(
                llm, history_store, session_id, message, language
            )
        
        # ================================================================
        # STEP 4: RESPONSE ASSEMBLER
        # ================================================================
        return assemble_response(content, intent, confidence, metadata)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent routing failed: {str(e)}"
        )
