"""
Portfolio API Router

Provides the /portfolio/chat endpoint for answering questions
about Mitesh Gupta using pre-ingested portfolio knowledge-base documents.

Fully isolated from the main RAG, agent, and chat routes.
Uses the same PostgreSQL-backed chat history infrastructure.

Includes structured LLM-based intent routing to avoid unnecessary
retrieval for greetings, follow-up questions.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from ..schemas.portfolio import PortfolioChatRequest, PortfolioChatResponse
from ..core.deps import get_chat_history_store, get_llm
from ..core.chat_history import PostgresChatHistoryStore

router = APIRouter(tags=["Portfolio"])

# ============================================================================
# STRUCTURED INTENT SCHEMA
# ============================================================================


class IntentSchema(BaseModel):
    """Schema for structured intent classification output."""

    intent: Literal["greeting", "followup", "profile_query"]


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

PORTFOLIO_SYSTEM_PROMPT = (
    "You are an assistant that answers questions about Mitesh Gupta using only the provided context. "
    "Use the context to generate clear, well-structured, and natural responses. "
    "You may summarize and combine information from multiple context sections if relevant and provided in context. "
    "Do not fabricate information or add external knowledge. "
    "If the answer is not present in the context, respond exactly with: "
    "This information is not available in Mitesh\'s profile."
    "When appropriate, organize the answer clearly (for example: overview first, then projects, skills, or experience). "
    "Write in a professional but conversational tone. "
    "Use emojis if needed."
    "Do not mention /'based on the provided context/' in your response. Just answer the question as naturally as possible using the context."
)

GREETING_SYSTEM_PROMPT = (
    "You are an assistant representing Mitesh Gupta. "
    "Respond naturally and professionally to greetings or small talk. "
    "Keep responses short and friendly."
)

PORTFOLIO_ROUTER_PROMPT = (
    "You are a classifier.\n\n"
    "Classify the user message into one of the following categories:\n\n"
    "- greeting\n"
    "- followup\n"
    "- profile_query\n\n"
    "Definitions:\n"
    "- greeting: simple greetings, small talk, or thanks.\n"
    "- followup: short references to previous response.\n"
    "- profile_query: questions about Mitesh Gupta's experience, skills, projects, education, or background.\n"
    "(e.g., general knowledge, politics, weather, math, coding problems).\n\n"
    "Respond strictly using the defined schema."
)



# ============================================================================
# INTENT CLASSIFIER
# ============================================================================


def classify_intent(message: str) -> str:
    """
    Classify a user message using structured LLM output.

    Uses a lightweight LLM call with temperature=0 and structured output
    enforced via Pydantic schema (IntentSchema). Falls back to
    'profile_query' on unexpected output or errors.

    Args:
        message: The user's message text.

    Returns:
        One of 'greeting', 'followup', 'profile_query'.
    """
    try:
        llm = get_llm(temperature=0, max_tokens=10)
        structured_llm = llm.with_structured_output(IntentSchema)
        prompt = f"{PORTFOLIO_ROUTER_PROMPT}\n\nUser message: {message}"
        result: IntentSchema = structured_llm.invoke(prompt)
        return result.intent
    except Exception:
        return "profile_query"


# ============================================================================
# ENDPOINT
# ============================================================================


@router.post(
    "/chat",
    response_model=PortfolioChatResponse,
    summary="Chat about Mitesh Gupta's portfolio",
    description="Ask a question about Mitesh Gupta. Answers are grounded in pre-ingested portfolio documents.",
)
def portfolio_chat(
    request: PortfolioChatRequest,
    req: Request,
    history_store: PostgresChatHistoryStore = Depends(get_chat_history_store),
) -> PortfolioChatResponse:
    """
    Handle a portfolio chat query with persistent history and intent routing.

    1. Validates request (session_id and message).
    2. Classifies intent via structured LLM call.
    3. Routes to appropriate handler:
       - greeting:       respond without retrieval.
       - followup:       respond using chat history, no retrieval.
       - profile_query:  retrieve context and generate RAG response.
    4. Persists messages to history.
    5. Returns the response.
    """
    # Validate session_id
    if not request.session_id or not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id must not be empty.")

    # Validate non-empty message
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be empty.")

    # Ensure portfolio retriever is available
    retriever = getattr(req.app.state, "portfolio_retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=500,
            detail="Portfolio retriever is not initialized.",
        )

    message = request.message.strip()
    session_id = request.session_id.strip()

    try:
        # Set session context and persist user message
        history_store.set_session(session_id)
        history_store.add_message("user", message)

        # Classify intent using structured output
        intent = classify_intent(message)

        # Get default LLM for response generation
        llm = get_llm()

        if intent == "greeting":
            # ── Greeting: no retrieval ──
            chat_history_langchain = history_store.get_langchain_messages()
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", GREETING_SYSTEM_PROMPT),
                    *chat_history_langchain[:-1],
                    ("human", "{message}"),
                ]
            )
            chain = prompt | llm
            result = chain.invoke({"message": message})

        elif intent == "followup":

            # ── Follow-up: use chat history, no retrieval ──
            chat_history_langchain = history_store.get_langchain_messages()
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", PORTFOLIO_SYSTEM_PROMPT),
                    *chat_history_langchain[:-1],
                    ("human", "{message}"),
                ]
            )
            chain = prompt | llm
            result = chain.invoke({"message": message})

        else:

            # ── Profile query: full RAG retrieval ──
            chat_history_langchain = history_store.get_langchain_messages()
            context_docs = retriever.invoke(message)
            context_text = "\n\n".join(doc.page_content for doc in context_docs)

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", PORTFOLIO_SYSTEM_PROMPT),
                    *chat_history_langchain[:-1],
                    ("human", "Context:\n{context}\n\nQuestion: {question}"),
                ]
            )
            chain = prompt | llm
            result = chain.invoke({"context": context_text, "question": message})

        # Persist assistant response
        history_store.add_message("assistant", result.content)

        return PortfolioChatResponse(response=result.content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate portfolio response: {str(e)}",
        )
