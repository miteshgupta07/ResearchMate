"""
Portfolio API Router

Provides the /portfolio/chat endpoint for answering questions
about Mitesh Gupta using pre-ingested portfolio knowledge-base documents.

Fully isolated from the main RAG, agent, and chat routes.
Uses the same PostgreSQL-backed chat history infrastructure.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.prompts import ChatPromptTemplate

from ..schemas.portfolio import PortfolioChatRequest, PortfolioChatResponse
from ..core.deps import get_chat_history_store, get_llm
from ..core.chat_history import PostgresChatHistoryStore

router = APIRouter(tags=["Portfolio"])

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
)


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
    Handle a portfolio chat query with persistent history.

    1. Validates request (session_id and message).
    2. Sets session context and adds user message to history.
    3. Retrieves top-3 context chunks from the portfolio retriever.
    4. Builds a prompt with portfolio system instruction, context, and chat history.
    5. Calls the default LLM.
    6. Persists assistant response to history.
    7. Returns the response.
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

    try:
        # Set session context
        history_store.set_session(request.session_id.strip())

        # Add user message to history
        history_store.add_message("user", request.message.strip())

        # Get chat history in LangChain format
        chat_history_langchain = history_store.get_langchain_messages()

        # Retrieve top-3 relevant chunks
        context_docs = retriever.invoke(request.message.strip())
        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        # Build prompt including chat history
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PORTFOLIO_SYSTEM_PROMPT),
                *chat_history_langchain[:-1],  # prior turns (exclude current user msg)
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        # Get default LLM (no dynamic model selection)
        llm = get_llm()

        # Generate response
        chain = prompt | llm
        result = chain.invoke({"context": context_text, "question": request.message.strip()})

        # Persist assistant response to history
        history_store.add_message("assistant", result.content)

        return PortfolioChatResponse(response=result.content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate portfolio response: {str(e)}",
        )
