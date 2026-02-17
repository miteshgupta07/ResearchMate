"""
Portfolio API Router

Provides the /portfolio/chat endpoint for answering questions
about Mitesh Gupta using pre-ingested portfolio knowledge-base documents.

Fully isolated from the main RAG, agent, and chat routes.
"""

from fastapi import APIRouter, HTTPException, Request
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from ..schemas.portfolio import PortfolioChatRequest, PortfolioChatResponse
from ..core.deps import get_llm

router = APIRouter(tags=["Portfolio"])

PORTFOLIO_SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly about Mitesh Gupta. "
    "Use only the provided context to answer. "
    "Do not fabricate information. "
    "If the answer is not present in the context, respond exactly with: "
    '"This information is not available in Mitesh\'s profile." '
    "Keep responses concise and factual."
)


@router.post(
    "/chat",
    response_model=PortfolioChatResponse,
    summary="Chat about Mitesh Gupta's portfolio",
    description="Ask a question about Mitesh Gupta. Answers are grounded in pre-ingested portfolio documents.",
)
def portfolio_chat(request: PortfolioChatRequest, req: Request) -> PortfolioChatResponse:
    """
    Handle a portfolio chat query.

    1. Validates the message is not empty.
    2. Retrieves top-3 context chunks from the portfolio retriever.
    3. Builds a prompt with the portfolio system instruction and context.
    4. Calls the default LLM via the existing service.
    5. Returns the response.
    """
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
        # Retrieve top-3 relevant chunks
        context_docs = retriever.invoke(request.message.strip())
        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        # Build prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PORTFOLIO_SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        # Get default LLM (no dynamic model selection)
        llm = get_llm()

        # Generate response
        chain = prompt | llm
        result = chain.invoke({"context": context_text, "question": request.message.strip()})

        return PortfolioChatResponse(response=result.content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate portfolio response: {str(e)}",
        )
