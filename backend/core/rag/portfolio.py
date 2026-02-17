"""
Portfolio RAG Module

Builds a FAISS retriever from pre-existing markdown files in the portfolio_kb/ directory.
This retriever is loaded once at FastAPI startup and used for portfolio-related queries.

Fully isolated from the main RAG pipeline — no document ingestion, no uploads, no persistence.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.core.rag.pipeline import _get_embeddings

# Path to the portfolio knowledge-base directory (relative to project root)
PORTFOLIO_KB_DIR = Path("portfolio_kb")


def build_portfolio_retriever():
    """
    Build a FAISS retriever from all markdown files in portfolio_kb/.

    Uses the same chunking strategy and embedding model as the main RAG pipeline
    (chunk_size=500, overlap=50, all-MiniLM-L6-v2).

    Returns:
        A LangChain FAISS retriever configured for top-3 retrieval.

    Raises:
        RuntimeError: If portfolio_kb/ directory is missing or contains no .md files.
    """
    if not PORTFOLIO_KB_DIR.exists() or not PORTFOLIO_KB_DIR.is_dir():
        raise RuntimeError(
            f"Portfolio knowledge-base directory not found: {PORTFOLIO_KB_DIR.resolve()}"
        )

    md_files = sorted(PORTFOLIO_KB_DIR.glob("*.md"))
    if not md_files:
        raise RuntimeError(
            f"No markdown files found in {PORTFOLIO_KB_DIR.resolve()}"
        )

    # Load markdown files into LangChain Document objects
    documents = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        documents.append(
            Document(page_content=content, metadata={"source": md_file.name})
        )

    # Chunk using same strategy as main RAG pipeline
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)

    # Build FAISS index using shared embedding model
    embeddings = _get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    # Return retriever with top-10 results
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever
