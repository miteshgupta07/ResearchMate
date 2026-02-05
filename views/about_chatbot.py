import streamlit as st

# Scoped CSS for layout customization
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 10rem;
        padding-right: 10rem;
        max-width: 1400px;
    }
    .content-wrapper {
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Setting the title for the page
st.markdown(
    "<h1 style='text-align: center;'>About ResearchMate 🤖</h1>",
    unsafe_allow_html=True
)

# Introduction Section
st.header("🚀 Introduction")
st.write("""
ResearchMate is a production-ready AI research assistant built on Retrieval-Augmented Generation (RAG) principles. 
Developed by Mitesh Gupta, it combines a FastAPI backend with a Streamlit frontend to deliver accurate, 
document-grounded answers to academic and research queries.

The system is designed with scalability and reliability in mind, featuring persistent storage, 
evaluated retrieval pipelines, and flexible LLM integration to support serious research workflows.
""")

# Purpose Section
st.header("🎯 Purpose")
st.write("""
ResearchMate is designed to:
- Enable **document-grounded question answering** over uploaded research papers.
- Improve **research productivity** by providing contextually relevant, citation-backed responses.
- Ensure **accuracy, grounding, and transparency** by retrieving information directly from ingested documents.
- Support **user-controlled agent workflows** for real-time retrieval from external sources like arXiv.
""")

# Architecture & Technologies Section
st.header("🏗️ Architecture & Technologies")
st.write("""
ResearchMate follows a decoupled client-server architecture:
""")
st.markdown("""
**Backend (FastAPI)**
- RESTful API orchestrating RAG pipelines, chat sessions, and document management.
- Asynchronous request handling for responsive performance.

**Frontend (Streamlit)**
- Interactive chat interface with session management.
- Document upload and ingestion controls.
- Dynamic configuration for model selection and generation parameters.

**Data & Storage**
- **PostgreSQL**: Persistent chat history with session isolation.
- **FAISS**: High-performance vector similarity search for document retrieval.

**Language Models**
- Integration with **HuggingFace** and **Groq** APIs.
- Dynamic model selection, temperature, and token limit configuration at runtime.
""")

# Features Section
st.header("🗝️ Key Features")
st.markdown("""
- **Persistent Document Ingestion**: Upload and index research papers for future querying across sessions.
- **Evaluated RAG Pipeline**: Retrieval quality is measured using standard metrics (e.g., Recall@K) to ensure reliable results.
- **Agent Mode Toggle**: Optionally enable an AI agent for real-time arXiv retrieval, activated manually by the user.
- **Dynamic LLM Configuration**: Select models, adjust temperature, and control max tokens directly from the interface.
- **Auto-Continuation**: Long responses are automatically continued to provide complete answers.
- **Chat History Persistence**: Conversations are stored in PostgreSQL, enabling context retention across sessions.
""")

# Evaluation & Reliability Section
st.header("📊 Evaluation & Reliability")
st.write("""
ResearchMate incorporates evaluation practices to ensure retrieval and response quality:
- RAG pipeline performance is assessed using open benchmarks and standard retrieval metrics.
- Retrieval accuracy and answer grounding are measured to validate that responses are backed by source documents.
- The system is designed with a focus on reducing hallucinations and improving trust in generated outputs.

This evaluation-driven approach ensures that ResearchMate delivers dependable results for research-critical tasks.
""")

# Benefits Section
st.header("🚨 Benefits")
st.write("""
- **Reliable Research Assistance**: Responses are grounded in uploaded documents, reducing speculation.
- **Transparent Answers**: Users can trace information back to source materials.
- **Scalable Architecture**: Decoupled backend and frontend enable independent scaling and extension.
- **Extensible Design**: Modular components allow integration of new models, storage backends, or retrieval methods.
""")

# Closing Section
st.markdown("---")
st.markdown("**ResearchMate is built to support serious research workflows with accuracy, transparency, and scalability at its core.**")
