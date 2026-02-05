# ResearchMate: Your Free AI Research Assistant 📚🤖

**ResearchMate** is a production-ready AI research assistant designed to streamline academic and research work through accurate, document-grounded question answering over research papers. Built with a scalable backend architecture for retrieval-augmented generation (RAG) and conversational interactions, it enables real-time retrieval of research papers, intelligent summarization, and insightful recommendations for related work—making it a reliable and indispensable tool for researchers and students alike.

---

## 🌍 Deployment

Research Mate can be accessed on [**https://researchmate.me/**](https://researchmate.me/) for a smooth and interactive user experience.

---

## 🚀 Key Features

- **PDF Document Ingestion & Querying**: Upload research papers and ask contextual questions with retrieval-augmented generation
- **Persistent Chat History**: PostgreSQL-backed conversation storage for maintaining session context across interactions
- **Agent Workflow**: Optional agent mode for multi-step reasoning, arXiv search, and paper summarization (user-controlled toggle)
- **Dynamic LLM Configuration**: Runtime control over model selection, temperature, and token limits
- **Auto-Continuation**: Automatic handling of long-form responses that exceed model context windows
- **Evaluated RAG Pipeline**: Retrieval accuracy and answer groundedness measured against open benchmarks to reduce hallucinations

---

## 🏗️ System Architecture

ResearchMate follows a decoupled frontend-backend architecture:

- **Frontend**: Streamlit-based user interface for document upload, chat interaction, and configuration controls
- **Backend**: FastAPI orchestration layer exposing RESTful endpoints for chat, document management, and history
- **Vector Store**: FAISS for efficient similarity search and document retrieval
- **Persistence Layer**: PostgreSQL for durable chat history and session management
- **LLM Service**: Abstracted model registry supporting multiple providers (Groq, HuggingFace, AWS Bedrock)

The backend API is deployed at `https://api.researchmate.me/` and handles all core logic, while the frontend acts as a lightweight client.

---

## 📊 RAG Evaluation & Reliability

ResearchMate emphasizes trustworthy information retrieval:

- **Retrieval Quality**: Evaluated using Recall@K metrics on open research paper benchmarks
- **Answer Groundedness**: Responses are measured for factual grounding in source documents
- **Hallucination Mitigation**: Evaluation-driven tuning of chunking strategies and retrieval parameters

The evaluation framework is located in the `eval/` directory and uses standard academic datasets to ensure consistent quality improvements.

---

## 🖥️ Running Locally

### Prerequisites
- Python 3.11+
- PostgreSQL (for chat history persistence)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/miteshgupta07/ResearchMate.git
cd ResearchMate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
HF_TOKEN= your_hf_token

# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=researchmate_db
DB_USER=postgres
DB_PASSWORD=your_database_password

BASE_URL="http://localhost:8000"
```

4. **Initialize Database**
```bash
# Initialize the PostgreSQL database and create necessary tables
cd backend
python init_db.py
```

5. **Start the backend API:**
```bash
uvicorn backend.main:app --reload --port 8000
```

6. **Start the frontend (in a separate terminal):**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` with API endpoints at `http://localhost:8000`.

---

## 🧰 Tech Stack

- **Backend Framework**: FastAPI
- **Frontend Framework**: Streamlit
- **Vector Database**: FAISS
- **Relational Database**: PostgreSQL
- **LLM Providers**: Groq, HuggingFace
- **Embedding Models**: HuggingFace Sentence Transformers
- **Orchestration**: LangChain (document loading, text splitting)

---

## 🗂️ Project Structure

```
ResearchMate/
├── backend/           # FastAPI application
│   ├── api/          # API route handlers
│   ├── core/         # Core services (LLM, RAG, chat history)
│   └── schemas/      # Pydantic models
├── frontend/         # Streamlit client utilities
├── eval/             # RAG evaluation framework
├── data/             # Document storage and vector indexes
└── views/            # Streamlit page components
```

---

## Contributing 🤝

Contributions are welcome. Please open an issue for discussion before submitting substantial changes.

---

## License 📝

This project is licensed under the MIT License. See the [LICENSE](https://github.com/miteshgupta07/ResearchMate/blob/main/LICENSE) file for details.

---

## Contact

For inquiries or collaborations, contact [miteshgupta2711@gmail.com](mailto:miteshgupta2711@gmail.com).
