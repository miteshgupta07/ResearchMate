# ResearchMate: Technical Deep Dive

## Project Overview

**ResearchMate** is a production-ready AI research assistant designed to streamline academic and research work through document-grounded question answering. It combines retrieval-augmented generation (RAG), multi-LLM support, persistent chat history, and intelligent agent routing to provide researchers with accurate, context-aware responses.

**Key Innovation**: A deterministic single-pass agent that intelligently routes user queries to specialized capabilities (normal chat, document RAG, arXiv search) with no loops or retries, ensuring predictable performance and cost efficiency.

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                           │
│  Streamlit UI (Chat Interface, Document Upload, Configuration)   │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP (REST)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Router Layer (API Endpoints)                            │   │
│  │  /chat, /rag/query, /agent/route, /documents/upload      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Core Service Layer                                      │   │
│  │  RAG Pipeline | LLM Service | Agent Router | Chat History│   │
│  └──────────────────────────────────────────────────────────┘   │
└────────┬──────────────────────────────────────────────────────┬─┘
         │                                                      │
         ▼                                                      ▼
    ┌──────────────┐                                    ┌──────────────┐
    │  LLM Providers                                   │  Persistence │
    │  (Groq API)                                      │  Layer       │
    │  • LLaMA 3.1-8B                                 │              │
    │  • Gemma2 9B                                    │ PostgreSQL   │
    │  • Mixtral 8x7B                                 │ (Chat Hist.) │
    └──────────────┘                                  │              │
                                                      └──────────────┘
         │                                                      │
         ▼                                                      ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                  DATA & STORAGE LAYER                         │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
    │  │ FAISS Vector │  │ PDF Documents│  │  Metadata    │        │
    │  │ Store Index  │  │ (Persistent) │  │  (JSON)      │        │
    │  │ (Embeddings) │  │              │  │              │        │
    │  └──────────────┘  └──────────────┘  └──────────────┘        │
    │  Location: data/rag/ (indexes, documents, metadata)          │
    └──────────────────────────────────────────────────────────────┘
```

### Architectural Principles

1. **Separation of Concerns**: Frontend (Streamlit) communicates only through FastAPI backend
2. **Transport Layer Pattern**: Backend is a thin orchestration layer that exposes core logic via HTTP
3. **Dependency Injection**: FastAPI's dependency system for injecting singletons (LLM registry, chat history store)
4. **Single Source of Truth**: Pre-initialized LLM registry at startup prevents cold-start delays
5. **Session Isolation**: PostgreSQL-backed chat history ensures isolated conversations per user session

---

## Data Pipeline & Flow

### Query Processing Flow

```
User Input
    │
    ├─ [Agent Mode ON]
    │    │
    │    ▼
    │  PreChecks (Rule-based)
    │    ├─ Greeting detection → CHAT
    │    ├─ Very short message → CHAT
    │    └─ Generic single word → CHAT
    │    │
    │    ├─ [No match]
    │    │    │
    │    │    ▼
    │    │  IntentClassifier (Single LLM call)
    │    │    ├─ INTENT_CHAT
    │    │    ├─ INTENT_RAG_QA (if document_id available)
    │    │    ├─ INTENT_RAG_SUMMARY (if document_id available)
    │    │    ├─ INTENT_ARXIV_SEARCH
    │    │    ├─ INTENT_ARXIV_RECOMMEND
    │    │    └─ INTENT_FALLBACK
    │    │
    │    ▼
    │  CapabilityRouter (Direct dispatch)
    │    ├─ route_to_chat()
    │    ├─ route_to_rag_qa()
    │    ├─ route_to_rag_summary()
    │    ├─ route_to_arxiv_search()
    │    └─ route_to_arxiv_recommend()
    │    │
    │    ▼
    │  ResponseAssembler
    │    └─ AgentResponse {content, intent, confidence, metadata}
    │
    ├─ [Agent Mode OFF, Document Uploaded]
    │    │
    │    ▼
    │  RAG Query Handler
    │    ├─ Load retriever from FAISS index
    │    ├─ Retrieve top-K chunks from document
    │    ├─ Augment prompt with context
    │    ├─ Generate response with LLM
    │    └─ RAGQueryResponse {content, sources}
    │
    └─ [Agent Mode OFF, No Document]
         │
         ▼
      Chat Handler
         ├─ Get chat history
         ├─ Generate response without RAG
         └─ ChatResponse {content}
```

### Document Ingestion Pipeline

```
PDF Upload
    │
    ▼
ingest_document(uploaded_file)
    │
    ├─ Generate unique document_id (UUID)
    │
    ├─ Save PDF to persistent storage
    │    └─ data/rag/documents/{document_id}.pdf
    │
    ├─ Load PDF with PyMuPDFLoader
    │    └─ Extract text + metadata (pages, titles)
    │
    ├─ RecursiveCharacterTextSplitter
    │    ├─ chunk_size=500 tokens
    │    ├─ chunk_overlap=50 tokens
    │    └─ Output: list of Document objects with page numbers
    │
    ├─ Generate embeddings
    │    ├─ Model: all-MiniLM-L6-v2 (384-dim)
    │    ├─ Provider: HuggingFace Sentence Transformers
    │    └─ Embedding type: Dense vectors
    │
    ├─ Build FAISS vector store
    │    ├─ Index type: IVF (flat initially, can be scaled)
    │    ├─ Similarity metric: L2 distance
    │    └─ Output: FAISS index with 512-dim vectors
    │
    ├─ Persist FAISS index to disk
    │    └─ data/rag/indexes/{document_id}/
    │        ├─ faiss_index
    │        └─ index.pkl
    │
    ├─ Save metadata (JSON)
    │    └─ data/rag/metadata/{document_id}.json
    │       {
    │         "document_id": "uuid",
    │         "filename": "original_name.pdf",
    │         "chunk_count": 42,
    │         "timestamp": "2024-01-15T10:30:00"
    │       }
    │
    └─ Cache retriever in session
         └─ _retriever_cache[cache_key] = retriever
```

### RAG Query Answering Pipeline

```
User Query + Document
    │
    ▼
answer_query(llm, retriever, chat_history, user_input, language)
    │
    ├─ Retrieve: retriever.invoke(user_input)
    │    ├─ FAISS search: query embedding vs. chunk embeddings
    │    ├─ Top-K selection (default: k=4)
    │    ├─ Reranking: None (direct FAISS ranking)
    │    └─ Output: list of Document chunks with metadata
    │
    ├─ Augment: Construct RAG prompt
    │    ├─ System role: Research assistant instructions
    │    ├─ Context injection: Retrieved chunks
    │    ├─ Chat history: Previous messages (HumanMessage/AIMessage)
    │    └─ User query: Latest question
    │
    ├─ Generate: LLM inference
    │    ├─ Model: LLaMA 3.1-8B (or user-selected)
    │    ├─ Temperature: 0.7 (controllable)
    │    ├─ Max tokens: 512 (controllable, auto-continuation if truncated)
    │    └─ Output: Response text grounded in retrieved context
    │
    ├─ Auto-continuation (if needed)
    │    ├─ Detect truncation: finish_reason == "length" OR not ends with .?!
    │    ├─ Continue: Recursively request continuation (max 2 attempts)
    │    └─ Concatenate: Full response = initial + continuations
    │
    ├─ Persist: Add to chat history
    │    ├─ Save user message → PostgreSQL
    │    └─ Save assistant response → PostgreSQL
    │
    └─ Return: RAGQueryResponse
         ├─ role: "assistant"
         ├─ content: Response text
         └─ sources: [] (reserved for future enhancement)
```

---

## Major Modules & Components

### 1. Backend Core Modules

#### A. `backend/core/llm/service.py` - LLM Orchestration

**Responsibilities:**
- Multi-model support (LLaMA, Gemma2, Mixtral)
- Pre-initialization at FastAPI startup
- Per-request parameter binding (temperature, max_tokens)
- Auto-continuation handling for truncated responses

**Key Classes:**

```python
class LLMRegistry:
    """Pre-initialized model cache"""
    - initialize_all_models()          # Called at startup
    - get_model(model_identifier)      # Get by backend ID
    - get_llm_with_params()            # Get with request params
    - available_models                 # List of initialized models

class AutoContinueLLM(Runnable):
    """Wrapper for auto-continuation"""
    - invoke(input, config)            # Main entry point
    - _is_truncated(response)          # Detect truncation
    - _request_continuation(partial)   # Request more tokens
    - _handle_continuation(response)   # Orchestrate continuation
    - bind(**kwargs)                   # Update parameters
```

**Frontend → Backend Model Mapping:**
| Frontend | Backend |
|----------|---------|
| LLaMA 3.1-8B | llama-3.1-8b-instant |
| Gemma2 9B | gemma2-9b-it |
| Mixtral | mixtral-8x7b-32768 |

**Configuration Defaults:**
```python
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
MAX_CONTINUATION_ATTEMPTS = 2
CONTINUATION_MAX_TOKENS = 512
```

#### B. `backend/core/rag/pipeline.py` - Document Processing

**Responsibilities:**
- PDF ingestion and chunking
- Embedding generation and vectorization
- FAISS index creation and persistence
- Retrieval and answer generation

**Functions:**

```python
ingest_document(uploaded_file, original_filename)
    └─ Returns: (document_id, retriever)

load_retriever(document_id)
    └─ Returns: FAISS retriever (or None)

answer_query(llm, retriever, chat_history, user_input, language)
    └─ Returns: Response text

process_uploaded_pdf(uploaded_file, temp_file_path)
    └─ Legacy wrapper for ingest_document()

answer_with_rag(llm, retriever, user_input, chat_history, language)
    └─ Legacy wrapper for answer_query()

answer_without_rag(llm, chat_history, language)
    └─ Normal chat without document context
```

**Storage Structure:**
```
data/rag/
├── documents/           # PDF files
│   └── {uuid}.pdf
├── indexes/             # FAISS vector indexes
│   └── {uuid}/
│       ├── faiss_index
│       └── index.pkl
└── metadata/            # Document metadata
    └── {uuid}.json
```

#### C. `backend/core/chat_history/postgres.py` - Chat Persistence

**Database Schema:**
```sql
CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(128) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**PostgresChatHistoryStore Class:**
```python
set_session(session_id)           # Set context for ops
add_message(role, content)        # Insert message
get_messages()                    # Fetch all for session
get_langchain_messages()          # Convert to LangChain format
get_session_messages(session_id)  # Fetch specific session
clear_session(session_id)         # Delete all messages
```

**Message Format Conversion:**
```
Python Role → LangChain Role
"user"      → ("human", content)
"assistant" → ("ai", content)
"system"    → ("system", content)
```

### 2. Backend API Routers

#### A. `/chat` - Normal Chat

**Endpoint:** `POST /chat`

**Request Schema:**
```python
class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: str = "English"
    model_type: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
```

**Flow:**
1. Set session context
2. Add user message to history
3. Get chat history in LangChain format
4. Call `answer_without_rag()`
5. Add assistant response to history
6. Return ChatResponse

#### B. `/rag/query` - Document Q&A

**Endpoint:** `POST /rag/query`

**Request Schema:**
```python
class RAGQueryRequest(BaseModel):
    session_id: str
    document_id: str
    message: str
    language: str = "English"
    model_type: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
```

**Flow:**
1. Load retriever from FAISS index
2. Set session context
3. Add user message to history
4. Call `answer_query()` with retriever
5. Add assistant response to history
6. Return RAGQueryResponse

#### C. `/documents/upload` - Document Ingestion

**Endpoint:** `POST /documents/upload`

**Request:** File upload (multipart/form-data)

**Response:**
```python
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    timestamp: str
```

**Flow:**
1. Call `ingest_document()` on uploaded file
2. Return document_id for future queries

#### D. `/history` - Chat History Management

**Endpoints:**
- `GET /history?session_id={id}` - Fetch chat history
- `DELETE /history?session_id={id}` - Clear chat history

**Response:**
```python
class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]
```

#### E. `/agent/route` - Deterministic Agent

**Endpoint:** `POST /agent/route`

**Request Schema:**
```python
class AgentRequest(BaseModel):
    session_id: str
    message: str
    document_id: Optional[str] = None
    language: str = "English"
    model_type: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
```

**Response Schema:**
```python
class AgentResponse(BaseModel):
    role: str = "assistant"
    content: str
    intent: str  # One of INTENT_*
    confidence: float
    metadata: Dict[str, Any]
```

### 3. Agent Router - Single-Pass Deterministic Design

**Purpose:** Intelligently route queries to specialized capabilities without loops

**Pipeline Stages:**

**Stage 1: PreChecks (Rule-based, No LLM)**
```
Greeting Detection:
  ├─ Patterns: "hi", "hello", "hey", "hola", "bonjour", etc.
  └─ Action: → INTENT_CHAT (confidence: 1.0)

Very Short Messages:
  ├─ Length < 4 chars
  └─ Action: → INTENT_CHAT (confidence: 1.0)

Generic Single Words:
  ├─ Words: "yes", "no", "ok", "thanks", "bye", etc.
  └─ Action: → INTENT_CHAT (confidence: 1.0)
```

**Stage 2: Intent Classification (Single LLM Call)**
```
Classifier Prompt Template:
  - Input: message, document_available
  - Output: Exactly ONE intent label
  - No retries, no tool calls
  - LLM response parsed for intent keywords

Allowed Intents:
  ├─ INTENT_CHAT              (general conversation)
  ├─ INTENT_RAG_QA            (doc question)
  ├─ INTENT_RAG_SUMMARY       (doc summary)
  ├─ INTENT_ARXIV_SEARCH      (paper search)
  ├─ INTENT_ARXIV_RECOMMEND   (paper recommendation)
  └─ INTENT_FALLBACK          (error recovery)

Classification Rules:
  1. If no document → block RAG intents
  2. "find papers about X" → ARXIV_SEARCH
  3. "recommend papers on X" → ARXIV_RECOMMEND
  4. Question about doc → RAG_QA
  5. "summarize" about doc → RAG_SUMMARY
  6. General talk → CHAT
```

**Stage 3: Capability Routing (Direct Dispatch)**

Each intent maps to exactly one route function:
```
route_to_chat()          → call answer_without_rag()
route_to_rag_qa()        → call answer_query() with retriever
route_to_rag_summary()   → call answer_query() with summary prompt
route_to_arxiv_search()  → call _search_arxiv() + format results
route_to_arxiv_recommend() → call _search_arxiv() + format recommendations
```

**Stage 4: Response Assembly**
```
AgentResponse:
  ├─ content: Generated response text
  ├─ intent: Classified intent
  ├─ confidence: 0.5 - 1.0 confidence score
  └─ metadata: Dict with route-specific data
     ├─ {document_id} for RAG routes
     ├─ {query, papers} for arXiv routes
     └─ {} for chat route
```

**arXiv Integration:**
```
_search_arxiv(query, max_results=5, sort_by)
  ├─ Search terms: keywords in title/abstract
  ├─ Sort options:
  │   ├─ SubmittedDate (default, most recent)
  │   └─ Relevance (ranking by LLM)
  └─ Returns: List of paper dicts with:
     ├─ title, authors, summary
     ├─ published date
     ├─ url (arXiv page)
     └─ pdf_url (direct PDF link)

Result Formatting:
  ├─ Search results: "**Found N papers for 'X':**"
  │   └─ Each with: title, authors, date, link, snippet
  └─ Recommendations: "**Recommended papers on 'X':**"
      └─ Same format, ranked by relevance
```

---

## Database Schema

### PostgreSQL Tables

**Table: `chat_messages`**

Purpose: Persist user-assistant conversations per session

```sql
CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(128) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Indexes (for performance):**
```sql
CREATE INDEX idx_session_id ON chat_messages(session_id);
CREATE INDEX idx_created_at ON chat_messages(created_at);
```

**Queries:**
```sql
-- Fetch all messages for a session, ordered by creation
SELECT role, content FROM chat_messages 
WHERE session_id = %s 
ORDER BY created_at ASC;

-- Insert a message
INSERT INTO chat_messages (session_id, role, content) 
VALUES (%s, %s, %s);

-- Clear a session
DELETE FROM chat_messages WHERE session_id = %s;
```

---

## LLM Service Architecture

### Model Configuration

**Pre-Initialization Pattern (at FastAPI startup):**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_llm_registry()  # Create all models once
    registry = get_llm_registry()
    print(f"Models: {registry.available_models}")
    
    yield
    
    # Shutdown
    print("[Shutdown] Application shutting down")
```

**Per-Request Parameter Binding:**
```python
llm = get_llm(
    model_type="LLaMA 3.1-8B",  # Frontend name (resolved to backend ID)
    temperature=0.7,             # Per-request override
    max_tokens=512               # Per-request override
)
# Returns: AutoContinueLLM wrapper with params applied
```

### Auto-Continuation Mechanism

**Problem:** Token limits can truncate responses

**Solution:** Automatic continuation requests

**Detection (two strategies):**
1. **Provider Finish Reason:** `finish_reason == "length"`
2. **Heuristic:** Response doesn't end with `.?!`

**Continuation Algorithm:**
```
Initial Response: "The model learns through backpropagation..."
                 (truncated at token limit)

Request 1: "Continue from where you left off..."
Response 1: "...gradient descent updates. Loss functions"
                 (still incomplete)

Request 2: "Continue from where you left off..."
Response 2: "...can be customized for specific objectives."
                 (ends with period)

Final Output: Concatenation of all parts
```

**Limits:**
- Max continuation attempts: 2
- Continuation token limit: 512
- Total max response: ~1500 tokens

---

## Frontend Implementation

### Streamlit Architecture

**File Structure:**
```
app.py                          # Entry point, page navigation
views/
├── chat.py                    # Main chat interface
├── about_chatbot.py           # Information page
└── about_developer.py         # Developer info

frontend/
├── api_client.py              # HTTP client for backend
└── css.py                     # Styling and rendering
```

### Chat View (`views/chat.py`)

**Session State Variables:**
```python
st.session_state.session_id     # UUID for user session
st.session_state.document_id    # UUID for uploaded document
st.session_state.agent_enabled  # Boolean for agent mode
st.session_state.language       # Selected language
st.session_state.model          # Selected model
```

**UI Components:**
```
Sidebar:
├── Agent Mode Toggle
│   ├─ Enable/disable agent routing
│   └─ Status indicator
├── Document Uploader
│   ├─ PDF file selection
│   └─ Processing status
├── Language Selector
│   └─ Options: English, Hindi, Spanish, French, German
└── Model Customization
    ├─ Model selector (3 options)
    ├─ Temperature slider (0.0-1.0)
    └─ Max tokens slider (1-2048)

Main Chat Area:
├─ Title: "Research Mate 🤖"
├─ Subtitle with developer credit
├─ Mode indicator banner
├─ Chat history (rendered messages)
└─ Chat input box
    └─ User enters questions
```

**Message Routing Logic:**
```python
if agent_enabled:
    result = send_agent_message()  # Agent endpoint
elif document_id:
    result = send_rag_query()      # RAG endpoint
else:
    result = send_chat_message()   # Chat endpoint
```

### API Client (`frontend/api_client.py`)

**Helper Functions:**
```python
send_chat_message(session_id, message, language, model_type, temperature, max_tokens)
    └─ POST /chat

send_rag_query(session_id, document_id, message, language, model_type, temperature, max_tokens)
    └─ POST /rag/query

send_agent_message(session_id, message, document_id, language, model_type, temperature, max_tokens)
    └─ POST /agent/route

upload_document(file)
    └─ POST /documents/upload (multipart)

get_chat_history(session_id)
    └─ GET /history?session_id={id}

clear_chat_history(session_id)
    └─ DELETE /history?session_id={id}

check_backend_health()
    └─ GET / (health check)
```

**Error Handling:**
```python
class APIError(Exception):
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
```

---

## Document Retrieval & Search

### FAISS Vector Store

**Vector Store Type:** Flat FAISS index (scalable to IVF for large datasets)

**Embedding Model:**
```
Model: all-MiniLM-L6-v2
Provider: HuggingFace Sentence Transformers
Dimension: 384 (MiniLM default)
Similarity Metric: L2 distance (Euclidean)
Max sequence length: 512 tokens
```

**Retrieval Process:**
```
User Query
    │
    ├─ Convert to embedding (384-dim)
    ├─ Search FAISS index (L2 distance)
    ├─ Retrieve top-K chunks (default K=4)
    │  └─ Each chunk: up to 500 tokens
    │  └─ Overlap: 50 tokens between consecutive chunks
    └─ Return: List of Document objects with metadata
       ├─ page_content (chunk text)
       └─ metadata (page number, doc filename)
```

### Chunking Strategy

**Text Splitter:** `RecursiveCharacterTextSplitter`

**Configuration:**
```python
chunk_size = 500        # Tokens per chunk
chunk_overlap = 50      # Overlap between chunks (10%)
separators = [          # Split priority
    "\n\n",            # Paragraph breaks first
    "\n",              # Line breaks second
    " ",               # Spaces third
    ""                 # Characters last
]
```

**Why This Strategy?**
- 500 tokens ≈ 1-2 paragraphs (readable context)
- 10% overlap prevents information loss at boundaries
- Recursive splitting preserves semantic structure

---

## Evaluation Framework

### Metrics (`eval/metrics.py`)

**Purpose:** Quantify RAG quality without external dependencies

**Metrics Implemented:**

#### 1. Retrieval Metrics

**Recall@K (K = 3, 5, 10)**
```
Formula: (# queries with ≥1 relevant chunk in top-K) / (# total queries)

Interpretation:
  Recall@3 = 0.80  →  80% of queries have relevant doc in top-3
  Ideal: > 0.70    →  70% hit rate is generally acceptable
```

**Mean Reciprocal Rank (MRR)**
```
Formula: (1/N) * Σ(1/rank_of_first_relevant)

Example:
  Query 1: First relevant at rank 2 → 1/2 = 0.5
  Query 2: First relevant at rank 1 → 1/1 = 1.0
  Query 3: No relevant chunk   → 1/∞ = 0.0
  MRR = (0.5 + 1.0 + 0.0) / 3 = 0.5

Interpretation:
  MRR = 0.67  →  On average, first relevant chunk found at rank 1.5
  Ideal: > 0.60  →  First relevant within top-2 on average
```

#### 2. Groundedness Metrics

**Overlap Ratio (Token-level)**
```
Formula: (# answer tokens in context) / (# total answer tokens)

Algorithm:
  1. Tokenize answer (remove stopwords)
  2. Tokenize context (remove stopwords)
  3. Count overlapping tokens
  4. Ratio = overlaps / answer_token_count

Example:
  Answer: "The model uses attention mechanisms"
  Context: "Attention mechanisms are key to the architecture"
  
  Answer tokens: [model, uses, attention, mechanisms]
  Context tokens: [attention, mechanisms, key, architecture]
  Overlaps: [attention, mechanisms]
  Ratio: 2/4 = 0.50 (50% grounded)
```

**Groundedness Threshold:** 0.20 (20% minimum overlap)
- Answers below threshold are flagged as potentially hallucinated
- Stopword filtering prevents false positives on common words

**Grounded Ratio**
```
Formula: (# grounded answers) / (# total answers)

Interpretation:
  Grounded Ratio = 0.85  →  85% of answers meet groundedness threshold
  Ideal: > 0.80          →  Most answers grounded in retrieved context
```

### MetricsCalculator Usage

```python
from eval.metrics import MetricsCalculator

calc = MetricsCalculator()

# Add retrieval results
calc.add_retrieval_result(
    query_id="q1",
    retrieved_chunk_ids=["c1", "c2", "c3", "c4", "c5"],
    relevant_chunk_ids=["c2", "c6"]  # c2 is at rank 2
)

# Add groundedness results
calc.add_groundedness_result(
    query_id="q1",
    answer_text="The model uses attention",
    context_text="Attention is the key mechanism"
)

# Compute metrics
metrics = calc.compute_all_metrics()
# Returns: EvaluationMetrics with all scores

# Get report
report = format_metrics_report(metrics)
print(report)
```

---

## Setup & Deployment

### Local Development Setup

#### Prerequisites
- Python 3.11+
- PostgreSQL 12+
- pip or conda

#### Step 1: Clone Repository
```bash
git clone https://github.com/miteshgupta07/ResearchMate.git
cd ResearchMate
```

#### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Configure Environment Variables

Create `.env` file in project root:
```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key
HF_TOKEN=your_huggingface_token

# PostgreSQL Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=researchmate_db
DB_USER=postgres
DB_PASSWORD=your_password_here

# Backend URL
BASE_URL=http://localhost:8000
```

**Getting API Keys:**
- **Groq API Key**: https://console.groq.com/keys
- **HuggingFace Token**: https://huggingface.co/settings/tokens
- **LangChain API Key**: https://smith.langchain.com/ (optional, for tracing)

#### Step 5: Initialize Database
```bash
cd backend
python init_db.py
cd ..
```

**What this does:**
- Creates PostgreSQL database if not exists
- Creates `chat_messages` table
- Establishes connection test

#### Step 6: Start Backend API
```bash
uvicorn backend.main:app --reload --port 8000
```

**Expected Output:**
```
[Startup] LLM Registry initialized with models: ['llama-3.1-8b-instant', 'gemma2-9b-it', 'mixtral-8x7b-32768']
[Startup] Portfolio retriever initialized successfully
Uvicorn running on http://127.0.0.1:8000
```

#### Step 7: Start Frontend (New Terminal)
```bash
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.
  URL: http://localhost:8501
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Backend
EXPOSE 8000

# Frontend
EXPOSE 8501

CMD ["bash", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501"]
```

**Build & Run:**
```bash
docker build -t researchmate .
docker run -p 8000:8000 -p 8501:8501 \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -e DB_HOST=postgres \
  researchmate
```

### Production Deployment

**Recommended Stack:**
```
Reverse Proxy:  Nginx / Traefik
Backend:        Gunicorn + FastAPI
Frontend:       Streamlit + Gunicorn
Database:       PostgreSQL (managed)
Vector Store:   FAISS on persistent volume
File Storage:   S3 / GCS (for PDFs and indexes)
```

**Gunicorn Backend:**
```bash
gunicorn backend.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Nginx Configuration (example):**
```nginx
server {
    listen 80;
    server_name api.researchmate.me;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Key Design Decisions & Trade-offs

### 1. FAISS Over LangChain VectorStore Abstractions
**Why:** 
- Direct control over embedding model and persistence
- Avoid vendor lock-in
- Deterministic retrieval for evaluation

**Trade-off:** 
- Lose automatic batching and distributed retrieval

### 2. Flat FAISS Index (Not Hierarchical)
**Why:** 
- Research papers are typically 10-50 pages → manageable size
- Perfect recall needed (no approximate search trade-offs)

**Trade-off:** 
- Doesn't scale to billions of documents
- Can upgrade to IVF/HNSW as dataset grows

### 3. PostgreSQL Over Document Database
**Why:**
- ACID transactions ensure consistent chat history
- Schema is simple (3 columns)
- Standard tooling for backups/monitoring

**Trade-off:** 
- Overkill for simple key-value storage
- Could use Redis for faster access

### 4. Single-Pass Agent (No Loops)
**Why:**
- Predictable latency (max 2 LLM calls: classify + execute)
- Reduced cost (no retries or backtracking)
- Clear error semantics (fails fast)

**Trade-off:** 
- Cannot refine intent after seeing context
- Cannot explore multiple paths

### 5. Token-Based Groundedness (Not LLM-based)
**Why:**
- Fast evaluation (no LLM calls)
- Reproducible and interpretable
- Doesn't require external models

**Trade-off:** 
- Crude heuristic (not semantic understanding)
- High false positive rate on paraphrases

### 6. 500-Token Chunks Over Sentence-Level
**Why:**
- Sufficient context for Q&A (1-2 paragraphs)
- Fewer chunks = faster retrieval
- Reduces vector store size

**Trade-off:** 
- May lose fine-grained relevance
- Paragraph-level queries might miss details

---

## Performance Characteristics

### Latency Profile (Typical)

| Operation | Time | Bottleneck |
|-----------|------|-----------|
| Chat query (no RAG) | 2-4s | LLM inference |
| RAG query (4 chunks) | 3-6s | LLM inference + retrieval |
| Agent routing | 4-7s | LLM inference (classify) + LLM inference (execute) |
| Document upload (10 pages) | 8-15s | PDF parsing + embedding generation |
| arXiv search (5 results) | 2-3s | Network latency + result formatting |

### Memory Profile

| Component | Memory |
|-----------|--------|
| LLM (LLaMA 3.1-8B) | ~8-16GB (on GPU) |
| FAISS index (100 papers) | ~200MB |
| Embeddings model (MiniLM) | ~50MB |
| PostgreSQL (10K messages) | ~50MB |

### Scalability Limits

**Current Limits:**
- Single-machine deployment
- Up to ~500 documents in FAISS (flat index)
- Chat history: millions of messages (PostgreSQL scales)

**Scaling Path:**
1. FAISS → IVF (10K docs)
2. FAISS → Distributed (Milvus/Weaviate)
3. Backend → Kubernetes
4. PostgreSQL → RDS/Managed DB

---

## Testing Strategy

### Unit Tests (eval/metrics.py)

```python
def test_recall_at_k():
    calc = MetricsCalculator()
    
    # Query 1: Relevant chunk at rank 2 → hit at 3, 5, 10
    calc.add_retrieval_result("q1", ["c1", "c2", "c3"], ["c2"])
    
    # Query 2: No relevant chunk → miss at all K
    calc.add_retrieval_result("q2", ["c4", "c5", "c6"], ["c100"])
    
    assert calc.compute_recall_at_k(3) == 0.5  # 1 hit out of 2
    assert calc.compute_recall_at_k(10) == 0.5

def test_groundedness():
    calc = MetricsCalculator()
    
    answer = "The model uses attention mechanisms"
    context = "Attention mechanisms are key to transformers"
    
    result = calc.add_groundedness_result("q1", answer, context)
    
    # Tokens: [model, uses, attention, mechanisms] (stopwords removed)
    # Overlaps: [attention, mechanisms]
    # Ratio: 2/4 = 0.50
    assert result.overlap_ratio == 0.50
    assert result.is_grounded == True  # 0.50 >= 0.20
```

### Integration Tests

**Test Flow:**
1. Upload test PDF
2. Query document via RAG endpoint
3. Check response contains context-relevant info
4. Verify PostgreSQL stores message
5. Retrieve chat history and validate

### Evaluation Benchmarks

**Standard Benchmark:** TREC-COVID (research papers + queries)

```
Target Metrics:
  Recall@3:  >= 0.65
  Recall@5:  >= 0.75
  MRR:       >= 0.60
  Groundedness: >= 0.80
```

---

## Troubleshooting

### Common Issues

**Issue 1: "GROQ_API_KEY not found"**
```
Solution: 
  1. Create .env file with GROQ_API_KEY=...
  2. Restart FastAPI server
  3. Check: echo $GROQ_API_KEY
```

**Issue 2: PostgreSQL connection refused**
```
Solution:
  1. Verify PostgreSQL is running: psql -U postgres
  2. Check DB_HOST, DB_PORT in .env
  3. Run: python backend/init_db.py (creates DB if missing)
```

**Issue 3: FAISS index not found after upload**
```
Solution:
  1. Check: ls data/rag/indexes/
  2. Verify document_id matches
  3. Check file permissions on data/rag/
```

**Issue 4: LLM response truncated**
```
Solution:
  1. Auto-continuation should handle it automatically
  2. If not working, check: MAX_CONTINUATION_ATTEMPTS in llm/service.py
  3. Increase max_tokens in Streamlit UI
```

### Debug Logging

**Enable verbose logging:**
```python
# backend/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check API health:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/
```

**Verify backend reachability from frontend:**
```python
# frontend/api_client.py
from frontend.api_client import check_backend_health
is_healthy = check_backend_health()
print(f"Backend healthy: {is_healthy}")
```

---

## Code Style & Conventions

### Naming Conventions
- `document_id`: UUID for uploaded documents
- `session_id`: UUID for user sessions
- `retriever`: FAISS retriever object
- `chat_history_langchain`: List of (role, content) tuples for LangChain

### Prompt Templates
```python
# System role always at top
("system", "You are a research assistant...")

# Context injected next
("human", "Context: {context}")

# Chat history in middle
MessagesPlaceholder(variable_name="rag_messages")

# User query last (implicit, added at invoke time)
```

### Error Handling
```python
try:
    # Operation
except HTTPException:
    raise  # Re-raise HTTP errors
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"Operation failed: {str(e)}"
    )
```

---

## Future Enhancements

### Short-term (1-3 months)
1. **Source Tracking**: Return specific chunks used in answers
2. **Multi-document RAG**: Query across multiple documents
3. **Reranking**: Add cross-encoder for better relevance
4. **Streaming**: Server-sent events for real-time responses

### Medium-term (3-6 months)
1. **Hybrid Search**: BM25 + semantic search
2. **Prompt Caching**: Reduce latency on repeated queries
3. **Fine-tuning**: Task-specific model adaptation
4. **Citation Management**: Inline references in responses

### Long-term (6+ months)
1. **Knowledge Graphs**: Explicit entity relationships
2. **Multi-modal RAG**: Images + tables from PDFs
3. **Collaborative Features**: Shared document collections
4. **API Marketplace**: Pre-built workflows for specific research domains

---

## Conclusion

ResearchMate demonstrates a production-grade RAG system with:
- **Robust architecture**: Separated concerns, dependency injection, persistence
- **Intelligent routing**: Single-pass deterministic agent avoiding costly loops
- **Evaluated quality**: Metrics-driven RAG pipeline with groundedness checks
- **Scalable design**: PostgreSQL for history, FAISS for vectors, multi-LLM support
- **Developer-friendly**: Clear module organization, comprehensive error handling

The system prioritizes **reliability and cost-efficiency** over feature complexity, making it suitable for enterprise research assistance applications.

---

**Document Generated:** 2026-05-30  
**Version:** 1.0  
**Project:** ResearchMate AI Research Assistant
