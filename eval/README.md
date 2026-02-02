# RAG Evaluation Module

Standalone evaluation suite for measuring RAG system performance using the **Open RAG Benchmark (arXiv)** dataset.

## 📊 Metrics Explained

### 1. Recall@K (Primary Retrieval Metric)

**What it measures:** The fraction of queries where *at least one* relevant chunk appears in the top-K retrieved chunks.

**Formula:**
```
Recall@K = (# queries with at least 1 relevant chunk in top-K) / (# total queries)
```

**Interpretation:**
- `Recall@3 = 0.60` means 60% of queries have a relevant chunk in top 3 results
- Higher is better (max = 1.0)
- We compute for K ∈ {3, 5, 10}

**Why it matters:** Measures whether the retriever can find *any* relevant context. If recall is low, the LLM has no chance to answer correctly.

---

### 2. Mean Reciprocal Rank (MRR)

**What it measures:** The average inverse rank of the first relevant chunk.

**Formula:**
```
MRR = (1/N) × Σ (1 / rank_of_first_relevant_chunk)
```

**Interpretation:**
- If the first relevant chunk is at rank 1 → RR = 1.0
- If at rank 2 → RR = 0.5
- If at rank 5 → RR = 0.2
- If no relevant chunk found → RR = 0

**Why it matters:** Penalizes retrievers that bury relevant results. Higher MRR means relevant chunks appear earlier.

---

### 3. Answer Groundedness (Heuristic)

**What it measures:** Token-level overlap between the generated answer and retrieved context.

**Formula:**
```
Groundedness = (# answer tokens present in context) / (# total answer tokens)
```

**Processing:**
1. Tokenize answer and context (words only, lowercased)
2. Remove stopwords
3. Count overlapping tokens

**Threshold:** Answers with < 10% overlap are flagged as **ungrounded**.

**Why it matters:** Detects hallucinations. If an answer contains many tokens not in the context, it may be making things up.

**Limitations:**
- Purely lexical (no semantic similarity)
- May undercount paraphrased content
- Not a substitute for human evaluation

---

## 🚀 Quick Start

### Basic Usage

```bash
# Run full evaluation (all queries)
python -m eval.run_eval

# Limit to 50 queries for testing
python -m eval.run_eval --max-queries 50

# Retrieval metrics only (skip LLM generation)
python -m eval.run_eval --skip-generation

# Verbose output
python -m eval.run_eval --verbose
```

### Output

The script outputs:
1. **Console report** with all metrics
2. **JSON file** at `eval/results.json`

Example console output:
```
==================================================
RAG Evaluation Report
==================================================

Retrieval Metrics:
  Recall@3:  0.4523
  Recall@5:  0.5891
  Recall@10: 0.7234
  MRR:       0.3567

Groundedness Metrics:
  Groundedness (avg): 0.4521
  Grounded Ratio:     0.8912

Query Counts:
  Total Queries:          500
  With Retrieval Eval:    500
  With Groundedness Eval: 500
==================================================
```

Example JSON output:
```json
{
  "recall@3": 0.4523,
  "recall@5": 0.5891,
  "recall@10": 0.7234,
  "mrr": 0.3567,
  "groundedness": 0.4521
}
```

---

## 📁 Module Structure

```
eval/
├── __init__.py           # Package marker
├── load_dataset.py       # Dataset loading and chunking
├── metrics.py            # Metric implementations
├── run_eval.py           # Main evaluation runner
├── README.md             # This file
└── results.json          # Output (generated)
```

### load_dataset.py

Handles loading the Open RAG Benchmark:
- `OpenRAGBenchmark` class - Main dataset interface
- `load_benchmark()` - Quick loader function
- Uses **same chunking parameters** as production RAG (500 chars, 50 overlap)

### metrics.py

Implements all evaluation metrics:
- `MetricsCalculator` - Stateful metric computation
- `RetrievalResult` / `GroundednessResult` - Per-query results
- `EvaluationMetrics` - Aggregated metrics dataclass

### run_eval.py

Main evaluation script:
- Builds FAISS index from benchmark chunks
- Retrieves chunks using production retriever
- Optionally generates answers using production LLM
- Computes and saves all metrics

---

## 📈 Interpreting Results

### Good Retrieval Performance
- Recall@10 > 0.80
- Recall@5 > 0.60
- MRR > 0.40

### Poor Retrieval Performance
- Recall@10 < 0.50
- MRR < 0.20

### Diagnosing Issues

| Symptom | Likely Cause | Investigation |
|---------|--------------|---------------|
| Low Recall@K | Poor embeddings or chunking | Check chunk boundaries, try different embedding model |
| High Recall@10, Low MRR | Relevant chunks ranked too low | Tune retriever scoring, consider reranking |
| Low Groundedness | Hallucination or paraphrasing | Check generated answers manually, inspect ungrounded queries |
| High Recall, Low Groundedness | Good retrieval, bad generation | LLM prompt or context formatting issue |

---

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-queries N` | Limit evaluation to N queries | All queries |
| `--skip-generation` | Skip LLM answer generation | False |
| `--output PATH` | Output JSON path | `eval/results.json` |
| `--verbose` | Show per-query progress | False |
| `--include-details` | Include per-query results in JSON | False |

---

## 🔧 Configuration

### Chunking Parameters

Located in `load_dataset.py`:
```python
CHUNK_SIZE = 500      # Characters per chunk
CHUNK_OVERLAP = 50    # Overlap between chunks
```

**Important:** These MUST match production settings in `core/rag_pipeline.py`.

### Groundedness Threshold

Located in `metrics.py`:
```python
GROUNDEDNESS_THRESHOLD = 0.1  # 10% token overlap minimum
```

Adjust based on your tolerance for ungrounded answers.

---

## 📝 Dataset Structure

The Open RAG Benchmark (arXiv) is located at `pdf/arxiv/`:

```
pdf/arxiv/
├── queries.json      # Query text and metadata
├── qrels.json        # Ground truth relevance judgments
├── answers.json      # Reference answers
├── pdf_urls.json     # Source PDF URLs
└── corpus/           # Document sections as JSON
    ├── 2401.01872v2.json
    ├── 2401.02247v4.json
    └── ...
```

### Ground Truth Format

**qrels.json** maps query → relevant document section:
```json
{
  "query_id": {
    "doc_id": "2401.07294v4",
    "section_id": 12
  }
}
```

A query is considered to have a "hit" if any chunk from the relevant section is retrieved.

---

## 🚫 Explicitly NOT Implemented

Per requirements, the following are excluded:
- ❌ BLEU / ROUGE scores
- ❌ Accuracy metrics
- ❌ RAGAS / TruLens frameworks
- ❌ LLM-as-judge evaluation
- ❌ Fine-tuning or model modification
- ❌ Retriever modification

---

## 📊 Baseline Establishment

This evaluation module establishes a **baseline** for the RAG system. Use these metrics to:

1. **Track improvements** - Compare before/after any changes
2. **Identify weaknesses** - Low recall points to retriever issues
3. **Justify modifications** - Metric deltas prove value of changes

---

## 💡 Tips for Analysis

### Finding Failure Cases

```python
from eval.metrics import MetricsCalculator

calc = MetricsCalculator()
# ... run evaluation ...

# Get queries where retrieval failed
for result in calc.retrieval_results:
    if not result.hit_at_k[10]:
        print(f"No hit: {result.query_id}")

# Get ungrounded answers
ungrounded = calc.get_ungrounded_queries()
```

### Debugging a Specific Query

```python
from eval.load_dataset import load_benchmark

benchmark = load_benchmark()
query_id = "your-query-id"

print(f"Query: {benchmark.get_query_text(query_id)}")
print(f"Relevant chunks: {benchmark.get_relevant_chunks(query_id)}")
print(f"Reference answer: {benchmark.get_reference_answer(query_id)}")
```

---

## 📜 License

This evaluation module is part of the ResearchMate project.
