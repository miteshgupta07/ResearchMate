"""
Evaluation Metrics for RAG System

Implements retrieval-focused metrics and answer groundedness:
1. Recall@K (K = 3, 5, 10) - Primary retrieval metric
2. Mean Reciprocal Rank (MRR) - Ranking quality metric
3. Answer Groundedness - Token-level overlap heuristic

All metrics are implemented in pure Python with minimal dependencies.
"""

import re
from typing import List, Set, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class RetrievalResult:
    """Result of retrieval evaluation for a single query."""
    query_id: str
    retrieved_chunk_ids: List[str]  # Ordered by retrieval rank
    relevant_chunk_ids: Set[str]    # Ground truth relevant chunks
    hit_at_k: Dict[int, bool]       # Whether relevant chunk found at each K
    reciprocal_rank: float          # 1/rank of first relevant chunk (0 if none)


@dataclass
class GroundednessResult:
    """Result of groundedness evaluation for a single answer."""
    query_id: str
    answer_text: str
    context_text: str
    overlap_ratio: float       # Token overlap ratio [0, 1]
    is_grounded: bool          # True if overlap >= threshold
    matched_tokens: int        # Number of overlapping tokens
    answer_tokens: int         # Total tokens in answer


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics across all queries."""
    # Retrieval metrics
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    
    # Groundedness metrics
    avg_groundedness: float
    grounded_ratio: float      # Fraction of answers that are grounded
    
    # Counts
    total_queries: int
    queries_with_retrieval: int
    queries_with_groundedness: int


class MetricsCalculator:
    """
    Calculator for RAG evaluation metrics.
    
    Computes:
    - Recall@K: Fraction of queries where at least one relevant chunk
                appears in top-K retrieved chunks
    - MRR: Mean of 1/(rank of first relevant chunk) across queries
    - Groundedness: Token-level overlap between answer and context
    """
    
    # K values for Recall@K
    K_VALUES = [3, 5, 10]
    
    # Groundedness threshold - answers below this are flagged
    GROUNDEDNESS_THRESHOLD = 0.2
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.retrieval_results: List[RetrievalResult] = []
        self.groundedness_results: List[GroundednessResult] = []
    
    def reset(self) -> None:
        """Reset all stored results."""
        self.retrieval_results = []
        self.groundedness_results = []
    
    # =========== Retrieval Metrics ===========
    
    def add_retrieval_result(
        self,
        query_id: str,
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str]
    ) -> RetrievalResult:
        """
        Add a retrieval result for evaluation.
        
        Args:
            query_id: The query identifier.
            retrieved_chunk_ids: List of chunk IDs in retrieval order (rank 1 first).
            relevant_chunk_ids: List of ground truth relevant chunk IDs.
        
        Returns:
            RetrievalResult with computed metrics.
        """
        relevant_set = set(relevant_chunk_ids)
        
        # Compute hit@K for each K value
        hit_at_k = {}
        for k in self.K_VALUES:
            top_k = set(retrieved_chunk_ids[:k])
            hit_at_k[k] = bool(top_k & relevant_set)  # Any overlap = hit
        
        # Compute reciprocal rank
        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
            if chunk_id in relevant_set:
                reciprocal_rank = 1.0 / rank
                break
        
        result = RetrievalResult(
            query_id=query_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=relevant_set,
            hit_at_k=hit_at_k,
            reciprocal_rank=reciprocal_rank
        )
        
        self.retrieval_results.append(result)
        return result
    
    def compute_recall_at_k(self, k: int) -> float:
        """
        Compute Recall@K across all queries.
        
        Recall@K = (# queries with hit in top-K) / (# total queries)
        
        Args:
            k: The cutoff rank.
        
        Returns:
            Recall@K score [0, 1].
        """
        if not self.retrieval_results:
            return 0.0
        
        hits = sum(1 for r in self.retrieval_results if r.hit_at_k.get(k, False))
        return hits / len(self.retrieval_results)
    
    def compute_mrr(self) -> float:
        """
        Compute Mean Reciprocal Rank across all queries.
        
        MRR = (1/N) * sum(1/rank_of_first_relevant)
        
        Returns:
            MRR score [0, 1].
        """
        if not self.retrieval_results:
            return 0.0
        
        total_rr = sum(r.reciprocal_rank for r in self.retrieval_results)
        return total_rr / len(self.retrieval_results)
    
    # =========== Groundedness Metrics ===========
    
    def add_groundedness_result(
        self,
        query_id: str,
        answer_text: str,
        context_text: str
    ) -> GroundednessResult:
        """
        Add a groundedness result for evaluation.
        
        Groundedness is computed as token-level overlap:
        - Tokenize answer and context (words)
        - Count how many answer tokens appear in context
        - Ratio = matched_tokens / total_answer_tokens
        
        Args:
            query_id: The query identifier.
            answer_text: The generated answer.
            context_text: The retrieved context (concatenated chunks).
        
        Returns:
            GroundednessResult with computed metrics.
        """
        # Tokenize both texts
        answer_tokens = self._tokenize(answer_text)
        context_tokens = self._tokenize(context_text)
        
        if not answer_tokens:
            # Empty answer is considered grounded (no claims made)
            result = GroundednessResult(
                query_id=query_id,
                answer_text=answer_text,
                context_text=context_text,
                overlap_ratio=1.0,
                is_grounded=True,
                matched_tokens=0,
                answer_tokens=0
            )
            self.groundedness_results.append(result)
            return result
        
        # Create context token set for efficient lookup
        context_token_set = set(context_tokens)
        
        # Count overlapping tokens
        matched_tokens = sum(1 for t in answer_tokens if t in context_token_set)
        
        # Compute overlap ratio
        overlap_ratio = matched_tokens / len(answer_tokens)
        
        result = GroundednessResult(
            query_id=query_id,
            answer_text=answer_text,
            context_text=context_text,
            overlap_ratio=overlap_ratio,
            is_grounded=overlap_ratio >= self.GROUNDEDNESS_THRESHOLD,
            matched_tokens=matched_tokens,
            answer_tokens=len(answer_tokens)
        )
        
        self.groundedness_results.append(result)
        return result
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        - Lowercases text
        - Extracts alphanumeric words
        - Removes stopwords (minimal set)
        
        Args:
            text: Text to tokenize.
        
        Returns:
            List of tokens.
        """
        if not text:
            return []
        
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        
        # Minimal stopwords to avoid
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
            'so', 'yet', 'both', 'either', 'neither', 'not', 'only', 'own',
            'same', 'than', 'too', 'very', 'just', 'it', 'its', 'this',
            'that', 'these', 'those', 'what', 'which', 'who', 'whom',
            'whose', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'any', 'some', 'no', 'such', 'here', 'there', 'i', 'you', 'he',
            'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
            'your', 'his', 'our', 'their', 'if', 'because', 'while'
        }
        
        # Filter out stopwords and very short tokens
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        
        return tokens
    
    def compute_avg_groundedness(self) -> float:
        """
        Compute average groundedness across all answers.
        
        Returns:
            Average overlap ratio [0, 1].
        """
        if not self.groundedness_results:
            return 0.0
        
        total = sum(r.overlap_ratio for r in self.groundedness_results)
        return total / len(self.groundedness_results)
    
    def compute_grounded_ratio(self) -> float:
        """
        Compute fraction of answers that are grounded.
        
        Returns:
            Ratio of grounded answers [0, 1].
        """
        if not self.groundedness_results:
            return 0.0
        
        grounded = sum(1 for r in self.groundedness_results if r.is_grounded)
        return grounded / len(self.groundedness_results)
    
    def get_ungrounded_queries(self) -> List[str]:
        """
        Get query IDs of ungrounded answers.
        
        Useful for analyzing failure cases.
        
        Returns:
            List of query IDs with ungrounded answers.
        """
        return [r.query_id for r in self.groundedness_results if not r.is_grounded]
    
    # =========== Aggregation ===========
    
    def compute_all_metrics(self) -> EvaluationMetrics:
        """
        Compute all evaluation metrics.
        
        Returns:
            EvaluationMetrics dataclass with all computed metrics.
        """
        return EvaluationMetrics(
            recall_at_3=self.compute_recall_at_k(3),
            recall_at_5=self.compute_recall_at_k(5),
            recall_at_10=self.compute_recall_at_k(10),
            mrr=self.compute_mrr(),
            avg_groundedness=self.compute_avg_groundedness(),
            grounded_ratio=self.compute_grounded_ratio(),
            total_queries=len(set(
                [r.query_id for r in self.retrieval_results] +
                [r.query_id for r in self.groundedness_results]
            )),
            queries_with_retrieval=len(self.retrieval_results),
            queries_with_groundedness=len(self.groundedness_results)
        )
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Get detailed per-query results for analysis.
        
        Returns:
            Dictionary with per-query breakdown.
        """
        retrieval_by_query = {r.query_id: r for r in self.retrieval_results}
        groundedness_by_query = {r.query_id: r for r in self.groundedness_results}
        
        all_query_ids = set(retrieval_by_query.keys()) | set(groundedness_by_query.keys())
        
        details = {}
        for qid in all_query_ids:
            ret = retrieval_by_query.get(qid)
            gnd = groundedness_by_query.get(qid)
            
            details[qid] = {
                "hit_at_3": ret.hit_at_k.get(3, False) if ret else None,
                "hit_at_5": ret.hit_at_k.get(5, False) if ret else None,
                "hit_at_10": ret.hit_at_k.get(10, False) if ret else None,
                "reciprocal_rank": ret.reciprocal_rank if ret else None,
                "groundedness": gnd.overlap_ratio if gnd else None,
                "is_grounded": gnd.is_grounded if gnd else None,
            }
        
        return details


def format_metrics_report(metrics: EvaluationMetrics) -> str:
    """
    Format evaluation metrics as a human-readable report.
    
    Args:
        metrics: Computed evaluation metrics.
    
    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 50,
        "RAG Evaluation Report",
        "=" * 50,
        "",
        "Retrieval Metrics:",
        f"  Recall@3:  {metrics.recall_at_3:.4f}",
        f"  Recall@5:  {metrics.recall_at_5:.4f}",
        f"  Recall@10: {metrics.recall_at_10:.4f}",
        f"  MRR:       {metrics.mrr:.4f}",
        "",
        "Groundedness Metrics:",
        f"  Groundedness (avg): {metrics.avg_groundedness:.4f}",
        f"  Grounded Ratio:     {metrics.grounded_ratio:.4f}",
        "",
        "Query Counts:",
        f"  Total Queries:          {metrics.total_queries}",
        f"  With Retrieval Eval:    {metrics.queries_with_retrieval}",
        f"  With Groundedness Eval: {metrics.queries_with_groundedness}",
        "=" * 50,
    ]
    
    return "\n".join(lines)


def metrics_to_dict(metrics: EvaluationMetrics) -> Dict[str, Any]:
    """
    Convert metrics to dictionary for JSON serialization.
    
    Args:
        metrics: Computed evaluation metrics.
    
    Returns:
        Dictionary representation.
    """
    return {
        "recall@3": round(metrics.recall_at_3, 4),
        "recall@5": round(metrics.recall_at_5, 4),
        "recall@10": round(metrics.recall_at_10, 4),
        "mrr": round(metrics.mrr, 4),
        "groundedness": round(metrics.avg_groundedness, 4),
        "grounded_ratio": round(metrics.grounded_ratio, 4),
        "total_queries": metrics.total_queries,
        "queries_with_retrieval": metrics.queries_with_retrieval,
        "queries_with_groundedness": metrics.queries_with_groundedness,
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    calc = MetricsCalculator()
    
    # Simulate some retrieval results
    calc.add_retrieval_result(
        query_id="q1",
        retrieved_chunk_ids=["c1", "c2", "c3", "c4", "c5"],
        relevant_chunk_ids=["c2", "c6"]  # c2 is at rank 2
    )
    
    calc.add_retrieval_result(
        query_id="q2",
        retrieved_chunk_ids=["c10", "c11", "c12", "c13", "c14"],
        relevant_chunk_ids=["c20"]  # No hit
    )
    
    calc.add_retrieval_result(
        query_id="q3",
        retrieved_chunk_ids=["c30", "c31", "c32"],
        relevant_chunk_ids=["c30"]  # Hit at rank 1
    )
    
    # Simulate groundedness results
    calc.add_groundedness_result(
        query_id="q1",
        answer_text="The model uses attention mechanisms for processing.",
        context_text="Attention mechanisms are key to the model architecture."
    )
    
    calc.add_groundedness_result(
        query_id="q2",
        answer_text="The sky is blue because of Rayleigh scattering.",
        context_text="The document discusses machine learning techniques."
    )
    
    # Compute metrics
    metrics = calc.compute_all_metrics()
    
    print(format_metrics_report(metrics))
    print("\nJSON Output:")
    import json
    print(json.dumps(metrics_to_dict(metrics), indent=2))
