"""
RAG Evaluation Runner

Main script for evaluating the RAG system against the Open RAG Benchmark.

This script:
1. Loads the benchmark dataset
2. For each query, retrieves top-K chunks using the existing RAG retriever
3. Computes retrieval metrics (Recall@K, MRR)
4. Generates answers using the existing RAG pipeline
5. Computes groundedness metrics
6. Outputs results to console and JSON file

Usage:
    python -m eval.run_eval [options]
    
Options:
    --max-queries N     Limit evaluation to N queries (default: all)
    --skip-generation   Skip answer generation (retrieval metrics only)
    --output PATH       Output JSON path (default: eval/results.json)
    --verbose          Show per-query progress
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
                                                        
# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.load_dataset import OpenRAGBenchmark, load_benchmark, Chunk
from eval.metrics import (
    MetricsCalculator,
    EvaluationMetrics,
    format_metrics_report,
    metrics_to_dict
)


class RAGEvaluator:
    """
    Evaluator for the RAG system using the Open RAG Benchmark.
    
    This class:
    - Loads the benchmark dataset
    - Builds a FAISS index from chunked documents
    - Retrieves chunks for queries
    - Generates answers using the existing RAG pipeline
    - Computes all evaluation metrics
    """
    
    def __init__(
        self,
        benchmark: OpenRAGBenchmark,
        verbose: bool = False
    ):
        """
        Initialize the evaluator.
        
        Args:
            benchmark: Loaded benchmark dataset.
            verbose: Whether to print progress for each query.
        """
        self.benchmark = benchmark
        self.verbose = verbose
        self.metrics_calc = MetricsCalculator()
        
        # Vector store components (lazy loaded)
        self._embeddings = None
        self._vectorstore = None
        self._retriever = None
        self._llm = None
    
    def _get_embeddings(self):
        """Get or create embeddings model (same as production)."""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return self._embeddings
    
    def build_vectorstore(self) -> None:
        """
        Build FAISS vector store from benchmark chunks.
        
        Uses the same embeddings model as the production RAG pipeline.
        """
        print("Building vector store from benchmark chunks...")
        
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        # Convert chunks to LangChain documents
        documents = []
        for chunk_id, chunk in self.benchmark.chunks.items():
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk_id,
                    "doc_id": chunk.doc_id,
                    "section_id": chunk.section_id,
                    "chunk_index": chunk.chunk_index
                }
            )
            documents.append(doc)
        
        print(f"  Creating embeddings for {len(documents)} chunks...")
        
        embeddings = self._get_embeddings()
        self._vectorstore = FAISS.from_documents(documents, embeddings)
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 10})
        
        print("  Vector store ready.")
    
    def retrieve_chunks(self, query_text: str, k: int = 10) -> List[Tuple[str, str]]:
        """
        Retrieve top-K chunks for a query.
        
        Args:
            query_text: The query text.
            k: Number of chunks to retrieve.
        
        Returns:
            List of (chunk_id, chunk_text) tuples in retrieval order.
        """
        if self._retriever is None:
            raise RuntimeError("Vector store not built. Call build_vectorstore() first.")
        
        # Retrieve documents
        docs = self._retriever.invoke(query_text)[:k]
        
        # Extract chunk IDs and text
        results = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id", "")
            results.append((chunk_id, doc.page_content))
        
        return results
    
    def generate_answer(
        self,
        query_text: str,
        context_chunks: List[str],
        language: str = "English"
    ) -> str:
        """
        Generate an answer using the RAG pipeline.
        
        Uses the production LLM and prompt configuration.
        
        Args:
            query_text: The query text.
            context_chunks: List of context chunk texts.
            language: Response language.
        
        Returns:
            Generated answer text.
        """
        if self._llm is None:
            # Import production LLM configuration
            from services.llm import create_llm
            self._llm = create_llm(
                model_name="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=512
            )
        
        # Combine context
        context_text = "\n\n".join(context_chunks)
        
        # Use production RAG prompt
        from services.llm import create_rag_chat_prompt
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.documents import Document
        
        # Create document objects for the chain
        context_docs = [Document(page_content=context_text)]
        
        # Create chain
        rag_prompt = create_rag_chat_prompt()
        chain = create_stuff_documents_chain(llm=self._llm, prompt=rag_prompt)
        
        # Generate response
        response = chain.invoke({
            "context": context_docs,
            "language": language,
            "rag_messages": [("user", query_text)]
        })
        
        return response
    
    def evaluate_query(
        self,
        query_id: str,
        skip_generation: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single query.
        
        Args:
            query_id: The query ID to evaluate.
            skip_generation: Skip answer generation.
        
        Returns:
            Dictionary with query results.
        """
        query_text = self.benchmark.get_query_text(query_id)
        relevant_chunk_ids = self.benchmark.get_relevant_chunks(query_id)
        
        if not query_text:
            return {"error": "Query not found"}
        
        # Retrieve chunks
        retrieved = self.retrieve_chunks(query_text, k=10)
        retrieved_chunk_ids = [chunk_id for chunk_id, _ in retrieved]
        retrieved_texts = [text for _, text in retrieved]
        
        # Add retrieval result
        self.metrics_calc.add_retrieval_result(
            query_id=query_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=relevant_chunk_ids
        )
        
        result = {
            "query_id": query_id,
            "query_text": query_text,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "relevant_chunk_ids": relevant_chunk_ids,
        }
        
        # Optional: Generate answer and evaluate groundedness
        if not skip_generation and retrieved_texts:
            try:
                answer = self.generate_answer(query_text, retrieved_texts)
                context_text = " ".join(retrieved_texts)
                
                self.metrics_calc.add_groundedness_result(
                    query_id=query_id,
                    answer_text=answer,
                    context_text=context_text
                )
                
                result["generated_answer"] = answer
                result["reference_answer"] = self.benchmark.get_reference_answer(query_id)
                
            except Exception as e:
                result["generation_error"] = str(e)
        
        return result
    
    def run_evaluation(
        self,
        max_queries: Optional[int] = None,
        skip_generation: bool = False
    ) -> Tuple[EvaluationMetrics, List[Dict[str, Any]]]:
        """
        Run full evaluation on the benchmark.
        
        Args:
            max_queries: Maximum number of queries to evaluate.
            skip_generation: Skip answer generation.
        
        Returns:
            Tuple of (EvaluationMetrics, per-query results list).
        """
        # Reset metrics
        self.metrics_calc.reset()
        
        # Get valid queries
        valid_query_ids = self.benchmark.get_valid_query_ids()
        
        if max_queries:
            valid_query_ids = valid_query_ids[:max_queries]
        
        print(f"\nEvaluating {len(valid_query_ids)} queries...")
        if skip_generation:
            print("  (Answer generation skipped)")
        
        results = []
        start_time = time.time()
        
        for i, query_id in enumerate(valid_query_ids, 1):
            if self.verbose:
                print(f"  [{i}/{len(valid_query_ids)}] Query: {query_id[:20]}...")
            
            try:
                result = self.evaluate_query(query_id, skip_generation)
                results.append(result)
            except Exception as e:
                results.append({
                    "query_id": query_id,
                    "error": str(e)
                })
            
            # Progress indicator
            if not self.verbose and i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(valid_query_ids) - i) / rate
                print(f"  Progress: {i}/{len(valid_query_ids)} ({rate:.1f} q/s, ETA: {eta:.0f}s)")
        
        elapsed = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed:.1f}s")
        
        # Compute metrics
        metrics = self.metrics_calc.compute_all_metrics()
        
        return metrics, results


def save_results(
    metrics: EvaluationMetrics,
    results: List[Dict[str, Any]],
    output_path: Path,
    include_details: bool = False
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        metrics: Computed metrics.
        results: Per-query results.
        output_path: Path to save JSON file.
        include_details: Include per-query details in output.
    """
    output = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics_to_dict(metrics),
    }
    
    if include_details:
        output["per_query_results"] = results
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system against Open RAG Benchmark"
    )
    parser.add_argument(
        "--max-queries", "-n",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate (default: all)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip answer generation (retrieval metrics only)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="eval/results.json",
        help="Output JSON path (default: eval/results.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-query progress"
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Include per-query details in JSON output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RAG Evaluation - Open RAG Benchmark (arXiv)")
    print("=" * 60)
    
    # Load benchmark
    print("\nLoading benchmark dataset...")
    benchmark = load_benchmark()
    stats = benchmark.stats()
    
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Valid queries: {stats['valid_queries']}")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Chunk size: {stats['chunk_size']}, overlap: {stats['chunk_overlap']}")
    
    # Create evaluator
    evaluator = RAGEvaluator(benchmark, verbose=args.verbose)
    
    # Build vector store
    evaluator.build_vectorstore()
    
    # Run evaluation
    metrics, results = evaluator.run_evaluation(
        max_queries=args.max_queries,
        skip_generation=args.skip_generation
    )
    
    # Print report
    print("\n" + format_metrics_report(metrics))
    
    # Save results
    output_path = Path(args.output)
    save_results(
        metrics,
        results,
        output_path,
        include_details=args.include_details
    )
    
    # Print JSON summary
    print("\nJSON Summary:")
    print(json.dumps(metrics_to_dict(metrics), indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
