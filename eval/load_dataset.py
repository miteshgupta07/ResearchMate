"""
Dataset Loader for Open RAG Benchmark (arXiv)

This module handles loading and preprocessing of the benchmark dataset:
- Queries with query IDs
- Ground truth relevance judgments (qrels)
- Document corpus with sections
- Reference answers

It also provides chunking functionality matching the production RAG pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# Default dataset path
DEFAULT_DATASET_PATH = Path("pdf/arxiv")

# Chunking parameters - MUST match production RAG pipeline
# See: core/rag_pipeline.py -> RecursiveCharacterTextSplitter settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass
class Query:
    """Represents a single query from the benchmark."""
    query_id: str
    text: str
    query_type: str  # 'abstractive' or 'extractive'
    source: str  # 'text', 'text-image', 'text-table', etc.


@dataclass
class RelevanceJudgment:
    """Ground truth relevance judgment for a query."""
    query_id: str
    doc_id: str
    section_id: int


@dataclass
class Section:
    """A section from a document."""
    section_id: int
    text: str
    tables: Dict[str, str] = field(default_factory=dict)
    images: Dict[str, str] = field(default_factory=dict)


@dataclass
class Document:
    """A document from the corpus."""
    doc_id: str
    title: str
    sections: List[Section]


@dataclass
class Chunk:
    """A text chunk from a document section."""
    chunk_id: str  # Format: {doc_id}_{section_id}_{chunk_index}
    doc_id: str
    section_id: int
    chunk_index: int
    text: str


class OpenRAGBenchmark:
    """
    Loader for the Open RAG Benchmark (arXiv) dataset.
    
    Provides access to:
    - Queries with ground truth
    - Document corpus
    - Reference answers
    - Chunked documents matching production RAG settings
    """
    
    def __init__(self, dataset_path: Optional[Path] = None):
        """
        Initialize the benchmark loader.
        
        Args:
            dataset_path: Path to the dataset directory. 
                          Defaults to 'pdf/arxiv' in project root.
        """
        self.dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        
        # Loaded data
        self._queries: Dict[str, Query] = {}
        self._qrels: Dict[str, RelevanceJudgment] = {}
        self._documents: Dict[str, Document] = {}
        self._answers: Dict[str, str] = {}
        self._chunks: Dict[str, Chunk] = {}
        
        # Mapping from (doc_id, section_id) to list of chunk_ids
        self._section_to_chunks: Dict[Tuple[str, int], List[str]] = {}
        
        self._loaded = False
    
    def load(self) -> "OpenRAGBenchmark":
        """
        Load all dataset components.
        
        Returns:
            Self for method chaining.
        """
        self._load_queries()
        self._load_qrels()
        self._load_answers()
        self._load_corpus()
        self._create_chunks()
        self._loaded = True
        return self
    
    def _load_queries(self) -> None:
        """Load queries from queries.json."""
        queries_path = self.dataset_path / "queries.json"
        
        with open(queries_path, "r", encoding="utf-8") as f:
            raw_queries = json.load(f)
        
        for query_id, data in raw_queries.items():
            self._queries[query_id] = Query(
                query_id=query_id,
                text=data["query"],
                query_type=data.get("type", "unknown"),
                source=data.get("source", "unknown")
            )
    
    def _load_qrels(self) -> None:
        """Load relevance judgments from qrels.json."""
        qrels_path = self.dataset_path / "qrels.json"
        
        with open(qrels_path, "r", encoding="utf-8") as f:
            raw_qrels = json.load(f)
        
        for query_id, data in raw_qrels.items():
            self._qrels[query_id] = RelevanceJudgment(
                query_id=query_id,
                doc_id=data["doc_id"],
                section_id=data["section_id"]
            )
    
    def _load_answers(self) -> None:
        """Load reference answers from answers.json."""
        answers_path = self.dataset_path / "answers.json"
        
        with open(answers_path, "r", encoding="utf-8") as f:
            self._answers = json.load(f)
    
    def _load_corpus(self) -> None:
        """Load all documents from corpus/ directory."""
        corpus_path = self.dataset_path / "corpus"
        
        for doc_file in corpus_path.glob("*.json"):
            doc_id = doc_file.stem  # e.g., "2401.07294v4"
            
            with open(doc_file, "r", encoding="utf-8") as f:
                raw_doc = json.load(f)
            
            sections = []
            for section_data in raw_doc.get("sections", []):
                section = Section(
                    section_id=section_data["section_id"],
                    text=section_data.get("text", ""),
                    tables=section_data.get("tables", {}),
                    images=section_data.get("images", {})
                )
                sections.append(section)
            
            self._documents[doc_id] = Document(
                doc_id=doc_id,
                title=raw_doc.get("title", ""),
                sections=sections
            )
    
    def _create_chunks(self) -> None:
        """
        Create text chunks from document sections.
        
        Uses the same chunking strategy as the production RAG pipeline:
        - Recursive character text splitting
        - chunk_size=500, chunk_overlap=50
        
        Chunk IDs are deterministic: {doc_id}_{section_id}_{chunk_index}
        """
        for doc_id, document in self._documents.items():
            for section in document.sections:
                section_text = section.text
                
                if not section_text.strip():
                    continue
                
                # Split into chunks using same logic as production
                chunks = self._split_text(section_text)
                
                chunk_ids = []
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc_id}_{section.section_id}_{chunk_idx}"
                    
                    self._chunks[chunk_id] = Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section_id=section.section_id,
                        chunk_index=chunk_idx,
                        text=chunk_text
                    )
                    chunk_ids.append(chunk_id)
                
                # Map section to its chunks
                self._section_to_chunks[(doc_id, section.section_id)] = chunk_ids
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting.
        
        Mimics LangChain's RecursiveCharacterTextSplitter behavior
        with chunk_size=500 and chunk_overlap=50.
        
        Args:
            text: The text to split.
        
        Returns:
            List of text chunks.
        """
        if len(text) <= CHUNK_SIZE:
            return [text] if text.strip() else []
        
        # Separators in order of preference (matching LangChain defaults)
        separators = ["\n\n", "\n", " ", ""]
        
        return self._recursive_split(text, separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using hierarchical separators.
        
        Args:
            text: Text to split.
            separators: List of separators to try in order.
        
        Returns:
            List of text chunks.
        """
        if not separators:
            # Base case: just split by character count
            return self._split_by_size(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Empty separator means split by character
            return self._split_by_size(text)
        
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            test_chunk = current_chunk + separator + split if current_chunk else split
            
            if len(test_chunk) <= CHUNK_SIZE:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle overlap
                if CHUNK_OVERLAP > 0 and current_chunk:
                    overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
                    current_chunk = overlap_text + separator + split if len(split) + len(overlap_text) + len(separator) <= CHUNK_SIZE else split
                else:
                    current_chunk = split
                
                # If single split is too large, recursively split it
                if len(current_chunk) > CHUNK_SIZE:
                    sub_chunks = self._recursive_split(current_chunk, remaining_separators)
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
                    else:
                        current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_size(self, text: str) -> List[str]:
        """
        Split text by size when no separator works.
        
        Args:
            text: Text to split.
        
        Returns:
            List of chunks.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - CHUNK_OVERLAP if CHUNK_OVERLAP > 0 else end
        
        return chunks
    
    # =========== Public API ===========
    
    @property
    def queries(self) -> Dict[str, Query]:
        """Get all queries."""
        return self._queries
    
    @property
    def qrels(self) -> Dict[str, RelevanceJudgment]:
        """Get all relevance judgments."""
        return self._qrels
    
    @property
    def documents(self) -> Dict[str, Document]:
        """Get all documents."""
        return self._documents
    
    @property
    def answers(self) -> Dict[str, str]:
        """Get all reference answers."""
        return self._answers
    
    @property
    def chunks(self) -> Dict[str, Chunk]:
        """Get all chunks."""
        return self._chunks
    
    def get_relevant_chunks(self, query_id: str) -> List[str]:
        """
        Get chunk IDs that are relevant to a query.
        
        Based on qrels, returns all chunks from the relevant section.
        
        Args:
            query_id: The query ID.
        
        Returns:
            List of chunk IDs that are relevant.
        """
        if query_id not in self._qrels:
            return []
        
        qrel = self._qrels[query_id]
        key = (qrel.doc_id, qrel.section_id)
        
        return self._section_to_chunks.get(key, [])
    
    def get_valid_query_ids(self) -> List[str]:
        """
        Get query IDs that have valid ground truth data.
        
        A query is valid if:
        1. It has a qrel entry
        2. The referenced document exists in corpus
        3. The referenced section has been chunked
        
        Returns:
            List of valid query IDs.
        """
        valid_ids = []
        
        for query_id in self._queries:
            if query_id not in self._qrels:
                continue
            
            qrel = self._qrels[query_id]
            
            if qrel.doc_id not in self._documents:
                continue
            
            key = (qrel.doc_id, qrel.section_id)
            if key not in self._section_to_chunks:
                continue
            
            if not self._section_to_chunks[key]:
                continue
            
            valid_ids.append(query_id)
        
        return valid_ids
    
    def get_reference_answer(self, query_id: str) -> Optional[str]:
        """
        Get the reference answer for a query.
        
        Args:
            query_id: The query ID.
        
        Returns:
            Reference answer text or None if not available.
        """
        return self._answers.get(query_id)
    
    def get_query_text(self, query_id: str) -> Optional[str]:
        """
        Get the query text for a query ID.
        
        Args:
            query_id: The query ID.
        
        Returns:
            Query text or None if not found.
        """
        query = self._queries.get(query_id)
        return query.text if query else None
    
    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """
        Get the text content of a chunk.
        
        Args:
            chunk_id: The chunk ID.
        
        Returns:
            Chunk text or None if not found.
        """
        chunk = self._chunks.get(chunk_id)
        return chunk.text if chunk else None
    
    def stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics.
        """
        valid_queries = self.get_valid_query_ids()
        
        return {
            "total_queries": len(self._queries),
            "total_qrels": len(self._qrels),
            "total_documents": len(self._documents),
            "total_answers": len(self._answers),
            "total_chunks": len(self._chunks),
            "valid_queries": len(valid_queries),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }


def load_benchmark(dataset_path: Optional[Path] = None) -> OpenRAGBenchmark:
    """
    Convenience function to load the benchmark.
    
    Args:
        dataset_path: Optional custom path to dataset.
    
    Returns:
        Loaded OpenRAGBenchmark instance.
    """
    return OpenRAGBenchmark(dataset_path).load()


if __name__ == "__main__":
    # Quick test
    print("Loading Open RAG Benchmark...")
    benchmark = load_benchmark()
    
    stats = benchmark.stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show sample query
    valid_ids = benchmark.get_valid_query_ids()
    if valid_ids:
        sample_id = valid_ids[0]
        print(f"\nSample Query ({sample_id}):")
        print(f"  Text: {benchmark.get_query_text(sample_id)[:100]}...")
        print(f"  Relevant chunks: {len(benchmark.get_relevant_chunks(sample_id))}")
