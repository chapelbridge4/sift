"""
Benchmark fixtures for local RAG evaluation.

Contains sample documents, questions, and expected evidence
for measuring retrieval and answer quality.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BenchmarkDocument:
    """A document fixture for benchmarking."""
    content: str
    source_file: str
    file_type: str
    expected_topics: List[str]


@dataclass
class BenchmarkQuery:
    """A query fixture with expected evidence."""
    query: str
    expected_evidence_topics: List[str]
    question_type: str  # "factual", "conceptual", "comparative", "procedural"


BENCHMARK_DOCUMENTS = [
    BenchmarkDocument(
        content="""The hippocampus is a major component of the brain of humans and other vertebrates.
It belongs to the limbic system and plays important roles in the consolidation of information
from short-term memory to long-term memory, and in spatial memory that enables navigation.
The hippocampus is located under the cerebral cortex in the medial temporal lobe.""",
        source_file="neuroscience_101.txt",
        file_type="txt",
        expected_topics=["hippocampus", "memory", "limbic system", "spatial navigation"]
    ),
    BenchmarkDocument(
        content="""Artificial Intelligence (AI) is a branch of computer science that aims to create
intelligent machines that can think and act like humans. Machine learning is a subset of AI
that allows systems to learn from data without being explicitly programmed. Deep learning
uses neural networks with many layers to model complex patterns in data.""",
        source_file="ai_intro.txt",
        file_type="txt",
        expected_topics=["AI", "machine learning", "deep learning", "neural networks"]
    ),
    BenchmarkDocument(
        content="""Python is a high-level programming language known for its readability and
simplicity. It supports multiple programming paradigms including procedural, object-oriented,
and functional programming. Python's extensive standard library and package ecosystem make
it popular for web development, data analysis, machine learning, and automation tasks.""",
        source_file="python_overview.txt",
        file_type="txt",
        expected_topics=["Python", "programming", "data analysis", "web development"]
    ),
    BenchmarkDocument(
        content="""The prefrontal cortex is the front part of the brain's frontal lobe.
It is responsible for executive functions such as decision-making, planning, and problem-solving.
The prefrontal cortex also helps regulate emotions and impulses by interacting with other
brain regions. Damage to this area can result in significant changes to personality and behavior.""",
        source_file="neuroscience_101.txt",
        file_type="txt",
        expected_topics=["prefrontal cortex", "decision making", "executive function", "brain"]
    ),
    BenchmarkDocument(
        content="""Retrieval-Augmented Generation (RAG) combines the power of large language models
with information retrieval systems. In a RAG pipeline, relevant documents are first retrieved
from a knowledge base based on the user's query. These documents are then used as context to
guide the LLM's response generation, improving factual accuracy and reducing hallucinations.""",
        source_file="rag_systems.txt",
        file_type="txt",
        expected_topics=["RAG", "LLM", "retrieval", "information retrieval", "context"]
    ),
]

BENCHMARK_QUERIES = [
    BenchmarkQuery(
        query="What is the hippocampus and what functions does it serve?",
        expected_evidence_topics=["hippocampus", "memory", "limbic system"],
        question_type="factual"
    ),
    BenchmarkQuery(
        query="How does deep learning differ from traditional machine learning?",
        expected_evidence_topics=["deep learning", "machine learning", "neural networks"],
        question_type="comparative"
    ),
    BenchmarkQuery(
        query="What are the key features of Python that make it popular?",
        expected_evidence_topics=["Python", "programming", "readability"],
        question_type="factual"
    ),
    BenchmarkQuery(
        query="What happens when the prefrontal cortex is damaged?",
        expected_evidence_topics=["prefrontal cortex", "decision making", "behavior"],
        question_type="factual"
    ),
    BenchmarkQuery(
        query="How does RAG improve the accuracy of language model responses?",
        expected_evidence_topics=["RAG", "retrieval", "LLM", "context"],
        question_type="conceptual"
    ),
]


def get_benchmark_documents() -> List[BenchmarkDocument]:
    """Return all benchmark documents."""
    return BENCHMARK_DOCUMENTS.copy()


def get_benchmark_queries() -> List[BenchmarkQuery]:
    """Return all benchmark queries."""
    return BENCHMARK_QUERIES.copy()


def export_fixtures_as_texts() -> List[str]:
    """Export benchmark documents as plain texts for indexing."""
    return [doc.content for doc in BENCHMARK_DOCUMENTS]


def export_fixtures_as_metadata() -> List[Dict[str, Any]]:
    """Export benchmark documents as metadata dicts."""
    return [
        {
            "source_file": doc.source_file,
            "file_type": doc.file_type,
            "expected_topics": doc.expected_topics,
            "chunk_index": i,
            "total_chunks": len(BENCHMARK_DOCUMENTS)
        }
        for i, doc in enumerate(BENCHMARK_DOCUMENTS)
    ]