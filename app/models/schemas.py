"""
Pydantic models for API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class FileFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    TXT = "txt"
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"


class FusionMethod(str, Enum):
    """Hybrid search fusion methods."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    DBSF = "dbsf"  # Distribution-Based Score Fusion


class ModelProfile(str, Enum):
    """LLM model profiles optimized for Apple Silicon via MLX."""
    FAST = "fast"  # 4B MLX profile, 400 token cap, thinking off
    BALANCED = "balanced"  # 4B MLX profile, 600 token cap, thinking off
    QUALITY = "quality"  # 4B MLX profile, 800 token cap, thinking on


# Collection Management Schemas
class CollectionCreate(BaseModel):
    """Request to create a new collection."""
    collection_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

    @field_validator('collection_name')
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Collection name must contain only alphanumeric characters, hyphens, and underscores')
        return v


class CollectionResponse(BaseModel):
    """Response after collection creation."""
    collection_name: str
    status: str
    vectors_count: int = 0
    message: str


class CollectionDelete(BaseModel):
    """Request to delete a collection."""
    collection_name: str


# Document Upload Schemas
class DocumentMetadata(BaseModel):
    """Metadata for a document chunk."""
    source_file: str
    file_type: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class KnowledgeStatsResponse(BaseModel):
    """Stats from make_knowledge corpus-intelligence ingest."""
    topics: int = 0
    papers: int = 0
    chunks: int = 0
    links: int = 0


class UploadFilesRequest(BaseModel):
    """Request to upload files to a collection."""
    collection_name: str
    file_paths: List[str] = Field(..., min_length=1)
    batch_size: Optional[int] = Field(default=32, ge=1, le=100)
    make_knowledge: bool = Field(
        default=False,
        description="Run corpus-intelligence pipeline (topic sheets + paper summaries)",
    )
    knowledge_profile: Optional[str] = Field(
        default="papers",
        description="Knowledge profile name (profiles/knowledge_<name>.toml)",
    )
    knowledge_model: Optional[str] = Field(
        default=None,
        description="Optional GGUF model path override for this ingest job only",
    )

    @field_validator('file_paths')
    @classmethod
    def validate_file_paths(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('At least one file path is required')
        return v


class UploadFilesResponse(BaseModel):
    """Response after file upload."""
    collection_name: str
    processed_files: int
    total_chunks: int
    failed_files: List[str] = []
    processing_time_seconds: float
    message: str
    knowledge: Optional[KnowledgeStatsResponse] = None
    knowledge_built: bool = False


# Query Schemas
class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    collection_name: str
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=10, ge=1, le=100)
    fusion_method: Optional[FusionMethod] = Field(default=FusionMethod.RRF)
    use_llm: Optional[bool] = Field(default=True)
    conversation_id: Optional[str] = None
    include_metadata: Optional[bool] = Field(default=True)
    model_profile: Optional[ModelProfile] = Field(default=None, description="LLM model profile to use (fast/balanced/quality)")


class RetrievedDocument(BaseModel):
    """A single retrieved document with metadata."""
    content: str
    score: float
    metadata: Dict[str, Any]
    source_file: str
    chunk_index: int


class QueryResponse(BaseModel):
    """Response to a query request."""
    query: str
    answer: Optional[str] = None
    retrieved_documents: List[RetrievedDocument]
    retrieval_method: str
    processing_time_seconds: float
    conversation_id: Optional[str] = None
    model_used: Optional[str] = Field(default=None, description="LLM model that was used for generation")


# Reranker importance-scoring schemas
class ImportanceContext(BaseModel):
    """Importance-scoring context for the reranker."""
    importance_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)


class ConversationContext(BaseModel):
    """Conversation context (history + metadata)."""
    conversation_id: str
    messages: List[Dict[str, str]]
    timestamp: datetime
    metadata: Dict[str, Any] = {}


# Health Check Schema
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    app_name: str
    version: str
    qdrant_connected: bool
    mlx_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
