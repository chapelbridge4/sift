"""
DocumentStore module - Document indexing and retrieval.

Responsible for:
- Indexing documents into the vector store
- Vector-based retrieval (dense and hybrid search)
- Collection lifecycle management
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import get_settings
from app.knowledge.backend import KnowledgeLLM, get_knowledge_backend
from app.knowledge.config import load_profile
from app.knowledge.index import index_artifacts
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.retrieval import (
    apply_topic_score_boost,
    build_paper_drill_down_filter,
    extract_drill_down_paper_ids,
    is_knowledge_collection,
    merge_drill_down_memories,
)
from app.services.document_parser import DocumentParser
from app.services.embeddings import EmbeddingService
from app.services.qdrant_service import QdrantService


class DocumentStore:
    """
    DocumentStore module for memory indexing and retrieval.

    This module handles:
    - Document indexing (memory formation)
    - Vector-based retrieval (memory recall)
    - Context-aware search (spatial navigation)
    """

    def __init__(self):
        self.qdrant_service = QdrantService()
        self.document_parser = DocumentParser()
        logger.info("DocumentStore module initialized")

    async def initialize(self):
        """Initialize the document store module."""
        await self.qdrant_service.initialize()
        logger.info("DocumentStore memory systems online")

    async def form_memories(
        self,
        collection_name: str,
        file_paths: List[str],
        batch_size: int = 32,
        *,
        make_knowledge: bool = False,
        knowledge_profile: Optional[str] = None,
        knowledge_model: Optional[str] = None,
        knowledge_pipeline: Optional[KnowledgePipeline] = None,
    ) -> Dict[str, Any]:
        """
        Form new memories by indexing documents (long-term potentiation).

        Args:
            collection_name: Memory collection name
            file_paths: Paths to documents to index
            batch_size: Batch size for processing
            make_knowledge: Run corpus-intelligence pipeline (topic sheets + paper summaries)
            knowledge_profile: Profile name (e.g. papers)
            knowledge_model: Optional GGUF path override for this ingest job only
            knowledge_pipeline: Injected pipeline for tests (DI)

        Returns:
            Dictionary with indexing statistics
        """
        logger.info(f"DocumentStore: Forming new memories from {len(file_paths)} documents")

        if make_knowledge:
            return await self._form_knowledge_memories(
                collection_name=collection_name,
                file_paths=file_paths,
                batch_size=batch_size,
                knowledge_profile=knowledge_profile,
                knowledge_model=knowledge_model,
                knowledge_pipeline=knowledge_pipeline,
            )

        try:
            # Parse documents into chunks (encoding)
            document_chunks = await self.document_parser.parse_files(file_paths)

            if not document_chunks:
                logger.warning("No document chunks extracted")
                return {
                    "success": False,
                    "total_chunks": 0,
                    "message": "No content could be extracted from the provided files"
                }

            # Extract texts and metadata
            texts = [chunk.content for chunk in document_chunks]
            metadatas = [chunk.metadata for chunk in document_chunks]

            # Store in long-term memory (consolidation)
            total_indexed = await self.qdrant_service.upsert_documents(
                collection_name=collection_name,
                texts=texts,
                metadatas=metadatas,
                batch_size=batch_size
            )

            logger.info(f"DocumentStore: Successfully formed {total_indexed} new memories")

            return {
                "success": True,
                "total_chunks": total_indexed,
                "processed_files": len(file_paths),
                "message": f"Successfully indexed {total_indexed} memory chunks"
            }

        except Exception as e:
            logger.error(f"DocumentStore: Error forming memories: {str(e)}")
            raise

    async def _form_knowledge_memories(
        self,
        *,
        collection_name: str,
        file_paths: List[str],
        batch_size: int,
        knowledge_profile: Optional[str],
        knowledge_model: Optional[str],
        knowledge_pipeline: Optional[KnowledgePipeline],
    ) -> Dict[str, Any]:
        """make_knowledge branch: Tier 0→2 artifacts, index canonical markdown only."""
        settings = get_settings()
        profile_name = knowledge_profile or settings.KNOWLEDGE_PROFILE
        profile = load_profile(profile_name)

        if knowledge_model:
            profile = replace(profile, llm=replace(profile.llm, model_path=knowledge_model))

        if settings.KNOWLEDGE_OUTPUT_DIR:
            output_root = Path(settings.KNOWLEDGE_OUTPUT_DIR).expanduser()
        else:
            output_root = Path(settings.ALLOWED_CORPUS_DIR) / ".knowledge"

        artifact_dir = output_root / collection_name

        pipeline = knowledge_pipeline or KnowledgePipeline(
            parser=self.document_parser,
            embedder=EmbeddingService(),
            llm=KnowledgeLLM(get_knowledge_backend(profile)),
            profile=profile,
            output_dir=output_root,
        )

        stats = await pipeline.run(file_paths, collection_name)

        indexed = await index_artifacts(
            collection_name=collection_name,
            artifact_dir=artifact_dir,
            profile=profile,
            qdrant_service=self.qdrant_service,
            batch_size=batch_size,
        )
        stats = stats.model_copy(update={"chunks": indexed})

        logger.info(
            "DocumentStore: Knowledge ingest complete topics={} papers={} chunks={}",
            stats.topics,
            stats.papers,
            stats.chunks,
        )

        return {
            "success": True,
            "total_chunks": indexed,
            "processed_files": len(file_paths),
            "knowledge_built": True,
            "knowledge_profile": profile.name,
            "knowledge": stats.model_dump(),
            "message": (
                f"Knowledge pipeline indexed {indexed} artifact chunks "
                f"({stats.papers} papers, {stats.topics} topics)"
            ),
        }

    async def recall_memories(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        fusion_method: str = "rrf",
        drill_down: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Recall memories through retrieval (memory recall/recognition).

        Args:
            collection_name: Memory collection to search
            query: Query to recall memories
            top_k: Number of memories to recall
            use_hybrid: Use hybrid search (dense + sparse)
            fusion_method: Fusion method for hybrid search

        Returns:
            List of recalled memories
        """
        logger.info(f"DocumentStore: Recalling memories for query_length={len(query)}")

        try:
            if use_hybrid:
                # Pattern completion using multiple cues (hybrid search)
                memories = await self.qdrant_service.hybrid_search(
                    collection_name=collection_name,
                    query_text=query,
                    top_k=top_k,
                    fusion_method=fusion_method
                )
            else:
                # Simple semantic recall (dense search only)
                memories = await self.qdrant_service.dense_search(
                    collection_name=collection_name,
                    query_text=query,
                    top_k=top_k
                )

            if is_knowledge_collection(memories):
                settings = get_settings()
                profile_name = next(
                    (
                        m.get("metadata", {}).get("knowledge_profile")
                        for m in memories
                        if m.get("metadata", {}).get("knowledge_profile")
                    ),
                    settings.KNOWLEDGE_PROFILE,
                )
                try:
                    profile = load_profile(profile_name)
                    topic_boost = profile.retrieval.topic_score_boost
                    drill_down_top_k = profile.retrieval.drill_down_top_k
                except FileNotFoundError:
                    topic_boost = settings.KNOWLEDGE_TOPIC_SCORE_BOOST
                    drill_down_top_k = 5

                memories = apply_topic_score_boost(memories, topic_boost)

                # v1: explicit drill_down flag only; auto citation-seeking heuristics deferred v2.
                if drill_down:
                    paper_ids = extract_drill_down_paper_ids(
                        memories,
                        top_k=drill_down_top_k,
                    )
                    if paper_ids:
                        paper_filter = build_paper_drill_down_filter(paper_ids)
                        if use_hybrid:
                            paper_memories = await self.qdrant_service.hybrid_search(
                                collection_name=collection_name,
                                query_text=query,
                                top_k=len(paper_ids),
                                fusion_method=fusion_method,
                                filters=paper_filter,
                            )
                        else:
                            paper_memories = await self.qdrant_service.dense_search(
                                collection_name=collection_name,
                                query_text=query,
                                top_k=len(paper_ids),
                                filters=paper_filter,
                            )
                        memories = merge_drill_down_memories(memories, paper_memories)
                        logger.info(
                            "DocumentStore: Drill-down added {} paper summaries "
                            "from {} linked IDs",
                            len(paper_memories),
                            len(paper_ids),
                        )

            logger.info(f"DocumentStore: Recalled {len(memories)} relevant memories")

            return memories

        except Exception as e:
            logger.error(f"DocumentStore: Error recalling memories: {str(e)}")
            raise

    async def create_memory_space(
        self,
        collection_name: str,
        vector_size: Optional[int] = None
    ) -> bool:
        """
        Create a new memory space (neurogenesis).

        Args:
            collection_name: Name for the new memory space
            vector_size: Size of embedding vectors

        Returns:
            True if successful
        """
        logger.info(f"DocumentStore: Creating new memory space: {collection_name}")

        try:
            success = await self.qdrant_service.create_collection(
                collection_name=collection_name,
                dense_vector_size=vector_size
            )

            if success:
                logger.info(f"DocumentStore: Memory space '{collection_name}' created successfully")
            else:
                logger.warning(f"DocumentStore: Memory space '{collection_name}' already exists")

            return success

        except Exception as e:
            logger.error(f"DocumentStore: Error creating memory space: {str(e)}")
            raise

    async def forget_memories(self, collection_name: str) -> bool:
        """
        Forget all memories in a collection (memory erasure).

        Args:
            collection_name: Collection to forget

        Returns:
            True if successful
        """
        logger.info(f"DocumentStore: Forgetting memory space: {collection_name}")

        try:
            success = await self.qdrant_service.delete_collection(
                collection_name=collection_name
            )

            logger.info(f"DocumentStore: Memory space '{collection_name}' forgotten")
            return success

        except Exception as e:
            logger.error(f"DocumentStore: Error forgetting memories: {str(e)}")
            raise

    async def get_memory_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Args:
            collection_name: Collection to analyze

        Returns:
            Dictionary with memory statistics
        """
        logger.debug(f"DocumentStore: Retrieving memory stats for '{collection_name}'")

        try:
            stats = await self.qdrant_service.get_collection_info(collection_name)

            logger.debug(f"DocumentStore: Memory stats retrieved for '{collection_name}'")
            return stats

        except Exception as e:
            logger.error(f"DocumentStore: Error getting memory stats: {str(e)}")
            raise

    async def memory_exists(self, collection_name: str) -> bool:
        """
        Check if a memory space exists.

        Args:
            collection_name: Collection name to check

        Returns:
            True if exists
        """
        return await self.qdrant_service.collection_exists(collection_name)
