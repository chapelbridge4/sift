"""
Hippocampus module - Memory indexing and retrieval.

Inspired by the hippocampus brain region responsible for:
- Formation of new memories (indexing documents)
- Storage and retrieval of long-term memories (vector search)
- Spatial navigation and context (semantic search)
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.document_parser import DocumentParser
from app.services.qdrant_service import QdrantService


class Hippocampus:
    """
    Hippocampus module for memory indexing and retrieval.

    This module handles:
    - Document indexing (memory formation)
    - Vector-based retrieval (memory recall)
    - Context-aware search (spatial navigation)
    """

    def __init__(self):
        self.qdrant_service = QdrantService()
        self.document_parser = DocumentParser()
        logger.info("Hippocampus module initialized")

    async def initialize(self):
        """Initialize the hippocampus module."""
        await self.qdrant_service.initialize()
        logger.info("Hippocampus memory systems online")

    async def form_memories(
        self,
        collection_name: str,
        file_paths: List[str],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Form new memories by indexing documents (long-term potentiation).

        Args:
            collection_name: Memory collection name
            file_paths: Paths to documents to index
            batch_size: Batch size for processing

        Returns:
            Dictionary with indexing statistics
        """
        logger.info(f"Hippocampus: Forming new memories from {len(file_paths)} documents")

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

            logger.info(f"Hippocampus: Successfully formed {total_indexed} new memories")

            return {
                "success": True,
                "total_chunks": total_indexed,
                "processed_files": len(file_paths),
                "message": f"Successfully indexed {total_indexed} memory chunks"
            }

        except Exception as e:
            logger.error(f"Hippocampus: Error forming memories: {str(e)}")
            raise

    async def recall_memories(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        fusion_method: str = "rrf"
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
        logger.info(f"Hippocampus: Recalling memories for query_length={len(query)}")

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

            logger.info(f"Hippocampus: Recalled {len(memories)} relevant memories")

            return memories

        except Exception as e:
            logger.error(f"Hippocampus: Error recalling memories: {str(e)}")
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
        logger.info(f"Hippocampus: Creating new memory space: {collection_name}")

        try:
            success = await self.qdrant_service.create_collection(
                collection_name=collection_name,
                dense_vector_size=vector_size
            )

            if success:
                logger.info(f"Hippocampus: Memory space '{collection_name}' created successfully")
            else:
                logger.warning(f"Hippocampus: Memory space '{collection_name}' already exists")

            return success

        except Exception as e:
            logger.error(f"Hippocampus: Error creating memory space: {str(e)}")
            raise

    async def forget_memories(self, collection_name: str) -> bool:
        """
        Forget all memories in a collection (memory erasure).

        Args:
            collection_name: Collection to forget

        Returns:
            True if successful
        """
        logger.info(f"Hippocampus: Forgetting memory space: {collection_name}")

        try:
            success = await self.qdrant_service.delete_collection(
                collection_name=collection_name
            )

            logger.info(f"Hippocampus: Memory space '{collection_name}' forgotten")
            return success

        except Exception as e:
            logger.error(f"Hippocampus: Error forgetting memories: {str(e)}")
            raise

    async def get_memory_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Args:
            collection_name: Collection to analyze

        Returns:
            Dictionary with memory statistics
        """
        logger.debug(f"Hippocampus: Retrieving memory stats for '{collection_name}'")

        try:
            stats = await self.qdrant_service.get_collection_info(collection_name)

            logger.debug(f"Hippocampus: Memory stats retrieved for '{collection_name}'")
            return stats

        except Exception as e:
            logger.error(f"Hippocampus: Error getting memory stats: {str(e)}")
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
