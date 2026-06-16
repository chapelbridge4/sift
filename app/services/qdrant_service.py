"""
Qdrant service for vector database operations with hybrid search support.
Uses AsyncQdrantClient for concurrent operations.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    Filter,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.config import get_settings
from app.services.embeddings import EmbeddingService
from app.utils.async_helpers import ProgressTracker, chunks


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[AsyncQdrantClient] = None
        self.embedding_service = EmbeddingService()

    async def initialize(self):
        """Initialize Qdrant client connection.

        Mode is selected by QDRANT_MODE (env var checked first, so monkeypatching
        works even when settings is cached by lru_cache), falling back to
        self.settings.QDRANT_MODE, then to 'embedded'.

        "embedded" → AsyncQdrantClient(path=...) — no server needed (zero-infra default)
        "server"   → AsyncQdrantClient(host=..., port=...) — requires a running Qdrant server
        """
        if self.client is None:
            mode = (
                os.getenv("QDRANT_MODE")
                or getattr(self.settings, "QDRANT_MODE", "embedded")
                or "embedded"
            ).lower()

            if mode == "embedded":
                qdrant_path = (
                    os.getenv("QDRANT_PATH")
                    or getattr(self.settings, "QDRANT_PATH", "./qdrant_data")
                    or "./qdrant_data"
                )
                logger.info(f"Starting Qdrant in embedded mode (path={qdrant_path})")
                self.client = AsyncQdrantClient(path=qdrant_path)
            else:
                logger.info(
                    f"Connecting to Qdrant server at {self.settings.QDRANT_HOST}:{self.settings.QDRANT_PORT}"
                )
                self.client = AsyncQdrantClient(
                    host=self.settings.QDRANT_HOST,
                    port=self.settings.QDRANT_PORT,
                    api_key=self.settings.QDRANT_API_KEY,
                    timeout=self.settings.QDRANT_TIMEOUT,
                    prefer_grpc=False,  # Disable gRPC to avoid SSL issues with local Qdrant
                    https=False,  # Use HTTP instead of HTTPS for local Qdrant
                )

            # Test connection
            try:
                collections = await self.client.get_collections()
                logger.info(f"Connected to Qdrant. Available collections: {len(collections.collections)}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {str(e)}")
                raise

            # Initialize embedding service
            await self.embedding_service.initialize()

    async def create_collection(
        self,
        collection_name: str,
        dense_vector_size: Optional[int] = None
    ) -> bool:
        """
        Create a new collection with hybrid search support (dense + sparse vectors).

        Args:
            collection_name: Name of the collection
            dense_vector_size: Size of dense vectors (default from config)

        Returns:
            True if successful
        """
        await self.initialize()

        if dense_vector_size is None:
            dense_vector_size = self.settings.DENSE_VECTOR_SIZE

        logger.info(f"Creating collection '{collection_name}' with hybrid search support")

        try:
            # Check if collection already exists
            collections = await self.client.get_collections()
            if collection_name in [c.name for c in collections.collections]:
                logger.warning(f"Collection '{collection_name}' already exists")
                return False

            # Create collection with dense and sparse vector configs
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        ),
                        modifier=Modifier.IDF,  # Important for BM25
                    )
                },
            )

            logger.info(f"Collection '{collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful
        """
        await self.initialize()

        logger.info(f"Deleting collection '{collection_name}'")

        try:
            await self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        await self.initialize()

        try:
            collections = await self.client.get_collections()
            return collection_name in [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        await self.initialize()

        try:
            collection_info = await self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise

    async def upsert_documents(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> int:
        """
        Upsert documents with hybrid embeddings (dense + sparse).

        Args:
            collection_name: Name of the collection
            texts: List of text chunks
            metadatas: List of metadata dicts
            batch_size: Batch size for processing

        Returns:
            Number of documents upserted
        """
        await self.initialize()

        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")

        logger.info(f"Upserting {len(texts)} documents to '{collection_name}'")

        try:
            progress = ProgressTracker(total=len(texts), description="Upserting documents")

            total_upserted = 0

            # Process in batches
            for batch_idx, batch_texts in enumerate(chunks(texts, batch_size)):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + len(batch_texts), len(texts))
                batch_metadatas = metadatas[batch_start:batch_end]

                # Generate embeddings for batch
                dense_embeddings, sparse_embeddings = await self.embedding_service.generate_hybrid_embeddings(
                    batch_texts
                )

                # Create points
                points = []
                for i, (text, metadata, dense_emb, sparse_emb) in enumerate(
                    zip(batch_texts, batch_metadatas, dense_embeddings, sparse_embeddings)
                ):
                    point_id = str(uuid.uuid4())

                    # Add timestamp to metadata
                    metadata["indexed_at"] = datetime.utcnow().isoformat()
                    metadata["text_preview"] = text[:200]  # Store preview

                    point = PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_emb,
                            "sparse": models.SparseVector(
                                indices=list(sparse_emb.keys()),
                                values=list(sparse_emb.values())
                            )
                        },
                        payload={
                            "text": text,
                            **metadata
                        }
                    )
                    points.append(point)

                # Upsert batch
                await self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )

                total_upserted += len(points)
                progress.update(len(points))

            logger.info(f"Successfully upserted {total_upserted} documents")
            return total_upserted

        except Exception as e:
            logger.error(f"Error upserting documents: {str(e)}")
            raise

    async def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (dense + sparse) with fusion.

        Args:
            collection_name: Name of the collection
            query_text: Query text
            top_k: Number of results to return
            fusion_method: Fusion method ("rrf" or "dbsf")
            filters: Optional filters

        Returns:
            List of search results with scores and metadata
        """
        await self.initialize()

        logger.info(f"Performing hybrid search on '{collection_name}' with {fusion_method} fusion")

        try:
            # Generate query embeddings
            dense_query, sparse_query = await self.embedding_service.generate_query_embeddings(
                query_text,
                include_sparse=True
            )

            # Convert sparse query to SparseVector
            sparse_vector = models.SparseVector(
                indices=list(sparse_query.keys()),
                values=list(sparse_query.values())
            )

            # Perform query with prefetch for hybrid search
            search_result = await self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using="dense",
                        limit=top_k * 2,  # Prefetch more for better fusion
                        filter=filters,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=top_k * 2,
                        filter=filters,
                    ),
                ],
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF if fusion_method == "rrf" else models.Fusion.DBSF
                ),
                limit=top_k,
                with_payload=True,
            )

            # Format results
            results = []
            for point in search_result.points:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in point.payload.items()
                        if k != "text"
                    }
                }
                results.append(result)

            logger.info(f"Hybrid search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            raise

    async def dense_search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform dense-only vector search.

        Args:
            collection_name: Name of the collection
            query_text: Query text
            top_k: Number of results to return
            filters: Optional filters

        Returns:
            List of search results
        """
        await self.initialize()

        logger.info(f"Performing dense search on '{collection_name}'")

        try:
            # Generate dense query embedding
            dense_query, _ = await self.embedding_service.generate_query_embeddings(
                query_text,
                include_sparse=False
            )

            # Perform search
            search_result = await self.client.query_points(
                collection_name=collection_name,
                query=dense_query,
                using="dense",
                limit=top_k,
                query_filter=filters,
                with_payload=True,
            )

            # Format results
            results = []
            for point in search_result.points:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in point.payload.items()
                        if k != "text"
                    }
                }
                results.append(result)

            # Log warning if top result score is below threshold
            if results and results[0]["score"] < self.settings.RETRIEVAL_SCORE_THRESHOLD:
                logger.warning(
                    "Low retrieval confidence: "
                    f"top score {results[0]['score']:.3f} < threshold {self.settings.RETRIEVAL_SCORE_THRESHOLD}; "
                    f"query_length={len(query_text)}"
                )

            logger.info(f"Dense search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during dense search: {str(e)}")
            raise

    async def sparse_search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform sparse-only (BM25) search.

        Args:
            collection_name: Name of the collection
            query_text: Query text
            top_k: Number of results to return
            filters: Optional filters

        Returns:
            List of search results
        """
        await self.initialize()

        logger.info(f"Performing sparse search on '{collection_name}'")

        try:
            # Generate sparse query embedding
            _, sparse_query = await self.embedding_service.generate_query_embeddings(
                query_text,
                include_sparse=True
            )

            # Convert to SparseVector
            sparse_vector = models.SparseVector(
                indices=list(sparse_query.keys()),
                values=list(sparse_query.values())
            )

            # Perform search using query_vector
            search_result = await self.client.query_points(
                collection_name=collection_name,
                query=sparse_vector,
                using="sparse",
                limit=top_k,
                query_filter=filters,
                with_payload=True,
            )

            # Format results
            results = []
            for point in search_result.points:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in point.payload.items()
                        if k != "text"
                    }
                }
                results.append(result)

            logger.info(f"Sparse search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during sparse search: {str(e)}")
            raise

    async def retrieve(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Unified retrieval method that dispatches based on RETRIEVAL_STRATEGY config.

        Args:
            collection_name: Name of the collection
            query_text: Query text
            top_k: Number of results to return
            filters: Optional filters

        Returns:
            List of search results
        """
        strategy = self.settings.RETRIEVAL_STRATEGY.lower()

        if strategy == "dense":
            return await self.dense_search(
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
                filters=filters,
            )
        elif strategy == "sparse":
            return await self.sparse_search(
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
                filters=filters,
            )
        elif strategy == "hybrid":
            fusion_method = self.settings.FUSION_METHOD.lower()
            return await self.hybrid_search(
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
                fusion_method=fusion_method,
                filters=filters,
            )
        else:
            logger.warning(f"Unknown retrieval strategy '{strategy}', defaulting to dense")
            return await self.dense_search(
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
                filters=filters,
            )

    async def close(self):
        """Close Qdrant client connection."""
        if self.client:
            await self.client.close()
            logger.info("Qdrant client connection closed")

    async def health_check(self) -> bool:
        """Check if Qdrant is accessible."""
        try:
            await self.initialize()
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False
