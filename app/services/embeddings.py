"""
Embedding service for dense and sparse (BM25) vector generation.
Uses FastEmbed for efficient embedding generation.
"""

from typing import List, Tuple, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastembed import TextEmbedding, SparseTextEmbedding
from loguru import logger

from app.config import get_settings
from app.utils.async_helpers import AsyncBatchProcessor, chunks


class EmbeddingService:
    """Service for generating dense and sparse embeddings."""

    def __init__(self):
        self.settings = get_settings()
        self.dense_model = None
        self.sparse_model = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize embedding models lazily."""
        async with self._lock:
            if self.dense_model is None:
                logger.info(f"Loading dense embedding model: {self.settings.DENSE_MODEL_NAME}")
                loop = asyncio.get_event_loop()
                self.dense_model = await loop.run_in_executor(
                    None,
                    lambda: TextEmbedding(model_name=self.settings.DENSE_MODEL_NAME)
                )
                logger.info("Dense embedding model loaded successfully")

            if self.sparse_model is None:
                logger.info(f"Loading sparse embedding model: {self.settings.SPARSE_MODEL_NAME}")
                loop = asyncio.get_event_loop()
                self.sparse_model = await loop.run_in_executor(
                    None,
                    lambda: SparseTextEmbedding(model_name=self.settings.SPARSE_MODEL_NAME)
                )
                logger.info("Sparse embedding model loaded successfully")

    async def generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of dense embedding vectors
        """
        await self.initialize()

        if not texts:
            return []

        logger.debug(f"Generating dense embeddings for {len(texts)} texts")

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: list(self.dense_model.embed(texts))
            )

            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]

            logger.debug(f"Generated {len(embeddings_list)} dense embeddings")
            return embeddings_list

        except Exception as e:
            logger.error(f"Error generating dense embeddings: {str(e)}")
            raise

    async def generate_sparse_embeddings(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Generate sparse BM25 embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of sparse embedding dictionaries {index: value}
        """
        await self.initialize()

        if not texts:
            return []

        logger.debug(f"Generating sparse embeddings for {len(texts)} texts")

        try:
            loop = asyncio.get_event_loop()
            sparse_embeddings = await loop.run_in_executor(
                None,
                lambda: list(self.sparse_model.embed(texts))
            )

            # Convert sparse embeddings to dict format
            sparse_dicts = []
            for sparse_emb in sparse_embeddings:
                # sparse_emb is a SparseEmbedding object with indices and values
                sparse_dict = {
                    int(idx): float(val)
                    for idx, val in zip(sparse_emb.indices, sparse_emb.values)
                }
                sparse_dicts.append(sparse_dict)

            logger.debug(f"Generated {len(sparse_dicts)} sparse embeddings")
            return sparse_dicts

        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {str(e)}")
            raise

    async def generate_hybrid_embeddings(
        self,
        texts: List[str]
    ) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Generate both dense and sparse embeddings concurrently.

        Args:
            texts: List of text strings

        Returns:
            Tuple of (dense_embeddings, sparse_embeddings)
        """
        await self.initialize()

        if not texts:
            return [], []

        logger.info(f"Generating hybrid embeddings for {len(texts)} texts")

        try:
            # Generate both types concurrently
            dense_task = self.generate_dense_embeddings(texts)
            sparse_task = self.generate_sparse_embeddings(texts)

            dense_embeddings, sparse_embeddings = await asyncio.gather(
                dense_task,
                sparse_task
            )

            logger.info(
                f"Generated {len(dense_embeddings)} dense and "
                f"{len(sparse_embeddings)} sparse embeddings"
            )

            return dense_embeddings, sparse_embeddings

        except Exception as e:
            logger.error(f"Error generating hybrid embeddings: {str(e)}")
            raise

    async def generate_query_embeddings(
        self,
        query: str,
        include_sparse: bool = True
    ) -> Tuple[List[float], Dict[int, float] | None]:
        """
        Generate embeddings for a query string.

        Args:
            query: Query text
            include_sparse: Whether to include sparse embeddings

        Returns:
            Tuple of (dense_embedding, sparse_embedding or None)
        """
        await self.initialize()

        logger.debug(f"Generating query embeddings for: {query[:50]}...")

        try:
            if include_sparse:
                dense_embs, sparse_embs = await self.generate_hybrid_embeddings([query])
                return dense_embs[0], sparse_embs[0]
            else:
                dense_embs = await self.generate_dense_embeddings([query])
                return dense_embs[0], None

        except Exception as e:
            logger.error(f"Error generating query embeddings: {str(e)}")
            raise

    async def batch_generate_hybrid_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """
        Generate hybrid embeddings in batches for large text lists.

        Args:
            texts: List of text strings
            batch_size: Size of each batch

        Returns:
            Tuple of (all_dense_embeddings, all_sparse_embeddings)
        """
        await self.initialize()

        if not texts:
            return [], []

        logger.info(f"Batch generating hybrid embeddings for {len(texts)} texts")

        all_dense = []
        all_sparse = []

        # Process in batches
        for i, batch in enumerate(chunks(texts, batch_size)):
            logger.debug(f"Processing batch {i + 1} ({len(batch)} texts)")

            dense_batch, sparse_batch = await self.generate_hybrid_embeddings(batch)

            all_dense.extend(dense_batch)
            all_sparse.extend(sparse_batch)

        logger.info(
            f"Completed batch generation: {len(all_dense)} dense, "
            f"{len(all_sparse)} sparse embeddings"
        )

        return all_dense, all_sparse

    def get_dense_dimension(self) -> int:
        """Get the dimension of dense embeddings."""
        return self.settings.DENSE_VECTOR_SIZE

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up embedding models")
        self.dense_model = None
        self.sparse_model = None
