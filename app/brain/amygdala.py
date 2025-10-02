"""
Amygdala module - Emotional importance and priority scoring.

Inspired by the amygdala brain region responsible for:
- Emotional processing and valence
- Salience detection (what's important)
- Priority assignment to memories
- Modulation of memory consolidation
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import math
from loguru import logger

from app.config import get_settings
from app.models.schemas import EmotionalContext


class Amygdala:
    """
    Amygdala module for importance and priority scoring.

    This module handles:
    - Importance scoring of retrieved documents
    - Recency bias (recent memories are more salient)
    - Relevance weighting based on search scores
    - Emotional tagging and filtering
    """

    def __init__(self):
        self.settings = get_settings()
        self.recency_weight = self.settings.RECENCY_WEIGHT
        self.relevance_weight = self.settings.RELEVANCE_WEIGHT
        self.importance_weight = self.settings.IMPORTANCE_WEIGHT
        logger.info("Amygdala module initialized")

    def calculate_importance_score(
        self,
        document: Dict[str, Any],
        query: str,
        current_time: datetime = None
    ) -> float:
        """
        Calculate overall importance score for a retrieved document.

        The score combines:
        - Relevance: How well the document matches the query (from search score)
        - Recency: How recent the document is (temporal salience)
        - Base importance: Pre-assigned importance in metadata

        Args:
            document: Retrieved document with metadata
            query: Original query
            current_time: Current timestamp (default: now)

        Returns:
            Combined importance score (0.0 to 1.0)
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # Extract scores
        relevance_score = self._normalize_relevance_score(document.get('score', 0.0))
        recency_score = self._calculate_recency_score(document, current_time)
        base_importance = document.get('metadata', {}).get('importance_score', 0.5)

        # Weighted combination
        importance_score = (
            self.relevance_weight * relevance_score +
            self.recency_weight * recency_score +
            self.importance_weight * base_importance
        )

        logger.debug(
            f"Amygdala: Importance score={importance_score:.3f} "
            f"(relevance={relevance_score:.3f}, recency={recency_score:.3f}, "
            f"base={base_importance:.3f})"
        )

        return min(max(importance_score, 0.0), 1.0)  # Clamp to [0, 1]

    def _normalize_relevance_score(self, raw_score: float) -> float:
        """
        Normalize relevance score to [0, 1] range.

        Qdrant scores vary by distance metric:
        - Cosine: typically [0, 2], higher is better
        - We normalize using sigmoid-like transformation
        """
        # Simple sigmoid normalization
        # Assuming scores are typically in [0, 1] range for cosine similarity
        return min(max(raw_score, 0.0), 1.0)

    def _calculate_recency_score(
        self,
        document: Dict[str, Any],
        current_time: datetime
    ) -> float:
        """
        Calculate recency score with exponential decay.

        Recent documents get higher scores, with exponential decay over time.

        Args:
            document: Document with metadata
            current_time: Current timestamp

        Returns:
            Recency score (0.0 to 1.0)
        """
        metadata = document.get('metadata', {})
        indexed_at_str = metadata.get('indexed_at')

        if not indexed_at_str:
            # No timestamp, return neutral score
            return 0.5

        try:
            indexed_at = datetime.fromisoformat(indexed_at_str)
            age_seconds = (current_time - indexed_at).total_seconds()

            # Exponential decay: score = exp(-age / half_life)
            # Half-life of 7 days = 604800 seconds
            half_life_seconds = 7 * 24 * 60 * 60

            recency_score = math.exp(-age_seconds / half_life_seconds)

            return recency_score

        except (ValueError, TypeError) as e:
            logger.warning(f"Amygdala: Error parsing timestamp: {str(e)}")
            return 0.5

    def rank_by_importance(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rank documents by importance score.

        Args:
            documents: List of retrieved documents
            query: Original query

        Returns:
            Documents sorted by importance (highest first)
        """
        logger.info(f"Amygdala: Ranking {len(documents)} documents by importance")

        current_time = datetime.utcnow()

        # Calculate importance scores
        scored_documents = []
        for doc in documents:
            importance_score = self.calculate_importance_score(
                doc, query, current_time
            )

            # Add importance score to document
            doc_with_importance = doc.copy()
            doc_with_importance['importance_score'] = importance_score

            # Store emotional context
            doc_with_importance['emotional_context'] = EmotionalContext(
                importance_score=importance_score,
                recency_score=self._calculate_recency_score(doc, current_time),
                relevance_score=self._normalize_relevance_score(doc.get('score', 0.0))
            ).dict()

            scored_documents.append(doc_with_importance)

        # Sort by importance score (descending)
        ranked_documents = sorted(
            scored_documents,
            key=lambda x: x['importance_score'],
            reverse=True
        )

        logger.info(
            f"Amygdala: Documents ranked. "
            f"Top score: {ranked_documents[0]['importance_score']:.3f}, "
            f"Bottom score: {ranked_documents[-1]['importance_score']:.3f}"
        )

        return ranked_documents

    def filter_by_importance_threshold(
        self,
        documents: List[Dict[str, Any]],
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Filter documents by minimum importance threshold.

        Args:
            documents: Documents with importance scores
            threshold: Minimum importance score (0.0 to 1.0)

        Returns:
            Filtered documents
        """
        filtered = [
            doc for doc in documents
            if doc.get('importance_score', 0.0) >= threshold
        ]

        logger.info(
            f"Amygdala: Filtered {len(documents)} documents to {len(filtered)} "
            f"with threshold {threshold}"
        )

        return filtered

    def detect_salient_features(
        self,
        document: Dict[str, Any]
    ) -> List[str]:
        """
        Detect salient features in a document (emotional markers).

        This is a simple heuristic-based approach.
        In a more advanced system, this could use NER or sentiment analysis.

        Args:
            document: Document to analyze

        Returns:
            List of salient features/keywords
        """
        text = document.get('text', '')
        metadata = document.get('metadata', {})

        salient_features = []

        # Check for numbers/statistics (high information content)
        if any(char.isdigit() for char in text):
            salient_features.append('contains_numbers')

        # Check file type (some formats may be more authoritative)
        file_type = metadata.get('file_type', '').lower()
        if file_type in ['.pdf', '.docx']:
            salient_features.append('formal_document')

        # Check for headings or structure indicators
        if any(indicator in text.lower() for indicator in ['chapter', 'section', '##', '###']):
            salient_features.append('structured_content')

        # Check length (longer may be more comprehensive)
        if len(text) > 1000:
            salient_features.append('comprehensive')

        return salient_features

    def apply_emotional_boost(
        self,
        documents: List[Dict[str, Any]],
        boost_keywords: List[str],
        boost_factor: float = 1.2
    ) -> List[Dict[str, Any]]:
        """
        Apply importance boost to documents containing specific keywords.

        Simulates emotional arousal increasing memory consolidation.

        Args:
            documents: Documents with importance scores
            boost_keywords: Keywords that trigger boost
            boost_factor: Multiplicative boost factor

        Returns:
            Documents with boosted scores
        """
        boosted_documents = []

        for doc in documents:
            text = doc.get('text', '').lower()
            importance = doc.get('importance_score', 0.5)

            # Check if any boost keyword is present
            if any(keyword.lower() in text for keyword in boost_keywords):
                new_importance = min(importance * boost_factor, 1.0)
                logger.debug(
                    f"Amygdala: Emotional boost applied. "
                    f"Score: {importance:.3f} -> {new_importance:.3f}"
                )
                doc['importance_score'] = new_importance

            boosted_documents.append(doc)

        return boosted_documents
