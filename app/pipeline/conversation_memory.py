"""
Conversation Memory module - Temporary contextual buffer.

Responsible for:
- Maintaining conversation context
- Short-term conversation history buffering
- Context window management
"""

import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import get_settings
from app.models.schemas import WorkingMemoryContext


class ConversationMemory:
    """
    Conversation Memory module for temporary context storage.

    This module handles:
    - Conversation history management
    - Contextual information buffering
    - Short-term state maintenance
    - Context window management
    """

    def __init__(self):
        self.settings = get_settings()
        self.context_window_size = self.settings.CONTEXT_WINDOW_SIZE
        self.max_conversation_history = self.settings.MAX_CONVERSATION_HISTORY

        # In-memory storage (in production, use Redis or similar)
        self.conversations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_conversation_history)
        )
        self.metadata: Dict[str, Dict[str, Any]] = {}

        logger.info("Conversation Memory module initialized")

    async def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Create a new conversation context.

        Args:
            conversation_id: Optional conversation ID (generates one if not provided)

        Returns:
            Conversation ID
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = deque(maxlen=self.max_conversation_history)
            self.metadata[conversation_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "message_count": 0
            }
            logger.info(f"Conversation Memory: Created conversation {conversation_id}")

        return conversation_id

    async def add_message(
        self,
        conversation_id: str,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a message to the conversation buffer.

        Args:
            conversation_id: Conversation ID
            message: Message dict with 'role' and 'content'
            metadata: Optional message metadata
        """
        if conversation_id not in self.conversations:
            await self.create_conversation(conversation_id)

        message_with_metadata = {
            **message,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        self.conversations[conversation_id].append(message_with_metadata)

        # Update conversation metadata
        self.metadata[conversation_id]["last_accessed"] = datetime.utcnow().isoformat()
        self.metadata[conversation_id]["message_count"] += 1

        logger.debug(
            f"Conversation Memory: Added message to conversation {conversation_id}. "
            f"Total messages: {len(self.conversations[conversation_id])}"
        )

    async def get_conversation_history(
        self,
        conversation_id: str,
        include_metadata: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get conversation history from working memory.

        Args:
            conversation_id: Conversation ID
            include_metadata: Whether to include message metadata

        Returns:
            List of messages
        """
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation Memory: Conversation {conversation_id} not found")
            return []

        # Update last accessed time
        self.metadata[conversation_id]["last_accessed"] = datetime.utcnow().isoformat()

        if include_metadata:
            return list(self.conversations[conversation_id])
        else:
            # Return only role and content for LLM
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.conversations[conversation_id]
            ]

    async def get_recent_context(
        self,
        conversation_id: str,
        num_messages: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get recent messages from conversation (sliding window).

        Args:
            conversation_id: Conversation ID
            num_messages: Number of recent messages (default: context_window_size)

        Returns:
            Recent messages
        """
        if num_messages is None:
            num_messages = self.context_window_size

        if conversation_id not in self.conversations:
            return []

        all_messages = list(self.conversations[conversation_id])

        # Get last N messages
        recent_messages = all_messages[-num_messages:] if num_messages > 0 else all_messages

        logger.debug(
            f"Conversation Memory: Retrieved {len(recent_messages)} recent messages "
            f"from conversation {conversation_id}"
        )

        # Return without metadata for LLM
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent_messages
        ]

    async def clear_conversation(self, conversation_id: str):
        """
        Clear a conversation from working memory.

        Args:
            conversation_id: Conversation ID
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            del self.metadata[conversation_id]
            logger.info(f"Conversation Memory: Cleared conversation {conversation_id}")

    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Summary dictionary
        """
        if conversation_id not in self.conversations:
            return {
                "exists": False,
                "conversation_id": conversation_id
            }

        messages = self.conversations[conversation_id]
        meta = self.metadata[conversation_id]

        return {
            "exists": True,
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "created_at": meta.get("created_at"),
            "last_accessed": meta.get("last_accessed"),
            "buffer_size": len(messages),
            "max_buffer_size": self.max_conversation_history
        }

    async def consolidate_to_summary(
        self,
        conversation_id: str,
        llm_service=None
    ) -> Optional[str]:
        """
        Consolidate conversation into a summary (memory consolidation).

        This simulates the process of converting working memory to long-term memory.

        Args:
            conversation_id: Conversation ID
            llm_service: Optional LLM service for generating summary

        Returns:
            Summary text or None
        """
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation Memory: Cannot consolidate, conversation {conversation_id} not found")
            return None

        messages = await self.get_conversation_history(conversation_id)

        if not messages:
            return None

        logger.info(f"Conversation Memory: Consolidating conversation {conversation_id} to summary")

        if llm_service:
            # Generate summary using LLM
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in messages
            ])

            summary_prompt = f"""Provide a concise summary of the following conversation:

{conversation_text}

Summary:"""

            try:
                summary = await llm_service.generate(
                    prompt=summary_prompt,
                    temperature=0.3,
                    max_tokens=200
                )
                return summary
            except Exception as e:
                logger.error(f"Conversation Memory: Error generating summary: {str(e)}")
                return None
        else:
            # Simple text-based summary
            summary = f"Conversation with {len(messages)} messages"
            return summary

    async def get_active_conversations(self) -> List[str]:
        """
        Get list of active conversation IDs.

        Returns:
            List of conversation IDs
        """
        return list(self.conversations.keys())

    async def prune_old_conversations(self, max_age_hours: int = 24):
        """
        Remove conversations older than specified age (memory decay).

        Args:
            max_age_hours: Maximum age in hours
        """
        current_time = datetime.utcnow()
        conversations_to_remove = []

        for conv_id, meta in self.metadata.items():
            last_accessed = datetime.fromisoformat(meta["last_accessed"])
            age_hours = (current_time - last_accessed).total_seconds() / 3600

            if age_hours > max_age_hours:
                conversations_to_remove.append(conv_id)

        for conv_id in conversations_to_remove:
            await self.clear_conversation(conv_id)
            logger.info(f"Conversation Memory: Pruned old conversation {conv_id}")

        if conversations_to_remove:
            logger.info(f"Conversation Memory: Pruned {len(conversations_to_remove)} old conversations")

    def get_buffer_usage(self) -> Dict[str, Any]:
        """
        Get working memory buffer usage statistics.

        Returns:
            Usage statistics
        """
        total_messages = sum(len(conv) for conv in self.conversations.values())

        return {
            "active_conversations": len(self.conversations),
            "total_messages_buffered": total_messages,
            "max_messages_per_conversation": self.max_conversation_history,
            "context_window_size": self.context_window_size
        }

    async def export_conversation(self, conversation_id: str) -> Optional[WorkingMemoryContext]:
        """
        Export conversation as WorkingMemoryContext object.

        Args:
            conversation_id: Conversation ID

        Returns:
            WorkingMemoryContext or None
        """
        if conversation_id not in self.conversations:
            return None

        messages = await self.get_conversation_history(conversation_id, include_metadata=True)
        meta = self.metadata[conversation_id]

        return WorkingMemoryContext(
            conversation_id=conversation_id,
            messages=messages,
            timestamp=datetime.fromisoformat(meta["last_accessed"]),
            metadata=meta
        )
