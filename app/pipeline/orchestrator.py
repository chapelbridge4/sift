"""
RagOrchestrator module - RAG pipeline orchestration.

Responsible for:
- Orchestrating the full retrieval-augmented generation pipeline
- Integrating information from multiple pipeline stages
- Driving reasoning and response generation with the LLM
- Deciding what retrieved context to use
"""

from typing import Any, Dict, List, Optional, cast

from loguru import logger

from app.pipeline.conversation_memory import ConversationMemory
from app.pipeline.document_store import DocumentStore
from app.pipeline.reranker import Reranker
from app.services.inference import RagBackend, get_inference_backend

# Methods a backend must provide to drive the profile-based RAG path. Mirrors the
# RagBackend Protocol (the static contract) but is checked explicitly at runtime,
# because isinstance() against a runtime_checkable Protocol rejects test doubles
# like MagicMock — this check accepts real backends and injected mocks alike, and
# still rejects GGUFService (which lacks chat / get_model_for_request).
RAG_BACKEND_METHODS = ("generate_rag_response", "generate", "chat", "get_model_for_request")


class RagOrchestrator:
    """
    RagOrchestrator module for executive control and reasoning.

    This module handles:
    - Orchestration of RAG pipeline
    - Integration of retrieved information
    - Complex reasoning with LLM
    - Decision making about what information to use
    """

    def __init__(
        self,
        llm_service: Optional[RagBackend] = None,
        document_store: Optional[DocumentStore] = None,
        reranker: Optional[Reranker] = None,
        conversation_memory: Optional[ConversationMemory] = None,
    ):
        # Dependencies are injected for testability and resolved from config by
        # default — never hardcoded. The inference backend is selected by
        # INFERENCE_BACKEND (default 'mlx'); MLX is the supported RAG backend.
        backend = llm_service or get_inference_backend()
        # Fail fast if the backend can't drive the profile-based RAG path.
        missing = [m for m in RAG_BACKEND_METHODS if not callable(getattr(backend, m, None))]
        if missing:
            raise TypeError(
                f"Inference backend {type(backend).__name__} cannot drive the RAG "
                f"orchestrator: missing {missing}. The profile-based RAG path requires "
                f"the MLX backend (INFERENCE_BACKEND=mlx); GGUF currently supports "
                f"direct generate/generate_rag_response only."
            )
        # Runtime contract verified above, so narrow the static type for callers.
        self.llm_service = cast(RagBackend, backend)
        self.document_store = document_store or DocumentStore()
        self.reranker = reranker or Reranker()
        self.conversation_memory = conversation_memory or ConversationMemory()
        logger.info("RagOrchestrator module initialized")

    async def initialize(self):
        """Initialize the orchestrator and its pipeline stages."""
        await self.document_store.initialize()
        logger.info("RagOrchestrator executive systems online")

    async def reason_with_context(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        conversation_id: Optional[str] = None,
        temperature: Optional[float] = None,
        model_profile: Optional[str] = None,
        fusion_method: str = "rrf",
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Perform complex reasoning using retrieved context (executive function).

        This orchestrates the full RAG pipeline:
        1. Retrieve relevant documents (DocumentStore)
        2. Rank by importance (Reranker)
        3. Integrate with conversation memory context
        4. Generate reasoned response (LLM)

        Args:
            query: User query
            collection_name: Memory collection to use
            top_k: Number of memories to retrieve
            use_hybrid: Use hybrid search
            conversation_id: Optional conversation ID for context
            temperature: LLM sampling temperature (optional, uses model profile default if not set)
            model_profile: Model profile to use (fast/balanced/quality/reasoning)

        Returns:
            Dictionary with answer and metadata including model_used
        """
        logger.info(f"Orchestrator: Beginning reasoning process for query_length={len(query)}")

        try:
            # Step 1: Retrieve relevant documents (DocumentStore recall)
            logger.debug("Orchestrator: Step 1 - Retrieving documents from DocumentStore")
            retrieved_memories = await self.document_store.recall_memories(
                collection_name=collection_name,
                query=query,
                top_k=top_k,
                use_hybrid=use_hybrid,
                fusion_method=fusion_method
            )

            # Step 2: Evaluate importance and rank (Reranker processing)
            if retrieved_memories:
                logger.debug("Orchestrator: Step 2 - Evaluating importance with Reranker")
                ranked_memories = self.reranker.rank_by_importance(
                    retrieved_memories,
                    query
                )
            else:
                ranked_memories = []

            # Step 3: Check if LLM generation is needed
            if not use_llm:
                logger.info("Orchestrator: use_llm=False, returning retrieved documents without generation")
                return {
                    "answer": None,
                    "retrieved_documents": ranked_memories,
                    "num_documents_used": len(ranked_memories),
                    "conversation_id": conversation_id,
                    "model_used": None
                }

            # Step 4: Get conversation context (Conversation Memory)
            if not ranked_memories:
                logger.warning("Orchestrator: No memories retrieved, generating response without context")
                return await self._generate_response_without_context(
                    query, conversation_id, temperature, model_profile
                )

            logger.debug("Orchestrator: Step 4 - Retrieving conversation context from Conversation Memory")
            conversation_history = None
            if conversation_id:
                conversation_history = await self.conversation_memory.get_conversation_history(
                    conversation_id
                )

            # Step 5: Integrate information and reason (Executive function)
            logger.debug("Orchestrator: Step 5 - Integrating information and reasoning")
            answer, model_used = await self._integrate_and_reason(
                query=query,
                ranked_memories=ranked_memories,
                conversation_history=conversation_history,
                temperature=temperature,
                model_profile=model_profile
            )

            # Step 6: Update conversation memory
            if conversation_id:
                await self.conversation_memory.add_message(
                    conversation_id,
                    {"role": "user", "content": query}
                )
                await self.conversation_memory.add_message(
                    conversation_id,
                    {"role": "assistant", "content": answer}
                )

            logger.info(f"Orchestrator: Reasoning complete with model '{model_used}'")

            return {
                "answer": answer,
                "retrieved_documents": ranked_memories,
                "num_documents_used": len(ranked_memories),
                "conversation_id": conversation_id,
                "model_used": model_used
            }

        except Exception as e:
            logger.error(f"Orchestrator: Error during reasoning: {str(e)}")
            raise

    async def _integrate_and_reason(
        self,
        query: str,
        ranked_memories: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        temperature: Optional[float],
        model_profile: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Integrate retrieved information and generate reasoned response.

        Args:
            query: User query
            ranked_memories: Ranked documents from Reranker
            conversation_history: Conversation history from Conversation Memory
            temperature: LLM temperature
            model_profile: Model profile to use

        Returns:
            Tuple of (generated answer, model_name used)
        """
        # Extract text contexts from top-ranked memories
        contexts = [
            memory['text']
            for memory in ranked_memories[:5]  # Use top 5 for faster response
        ]

        # Get model for this request without modifying persistent state
        model_name, model_config = self.llm_service.get_model_for_request(model_profile)

        # Generate response using LLM with specified model profile
        answer = await self.llm_service.generate_rag_response(
            query=query,
            retrieved_contexts=contexts,
            conversation_history=conversation_history,
            temperature=temperature,
            model_profile=model_profile
        )

        return answer, model_name

    async def _generate_response_without_context(
        self,
        query: str,
        conversation_id: Optional[str],
        temperature: float,
        model_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response when no relevant context is found.

        Args:
            query: User query
            conversation_id: Conversation ID
            temperature: LLM temperature
            model_profile: Model profile to use

        Returns:
            Response dictionary
        """
        logger.info("Orchestrator: Generating response without retrieved context")

        conversation_history = None
        if conversation_id:
            conversation_history = await self.conversation_memory.get_conversation_history(
                conversation_id
            )

        # Get model for this request without modifying persistent state
        model_name, model_config = self.llm_service.get_model_for_request(model_profile)

        final_temperature = temperature if temperature is not None else model_config.get('temperature', 0.7)
        max_tokens = model_config.get('max_tokens', 300)

        # Generate response using LLM without context
        if conversation_history:
            answer = await self.llm_service.chat(
                messages=conversation_history + [
                    {"role": "user", "content": query}
                ],
                temperature=final_temperature,
                max_tokens=max_tokens,
                model_name=model_name
            )
        else:
            answer = await self.llm_service.generate(
                prompt=query,
                temperature=final_temperature,
                max_tokens=max_tokens,
                model_name=model_name
            )

        return {
            "answer": answer,
            "retrieved_documents": [],
            "num_documents_used": 0,
            "conversation_id": conversation_id,
            "model_used": model_name,
            "note": "No relevant context found in memory"
        }

    async def make_decision(
        self,
        options: List[str],
        criteria: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a decision between multiple options (executive decision making).

        Args:
            options: List of options to choose from
            criteria: Decision criteria
            context: Optional context

        Returns:
            Decision result with reasoning
        """
        logger.info(f"Orchestrator: Making decision between {len(options)} options")

        decision_prompt = f"""Given the following options and criteria, make a decision.

Criteria: {criteria}

Options:
{chr(10).join([f"{i+1}. {opt}" for i, opt in enumerate(options)])}

{f"Context: {context}" if context else ""}

Provide your decision with clear reasoning."""

        try:
            decision = await self.llm_service.generate(
                prompt=decision_prompt,
                temperature=0.3  # Lower temperature for more deterministic decisions
            )

            return {
                "decision": decision,
                "options": options,
                "criteria": criteria
            }

        except Exception as e:
            logger.error(f"Orchestrator: Error making decision: {str(e)}")
            raise

    async def plan_task(
        self,
        task_description: str,
        constraints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a plan for accomplishing a task (planning and organization).

        Args:
            task_description: Description of the task
            constraints: Optional list of constraints

        Returns:
            Task plan with steps
        """
        logger.info(f"Orchestrator: Planning task with description_length={len(task_description)}")

        constraints_str = ""
        if constraints:
            constraints_str = f"\nConstraints:\n{chr(10).join([f'- {c}' for c in constraints])}"

        planning_prompt = f"""Create a detailed plan to accomplish the following task:

Task: {task_description}
{constraints_str}

Provide a step-by-step plan with clear actions."""

        try:
            plan = await self.llm_service.generate(
                prompt=planning_prompt,
                temperature=0.5
            )

            return {
                "plan": plan,
                "task": task_description,
                "constraints": constraints or []
            }

        except Exception as e:
            logger.error(f"Orchestrator: Error planning task: {str(e)}")
            raise

    async def evaluate_response_quality(
        self,
        query: str,
        response: str,
        expected_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated response (meta-cognition).

        Args:
            query: Original query
            response: Generated response
            expected_criteria: Optional quality criteria

        Returns:
            Evaluation results
        """
        logger.info("Orchestrator: Evaluating response quality (meta-cognition)")

        criteria_str = ""
        if expected_criteria:
            criteria_str = f"Criteria: {', '.join(expected_criteria)}\n"

        evaluation_prompt = f"""Evaluate the quality of the following response to the query.

Query: {query}

Response: {response}

{criteria_str}
Provide an evaluation of relevance, accuracy, completeness, and clarity.
Rate each aspect on a scale of 1-10."""

        try:
            evaluation = await self.llm_service.generate(
                prompt=evaluation_prompt,
                temperature=0.3
            )

            return {
                "evaluation": evaluation,
                "query": query,
                "response": response
            }

        except Exception as e:
            logger.error(f"Orchestrator: Error evaluating response: {str(e)}")
            raise
