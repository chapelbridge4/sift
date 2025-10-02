"""
Prefrontal Cortex module - Executive function and reasoning.

Inspired by the prefrontal cortex brain region responsible for:
- Executive function and decision making
- Integration of information from multiple sources
- Complex reasoning and planning
- Goal-directed behavior
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from app.services.llm_service import LLMService
from app.brain.hippocampus import Hippocampus
from app.brain.amygdala import Amygdala
from app.brain.working_memory import WorkingMemory


class PrefrontalCortex:
    """
    Prefrontal Cortex module for executive control and reasoning.

    This module handles:
    - Orchestration of RAG pipeline
    - Integration of retrieved information
    - Complex reasoning with LLM
    - Decision making about what information to use
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.hippocampus = Hippocampus()
        self.amygdala = Amygdala()
        self.working_memory = WorkingMemory()
        logger.info("Prefrontal Cortex module initialized")

    async def initialize(self):
        """Initialize the prefrontal cortex and its subsystems."""
        await self.llm_service.initialize()
        await self.hippocampus.initialize()
        logger.info("Prefrontal Cortex executive systems online")

    async def reason_with_context(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        conversation_id: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Perform complex reasoning using retrieved context (executive function).

        This orchestrates the full RAG pipeline:
        1. Retrieve relevant memories (Hippocampus)
        2. Rank by importance (Amygdala)
        3. Integrate with working memory context
        4. Generate reasoned response (PFC)

        Args:
            query: User query
            collection_name: Memory collection to use
            top_k: Number of memories to retrieve
            use_hybrid: Use hybrid search
            conversation_id: Optional conversation ID for context
            temperature: LLM sampling temperature

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"PFC: Beginning reasoning process for query: {query[:50]}...")

        try:
            # Step 1: Retrieve relevant memories (Hippocampus recall)
            logger.debug("PFC: Step 1 - Retrieving memories from Hippocampus")
            retrieved_memories = await self.hippocampus.recall_memories(
                collection_name=collection_name,
                query=query,
                top_k=top_k,
                use_hybrid=use_hybrid
            )

            if not retrieved_memories:
                logger.warning("PFC: No memories retrieved, generating response without context")
                return await self._generate_response_without_context(
                    query, conversation_id, temperature
                )

            # Step 2: Evaluate importance and rank (Amygdala processing)
            logger.debug("PFC: Step 2 - Evaluating importance with Amygdala")
            ranked_memories = self.amygdala.rank_by_importance(
                retrieved_memories,
                query
            )

            # Step 3: Get conversation context (Working Memory)
            logger.debug("PFC: Step 3 - Retrieving conversation context from Working Memory")
            conversation_history = None
            if conversation_id:
                conversation_history = await self.working_memory.get_conversation_history(
                    conversation_id
                )

            # Step 4: Integrate information and reason (Executive function)
            logger.debug("PFC: Step 4 - Integrating information and reasoning")
            answer = await self._integrate_and_reason(
                query=query,
                ranked_memories=ranked_memories,
                conversation_history=conversation_history,
                temperature=temperature
            )

            # Step 5: Update working memory
            if conversation_id:
                await self.working_memory.add_message(
                    conversation_id,
                    {"role": "user", "content": query}
                )
                await self.working_memory.add_message(
                    conversation_id,
                    {"role": "assistant", "content": answer}
                )

            logger.info("PFC: Reasoning complete, response generated")

            return {
                "answer": answer,
                "retrieved_documents": ranked_memories,
                "num_documents_used": len(ranked_memories),
                "conversation_id": conversation_id
            }

        except Exception as e:
            logger.error(f"PFC: Error during reasoning: {str(e)}")
            raise

    async def _integrate_and_reason(
        self,
        query: str,
        ranked_memories: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        temperature: float
    ) -> str:
        """
        Integrate retrieved information and generate reasoned response.

        Args:
            query: User query
            ranked_memories: Ranked documents from Amygdala
            conversation_history: Conversation history from Working Memory
            temperature: LLM temperature

        Returns:
            Generated answer
        """
        # Extract text contexts from top-ranked memories
        contexts = [
            memory['text']
            for memory in ranked_memories[:5]  # Use top 5 for faster response
        ]

        # Generate response using LLM
        answer = await self.llm_service.generate_rag_response(
            query=query,
            retrieved_contexts=contexts,
            conversation_history=conversation_history,
            temperature=temperature
        )

        return answer

    async def _generate_response_without_context(
        self,
        query: str,
        conversation_id: Optional[str],
        temperature: float
    ) -> Dict[str, Any]:
        """
        Generate response when no relevant context is found.

        Args:
            query: User query
            conversation_id: Conversation ID
            temperature: LLM temperature

        Returns:
            Response dictionary
        """
        logger.info("PFC: Generating response without retrieved context")

        conversation_history = None
        if conversation_id:
            conversation_history = await self.working_memory.get_conversation_history(
                conversation_id
            )

        # Generate response using LLM without context
        if conversation_history:
            answer = await self.llm_service.chat(
                messages=conversation_history + [
                    {"role": "user", "content": query}
                ],
                temperature=temperature
            )
        else:
            answer = await self.llm_service.generate(
                prompt=query,
                temperature=temperature
            )

        return {
            "answer": answer,
            "retrieved_documents": [],
            "num_documents_used": 0,
            "conversation_id": conversation_id,
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
        logger.info(f"PFC: Making decision between {len(options)} options")

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
            logger.error(f"PFC: Error making decision: {str(e)}")
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
        logger.info(f"PFC: Planning task: {task_description[:50]}...")

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
            logger.error(f"PFC: Error planning task: {str(e)}")
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
        logger.info("PFC: Evaluating response quality (meta-cognition)")

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
            logger.error(f"PFC: Error evaluating response: {str(e)}")
            raise
