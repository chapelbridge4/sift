"""
LLM service for interaction with Ollama (llama3.2:3b).
Provides async text generation and chat capabilities.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio

import ollama
from ollama import AsyncClient
from loguru import logger

from app.config import get_settings
from app.utils.async_helpers import async_retry


class LLMService:
    """Service for interacting with Ollama LLM."""

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[AsyncClient] = None
        self.model_name = self.settings.OLLAMA_MODEL

    async def initialize(self):
        """Initialize Ollama client."""
        if self.client is None:
            logger.info(f"Initializing Ollama client for model: {self.model_name}")

            self.client = AsyncClient(
                host=self.settings.OLLAMA_HOST,
                timeout=self.settings.OLLAMA_TIMEOUT
            )

            # Verify model availability
            try:
                await self._check_model_availability()
                logger.info(f"Ollama client initialized successfully with {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {str(e)}")
                raise

    async def _check_model_availability(self):
        """Check if the specified model is available."""
        try:
            models_response = await self.client.list()
            available_models = [model['name'] for model in models_response.get('models', [])]

            if not any(self.model_name in model for model in available_models):
                logger.warning(
                    f"Model {self.model_name} not found in available models: {available_models}"
                )
                logger.info(f"Attempting to use {self.model_name} anyway...")

        except Exception as e:
            logger.warning(f"Could not list models: {str(e)}")

    @async_retry(max_retries=3, delay=1.0, backoff=2.0)
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text completion from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        await self.initialize()

        logger.debug(f"Generating completion for prompt: {prompt[:100]}...")

        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_prompt,
                options=options,
            )

            generated_text = response.get('response', '')
            logger.debug(f"Generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

    @async_retry(max_retries=3, delay=1.0, backoff=2.0)
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate chat completion from message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        await self.initialize()

        logger.debug(f"Generating chat completion with {len(messages)} messages")

        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            response = await self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
            )

            generated_text = response['message']['content']
            logger.debug(f"Generated chat response: {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise

    async def generate_rag_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate RAG response using retrieved contexts.

        Args:
            query: User query
            retrieved_contexts: List of retrieved document texts
            conversation_history: Optional conversation history
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        await self.initialize()

        logger.info(f"Generating RAG response for query with {len(retrieved_contexts)} contexts")

        # Build context string
        context_str = "\n\n---\n\n".join([
            f"Context {i+1}:\n{ctx}"
            for i, ctx in enumerate(retrieved_contexts)
        ])

        # Build RAG prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Provide concise, accurate answers (2-4 sentences maximum unless more detail is explicitly requested).
If the context doesn't contain enough information, acknowledge this briefly.
Always ground your responses in the provided context."""

        rag_prompt = f"""Context Information:
{context_str}

Question: {query}

Please provide a concise answer (2-4 sentences) based on the context above."""

        try:
            if conversation_history:
                # Use chat mode with conversation history
                messages = conversation_history.copy()
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
                messages.append({
                    "role": "user",
                    "content": rag_prompt
                })

                response = await self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=300  # Limit response length
                )
            else:
                # Use simple generation
                response = await self.generate(
                    prompt=rag_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=300  # Limit response length
                )

            logger.info("RAG response generated successfully")
            return response

        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            raise

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation (for future streaming endpoints).

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature

        Yields:
            Generated text chunks
        """
        await self.initialize()

        logger.debug(f"Starting streaming generation for prompt: {prompt[:100]}...")

        try:
            stream = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system_prompt,
                stream=True,
                options={"temperature": temperature},
            )

            async for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']

        except Exception as e:
            logger.error(f"Error during streaming generation: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """Check if Ollama is accessible and model is available."""
        try:
            await self.initialize()
            # Try a simple generation to verify functionality
            await self.generate(
                prompt="Hello",
                max_tokens=5,
                temperature=0.1
            )
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return False

    async def close(self):
        """Cleanup resources."""
        logger.info("Closing LLM service")
        self.client = None
