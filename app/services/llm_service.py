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


class ModelManager:
    """Manages LLM model selection and configuration."""

    def __init__(self, settings):
        self.settings = settings
        self.model_profiles = settings.MODEL_PROFILES

    def get_model_for_profile(self, profile: str = None) -> str:
        """
        Get model name for a given profile.

        Args:
            profile: Model profile name (fast/balanced/quality/reasoning)

        Returns:
            Model name string
        """
        # Check for custom model override
        if self.settings.CUSTOM_MODEL_NAME:
            logger.info(f"Using custom model override: {self.settings.CUSTOM_MODEL_NAME}")
            return self.settings.CUSTOM_MODEL_NAME

        # Use specified profile or default
        profile_to_use = profile or self.settings.MODEL_PROFILE

        if profile_to_use not in self.model_profiles:
            logger.warning(
                f"Unknown profile '{profile_to_use}', falling back to 'balanced'"
            )
            profile_to_use = "balanced"

        model_config = self.model_profiles[profile_to_use]
        model_name = model_config["model"]

        logger.info(f"Selected model '{model_name}' for profile '{profile_to_use}'")
        return model_name

    def get_model_config(self, profile: str = None) -> Dict[str, Any]:
        """
        Get full configuration for a model profile.

        Args:
            profile: Model profile name

        Returns:
            Dictionary with model configuration
        """
        profile_to_use = profile or self.settings.MODEL_PROFILE

        if profile_to_use not in self.model_profiles:
            profile_to_use = "balanced"

        return self.model_profiles[profile_to_use].copy()

    async def validate_model_availability(self, client: AsyncClient, model_name: str) -> bool:
        """
        Validate if a model is available in Ollama.

        Args:
            client: Ollama async client
            model_name: Model name to check

        Returns:
            True if model is available
        """
        try:
            models_response = await client.list()
            available_models = [model.get('name', '') for model in models_response.get('models', [])]

            is_available = any(model_name in model for model in available_models)

            if is_available:
                logger.info(f"Model '{model_name}' is available in Ollama")
            else:
                logger.warning(
                    f"Model '{model_name}' not found. Available models: {available_models}. "
                    f"Download with: ollama pull {model_name}"
                )

            return is_available

        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False

    def list_available_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all available model profiles with their configurations."""
        return self.model_profiles.copy()


class LLMService:
    """Service for interacting with Ollama LLM with dynamic model selection."""

    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[AsyncClient] = None
        self.model_manager = ModelManager(self.settings)

        # Default model (can be changed with set_model_profile)
        self.current_profile = None
        self.model_name = self.model_manager.get_model_for_profile()
        self.current_config = self.model_manager.get_model_config()

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

    def set_model_profile(self, profile: str = None) -> str:
        """
        Set model based on profile and return the model name.

        Args:
            profile: Model profile (fast/balanced/quality/reasoning)

        Returns:
            Selected model name
        """
        self.current_profile = profile
        self.model_name = self.model_manager.get_model_for_profile(profile)
        self.current_config = self.model_manager.get_model_config(profile)

        logger.info(
            f"Model switched to '{self.model_name}' "
            f"(profile: {profile or 'default'}, "
            f"max_tokens: {self.current_config['max_tokens']}, "
            f"temperature: {self.current_config['temperature']})"
        )

        return self.model_name

    def get_current_model(self) -> str:
        """Get currently selected model name."""
        return self.model_name

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
        temperature: Optional[float] = None,
        model_profile: Optional[str] = None,
    ) -> str:
        """
        Generate RAG response using retrieved contexts with dynamic model selection.

        Args:
            query: User query
            retrieved_contexts: List of retrieved document texts
            conversation_history: Optional conversation history
            temperature: Sampling temperature (overrides profile default if specified)
            model_profile: Model profile to use (fast/balanced/quality/reasoning)

        Returns:
            Generated response
        """
        await self.initialize()

        # Set model profile if specified
        if model_profile:
            self.set_model_profile(model_profile)

        # Use model config values or provided overrides
        final_temperature = temperature if temperature is not None else self.current_config.get('temperature', 0.7)
        max_tokens = self.current_config.get('max_tokens', 300)

        logger.info(
            f"Generating RAG response with model '{self.model_name}' "
            f"(profile: {model_profile or 'default'}, temp: {final_temperature}, "
            f"max_tokens: {max_tokens}, contexts: {len(retrieved_contexts)})"
        )

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
                    temperature=final_temperature,
                    max_tokens=max_tokens
                )
            else:
                # Use simple generation
                response = await self.generate(
                    prompt=rag_prompt,
                    system_prompt=system_prompt,
                    temperature=final_temperature,
                    max_tokens=max_tokens
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
