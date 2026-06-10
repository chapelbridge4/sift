"""
LLM service for interaction with MLX-VLM (Apple Silicon native LLM + vision).
Provides async text generation, chat, and vision capabilities using mlx-vlm.

Single code path for all requests — mlx-vlm handles both text-only and vision.
"""

import asyncio
import os
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from loguru import logger

from app.config import get_settings
import app.tuning.quality as _quality


class ModelManager:
    """Manages LLM model selection and configuration."""

    def __init__(self, settings):
        self.settings = settings
        self.model_profiles = settings.MODEL_PROFILES

    def get_model_for_profile(self, profile: str = None) -> str:
        """
        Get model name for a given profile.

        Args:
            profile: Model profile name (fast/balanced/quality)

        Returns:
            Model name string (mlx-community model ID)
        """
        if self.settings.CUSTOM_MODEL_NAME:
            logger.info(f"Using custom model override: {self.settings.CUSTOM_MODEL_NAME}")
            return self.settings.CUSTOM_MODEL_NAME

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

    def list_available_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all available model profiles with their configurations."""
        return self.model_profiles.copy()


class LLMService:
    """Service for interacting with MLX-VLM LLM with dynamic model selection."""

    def __init__(self):
        self.settings = get_settings()
        self.model_manager = ModelManager(self.settings)

        self._model_cache: Dict[str, tuple[Any, Any]] = {}
        self._current_model_id: Optional[str] = None
        self._current_model: Optional[tuple[Any, Any]] = None

        self.current_profile = None
        self.model_name = self.model_manager.get_model_for_profile()
        self.current_config = self.model_manager.get_model_config()

    def _get_model_sync(self, model_id: str) -> tuple[Any, Any]:
        """
        Load or return cached model. Synchronous — runs in thread pool.

        Args:
            model_id: mlx-community model identifier

        Returns:
            Tuple of (model, tokenizer)
        """
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        import mlx_vlm
        from mlx_vlm import load as vlm_load
        import mlx.core as mx
        import psutil

        # Check swap pressure before loading
        swap = psutil.swap_memory()
        swap_mb = swap.used / (1024 * 1024)
        if swap_mb > 100:
            logger.warning(
                f"Swap pressure detected: {swap_mb:.1f}MB used. "
                f"Loading model '{model_id}' may cause memory pressure."
            )

        logger.info(f"Loading MLX-VLM model: {model_id}")
        try:
            loaded = vlm_load(model_id)
        except MemoryError as e:
            raise MemoryError(
                f"Failed to load model '{model_id}' — not enough memory on M1 8GB. "
                f"Swap used: {swap_mb:.1f}MB. Try the 'fast' profile or restart the shell."
            ) from e

        self._model_cache[model_id] = loaded
        return loaded

    async def initialize(self):
        """Initialize with default model pre-loaded."""
        logger.info(f"Initializing MLX-VLM service with default model: {self.model_name}")

        model_id = self.model_manager.get_model_for_profile()
        try:
            self._current_model_id = model_id
            self._current_model = await asyncio.to_thread(
                self._get_model_sync, model_id
            )
            logger.info(f"MLX-VLM model '{model_id}' loaded successfully")

            # Warmup: generate a single token to finalize Metal kernel compilation
            if self.settings.MLX_WARMUP_ON_LOAD:
                import mlx_vlm
                model, tokenizer = self._current_model
                logger.info("Running MLX warmup (1-token generation)...")
                await asyncio.to_thread(
                    mlx_vlm.generate,
                    model, tokenizer,
                    prompt=" ",
                    max_tokens=1,
                    temperature=0.1,
                )
                logger.info("MLX warmup complete")
        except Exception as e:
            logger.error(f"Failed to load MLX-VLM model '{model_id}': {str(e)}")
            raise

    async def _ensure_model(self, model_id: str):
        """Ensure the specified model is loaded."""
        if self._current_model_id != model_id or self._current_model is None:
            self._current_model_id = model_id
            self._current_model = await asyncio.to_thread(
                self._get_model_sync, model_id
            )

    def set_model_profile(self, profile: str = None) -> str:
        """
        Set model based on profile and return the model name.

        Args:
            profile: Model profile (fast/balanced/quality)

        Returns:
            Selected model name
        """
        import mlx.core as mx

        # Clear Metal cache between profile switches to free memory
        if self.settings.MLX_METAL_CLEAR_CACHE_BETWEEN_PROFILES:
            mx.metal.clear_cache()

        self.current_profile = profile
        self.model_name = self.model_manager.get_model_for_profile(profile)
        self.current_config = self.model_manager.get_model_config(profile)

        logger.info(
            f"Model switched to '{self.model_name}' "
            f"(profile: {profile or 'default'}, "
            f"max_tokens: {self.current_config.get('max_tokens')}, "
            f"temperature: {self.current_config.get('temperature')}, "
            f"thinking: {self.current_config.get('thinking', False)})"
        )

        return self.model_name

    def get_model_for_request(
        self, model_profile: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Get model name and config for a specific request without modifying persistent state.

        Args:
            model_profile: Model profile to use for this request

        Returns:
            Tuple of (model_name, config_dict)
        """
        if model_profile:
            model_name = self.model_manager.get_model_for_profile(model_profile)
            config = self.model_manager.get_model_config(model_profile)
        else:
            model_name = self.model_name
            config = self.current_config

        return model_name, config

    def get_current_model(self) -> str:
        """Get currently selected model name."""
        return self.model_name

    @staticmethod
    def _sanitize_output(text: str) -> str:
        import re
        text = re.sub(r"Thinking Process:.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
        text = text.replace("(思考中)", "").replace("(思考完毕)", "").strip()
        text = re.sub(r"<\|[^|]+\|>", "", text)
        return text.strip()

    def _sanitize_qwen_output(self, text: str) -> str:
        """
        Strip any leaked thinking blocks from Qwen model output.
        """
        import re
        # Strip (思考中)...(思考完毕) blocks (Chinese)
        text = re.sub(r"\(思考中\).*?\(思考完毕\)\s*", "", text, flags=re.DOTALL)
        # Strip <think>...</think> blocks (English XML-style)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Strip numbered thinking steps (1. **Analyze..., 2. **Analyze..., etc.)
        text = re.sub(r"\n?\d+\.\s+\*\*[A-Za-z]+.*?(?=\n\d+\.|\n[A-Z]|\Z)", "", text, flags=re.DOTALL)
        # Strip "Thinking Process:" section
        text = re.sub(r"Thinking Process:.*?(?=\n\n|\n[A-Z]|$)", "", text, flags=re.DOTALL)
        # Also strip any remaining thinking tokens
        text = text.replace("(思考中)", "").replace("(思考完毕)", "").strip()
        text = text.replace("<think>", "").replace("</think>", "").strip()
        return text

    def _build_generate_args(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
        repetition_penalty: float = 1.15,
    ) -> dict:
        """
        Build arguments dict for mlx_vlm.generate.

        All values come from config — no hardcoding.
        """
        args = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
        }

        if enable_thinking:
            args["enable_thinking"] = True

        return args

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        enable_thinking: bool = False,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate text completion from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (prepended to prompt)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (from config if not specified)
            model_name: Override model name for this request
            enable_thinking: Toggle thinking mode for this request
            repetition_penalty: Override config default if specified

        Returns:
            Generated text
        """
        await self.initialize()

        model_id = model_name or self.model_name
        await self._ensure_model(model_id)

        logger.debug(f"Generating completion for prompt_length={len(prompt)}")

        try:
            import mlx_vlm

            model, tokenizer = self._current_model

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            if max_tokens is None:
                max_tokens = self.current_config.get("max_tokens", 500)

            final_repetition_penalty = (
                repetition_penalty if repetition_penalty is not None
                else self.current_config.get("repetition_penalty", 1.15)
            )

            generate_args = self._build_generate_args(
                full_prompt, temperature, max_tokens, enable_thinking,
                final_repetition_penalty
            )

            result = await asyncio.to_thread(
                mlx_vlm.generate,
                model,
                tokenizer,
                **generate_args
            )

            # mlx_vlm.generate returns GenerationResult object
            text = result.text if hasattr(result, 'text') else str(result)
            text = self._sanitize_qwen_output(text)
            text = LLMService._sanitize_output(text)
            logger.debug(f"Generated {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

    async def generate_with_image(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        enable_thinking: bool = False,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate text with optional image input (vision-enabled).

        When no image is provided, falls back to text-only generation.

        Args:
            prompt: User prompt
            image_path: Local path to image file (optional)
            image_url: URL to image (optional)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model_name: Override model name for this request
            enable_thinking: Toggle thinking mode for this request
            repetition_penalty: Override config default if specified

        Returns:
            Generated text
        """
        await self.initialize()

        model_id = model_name or self.model_name
        await self._ensure_model(model_id)

        if image_path is None and image_url is None:
            return await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model_name=model_id,
                enable_thinking=enable_thinking,
                repetition_penalty=repetition_penalty,
            )

        logger.debug(f"Generating vision completion for prompt_length={len(prompt)}")

        try:
            import mlx_vlm

            model, tokenizer = self._current_model

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            if max_tokens is None:
                max_tokens = self.current_config.get("max_tokens", 500)

            final_repetition_penalty = (
                repetition_penalty if repetition_penalty is not None
                else self.current_config.get("repetition_penalty", 1.15)
            )

            if image_path and os.path.exists(image_path):
                generate_args = self._build_generate_args(
                    full_prompt, temperature, max_tokens, enable_thinking,
                    final_repetition_penalty
                )
                generate_args["image"] = image_path

                result = await asyncio.to_thread(
                    mlx_vlm.generate,
                    model,
                    tokenizer,
                    **generate_args
                )
            elif image_url:
                generate_args = self._build_generate_args(
                    full_prompt, temperature, max_tokens, enable_thinking,
                    final_repetition_penalty
                )
                generate_args["image"] = image_url

                result = await asyncio.to_thread(
                    mlx_vlm.generate,
                    model,
                    tokenizer,
                    **generate_args
                )

            text = result.text if hasattr(result, 'text') else str(result)
            text = LLMService._sanitize_output(text)
            valid, reason = _quality.is_valid_response(text)
            if not valid:
                raise ValueError(f"Invalid model output: {reason}")
            logger.debug(f"Generated vision response: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error generating vision completion: {str(e)}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        enable_thinking: bool = False,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate chat completion from message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model_name: Override model name for this request
            enable_thinking: Toggle thinking mode for this request
            repetition_penalty: Override config default if specified

        Returns:
            Generated response text
        """
        await self.initialize()

        model_id = model_name or self.model_name
        await self._ensure_model(model_id)

        logger.debug(f"Generating chat completion with {len(messages)} messages")

        try:
            import mlx_vlm

            model, tokenizer = self._current_model

            if max_tokens is None:
                max_tokens = self.current_config.get("max_tokens", 500)

            final_repetition_penalty = (
                repetition_penalty if repetition_penalty is not None
                else self.current_config.get("repetition_penalty", 1.15)
            )

            # Apply chat template (Qwen3.5 defaults to thinking OFF, no explicit flag needed)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            generate_args = self._build_generate_args(
                prompt, temperature, max_tokens, enable_thinking,
                final_repetition_penalty
            )

            result = await asyncio.to_thread(
                mlx_vlm.generate,
                model,
                tokenizer,
                **generate_args
            )

            text = result.text if hasattr(result, 'text') else str(result)
            text = self._sanitize_qwen_output(text)
            text = LLMService._sanitize_output(text)
            valid, reason = _quality.is_valid_response(text)
            if not valid:
                raise ValueError(f"Invalid model output: {reason}")
            logger.debug(f"Generated chat response: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise

    async def generate_rag_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_profile: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """
        Generate RAG response using retrieved contexts with dynamic model selection.

        Args:
            query: User query
            retrieved_contexts: List of retrieved document texts
            conversation_history: Optional conversation history
            temperature: Sampling temperature (overrides profile default if specified)
            max_tokens: Per-request token cap (overrides profile default if specified)
            model_profile: Model profile to use (fast/balanced/quality)
            enable_thinking: Override thinking mode for this request

        Returns:
            Generated response
        """
        await self.initialize()

        model_name, model_config = self.get_model_for_request(model_profile)
        await self._ensure_model(model_name)

        final_temperature = (
            temperature if temperature is not None
            else model_config.get('temperature', 0.7)
        )
        # Cap from config profile, overridable per-request
        profile_max_tokens = model_config.get('max_tokens', 200)
        final_max_tokens = max_tokens if max_tokens is not None else profile_max_tokens

        thinking_for_request = (
            enable_thinking if enable_thinking is not None
            else model_config.get('thinking', False)
        )

        logger.info(
            f"Generating RAG response with model '{model_name}' "
            f"(profile: {model_profile or 'default'}, temp: {final_temperature}, "
            f"max_tokens: {final_max_tokens}, thinking: {thinking_for_request}, "
            f"contexts: {len(retrieved_contexts)})"
        )

        context_str = "\n\n---\n\n".join([
            f"Context {i+1}:\n{ctx}"
            for i, ctx in enumerate(retrieved_contexts)
        ])

        system_prompt = (
            "You are a helpful AI assistant that answers questions based on the "
            "provided context. Provide concise, accurate answers (2-4 sentences "
            "maximum unless more detail is explicitly requested). If the context "
            "doesn't contain enough information, acknowledge this briefly. Always "
            "ground your responses in the provided context."
        )

        # Prepend /no_think for Qwen models to suppress thinking output
        if "qwen" in model_name.lower():
            system_prompt = "/no_think " + system_prompt

        rag_prompt = f"""Context Information:
{context_str}

Question: {query}

Please provide a concise answer (2-4 sentences) based on the context above."""

        try:
            if conversation_history:
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
                    max_tokens=final_max_tokens,
                    model_name=model_name,
                    enable_thinking=thinking_for_request
                )
            else:
                response = await self.generate(
                    prompt=rag_prompt,
                    system_prompt=system_prompt,
                    temperature=final_temperature,
                    max_tokens=final_max_tokens,
                    model_name=model_name,
                    enable_thinking=thinking_for_request
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
        enable_thinking: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation (for future streaming endpoints).

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            enable_thinking: Toggle thinking mode

        Yields:
            Generated text chunks
        """
        await self.initialize()

        logger.debug(f"Starting streaming generation for prompt_length={len(prompt)}")

        try:
            import mlx_vlm

            model, tokenizer = self._current_model

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            generate_args = self._build_generate_args(
                full_prompt,
                temperature,
                self.current_config.get("max_tokens", 500),
                enable_thinking
            )

            stream = mlx_vlm.stream_generate(
                model, tokenizer, **generate_args
            )

            for chunk in stream:
                if hasattr(chunk, 'text'):
                    yield chunk.text
                elif isinstance(chunk, str):
                    yield chunk

        except Exception as e:
            logger.error(f"Error during streaming generation: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """Check if MLX-VLM models are loadable and generation works."""
        try:
            import mlx_vlm

            await self.initialize()

            model, tokenizer = self._current_model

            def do_generate():
                return mlx_vlm.generate(
                    model, tokenizer,
                    prompt="hello",
                    max_tokens=5,
                    temperature=0.1,
                )

            result = await asyncio.to_thread(do_generate)
            return result is not None and len(result) > 0

        except Exception as e:
            logger.error(f"MLX-VLM health check failed: {str(e)}")
            return False

    async def close(self):
        """Cleanup resources — clear model cache."""
        logger.info("Closing MLX-VLM service")
        self._model_cache.clear()
        self._current_model = None
        self._current_model_id = None
