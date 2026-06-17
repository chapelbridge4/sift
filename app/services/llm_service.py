"""
LLM service for interaction with MLX-VLM (Apple Silicon native LLM + vision).
Provides async text generation, chat, and vision capabilities using mlx-vlm.

Single code path for all requests — mlx-vlm handles both text-only and vision.
"""

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger

import app.tuning.quality as _quality
from app.config import get_settings


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

        # MLX-Metal GPU streams are thread-affine: a model's GPU stream lives on
        # the thread that created it, and generating from another thread raises
        # "There is no Stream(gpu, 1) in current thread." We pin every MLX op
        # (load, warmup, generate, chat, stream) to ONE dedicated worker thread
        # so load and generation always share the same Metal stream.
        self._mlx_executor: Optional[ThreadPoolExecutor] = None

        self.current_profile = None
        self.model_name = self.model_manager.get_model_for_profile()
        self.current_config = self.model_manager.get_model_config()

    def _get_mlx_executor(self) -> ThreadPoolExecutor:
        """Return the single-worker executor that owns all MLX/Metal ops.

        Created lazily so constructing an LLMService stays cheap (no thread
        spawned until the first MLX call).
        """
        if self._mlx_executor is None:
            self._mlx_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mlx"
            )
        return self._mlx_executor

    async def _run_mlx(self, fn, *args, **kwargs):
        """Run a blocking MLX call on the dedicated single MLX thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_mlx_executor(), functools.partial(fn, *args, **kwargs)
        )

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

        import psutil
        from mlx_vlm import load as vlm_load

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
            self._current_model = await self._run_mlx(
                self._get_model_sync, model_id
            )
            logger.info(f"MLX-VLM model '{model_id}' loaded successfully")

            # Warmup: generate a single token to finalize Metal kernel compilation
            if self.settings.MLX_WARMUP_ON_LOAD:
                import mlx_vlm
                model, tokenizer = self._current_model
                logger.info("Running MLX warmup (1-token generation)...")
                await self._run_mlx(
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
            self._current_model = await self._run_mlx(
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
    def _passes_validity(text: str, enable_thinking: bool) -> tuple[bool, str]:
        """Decide whether a generated answer is acceptable.

        Non-thinking responses use the strict gate unchanged. Thinking responses
        may be short after the reasoning block is sanitized away, so we accept any
        NON-EMPTY thinking answer (reason "too_short_but_accepted") while still
        rejecting truly empty output or special-token garbage.
        """
        valid, reason = _quality.is_valid_response(text)
        if valid or not enable_thinking:
            return valid, reason
        # Thinking path: salvage short-but-non-empty answers.
        if not text.strip():
            return False, "too_short"
        if reason == "special_token_leak":
            return False, reason
        return True, "too_short_but_accepted"

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

    @staticmethod
    def _apply_chat_template(tokenizer, messages, enable_thinking: bool) -> str:
        """Render messages to a prompt string, controlling Qwen thinking mode.

        ``enable_thinking`` is passed to the tokenizer template (the only place
        that actually toggles reasoning for Qwen3.5). Tokenizers whose template
        predates this kwarg fall back to the default rendering.
        """
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

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

            result = await self._run_mlx(
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

                result = await self._run_mlx(
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

                result = await self._run_mlx(
                    mlx_vlm.generate,
                    model,
                    tokenizer,
                    **generate_args
                )
            else:
                raise ValueError(
                    "generate_with_image requires a readable image; got "
                    f"image_path={image_path!r}, image_url={image_url!r}"
                )

            text = result.text if hasattr(result, 'text') else str(result)
            text = self._sanitize_qwen_output(text)
            text = LLMService._sanitize_output(text)
            valid, reason = self._passes_validity(text, enable_thinking)
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

            # Thinking is controlled by the chat template, NOT by mlx_vlm.generate.
            # The Qwen3.5 template defaults thinking ON (it opens a <think> block
            # with no close); enable_thinking=False injects a closed, empty
            # <think></think> block that suppresses reasoning. Without this the
            # model emits a long reasoning dump that the sanitizer strips to an
            # empty string.
            prompt = self._apply_chat_template(tokenizer, messages, enable_thinking)

            generate_args = self._build_generate_args(
                prompt, temperature, max_tokens, enable_thinking,
                final_repetition_penalty
            )

            result = await self._run_mlx(
                mlx_vlm.generate,
                model,
                tokenizer,
                **generate_args
            )

            text = result.text if hasattr(result, 'text') else str(result)
            text = self._sanitize_qwen_output(text)
            text = LLMService._sanitize_output(text)
            valid, reason = self._passes_validity(text, enable_thinking)
            if not valid:
                raise ValueError(f"Invalid model output: {reason}")
            if enable_thinking and reason == "too_short_but_accepted":
                logger.warning(
                    "Thinking response was short after sanitizing the reasoning "
                    "block; accepting the non-empty answer "
                    f"({len(text.strip())} chars)."
                )
            logger.debug(f"Generated chat response: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise

    @staticmethod
    def _build_rag_system_prompt(model_name: str) -> str:
        """Build the RAG system prompt.

        Note: we deliberately do NOT prepend a "/no_think" control token for
        Qwen models. On this MLX build that token lands at the start of the raw
        user turn and produces malformed output (a leaked empty <think></think>
        block, or an empty string). Thinking is instead disabled the correct
        way — via the chat template's enable_thinking flag in chat() — so the
        prefix is both unnecessary and harmful. ``model_name`` is kept for
        forward compatibility with model-specific prompt tweaks.
        """
        return (
            "You are a helpful AI assistant that answers questions based on the "
            "provided context. Provide concise, accurate answers (2-4 sentences "
            "maximum unless more detail is explicitly requested). If the context "
            "doesn't contain enough information, acknowledge this briefly. Always "
            "ground your responses in the provided context."
        )

    async def generate_rag_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_profile: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        extra_system_instruction: Optional[str] = None,
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

        system_prompt = self._build_rag_system_prompt(model_name)
        if extra_system_instruction:
            system_prompt = f"{system_prompt}\n\n{extra_system_instruction}"

        rag_prompt = f"""Context Information:
{context_str}

Question: {query}

Please provide a concise answer (2-4 sentences) based on the context above."""

        # Route through chat() so the prompt is built with the proper chat
        # template (which is the only mechanism that toggles Qwen thinking).
        # The system message MUST come first — the Qwen template rejects a
        # system turn placed after conversation history.
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": rag_prompt})

        try:
            response = await self.chat(
                messages=messages,
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

        MLX-Metal GPU ops are thread-affine, so the stream is produced on the
        dedicated single MLX thread (the one that owns the model) and bridged to
        this async caller via a thread-safe queue. Iterating ``mlx_vlm`` directly
        on the event-loop thread would raise "There is no Stream(gpu, 1) in
        current thread" under a server (same root cause as the load/generate path).

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

        model, tokenizer = self._current_model

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generate_args = self._build_generate_args(
            full_prompt,
            temperature,
            self.current_config.get("max_tokens", 500),
            enable_thinking,
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        def _produce() -> None:
            # Runs on the single dedicated MLX thread.
            try:
                import mlx_vlm

                for chunk in mlx_vlm.stream_generate(model, tokenizer, **generate_args):
                    text = getattr(chunk, "text", chunk if isinstance(chunk, str) else None)
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as e:  # surface to the async consumer
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        # Submit onto the dedicated MLX executor (NOT the default thread pool),
        # so the stream shares the one Metal thread the model lives on.
        self._get_mlx_executor().submit(_produce)

        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, Exception):
                logger.error(f"Error during streaming generation: {item}")
                raise item
            yield item

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

            result = await self._run_mlx(do_generate)
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
        if self._mlx_executor is not None:
            self._mlx_executor.shutdown(wait=False)
            self._mlx_executor = None
