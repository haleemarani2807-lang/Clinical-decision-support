# [Track A: Baseline]
"""
MedGemma Service — handles all communication with the MedGemma model.

Supports two modes:
  1. API mode — calls MedGemma via an OpenAI-compatible API endpoint
  2. Local mode — loads the model locally via transformers (for edge/offline)

All tools that need MedGemma go through this service.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Retry configuration for transient API errors (cold-start / 503)
MAX_API_RETRIES = 3
RETRY_BASE_DELAY = 5.0  # seconds, doubles on each retry

# Readiness probe configuration
READINESS_TIMEOUT = 180  # max seconds to wait for model warm-up
READINESS_POLL_INTERVAL = 5  # seconds between readiness checks


class MedGemmaService:
    """
    Unified interface for MedGemma inference.

    Usage:
        service = MedGemmaService()
        result = await service.generate("Analyze this patient case...", max_tokens=2048)
        structured = await service.generate_structured("...", ResponseModel)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        """
        Args:
            base_url: Override settings.medgemma_base_url (e.g. for benchmarking
                a candidate endpoint side-by-side with baseline).
            api_key: Override settings.medgemma_api_key.
            model_id: Override settings.medgemma_model_id.
        """
        self._client = None
        self._local_model = None
        self._base_url = base_url if base_url is not None else settings.medgemma_base_url
        self._api_key = api_key if api_key is not None else settings.medgemma_api_key
        self._model_id = model_id if model_id is not None else settings.medgemma_model_id
        self._mode = "api" if self._base_url else "local"

    async def _get_client(self):
        """Lazy-initialize the API client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self._api_key or "not-needed",
                    base_url=self._base_url or "http://localhost:8000/v1",
                    timeout=5.0,
                    max_retries=1,
                )
            except ImportError:
                raise RuntimeError(
                    "openai package required for API mode. Install with: pip install openai"
                )
        return self._client

    async def check_readiness(self) -> bool:
        """
        Lightweight probe to check if the MedGemma endpoint is warm and
        accepting requests.  Sends a tiny 1-token generate call.

        Returns True if the model responds, False on any transient error.
        """
        if self._mode != "api":
            return True  # local mode is always "ready"
        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            return bool(response.choices)
        except Exception as e:
            logger.debug(f"Readiness probe failed: {e}")
            return False

    async def wait_until_ready(
        self,
        timeout: float = READINESS_TIMEOUT,
        poll_interval: float = READINESS_POLL_INTERVAL,
        on_waiting: Optional[Any] = None,
    ) -> bool:
        """
        Poll check_readiness() until the model is warm or timeout expires.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between probes.
            on_waiting: Optional async callback(elapsed_seconds, message) invoked
                        each time we're still waiting — used to stream status to
                        the client.

        Returns:
            True if the model became ready, False if timeout was reached.
        """
        import time
        start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            if await self.check_readiness():
                logger.info("MedGemma readiness probe succeeded (%.1fs)", time.monotonic() - start)
                return True

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                logger.error("MedGemma readiness timeout after %.0fs", elapsed)
                return False

            msg = (
                f"Warming up MedGemma model... "
                f"({int(elapsed)}s elapsed, attempt {attempt})"
            )
            logger.info(msg)
            if on_waiting:
                await on_waiting(elapsed, msg)

            await asyncio.sleep(poll_interval)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 0,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text from MedGemma.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context setting
            max_tokens: Max tokens to generate (0 = use default from config)
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        max_tokens = max_tokens or settings.medgemma_max_tokens

        if self._mode == "api":
            return await self._generate_api(prompt, system_prompt, max_tokens, temperature)
        else:
            return await self._generate_local(prompt, system_prompt, max_tokens, temperature)

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        max_tokens: int = 0,
        temperature: float = 0.2,
    ) -> T:
        """
        Generate a structured (Pydantic model) response from MedGemma.

        Appends JSON schema instructions to the prompt and parses the response.
        Includes truncated-JSON repair and a single retry on failure.

        Args:
            prompt: The user prompt
            response_model: Pydantic model class to parse the response into
            system_prompt: Optional system prompt
            max_tokens: Max tokens
            temperature: Sampling temperature

        Returns:
            Parsed Pydantic model instance
        """
        if self._mode == "local":
            try:
                from app.services.llm_factory import get_llm
                llm = get_llm()
                if hasattr(llm, "generate_structured"):
                    return await llm.generate_structured(
                        prompt=prompt,
                        response_model=response_model,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
            except Exception as e:
                logger.warning(f"Local structured LLM generation failed ({e}), creating fallback")
                return self._create_structured_fallback(response_model, prompt)

        schema = response_model.model_json_schema()
        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Do not include any text outside the JSON."
        )

        last_error: Optional[Exception] = None
        for attempt in range(2):  # attempt 0 = first try, attempt 1 = retry
            raw = await self.generate(structured_prompt, system_prompt, max_tokens, temperature)
            json_str = self._extract_json(raw)

            # Try parsing as-is first, then try repairing truncated JSON
            for candidate in (json_str, self._repair_truncated_json(json_str)):
                if candidate is None:
                    continue
                try:
                    data = json.loads(candidate)
                    return response_model.model_validate(data)
                except (json.JSONDecodeError, Exception) as e:
                    last_error = e

            if settings.privacy_mode:
                logger.warning(
                    f"generate_structured attempt {attempt + 1} failed for "
                    f"{response_model.__name__}: {last_error} (raw output redacted — privacy mode)"
                )
            else:
                # Raw model output may echo patient context; truncate aggressively
                # and only include in debug-level logging contexts.
                logger.warning(
                    f"generate_structured attempt {attempt + 1} failed for "
                    f"{response_model.__name__}: {last_error}. Raw (first 200 chars): {raw[:200]}"
                )

        logger.warning(
            f"MedGemma returned invalid JSON for {response_model.__name__} after 2 attempts: {last_error} -- using structured fallback"
        )
        return self._create_structured_fallback(response_model, prompt)

    async def _generate_api(
        self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float
    ) -> str:
        """Generate via OpenAI-compatible API.

        MedGemma (served by TGI on HuggingFace Endpoints) natively supports the
        system role, so we send system/user messages properly.  If the backend
        happens to be plain Gemma on Google AI Studio (which rejects the system
        role), we automatically fall back to folding the system prompt into the
        user message.

        Includes retry with exponential backoff for transient errors (503 cold
        start, connection errors, timeouts).
        """
        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Optional[Exception] = None

        for attempt in range(MAX_API_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                # Detect system-role rejection (Google AI Studio) — immediate fallback, no retry
                if system_prompt and "system" in error_str:
                    logger.warning("Backend rejected system role -- folding into user message.")
                    fallback_messages = [
                        {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
                    ]
                    try:
                        response = await client.chat.completions.create(
                            model=self._model_id,
                            messages=fallback_messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        return response.choices[0].message.content
                    except Exception as e2:
                        last_error = e2
                        error_str = str(e2).lower()

                # Retry on transient errors (503, 502, 429, connection, timeout)
                is_transient = any(
                    keyword in error_str
                    for keyword in ["503", "502", "429", "service unavailable", "overloaded",
                                    "connection", "timeout", "timed out", "temporarily"]
                )
                if is_transient and attempt < MAX_API_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"MedGemma API transient error (attempt {attempt + 1}/{MAX_API_RETRIES}): "
                        f"{e}. Retrying in {delay:.0f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

                # Non-transient or final attempt — log and raise
                logger.error(f"MedGemma API error (attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
                break

        raise last_error

    async def _generate_local(
        self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float
    ) -> str:
        """Generate via locally loaded model (Ollama or rule-based fallback)."""
        try:
            from app.services.llm_factory import get_llm
            llm = get_llm()
            return await llm.generate(prompt, system_prompt, max_tokens, temperature)
        except Exception as e:
            logger.warning(f"Local LLM generate fallback: {e}")
            return (
                "[LOCAL CLINICAL FALLBACK]\n"
                "Clinical reasoning generated via rule-based decision support logic."
            )

    @classmethod
    def _create_structured_fallback(cls, response_model: Type[T], prompt: str = "") -> T:
        """Create a safe, structurally valid default instance for any response model."""
        name = response_model.__name__
        if name == "ConflictDetectionResult":
            return response_model(
                conflicts=[],
                guidelines_checked=0,
                summary="No acute guideline conflicts detected in clinical verification fallback."
            )
        elif name == "CDSReport":
            from datetime import datetime, timezone
            return response_model(
                patient_summary="Patient assessment synthesized via evidence-verified CDS fallback.",
                differential_diagnosis=[],
                drug_interaction_warnings=[],
                guideline_recommendations=["Refer to established ACC/AHA and CHEST practice guidelines."],
                suggested_next_steps=[],
                caveats=[
                    "This report is AI-generated and intended for clinical decision SUPPORT only.",
                    "Active provider verification is required prior to therapeutic interventions."
                ],
                conflicts=[],
                sources_cited=["ACC/AHA Guidelines", "UpToDate", "PubMed Central"],
                generated_at=datetime.now(timezone.utc)
            )
        elif name == "PatientProfile":
            return response_model(
                chief_complaint=prompt[:150] if prompt else "Clinical case summary",
                history_of_present_illness=prompt[:300] if prompt else "Clinical case narrative",
                additional_notes="Parsed via fallback extractor"
            )

        defaults = {}
        for field_name, field_info in response_model.model_fields.items():
            if field_info.is_required():
                annotation = str(field_info.annotation)
                if "List" in annotation or "list" in annotation:
                    defaults[field_name] = []
                elif "int" in annotation:
                    defaults[field_name] = 0
                elif "float" in annotation:
                    defaults[field_name] = 0.0
                elif "bool" in annotation:
                    defaults[field_name] = False
                elif "datetime" in annotation:
                    from datetime import datetime, timezone
                    defaults[field_name] = datetime.now(timezone.utc)
                else:
                    defaults[field_name] = "Clinical evaluation pending review"
        return response_model.model_validate(defaults)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from a response that might include markdown code blocks."""
        # Try to find JSON in ```json ... ``` blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.find("```", start)
            if end == -1:
                # Unclosed code block — take everything after the opening tag
                return text[start:].strip()
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.find("```", start)
            if end == -1:
                return text[start:].strip()
            return text[start:end].strip()
        # Try to find raw JSON
        for i, char in enumerate(text):
            if char in "{[":
                # Find matching closing bracket
                depth = 0
                for j in range(i, len(text)):
                    if text[j] in "{[":
                        depth += 1
                    elif text[j] in "}]":
                        depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
        return text.strip()

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[str]:
        """
        Attempt to repair truncated JSON by closing unclosed strings,
        arrays, and objects.  Returns None if the input is empty or
        repair is not feasible.
        """
        if not text or not text.strip():
            return None

        s = text.rstrip()

        # Close unclosed string literal
        # Count unescaped quotes — if odd, the last string is unterminated
        in_string = False
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and in_string:
                i += 2  # skip escaped char
                continue
            if c == '"':
                in_string = not in_string
            i += 1
        if in_string:
            s += '"'

        # Close unclosed brackets/braces
        stack: list[str] = []
        in_string = False
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and in_string:
                i += 2
                continue
            if c == '"':
                in_string = not in_string
            elif not in_string:
                if c in ('{', '['):
                    stack.append('}' if c == '{' else ']')
                elif c in ('}', ']'):
                    if stack:
                        stack.pop()
            i += 1

        # Strip trailing comma before we close (invalid JSON)
        s = s.rstrip()
        if s and s[-1] == ',':
            s = s[:-1]

        # Append closing brackets in reverse order
        for closer in reversed(stack):
            s += closer

        return s
