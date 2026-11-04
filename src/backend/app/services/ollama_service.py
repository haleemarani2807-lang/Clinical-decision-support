"""
Ollama Service -- handles all communication with the local Ollama server.

Supports:
- Streaming and non-streaming generation
- Structured (Pydantic) output via JSON mode
- Automatic fallback to smaller model when RAM is constrained
- Rule-based fallbacks when Ollama is unavailable
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_TAGS_ENDPOINT = "/api/tags"
OLLAMA_GENERATE_ENDPOINT = "/api/generate"


class OllamaService:
    """
    Unified interface for Ollama local LLM inference.

    Usage:
        service = OllamaService()
        result = await service.generate("Analyze this patient case...")
        structured = await service.generate_structured("...", ResponseModel)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model
        self._fallback_model = fallback_model or settings.ollama_fallback_model
        self._active_model: Optional[str] = None  # set after availability check
        self._is_server_available: Optional[bool] = None  # Cached server availability flag

    # ──────────────────────────────────────────────────────────────────────────
    # Public API (matches MedGemmaService interface for drop-in compatibility)
    # ──────────────────────────────────────────────────────────────────────────

    def _is_endpoint_alive(self) -> bool:
        """Instant 50ms socket probe to check if Ollama daemon is reachable."""
        if self._is_server_available is not None:
            return self._is_server_available
        import socket
        try:
            url_str = self._base_url.replace("http://", "").replace("https://", "").split("/")[0]
            if ":" in url_str:
                host, port = url_str.split(":")
                port = int(port)
            else:
                host, port = url_str, 11434
            if host in ("localhost", ""):
                host = "127.0.0.1"
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.05)
            s.connect((host, port))
            s.close()
            self._is_server_available = True
            return True
        except Exception:
            self._is_server_available = False
            return False

    async def check_readiness(self) -> bool:
        """Returns True if Ollama is running and responsive."""
        return self._is_endpoint_alive()

    async def wait_until_ready(
        self,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
        on_waiting: Any = None,
    ) -> bool:
        """Poll until Ollama is ready or timeout expires."""
        import time
        start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            if await self.check_readiness():
                logger.info("Ollama readiness probe succeeded (%.1fs)", time.monotonic() - start)
                return True
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                logger.warning("Ollama not available after %.0fs -- will use rule-based fallbacks", elapsed)
                return False
            msg = f"Waiting for Ollama... ({int(elapsed)}s elapsed, attempt {attempt})"
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
        Generate text from Ollama.

        Falls back to rule-based response if Ollama is unavailable.
        """
        if self._is_server_available is False:
            return self._rule_based_fallback(prompt, system_prompt)
        if not self._is_endpoint_alive():
            return self._rule_based_fallback(prompt, system_prompt)

        max_tokens = max_tokens or settings.medgemma_max_tokens

        model = await self._resolve_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=float(settings.ollama_timeout), connect=2.0)
            ) as client:
                resp = await client.post(
                    f"{self._base_url}{OLLAMA_CHAT_ENDPOINT}",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]

        except Exception as e:
            logger.warning("Ollama generation failed: %s -- using rule-based fallback", e)
            return self._rule_based_fallback(prompt, system_prompt)

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        max_tokens: int = 0,
        temperature: float = 0.2,
    ) -> T:
        """
        Generate a structured (Pydantic model) response from Ollama.

        Strategy:
          1. Ask Ollama to output JSON matching the schema.
          2. Parse and validate with Pydantic.
          3. On failure, retry once with explicit schema reminder.
          4. On repeated failure, return a safe rule-based default.
        """
        if self._is_server_available is False:
            return self._structured_fallback(response_model)
        if not self._is_endpoint_alive():
            return self._structured_fallback(response_model)
        schema = response_model.model_json_schema()
        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with valid JSON matching this exact schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Do not include any text outside the JSON object."
        )

        last_error: Optional[Exception] = None
        for attempt in range(2):
            raw = await self.generate(structured_prompt, system_prompt, max_tokens, temperature)
            json_str = self._extract_json(raw)

            for candidate in (json_str, self._repair_truncated_json(json_str)):
                if candidate is None:
                    continue
                try:
                    data = json.loads(candidate)
                    return response_model.model_validate(data)
                except Exception as e:
                    last_error = e

            logger.warning(
                "generate_structured attempt %d failed for %s: %s",
                attempt + 1, response_model.__name__, last_error,
            )

        # Rule-based fallback: return a minimal valid instance
        logger.error(
            "All Ollama structured generation attempts failed for %s -- using default",
            response_model.__name__,
        )
        return self._structured_fallback(response_model)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _resolve_model(self) -> str:
        """
        Select the appropriate model based on availability.
        Does NOT auto-pull -- respects model_config.yaml auto_pull=false.
        """
        if self._active_model:
            return self._active_model

        for candidate in [self._model, self._fallback_model]:
            if await self._model_available(candidate):
                self._active_model = candidate
                logger.info("Using Ollama model: %s", candidate)
                return candidate

        # Neither model pulled -- log clear instructions and use fallback name
        logger.error(
            "Neither '%s' nor '%s' found in Ollama. "
            "Run: ollama pull %s (or %s for development).",
            self._model, self._fallback_model,
            self._model, self._fallback_model,
        )
        # Still return the primary name so the error comes from the API call
        return self._model

    async def _model_available(self, model_name: str) -> bool:
        """Check if a model is already pulled in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=1.0)) as client:
                resp = await client.get(f"{self._base_url}{OLLAMA_TAGS_ENDPOINT}")
                if resp.status_code != 200:
                    return False
                tags = resp.json().get("models", [])
                available = {m.get("name", "").split(":")[0] for m in tags}
                # also check full name
                available_full = {m.get("name", "") for m in tags}
                check_name = model_name.split(":")[0]
                return check_name in available or model_name in available_full
        except Exception:
            return False

    @staticmethod
    def _rule_based_fallback(prompt: str, system_prompt: Optional[str]) -> str:
        """
        Minimal rule-based text response when Ollama is unavailable.
        Returns a clearly-labelled placeholder so the pipeline does not silently fail.
        """
        return (
            "[RULE-BASED FALLBACK -- LLM UNAVAILABLE]\n"
            "The local Ollama LLM service is not reachable. "
            "This response was generated by a rule-based placeholder and requires human review. "
            "Please ensure Ollama is running: `ollama serve` and the required model is pulled."
        )

    @staticmethod
    def _structured_fallback(response_model: Type[T]) -> T:
        """
        Return the safest possible default instance for a Pydantic model
        when LLM generation fails completely.
        """
        try:
            # Try to construct with empty/default values
            defaults = {}
            for field_name, field_info in response_model.model_fields.items():
                if field_info.is_required():
                    # Use empty string, empty list, or 0 depending on annotation
                    annotation = str(field_info.annotation)
                    if "List" in annotation or "list" in annotation:
                        defaults[field_name] = []
                    elif "int" in annotation:
                        defaults[field_name] = 0
                    elif "float" in annotation:
                        defaults[field_name] = 0.0
                    elif "bool" in annotation:
                        defaults[field_name] = False
                    else:
                        defaults[field_name] = "[LLM unavailable -- requires human review]"
            return response_model.model_validate(defaults)
        except Exception as e:
            raise RuntimeError(
                f"Cannot create fallback for {response_model.__name__}: {e}. "
                "Ensure Ollama is running and model is pulled."
            ) from e

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from a response that might include markdown code blocks."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip() if end != -1 else text[start:].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.find("```", start)
            return text[start:end].strip() if end != -1 else text[start:].strip()
        # Find raw JSON object or array
        for i, char in enumerate(text):
            if char in "{[":
                depth = 0
                for j in range(i, len(text)):
                    if text[j] in "{[":
                        depth += 1
                    elif text[j] in "}]":
                        depth -= 1
                    if depth == 0:
                        return text[i: j + 1]
        return text.strip()

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[str]:
        """Attempt to repair truncated JSON by closing unclosed brackets."""
        if not text or not text.strip():
            return None
        s = text.rstrip()
        # Close unclosed string
        in_string = False
        i = 0
        while i < len(s):
            c = s[i]
            if c == "\\" and in_string:
                i += 2
                continue
            if c == '"':
                in_string = not in_string
            i += 1
        if in_string:
            s += '"'
        # Close unclosed brackets
        stack: list[str] = []
        in_string = False
        i = 0
        while i < len(s):
            c = s[i]
            if c == "\\" and in_string:
                i += 2
                continue
            if c == '"':
                in_string = not in_string
            elif not in_string:
                if c in ("{", "["):
                    stack.append("}" if c == "{" else "]")
                elif c in ("}", "]"):
                    if stack:
                        stack.pop()
            i += 1
        s = s.rstrip()
        if s and s[-1] == ",":
            s = s[:-1]
        for closer in reversed(stack):
            s += closer
        return s
