"""
LLM Factory — returns the correct LLM service based on model_config.yaml.

Usage:
    llm = get_llm()
    result = await llm.generate("...")
    structured = await llm.generate_structured("...", MyModel)

The returned object is always interface-compatible:
  .generate(prompt, system_prompt, max_tokens, temperature) -> str
  .generate_structured(prompt, model, system_prompt, max_tokens, temperature) -> Model
  .check_readiness() -> bool
  .wait_until_ready(timeout, poll_interval, on_waiting) -> bool
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Lazy import to avoid circular deps
_llm_instance = None


def get_llm(
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Return the configured LLM service instance (singleton per process).

    Args:
        provider_override: Force a specific provider ('ollama', 'medgemma', 'openai_compat').
        model_override: Force a specific model name.
    """
    global _llm_instance
    if _llm_instance is not None and not provider_override and not model_override:
        return _llm_instance

    config = _load_model_config()
    provider = provider_override or config.get("llm", {}).get("provider", "ollama")

    logger.info("LLM provider: %s", provider)

    if provider == "ollama":
        from app.services.ollama_service import OllamaService
        ollama_cfg = config.get("llm", {}).get("ollama", {})
        instance = OllamaService(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            model=model_override or ollama_cfg.get("model", "llama3.1:8b"),
            fallback_model=ollama_cfg.get("fallback_model", "llama3.2:3b"),
        )

    elif provider == "medgemma":
        from app.services.medgemma import MedGemmaService
        mg_cfg = config.get("llm", {}).get("medgemma", {})
        api_key_env = mg_cfg.get("api_key_env", "MEDGEMMA_API_KEY")
        instance = MedGemmaService(
            base_url=mg_cfg.get("base_url", ""),
            api_key=os.environ.get(api_key_env, ""),
            model_id=model_override or mg_cfg.get("model", "tgi"),
        )

    elif provider == "openai_compat":
        from app.services.medgemma import MedGemmaService  # reuses same client
        oc_cfg = config.get("llm", {}).get("openai_compat", {})
        api_key_env = oc_cfg.get("api_key_env", "LLM_API_KEY")
        instance = MedGemmaService(
            base_url=oc_cfg.get("base_url", ""),
            api_key=os.environ.get(api_key_env, ""),
            model_id=model_override or oc_cfg.get("model", ""),
        )

    else:
        logger.warning("Unknown provider '%s', defaulting to Ollama", provider)
        from app.services.ollama_service import OllamaService
        instance = OllamaService()

    if not provider_override and not model_override:
        _llm_instance = instance

    return instance


def get_agent_llm(agent_name: str):
    """
    Return an LLM service configured for a specific agent.

    Per-agent overrides in model_config.yaml take precedence.
    Falls back to the primary LLM if no override is defined.
    """
    config = _load_model_config()
    overrides = config.get("agent_overrides", {}).get(agent_name, {})

    base_llm = get_llm()

    if not overrides:
        return base_llm

    # Wrap the base LLM with per-agent parameter overrides
    return _AgentLLMWrapper(base_llm, overrides)


def _load_model_config() -> dict:
    """Load model_config.yaml, return empty dict on failure."""
    from app.config import settings
    config_path = Path(settings.model_config_path)
    if not config_path.exists():
        logger.warning("model_config.yaml not found at %s — using defaults", config_path)
        return {}
    try:
        import yaml  # type: ignore
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed — model_config.yaml cannot be read")
        return {}
    except Exception as e:
        logger.error("Failed to load model_config.yaml: %s", e)
        return {}


class _AgentLLMWrapper:
    """
    Thin wrapper that applies per-agent temperature and max_tokens overrides.
    Delegates everything else to the underlying LLM service.
    """

    def __init__(self, base_llm, overrides: dict):
        self._base = base_llm
        self._temperature = overrides.get("temperature")
        self._max_tokens = overrides.get("max_tokens")

    async def generate(self, prompt, system_prompt=None, max_tokens=0, temperature=None):
        return await self._base.generate(
            prompt,
            system_prompt,
            max_tokens or self._max_tokens or 0,
            temperature if temperature is not None else (self._temperature or 0.3),
        )

    async def generate_structured(self, prompt, response_model, system_prompt=None,
                                   max_tokens=0, temperature=None):
        return await self._base.generate_structured(
            prompt,
            response_model,
            system_prompt,
            max_tokens or self._max_tokens or 0,
            temperature if temperature is not None else (self._temperature or 0.2),
        )

    async def check_readiness(self):
        return await self._base.check_readiness()

    async def wait_until_ready(self, **kwargs):
        return await self._base.wait_until_ready(**kwargs)
