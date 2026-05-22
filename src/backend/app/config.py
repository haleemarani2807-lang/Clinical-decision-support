# [Track A: Baseline]
"""
Application configuration via environment variables.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # App
    app_name: str = "CDS Agent"
    debug: bool = True

    # Privacy mode: when True, disables data retention for open-web deployment
    # When False (default), full traceability is preserved for clinical environments
    privacy_mode: bool = False
    case_ttl_seconds: int = 300  # TTL for in-memory cases when privacy_mode=True

    # CORS
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://demo.briansheppard.com",
        "https://bshepp-cds-agent.hf.space",
    ]

    # MedGemma
    # Model ID varies by backend (env override wins):
    #   - HuggingFace TGI / Inference Endpoints: "tgi"
    #   - Google AI Studio (OpenAI-compatible): "google/medgemma-27b-text-it"
    #   - Local transformers: full model path, e.g. "google/medgemma-4b-it"
    medgemma_model_id: str = "google/medgemma-27b-text-it"
    medgemma_api_key: str = ""
    medgemma_base_url: str = ""  # For API-based access
    medgemma_device: str = "auto"  # "cpu", "cuda", "auto"
    medgemma_max_tokens: int = 4096

    # External APIs
    openfda_api_key: str = ""  # Optional, increases rate limits
    rxnorm_base_url: str = "https://rxnav.nlm.nih.gov/REST"
    pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    pubmed_api_key: str = ""  # Optional, increases rate limits
    hf_token: str = ""  # HuggingFace token for dataset downloads

    # RAG
    chroma_persist_dir: str = "./data/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Agent
    agent_max_retries: int = 2
    # Wall-clock timeout for the orchestrator pipeline (used by validation harness).
    # Observed baseline: ~75–90s; allow headroom for cold-start + complex cases.
    agent_timeout_seconds: int = 180
    agent_max_steps: int = 10
    default_include_drug_check: bool = True
    default_include_guidelines: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
