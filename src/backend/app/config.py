"""
Application configuration via environment variables.

Extended from the original to support:
- Ollama (local, default LLM provider)
- SQLite / PostgreSQL (pluggable database)
- model_config.yaml for per-component LLM overrides
- LangGraph orchestrator toggle
- Evaluation and experiment settings
"""
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "Evidence-Verified CDSS"
    debug: bool = True

    # Privacy mode: when True, disables data retention for open-web deployment
    privacy_mode: bool = False
    case_ttl_seconds: int = 300

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://demo.briansheppard.com",
        "https://bshepp-cds-agent.hf.space",
    ]

    # ── Model Config ─────────────────────────────────────────────────────────
    # Path to model_config.yaml — controls all LLM/embedding/reranker choices
    model_config_path: str = "./model_config.yaml"

    # ── Ollama (default LLM provider) ─────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_fallback_model: str = "llama3.2:3b"
    ollama_timeout: int = 300

    # ── MedGemma (optional cloud provider) ───────────────────────────────────
    # Preserved for backward compatibility and optional cloud use
    medgemma_model_id: str = "google/medgemma-27b-text-it"
    medgemma_api_key: str = ""
    medgemma_base_url: str = ""
    medgemma_device: str = "auto"
    medgemma_max_tokens: int = 4096

    # ── External APIs ─────────────────────────────────────────────────────────
    openfda_api_key: str = ""
    rxnorm_base_url: str = "https://rxnav.nlm.nih.gov/REST"
    pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    pubmed_api_key: str = ""
    hf_token: str = ""

    # ── Database ──────────────────────────────────────────────────────────────
    # SQLite by default (zero setup); set to postgresql://... for production
    database_url: str = "sqlite:///./data/cdss.db"

    # ── RAG / Vector Store ────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Hybrid retrieval weights
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    retrieval_top_k: int = 10

    # ── Agent Pipeline ────────────────────────────────────────────────────────
    agent_max_retries: int = 2
    agent_timeout_seconds: int = 300
    agent_max_steps: int = 15
    default_include_drug_check: bool = True
    default_include_guidelines: bool = True

    # Use LangGraph orchestrator for enhanced pipeline (Experiment 4)
    # False = original 6-step orchestrator (Experiment 1 baseline)
    use_langgraph: bool = False

    # ── Evidence Verification ─────────────────────────────────────────────────
    entailment_model: str = "cross-encoder/nli-deberta-v3-small"
    entailment_support_threshold: float = 0.5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Kaggle ────────────────────────────────────────────────────────────────
    kaggle_username: str = ""
    kaggle_key: str = ""

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_results_dir: str = "./experiments/results"
    eval_max_cases: int = 50

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Prevent "model_config" field from clashing with pydantic-settings
        "populate_by_name": True,
        "protected_namespaces": (),
    }


settings = Settings()
