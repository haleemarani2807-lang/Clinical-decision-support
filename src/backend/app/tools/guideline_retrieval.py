# [Track A: Baseline]
"""
Tool: Guideline Retrieval (RAG)

Retrieves relevant clinical guideline excerpts using retrieval-augmented generation.
Uses ChromaDB for vector storage and sentence-transformers for embeddings.

This demonstrates RAG as a tool within the agent pipeline — the orchestrator
invokes it when clinical guidelines are needed for the current diagnosis.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.models.schemas import GuidelineExcerpt, GuidelineRetrievalResult

logger = logging.getLogger(__name__)

# Path to the comprehensive clinical guidelines corpus
GUIDELINES_DATA_PATH = Path(__file__).parent.parent / "data" / "clinical_guidelines.json"


class GuidelineRetrievalTool:
    """RAG-based clinical guideline retrieval."""

    def __init__(self):
        self._collection = None
        self._embedding_fn = None

    async def _ensure_initialized(self):
        """Lazy-initialize ChromaDB and embeddings."""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model,
            )

            client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

            self._collection = client.get_or_create_collection(
                name="clinical_guidelines",
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            # If collection is empty, load seed guidelines
            if self._collection.count() == 0:
                await self._load_seed_guidelines()

        except Exception as e:
            logger.warning(f"Vector search init skipped ({e}), falling back to lexical matching on guidelines corpus")
            self._collection = None

    async def run(self, query: str, n_results: int = 5) -> GuidelineRetrievalResult:
        """
        Retrieve relevant clinical guideline excerpts for a query.

        Args:
            query: Clinical query (e.g., "heart failure management guidelines")
            n_results: Number of excerpts to retrieve

        Returns:
            GuidelineRetrievalResult with relevant excerpts
        """
        await self._ensure_initialized()

        if self._collection is not None:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self._collection.count() or 1),
                    include=["documents", "metadatas", "distances"],
                )
                excerpts = []
                if results and results["documents"] and results["documents"][0]:
                    for doc, meta, distance in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    ):
                        excerpts.append(
                            GuidelineExcerpt(
                                title=meta.get("title", "Clinical Guideline"),
                                excerpt=doc,
                                source=meta.get("source", "Unknown"),
                                url=meta.get("url"),
                                relevance_score=round(1 - distance, 4),
                            )
                        )
                return GuidelineRetrievalResult(query=query, excerpts=excerpts)
            except Exception as e:
                logger.warning(f"Chroma query failed ({e}), falling back to lexical search")

        # Lexical matching fallback over the guidelines JSON corpus
        seed = self._get_seed_guidelines()
        q_tokens = [w for w in query.lower().split() if len(w) > 3]
        scored = []
        for g in seed:
            text_blob = f"{g.get('title', '')} {g.get('text', '')} {g.get('specialty', '')}".lower()
            score = sum(text_blob.count(t) for t in q_tokens)
            if score > 0:
                scored.append((score, g))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [g for _, g in scored[:n_results]] if scored else seed[:n_results]

        excerpts = [
            GuidelineExcerpt(
                title=g.get("title", "Clinical Practice Guideline"),
                excerpt=g.get("text", "")[:400],
                source=g.get("source", "ACC/AHA / CHEST Guidelines"),
                url=g.get("url"),
                relevance_score=0.85 if scored else 0.60,
            )
            for g in chosen
        ]
        return GuidelineRetrievalResult(query=query, excerpts=excerpts)

    async def _load_seed_guidelines(self):
        """
        Load seed clinical guidelines into the vector store.

        Loads from the comprehensive JSON corpus covering 14+ medical specialties.
        Each guideline is stored with metadata for filtering and attribution.
        """
        seed_guidelines = self._get_seed_guidelines()

        if not seed_guidelines:
            logger.warning("No seed guidelines to load")
            return

        documents = [g["text"] for g in seed_guidelines]
        metadatas = [
            {
                "title": g["title"],
                "source": g["source"],
                "url": g.get("url", ""),
                "specialty": g.get("specialty", "General"),
                "guideline_id": g.get("id", f"guideline_{i}"),
            }
            for i, g in enumerate(seed_guidelines)
        ]
        ids = [g.get("id", f"guideline_{i}") for i, g in enumerate(seed_guidelines)]

        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Loaded {len(seed_guidelines)} seed guidelines into vector store")

    @staticmethod
    def _get_seed_guidelines() -> List[dict]:
        """
        Load clinical guidelines from the comprehensive JSON corpus.

        The guidelines cover 14+ specialties including cardiology, emergency medicine,
        endocrinology, pulmonology, neurology, gastroenterology, infectious disease,
        psychiatry, pediatrics, nephrology, hematology, rheumatology, OB/GYN,
        dermatology, preventive medicine, and perioperative medicine.
        """
        if GUIDELINES_DATA_PATH.exists():
            try:
                with open(GUIDELINES_DATA_PATH, "r", encoding="utf-8") as f:
                    guidelines = json.load(f)
                logger.info(
                    f"Loaded {len(guidelines)} guidelines from {GUIDELINES_DATA_PATH.name} "
                    f"covering specialties: {', '.join(sorted(set(g.get('specialty', 'Unknown') for g in guidelines)))}"
                )
                return guidelines
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load guidelines from {GUIDELINES_DATA_PATH}: {e}")

        # Fallback: minimal seed guidelines if JSON file not found
        logger.warning("Guidelines JSON not found, using minimal fallback set")
        return [
            {
                "title": "ACC/AHA Chest Pain Guidelines (2021)",
                "source": "American College of Cardiology / American Heart Association",
                "url": "https://www.jacc.org/doi/10.1016/j.jacc.2021.07.053",
                "text": (
                    "Acute chest pain evaluation: Assess pretest probability of ACS. "
                    "High-sensitivity troponin is the preferred biomarker. Use HEART score "
                    "for risk stratification. Low-risk (HEART 0-3): consider early discharge. "
                    "High-risk (HEART 7-10): invasive strategy with cardiology consultation. "
                    "For STEMI, activate cath lab with door-to-balloon time <90 minutes."
                ),
            },
            {
                "title": "Surviving Sepsis Campaign (2021)",
                "source": "SCCM / ESICM",
                "url": "https://www.sccm.org/SurvivingSepsisCampaign/Guidelines",
                "text": (
                    "Sepsis hour-1 bundle: measure lactate, obtain blood cultures before "
                    "antibiotics, administer broad-spectrum antibiotics within 1 hour, "
                    "begin 30 mL/kg crystalloid for hypotension or lactate ≥4. "
                    "Norepinephrine first-line vasopressor. Target MAP ≥65."
                ),
            },
        ]

    async def add_guidelines(self, guidelines: List[dict]):
        """
        Add custom guidelines to the vector store.

        Args:
            guidelines: List of dicts with 'title', 'source', 'text', optional 'url'
        """
        await self._ensure_initialized()

        existing_count = self._collection.count()
        documents = [g["text"] for g in guidelines]
        metadatas = [
            {"title": g["title"], "source": g["source"], "url": g.get("url", "")}
            for g in guidelines
        ]
        ids = [f"guideline_{existing_count + i}" for i in range(len(guidelines))]

        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(guidelines)} guidelines to vector store")
