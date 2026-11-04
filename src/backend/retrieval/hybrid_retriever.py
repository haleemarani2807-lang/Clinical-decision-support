"""
Hybrid Retriever — combines BM25 + dense embeddings + CrossEncoder reranking.

Pipeline:
  1. BM25 retrieval over the document corpus
  2. Dense retrieval via ChromaDB (existing)
  3. Reciprocal Rank Fusion to merge scores
  4. CrossEncoder reranking of top candidates
  5. Deduplication by passage_id

The original GuidelineRetrievalTool (dense-only) is preserved as Experiment 1.
This hybrid retriever is used for Experiments 2, 3, and 4.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models.schemas import EvidenceBundle, EvidencePassage, EvidenceSource

logger = logging.getLogger(__name__)

GUIDELINES_DATA_PATH = Path(__file__).parent.parent / "app" / "data" / "clinical_guidelines.json"


class HybridRetriever:
    """
    Hybrid BM25 + Dense + Reranker retrieval pipeline.

    Usage:
        retriever = HybridRetriever()
        await retriever.initialize()
        bundle = await retriever.retrieve("chest pain management", top_k=10)
    """

    def __init__(
        self,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        top_k_bm25: int = 20,
        top_k_dense: int = 20,
        top_k_final: int = 10,
    ):
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._top_k_bm25 = top_k_bm25
        self._top_k_dense = top_k_dense
        self._top_k_final = top_k_final

        self._corpus: list[dict] = []
        self._bm25 = None
        self._chroma_collection = None
        self._reranker = None
        self._initialized = False

    async def initialize(self):
        """Load corpus and initialize BM25 + ChromaDB. Idempotent."""
        if self._initialized:
            return
        self._corpus = self._load_corpus()
        self._bm25 = self._build_bm25(self._corpus)
        self._chroma_collection = await self._get_chroma_collection(self._corpus)
        self._reranker = self._load_reranker()
        self._initialized = True
        logger.info(
            "HybridRetriever initialized: %d documents, BM25=%s, Reranker=%s",
            len(self._corpus),
            "OK" if self._bm25 else "UNAVAILABLE",
            "OK" if self._reranker else "UNAVAILABLE",
        )

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_pubmed: bool = False,
        pubmed_passages: Optional[list[EvidencePassage]] = None,
    ) -> EvidenceBundle:
        """
        Retrieve evidence using the hybrid pipeline.

        Args:
            query: Clinical query string.
            top_k: Number of final results (defaults to settings.retrieval_top_k).
            include_pubmed: Whether to include PubMed passages in hybrid ranking.
            pubmed_passages: Pre-fetched PubMed passages to include.
        """
        if not self._initialized:
            await self.initialize()

        top_k = top_k or settings.retrieval_top_k

        # ── Step 1: BM25 retrieval ─────────────────────────────────────────────
        bm25_results = self._bm25_retrieve(query, self._top_k_bm25)

        # ── Step 2: Dense retrieval (ChromaDB) ─────────────────────────────────
        dense_results = await self._dense_retrieve(query, self._top_k_dense)

        # ── Step 3: Add PubMed passages ───────────────────────────────────────
        external_passages = pubmed_passages or []

        # ── Step 4: Reciprocal Rank Fusion ────────────────────────────────────
        all_passages = self._reciprocal_rank_fusion(
            bm25_results, dense_results, external_passages
        )

        # ── Step 5: Rerank ────────────────────────────────────────────────────
        reranked = self._rerank(query, all_passages)

        # ── Step 6: Deduplicate and return top-k ──────────────────────────────
        final = self._deduplicate(reranked)[:top_k]

        for rank, passage in enumerate(final, 1):
            passage.final_rank = rank

        return EvidenceBundle(
            query=query,
            passages=final,
            total_retrieved=len(bm25_results) + len(dense_results) + len(external_passages),
            retrieval_strategy="hybrid",
            reranked=self._reranker is not None,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # BM25
    # ──────────────────────────────────────────────────────────────────────────

    def _bm25_retrieve(self, query: str, top_k: int) -> list[EvidencePassage]:
        """BM25 retrieval. Returns empty list if rank-bm25 not installed."""
        if not self._bm25:
            return []
        try:
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for rank, idx in enumerate(top_indices):
                if scores[idx] <= 0:
                    break
                doc = self._corpus[idx]
                passage = self._doc_to_passage(doc, idx)
                passage.bm25_score = float(scores[idx])
                results.append(passage)
            return results
        except Exception as e:
            logger.warning("BM25 retrieval failed: %s", e)
            return []

    @staticmethod
    def _build_bm25(corpus: list[dict]):
        """Build BM25 index from corpus. Returns None if rank-bm25 not installed."""
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.get("text", "").lower().split() for doc in corpus]
            return BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank-bm25 not installed — BM25 retrieval disabled. Run: pip install rank-bm25")
            return None
        except Exception as e:
            logger.error("BM25 index build failed: %s", e)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Dense (ChromaDB)
    # ──────────────────────────────────────────────────────────────────────────

    async def _dense_retrieve(self, query: str, top_k: int) -> list[EvidencePassage]:
        """Dense retrieval via ChromaDB cosine similarity."""
        if not self._chroma_collection:
            return []
        try:
            n = min(top_k, self._chroma_collection.count() or 1)
            results = self._chroma_collection.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            passages = []
            if results and results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    p = EvidencePassage(
                        passage_id=meta.get("guideline_id", str(uuid.uuid4())),
                        text=doc,
                        source=EvidenceSource.GUIDELINE,
                        title=meta.get("title", "Clinical Guideline"),
                        url=meta.get("url"),
                        dense_score=round(1 - dist, 4),
                    )
                    passages.append(p)
            return passages
        except Exception as e:
            logger.warning("Dense retrieval failed: %s", e)
            return []

    async def _get_chroma_collection(self, corpus: list[dict]):
        """Get or create ChromaDB collection with the guideline corpus."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            # Guard against network hang if model not pre-downloaded
            from sentence_transformers import SentenceTransformer
            _ = SentenceTransformer(settings.embedding_model, local_files_only=True)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
            client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            collection = client.get_or_create_collection(
                name="clinical_guidelines",
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            if collection.count() == 0 and corpus:
                collection.add(
                    documents=[d["text"] for d in corpus],
                    metadatas=[{
                        "title": d.get("title", ""),
                        "source": d.get("source", ""),
                        "url": d.get("url", ""),
                        "specialty": d.get("specialty", ""),
                        "guideline_id": d.get("id", f"g_{i}"),
                    } for i, d in enumerate(corpus)],
                    ids=[d.get("id", f"g_{i}") for i, d in enumerate(corpus)],
                )
                logger.info("Loaded %d guidelines into ChromaDB", len(corpus))
            return collection
        except Exception as e:
            logger.warning("ChromaDB initialization failed: %s", e)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Reciprocal Rank Fusion
    # ──────────────────────────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[EvidencePassage],
        dense_results: list[EvidencePassage],
        external_passages: list[EvidencePassage],
        k: int = 60,
    ) -> list[EvidencePassage]:
        """
        Merge BM25 and dense results using Reciprocal Rank Fusion.
        RRF score = sum(weight / (k + rank)) across lists.
        """
        passage_map: dict[str, EvidencePassage] = {}
        rrf_scores: dict[str, float] = {}

        def add_list(passages: list[EvidencePassage], weight: float):
            for rank, passage in enumerate(passages, 1):
                pid = passage.passage_id
                if pid not in passage_map:
                    passage_map[pid] = passage
                    rrf_scores[pid] = 0.0
                rrf_scores[pid] += weight * (1.0 / (k + rank))

        add_list(bm25_results, self._bm25_weight)
        add_list(dense_results, self._dense_weight)
        add_list(external_passages, 0.5)  # External passages get equal weight

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        results = []
        for pid in sorted_ids:
            p = passage_map[pid]
            p.hybrid_score = round(rrf_scores[pid], 6)
            results.append(p)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Reranker
    # ──────────────────────────────────────────────────────────────────────────

    def _rerank(self, query: str, passages: list[EvidencePassage]) -> list[EvidencePassage]:
        """CrossEncoder reranking. Falls back to hybrid scores if unavailable."""
        if not self._reranker or not passages:
            return passages
        try:
            pairs = [[query, p.text[:512]] for p in passages]
            scores = self._reranker.predict(pairs)
            for p, score in zip(passages, scores):
                p.reranker_score = float(score)
            return sorted(passages, key=lambda p: p.reranker_score or 0, reverse=True)
        except Exception as e:
            logger.warning("Reranking failed: %s — using hybrid scores", e)
            return passages

    @staticmethod
    def _load_reranker():
        """Load CrossEncoder reranker. Returns None if not installed."""
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(settings.reranker_model, local_files_only=True)
            logger.info("Reranker loaded: %s", settings.reranker_model)
            return model
        except ImportError:
            logger.warning("sentence-transformers not installed — reranker disabled")
            return None
        except Exception as e:
            logger.warning("Reranker load failed: %s — reranker disabled", e)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(passages: list[EvidencePassage]) -> list[EvidencePassage]:
        """Remove duplicate passages by passage_id."""
        seen = set()
        result = []
        for p in passages:
            if p.passage_id not in seen:
                seen.add(p.passage_id)
                result.append(p)
        return result

    @staticmethod
    def _load_corpus() -> list[dict]:
        """Load the guideline corpus from JSON file."""
        if GUIDELINES_DATA_PATH.exists():
            try:
                import json
                with open(GUIDELINES_DATA_PATH, encoding="utf-8") as f:
                    corpus = json.load(f)
                logger.info("Loaded %d guidelines from %s", len(corpus), GUIDELINES_DATA_PATH.name)
                return corpus
            except Exception as e:
                logger.error("Failed to load guidelines: %s", e)
        return []

    @staticmethod
    def _doc_to_passage(doc: dict, idx: int) -> EvidencePassage:
        return EvidencePassage(
            passage_id=doc.get("id", f"guideline_{idx}"),
            text=doc.get("text", ""),
            source=EvidenceSource.GUIDELINE,
            title=doc.get("title", "Clinical Guideline"),
            url=doc.get("url"),
        )
