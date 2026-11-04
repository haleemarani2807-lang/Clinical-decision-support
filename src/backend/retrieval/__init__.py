"""
Evidence-Verified CDSS Retrieval Module.

Hybrid retrieval pipeline for clinical evidence:
  - PubMedClient: Real PubMed E-utilities integration
  - HybridRetriever: BM25 + Dense + RRF + CrossEncoder reranking
"""
from retrieval.pubmed_client import PubMedClient
from retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "PubMedClient",
    "HybridRetriever",
]
