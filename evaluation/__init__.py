"""
Evidence-Verified CDSS Evaluation Framework.

Supports 4 experimental comparisons:
  Experiment 1: Baseline 6-step orchestrator
  Experiment 2: + Hybrid retrieval (BM25 + Dense + PubMed)
  Experiment 3: + Evidence verification (NLI entailment)
  Experiment 4: Full LangGraph pipeline with uncertainty assessment
"""
from evaluation.eval_framework import (
    EvaluationMetrics,
    EvaluationRunner,
    ExperimentResult,
    ExperimentSuite,
)

__all__ = [
    "EvaluationMetrics",
    "EvaluationRunner",
    "ExperimentResult",
    "ExperimentSuite",
]
