"""
Base classes and utilities for the validation framework.

Provides:
  - ValidationCase: a single test case with input + ground truth
  - ValidationResult: scored result for a single case
  - ValidationSummary: aggregate metrics for a dataset
  - run_cds_pipeline(): runs a case through the orchestrator directly
  - fuzzy_match(): soft string matching for diagnosis comparison
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── CDS pipeline imports ──
import sys

# Ensure the backend app is importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agent.orchestrator import Orchestrator
from app.models.schemas import (
    CaseSubmission,
    CDSReport,
    ClinicalReasoningResult,
    AgentState,
    AgentStepStatus,
)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ValidationCase:
    """A single validation test case."""
    case_id: str
    source_dataset: str                    # "medqa", "mtsamples", "pmc"
    input_text: str                        # Clinical text fed to the pipeline
    ground_truth: Dict[str, Any]           # Dataset-specific ground truth
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of running one case through the pipeline + scoring."""
    case_id: str
    source_dataset: str
    success: bool                          # Pipeline completed without crash
    scores: Dict[str, float]              # Metric name → score (0.0–1.0)
    pipeline_time_ms: int = 0
    step_results: Dict[str, str] = field(default_factory=dict)  # step_id → status
    report_summary: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)       # Extra scoring info


@dataclass
class ValidationSummary:
    """Aggregate metrics for a dataset validation run."""
    dataset: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    metrics: Dict[str, float]              # Metric name → average score
    per_case: List[ValidationResult]
    run_duration_sec: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────

async def run_cds_pipeline(
    patient_text: str,
    include_drug_check: bool = True,
    include_guidelines: bool = True,
    timeout_sec: int = 180,
) -> tuple[Optional[AgentState], Optional[CDSReport], Optional[str]]:
    """
    Run a single case through the CDS pipeline directly (no HTTP server needed).

    Returns:
        (state, report, error) — error is None on success
    """
    case = CaseSubmission(
        patient_text=patient_text,
        include_drug_check=include_drug_check,
        include_guidelines=include_guidelines,
    )
    orchestrator = Orchestrator()

    async def _run_pipeline() -> Optional[CDSReport]:
        async for _step_update in orchestrator.run(case):
            pass  # consume all step updates
        return orchestrator.get_result()

    try:
        report = await asyncio.wait_for(_run_pipeline(), timeout=timeout_sec)

        # If no report was produced, collect errors from failed steps
        if report is None and orchestrator.state:
            failed_steps = [
                s for s in orchestrator.state.steps
                if s.status == AgentStepStatus.FAILED
            ]
            if failed_steps:
                error_msgs = [f"{s.step_id}: {s.error}" for s in failed_steps]
                return orchestrator.state, None, "; ".join(error_msgs)
            return orchestrator.state, None, "Pipeline completed but produced no report"

        return orchestrator.state, report, None
    except asyncio.TimeoutError:
        return orchestrator.state, None, f"Pipeline timed out after {timeout_sec}s"
    except Exception as e:
        return orchestrator.state, None, str(e)


# ──────────────────────────────────────────────
# Fuzzy string matching for diagnosis comparison
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Medical stopwords that don't carry diagnostic meaning
_MEDICAL_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "was",
    "were", "be", "been", "with", "for", "on", "at", "by", "from", "this",
    "that", "these", "those", "it", "its", "has", "have", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "most", "likely", "following", "which", "what", "patient", "patients",
})


def _content_tokens(text: str) -> set:
    """Extract meaningful content tokens, removing medical stopwords."""
    tokens = set(normalize_text(text).split())
    return tokens - _MEDICAL_STOPWORDS


def fuzzy_match(candidate: str, target: str, threshold: float = 0.6) -> bool:
    """
    Check if candidate text is a fuzzy match for target.

    Strategy (checked in order, first match wins):
      1. Normalized substring containment (either direction)
      2. All content tokens of target appear in candidate (recall=1.0)
      3. Token overlap ratio >= threshold (using content tokens, recall-based)

    Args:
        candidate: Text from the pipeline output (may be long)
        target: Ground truth text (usually short)
        threshold: Minimum token overlap ratio (0.0-1.0)
    """
    c_norm = normalize_text(candidate)
    t_norm = normalize_text(target)

    if not t_norm:
        return False

    # 1. Substring containment (either direction)
    if t_norm in c_norm or c_norm in t_norm:
        return True

    # 2. All content tokens of target present in candidate
    t_content = _content_tokens(target)
    c_content = _content_tokens(candidate)

    if t_content and t_content.issubset(c_content):
        return True

    # 3. Token overlap ratio (recall-based: what fraction of target is present?)
    if not t_content or not c_content:
        return False

    overlap = len(t_content & c_content)
    recall = overlap / len(t_content)

    return recall >= threshold


def diagnosis_in_differential(
    target_diagnosis: str,
    report: CDSReport,
    top_n: Optional[int] = None,
) -> tuple[bool, int, str]:
    """
    Check if target_diagnosis appears in the report's differential.

    Returns:
        (found, rank, match_location) — rank is 0-indexed position, or -1 if not found.
        match_location is one of: "differential", "next_steps", "recommendations",
        "fulltext", or "not_found".
    """
    diagnoses = report.differential_diagnosis
    if top_n:
        diagnoses = diagnoses[:top_n]

    for i, dx in enumerate(diagnoses):
        if fuzzy_match(dx.diagnosis, target_diagnosis):
            return True, i, "differential"

    # Check suggested_next_steps (for management-type answers)
    for i, action in enumerate(report.suggested_next_steps):
        if fuzzy_match(action.action, target_diagnosis):
            return True, len(diagnoses) + i, "next_steps"

    # Check guideline recommendations (for treatment-type answers)
    for i, rec in enumerate(report.guideline_recommendations):
        if fuzzy_match(rec, target_diagnosis):
            return True, len(diagnoses) + len(report.suggested_next_steps) + i, "recommendations"

    # Broad fulltext check (patient_summary, recommendations, next steps combined)
    full_text = " ".join([
        report.patient_summary or "",
        " ".join(report.guideline_recommendations),
        " ".join(a.action for a in report.suggested_next_steps),
        " ".join(dx.reasoning for dx in report.differential_diagnosis),
    ])
    if fuzzy_match(full_text, target_diagnosis, threshold=0.3):
        return True, len(diagnoses), "fulltext"

    return False, -1, "not_found"


# ──────────────────────────────────────────────
# Type-aware scoring (P4)
# ──────────────────────────────────────────────

def score_case(
    target_answer: str,
    report: CDSReport,
    question_type: str = "diagnostic",
    reasoning_result: Optional[ClinicalReasoningResult] = None,
) -> dict:
    """
    Score a case based on its question type.

    Returns a dict of metric_name -> score (0.0 or 1.0), plus
    'match_location' (str) and 'match_rank' (int) detail fields.
    """
    qt = question_type.lower()

    if qt == "diagnostic":
        return _score_diagnostic(target_answer, report)
    elif qt == "treatment":
        return _score_treatment(target_answer, report)
    elif qt == "mechanism":
        return _score_mechanism(target_answer, report, reasoning_result)
    elif qt == "lab_finding":
        return _score_lab_finding(target_answer, report, reasoning_result)
    else:
        return _score_generic(target_answer, report, reasoning_result)


def _score_diagnostic(target: str, report: CDSReport) -> dict:
    """Score a diagnostic question -- primary field is differential_diagnosis."""
    found_top1, r1, l1 = diagnosis_in_differential(target, report, top_n=1)
    found_top3, r3, l3 = diagnosis_in_differential(target, report, top_n=3)
    found_any, ra, la = diagnosis_in_differential(target, report)

    return {
        "top1_accuracy": 1.0 if found_top1 else 0.0,
        "top3_accuracy": 1.0 if found_top3 else 0.0,
        "mentioned_accuracy": 1.0 if found_any else 0.0,
        "differential_accuracy": 1.0 if (found_any and la == "differential") else 0.0,
        "match_location": la,
        "match_rank": ra,
    }


def _score_treatment(target: str, report: CDSReport) -> dict:
    """Score a treatment question -- primary fields are next_steps + recommendations."""
    # Check suggested_next_steps first (most specific for treatment)
    for i, action in enumerate(report.suggested_next_steps):
        if fuzzy_match(action.action, target):
            return {
                "top1_accuracy": 1.0 if i == 0 else 0.0,
                "top3_accuracy": 1.0 if i < 3 else 0.0,
                "mentioned_accuracy": 1.0,
                "differential_accuracy": 0.0,
                "match_location": "next_steps",
                "match_rank": i,
            }

    # Check guideline_recommendations
    for i, rec in enumerate(report.guideline_recommendations):
        if fuzzy_match(rec, target):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "differential_accuracy": 0.0,
                "match_location": "recommendations",
                "match_rank": i,
            }

    # Check differential reasoning text (treatment may appear in reasoning)
    for dx in report.differential_diagnosis:
        if fuzzy_match(dx.reasoning, target, threshold=0.3):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "differential_accuracy": 0.0,
                "match_location": "reasoning_text",
                "match_rank": -1,
            }

    # Fulltext fallback
    full_text = _build_fulltext(report)
    if fuzzy_match(full_text, target, threshold=0.3):
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mentioned_accuracy": 1.0,
            "differential_accuracy": 0.0,
            "match_location": "fulltext",
            "match_rank": -1,
        }

    return _not_found()


def _score_mechanism(
    target: str,
    report: CDSReport,
    reasoning_result: Optional[ClinicalReasoningResult] = None,
) -> dict:
    """Score a mechanism question -- primary field is reasoning_chain."""
    # Check reasoning chain from clinical reasoning step
    if reasoning_result and reasoning_result.reasoning_chain:
        if fuzzy_match(reasoning_result.reasoning_chain, target, threshold=0.3):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "differential_accuracy": 0.0,
                "match_location": "reasoning_chain",
                "match_rank": -1,
            }

    # Check differential reasoning text
    for dx in report.differential_diagnosis:
        if fuzzy_match(dx.reasoning, target, threshold=0.3):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "differential_accuracy": 0.0,
                "match_location": "differential_reasoning",
                "match_rank": -1,
            }

    # Fulltext fallback
    full_text = _build_fulltext(report)
    if fuzzy_match(full_text, target, threshold=0.3):
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mentioned_accuracy": 1.0,
            "differential_accuracy": 0.0,
            "match_location": "fulltext",
            "match_rank": -1,
        }

    return _not_found()


def _score_lab_finding(
    target: str,
    report: CDSReport,
    reasoning_result: Optional[ClinicalReasoningResult] = None,
) -> dict:
    """Score a lab/finding question -- primary field is recommended_workup."""
    # Check recommended workup from clinical reasoning step
    if reasoning_result:
        for i, action in enumerate(reasoning_result.recommended_workup):
            if fuzzy_match(action.action, target, threshold=0.4):
                return {
                    "top1_accuracy": 1.0 if i == 0 else 0.0,
                    "top3_accuracy": 1.0 if i < 3 else 0.0,
                    "mentioned_accuracy": 1.0,
                    "differential_accuracy": 0.0,
                    "match_location": "recommended_workup",
                    "match_rank": i,
                }

    # Check next steps in final report
    for i, action in enumerate(report.suggested_next_steps):
        if fuzzy_match(action.action, target, threshold=0.4):
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mentioned_accuracy": 1.0,
                "differential_accuracy": 0.0,
                "match_location": "next_steps",
                "match_rank": i,
            }

    # Fulltext fallback
    full_text = _build_fulltext(report)
    if fuzzy_match(full_text, target, threshold=0.3):
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "mentioned_accuracy": 1.0,
            "differential_accuracy": 0.0,
            "match_location": "fulltext",
            "match_rank": -1,
        }

    return _not_found()


def _score_generic(
    target: str,
    report: CDSReport,
    reasoning_result: Optional[ClinicalReasoningResult] = None,
) -> dict:
    """Score any question type -- searches all fields broadly."""
    # Try diagnostic scoring first
    result = _score_diagnostic(target, report)
    if result.get("mentioned_accuracy", 0.0) > 0.0:
        return result

    # Try treatment scoring
    result = _score_treatment(target, report)
    if result.get("mentioned_accuracy", 0.0) > 0.0:
        return result

    # Try mechanism scoring
    if reasoning_result:
        result = _score_mechanism(target, report, reasoning_result)
        if result.get("mentioned_accuracy", 0.0) > 0.0:
            return result

    return _not_found()


def _build_fulltext(report: CDSReport) -> str:
    """Concatenate all report fields into a single searchable string."""
    parts = [
        report.patient_summary or "",
        " ".join(report.guideline_recommendations),
        " ".join(a.action for a in report.suggested_next_steps),
        " ".join(dx.diagnosis + " " + dx.reasoning for dx in report.differential_diagnosis),
        " ".join(report.sources_cited),
    ]
    if report.conflicts:
        parts.append(" ".join(c.description for c in report.conflicts))
    return " ".join(parts)


def _not_found() -> dict:
    """Return a zero-score result dict."""
    return {
        "top1_accuracy": 0.0,
        "top3_accuracy": 0.0,
        "mentioned_accuracy": 0.0,
        "differential_accuracy": 0.0,
        "match_location": "not_found",
        "match_rank": -1,
    }


# ──────────────────────────────────────────────
# I/O utilities
# ──────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _result_to_dict(r: ValidationResult) -> dict:
    """Convert a ValidationResult to a serialisable dict."""
    return {
        "case_id": r.case_id,
        "source_dataset": r.source_dataset,
        "success": r.success,
        "scores": r.scores,
        "pipeline_time_ms": r.pipeline_time_ms,
        "step_results": r.step_results,
        "report_summary": r.report_summary,
        "error": r.error,
        "details": r.details,
    }


# ──────────────────────────────────────────────
# Incremental checkpoint (JSONL)
# ──────────────────────────────────────────────

def checkpoint_path(dataset: str) -> Path:
    """Return the path to the checkpoint JSONL for *dataset*."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{dataset}_checkpoint.jsonl"


def save_incremental(result: ValidationResult, dataset: str) -> None:
    """Append a single case result to the checkpoint JSONL file."""
    path = checkpoint_path(dataset)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_result_to_dict(result), default=str) + "\n")


def load_checkpoint(dataset: str) -> List[ValidationResult]:
    """
    Load previously-completed results from the checkpoint file.

    Returns a list of ValidationResult objects (may be empty).
    """
    path = checkpoint_path(dataset)
    if not path.exists():
        return []

    results: List[ValidationResult] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        results.append(ValidationResult(
            case_id=d["case_id"],
            source_dataset=d.get("source_dataset", dataset),
            success=d["success"],
            scores=d["scores"],
            pipeline_time_ms=d.get("pipeline_time_ms", 0),
            step_results=d.get("step_results", {}),
            report_summary=d.get("report_summary"),
            error=d.get("error"),
            details=d.get("details", {}),
        ))
    return results


def clear_checkpoint(dataset: str) -> None:
    """Delete checkpoint file for a fresh run."""
    path = checkpoint_path(dataset)
    if path.exists():
        path.unlink()


# ──────────────────────────────────────────────
# Final results save
# ──────────────────────────────────────────────

def save_results(summary: ValidationSummary, filename: Optional[str] = None):
    """Save validation results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{summary.dataset}_{ts}.json"

    path = RESULTS_DIR / filename

    # Convert to serializable dict
    data = {
        "dataset": summary.dataset,
        "total_cases": summary.total_cases,
        "successful_cases": summary.successful_cases,
        "failed_cases": summary.failed_cases,
        "metrics": summary.metrics,
        "run_duration_sec": summary.run_duration_sec,
        "timestamp": summary.timestamp,
        "per_case": [_result_to_dict(r) for r in summary.per_case],
    }

    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def print_summary(summary: ValidationSummary):
    """Pretty-print validation results with stratified breakdown."""
    print(f"\n{'='*60}")
    print(f"  Validation Results: {summary.dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Total cases:      {summary.total_cases}")
    print(f"  Successful:       {summary.successful_cases}")
    print(f"  Failed:           {summary.failed_cases}")
    print(f"  Duration:         {summary.run_duration_sec:.1f}s")

    # Known question types used for stratification
    _KNOWN_TYPES = {
        "diagnostic", "treatment", "mechanism", "lab_finding",
        "pharmacology", "epidemiology", "ethics", "anatomy", "other",
    }

    # Overall metrics (exclude per-type breakdowns)
    print(f"\n  Overall Metrics:")
    for metric, value in sorted(summary.metrics.items()):
        if metric.startswith("count_"):
            continue
        if any(metric.endswith(f"_{qt}") for qt in _KNOWN_TYPES):
            continue
        if metric.endswith("_pipeline_appropriate"):
            continue
        if "time" in metric and isinstance(value, (int, float)):
            print(f"    {metric:35s} {value:.0f}ms")
        elif isinstance(value, float):
            print(f"    {metric:35s} {value:.1%}")
        else:
            print(f"    {metric:35s} {value}")

    # Stratified breakdown by question type
    type_keys = sorted(
        k[len("count_"):] for k in summary.metrics
        if k.startswith("count_") and k != "count_pipeline_appropriate"
    )
    if type_keys:
        print(f"\n  By Question Type:")
        print(f"    {'Type':15s} {'Count':>6s} {'Top-1':>7s} {'Top-3':>7s} {'Mentioned':>10s} {'MCQ':>7s}")
        print(f"    {'-'*15} {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*7}")
        for qt in type_keys:
            count = int(summary.metrics.get(f"count_{qt}", 0))
            t1 = summary.metrics.get(f"top1_accuracy_{qt}")
            t3 = summary.metrics.get(f"top3_accuracy_{qt}")
            ma = summary.metrics.get(f"mentioned_accuracy_{qt}")
            mcq = summary.metrics.get(f"mcq_accuracy_{qt}")
            t1_s = f"{t1:.0%}" if t1 is not None else "-"
            t3_s = f"{t3:.0%}" if t3 is not None else "-"
            ma_s = f"{ma:.0%}" if ma is not None else "-"
            mcq_s = f"{mcq:.0%}" if mcq is not None else "-"
            print(f"    {qt:15s} {count:6d} {t1_s:>7s} {t3_s:>7s} {ma_s:>10s} {mcq_s:>7s}")

    # Pipeline-appropriate subset
    pa_count = summary.metrics.get("count_pipeline_appropriate", 0)
    if pa_count > 0:
        print(f"\n  Pipeline-Appropriate Subset ({int(pa_count)} cases):")
        for m in ["top1_accuracy", "top3_accuracy", "mentioned_accuracy"]:
            v = summary.metrics.get(f"{m}_pipeline_appropriate")
            if v is not None:
                print(f"    {m:35s} {v:.1%}")

    print(f"{'='*60}\n")
