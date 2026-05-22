"""
Upgrade benchmark — A/B compare two MedGemma endpoints (baseline vs candidate)
on a sample of MedQA cases.

Used to gate model upgrades: run before swapping the production endpoint to
verify the candidate is at least as accurate and not significantly slower.

How it works
------------
The CDS pipeline tools each construct their own MedGemmaService(), which reads
from app.config.settings at instantiation time. To run two configurations in
the same process we mutate `settings.medgemma_*` between runs (single-threaded;
each case completes before settings are swapped for the next side).

Usage
-----
    python upgrade_benchmark.py \
        --baseline-url https://baseline.endpoint/v1 --baseline-key sk-base \
        --candidate-url https://candidate.endpoint/v1 --candidate-key sk-cand \
        --baseline-model tgi --candidate-model tgi \
        --max-cases 20 --seed 42 \
        --out benchmark_results.json

Exit codes
----------
    0 — candidate accuracy >= baseline AND latency regression <= --max-latency-regression
    1 — candidate fails one or more gates
    2 — execution error (no cases scored on either side)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from app.config import settings
from validation.base import (
    ValidationCase,
    run_cds_pipeline,
    score_case,
)
from validation.harness_medqa import fetch_medqa

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("upgrade_benchmark")
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    success: bool
    latency_ms: float
    top1: float = 0.0
    top3: float = 0.0
    mentioned: float = 0.0
    error: Optional[str] = None


@dataclass
class SideSummary:
    label: str
    base_url: str
    model_id: str
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def successful(self) -> List[CaseResult]:
        return [c for c in self.cases if c.success]

    @property
    def success_rate(self) -> float:
        return len(self.successful) / len(self.cases) if self.cases else 0.0

    @property
    def avg_latency_ms(self) -> float:
        succ = self.successful
        return statistics.mean(c.latency_ms for c in succ) if succ else 0.0

    @property
    def p95_latency_ms(self) -> float:
        succ = self.successful
        if not succ:
            return 0.0
        sorted_lat = sorted(c.latency_ms for c in succ)
        idx = max(0, int(round(0.95 * len(sorted_lat))) - 1)
        return sorted_lat[idx]

    @property
    def avg_top1(self) -> float:
        succ = self.successful
        return statistics.mean(c.top1 for c in succ) if succ else 0.0

    @property
    def avg_top3(self) -> float:
        succ = self.successful
        return statistics.mean(c.top3 for c in succ) if succ else 0.0

    @property
    def avg_mentioned(self) -> float:
        succ = self.successful
        return statistics.mean(c.mentioned for c in succ) if succ else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "base_url": self.base_url,
            "model_id": self.model_id,
            "n_cases": len(self.cases),
            "n_successful": len(self.successful),
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "avg_top1": round(self.avg_top1, 4),
            "avg_top3": round(self.avg_top3, 4),
            "avg_mentioned": round(self.avg_mentioned, 4),
            "cases": [asdict(c) for c in self.cases],
        }


# ──────────────────────────────────────────────
# Side runner
# ──────────────────────────────────────────────

async def _run_side(
    label: str,
    base_url: str,
    api_key: str,
    model_id: str,
    cases: List[ValidationCase],
    timeout_sec: int,
) -> SideSummary:
    """Mutate settings to point at this endpoint, then run all cases."""
    print(f"\n{'=' * 60}")
    print(f"Running {label}: {model_id} @ {base_url}")
    print(f"{'=' * 60}")

    # Mutate settings — tool instances created during run_cds_pipeline will pick these up
    settings.medgemma_base_url = base_url
    settings.medgemma_api_key = api_key
    settings.medgemma_model_id = model_id

    summary = SideSummary(label=label, base_url=base_url, model_id=model_id)

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.case_id} ... ", end="", flush=True)
        t0 = time.perf_counter()
        _state, report, error = await run_cds_pipeline(
            case.input_text,
            include_drug_check=True,
            include_guidelines=True,
            timeout_sec=timeout_sec,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if error or report is None:
            print(f"FAIL ({elapsed_ms:.0f}ms): {error or 'no report'}")
            summary.cases.append(CaseResult(
                case_id=case.case_id,
                success=False,
                latency_ms=elapsed_ms,
                error=error or "no report",
            ))
            continue

        target = case.ground_truth.get("correct_answer", "")
        scores = score_case(target, report, question_type="diagnostic")
        cr = CaseResult(
            case_id=case.case_id,
            success=True,
            latency_ms=elapsed_ms,
            top1=scores.get("top1_accuracy", 0.0),
            top3=scores.get("top3_accuracy", 0.0),
            mentioned=scores.get("mentioned_accuracy", 0.0),
        )
        summary.cases.append(cr)
        print(f"OK ({elapsed_ms:.0f}ms) top1={cr.top1:.0f} top3={cr.top3:.0f}")

    return summary


# ──────────────────────────────────────────────
# Comparison + gate
# ──────────────────────────────────────────────

def _print_comparison(baseline: SideSummary, candidate: SideSummary) -> None:
    print()
    print("=" * 60)
    print("Baseline vs Candidate")
    print("=" * 60)
    rows = [
        ("metric", "baseline", "candidate", "delta"),
        ("success_rate", baseline.success_rate, candidate.success_rate,
         candidate.success_rate - baseline.success_rate),
        ("avg_top1", baseline.avg_top1, candidate.avg_top1,
         candidate.avg_top1 - baseline.avg_top1),
        ("avg_top3", baseline.avg_top3, candidate.avg_top3,
         candidate.avg_top3 - baseline.avg_top3),
        ("avg_mentioned", baseline.avg_mentioned, candidate.avg_mentioned,
         candidate.avg_mentioned - baseline.avg_mentioned),
        ("avg_latency_ms", baseline.avg_latency_ms, candidate.avg_latency_ms,
         candidate.avg_latency_ms - baseline.avg_latency_ms),
        ("p95_latency_ms", baseline.p95_latency_ms, candidate.p95_latency_ms,
         candidate.p95_latency_ms - baseline.p95_latency_ms),
    ]
    for row in rows:
        if isinstance(row[1], str):
            print(f"  {row[0]:<18} {row[1]:>12} {row[2]:>12} {row[3]:>12}")
        else:
            print(f"  {row[0]:<18} {row[1]:>12.4f} {row[2]:>12.4f} {row[3]:>+12.4f}")


def _gate(
    baseline: SideSummary,
    candidate: SideSummary,
    min_success_rate: float,
    max_accuracy_regression: float,
    max_latency_regression_pct: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    if candidate.success_rate < min_success_rate:
        failures.append(
            f"candidate success_rate {candidate.success_rate:.2%} "
            f"< minimum {min_success_rate:.2%}"
        )

    acc_delta = candidate.avg_top1 - baseline.avg_top1
    if acc_delta < -max_accuracy_regression:
        failures.append(
            f"top1 regression {acc_delta:+.4f} exceeds threshold "
            f"-{max_accuracy_regression:.4f}"
        )

    if baseline.avg_latency_ms > 0:
        lat_pct = (candidate.avg_latency_ms - baseline.avg_latency_ms) / baseline.avg_latency_ms
        if lat_pct > max_latency_regression_pct:
            failures.append(
                f"latency regression {lat_pct:+.1%} exceeds threshold "
                f"+{max_latency_regression_pct:.1%}"
            )

    return (len(failures) == 0, failures)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark two MedGemma endpoints")
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--baseline-key", default="")
    parser.add_argument("--baseline-model", default="tgi")
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--candidate-key", default="")
    parser.add_argument("--candidate-model", default="tgi")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--min-success-rate", type=float, default=0.80,
                        help="Candidate must achieve at least this success rate (default 0.80)")
    parser.add_argument("--max-accuracy-regression", type=float, default=0.05,
                        help="Max allowed drop in top1 accuracy (default 0.05 = 5pp)")
    parser.add_argument("--max-latency-regression", type=float, default=0.20,
                        help="Max allowed avg latency increase (default 0.20 = 20%%)")
    parser.add_argument("--out", default="upgrade_benchmark_results.json")
    args = parser.parse_args()

    # Snapshot original settings so we can restore them
    orig_url = settings.medgemma_base_url
    orig_key = settings.medgemma_api_key
    orig_model = settings.medgemma_model_id

    print(f"Fetching {args.max_cases} MedQA cases (seed={args.seed})...")
    cases = await fetch_medqa(max_cases=args.max_cases, seed=args.seed)
    print(f"Loaded {len(cases)} cases.")

    try:
        baseline = await _run_side(
            "baseline", args.baseline_url, args.baseline_key, args.baseline_model,
            cases, args.timeout_sec,
        )
        candidate = await _run_side(
            "candidate", args.candidate_url, args.candidate_key, args.candidate_model,
            cases, args.timeout_sec,
        )
    finally:
        settings.medgemma_base_url = orig_url
        settings.medgemma_api_key = orig_key
        settings.medgemma_model_id = orig_model

    if not baseline.cases or not candidate.cases:
        print("ERROR: no cases were run on at least one side", file=sys.stderr)
        return 2

    _print_comparison(baseline, candidate)

    passed, failures = _gate(
        baseline, candidate,
        min_success_rate=args.min_success_rate,
        max_accuracy_regression=args.max_accuracy_regression,
        max_latency_regression_pct=args.max_latency_regression,
    )

    print()
    print("=" * 60)
    print("Upgrade gate")
    print("=" * 60)
    if passed:
        print("  PASS  candidate is suitable for promotion")
    else:
        print("  FAIL  candidate did not meet all gates:")
        for f in failures:
            print(f"        - {f}")

    payload = {
        "args": vars(args),
        "gate_passed": passed,
        "gate_failures": failures,
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
