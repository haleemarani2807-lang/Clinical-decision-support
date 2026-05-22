"""
Unified validation runner for the Clinical Decision Support Agent.

Runs dataset validations (MedQA, MedQA-5, MedMCQA, PubMedQA, MMLU-Medical,
HeadQA, MTSamples, PMC Case Reports), adversarial edge-case tests, and
regression tests.

Usage:
    # From src/backend directory:
    python -m validation.run_validation --all --max-cases 10
    python -m validation.run_validation --medqa --max-cases 20
    python -m validation.run_validation --medqa5 --max-cases 50
    python -m validation.run_validation --medmcqa --max-cases 50
    python -m validation.run_validation --pubmedqa --max-cases 50
    python -m validation.run_validation --mmlu --max-cases 50
    python -m validation.run_validation --headqa --max-cases 50
    python -m validation.run_validation --mtsamples --max-cases 15
    python -m validation.run_validation --pmc --max-cases 10
    python -m validation.run_validation --adversarial
    python -m validation.run_validation --regression

    # Fetch data only (no pipeline execution):
    python -m validation.run_validation --fetch-only
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to avoid cp1252 encoding errors
# when LLM output contains Unicode characters (e.g., Greek letters in diagnoses)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure backend is importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load .env and export HF_TOKEN so huggingface_hub picks it up
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

from validation.base import (
    ValidationSummary,
    print_summary,
    save_results,
)
from validation.harness_medqa import fetch_medqa, validate_medqa
from validation.harness_medqa5 import fetch_medqa5, validate_medqa5
from validation.harness_medmcqa import fetch_medmcqa, validate_medmcqa
from validation.harness_pubmedqa import fetch_pubmedqa, validate_pubmedqa
from validation.harness_mmlu_medical import fetch_mmlu_medical, validate_mmlu_medical
from validation.harness_headqa import fetch_headqa, validate_headqa
from validation.harness_mtsamples import fetch_mtsamples, validate_mtsamples
from validation.harness_pmc import fetch_pmc_cases, validate_pmc
from validation.test_adversarial import validate_adversarial
from validation.test_regression import run_regression_suite


async def run_all_validations(
    run_medqa: bool = True,
    run_medqa5: bool = False,
    run_mtsamples: bool = True,
    run_pmc: bool = True,
    run_medmcqa: bool = False,
    run_pubmedqa: bool = False,
    run_mmlu: bool = False,
    run_headqa: bool = False,
    run_adversarial: bool = False,
    run_regression: bool = False,
    max_cases: int = 10,
    seed: int = 42,
    include_drug_check: bool = True,
    include_guidelines: bool = True,
    delay: float = 2.0,
    fetch_only: bool = False,
    resume: bool = False,
) -> dict:
    """
    Run validation against selected datasets.

    Returns dict of {dataset_name: ValidationSummary}
    """
    results = {}
    start = time.time()

    # ── MedQA ──
    if run_medqa:
        print("\n" + "=" * 60)
        print("  DATASET 1: MedQA (USMLE-style diagnostic accuracy)")
        print("=" * 60)

        cases = await fetch_medqa(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} MedQA cases (fetch-only mode)")
        else:
            summary = await validate_medqa(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
                resume=resume,
            )
            print_summary(summary)
            save_results(summary)
            results["medqa"] = summary

    # ── MedQA-5 (5-option variant) ──
    if run_medqa5:
        print("\n" + "=" * 60)
        print("  DATASET: MedQA-5 (5-option USMLE-style, harder baseline)")
        print("=" * 60)

        cases = await fetch_medqa5(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} MedQA-5 cases (fetch-only mode)")
        else:
            summary = await validate_medqa5(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
                resume=resume,
            )
            print_summary(summary)
            save_results(summary)
            results["medqa5"] = summary

    # ── MTSamples ──
    if run_mtsamples:
        print("\n" + "=" * 60)
        print("  DATASET 2: MTSamples (clinical note parsing robustness)")
        print("=" * 60)

        cases = await fetch_mtsamples(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} MTSamples cases (fetch-only mode)")
        else:
            summary = await validate_mtsamples(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
                resume=resume,
            )
            print_summary(summary)
            save_results(summary)
            results["mtsamples"] = summary

    # ── PMC Case Reports ──
    if run_pmc:
        print("\n" + "=" * 60)
        print("  DATASET 3: PMC Case Reports (real-world diagnostic accuracy)")
        print("=" * 60)

        cases = await fetch_pmc_cases(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} PMC cases (fetch-only mode)")
        else:
            summary = await validate_pmc(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
                resume=resume,
            )
            print_summary(summary)
            save_results(summary)
            results["pmc"] = summary

    # ── MedMCQA ──
    if run_medmcqa:
        print("\n" + "=" * 60)
        print("  DATASET 4: MedMCQA (Indian medical entrance exam accuracy)")
        print("=" * 60)

        cases = await fetch_medmcqa(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} MedMCQA cases (fetch-only mode)")
        else:
            summary = await validate_medmcqa(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
                resume=resume,
            )
            print_summary(summary)
            save_results(summary)
            results["medmcqa"] = summary

    # ── PubMedQA ──
    if run_pubmedqa:
        print("\n" + "=" * 60)
        print("  DATASET 5: PubMedQA (biomedical yes/no/maybe reasoning)")
        print("=" * 60)

        cases = await fetch_pubmedqa(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} PubMedQA cases (fetch-only mode)")
        else:
            summary = await validate_pubmedqa(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
                resume=resume,
            )
            print_summary(summary)
            save_results(summary)
            results["pubmedqa"] = summary

    # ── MMLU-Medical ──
    if run_mmlu:
        print("\n" + "=" * 60)
        print("  DATASET: MMLU-Medical (6 medical subtasks)")
        print("=" * 60)

        cases = await fetch_mmlu_medical(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} MMLU-Medical cases (fetch-only mode)")
        else:
            summary = await validate_mmlu_medical(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
                resume=resume,
            )
            print_summary(summary)
            save_results(summary)
            results["mmlu_medical"] = summary

    # ── HeadQA ──
    if run_headqa:
        print("\n" + "=" * 60)
        print("  DATASET: HeadQA (Spanish healthcare exams, English)")
        print("=" * 60)

        cases = await fetch_headqa(max_cases=max_cases, seed=seed)

        if fetch_only:
            print(f"  Fetched {len(cases)} HeadQA cases (fetch-only mode)")
        else:
            summary = await validate_headqa(
                cases,
                include_drug_check=include_drug_check,
                include_guidelines=include_guidelines,
                delay_between_cases=delay,
                resume=resume,
            )
            print_summary(summary)
            save_results(summary)
            results["headqa"] = summary

    # ── Adversarial Tests ──
    if run_adversarial and not fetch_only:
        print("\n" + "=" * 60)
        print("  ADVERSARIAL: Edge case & robustness tests")
        print("=" * 60)

        summary = await validate_adversarial(
            include_drug_check=include_drug_check,
            include_guidelines=include_guidelines,
            delay_between_cases=delay,
        )
        print_summary(summary)
        save_results(summary)
        results["adversarial"] = summary

    # ── Regression Tests ──
    if run_regression and not fetch_only:
        print("\n" + "=" * 60)
        print("  REGRESSION: Known failure case tests")
        print("=" * 60)

        passed, total, failures = await run_regression_suite()
        print(f"\n  Regression: {passed}/{total} passed")
        if failures:
            for f in failures:
                print(f"    FAIL: {f}")
        results["regression"] = {"passed": passed, "total": total, "failures": failures}

    # ── Combined Summary ──
    total_duration = time.time() - start

    if results and not fetch_only:
        _print_combined_summary(results, total_duration)
        _save_combined_report(results, total_duration)

    return results


def _print_combined_summary(results: dict, total_duration: float):
    """Print a combined summary across all datasets."""
    print("\n" + "=" * 70)
    print("  COMBINED VALIDATION REPORT")
    print("=" * 70)

    # Header
    print(f"\n  {'Dataset':<15} {'Cases':>6} {'Success':>8} {'Key Metric':>25} {'Value':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*25} {'-'*8}")

    for name, summary in results.items():
        # Skip regression (different format)
        if name == "regression":
            reg = summary
            print(
                f"  {'regression':<15} {reg['total']:>6} "
                f"{reg['passed']:>8} "
                f"{'pass_rate':>25} {reg['passed']/max(reg['total'],1):>7.1%}"
            )
            continue

        # Pick the most important metric for each dataset
        if name == "medqa":
            key_metric = "top3_accuracy"
        elif name == "mtsamples":
            key_metric = "parse_success"
        elif name == "pmc":
            key_metric = "diagnostic_accuracy"
        elif name == "medmcqa":
            key_metric = "top3_accuracy"
        elif name == "pubmedqa":
            key_metric = "decision_accuracy"
        elif name == "medqa5":
            key_metric = "top3_accuracy"
        elif name == "mmlu_medical":
            key_metric = "mcq_accuracy"
        elif name == "headqa":
            key_metric = "mcq_accuracy"
        elif name == "adversarial":
            key_metric = "pass_rate"
        else:
            key_metric = list(summary.metrics.keys())[0] if summary.metrics else "N/A"

        value = summary.metrics.get(key_metric, 0.0)
        print(
            f"  {name:<15} {summary.total_cases:>6} "
            f"{summary.successful_cases:>8} "
            f"{key_metric:>25} {value:>7.1%}"
        )

    # All metrics
    print(f"\n  {'-' * 66}")
    for name, summary in results.items():
        if name == "regression":
            print(f"\n  REGRESSION metrics:")
            print(f"    {'passed':<35} {summary['passed']}")
            print(f"    {'total':<35} {summary['total']}")
            if summary['failures']:
                print(f"    {'failures':<35} {', '.join(summary['failures'])}")
            continue
        print(f"\n  {name.upper()} metrics:")
        for metric, value in sorted(summary.metrics.items()):
            if "time" in metric and isinstance(value, (int, float)):
                print(f"    {metric:<35} {value:.0f}ms")
            elif isinstance(value, float):
                print(f"    {metric:<35} {value:.1%}")

    # Totals
    total_cases = sum(
        s.total_cases if hasattr(s, 'total_cases') else s.get('total', 0)
        for s in results.values()
    )
    total_success = sum(
        s.successful_cases if hasattr(s, 'successful_cases') else s.get('passed', 0)
        for s in results.values()
    )
    print(f"\n  Total cases:     {total_cases}")
    print(f"  Total success:   {total_success}")
    print(f"  Total duration:  {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"  Timestamp:       {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)


def _save_combined_report(results: dict, total_duration: float):
    """Save combined report to JSON."""
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"combined_{ts}.json"

    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_duration_sec": total_duration,
        "datasets": {},
    }

    for name, summary in results.items():
        if name == "regression":
            combined["datasets"][name] = summary
            continue
        combined["datasets"][name] = {
            "total_cases": summary.total_cases,
            "successful_cases": summary.successful_cases,
            "failed_cases": summary.failed_cases,
            "metrics": summary.metrics,
            "run_duration_sec": summary.run_duration_sec,
        }

    path.write_text(json.dumps(combined, indent=2, default=str))
    print(f"\n  Combined report saved to: {path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CDS Agent Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m validation.run_validation --all --max-cases 10
  python -m validation.run_validation --medqa --max-cases 50
  python -m validation.run_validation --medqa5 --max-cases 50
  python -m validation.run_validation --medmcqa --max-cases 50
  python -m validation.run_validation --pubmedqa --max-cases 50
  python -m validation.run_validation --mmlu --max-cases 50
  python -m validation.run_validation --headqa --max-cases 50
  python -m validation.run_validation --adversarial
  python -m validation.run_validation --regression
  python -m validation.run_validation --fetch-only
  python -m validation.run_validation --medqa --pmc --max-cases 20 --no-drugs
        """,
    )

    # Dataset selection
    data_group = parser.add_argument_group("Datasets")
    data_group.add_argument("--all", action="store_true", help="Run all benchmark datasets")
    data_group.add_argument("--medqa", action="store_true", help="Run MedQA (USMLE 4-option) validation")
    data_group.add_argument("--medqa5", action="store_true", help="Run MedQA-5 (USMLE 5-option) validation")
    data_group.add_argument("--medmcqa", action="store_true", help="Run MedMCQA (Indian medical entrance) validation")
    data_group.add_argument("--pubmedqa", action="store_true", help="Run PubMedQA (biomedical yes/no/maybe) validation")
    data_group.add_argument("--mmlu", action="store_true", help="Run MMLU-Medical (6 medical subtasks) validation")
    data_group.add_argument("--headqa", action="store_true", help="Run HeadQA (Spanish healthcare exams) validation")
    data_group.add_argument("--mtsamples", action="store_true", help="Run MTSamples validation")
    data_group.add_argument("--pmc", action="store_true", help="Run PMC Case Reports validation")
    data_group.add_argument("--adversarial", action="store_true", help="Run adversarial edge-case tests")
    data_group.add_argument("--regression", action="store_true", help="Run regression test suite")

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--max-cases", type=int, default=10, help="Cases per dataset (default: 10)")
    config_group.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    config_group.add_argument("--delay", type=float, default=2.0, help="Delay between cases in seconds (default: 2.0)")
    config_group.add_argument("--no-drugs", action="store_true", help="Skip drug interaction checks")
    config_group.add_argument("--no-guidelines", action="store_true", help="Skip guideline retrieval")
    config_group.add_argument("--resume", action="store_true", help="Resume from checkpoint (skip already-completed cases)")
    config_group.add_argument("--fetch-only", action="store_true", help="Only download data, don't run pipeline")

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.all, args.medqa, args.medqa5, args.medmcqa, args.pubmedqa,
                args.mmlu, args.headqa, args.mtsamples, args.pmc,
                args.adversarial, args.regression]):
        args.all = True

    run_medqa = args.all or args.medqa
    run_medqa5 = args.all or args.medqa5
    run_mtsamples = args.all or args.mtsamples
    run_pmc = args.all or args.pmc
    run_medmcqa = args.all or args.medmcqa
    run_pubmedqa = args.all or args.pubmedqa
    run_mmlu = args.all or args.mmlu
    run_headqa = args.all or args.headqa
    run_adversarial_flag = args.adversarial  # --all does NOT auto-run adversarial/regression
    run_regression_flag = args.regression

    datasets_str = ""
    for name, active in [
        ("MedQA", run_medqa), ("MedQA-5", run_medqa5),
        ("MedMCQA", run_medmcqa), ("PubMedQA", run_pubmedqa),
        ("MMLU-Medical", run_mmlu), ("HeadQA", run_headqa),
        ("MTSamples", run_mtsamples), ("PMC", run_pmc),
        ("Adversarial", run_adversarial_flag),
        ("Regression", run_regression_flag),
    ]:
        if active:
            datasets_str += name + " "

    print("=" * 58)
    print("   Clinical Decision Support Agent - Validation Suite")
    print("=" * 58)
    print(f"\n  Datasets:     {datasets_str.strip()}")
    print(f"  Cases/dataset: {args.max_cases}")
    print(f"  Drug check:    {'Yes' if not args.no_drugs else 'No'}")
    print(f"  Guidelines:    {'Yes' if not args.no_guidelines else 'No'}")
    print(f"  Resume:        {'Yes' if args.resume else 'No'}")
    print(f"  Fetch only:    {'Yes' if args.fetch_only else 'No'}")

    asyncio.run(run_all_validations(
        run_medqa=run_medqa,
        run_medqa5=run_medqa5,
        run_mtsamples=run_mtsamples,
        run_pmc=run_pmc,
        run_medmcqa=run_medmcqa,
        run_pubmedqa=run_pubmedqa,
        run_mmlu=run_mmlu,
        run_headqa=run_headqa,
        run_adversarial=run_adversarial_flag,
        run_regression=run_regression_flag,
        max_cases=args.max_cases,
        seed=args.seed,
        include_drug_check=not args.no_drugs,
        include_guidelines=not args.no_guidelines,
        delay=args.delay,
        fetch_only=args.fetch_only,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
