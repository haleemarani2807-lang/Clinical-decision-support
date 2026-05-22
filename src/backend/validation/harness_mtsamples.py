"""
MTSamples dataset fetcher and validation harness.

Downloads medical transcription samples and evaluates the CDS pipeline's
ability to parse diverse clinical note formats and reason across specialties.

Source: https://mtsamples.com (via GitHub mirrors)
Format: CSV with columns: description, medical_specialty, sample_name, transcription, keywords

Metrics:
  - parse_success_rate: Pipeline completed without crashing
  - field_completeness: How many structured fields were extracted
  - specialty_alignment: System reasoning aligns with correct specialty
  - has_differential: Report includes at least one diagnosis
  - has_recommendations: Report includes next steps
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import random
import re
import time
from pathlib import Path
from typing import List, Optional

import httpx

from validation.base import (
    DATA_DIR,
    ValidationCase,
    ValidationResult,
    ValidationSummary,
    clear_checkpoint,
    ensure_data_dir,
    fuzzy_match,
    load_checkpoint,
    normalize_text,
    print_summary,
    run_cds_pipeline,
    save_incremental,
    save_results,
)


# ──────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────

MTSAMPLES_URL = "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"
MTSAMPLES_FALLBACK_URL = "https://raw.githubusercontent.com/Abonia1/Clinical-NLP-on-MTSamples/master/mtsamples.csv"

# Specialties most relevant to CDS
RELEVANT_SPECIALTIES = {
    "Cardiovascular / Pulmonary",
    "Gastroenterology",
    "General Medicine",
    "Neurology",
    "Orthopedic",
    "Urology",
    "Nephrology",
    "Endocrinology",
    "Hematology - Oncology",
    "Obstetrics / Gynecology",
    "Emergency Room Reports",
    "Consult - History and Phy.",
    "Discharge Summary",
    "SOAP / Chart / Progress Notes",
    "Internal Medicine",
}


async def fetch_mtsamples(
    max_cases: int = 30,
    seed: int = 42,
    specialties: Optional[set] = None,
    min_length: int = 200,
) -> List[ValidationCase]:
    """
    Download MTSamples and convert to ValidationCase objects.

    Args:
        max_cases: Maximum number of cases to sample
        seed: Random seed for reproducible sampling
        specialties: Filter to these specialties (None = use RELEVANT_SPECIALTIES)
        min_length: Minimum transcription length to include
    """
    ensure_data_dir()
    cache_path = DATA_DIR / "mtsamples.csv"

    if cache_path.exists():
        print(f"  Loading MTSamples from cache: {cache_path}")
        raw_text = cache_path.read_text(encoding="utf-8")
    else:
        print(f"  Downloading MTSamples...")
        raw_text = await _download_mtsamples(cache_path)

    if not raw_text:
        raise RuntimeError("Failed to fetch MTSamples data.")

    # Parse CSV
    reader = csv.DictReader(io.StringIO(raw_text))
    rows = list(reader)

    # Filter
    target_specialties = specialties or RELEVANT_SPECIALTIES
    filtered = []
    for row in rows:
        specialty = row.get("medical_specialty", "").strip()
        transcription = row.get("transcription", "").strip()
        if not transcription or len(transcription) < min_length:
            continue
        if specialty in target_specialties:
            filtered.append(row)

    # Sample
    random.seed(seed)
    if len(filtered) > max_cases:
        # Stratified sample: try to get cases from diverse specialties
        by_specialty = {}
        for row in filtered:
            sp = row.get("medical_specialty", "Other")
            by_specialty.setdefault(sp, []).append(row)

        sampled = []
        per_specialty = max(1, max_cases // len(by_specialty))
        for sp, sp_rows in by_specialty.items():
            sampled.extend(random.sample(sp_rows, min(per_specialty, len(sp_rows))))

        # Fill remaining slots randomly
        remaining = [r for r in filtered if r not in sampled]
        if len(sampled) < max_cases and remaining:
            sampled.extend(random.sample(remaining, min(max_cases - len(sampled), len(remaining))))

        filtered = sampled[:max_cases]

    # Convert to ValidationCase
    cases = []
    for i, row in enumerate(filtered):
        transcription = row.get("transcription", "").strip()
        specialty = row.get("medical_specialty", "Unknown").strip()
        description = row.get("description", "").strip()
        keywords = row.get("keywords", "").strip()

        cases.append(ValidationCase(
            case_id=f"mts_{i:04d}",
            source_dataset="mtsamples",
            input_text=transcription,
            ground_truth={
                "specialty": specialty,
                "description": description,
                "keywords": keywords,
            },
            metadata={
                "sample_name": row.get("sample_name", ""),
                "text_length": len(transcription),
            },
        ))

    print(f"  Loaded {len(cases)} MTSamples cases across {len(set(c.ground_truth['specialty'] for c in cases))} specialties")
    return cases


async def _download_mtsamples(cache_path: Path) -> str:
    """Download MTSamples CSV."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for url in [MTSAMPLES_URL, MTSAMPLES_FALLBACK_URL]:
            try:
                r = await client.get(url)
                r.raise_for_status()
                cache_path.write_text(r.text, encoding="utf-8")
                print(f"  Cached MTSamples ({len(r.text)} bytes) to {cache_path}")
                return r.text
            except Exception as e:
                print(f"  Warning: Failed to download from {url}: {e}")
                continue
    return ""


# ──────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────

SPECIALTY_KEYWORDS = {
    "Cardiovascular / Pulmonary": ["cardiac", "heart", "coronary", "pulmonary", "lung", "chest", "hypertension", "arrhythmia"],
    "Gastroenterology": ["gastro", "liver", "hepat", "colon", "bowel", "gi ", "abdominal", "pancrea"],
    "General Medicine": ["general", "medicine", "primary", "routine"],
    "Neurology": ["neuro", "brain", "seizure", "stroke", "headache", "neuropathy", "ms "],
    "Orthopedic": ["ortho", "fracture", "bone", "joint", "knee", "hip", "shoulder", "spine"],
    "Urology": ["urol", "kidney", "bladder", "prostate", "renal", "urinary"],
    "Nephrology": ["renal", "kidney", "dialysis", "nephr", "creatinine"],
    "Endocrinology": ["diabet", "thyroid", "endocrin", "insulin", "glucose", "adrenal"],
    "Hematology - Oncology": ["cancer", "tumor", "leukemia", "lymphoma", "anemia", "oncol"],
    "Obstetrics / Gynecology": ["pregnan", "obstet", "gynecol", "uterus", "ovarian", "menstrual"],
    "Emergency Room Reports": ["emergency", "trauma", "acute", "er ", "ed "],
    "Internal Medicine": ["internal", "medicine"],
}


def check_specialty_alignment(report_text: str, target_specialty: str) -> bool:
    """Check if the report's content aligns with the expected specialty."""
    keywords = SPECIALTY_KEYWORDS.get(target_specialty, [])
    if not keywords:
        return True  # Can't check, assume aligned

    report_lower = report_text.lower()
    matches = sum(1 for kw in keywords if kw in report_lower)
    return matches >= 1  # At least one specialty keyword present


def score_field_completeness(state) -> float:
    """Score how many structured fields were successfully extracted from parsing."""
    if not state or not state.patient_profile:
        return 0.0

    profile = state.patient_profile
    fields = [
        profile.age is not None,
        profile.gender.value != "unknown",
        bool(profile.chief_complaint),
        bool(profile.history_of_present_illness),
        len(profile.past_medical_history) > 0,
        len(profile.current_medications) > 0,
        len(profile.allergies) > 0,
        len(profile.lab_results) > 0,
        profile.vital_signs is not None,
        bool(profile.social_history),
        bool(profile.family_history),
    ]
    return sum(fields) / len(fields)


# ──────────────────────────────────────────────
# Validation harness
# ──────────────────────────────────────────────

async def validate_mtsamples(
    cases: List[ValidationCase],
    include_drug_check: bool = True,
    include_guidelines: bool = True,
    delay_between_cases: float = 2.0,
    resume: bool = False,
) -> ValidationSummary:
    """
    Run MTSamples cases through the CDS pipeline and score results.
    """
    results: List[ValidationResult] = []
    start_time = time.time()

    # Resume support
    completed_ids: set = set()
    if resume:
        prior = load_checkpoint("mtsamples")
        if prior:
            results.extend(prior)
            completed_ids = {r.case_id for r in prior}
            print(f"  Resuming: {len(prior)} cases loaded from checkpoint, {len(cases) - len(completed_ids)} remaining")
    else:
        clear_checkpoint("mtsamples")

    for i, case in enumerate(cases):
        specialty = case.ground_truth.get("specialty", "?")
        if case.case_id in completed_ids:
            print(f"\n  [{i+1}/{len(cases)}] {case.case_id} ({specialty}): (cached) skipped")
            continue

        print(f"\n  [{i+1}/{len(cases)}] {case.case_id} ({specialty}): ", end="", flush=True)

        case_start = time.monotonic()

        state, report, error = await run_cds_pipeline(
            patient_text=case.input_text,
            include_drug_check=include_drug_check,
            include_guidelines=include_guidelines,
        )

        elapsed_ms = int((time.monotonic() - case_start) * 1000)

        # Step results
        step_results = {}
        if state:
            step_results = {s.step_id: s.status.value for s in state.steps}

        # Score
        scores = {}
        details = {}

        # Parse success
        scores["parse_success"] = 1.0 if (state and state.patient_profile) else 0.0

        # Field completeness
        scores["field_completeness"] = score_field_completeness(state)

        if report:
            # Has differential
            scores["has_differential"] = 1.0 if len(report.differential_diagnosis) > 0 else 0.0

            # Has recommendations
            scores["has_recommendations"] = 1.0 if len(report.suggested_next_steps) > 0 else 0.0

            # Has guideline recommendations
            scores["has_guidelines"] = 1.0 if len(report.guideline_recommendations) > 0 else 0.0

            # Specialty alignment
            full_report_text = " ".join([
                report.patient_summary or "",
                " ".join(d.diagnosis for d in report.differential_diagnosis),
                " ".join(report.guideline_recommendations),
                " ".join(a.action for a in report.suggested_next_steps),
            ])
            scores["specialty_alignment"] = 1.0 if check_specialty_alignment(
                full_report_text, specialty
            ) else 0.0

            # Conflict detection worked (if applicable)
            if state and state.conflict_detection:
                scores["conflict_detection_ran"] = 1.0
            else:
                scores["conflict_detection_ran"] = 0.0

            details = {
                "specialty": specialty,
                "num_diagnoses": len(report.differential_diagnosis),
                "num_recommendations": len(report.suggested_next_steps),
                "field_completeness": scores["field_completeness"],
                "num_conflicts": len(report.conflicts) if report.conflicts else 0,
            }

            print(f"+ fields={scores['field_completeness']:.0%} dx={len(report.differential_diagnosis)} ({elapsed_ms}ms)")
        else:
            scores.update({
                "has_differential": 0.0,
                "has_recommendations": 0.0,
                "has_guidelines": 0.0,
                "specialty_alignment": 0.0,
                "conflict_detection_ran": 0.0,
            })
            details = {"specialty": specialty, "error": error}
            print(f"- FAILED: {error[:80] if error else 'unknown'}")

        result = ValidationResult(
            case_id=case.case_id,
            source_dataset="mtsamples",
            success=report is not None,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            step_results=step_results,
            report_summary=report.patient_summary[:200] if report else None,
            error=error,
            details=details,
        )
        results.append(result)
        save_incremental(result, "mtsamples")  # checkpoint after every case

        if i < len(cases) - 1:
            await asyncio.sleep(delay_between_cases)

    # Aggregate
    total = len(results)
    successful = sum(1 for r in results if r.success)

    metric_names = [
        "parse_success", "field_completeness", "has_differential",
        "has_recommendations", "has_guidelines", "specialty_alignment",
        "conflict_detection_ran",
    ]
    metrics = {}
    for m in metric_names:
        values = [r.scores.get(m, 0.0) for r in results]
        metrics[m] = sum(values) / len(values) if values else 0.0

    times = [r.pipeline_time_ms for r in results if r.success]
    metrics["avg_pipeline_time_ms"] = sum(times) / len(times) if times else 0

    summary = ValidationSummary(
        dataset="mtsamples",
        total_cases=total,
        successful_cases=successful,
        failed_cases=total - successful,
        metrics=metrics,
        per_case=results,
        run_duration_sec=time.time() - start_time,
    )

    return summary


# ──────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────

async def main():
    """Run MTSamples validation standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="MTSamples Validation")
    parser.add_argument("--max-cases", type=int, default=10, help="Number of cases to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-drugs", action="store_true", help="Skip drug interaction check")
    parser.add_argument("--no-guidelines", action="store_true", help="Skip guideline retrieval")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between cases (seconds)")
    args = parser.parse_args()

    print("MTSamples Validation Harness")
    print("=" * 40)

    cases = await fetch_mtsamples(max_cases=args.max_cases, seed=args.seed)
    summary = await validate_mtsamples(
        cases,
        include_drug_check=not args.no_drugs,
        include_guidelines=not args.no_guidelines,
        delay_between_cases=args.delay,
    )

    print_summary(summary)
    path = save_results(summary)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())
