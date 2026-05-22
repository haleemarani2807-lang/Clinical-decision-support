"""
PMC Case Reports fetcher and validation harness.

Fetches published clinical case reports from PubMed Central and evaluates
the CDS pipeline's diagnostic accuracy against gold-standard diagnoses.

Source: NCBI PubMed / PubMed Central (E-utilities API)
Format: XML abstracts with case presentations and final diagnoses

Metrics:
  - diagnostic_accuracy: Correct diagnosis appears in differential
  - top3_accuracy: Correct diagnosis in top 3
  - parse_success_rate: Pipeline completed without crashing
  - has_recommendations: Report includes actionable next steps
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from validation.base import (
    DATA_DIR,
    ValidationCase,
    ValidationResult,
    ValidationSummary,
    clear_checkpoint,
    diagnosis_in_differential,
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
# NCBI E-utilities configuration
# ──────────────────────────────────────────────

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

# Curated search queries for case reports with clear diagnoses
# Each tuple: (search_query, expected_specialty)
CASE_REPORT_QUERIES = [
    ('"case report"[Title] AND "myocardial infarction"[Title] AND diagnosis', "Cardiology"),
    ('"case report"[Title] AND "pneumonia"[Title] AND diagnosis', "Pulmonology"),
    ('"case report"[Title] AND "diabetic ketoacidosis"[Title]', "Endocrinology"),
    ('"case report"[Title] AND "stroke"[Title] AND diagnosis', "Neurology"),
    ('"case report"[Title] AND "appendicitis"[Title] AND diagnosis', "Surgery"),
    ('"case report"[Title] AND "pulmonary embolism"[Title]', "Pulmonology"),
    ('"case report"[Title] AND "sepsis"[Title] AND management', "Critical Care"),
    ('"case report"[Title] AND "heart failure"[Title] AND management', "Cardiology"),
    ('"case report"[Title] AND "pancreatitis"[Title] AND diagnosis', "Gastroenterology"),
    ('"case report"[Title] AND "meningitis"[Title] AND diagnosis', "Neurology/ID"),
    ('"case report"[Title] AND "urinary tract infection"[Title]', "Urology/ID"),
    ('"case report"[Title] AND "thyroid"[Title] AND "nodule"', "Endocrinology"),
    ('"case report"[Title] AND "deep vein thrombosis"[Title]', "Hematology"),
    ('"case report"[Title] AND "anaphylaxis"[Title]', "Allergy/EM"),
    ('"case report"[Title] AND "renal failure"[Title] AND acute', "Nephrology"),
    ('"case report"[Title] AND "liver cirrhosis"[Title]', "Hepatology"),
    ('"case report"[Title] AND "asthma"[Title] AND exacerbation', "Pulmonology"),
    ('"case report"[Title] AND "seizure"[Title] AND diagnosis', "Neurology"),
    ('"case report"[Title] AND "hypoglycemia"[Title]', "Endocrinology"),
    ('"case report"[Title] AND "gastrointestinal bleeding"[Title]', "Gastroenterology"),
]


async def fetch_pmc_cases(
    max_cases: int = 20,
    seed: int = 42,
) -> List[ValidationCase]:
    """
    Fetch case reports from PubMed and convert to ValidationCase objects.

    Uses PubMed E-utilities to search for case reports with clear diagnoses,
    then extracts the clinical presentation and diagnosis from abstracts.

    Args:
        max_cases: Maximum number of cases to fetch
        seed: Random seed for reproducible selection
    """
    ensure_data_dir()
    cache_path = DATA_DIR / "pmc_cases.json"

    if cache_path.exists():
        print(f"  Loading PMC cases from cache: {cache_path}")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cases = [ValidationCase(**c) for c in cached]
        if len(cases) >= max_cases:
            random.seed(seed)
            return random.sample(cases, min(max_cases, len(cases)))
        # Fall through to fetch more if cache is insufficient

    print(f"  Fetching case reports from PubMed...")
    cases = await _fetch_from_pubmed(max_cases, seed)

    if cases:
        # Cache
        cached_data = [
            {
                "case_id": c.case_id,
                "source_dataset": c.source_dataset,
                "input_text": c.input_text,
                "ground_truth": c.ground_truth,
                "metadata": c.metadata,
            }
            for c in cases
        ]
        cache_path.write_text(json.dumps(cached_data, indent=2), encoding="utf-8")
        print(f"  Cached {len(cases)} PMC cases to {cache_path}")

    print(f"  Loaded {len(cases)} PMC case reports")
    return cases


async def _fetch_from_pubmed(max_cases: int, seed: int) -> List[ValidationCase]:
    """Fetch case reports via PubMed E-utilities."""
    cases = []
    random.seed(seed)
    queries = random.sample(CASE_REPORT_QUERIES, min(max_cases, len(CASE_REPORT_QUERIES)))

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for query_text, specialty in queries:
            if len(cases) >= max_cases:
                break

            try:
                # Step 1: Search for PMIDs
                pmids = await _esearch(client, query_text, retmax=3)
                if not pmids:
                    continue

                # Step 2: Fetch abstracts
                for pmid in pmids[:1]:  # Take first result per query
                    abstract_data = await _efetch_abstract(client, pmid)
                    if not abstract_data:
                        continue

                    title, abstract = abstract_data

                    # Step 3: Extract case presentation and diagnosis
                    presentation, diagnosis = _extract_case_and_diagnosis(title, abstract, query_text)
                    if not presentation or not diagnosis:
                        continue

                    cases.append(ValidationCase(
                        case_id=f"pmc_{pmid}",
                        source_dataset="pmc",
                        input_text=presentation,
                        ground_truth={
                            "diagnosis": diagnosis,
                            "specialty": specialty,
                            "title": title,
                        },
                        metadata={
                            "pmid": pmid,
                            "full_abstract": abstract,
                        },
                    ))

                    if len(cases) >= max_cases:
                        break

                # NCBI rate limit: max 3 requests/second without API key
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"  Warning: Query failed '{query_text[:40]}...': {e}")
                continue

    return cases


async def _esearch(client: httpx.AsyncClient, query: str, retmax: int = 3) -> List[str]:
    """Search PubMed and return PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    r = await client.get(ESEARCH_URL, params=params)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


async def _efetch_abstract(client: httpx.AsyncClient, pmid: str) -> Optional[Tuple[str, str]]:
    """Fetch the title and abstract for a PMID."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
    }
    r = await client.get(EFETCH_URL, params=params)
    r.raise_for_status()

    try:
        root = ET.fromstring(r.text)

        # Extract title
        title_el = root.find(".//ArticleTitle")
        title = title_el.text if title_el is not None and title_el.text else ""

        # Extract abstract
        abstract_parts = []
        for abs_text in root.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            # Collect tail text from sub-elements
            full_text = (abs_text.text or "") + "".join(
                (child.text or "") + (child.tail or "") for child in abs_text
            )
            if label:
                abstract_parts.append(f"{label}: {full_text.strip()}")
            else:
                abstract_parts.append(full_text.strip())

        abstract = " ".join(abstract_parts)

        if len(abstract) < 100:
            return None

        return title, abstract

    except ET.ParseError:
        return None


def _extract_case_and_diagnosis(
    title: str, abstract: str, search_query: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the clinical presentation and final diagnosis from a case report abstract.

    Strategy:
    1. Try structured abstract sections (CASE PRESENTATION, DIAGNOSIS, etc.)
    2. Extract diagnosis from the title (common pattern: "A case of [diagnosis]")
    3. Fall back to using the search condition as the expected diagnosis
    """
    # Try to extract diagnosis from title
    diagnosis = None
    title_patterns = [
        r"case (?:report )?of (.+?)(?:\.|:|$)",
        r"presenting (?:as|with) (.+?)(?:\.|:|$)",
        r"diagnosed (?:as|with) (.+?)(?:\.|:|$)",
        r"rare case of (.+?)(?:\.|:|$)",
        r"unusual (?:case|presentation) of (.+?)(?:\.|:|$)",
        # Pattern: "Diagnosis Name: A Case Report"
        r"^(.+?):\s*[Aa]\s*[Cc]ase\s*[Rr]eport",
        # Pattern: "Diagnosis Name - Case Report"
        r"^(.+?)\s*[-–—]\s*[Cc]ase\s*[Rr]eport",
        # Pattern: "Case of Diagnosis Name"
        r"[Cc]ase\s+of\s+(.+?)(?:\.|:|,|$)",
    ]
    for pattern in title_patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            diagnosis = match.group(1).strip()
            break

    if not diagnosis:
        # Extract from search query
        # queries look like: '"case report"[Title] AND "myocardial infarction"[Title]'
        # Find all quoted terms and pick the one that isn't "case report"
        matches = re.findall(r'"([^"]+)"', search_query)
        for m in matches:
            if m.lower() != "case report":
                diagnosis = m
                break

    if not diagnosis:
        return None, None

    # Clean diagnosis text
    diagnosis = diagnosis.strip().rstrip('.')

    # Extract clinical presentation
    # For structured abstracts, look for specific sections
    presentation_sections = ["CASE PRESENTATION", "CASE REPORT", "CASE", "CLINICAL PRESENTATION", "HISTORY"]
    conclusion_sections = ["CONCLUSION", "DISCUSSION", "OUTCOME", "DIAGNOSIS", "RESULTS"]

    # Try to split abstract into presentation vs conclusion
    presentation = abstract

    # Look for section boundaries in structured abstracts
    for cs in conclusion_sections:
        pattern = re.compile(rf'\b{cs}\b[:\s]', re.IGNORECASE)
        match = pattern.search(abstract)
        if match:
            # Everything before the conclusion is the presentation
            candidate = abstract[:match.start()].strip()
            if len(candidate) > 100:
                presentation = candidate
                break

    # Clean up
    presentation = presentation.strip()
    if len(presentation) < 50:
        presentation = abstract  # Use full abstract if extraction is too short

    return presentation, diagnosis


# ──────────────────────────────────────────────
# Validation harness
# ──────────────────────────────────────────────

async def validate_pmc(
    cases: List[ValidationCase],
    include_drug_check: bool = True,
    include_guidelines: bool = True,
    delay_between_cases: float = 2.0,
    resume: bool = False,
) -> ValidationSummary:
    """
    Run PMC case reports through the CDS pipeline and score results.
    """
    results: List[ValidationResult] = []
    start_time = time.time()

    # Resume support
    completed_ids: set = set()
    if resume:
        prior = load_checkpoint("pmc")
        if prior:
            results.extend(prior)
            completed_ids = {r.case_id for r in prior}
            print(f"  Resuming: {len(prior)} cases loaded from checkpoint, {len(cases) - len(completed_ids)} remaining")
    else:
        clear_checkpoint("pmc")

    for i, case in enumerate(cases):
        dx = case.ground_truth.get("diagnosis", "?")
        specialty = case.ground_truth.get("specialty", "?")
        if case.case_id in completed_ids:
            print(f"\n  [{i+1}/{len(cases)}] {case.case_id} ({specialty}): (cached) skipped")
            continue

        print(f"\n  [{i+1}/{len(cases)}] {case.case_id} ({specialty} — {dx[:40]}): ", end="", flush=True)

        case_start = time.monotonic()

        state, report, error = await run_cds_pipeline(
            patient_text=case.input_text,
            include_drug_check=include_drug_check,
            include_guidelines=include_guidelines,
        )

        elapsed_ms = int((time.monotonic() - case_start) * 1000)

        step_results = {}
        if state:
            step_results = {s.step_id: s.status.value for s in state.steps}

        scores = {}
        details = {}
        target_diagnosis = case.ground_truth["diagnosis"]

        if report:
            # Diagnostic accuracy (anywhere in differential)
            found_any, rank_any, loc_any = diagnosis_in_differential(target_diagnosis, report)
            scores["diagnostic_accuracy"] = 1.0 if found_any else 0.0

            # Top-3 accuracy
            found_top3, rank3, loc3 = diagnosis_in_differential(target_diagnosis, report, top_n=3)
            scores["top3_accuracy"] = 1.0 if found_top3 else 0.0

            # Top-1 accuracy
            found_top1, rank1, loc1 = diagnosis_in_differential(target_diagnosis, report, top_n=1)
            scores["top1_accuracy"] = 1.0 if found_top1 else 0.0

            # Parse success
            scores["parse_success"] = 1.0

            # Has recommendations
            scores["has_recommendations"] = 1.0 if len(report.suggested_next_steps) > 0 else 0.0

            details = {
                "target_diagnosis": target_diagnosis,
                "top_diagnosis": report.differential_diagnosis[0].diagnosis if report.differential_diagnosis else "NONE",
                "num_diagnoses": len(report.differential_diagnosis),
                "found_at_rank": rank_any if found_any else -1,
                "all_diagnoses": [d.diagnosis for d in report.differential_diagnosis[:5]],
            }

            icon = "+" if found_any else "-"
            top_dx = report.differential_diagnosis[0].diagnosis if report.differential_diagnosis else "NONE"
            # Sanitize for Windows console encoding
            safe_dx = top_dx[:30].encode('ascii', errors='replace').decode('ascii')
            print(f"{icon} top1={'Y' if found_top1 else 'N'} diag={'Y' if found_any else 'N'} | top: {safe_dx} ({elapsed_ms}ms)")
        else:
            scores = {
                "diagnostic_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "top1_accuracy": 0.0,
                "parse_success": 0.0,
                "has_recommendations": 0.0,
            }
            details = {"target_diagnosis": target_diagnosis, "error": error}
            print(f"- FAILED: {error[:80] if error else 'unknown'}")

        result = ValidationResult(
            case_id=case.case_id,
            source_dataset="pmc",
            success=report is not None,
            scores=scores,
            pipeline_time_ms=elapsed_ms,
            step_results=step_results,
            report_summary=report.patient_summary[:200] if report else None,
            error=error,
            details=details,
        )
        results.append(result)
        save_incremental(result, "pmc")  # checkpoint after every case

        if i < len(cases) - 1:
            await asyncio.sleep(delay_between_cases)

    # Aggregate
    total = len(results)
    successful = sum(1 for r in results if r.success)

    metric_names = ["diagnostic_accuracy", "top3_accuracy", "top1_accuracy", "parse_success", "has_recommendations"]
    metrics = {}
    for m in metric_names:
        values = [r.scores.get(m, 0.0) for r in results]
        metrics[m] = sum(values) / len(values) if values else 0.0

    times = [r.pipeline_time_ms for r in results if r.success]
    metrics["avg_pipeline_time_ms"] = sum(times) / len(times) if times else 0

    summary = ValidationSummary(
        dataset="pmc",
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
    """Run PMC Case Reports validation standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="PMC Case Reports Validation")
    parser.add_argument("--max-cases", type=int, default=10, help="Number of cases to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-drugs", action="store_true", help="Skip drug interaction check")
    parser.add_argument("--no-guidelines", action="store_true", help="Skip guideline retrieval")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between cases (seconds)")
    args = parser.parse_args()

    print("PMC Case Reports Validation Harness")
    print("=" * 40)

    cases = await fetch_pmc_cases(max_cases=args.max_cases, seed=args.seed)
    summary = await validate_pmc(
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
