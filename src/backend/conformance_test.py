"""
Conformance test suite — schema/contract regression tests.

Two test groups:
  A. Offline tests (run always, no network/model required) verify that the
     Pydantic field validators normalize free-text enum values the way every
     downstream tool expects.
  B. Online probes (skipped if MedGemma endpoint is unreachable) verify that
     the live model still returns JSON that parses cleanly into our enums.

Run from src/backend:
    python conformance_test.py            # full suite (online + offline)
    python conformance_test.py --offline  # offline only (CI/precommit safe)

Exit code 0 on full pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.models.schemas import (
    ClinicalConflict,
    Confidence,
    ConflictType,
    DiagnosisCandidate,
    Severity,
)
from app.services.medgemma import MedGemmaService

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("conformance")
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────
# Offline normalization tests
# ──────────────────────────────────────────────

def test_confidence_enum_direct() -> bool:
    """Every documented Confidence value constructs successfully."""
    for member in Confidence:
        try:
            dx = DiagnosisCandidate(diagnosis="x", likelihood=member.value)
            assert dx.likelihood == member, f"{member.value!r} did not round-trip"
        except Exception as e:
            logger.error(f"  FAIL Confidence.{member.name}: {e}")
            return False
    return True


def test_confidence_fuzzy_mapping() -> bool:
    """Common LLM phrasings fold into the expected canonical Confidence."""
    cases: list[tuple[str, Confidence]] = [
        ("Unlikely", Confidence.VERY_LOW),
        ("possible", Confidence.LOW),
        ("medium", Confidence.MODERATE),
        ("likely", Confidence.HIGH),
        ("Highly Likely", Confidence.VERY_HIGH),
        ("almost certain", Confidence.VERY_HIGH),
    ]
    ok = True
    for raw, expected in cases:
        dx = DiagnosisCandidate(diagnosis="x", likelihood=raw)
        if dx.likelihood != expected:
            logger.error(
                f"  FAIL fuzzy {raw!r}: got {dx.likelihood.value!r}, expected {expected.value!r}"
            )
            ok = False
    return ok


def test_confidence_unknown_preserves_raw() -> bool:
    """Unrecognized confidence values fall back to UNRECOGNIZED with raw preserved."""
    dx = DiagnosisCandidate(diagnosis="x", likelihood="quasi-plausible")
    if dx.likelihood != Confidence.UNRECOGNIZED:
        logger.error(f"  FAIL unknown likelihood: got {dx.likelihood}")
        return False
    if dx.likelihood_raw != "quasi-plausible":
        logger.error(f"  FAIL likelihood_raw not preserved: got {dx.likelihood_raw!r}")
        return False
    return True


def test_severity_case_insensitive() -> bool:
    """Severity normalizes mixed-case input via the validator."""
    for raw in ["LOW", "Moderate", "HIGH", "Critical"]:
        try:
            c = ClinicalConflict(
                conflict_type="omission",
                severity=raw,
            )
            assert c.severity == Severity(raw.lower())
        except Exception as e:
            logger.error(f"  FAIL severity {raw!r}: {e}")
            return False
    return True


def test_conflict_type_case_insensitive() -> bool:
    """ConflictType normalizes mixed-case input via the validator."""
    for raw, expected in [
        ("OMISSION", ConflictType.OMISSION),
        ("Contradiction", ConflictType.CONTRADICTION),
        ("ALLERGY_RISK", ConflictType.ALLERGY_RISK),
    ]:
        try:
            c = ClinicalConflict(conflict_type=raw, severity="low")
            assert c.conflict_type == expected
        except Exception as e:
            logger.error(f"  FAIL conflict_type {raw!r}: {e}")
            return False
    return True


# ──────────────────────────────────────────────
# Online probes (require MedGemma endpoint)
# ──────────────────────────────────────────────

class EnumProbe(BaseModel):
    """Minimal schema for the LLM enum-shape probe."""
    severity: Severity = Field(..., description="One of: low, moderate, high, critical")
    confidence: Confidence = Field(..., description="One of: very low, low, moderate, high, very high")


async def test_enum_probe_online(medgemma: MedGemmaService) -> bool:
    """Live model returns JSON that maps cleanly into Severity + Confidence."""
    prompt = (
        "Return ONLY a JSON object with two fields:\n"
        '  "severity": one of "low", "moderate", "high", "critical"\n'
        '  "confidence": one of "very low", "low", "moderate", "high", "very high"\n'
        'Use severity="High" and confidence="Moderate". '
        "Do not include any other text."
    )
    try:
        result = await medgemma.generate_structured(
            prompt=prompt,
            response_model=EnumProbe,
            system_prompt="You are a clinical assistant. Respond with valid JSON only.",
            temperature=0.0,
            max_tokens=200,
        )
        if result.severity != Severity.HIGH:
            logger.error(f"  FAIL probe severity: got {result.severity}")
            return False
        if result.confidence != Confidence.MODERATE:
            logger.error(f"  FAIL probe confidence: got {result.confidence}")
            return False
        return True
    except Exception as e:
        logger.error(f"  FAIL enum probe: {type(e).__name__}: {e}")
        return False


async def test_clinical_conflict_online(medgemma: MedGemmaService) -> bool:
    """Live model returns a ClinicalConflict that passes our validators."""
    prompt = (
        "A patient with diabetes has HbA1c=9.2% but is not on metformin. "
        "Return ONLY a JSON object with these fields: "
        '"conflict_type" (use "omission"), "severity" (use "high"), '
        '"guideline_source", "guideline_text", "patient_data", "description". '
        "Do not include any other text."
    )
    try:
        result = await medgemma.generate_structured(
            prompt=prompt,
            response_model=ClinicalConflict,
            system_prompt="You are a clinical assistant. Respond with valid JSON only.",
            temperature=0.0,
            max_tokens=400,
        )
        if result.conflict_type != ConflictType.OMISSION:
            logger.error(f"  FAIL conflict_type: got {result.conflict_type}")
            return False
        if result.severity != Severity.HIGH:
            logger.error(f"  FAIL severity: got {result.severity}")
            return False
        return True
    except Exception as e:
        logger.error(f"  FAIL clinical conflict probe: {type(e).__name__}: {e}")
        return False


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

OFFLINE_TESTS: list[tuple[str, Callable[[], bool]]] = [
    ("confidence_enum_direct", test_confidence_enum_direct),
    ("confidence_fuzzy_mapping", test_confidence_fuzzy_mapping),
    ("confidence_unknown_preserves_raw", test_confidence_unknown_preserves_raw),
    ("severity_case_insensitive", test_severity_case_insensitive),
    ("conflict_type_case_insensitive", test_conflict_type_case_insensitive),
]


async def main(offline_only: bool = False) -> int:
    results: list[tuple[str, bool]] = []

    print("=" * 60)
    print("Offline schema conformance")
    print("=" * 60)
    for name, fn in OFFLINE_TESTS:
        ok = fn()
        results.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if not offline_only:
        print()
        print("=" * 60)
        print("Online enum probes (live MedGemma)")
        print("=" * 60)
        medgemma = MedGemmaService()
        ready = await medgemma.check_readiness()
        if not ready:
            print("  SKIP  endpoint not reachable — set MEDGEMMA_BASE_URL/API_KEY to run online probes")
        else:
            for name, coro in [
                ("enum_probe_online", test_enum_probe_online(medgemma)),
                ("clinical_conflict_online", test_clinical_conflict_online(medgemma)),
            ]:
                ok = await coro
                results.append((name, ok))
                print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    print()
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    print(f"Conformance: {passed}/{len(results)} passed")
    print("=" * 60)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Skip live-model probes")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(offline_only=args.offline)))
