"""
Clinical Context Extraction Agent.

Responsibilities:
  1. Extract clinical entities (symptoms, diagnoses, medications, lab values)
  2. Structure patient history into normalized fields
  3. Detect missing required clinical information
  4. Detect inconsistent/contradictory patient data
  5. Produce a NormalizedPatientContext object

Strategy:
  - Primary: LLM-based extraction (via llm_factory)
  - Fallback: Rule-based regex NER when LLM is unavailable
  - Missing-field detection is always deterministic (rule-based)

Does NOT invent missing patient information.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.models.schemas import (
    MissingFieldFlags,
    NormalizedPatientContext,
    PatientProfile,
    InconsistencyFlag,
)
from app.services.llm_factory import get_agent_llm

logger = logging.getLogger(__name__)

# ── Regex patterns for rule-based NER ────────────────────────────────────────
_RE_AGE = re.compile(r'\b(\d{1,3})[\s-]year[\s-]old\b', re.IGNORECASE)
_RE_GENDER = re.compile(r'\b(male|female|man|woman|boy|girl|he\b|she\b)', re.IGNORECASE)
_RE_BP = re.compile(r'\b(?:BP|blood pressure)[:\s]+(\d{2,3}/\d{2,3})\s*mmHg\b', re.IGNORECASE)
_RE_HR = re.compile(r'\b(?:HR|heart rate|pulse)[:\s]+(\d{2,3})\s*(?:bpm|/min)\b', re.IGNORECASE)
_RE_TEMP = re.compile(r'\b(?:Temp|temperature)[:\s]+(\d{2,3}\.?\d*)[\s°]*(?:°?C|°?F)\b', re.IGNORECASE)
_RE_RR = re.compile(r'\b(?:RR|respiratory rate)[:\s]+(\d{1,3})\s*/min\b', re.IGNORECASE)
_RE_SPO2 = re.compile(r'\b(?:SpO2|O2 sat|oxygen saturation)[:\s]+(\d{2,3})%\b', re.IGNORECASE)
_RE_WEIGHT = re.compile(r'\b(?:weight|wt)[:\s]+(\d{2,4}\.?\d*)\s*(?:kg|lbs?)\b', re.IGNORECASE)
_RE_MED = re.compile(r'\b([A-Za-z]+(?:\s[A-Za-z]+)?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|units?|g)\b')
_RE_ALLERGY_NKDA = re.compile(r'\b(?:NKDA|no known drug allerg|no allerg)\b', re.IGNORECASE)
_RE_ALLERGY = re.compile(r'\bAllerg(?:ic|ies?)[:\s]+([^.]+)', re.IGNORECASE)

REQUIRED_FIELDS = [
    "age", "gender", "chief_complaint", "vital_signs",
    "current_medications", "allergy_status", "lab_results",
]

INCONSISTENCY_CHECKS = [
    ("heart_rate_range", _RE_HR, lambda vals: _check_range_conflict(vals, 20, 300)),
    ("blood_pressure", _RE_BP, lambda vals: _check_bp_conflict(vals)),
    ("temperature_range", _RE_TEMP, lambda vals: _check_range_conflict(vals, 34.0, 42.5)),
]


def _check_range_conflict(values: list[str], low: float, high: float) -> Optional[str]:
    """Return conflict message if extracted values are physiologically impossible."""
    nums = []
    for v in values:
        try:
            nums.append(float(v.split("/")[0]))
        except Exception:
            pass
    if not nums:
        return None
    if max(nums) > high or min(nums) < low:
        return f"Value outside physiological range: {nums}"
    if len(nums) > 1 and abs(max(nums) - min(nums)) > 50:
        return f"Conflicting values found: {nums}"
    return None


def _check_bp_conflict(values: list[str]) -> Optional[str]:
    """Check for contradictory blood pressure readings."""
    if len(values) < 2:
        return None
    systolics = []
    for v in values:
        try:
            systolics.append(int(v.split("/")[0]))
        except Exception:
            pass
    if len(systolics) > 1 and abs(max(systolics) - min(systolics)) > 40:
        return f"Contradictory BP readings: {values}"
    return None


class ClinicalContextAgent:
    """
    Extracts and normalises clinical context from free-text patient input.

    Usage:
        agent = ClinicalContextAgent()
        context = await agent.run("65yo male with chest pain...")
    """

    def __init__(self):
        self._llm = get_agent_llm("patient_parser")

    async def run(self, patient_text: str) -> NormalizedPatientContext:
        """
        Extract clinical context from patient text.

        Returns NormalizedPatientContext. Never invents missing information.
        """
        if not patient_text or len(patient_text.strip()) < 10:
            return self._empty_context(patient_text, reason="Input too short")

        # Step 1: Rule-based extraction (always runs, never fails)
        rule_extracted = self._rule_based_extract(patient_text)

        # Step 2: LLM-based extraction (enhances rule-based output)
        try:
            llm_context = await self._llm_extract(patient_text)
            # Merge: LLM fills in what rules missed; rules fill in what LLM missed
            merged = self._merge(rule_extracted, llm_context, patient_text)
        except Exception as e:
            logger.warning("LLM extraction failed: %s — using rule-based only", e)
            merged = rule_extracted

        # Step 3: Missing-field detection (deterministic, always runs)
        missing = self._detect_missing_fields(merged, patient_text)

        # Step 4: Inconsistency detection (deterministic)
        inconsistencies = self._detect_inconsistencies(patient_text)

        return NormalizedPatientContext(
            patient_profile=merged,
            missing_fields=missing,
            inconsistencies=inconsistencies,
            extraction_method="llm+rule_based",
            raw_input_length=len(patient_text),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # LLM extraction
    # ──────────────────────────────────────────────────────────────────────────

    async def _llm_extract(self, patient_text: str) -> PatientProfile:
        """Use LLM to extract structured PatientProfile."""
        system = (
            "You are a clinical NLP system that extracts structured patient information. "
            "Extract ONLY information explicitly stated in the text. "
            "Do NOT invent, assume, or hallucinate missing fields. "
            "Use null/empty for any field not mentioned."
        )
        prompt = (
            f"Extract all clinical information from this patient case:\n\n"
            f"{patient_text}\n\n"
            f"Return a structured PatientProfile with every field that can be determined from the text. "
            f"Leave fields null/empty if not mentioned."
        )
        return await self._llm.generate_structured(
            prompt, PatientProfile, system_prompt=system, temperature=0.1
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Rule-based extraction (fallback / supplement)
    # ──────────────────────────────────────────────────────────────────────────

    def _rule_based_extract(self, text: str) -> PatientProfile:
        """
        Minimal rule-based patient profile extraction.
        Produces reliable but incomplete results.
        """
        from app.models.schemas import Medication, LabResult, VitalSigns, Gender

        # Age
        age = None
        m = _RE_AGE.search(text)
        if m:
            try:
                age = int(m.group(1))
            except Exception:
                pass

        # Gender
        gender = Gender.UNKNOWN
        gm = _RE_GENDER.search(text)
        if gm:
            g = gm.group(1).lower()
            if g in ("male", "man", "boy", "he"):
                gender = Gender.MALE
            elif g in ("female", "woman", "girl", "she"):
                gender = Gender.FEMALE

        # Vitals
        vitals = VitalSigns(
            blood_pressure=self._extract_first(_RE_BP, text),
            heart_rate=self._extract_first(_RE_HR, text),
            temperature=self._extract_first(_RE_TEMP, text),
            respiratory_rate=self._extract_first(_RE_RR, text),
            oxygen_saturation=self._extract_first(_RE_SPO2, text),
            weight=self._extract_first(_RE_WEIGHT, text),
        )
        has_any_vital = any([
            vitals.blood_pressure, vitals.heart_rate, vitals.temperature,
            vitals.respiratory_rate, vitals.oxygen_saturation
        ])

        # Chief complaint (heuristic: first sentence)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        chief_complaint = sentences[0] if sentences else "Unknown"

        # Allergies
        allergies = []
        if _RE_ALLERGY_NKDA.search(text):
            allergies = ["NKDA"]
        else:
            am = _RE_ALLERGY.search(text)
            if am:
                allergies = [a.strip() for a in am.group(1).split(",")]

        return PatientProfile(
            age=age,
            gender=gender,
            chief_complaint=chief_complaint,
            vital_signs=vitals if has_any_vital else None,
            allergies=allergies,
        )

    @staticmethod
    def _extract_first(pattern: re.Pattern, text: str) -> Optional[str]:
        m = pattern.search(text)
        return m.group(1) if m else None

    # ──────────────────────────────────────────────────────────────────────────
    # Merge strategy
    # ──────────────────────────────────────────────────────────────────────────

    def _merge(self, rule: PatientProfile, llm: PatientProfile,
               original_text: str) -> PatientProfile:
        """
        Merge rule-based and LLM extractions.
        LLM values take precedence; rule-based fills gaps.
        """
        merged = llm.model_copy(deep=True)

        # Fill gaps from rule-based extraction
        if not merged.age and rule.age:
            merged.age = rule.age
        if merged.gender.value == "unknown" and rule.gender.value != "unknown":
            merged.gender = rule.gender
        if not merged.chief_complaint or merged.chief_complaint == "Unknown":
            merged.chief_complaint = rule.chief_complaint
        if not merged.vital_signs and rule.vital_signs:
            merged.vital_signs = rule.vital_signs
        if not merged.allergies and rule.allergies:
            merged.allergies = rule.allergies

        return merged

    # ──────────────────────────────────────────────────────────────────────────
    # Missing-field detection (deterministic)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_missing_fields(self, profile: PatientProfile, text: str) -> MissingFieldFlags:
        """
        Deterministically detect which required fields are missing.
        Never marks a field missing if it was extracted.
        """
        text_lower = text.lower()

        missing_flags = MissingFieldFlags(
            age=profile.age is None,
            gender=profile.gender.value == "unknown",
            chief_complaint=not profile.chief_complaint or profile.chief_complaint == "Unknown",
            vital_signs=(
                profile.vital_signs is None
                and not re.search(r'\b(bp|blood pressure|hr|heart rate|temp|spo2)\b', text_lower)
            ),
            current_medications=(
                not profile.current_medications
                and not re.search(r'\b(medication|drug|tablet|capsule|mg|mcg)\b', text_lower)
            ),
            allergy_status=(
                not profile.allergies
                and not re.search(r'\b(allerg|nkda|no known)\b', text_lower)
            ),
            lab_results=(
                not profile.lab_results
                and not re.search(r'\b(labs?:|wbc|hemoglobin|creatinine|glucose|troponin)\b', text_lower)
            ),
            past_medical_history=not profile.past_medical_history,
            social_history=not profile.social_history,
            family_history=not profile.family_history,
        )

        missing_count = sum([
            missing_flags.age, missing_flags.gender, missing_flags.chief_complaint,
            missing_flags.vital_signs, missing_flags.current_medications,
            missing_flags.allergy_status, missing_flags.lab_results,
        ])
        missing_flags.missing_count = missing_count
        missing_flags.critical_missing = missing_count >= 3

        return missing_flags

    # ──────────────────────────────────────────────────────────────────────────
    # Inconsistency detection (deterministic)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_inconsistencies(self, text: str) -> list[InconsistencyFlag]:
        """
        Check for physiologically impossible or contradictory values.
        Rule-based — no LLM required.
        """
        flags = []
        for field_name, pattern, checker in INCONSISTENCY_CHECKS:
            values = pattern.findall(text)
            if len(values) > 1:
                conflict = checker(values)
                if conflict:
                    flags.append(InconsistencyFlag(
                        field=field_name,
                        description=conflict,
                        values_found=values,
                        severity="warning",
                    ))

        # Allergy-medication conflict
        if _RE_ALLERGY.search(text):
            allergy_match = _RE_ALLERGY.search(text)
            if allergy_match:
                allergen = allergy_match.group(1).strip().lower()
                # Check if the allergen appears as a prescribed medication
                if re.search(rf'\b{re.escape(allergen.split()[0])}\b.*(?:prescribed|ordered|given|started)',
                              text, re.IGNORECASE):
                    flags.append(InconsistencyFlag(
                        field="medication_allergy_conflict",
                        description=f"Patient allergic to '{allergen}' but appears in treatment plan",
                        values_found=[allergen],
                        severity="critical",
                    ))

        return flags

    @staticmethod
    def _empty_context(text: str, reason: str = "") -> NormalizedPatientContext:
        """Return a safe empty context when extraction cannot proceed."""
        from app.models.schemas import Gender
        return NormalizedPatientContext(
            patient_profile=PatientProfile(
                chief_complaint=f"[Extraction failed: {reason}]",
                gender=Gender.UNKNOWN,
            ),
            missing_fields=MissingFieldFlags(
                age=True, gender=True, chief_complaint=True,
                vital_signs=True, current_medications=True,
                allergy_status=True, lab_results=True,
                missing_count=7, critical_missing=True,
            ),
            inconsistencies=[],
            extraction_method="none",
            raw_input_length=len(text),
        )
