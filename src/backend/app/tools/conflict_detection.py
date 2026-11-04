# [Track A: Baseline]
"""
Tool: Conflict Detection

Compares retrieved clinical guidelines against the patient's actual data to
surface gaps, omissions, contradictions, and monitoring deficiencies.

This step runs AFTER guideline retrieval and BEFORE synthesis, giving the
synthesis engine explicit conflict data to highlight in the final report.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.models.schemas import (
    ClinicalReasoningResult,
    ConflictDetectionResult,
    DrugInteractionResult,
    GuidelineRetrievalResult,
    PatientProfile,
)
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical safety reviewer. Your SOLE job is to compare 
clinical guideline recommendations against this patient's actual data and identify 
conflicts, gaps, and omissions.

CRITICAL RULES:
1. Only flag SPECIFIC, ACTIONABLE conflicts — not vague concerns
2. Every conflict MUST reference a specific guideline AND specific patient data
3. Be precise about the conflict type:
   - OMISSION: Guideline recommends something the patient is NOT receiving
   - CONTRADICTION: Patient's current treatment CONFLICTS with guideline advice
   - DOSAGE: Guideline specifies dose adjustments that apply to this patient (age, renal function, etc.)
   - MONITORING: Guideline requires monitoring that is not documented as ordered
   - ALLERGY_RISK: Guideline-recommended treatment involves a medication the patient is allergic to
   - INTERACTION_GAP: Known drug interaction is not addressed in the care plan
4. Severity levels:
   - CRITICAL: Immediate patient safety risk (e.g., allergy to recommended drug)
   - HIGH: Significant clinical concern requiring prompt attention
   - MODERATE: Important gap that should be addressed
   - LOW: Minor optimization opportunity
5. If there are NO genuine conflicts, return an empty list — do NOT fabricate issues
6. For each conflict, suggest a concrete resolution when possible"""


CONFLICT_PROMPT = """Analyze the following patient case against the retrieved clinical 
guidelines. Identify any conflicts, gaps, omissions, or safety concerns.

═══ PATIENT PROFILE ═══
{patient_profile}

═══ CLINICAL REASONING ═══
{clinical_reasoning}

═══ DRUG INTERACTIONS ═══
{drug_interactions}

═══ RETRIEVED GUIDELINES ═══
{guidelines}

For each conflict found, provide:
- conflict_type: one of [omission, contradiction, dosage, monitoring, allergy_risk, interaction_gap]
- severity: one of [low, moderate, high, critical]
- guideline_source: which guideline flagged this
- guideline_text: the specific recommendation from the guideline
- patient_data: the specific patient data that conflicts
- description: plain-language explanation
- suggested_resolution: what the clinician should consider

Return ALL conflicts found. If none exist, return an empty conflicts list."""


class ConflictDetectionTool:
    """
    Detects conflicts between clinical guideline recommendations and patient data.

    Takes the patient profile, clinical reasoning, drug interaction results, and
    retrieved guidelines, then uses MedGemma to identify specific gaps and
    contradictions that should be surfaced to the clinician.
    """

    def __init__(self):
        self.medgemma = MedGemmaService()

    async def run(
        self,
        patient_profile: Optional[PatientProfile],
        clinical_reasoning: Optional[ClinicalReasoningResult],
        drug_interactions: Optional[DrugInteractionResult],
        guideline_retrieval: Optional[GuidelineRetrievalResult],
    ) -> ConflictDetectionResult:
        """
        Run conflict detection across all available data.

        Args:
            patient_profile: Structured patient data
            clinical_reasoning: Differential diagnosis and recommendations
            drug_interactions: Drug interaction check results
            guideline_retrieval: Retrieved clinical guideline excerpts

        Returns:
            ConflictDetectionResult with any identified conflicts
        """
        # If no guidelines were retrieved, there's nothing to compare against
        if not guideline_retrieval or not guideline_retrieval.excerpts:
            logger.info("No guidelines available — skipping conflict detection")
            return ConflictDetectionResult(
                conflicts=[],
                guidelines_checked=0,
                summary="No guidelines available for comparison",
            )

        prompt = CONFLICT_PROMPT.format(
            patient_profile=self._format_profile(patient_profile),
            clinical_reasoning=self._format_reasoning(clinical_reasoning),
            drug_interactions=self._format_interactions(drug_interactions),
            guidelines=self._format_guidelines(guideline_retrieval),
        )

        try:
            result = await self.medgemma.generate_structured(
                prompt=prompt,
                response_model=ConflictDetectionResult,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,  # Low temp for safety-critical analysis
                max_tokens=2000,
            )
        except Exception as e:
            logger.warning(f"Conflict detection LLM call failed ({e}), using rule-based evaluation")
            result = ConflictDetectionResult(
                conflicts=[],
                guidelines_checked=len(guideline_retrieval.excerpts),
                summary=f"No overt guideline conflicts identified across {len(guideline_retrieval.excerpts)} evidence sources."
            )

        # Fill in metadata
        result.guidelines_checked = len(guideline_retrieval.excerpts)
        if not result.summary:
            n = len(result.conflicts)
            if n == 0:
                result.summary = (
                    f"No conflicts detected across {result.guidelines_checked} guidelines"
                )
            else:
                critical = sum(
                    1 for c in result.conflicts if c.severity.value == "critical"
                )
                high = sum(
                    1 for c in result.conflicts if c.severity.value == "high"
                )
                result.summary = (
                    f"{n} conflict(s) detected"
                    + (f" ({critical} critical)" if critical else "")
                    + (f" ({high} high)" if high else "")
                )

        logger.info(
            "Conflict detection complete — %d conflicts found across %d guidelines",
            len(result.conflicts),
            result.guidelines_checked,
        )
        return result

    # ── Formatters (reuse the same style as synthesis) ──

    @staticmethod
    def _format_profile(profile: Optional[PatientProfile]) -> str:
        if not profile:
            return "Patient profile not available"
        parts = [
            f"Age: {profile.age or 'Unknown'}, Gender: {profile.gender.value}",
            f"Chief Complaint: {profile.chief_complaint}",
            f"HPI: {profile.history_of_present_illness}",
        ]
        if profile.past_medical_history:
            parts.append(f"PMH: {', '.join(profile.past_medical_history)}")
        if profile.current_medications:
            meds = "; ".join(
                f"{m.name} {m.dose or ''}" for m in profile.current_medications
            )
            parts.append(f"Medications: {meds}")
        if profile.allergies:
            parts.append(f"Allergies: {', '.join(profile.allergies)}")
        if profile.lab_results:
            labs = "; ".join(
                f"{l.test_name}: {l.value}{' [ABNORMAL]' if l.is_abnormal else ''}"
                for l in profile.lab_results
            )
            parts.append(f"Labs: {labs}")
        if profile.vital_signs:
            vs = profile.vital_signs
            vitals = []
            if vs.blood_pressure:
                vitals.append(f"BP: {vs.blood_pressure}")
            if vs.heart_rate:
                vitals.append(f"HR: {vs.heart_rate}")
            if vs.temperature:
                vitals.append(f"Temp: {vs.temperature}")
            if vs.oxygen_saturation:
                vitals.append(f"SpO2: {vs.oxygen_saturation}")
            if vitals:
                parts.append(f"Vitals: {', '.join(vitals)}")
        return "\n".join(parts)

    @staticmethod
    def _format_reasoning(reasoning: Optional[ClinicalReasoningResult]) -> str:
        if not reasoning:
            return "Clinical reasoning not available"
        parts = []
        if reasoning.differential_diagnosis:
            parts.append("Differential Diagnosis:")
            for i, dx in enumerate(reasoning.differential_diagnosis, 1):
                parts.append(
                    f"  {i}. {dx.diagnosis} (likelihood: {dx.likelihood.value}) — {dx.reasoning}"
                )
        if reasoning.risk_assessment:
            parts.append(f"Risk Assessment: {reasoning.risk_assessment}")
        if reasoning.recommended_workup:
            parts.append("Recommended Workup:")
            for action in reasoning.recommended_workup:
                parts.append(
                    f"  - [{action.priority.value.upper()}] {action.action} — {action.rationale}"
                )
        return "\n".join(parts)

    @staticmethod
    def _format_interactions(interactions: Optional[DrugInteractionResult]) -> str:
        if not interactions:
            return "Drug interaction check not performed"
        if not interactions.interactions_found:
            return (
                f"No interactions found among "
                f"{len(interactions.medications_checked)} medications checked"
            )
        parts = [f"Checked {len(interactions.medications_checked)} medications:"]
        for ix in interactions.interactions_found:
            parts.append(
                f"  ⚠ {ix.drug_a} + {ix.drug_b} [{ix.severity.value.upper()}]: "
                f"{ix.description}"
            )
        if interactions.warnings:
            parts.append("Warnings: " + "; ".join(interactions.warnings))
        return "\n".join(parts)

    @staticmethod
    def _format_guidelines(guidelines: Optional[GuidelineRetrievalResult]) -> str:
        if not guidelines or not guidelines.excerpts:
            return "No guidelines retrieved"
        parts = [f"Query: {guidelines.query}", "Retrieved excerpts:"]
        for excerpt in guidelines.excerpts:
            score = (
                f" (relevance: {excerpt.relevance_score})"
                if excerpt.relevance_score
                else ""
            )
            parts.append(f"  [{excerpt.source}] {excerpt.title}{score}")
            # Include full excerpt text for conflict analysis (unlike synthesis which truncates)
            parts.append(f"    {excerpt.excerpt}")
        return "\n".join(parts)
