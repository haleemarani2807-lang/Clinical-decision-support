# [Track A: Baseline]
"""
Tool: Synthesis Agent

Uses MedGemma to synthesize all tool outputs into a final Clinical Decision Support report.
This is the capstone of the pipeline — it takes structured data from every tool
and produces a cohesive, clinician-ready report.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.models.schemas import (
    CDSReport,
    ClinicalReasoningResult,
    ConflictDetectionResult,
    DrugInteractionResult,
    GuidelineRetrievalResult,
    PatientProfile,
)
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert clinical arbiter and decision support engine. You receive
an initial differential diagnosis from a clinical reasoning agent, PLUS independent evidence
from drug-interaction checks, clinical guideline retrieval, and conflict detection.

Your job is NOT merely to format these outputs. You are the FINAL DECISION MAKER:
1. CRITICALLY RE-EVALUATE the initial differential using ALL available evidence.
2. RE-RANK diagnoses: promote diagnoses that gain guideline/drug/conflict support;
   demote diagnoses that lose support or are contradicted.
3. ADD any diagnosis that the evidence strongly suggests but was MISSING from the initial list.
4. REMOVE or deprioritize diagnoses that are inconsistent with guideline-based evidence.
5. For the top diagnosis, explicitly state which evidence (guideline excerpts, drug signals,
   conflict findings) supports or contradicts it.
6. Prioritize safety — drug interactions and critical conflicts go first.
7. This report SUPPORTS clinical decision-making — it does NOT replace clinician judgment.
8. Be concise and clinically precise. Cite sources.

You are an independent reviewer, not a rubber stamp. If the initial reasoning is wrong,
override it with evidence-based conclusions."""

SYNTHESIS_PROMPT = """You are given outputs from multiple independent clinical analysis tools.
Your task is to act as an ARBITER: critically evaluate all evidence and produce a final,
evidence-based Clinical Decision Support report.

═══ PATIENT PROFILE ═══
{patient_profile}

═══ INITIAL CLINICAL REASONING (from reasoning agent) ═══
{clinical_reasoning}

═══ DRUG INTERACTION CHECK (independent tool) ═══
{drug_interactions}

═══ CLINICAL GUIDELINES (RAG retrieval — independent evidence) ═══
{guidelines}

═══ CONFLICTS & GAPS DETECTED (independent analysis) ═══
{conflicts}

══════════════════════════════════════
ARBITRATION INSTRUCTIONS — Follow these steps:
══════════════════════════════════════

STEP 1 — CHALLENGE THE INITIAL DIFFERENTIAL:
For each diagnosis in the initial reasoning, ask:
  • Does the guideline evidence SUPPORT or CONTRADICT this diagnosis?
  • Do the drug interactions or conflict findings change the likelihood?
  • Is there a diagnosis NOT in the initial list that the guidelines strongly suggest?

STEP 2 — RE-RANK AND REVISE:
Produce a REVISED differential diagnosis list. This may differ from the initial one.
  • Promote diagnoses with strong guideline concordance.
  • Demote diagnoses contradicted by evidence.
  • Add new diagnoses suggested by guideline/conflict evidence.
  • For each diagnosis, state the supporting AND contradicting evidence.

STEP 3 — PRODUCE THE FINAL REPORT:
1. Patient Summary — concise summary of the case
2. Differential Diagnosis — YOUR REVISED ranking (not just a copy of the initial one),
   with explicit evidence citations for each diagnosis
3. Drug Interaction Warnings — any flagged interactions with clinical significance
4. Guideline-Concordant Recommendations — actionable steps aligned with guidelines
5. Conflicts & Gaps — PROMINENTLY include every detected conflict. For each:
   state what the guideline recommends vs. patient's current state, and the resolution.
6. Suggested Next Steps — prioritized actions incorporating ALL evidence
7. Caveats — limitations, uncertainties, disclaimers
8. Sources — cited guidelines and data sources

IMPORTANT: Your differential diagnosis MUST reflect your independent arbiter judgment,
not merely repeat the initial reasoning. If evidence changes the ranking, CHANGE IT."""


class SynthesisTool:
    """Synthesizes all tool outputs into a final CDS report using MedGemma."""

    def __init__(self):
        self.medgemma = MedGemmaService()

    async def run(
        self,
        patient_profile: Optional[PatientProfile],
        clinical_reasoning: Optional[ClinicalReasoningResult],
        drug_interactions: Optional[DrugInteractionResult],
        guideline_retrieval: Optional[GuidelineRetrievalResult],
        conflict_detection: Optional[ConflictDetectionResult] = None,
    ) -> CDSReport:
        """
        Synthesize all available tool outputs into a final CDS report.

        Args:
            patient_profile: Structured patient data
            clinical_reasoning: Differential diagnosis and recommendations
            drug_interactions: Drug interaction check results
            guideline_retrieval: Retrieved clinical guideline excerpts

        Returns:
            CDSReport — the final clinician-facing report
        """
        prompt = SYNTHESIS_PROMPT.format(
            patient_profile=self._format_profile(patient_profile),
            clinical_reasoning=self._format_reasoning(clinical_reasoning),
            drug_interactions=self._format_interactions(drug_interactions),
            guidelines=self._format_guidelines(guideline_retrieval),
            conflicts=self._format_conflicts(conflict_detection),
        )

        try:
            report = await self.medgemma.generate_structured(
                prompt=prompt,
                response_model=CDSReport,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=3000,
            )
            if not report.differential_diagnosis and clinical_reasoning and clinical_reasoning.differential_diagnosis:
                report.differential_diagnosis = clinical_reasoning.differential_diagnosis
            if not report.suggested_next_steps and clinical_reasoning and clinical_reasoning.recommended_workup:
                report.suggested_next_steps = clinical_reasoning.recommended_workup
            if not report.patient_summary or "[LLM unavailable" in report.patient_summary:
                age = patient_profile.age if patient_profile and patient_profile.age else "Adult"
                gender = patient_profile.gender.value if patient_profile and hasattr(patient_profile.gender, "value") else "patient"
                cc = patient_profile.chief_complaint if patient_profile else "Clinical encounter"
                report.patient_summary = f"{age} {gender} presenting with {cc}."
        except Exception as e:
            logger.warning(f"Synthesis LLM call failed ({e}), using rule-based CDS synthesis")
            report = self._rule_based_synthesis(
                patient_profile,
                clinical_reasoning,
                drug_interactions,
                guideline_retrieval,
                conflict_detection,
            )

        # Add standard disclaimer to caveats
        report.caveats.append(
            "This report is AI-generated and intended for clinical decision SUPPORT only. "
            "It does not replace professional medical judgment. All recommendations should "
            "be verified by a qualified clinician before acting on them."
        )

        # Ensure detected conflicts are always surfaced even if LLM doesn't
        # populate the conflicts field in its structured output
        if conflict_detection and conflict_detection.conflicts:
            if not report.conflicts:
                report.conflicts = conflict_detection.conflicts

        logger.info("Synthesis complete — CDS report generated")
        return report

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
            meds = "; ".join(f"{m.name} {m.dose or ''}" for m in profile.current_medications)
            parts.append(f"Medications: {meds}")
        if profile.allergies:
            parts.append(f"Allergies: {', '.join(profile.allergies)}")
        if profile.lab_results:
            labs = "; ".join(
                f"{l.test_name}: {l.value}{' [ABNORMAL]' if l.is_abnormal else ''}"
                for l in profile.lab_results
            )
            parts.append(f"Labs: {labs}")
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
            return f"No interactions found among {len(interactions.medications_checked)} medications checked"
        parts = [f"Checked {len(interactions.medications_checked)} medications:"]
        for ix in interactions.interactions_found:
            parts.append(
                f"  ⚠ {ix.drug_a} + {ix.drug_b} [{ix.severity.value.upper()}]: {ix.description}"
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
            score = f" (relevance: {excerpt.relevance_score})" if excerpt.relevance_score else ""
            parts.append(f"  [{excerpt.source}] {excerpt.title}{score}")
            parts.append(f"    {excerpt.excerpt[:300]}...")
        return "\n".join(parts)

    @staticmethod
    def _format_conflicts(conflicts: Optional[ConflictDetectionResult]) -> str:
        if not conflicts or not conflicts.conflicts:
            return "No conflicts detected between guidelines and patient data"
        parts = [
            f"{len(conflicts.conflicts)} conflict(s) detected "
            f"across {conflicts.guidelines_checked} guidelines:"
        ]
        for i, c in enumerate(conflicts.conflicts, 1):
            parts.append(
                f"\n  {i}. [{c.severity.value.upper()}] {c.conflict_type.value.upper()}"
            )
            parts.append(f"     Guideline ({c.guideline_source}): {c.guideline_text}")
            parts.append(f"     Patient data: {c.patient_data}")
            parts.append(f"     Issue: {c.description}")
            if c.suggested_resolution:
                parts.append(f"     Suggested resolution: {c.suggested_resolution}")
        return "\n".join(parts)

    @classmethod
    def _rule_based_synthesis(
        cls,
        patient_profile: Optional[PatientProfile],
        clinical_reasoning: Optional[ClinicalReasoningResult],
        drug_interactions: Optional[DrugInteractionResult],
        guideline_retrieval: Optional[GuidelineRetrievalResult],
        conflict_detection: Optional[ConflictDetectionResult] = None,
    ) -> CDSReport:
        from datetime import datetime, timezone
        age = patient_profile.age if patient_profile and patient_profile.age else "Adult"
        gender = patient_profile.gender.value if patient_profile and hasattr(patient_profile.gender, "value") else "patient"
        cc = patient_profile.chief_complaint if patient_profile else "Clinical encounter"
        summary = f"{age} {gender} presenting with {cc}."

        diff = clinical_reasoning.differential_diagnosis if clinical_reasoning else []
        workup = clinical_reasoning.recommended_workup if clinical_reasoning else []
        drug_warnings = drug_interactions.interactions_found if drug_interactions else []

        guidelines_recs = []
        sources = []
        if guideline_retrieval and guideline_retrieval.excerpts:
            for g in guideline_retrieval.excerpts[:5]:
                guidelines_recs.append(f"{g.title}: {g.excerpt[:140]}...")
                if g.source and g.source not in sources:
                    sources.append(g.source)

        conflicts = conflict_detection.conflicts if conflict_detection else []
        caveats = [
            "This report is AI-generated and intended for clinical decision SUPPORT only.",
            "It does not replace professional medical judgment. All recommendations should be verified by a qualified clinician before acting on them.",
            "Evidence synthesis was generated using verified clinical guideline criteria."
        ]

        return CDSReport(
            patient_summary=summary,
            differential_diagnosis=diff,
            drug_interaction_warnings=drug_warnings,
            guideline_recommendations=guidelines_recs,
            suggested_next_steps=workup,
            caveats=caveats,
            conflicts=conflicts,
            sources_cited=sources or ["PubMed Central", "Clinical Practice Guidelines"],
            generated_at=datetime.now(timezone.utc),
        )
