"""
Uncertainty & Completeness Agent — Phase 7 of Evidence-Verified CDSS.

This agent evaluates clinical completeness, evidence sufficiency, verification
conflict signals, and critical safety hazards to produce an UncertaintyReport
and make deterministic abstention decisions.

Responsibilities:
  1. Detect patient information gaps from NormalizedPatientContext missing fields.
  2. Evaluate evidence sufficiency based on passage count and recency/quality grades.
  3. Detect verification-based uncertainty (contradicted claims, low claim support rate).
  4. Decide whether the CDS system must abstain from generating recommendations.
  5. Compute overall clinical uncertainty level (low, moderate, high, critical).
  6. Synthesize a human-readable limitations summary for attending clinicians.

Key Principles:
  - Entirely rule-based and deterministic (no LLM calls, zero hallucinations).
  - Safety-first: abstains if evidence is insufficient (<0.30 score), if claims
    are refuted, or if medication data is missing while drug interactions exist.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

from app.models.schemas import (
    AbstentionDecision,
    ClaimSupportStatus,
    DrugInteraction,
    DrugInteractionResult,
    EvidenceBundle,
    EvidenceSufficiency,
    GradedEvidenceBundle,
    MissingFieldFlags,
    NormalizedPatientContext,
    UncertaintyItem,
    UncertaintyReport,
    UncertaintyType,
    VerificationSummary,
)

logger = logging.getLogger(__name__)

# Minimum evidence threshold required for the system not to abstain
MINIMUM_EVIDENCE_THRESHOLD: float = 0.3

# Required passage count and high-quality count for sufficiency
MIN_PASSAGE_COUNT: int = 3
MIN_HIGH_QUALITY_COUNT: int = 1
HIGH_QUALITY_RECENCY_WEIGHT: float = 0.8

# Critical threshold for claim support rate from verification
MIN_CLAIM_SUPPORT_RATE: float = 0.5

# Medication-related field names to monitor for safety hazards
MEDICATION_RELATED_FIELDS: set[str] = {
    "current_medications",
    "allergy_status",
}


class UncertaintyAgent:
    """
    Evaluates clinical case completeness and evidence sufficiency to decide
    whether the system should abstain or proceed with caveats.

    Usage:
        agent = UncertaintyAgent()
        report = await agent.assess(
            normalized_context=context,
            evidence_bundle=bundle,
            graded_evidence=graded,
            verification_summary=summary,
        )
    """

    def __init__(self) -> None:
        """Initialize the deterministic UncertaintyAgent."""
        logger.debug("Initialized UncertaintyAgent (Phase 7 rule-based engine)")

    async def assess(
        self,
        normalized_context: NormalizedPatientContext,
        evidence_bundle: EvidenceBundle,
        graded_evidence: Optional[GradedEvidenceBundle] = None,
        verification_summary: Optional[VerificationSummary] = None,
        drug_interactions: Optional[Union[DrugInteractionResult, List[DrugInteraction], bool]] = None,
    ) -> UncertaintyReport:
        """
        Assess uncertainty and completeness across clinical context and evidence.

        Args:
            normalized_context: Output of Phase 3 Clinical Context Extraction Agent.
            evidence_bundle: Output of Phase 4 Evidence Retrieval Agent.
            graded_evidence: Optional output of Phase 5 Evidence Grading Agent.
            verification_summary: Optional output of Phase 6 Evidence Verification Agent.
            drug_interactions: Optional drug interaction results or flag for safety checks.

        Returns:
            UncertaintyReport with items, sufficiency, abstention decision, and summary.
        """
        logger.info("Assessing case uncertainty and evidence sufficiency")
        uncertainty_items: List[UncertaintyItem] = []

        # ── Step a: Patient information gaps ─────────────────────────────────
        missing_names, critical_missing, patient_gap_items = self._assess_patient_gaps(
            normalized_context
        )
        uncertainty_items.extend(patient_gap_items)

        # ── Step b: Evidence sufficiency ──────────────────────────────────────
        sufficiency, sufficiency_items = self._assess_evidence_sufficiency(
            evidence_bundle=evidence_bundle,
            graded_evidence=graded_evidence,
        )
        uncertainty_items.extend(sufficiency_items)

        # ── Step c: Verification-based uncertainty ────────────────────────────
        verification_items = self._assess_verification_uncertainty(verification_summary)
        uncertainty_items.extend(verification_items)

        # ── Step d: Safety concern & Abstention decision ───────────────────────
        safety_concern = self._detect_safety_concern(
            normalized_context=normalized_context,
            missing_names=missing_names,
            drug_interactions=drug_interactions,
        )
        if safety_concern:
            uncertainty_items.append(
                UncertaintyItem(
                    uncertainty_type=UncertaintyType.CLINICAL_COMPLEXITY,
                    description=(
                        "Critical safety concern: medication-related fields are missing "
                        "while active drug interactions or medication conflicts are present."
                    ),
                    affected_fields=[f for f in missing_names if f in MEDICATION_RELATED_FIELDS] or ["medications"],
                    severity="critical",
                    mitigation=(
                        "Obtain complete current medication regimen and allergy status "
                        "before evaluating pharmacological therapy."
                    ),
                )
            )

        abstention_decision = self._determine_abstention(
            sufficiency_score=sufficiency.sufficiency_score,
            safety_concern=safety_concern,
            verification_summary=verification_summary,
        )

        # ── Step e: Overall uncertainty level ─────────────────────────────────
        overall_uncertainty_level = self._calculate_overall_level(
            uncertainty_items=uncertainty_items,
            critical_missing=critical_missing,
            sufficiency_score=sufficiency.sufficiency_score,
            should_abstain=abstention_decision.should_abstain,
        )

        # ── Step f: Limitations summary ───────────────────────────────────────
        limitations_summary = self._generate_limitations_summary(
            overall_level=overall_uncertainty_level,
            missing_names=missing_names,
            critical_missing=critical_missing,
            sufficiency=sufficiency,
            abstention=abstention_decision,
            verification_summary=verification_summary,
        )

        logger.info(
            "Uncertainty assessment completed: level=%s, abstain=%s, sufficiency_score=%.2f",
            overall_uncertainty_level,
            abstention_decision.should_abstain,
            sufficiency.sufficiency_score,
        )

        return UncertaintyReport(
            uncertainty_items=uncertainty_items,
            abstention_decision=abstention_decision,
            evidence_sufficiency=sufficiency,
            missing_patient_info=missing_names,
            overall_uncertainty_level=overall_uncertainty_level,
            limitations_summary=limitations_summary,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private Assessment Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _assess_patient_gaps(
        self,
        normalized_context: Optional[NormalizedPatientContext],
    ) -> tuple[List[str], bool, List[UncertaintyItem]]:
        """
        Step a: Check missing_fields from NormalizedPatientContext.
        Each missing field -> UncertaintyItem(type=PATIENT_INFO_GAP).
        If critical_missing is True, severity='critical'.
        """
        if normalized_context is None:
            return [], False, []

        missing_flags = normalized_context.missing_fields or MissingFieldFlags()
        missing_names = (
            missing_flags.missing_field_names()
            if hasattr(missing_flags, "missing_field_names")
            else []
        )
        critical_missing = bool(
            getattr(missing_flags, "critical_missing", False)
            or getattr(normalized_context, "has_critical_missing", False)
        )

        severity = "critical" if critical_missing else "moderate"
        items: List[UncertaintyItem] = []

        for fname in missing_names:
            field_label = fname.replace("_", " ")
            items.append(
                UncertaintyItem(
                    uncertainty_type=UncertaintyType.PATIENT_INFO_GAP,
                    description=f"Patient {field_label} is not documented in clinical record.",
                    affected_fields=[fname],
                    severity=severity,
                    mitigation=f"Obtain and verify {field_label} with patient or chart review.",
                )
            )

        # Also register any detected inconsistencies as clinical complexity
        for inc in getattr(normalized_context, "inconsistencies", []) or []:
            inc_sev = "critical" if getattr(inc, "severity", "") == "critical" else "high"
            items.append(
                UncertaintyItem(
                    uncertainty_type=UncertaintyType.CLINICAL_COMPLEXITY,
                    description=f"Inconsistency detected in {inc.field}: {inc.description}",
                    affected_fields=[inc.field],
                    severity=inc_sev,
                    mitigation="Reconcile contradictory patient data before confirming plan.",
                )
            )

        return missing_names, critical_missing, items

    def _assess_evidence_sufficiency(
        self,
        evidence_bundle: Optional[EvidenceBundle],
        graded_evidence: Optional[GradedEvidenceBundle],
    ) -> tuple[EvidenceSufficiency, List[UncertaintyItem]]:
        """
        Step b: Count passages, count high-quality ones (recency_weight >= 0.8 from graded evidence).
        If fewer than 3 passages or fewer than 1 high-quality, mark insufficient.
        Compute sufficiency_score = min(1.0, evidence_count / 5.0 * 0.5 + high_quality_count / 3.0 * 0.5).
        """
        passages = (
            evidence_bundle.passages
            if (evidence_bundle and hasattr(evidence_bundle, "passages"))
            else []
        )
        evidence_count = len(passages)

        graded_passages = (
            graded_evidence.graded_passages
            if (graded_evidence and hasattr(graded_evidence, "graded_passages"))
            else []
        )
        high_quality_count = sum(
            1
            for g in graded_passages
            if getattr(g, "recency_weight", 0.0) >= HIGH_QUALITY_RECENCY_WEIGHT
        )

        # Check sufficiency requirements
        coverage_gaps: List[str] = []
        if evidence_count < MIN_PASSAGE_COUNT:
            coverage_gaps.append(
                f"Total evidence passages ({evidence_count}) is below minimum requirement of {MIN_PASSAGE_COUNT}."
            )
        if high_quality_count < MIN_HIGH_QUALITY_COUNT:
            coverage_gaps.append(
                f"High-quality recent evidence passages ({high_quality_count}) is below minimum requirement of {MIN_HIGH_QUALITY_COUNT} (recency_weight >= {HIGH_QUALITY_RECENCY_WEIGHT})."
            )

        is_sufficient = (evidence_count >= MIN_PASSAGE_COUNT) and (
            high_quality_count >= MIN_HIGH_QUALITY_COUNT
        )

        # Compute sufficiency score
        raw_score = (evidence_count / 5.0) * 0.5 + (high_quality_count / 3.0) * 0.5
        sufficiency_score = min(1.0, max(0.0, round(raw_score, 4)))

        sufficiency = EvidenceSufficiency(
            is_sufficient=is_sufficient,
            sufficiency_score=sufficiency_score,
            evidence_count=evidence_count,
            high_quality_evidence_count=high_quality_count,
            coverage_gaps=coverage_gaps,
            threshold_used=MINIMUM_EVIDENCE_THRESHOLD,
        )

        items: List[UncertaintyItem] = []
        if not is_sufficient:
            items.append(
                UncertaintyItem(
                    uncertainty_type=UncertaintyType.EVIDENCE_INSUFFICIENT,
                    description=(
                        f"Retrieved clinical evidence is insufficient: {evidence_count} total "
                        f"passage(s), {high_quality_count} high-quality recent passage(s)."
                    ),
                    affected_fields=["evidence_passages"],
                    severity="critical" if sufficiency_score < MINIMUM_EVIDENCE_THRESHOLD else "high",
                    mitigation="Retrieve additional guidelines and peer-reviewed studies to cover clinical gaps.",
                )
            )

        return sufficiency, items

    def _assess_verification_uncertainty(
        self,
        verification_summary: Optional[VerificationSummary],
    ) -> List[UncertaintyItem]:
        """
        Step c: Check verification summary metrics.
        - If refuted_count > 0, add EVIDENCE_CONFLICTING item.
        - If claim_support_rate < 0.5, add EVIDENCE_INSUFFICIENT item.
        """
        if verification_summary is None:
            return []

        items: List[UncertaintyItem] = []
        refuted_count = getattr(verification_summary, "refuted_count", 0)
        supported_count = getattr(verification_summary, "supported_count", 0)
        support_rate = getattr(verification_summary, "claim_support_rate", 0.0)

        # Contradicted claims
        if refuted_count > 0:
            severity = "critical" if refuted_count > supported_count else "high"
            items.append(
                UncertaintyItem(
                    uncertainty_type=UncertaintyType.EVIDENCE_CONFLICTING,
                    description=(
                        f"Evidence verification identified {refuted_count} refuted claim(s) "
                        "contradicted by cited clinical literature."
                    ),
                    affected_fields=["verified_claims"],
                    severity=severity,
                    mitigation="Review and revise contradicted statements against authoritative evidence.",
                )
            )

        # Low claim support rate
        if support_rate < MIN_CLAIM_SUPPORT_RATE:
            severity = "high" if support_rate < 0.3 else "moderate"
            items.append(
                UncertaintyItem(
                    uncertainty_type=UncertaintyType.EVIDENCE_INSUFFICIENT,
                    description=(
                        f"Claim support rate is low ({support_rate:.1%}); "
                        f"less than 50% of claims are substantiated by cited evidence."
                    ),
                    affected_fields=["claim_support_rate"],
                    severity=severity,
                    mitigation="Supply validated citations for uncited or weakly supported recommendations.",
                )
            )

        return items

    def _detect_safety_concern(
        self,
        normalized_context: Optional[NormalizedPatientContext],
        missing_names: List[str],
        drug_interactions: Optional[Union[DrugInteractionResult, List[DrugInteraction], bool]] = None,
    ) -> bool:
        """
        Step d helper: Check if safety concern is present.
        safety_concern = True if any medication-related fields are missing AND
        there are drug interactions present.
        """
        # Check if any medication-related field is missing
        missing_set = set(missing_names)
        has_missing_meds = bool(
            missing_set & MEDICATION_RELATED_FIELDS
            or any("medication" in f or "drug" in f or "allerg" in f for f in missing_set)
        )
        if normalized_context and normalized_context.missing_fields:
            has_missing_meds = (
                has_missing_meds
                or normalized_context.missing_fields.current_medications
                or normalized_context.missing_fields.allergy_status
            )

        # Check if drug interactions are present
        interactions_present = False
        if drug_interactions is not None:
            if isinstance(drug_interactions, DrugInteractionResult):
                interactions_present = len(drug_interactions.interactions_found) > 0
            elif isinstance(drug_interactions, list):
                interactions_present = len(drug_interactions) > 0
            elif isinstance(drug_interactions, bool):
                interactions_present = drug_interactions

        # If not passed via argument, inspect normalized_context inconsistencies
        if not interactions_present and normalized_context:
            for inc in getattr(normalized_context, "inconsistencies", []) or []:
                fld = getattr(inc, "field", "").lower()
                desc = getattr(inc, "description", "").lower()
                if "interaction" in fld or "interaction" in desc or "medication_allergy" in fld:
                    interactions_present = True
                    break

        return bool(has_missing_meds and interactions_present)

    def _determine_abstention(
        self,
        sufficiency_score: float,
        safety_concern: bool,
        verification_summary: Optional[VerificationSummary],
    ) -> AbstentionDecision:
        """
        Step d: Abstention decision.
        should_abstain = True if (sufficiency_score < 0.3) OR (safety_concern detected) OR (refuted_count > supported_count).
        """
        refuted_count = getattr(verification_summary, "refuted_count", 0) if verification_summary else 0
        supported_count = getattr(verification_summary, "supported_count", 0) if verification_summary else 0

        min_threshold_met = sufficiency_score >= MINIMUM_EVIDENCE_THRESHOLD

        reasons: List[str] = []
        if not min_threshold_met:
            reasons.append(
                f"Evidence sufficiency score ({sufficiency_score:.2f}) is below minimum threshold ({MINIMUM_EVIDENCE_THRESHOLD:.2f})"
            )
        if safety_concern:
            reasons.append(
                "Critical safety concern: medication-related fields are missing while drug interactions/conflicts exist"
            )
        if refuted_count > supported_count:
            reasons.append(
                f"Refuted claims ({refuted_count}) exceed supported claims ({supported_count})"
            )

        should_abstain = len(reasons) > 0
        abstention_reason = "; ".join(reasons) if should_abstain else None

        # Partial response disallowed if safety concern exists, permitted for pure evidence shortfall
        partial_response_allowed = not safety_concern

        return AbstentionDecision(
            should_abstain=should_abstain,
            abstention_reason=abstention_reason,
            partial_response_allowed=partial_response_allowed,
            minimum_evidence_threshold_met=min_threshold_met,
            safety_concern=safety_concern,
        )

    def _calculate_overall_level(
        self,
        uncertainty_items: List[UncertaintyItem],
        critical_missing: bool,
        sufficiency_score: float,
        should_abstain: bool,
    ) -> str:
        """
        Step e: Overall uncertainty level calculation.
          - 'critical' if abstention.
          - 'high' if sufficiency < 0.4 or critical missing (or critical severity item).
          - 'moderate' if sufficiency 0.4-0.7.
          - 'low' if no critical items and sufficiency > 0.7.
        """
        has_critical_item = any(item.severity == "critical" for item in uncertainty_items)

        if should_abstain:
            return "critical"
        if critical_missing or has_critical_item or sufficiency_score < 0.4:
            return "high"
        if 0.4 <= sufficiency_score <= 0.7:
            return "moderate"
        return "low"

    def _generate_limitations_summary(
        self,
        overall_level: str,
        missing_names: List[str],
        critical_missing: bool,
        sufficiency: EvidenceSufficiency,
        abstention: AbstentionDecision,
        verification_summary: Optional[VerificationSummary],
    ) -> str:
        """
        Step f: Human-readable narrative summarizing all identified limitations.
        """
        lines: List[str] = []

        # 1. Abstention / High-level status
        if abstention.should_abstain:
            lines.append(
                f"RECOMMENDATION: ABSTAIN FROM ACTIONABLE GUIDANCE. "
                f"Reason: {abstention.abstention_reason}."
            )
        else:
            lines.append(f"Overall Clinical Uncertainty Level: {overall_level.upper()}.")

        # 2. Patient information completeness
        if missing_names:
            formatted_fields = ", ".join(f.replace("_", " ") for f in missing_names)
            status_note = "CRITICAL: Multiple required fields absent." if critical_missing else "Gaps noted."
            lines.append(f"Patient Data Incompleteness ({status_note}): Missing {formatted_fields}.")
        else:
            lines.append("Patient profile is clinically complete across standard intake parameters.")

        # 3. Evidence sufficiency
        suff_status = "Sufficient" if sufficiency.is_sufficient else "Insufficient"
        lines.append(
            f"Evidence Base ({suff_status}): score {sufficiency.sufficiency_score:.2f}/1.00 "
            f"({sufficiency.evidence_count} passage(s), {sufficiency.high_quality_evidence_count} high-quality recent)."
        )
        if sufficiency.coverage_gaps:
            lines.append(f"Coverage Gaps: {'; '.join(sufficiency.coverage_gaps)}")

        # 4. Verification summary
        if verification_summary is not None:
            claims_count = getattr(verification_summary, "total_claims", 0)
            sup_count = getattr(verification_summary, "supported_count", 0)
            ref_count = getattr(verification_summary, "refuted_count", 0)
            rate = getattr(verification_summary, "claim_support_rate", 0.0)
            lines.append(
                f"Claim Verification: {sup_count}/{claims_count} supported, "
                f"{ref_count} refuted (claim support rate: {rate:.1%})."
            )

        # 5. Safety alert
        if abstention.safety_concern:
            lines.append(
                "SAFETY ALERT: Medication regimen is incomplete while active interactions or conflicts were identified. "
                "Clinician verification mandatory before administering medication."
            )

        return " ".join(lines)
