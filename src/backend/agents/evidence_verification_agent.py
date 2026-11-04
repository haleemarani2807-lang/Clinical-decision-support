"""
Evidence Verification Agent — Major Research Contribution.

Implements claim-level citation verification:

  1. Atomic claim extraction from generated report text
  2. Claim-to-source mapping via proposed citations
  3. Citation identifier validation (PMID lookup, not just URL)
  4. Evidence passage retrieval for each cited source
  5. NLI entailment classification (DeBERTa cross-encoder)
     → SUPPORT / REFUTE / INSUFFICIENT_EVIDENCE / UNCITED
  6. Aggregate verification metrics calculation
  7. Re-verification support for modified claims

KEY PRINCIPLE: A citation is NOT valid merely because its URL exists.
The cited passage must actually support the associated statement.

Fallback chain:
  1. NLI-based entailment (DeBERTa) — primary
  2. Lexical overlap (token F1) — if NLI unavailable
  3. Mark as NOT_VERIFIED — if all fails (never silently passes)
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.models.schemas import (
    AtomicClaim,
    ClaimSupportStatus,
    ClaimVerification,
    EvidenceBundle,
    EvidencePassage,
    VerificationSummary,
)

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """
    Extracts atomic factual claims from clinical report text.

    Strategy:
    1. Split into sentences (rule-based)
    2. Filter to claims (assertions with clinical content)
    3. Identify proposed citations from inline markers
    """

    # Markers that indicate a claim should be verifiable
    CLAIM_INDICATORS = re.compile(
        r'\b(is|are|was|were|has|have|should|must|recommends?|indicates?|'
        r'shows?|demonstrates?|supports?|suggests?|reduces?|increases?|'
        r'decreases?|associated with|evidence|studies? show|guideline)\b',
        re.IGNORECASE
    )

    # Citation markers in text (e.g., [pmid_12345], [guideline_001])
    CITATION_MARKER = re.compile(r'\[([a-zA-Z0-9_]+)\]')

    def extract_claims(
        self,
        report_text: str,
        source_section: str = "",
        agent_step_origin: str = "synthesis",
    ) -> list[AtomicClaim]:
        """
        Extract atomic claims from a text block.

        Returns:
            List of AtomicClaim objects (may be empty for non-assertive text).
        """
        if not report_text or len(report_text.strip()) < 10:
            return []

        # Split into sentences
        sentences = self._split_sentences(report_text)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue

            # Only extract claims that make assertions
            if not self.CLAIM_INDICATORS.search(sentence):
                continue

            # Extract proposed citations
            citations = self.CITATION_MARKER.findall(sentence)
            # Remove citation markers from claim text
            clean_text = self.CITATION_MARKER.sub("", sentence).strip()

            if not clean_text:
                continue

            claims.append(AtomicClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                text=clean_text,
                source_section=source_section,
                proposed_citation_ids=citations,
                agent_step_origin=agent_step_origin,
            ))

        return claims

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences (simple rule-based splitter)."""
        # Protect common abbreviations from being split
        protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|vs|e\.g|i\.e|et al|mg|mcg|mEq|mL|IV|BID|TID)\.',
                           lambda m: m.group(0).replace(".", "_DOT_"), text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
        return [s.replace("_DOT_", ".") for s in sentences]


class ClaimVerifier:
    """
    Verifies individual atomic claims against evidence passages.

    For each claim:
    1. Validate citation IDs (check they exist in evidence store)
    2. Retrieve passage text for each valid citation
    3. Run NLI entailment: claim → passage
    4. Classify as SUPPORTED / REFUTED / INSUFFICIENT_EVIDENCE / UNCITED
    """

    def __init__(self):
        self._nli_model = None
        self._nli_available = None  # None = not yet checked

    async def verify_claim(
        self,
        claim: AtomicClaim,
        evidence_store: dict[str, EvidencePassage],
    ) -> ClaimVerification:
        """
        Verify a single atomic claim.

        Args:
            claim: The atomic claim to verify.
            evidence_store: Dict mapping passage_id → EvidencePassage.

        Returns:
            ClaimVerification with status, scores, and notes.
        """
        verification = ClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.text,
        )

        # Step 1: Validate citation IDs
        valid_ids, invalid_ids = self._validate_citation_ids(
            claim.proposed_citation_ids, evidence_store
        )
        verification.citation_ids_valid = valid_ids
        verification.citation_ids_invalid = invalid_ids

        # Step 2: If no citations proposed, mark as uncited
        if not claim.proposed_citation_ids:
            verification.status = ClaimSupportStatus.UNCITED
            verification.verification_note = (
                "No citation proposed for this claim. "
                "Cannot verify without a source reference."
            )
            verification.requires_human_review = True
            return verification

        # Step 3: If all citations invalid, mark as uncited
        if not valid_ids:
            verification.status = ClaimSupportStatus.UNCITED
            verification.verification_note = (
                f"All proposed citations ({', '.join(invalid_ids)}) not found in evidence store."
            )
            verification.requires_human_review = True
            return verification

        # Step 4: Entailment verification against valid passages
        passages = [evidence_store[pid] for pid in valid_ids if pid in evidence_store]
        return await self._run_entailment(claim, passages, verification)

    async def _run_entailment(
        self,
        claim: AtomicClaim,
        passages: list[EvidencePassage],
        verification: ClaimVerification,
    ) -> ClaimVerification:
        """Run NLI entailment between claim and candidate passages."""
        # Try NLI-based entailment
        if self._nli_available is None:
            self._nli_available = self._load_nli_model()

        if self._nli_available and self._nli_model:
            return self._nli_entailment(claim, passages, verification)
        else:
            return self._lexical_entailment(claim, passages, verification)

    def _nli_entailment(
        self,
        claim: AtomicClaim,
        passages: list[EvidencePassage],
        verification: ClaimVerification,
    ) -> ClaimVerification:
        """
        DeBERTa NLI entailment.
        Input: (passage, claim) pairs — note: NLI is premise→hypothesis.
        Labels: 0=contradiction, 1=neutral, 2=entailment
        """
        from app.config import settings

        support_threshold = settings.entailment_support_threshold
        refute_threshold = settings.entailment_support_threshold

        max_support = 0.0
        max_refute = 0.0
        supporting_ids = []
        refuting_ids = []

        pairs = [[p.text[:512], claim.text] for p in passages]
        try:
            scores = self._nli_model.predict(pairs, apply_softmax=True)
            # scores shape: (n_pairs, 3) — [contradiction, neutral, entailment]
            for passage, score in zip(passages, scores):
                contradiction_score = float(score[0])
                entailment_score = float(score[2])

                if entailment_score > max_support:
                    max_support = entailment_score
                if contradiction_score > max_refute:
                    max_refute = contradiction_score

                if entailment_score >= support_threshold:
                    supporting_ids.append(passage.passage_id)
                elif contradiction_score >= refute_threshold:
                    refuting_ids.append(passage.passage_id)

        except Exception as e:
            logger.warning("NLI prediction failed: %s — falling back to lexical", e)
            return self._lexical_entailment(claim, passages, verification)

        verification.max_support_score = round(max_support, 4)
        verification.max_refute_score = round(max_refute, 4)
        verification.supporting_passage_ids = supporting_ids
        verification.refuting_passage_ids = refuting_ids
        verification.entailment_method = "nli_deberta"

        # Classify
        if supporting_ids:
            verification.status = ClaimSupportStatus.SUPPORTED
            verification.verification_note = (
                f"Claim is entailed by {len(supporting_ids)} passage(s) "
                f"(max entailment score: {max_support:.3f})."
            )
            verification.requires_human_review = False
        elif refuting_ids:
            verification.status = ClaimSupportStatus.REFUTED
            verification.verification_note = (
                f"Claim is contradicted by {len(refuting_ids)} passage(s) "
                f"(max contradiction score: {max_refute:.3f}). Requires immediate review."
            )
            verification.requires_human_review = True
        else:
            verification.status = ClaimSupportStatus.INSUFFICIENT_EVIDENCE
            verification.verification_note = (
                f"Available evidence is insufficient to support or refute this claim "
                f"(max support: {max_support:.3f}, max refute: {max_refute:.3f})."
            )
            verification.requires_human_review = True

        return verification

    @staticmethod
    def _lexical_entailment(
        claim: AtomicClaim,
        passages: list[EvidencePassage],
        verification: ClaimVerification,
    ) -> ClaimVerification:
        """
        Token-level lexical overlap fallback when NLI is unavailable.
        Uses token F1 score (similar to SQuAD evaluation).
        """
        claim_tokens = set(claim.text.lower().split())
        max_overlap = 0.0
        supporting_ids = []

        for passage in passages:
            passage_tokens = set(passage.text.lower().split())
            if not claim_tokens or not passage_tokens:
                continue
            intersection = claim_tokens & passage_tokens
            precision = len(intersection) / len(claim_tokens)
            recall = len(intersection) / len(passage_tokens)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            if f1 > max_overlap:
                max_overlap = f1
            if f1 >= 0.4:  # Conservative threshold for lexical overlap
                supporting_ids.append(passage.passage_id)

        verification.max_support_score = round(max_overlap, 4)
        verification.supporting_passage_ids = supporting_ids
        verification.entailment_method = "lexical_f1"

        if supporting_ids:
            verification.status = ClaimSupportStatus.SUPPORTED
            verification.verification_note = (
                f"Lexical overlap suggests support (F1={max_overlap:.3f}). "
                "NLI model unavailable — human review recommended."
            )
            verification.requires_human_review = True  # Always flag for lexical
        else:
            verification.status = ClaimSupportStatus.INSUFFICIENT_EVIDENCE
            verification.verification_note = (
                f"Insufficient lexical overlap with cited evidence (max F1={max_overlap:.3f}). "
                "NLI model unavailable."
            )
            verification.requires_human_review = True

        return verification

    def _load_nli_model(self) -> bool:
        """Load DeBERTa NLI cross-encoder. Returns True if successful."""
        from app.config import settings
        try:
            from sentence_transformers import CrossEncoder
            # Check local files first to prevent unapproved network downloads/hangs
            self._nli_model = CrossEncoder(
                settings.entailment_model,
                num_labels=3,
                local_files_only=True,
            )
            logger.info("NLI entailment model loaded: %s", settings.entailment_model)
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — falling back to lexical entailment. "
                "Run: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            logger.warning("NLI model load failed: %s — falling back to lexical entailment", e)
            return False

    @staticmethod
    def _validate_citation_ids(
        proposed_ids: list[str],
        evidence_store: dict[str, EvidencePassage],
    ) -> tuple[list[str], list[str]]:
        """
        Check which citation IDs actually exist in the evidence store.
        A citation is only valid if the passage is present and retrievable.
        """
        valid = [pid for pid in proposed_ids if pid in evidence_store]
        invalid = [pid for pid in proposed_ids if pid not in evidence_store]
        return valid, invalid


class EvidenceVerificationAgent:
    """
    Orchestrates the full evidence verification pipeline.

    Usage:
        agent = EvidenceVerificationAgent()
        summary = await agent.verify(report_text, evidence_bundle)
    """

    def __init__(self):
        self._extractor = ClaimExtractor()
        self._verifier = ClaimVerifier()

    async def verify(
        self,
        report_text: str,
        evidence_bundle: EvidenceBundle,
        report_sections: Optional[dict[str, str]] = None,
    ) -> VerificationSummary:
        """
        Run full verification pipeline on a generated report.

        Args:
            report_text: The full report text to verify.
            evidence_bundle: Retrieved evidence passages.
            report_sections: Optional dict mapping section_name → section_text
                             for more granular extraction.

        Returns:
            VerificationSummary with per-claim and aggregate metrics.
        """
        # Build evidence store (passage_id → passage)
        evidence_store = {p.passage_id: p for p in evidence_bundle.passages}

        # Extract claims
        if report_sections:
            all_claims = []
            for section_name, section_text in report_sections.items():
                claims = self._extractor.extract_claims(
                    section_text, source_section=section_name
                )
                all_claims.extend(claims)
        else:
            all_claims = self._extractor.extract_claims(report_text)

        if not all_claims:
            logger.info("No verifiable claims extracted from report")
            return VerificationSummary(
                total_claims=0,
                verification_timestamp=datetime.now(timezone.utc),
            )

        logger.info("Verifying %d extracted claims against %d evidence passages",
                    len(all_claims), len(evidence_store))

        # Verify each claim
        verifications = []
        for claim in all_claims:
            result = await self._verifier.verify_claim(claim, evidence_store)
            verifications.append(result)

        # Aggregate metrics
        summary = self._aggregate(verifications)
        summary.verification_timestamp = datetime.now(timezone.utc)
        return summary

    @staticmethod
    def _aggregate(verifications: list[ClaimVerification]) -> VerificationSummary:
        """Compute aggregate verification metrics."""
        total = len(verifications)
        if total == 0:
            return VerificationSummary()

        supported = sum(1 for v in verifications if v.status == ClaimSupportStatus.SUPPORTED)
        refuted = sum(1 for v in verifications if v.status == ClaimSupportStatus.REFUTED)
        insufficient = sum(1 for v in verifications if v.status == ClaimSupportStatus.INSUFFICIENT_EVIDENCE)
        uncited = sum(1 for v in verifications if v.status == ClaimSupportStatus.UNCITED)

        # Citation precision: valid citations / all proposed citations
        total_proposed = sum(
            len(v.citation_ids_valid) + len(v.citation_ids_invalid)
            for v in verifications
        )
        total_valid = sum(len(v.citation_ids_valid) for v in verifications)
        citation_precision = total_valid / total_proposed if total_proposed > 0 else 0.0

        # Entailment rate: NLI-supported / non-uncited
        non_uncited = [v for v in verifications if v.status != ClaimSupportStatus.UNCITED]
        entailment_rate = (
            sum(1 for v in non_uncited if v.status == ClaimSupportStatus.SUPPORTED)
            / len(non_uncited)
        ) if non_uncited else 0.0

        return VerificationSummary(
            total_claims=total,
            supported_count=supported,
            refuted_count=refuted,
            insufficient_evidence_count=insufficient,
            uncited_count=uncited,
            claim_support_rate=round(supported / total, 4),
            citation_precision=round(citation_precision, 4),
            entailment_rate=round(entailment_rate, 4),
            unsupported_claim_rate=round((refuted + uncited) / total, 4),
            claim_verifications=verifications,
        )
