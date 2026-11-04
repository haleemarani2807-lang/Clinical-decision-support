"""
Evidence Grading Agent – assigns quality grades to retrieved evidence passages.

Grading criteria (based on GRADE framework):
  • Study design (RCT, meta‑analysis, etc.)
  • Sample size & statistical power (approximated by publication year)
  • Recency (discount older publications)
  • Explicit study limitations (if present in abstract)

Outputs a GradedEvidenceBundle that aligns with the schema definitions.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional

from app.models.schemas import (
    EvidenceBundle,
    EvidencePassage,
    EvidenceGrade,
    GradedEvidenceBundle,
    PublicationType,
)

logger = logging.getLogger(__name__)


class EvidenceGrader:
    """Assigns grades to evidence passages using rule‑based heuristics.

    The grader is intentionally lightweight – it does not require a heavy model.
    It can be swapped out later for a learned classifier.
    """

    # Mapping of regex patterns to PublicationType
    _type_patterns = {
        PublicationType.RCT: re.compile(r"randomized controlled trial|rct", re.I),
        PublicationType.META_ANALYSIS: re.compile(r"meta[- ]analysis", re.I),
        PublicationType.SYSTEMATIC_REVIEW: re.compile(r"systematic review", re.I),
        PublicationType.COHORT_STUDY: re.compile(r"cohort", re.I),
        PublicationType.CASE_CONTROL: re.compile(r"case[- ]control", re.I),
        PublicationType.CASE_REPORT: re.compile(r"case report", re.I),
        PublicationType.CLINICAL_GUIDELINE: re.compile(r"guideline|clinical practice guideline", re.I),
        PublicationType.EXPERT_OPINION: re.compile(r"expert opinion|consensus", re.I),
        PublicationType.REVIEW_ARTICLE: re.compile(r"review", re.I),
    }

    def grade_bundle(self, bundle: EvidenceBundle) -> GradedEvidenceBundle:
        """Grade every passage in an EvidenceBundle.

        Returns:
            GradedEvidenceBundle with parallel lists of graded and ungraded passages.
        """
        graded: List[EvidenceGrade] = []
        ungraded: List[EvidencePassage] = []

        for passage in bundle.passages:
            try:
                grade = self._grade_passage(passage)
                graded.append(grade)
            except Exception as e:
                logger.debug("Failed to grade passage %s: %s", passage.passage_id, e)
                ungraded.append(passage)

        return GradedEvidenceBundle(
            query=bundle.query,
            graded_passages=graded,
            ungraded_passages=ungraded,
        )

    def _grade_passage(self, passage: EvidencePassage) -> EvidenceGrade:
        """Produce an EvidenceGrade for a single passage.

        Heuristics applied:
        1. Detect study type via regex on title/abstract.
        2. Compute relevance_score (placeholder – set to 1.0 for now).
        3. Derive recency weight from publication year (0.5‑1.0 scale).
        4. Scan abstract for common limitation phrases.
        5. Infer source authority from source enum.
        """
        # 1. Publication type detection
        pub_type = self._detect_publication_type(passage.title, passage.text)

        # 2. Relevance – simple placeholder (future work: use cross‑encoder)
        relevance = 1.0

        # 3. Recency weight
        recency = self._recency_weight(passage.publication_date)

        # 4. Limitations extraction (simple keyword search)
        limitations = self._extract_limitations(passage.text)

        # 5. Authority – heuristic based on source
        authority = self._source_authority(passage.source)

        # Publication year extraction for optional field
        pub_year = self._extract_year(passage.publication_date)

        return EvidenceGrade(
            passage_id=passage.passage_id,
            publication_type=pub_type,
            relevance_score=relevance,
            source_authority=authority,
            publication_year=pub_year,
            recency_weight=recency,
            study_limitations=limitations,
            grade_notes="Rule‑based grading; replace with ML model for research.",
            clinically_relevant=None,
            evidence_quality_note="",
        )

    def _detect_publication_type(self, title: str, abstract: str) -> PublicationType:
        """Match title/abstract against known patterns to infer type."""
        combined = f"{title} {abstract}".lower()
        for ptype, pattern in self._type_patterns.items():
            if pattern.search(combined):
                return ptype
        return PublicationType.UNKNOWN

    @staticmethod
    def _recency_weight(pub_date: Optional[str]) -> float:
        """Compute a weight between 0.3 and 1.0 based on how recent a publication is.

        If the date is missing or unparsable, return a neutral weight of 0.6.
        """
        try:
            if not pub_date:
                return 0.6
            # Accept formats YYYY or YYYY‑MM or YYYY‑MM‑DD
            year = int(pub_date.split("-")[0])
            current_year = datetime.utcnow().year
            age = current_year - year
            if age < 0:
                age = 0
            # Linear decay: 0‑5 years => 1.0, 6‑10 => 0.8, 11‑20 => 0.6, >20 => 0.4
            if age <= 5:
                return 1.0
            if age <= 10:
                return 0.8
            if age <= 20:
                return 0.6
            return 0.4
        except Exception:
            return 0.6

    @staticmethod
    def _extract_limitations(text: str) -> List[str]:
        """Simple keyword based extraction of limitation statements.

        Looks for phrases like "small sample", "single centre", "short follow‑up",
        "preliminary", "limited power".
        """
        limitation_keywords = [
            "small sample",
            "single centre",
            "single center",
            "short follow-up",
            "preliminary",
            "limited power",
            "retrospective",
            "observational",
            "case series",
        ]
        lowered = text.lower()
        found = [kw for kw in limitation_keywords if kw in lowered]
        return found

    @staticmethod
    def _source_authority(source) -> str:
        """Heuristic authority label based on source enum.

        PubMed & ClinicalGuideline are considered high authority, others moderate.
        """
        if source.name in {"PUBMED", "GUIDELINE"}:
            return "high"
        if source.name == "OPENFDA":
            return "moderate"
        return "low"

    @staticmethod
    def _extract_year(pub_date: Optional[str]) -> Optional[int]:
        """Return integer year if a YYYY pattern is present."""
        if not pub_date:
            return None
        try:
            return int(pub_date.split("-")[0])
        except Exception:
            return None
