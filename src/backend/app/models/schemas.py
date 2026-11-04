# [Track A: Baseline]
"""
Domain models for the Clinical Decision Support Agent.

These Pydantic models define the structured data flowing through the agent pipeline.
Every tool consumes and produces typed models — no loose dicts or unstructured text.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Confidence(str, Enum):
    VERY_LOW = "very low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very high"
    UNRECOGNIZED = "unrecognized — requires review"


class AgentStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ──────────────────────────────────────────────
# Patient Data Models
# ──────────────────────────────────────────────

class Medication(BaseModel):
    name: str = Field(..., description="Medication name")
    dose: Optional[str] = Field(None, description="Dosage, e.g. '10mg daily'")
    rxcui: Optional[str] = Field(None, description="RxNorm concept ID")


class LabResult(BaseModel):
    test_name: str = Field(..., description="Lab test name")
    value: str = Field(..., description="Result value with units")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    is_abnormal: Optional[bool] = Field(None, description="Whether result is abnormal")


class VitalSigns(BaseModel):
    blood_pressure: Optional[str] = None
    heart_rate: Optional[str] = None
    temperature: Optional[str] = None
    respiratory_rate: Optional[str] = None
    oxygen_saturation: Optional[str] = None
    weight: Optional[str] = None
    height: Optional[str] = None


class PatientProfile(BaseModel):
    """Structured patient profile — output of the Patient Data Parser tool."""
    age: Optional[int] = None
    gender: Gender = Gender.UNKNOWN
    chief_complaint: str = Field(..., description="Primary reason for visit")
    history_of_present_illness: str = Field("", description="HPI narrative")
    past_medical_history: List[str] = Field(default_factory=list)
    current_medications: List[Medication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    lab_results: List[LabResult] = Field(default_factory=list)
    vital_signs: Optional[VitalSigns] = None
    social_history: Optional[str] = None
    family_history: Optional[str] = None
    additional_notes: Optional[str] = None


# ──────────────────────────────────────────────
# Clinical Reasoning Models
# ──────────────────────────────────────────────

class DiagnosisCandidate(BaseModel):
    diagnosis: str = Field(..., description="Diagnosis name")
    icd10_code: Optional[str] = Field(None, description="ICD-10 code if known")
    likelihood: Confidence = Field(..., description="Estimated likelihood")
    likelihood_raw: Optional[str] = Field(None, description="Original model output if it didn't match a standard value")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence from patient data")
    reasoning: str = Field("", description="Clinical reasoning chain")

    @field_validator("likelihood", mode="before")
    @classmethod
    def normalize_likelihood(cls, v: str) -> str:
        """Map non-standard likelihood values to the closest Confidence enum member."""
        if isinstance(v, Confidence):
            return v
        raw = str(v).strip().lower()
        # Direct match
        for member in Confidence:
            if raw == member.value:
                return member
        # Fuzzy mapping for common LLM variations
        mapping = {
            "unlikely": "very low",
            "very unlikely": "very low",
            "rare": "very low",
            "possible": "low",
            "less likely": "low",
            "somewhat likely": "moderate",
            "medium": "moderate",
            "intermediate": "moderate",
            "likely": "high",
            "probable": "high",
            "very likely": "very high",
            "highly likely": "very high",
            "almost certain": "very high",
        }
        if raw in mapping:
            return Confidence(mapping[raw])
        # Unknown value — flag for clinical review, raw value preserved in likelihood_raw
        return Confidence.UNRECOGNIZED

    def __init__(self, **data):
        # Capture the raw value before validation normalizes it
        raw_val = data.get("likelihood")
        super().__init__(**data)
        if isinstance(raw_val, str):
            raw_lower = raw_val.strip().lower()
            known = {m.value for m in Confidence}
            # Also check the fuzzy mapping keys
            mapped = {
                "unlikely", "very unlikely", "rare", "possible", "less likely",
                "somewhat likely", "medium", "intermediate", "likely",
                "probable", "very likely", "highly likely", "almost certain",
            }
            if raw_lower not in known and raw_lower not in mapped:
                self.likelihood_raw = raw_val.strip()


class RecommendedAction(BaseModel):
    action: str = Field(..., description="Recommended action (test, referral, treatment)")
    priority: Severity = Field(..., description="Priority level")
    rationale: str = Field("", description="Why this action is recommended")


class ClinicalReasoningResult(BaseModel):
    """Output of the Clinical Reasoning Agent (MedGemma)."""
    differential_diagnosis: List[DiagnosisCandidate] = Field(
        default_factory=list, description="Ranked differential diagnosis"
    )
    risk_assessment: Optional[str] = Field(None, description="Overall risk assessment")
    recommended_workup: List[RecommendedAction] = Field(
        default_factory=list, description="Recommended tests, referrals, treatments"
    )
    reasoning_chain: str = Field("", description="Full chain-of-thought reasoning")


# ──────────────────────────────────────────────
# Drug Interaction Models
# ──────────────────────────────────────────────

class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: Severity
    description: str
    clinical_significance: Optional[str] = None
    source: str = Field("OpenFDA", description="Data source")


class DrugInteractionResult(BaseModel):
    """Output of the Drug Interaction Checker tool."""
    interactions_found: List[DrugInteraction] = Field(default_factory=list)
    medications_checked: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Guideline Retrieval Models
# ──────────────────────────────────────────────

class GuidelineExcerpt(BaseModel):
    title: str = Field(..., description="Guideline or source title")
    excerpt: str = Field(..., description="Relevant excerpt text")
    source: str = Field(..., description="Publication or organization")
    url: Optional[str] = None
    relevance_score: Optional[float] = None


class GuidelineRetrievalResult(BaseModel):
    """Output of the Guideline Retrieval (RAG) tool."""
    query: str = Field(..., description="The query used for retrieval")
    excerpts: List[GuidelineExcerpt] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Conflict Detection Models
# ──────────────────────────────────────────────

class ConflictType(str, Enum):
    OMISSION = "omission"                # Guideline recommends X, patient not receiving X
    CONTRADICTION = "contradiction"      # Patient's current care contradicts guideline
    DOSAGE = "dosage"                    # Dose adjustment criteria apply to this patient
    MONITORING = "monitoring"            # Required monitoring not documented/ordered
    ALLERGY_RISK = "allergy_risk"        # Guideline suggests drug patient is allergic to
    INTERACTION_GAP = "interaction_gap"  # Drug interaction not addressed in current plan


class ClinicalConflict(BaseModel):
    """A single detected conflict between guidelines and patient data."""
    conflict_type: ConflictType = Field(..., description="Category of the conflict")
    severity: Severity = Field(..., description="Potential clinical impact")

    @field_validator("conflict_type", mode="before")
    @classmethod
    def _normalise_conflict_type(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    @field_validator("severity", mode="before")
    @classmethod
    def _normalise_severity(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v
    guideline_source: str = Field("", description="Which guideline flagged this")
    guideline_text: str = Field("", description="What the guideline recommends")
    patient_data: str = Field("", description="Relevant patient data that conflicts")
    description: str = Field("", description="Plain-language explanation of the gap")
    suggested_resolution: Optional[str] = Field(
        None, description="Potential resolution for the clinician to consider"
    )


class ConflictDetectionResult(BaseModel):
    """Output of the Conflict Detection tool."""
    conflicts: List[ClinicalConflict] = Field(default_factory=list)
    guidelines_checked: int = Field(0, description="Number of guidelines compared")
    summary: str = Field("", description="Brief summary of conflict analysis")


# ──────────────────────────────────────────────
# Final CDS Report
# ──────────────────────────────────────────────

class CDSReport(BaseModel):
    """
    The final Clinical Decision Support report — synthesized by MedGemma
    from all tool outputs. This is what the clinician sees.
    """
    patient_summary: str = Field(..., description="Concise patient summary")
    differential_diagnosis: List[DiagnosisCandidate] = Field(default_factory=list)
    drug_interaction_warnings: List[DrugInteraction] = Field(default_factory=list)
    guideline_recommendations: List[str] = Field(
        default_factory=list, description="Guideline-concordant recommendations"
    )
    suggested_next_steps: List[RecommendedAction] = Field(default_factory=list)
    caveats: List[str] = Field(
        default_factory=list,
        description="Limitations, uncertainties, and disclaimers"
    )
    conflicts: List[ClinicalConflict] = Field(
        default_factory=list,
        description="Detected conflicts between guidelines and patient care"
    )
    sources_cited: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Agent Orchestration Models
# ──────────────────────────────────────────────

class AgentStep(BaseModel):
    """Represents a single step in the agent pipeline, streamed to the frontend."""
    step_id: str
    step_name: str
    status: AgentStepStatus = AgentStepStatus.PENDING
    tool_name: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class AgentState(BaseModel):
    """Full state of the agent pipeline for a given case."""
    case_id: str
    steps: List[AgentStep] = Field(default_factory=list)
    patient_profile: Optional[PatientProfile] = None
    clinical_reasoning: Optional[ClinicalReasoningResult] = None
    drug_interactions: Optional[DrugInteractionResult] = None
    guideline_retrieval: Optional[GuidelineRetrievalResult] = None
    conflict_detection: Optional[ConflictDetectionResult] = None
    final_report: Optional[CDSReport] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ──────────────────────────────────────────────
# API Request / Response Models
# ──────────────────────────────────────────────

class CaseSubmission(BaseModel):
    """API request to submit a new patient case for analysis."""
    patient_text: str = Field(
        ...,
        description="Free-text patient case description or structured data",
        min_length=10,
    )
    include_drug_check: bool = Field(True, description="Run drug interaction check")
    include_guidelines: bool = Field(True, description="Retrieve relevant guidelines")
    use_langgraph: Optional[bool] = Field(
        None,
        description="Enable 10-step LangGraph-based evidence-verified pipeline. Defaults to system config."
    )


class CaseResponse(BaseModel):
    """API response for a submitted case."""
    case_id: str
    status: str
    message: str


class CaseResult(BaseModel):
    """API response with the full case result."""
    case_id: str
    state: Union[EnhancedAgentState, AgentState]
    report: Optional[Union[EnhancedCDSReport, CDSReport]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Clinical Context Extraction Models
# ──────────────────────────────────────────────────────────────────────────────

class MissingFieldFlags(BaseModel):
    """Tracks which required clinical fields are absent from the patient record."""
    age: bool = False
    gender: bool = False
    chief_complaint: bool = False
    vital_signs: bool = False
    current_medications: bool = False
    allergy_status: bool = False
    lab_results: bool = False
    past_medical_history: bool = False
    social_history: bool = False
    family_history: bool = False
    missing_count: int = 0
    critical_missing: bool = False  # True when >= 3 required fields missing

    def missing_field_names(self) -> List[str]:
        """Return list of field names that are missing."""
        fields = []
        for fname in ["age", "gender", "chief_complaint", "vital_signs",
                      "current_medications", "allergy_status", "lab_results",
                      "past_medical_history", "social_history", "family_history"]:
            if getattr(self, fname):
                fields.append(fname)
        return fields


class InconsistencyFlag(BaseModel):
    """A detected inconsistency or contradiction in patient data."""
    field: str = Field(..., description="Clinical field where inconsistency was found")
    description: str = Field(..., description="Description of the inconsistency")
    values_found: List[str] = Field(default_factory=list, description="Conflicting values extracted")
    severity: str = Field("warning", description="warning | critical")


class NormalizedPatientContext(BaseModel):
    """
    Output of the Clinical Context Extraction Agent.

    Contains:
    - Structured PatientProfile
    - Missing field flags
    - Detected inconsistencies
    - Extraction metadata
    """
    patient_profile: PatientProfile
    missing_fields: MissingFieldFlags = Field(default_factory=MissingFieldFlags)
    inconsistencies: List[InconsistencyFlag] = Field(default_factory=list)
    extraction_method: str = Field("unknown", description="llm+rule_based | rule_based | none")
    raw_input_length: int = 0
    has_critical_missing: bool = False
    has_inconsistencies: bool = False

    def model_post_init(self, __context) -> None:
        self.has_critical_missing = self.missing_fields.critical_missing
        self.has_inconsistencies = len(self.inconsistencies) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Evidence Retrieval Models
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceSource(str, Enum):
    PUBMED = "PubMed"
    GUIDELINE = "ClinicalGuideline"
    OPENFDA = "OpenFDA"
    RXNORM = "RxNorm"
    SYNTHETIC = "Synthetic"


class EvidencePassage(BaseModel):
    """A single retrieved evidence passage with full provenance."""
    passage_id: str = Field(..., description="Unique passage identifier")
    text: str = Field(..., description="Passage text")
    source: EvidenceSource = EvidenceSource.GUIDELINE
    title: str = Field("", description="Article or guideline title")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    url: Optional[str] = Field(None, description="Direct link to source")
    publication_date: Optional[str] = Field(None, description="Publication date (YYYY-MM-DD or YYYY)")
    authors: Optional[str] = Field(None, description="Author list")
    journal: Optional[str] = Field(None, description="Journal name")
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    reranker_score: Optional[float] = None
    final_rank: Optional[int] = None


class EvidenceBundle(BaseModel):
    """Output of the Evidence Retrieval Agent — deduplicated, ranked passages."""
    query: str
    passages: List[EvidencePassage] = Field(default_factory=list)
    total_retrieved: int = 0
    retrieval_strategy: str = Field("hybrid", description="bm25 | dense | hybrid")
    reranked: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Evidence Grading Models
# ──────────────────────────────────────────────────────────────────────────────

class PublicationType(str, Enum):
    RCT = "randomized_controlled_trial"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL = "case_control"
    CASE_REPORT = "case_report"
    CLINICAL_GUIDELINE = "clinical_guideline"
    EXPERT_OPINION = "expert_opinion"
    REVIEW_ARTICLE = "review_article"
    UNKNOWN = "unknown"


class EvidenceGrade(BaseModel):
    """Grading assessment for a single evidence passage."""
    passage_id: str
    publication_type: PublicationType = PublicationType.UNKNOWN
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    source_authority: str = Field("", description="Authority/trustworthiness of source")
    publication_year: Optional[int] = None
    recency_weight: float = Field(1.0, description="Discount factor for older publications")
    study_limitations: List[str] = Field(default_factory=list)
    grade_notes: str = Field("", description="Additional grading notes")
    # Separation of relevance from clinical evidence quality
    clinically_relevant: Optional[bool] = None
    evidence_quality_note: str = Field(
        "", description="Note distinguishing relevance from evidence quality"
    )


class GradedEvidenceBundle(BaseModel):
    """Evidence bundle with grading annotations."""
    query: str
    graded_passages: List[EvidenceGrade] = Field(default_factory=list)
    ungraded_passages: List[EvidencePassage] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Evidence Verification (Claim-Level) Models
# ──────────────────────────────────────────────────────────────────────────────

class ClaimSupportStatus(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNCITED = "uncited"
    NOT_VERIFIED = "not_verified"


class AtomicClaim(BaseModel):
    """A single atomic factual claim extracted from the generated report."""
    claim_id: str
    text: str = Field(..., description="The atomic claim text")
    source_section: str = Field("", description="Report section this claim came from")
    proposed_citation_ids: List[str] = Field(
        default_factory=list, description="Passage IDs the claim should be traceable to"
    )
    agent_step_origin: str = Field("", description="Which agent step generated this claim")


class ClaimVerification(BaseModel):
    """Verification result for a single atomic claim."""
    claim_id: str
    claim_text: str
    status: ClaimSupportStatus = ClaimSupportStatus.NOT_VERIFIED
    # Citation validation
    citation_ids_valid: List[str] = Field(default_factory=list, description="IDs that exist in evidence store")
    citation_ids_invalid: List[str] = Field(default_factory=list, description="IDs that could not be found")
    # Passage entailment
    supporting_passage_ids: List[str] = Field(default_factory=list)
    refuting_passage_ids: List[str] = Field(default_factory=list)
    # Entailment scores
    max_support_score: float = Field(0.0, ge=0.0, le=1.0)
    max_refute_score: float = Field(0.0, ge=0.0, le=1.0)
    entailment_method: str = Field("nli_deberta", description="nli_deberta | lexical | none")
    # Human-readable explanation
    verification_note: str = ""
    requires_human_review: bool = True


class VerificationSummary(BaseModel):
    """Aggregate verification results for the full report."""
    total_claims: int = 0
    supported_count: int = 0
    refuted_count: int = 0
    insufficient_evidence_count: int = 0
    uncited_count: int = 0
    claim_support_rate: float = 0.0      # supported / total
    citation_precision: float = 0.0      # valid_citations / total_citations
    entailment_rate: float = 0.0         # entailed / verified
    unsupported_claim_rate: float = 0.0  # (refuted + uncited) / total
    claim_verifications: List[ClaimVerification] = Field(default_factory=list)
    verification_timestamp: Optional[datetime] = None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 7: Uncertainty & Completeness Models
# ──────────────────────────────────────────────────────────────────────────────

class UncertaintyType(str, Enum):
    PATIENT_INFO_GAP = "patient_info_gap"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    EVIDENCE_CONFLICTING = "evidence_conflicting"
    OUT_OF_SCOPE = "out_of_scope"
    CLINICAL_COMPLEXITY = "clinical_complexity"


class UncertaintyItem(BaseModel):
    """A single identified source of uncertainty."""
    uncertainty_type: UncertaintyType
    description: str
    affected_fields: List[str] = Field(default_factory=list)
    severity: str = Field("moderate", description="low | moderate | high | critical")
    mitigation: Optional[str] = Field(None, description="Suggested action to resolve uncertainty")


class AbstentionDecision(BaseModel):
    """Decision on whether the system should abstain from making a recommendation."""
    should_abstain: bool = False
    abstention_reason: Optional[str] = None
    partial_response_allowed: bool = True
    minimum_evidence_threshold_met: bool = True
    safety_concern: bool = False


class EvidenceSufficiency(BaseModel):
    """Assessment of whether retrieved evidence is sufficient for clinical recommendations."""
    is_sufficient: bool = True
    sufficiency_score: float = Field(1.0, ge=0.0, le=1.0)
    evidence_count: int = 0
    high_quality_evidence_count: int = 0
    coverage_gaps: List[str] = Field(default_factory=list, description="Topics not covered by evidence")
    threshold_used: float = Field(0.3, description="Minimum sufficiency score threshold")


class UncertaintyReport(BaseModel):
    """Complete uncertainty and completeness assessment."""
    uncertainty_items: List[UncertaintyItem] = Field(default_factory=list)
    abstention_decision: AbstentionDecision = Field(default_factory=AbstentionDecision)
    evidence_sufficiency: EvidenceSufficiency = Field(default_factory=EvidenceSufficiency)
    missing_patient_info: List[str] = Field(default_factory=list)
    overall_uncertainty_level: str = Field("low", description="low | moderate | high | critical")
    limitations_summary: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Phase 8: Enhanced CDS Report
# ──────────────────────────────────────────────────────────────────────────────

class HumanReviewStatus(str, Enum):
    PENDING = "pending_review"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REQUIRES_CORRECTION = "requires_correction"


class EnhancedCDSReport(CDSReport):
    """
    Extended Clinical Decision Support Report with:
    - Claim-level verification
    - Evidence grading
    - Uncertainty assessment
    - Human review tracking
    - Safety warnings

    Extends the original CDSReport for backward compatibility.
    """
    # Verification
    verification_summary: Optional[VerificationSummary] = None
    verified_claims: List[ClaimVerification] = Field(default_factory=list)

    # Evidence
    evidence_bundle: Optional[EvidenceBundle] = None
    evidence_grades: List[EvidenceGrade] = Field(default_factory=list)

    # Uncertainty
    uncertainty_report: Optional[UncertaintyReport] = None
    missing_patient_information: List[str] = Field(default_factory=list)

    # Safety
    safety_warnings: List[str] = Field(
        default_factory=list,
        description="Critical safety alerts requiring immediate clinician attention"
    )
    unsafe_recommendation_flags: List[str] = Field(default_factory=list)

    # Human review
    human_review_status: HumanReviewStatus = HumanReviewStatus.PENDING
    reviewer_notes: Optional[str] = None

    # Traceability
    pipeline_version: str = "2.0.0-evidence-verified"
    experiment_id: Optional[str] = None

    # Disclaimer (always present)
    disclaimer: str = (
        "This report is generated by an AI system for research and clinical decision support "
        "purposes only. It does not constitute medical advice, diagnosis, or treatment. "
        "All outputs must be reviewed by a qualified healthcare professional before any "
        "clinical action is taken. AI-generated statements that are unsupported by cited "
        "evidence are explicitly marked as requiring review."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Extended Agent State (LangGraph pipeline)
# ──────────────────────────────────────────────────────────────────────────────

class EnhancedAgentState(AgentState):
    """
    Extended state for the LangGraph-based enhanced pipeline (Experiment 4).
    Preserves all original AgentState fields for backward compatibility.
    """
    # Phase 3
    normalized_context: Optional[NormalizedPatientContext] = None

    # Phase 4
    evidence_bundle: Optional[EvidenceBundle] = None

    # Phase 5
    graded_evidence: Optional[GradedEvidenceBundle] = None

    # Phase 6
    verification_summary: Optional[VerificationSummary] = None

    # Phase 7
    uncertainty_report: Optional[UncertaintyReport] = None

    # Phase 8
    enhanced_report: Optional[EnhancedCDSReport] = None

    # Metadata
    pipeline_type: str = "langgraph_enhanced"
    experiment_id: Optional[str] = None
