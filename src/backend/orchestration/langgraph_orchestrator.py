# [Track B: Evidence-Verified CDSS — Phase 8]
"""
LangGraph-Based Enhanced Clinical Decision Support Orchestrator.

Architecture Overview:
----------------------
This module implements Phase 8 of the Evidence-Verified CDSS:
an enhanced 10-step orchestrator that layers on top of the original 6-step
baseline orchestrator (Experiment 1). It introduces claim-level verification,
hybrid retrieval (BM25 + ChromaDB + PubMed), evidence quality grading (GRADE),
context normalization, and uncertainty quantification.

10-Step Pipeline Workflow:
--------------------------
  1. parse: PatientParserTool (original) — Extracts structured patient profile from raw text.
  2. context_extraction: ClinicalContextAgent (new, parallel with parse) —
     Normalizes clinical entities, flags missing required fields, and identifies inconsistencies.
  3. reason: ClinicalReasoningTool (original) — Produces ranked differential diagnoses and initial workup.
  4. drugs: DrugInteractionTool (original, parallel with retrieval) — Checks drug-drug interactions.
  5. evidence_retrieval: HybridRetriever + PubMedClient (new) —
     Executes hybrid BM25 + dense vector + PubMed retrieval with CrossEncoder reranking.
  6. evidence_grading: EvidenceGrader (new) — Assesses study design, recency, and source authority.
  7. conflicts: ConflictDetectionTool (original) — Identifies contradictions between evidence & care plan.
  8. synthesize: SynthesisTool (original) — Arbitrates and synthesizes all evidence into a CDS report.
  9. verify: EvidenceVerificationAgent (new, runs post-synthesis) —
     Extracts atomic claims and verifies them against cited passages using NLI entailment.
 10. uncertainty: UncertaintyAgent (new, runs post-verification) —
     Quantifies information gaps, evidence sufficiency, and determines abstention / safety warnings.

Dual Execution Engine:
----------------------
  - Primary Engine: When `langgraph` is available and enabled, the workflow is compiled into
    a `StateGraph` with conditional edges to enforce dependency gates and abstention branches.
  - Fallback Engine: When `langgraph` is not installed (ImportError) or toggled off,
    execution falls back seamlessly to an asynchronous sequential execution flow with
    asyncio.gather parallel branches.

Factory Pattern:
----------------
  - OrchestratorFactory provides unified instantiation allowing runtime switching between
    the baseline Orchestrator (Experiment 1) and the EnhancedOrchestrator (Experiments 2-4).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from app.agent.orchestrator import Orchestrator, StepCallback
from app.config import settings
from app.models.schemas import (
    AbstentionDecision,
    AgentState,
    AgentStep,
    AgentStepStatus,
    CaseSubmission,
    CDSReport,
    ClaimSupportStatus,
    ClaimVerification,
    ClinicalReasoningResult,
    ConflictDetectionResult,
    DrugInteractionResult,
    EnhancedAgentState,
    EnhancedCDSReport,
    EvidenceBundle,
    EvidenceGrade,
    EvidencePassage,
    EvidenceSufficiency,
    GradedEvidenceBundle,
    GuidelineExcerpt,
    GuidelineRetrievalResult,
    HumanReviewStatus,
    NormalizedPatientContext,
    PatientProfile,
    Severity,
    UncertaintyItem,
    UncertaintyReport,
    UncertaintyType,
    VerificationSummary,
)
from app.tools.clinical_reasoning import ClinicalReasoningTool
from app.tools.conflict_detection import ConflictDetectionTool
from app.tools.drug_interactions import DrugInteractionTool
from app.tools.guideline_retrieval import GuidelineRetrievalTool
from app.tools.patient_parser import PatientParserTool
from app.tools.synthesis import SynthesisTool

from agents.clinical_context_agent import ClinicalContextAgent
from agents.evidence_grading_agent import EvidenceGrader
from agents.evidence_verification_agent import EvidenceVerificationAgent
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.pubmed_client import PubMedClient

# Optional import for UncertaintyAgent (Phase 7 agent) with fallback
try:
    from agents.uncertainty_agent import UncertaintyAgent  # type: ignore
except (ImportError, ModuleNotFoundError):
    UncertaintyAgent = None

# Optional import for LangGraph
try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Built-in Fallback Uncertainty Agent
# ──────────────────────────────────────────────────────────────────────────────

class FallbackUncertaintyAgent:
    """
    Deterministic rule-based uncertainty and safety assessment agent.

    Used when `agents.uncertainty_agent` is not installed or available. Evaluates:
      1. Missing clinical information and critical flags from NormalizedPatientContext
      2. Inconsistencies and contradictory patient values
      3. Evidence sufficiency score and coverage gaps from EvidenceBundle
      4. Guideline conflicts and safety contradictions
      5. Refuted and uncited claims from VerificationSummary
      6. System-level abstention recommendations for clinician safety
    """

    async def assess(
        self,
        normalized_context: Optional[NormalizedPatientContext] = None,
        patient_profile: Optional[PatientProfile] = None,
        clinical_reasoning: Optional[ClinicalReasoningResult] = None,
        evidence_bundle: Optional[EvidenceBundle] = None,
        graded_evidence: Optional[GradedEvidenceBundle] = None,
        conflicts: Optional[ConflictDetectionResult] = None,
        verification_summary: Optional[VerificationSummary] = None,
        final_report: Optional[CDSReport] = None,
        **kwargs: Any,
    ) -> UncertaintyReport:
        """Evaluate pipeline state and produce an UncertaintyReport."""
        items: List[UncertaintyItem] = []
        missing_info: List[str] = []

        # 1. Patient Information Gaps
        if normalized_context and normalized_context.missing_fields:
            missing_info = normalized_context.missing_fields.missing_field_names()
            if normalized_context.missing_fields.critical_missing:
                items.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.PATIENT_INFO_GAP,
                        description=(
                            f"Critical patient fields absent ({len(missing_info)} fields): "
                            f"{', '.join(missing_info[:5])}"
                        ),
                        affected_fields=missing_info,
                        severity="high",
                        mitigation="Gather missing clinical parameters before finalizing therapy decisions.",
                    )
                )
            elif missing_info:
                items.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.PATIENT_INFO_GAP,
                        description=f"Non-critical fields omitted: {', '.join(missing_info)}",
                        affected_fields=missing_info,
                        severity="low",
                        mitigation="Verify if absent fields impact long-term management.",
                    )
                )

            # Inconsistencies
            if normalized_context.inconsistencies:
                for inc in normalized_context.inconsistencies:
                    items.append(
                        UncertaintyItem(
                            uncertainty_type=UncertaintyType.PATIENT_INFO_GAP,
                            description=f"Inconsistent {inc.field}: {inc.description}",
                            affected_fields=[inc.field],
                            severity="high" if inc.severity == "critical" else "moderate",
                            mitigation="Confirm vital signs and intake notes with attending staff.",
                        )
                    )

        # 2. Evidence Sufficiency
        passage_count = len(evidence_bundle.passages) if evidence_bundle else 0
        high_quality_count = 0
        if graded_evidence and graded_evidence.graded_passages:
            high_quality_count = sum(
                1
                for g in graded_evidence.graded_passages
                if getattr(g, "source_authority", "") == "high" or (g.relevance_score and g.relevance_score >= 0.8)
            )

        sufficiency_score = min(1.0, passage_count / 5.0) if passage_count > 0 else 0.0
        is_sufficient = sufficiency_score >= 0.3
        coverage_gaps: List[str] = []

        if passage_count == 0:
            coverage_gaps.append("Zero guideline or literature passages retrieved for top differential.")
            items.append(
                UncertaintyItem(
                    uncertainty_type=UncertaintyType.EVIDENCE_INSUFFICIENT,
                    description="No relevant clinical evidence retrieved from guidelines or PubMed.",
                    severity="high",
                    mitigation="Broaden search terms or consult subspecialty practice guidelines.",
                )
            )
        elif passage_count < 3:
            coverage_gaps.append(f"Limited literature support ({passage_count} passages).")
            items.append(
                UncertaintyItem(
                    uncertainty_type=UncertaintyType.EVIDENCE_INSUFFICIENT,
                    description=f"Sparse evidence coverage: only {passage_count} passages retrieved.",
                    severity="moderate",
                    mitigation="Review additional institutional clinical pathways.",
                )
            )

        evidence_sufficiency = EvidenceSufficiency(
            is_sufficient=is_sufficient,
            sufficiency_score=round(sufficiency_score, 3),
            evidence_count=passage_count,
            high_quality_evidence_count=high_quality_count,
            coverage_gaps=coverage_gaps,
            threshold_used=0.3,
        )

        # 3. Clinical Conflicts & Contradictions
        safety_concern = False
        if conflicts and conflicts.conflicts:
            crit_conflicts = [
                c for c in conflicts.conflicts
                if c.severity == Severity.CRITICAL or getattr(c.severity, "value", "") == "critical"
            ]
            if crit_conflicts:
                safety_concern = True
                items.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.EVIDENCE_CONFLICTING,
                        description=(
                            f"{len(crit_conflicts)} critical conflict(s) detected between patient state "
                            f"and clinical practice guidelines."
                        ),
                        severity="critical",
                        mitigation="Immediate clinician reconciliation of conflicting orders is advised.",
                    )
                )

            mod_conflicts = [c for c in conflicts.conflicts if c not in crit_conflicts]
            if mod_conflicts:
                items.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.EVIDENCE_CONFLICTING,
                        description=f"{len(mod_conflicts)} clinical care gap(s) or omission(s) noted.",
                        severity="moderate",
                        mitigation="Evaluate suggested resolutions during clinical rounds.",
                    )
                )

        # 4. Claim Entailment Verification
        if verification_summary:
            if verification_summary.refuted_count > 0:
                safety_concern = True
                items.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.EVIDENCE_CONFLICTING,
                        description=(
                            f"{verification_summary.refuted_count} claim(s) in the generated report are "
                            f"refuted or contradicted by cited literature."
                        ),
                        severity="critical",
                        mitigation="Review flagged claims and remove contradicted statements.",
                    )
                )
            if verification_summary.uncited_count > 0:
                items.append(
                    UncertaintyItem(
                        uncertainty_type=UncertaintyType.EVIDENCE_INSUFFICIENT,
                        description=(
                            f"{verification_summary.uncited_count} generated statement(s) have no "
                            f"verifiable citation."
                        ),
                        severity="moderate",
                        mitigation="Corroborate uncited recommendations against primary sources.",
                    )
                )

        # 5. Abstention Evaluation
        should_abstain = False
        abstention_reason: Optional[str] = None

        if safety_concern and not is_sufficient:
            should_abstain = True
            abstention_reason = (
                "Critical safety conflicts present alongside insufficient evidentiary support."
            )
        elif passage_count == 0 and (normalized_context and normalized_context.has_critical_missing):
            should_abstain = True
            abstention_reason = (
                "Severe patient data incompleteness combined with zero evidence retrieval."
            )

        abstention = AbstentionDecision(
            should_abstain=should_abstain,
            abstention_reason=abstention_reason,
            partial_response_allowed=True,
            minimum_evidence_threshold_met=is_sufficient,
            safety_concern=safety_concern,
        )

        # 6. Overall Uncertainty Level
        if safety_concern or should_abstain:
            overall_level = "critical"
        elif any(i.severity in ("critical", "high") for i in items):
            overall_level = "high"
        elif any(i.severity == "moderate" for i in items):
            overall_level = "moderate"
        else:
            overall_level = "low"

        limitations = (
            f"Overall clinical uncertainty is rated {overall_level.upper()}. "
            f"Identified {len(items)} uncertainty factor(s). "
            f"Evidence sufficiency score: {evidence_sufficiency.sufficiency_score:.2f}."
        )

        return UncertaintyReport(
            uncertainty_items=items,
            abstention_decision=abstention,
            evidence_sufficiency=evidence_sufficiency,
            missing_patient_info=missing_info,
            overall_uncertainty_level=overall_level,
            limitations_summary=limitations,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Enhanced Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class EnhancedOrchestrator(Orchestrator):
    """
    Enhanced 10-step Clinical Decision Support Orchestrator.

    Wraps the baseline Orchestrator and extends it with:
      - Clinical context extraction & validation (ClinicalContextAgent)
      - Hybrid PubMed + Vector + BM25 retrieval (HybridRetriever + PubMedClient)
      - GRADE-based evidence quality grading (EvidenceGrader)
      - Post-synthesis claim-level verification (EvidenceVerificationAgent)
      - Uncertainty, sufficiency, and abstention assessment (UncertaintyAgent)
      - Optional LangGraph StateGraph execution with graceful sequential fallback

    Usage:
        orchestrator = EnhancedOrchestrator()
        async for step_update in orchestrator.run(case):
            # stream step_update to frontend
            ...
        report = orchestrator.get_result()
    """

    def __init__(self, use_langgraph: Optional[bool] = None) -> None:
        super().__init__()

        # Determine whether to use LangGraph engine
        if use_langgraph is None:
            configured_flag = getattr(settings, "use_langgraph", False)
            self.use_langgraph = configured_flag and LANGGRAPH_AVAILABLE
        else:
            self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE

        if use_langgraph and not LANGGRAPH_AVAILABLE:
            logger.warning(
                "LangGraph was requested but is not installed. Falling back to sequential execution. "
                "Install langgraph with: pip install langgraph"
            )

        # Enhanced Pipeline Components
        self.clinical_context = ClinicalContextAgent()
        self.hybrid_retriever = HybridRetriever()
        self.pubmed_client = PubMedClient()
        self.evidence_grader = EvidenceGrader()
        self.evidence_verifier = EvidenceVerificationAgent()

        # Uncertainty agent setup with graceful fallback
        if UncertaintyAgent is not None:
            try:
                self.uncertainty_agent = UncertaintyAgent()
            except Exception as e:
                logger.warning(
                    "Failed to initialize UncertaintyAgent (%s). Using FallbackUncertaintyAgent.", e
                )
                self.uncertainty_agent = FallbackUncertaintyAgent()
        else:
            self.uncertainty_agent = FallbackUncertaintyAgent()

        # State and execution internals
        self._state: Optional[EnhancedAgentState] = None
        self._current_case: Optional[CaseSubmission] = None
        self._step_queue: Optional[asyncio.Queue[Optional[AgentStep]]] = None

        # Compile LangGraph workflow if enabled
        self._compiled_graph = None
        if self.use_langgraph:
            self._compiled_graph = self._build_langgraph_workflow()

    @property
    def state(self) -> Optional[EnhancedAgentState]:
        """Return the current EnhancedAgentState."""
        return self._state

    def get_result(self) -> Optional[EnhancedCDSReport]:
        """
        Return the final EnhancedCDSReport, if synthesis and enhancement completed.
        Falls back to final_report wrapped as an EnhancedCDSReport if needed.
        """
        if not self._state:
            return None
        if self._state.enhanced_report:
            return self._state.enhanced_report
        if self._state.final_report:
            return self._build_enhanced_report()
        return None

    def _create_steps(self, case: CaseSubmission) -> List[AgentStep]:
        """Define the 10-step pipeline based on the case submission configuration."""
        steps: List[AgentStep] = [
            AgentStep(
                step_id="parse",
                step_name="Parsing Patient Data",
                tool_name="patient_parser",
            ),
            AgentStep(
                step_id="context_extraction",
                step_name="Clinical Context Extraction",
                tool_name="clinical_context_agent",
            ),
            AgentStep(
                step_id="reason",
                step_name="Clinical Reasoning",
                tool_name="clinical_reasoning",
            ),
        ]

        if case.include_drug_check:
            steps.append(
                AgentStep(
                    step_id="drugs",
                    step_name="Drug Interaction Check",
                    tool_name="drug_interactions",
                )
            )

        if case.include_guidelines:
            steps.append(
                AgentStep(
                    step_id="evidence_retrieval",
                    step_name="Hybrid Evidence Retrieval",
                    tool_name="hybrid_retriever",
                )
            )
            steps.append(
                AgentStep(
                    step_id="evidence_grading",
                    step_name="Evidence Quality Grading",
                    tool_name="evidence_grading_agent",
                )
            )
            steps.append(
                AgentStep(
                    step_id="conflicts",
                    step_name="Conflict Detection",
                    tool_name="conflict_detection",
                )
            )

        steps.extend([
            AgentStep(
                step_id="synthesize",
                step_name="Synthesizing Report",
                tool_name="synthesis",
            ),
            AgentStep(
                step_id="verify",
                step_name="Evidence Verification",
                tool_name="evidence_verification_agent",
            ),
            AgentStep(
                step_id="uncertainty",
                step_name="Uncertainty & Safety Assessment",
                tool_name="uncertainty_agent",
            ),
        ])

        return steps

    # ──────────────────────────────────────────────────────────────────────────
    # Execution Entry Point
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self, case: CaseSubmission) -> AsyncGenerator[AgentStep, None]:
        """
        Run the full 10-step agent pipeline, yielding AgentStep updates.

        Executes via LangGraph StateGraph if compiled and enabled; otherwise
        executes via the sequential async fallback pipeline.
        """
        case_id = str(uuid.uuid4())[:8]
        steps = self._create_steps(case)

        self._state = EnhancedAgentState(
            case_id=case_id,
            steps=steps,
            started_at=datetime.now(timezone.utc),
            pipeline_type="langgraph_enhanced" if (self.use_langgraph and self._compiled_graph) else "sequential_enhanced",
            experiment_id="experiment_4",
        )
        self._current_case = case

        if self.use_langgraph and self._compiled_graph:
            async for step_update in self._run_langgraph(case):
                yield step_update
        else:
            async for step_update in self._run_sequential(case):
                yield step_update

    # ──────────────────────────────────────────────────────────────────────────
    # Sequential Execution Engine (Fallback / Default)
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_sequential(self, case: CaseSubmission) -> AsyncGenerator[AgentStep, None]:
        """
        Execute the 10 steps sequentially with parallel branches for steps 1-2 and 4-5.
        """
        try:
            # ── Steps 1 & 2: Parse + Context Extraction (Parallel) ──
            yield self._mark_running("parse")
            yield self._mark_running("context_extraction")

            parse_task = asyncio.create_task(
                self._run_step("parse", self._step_parse, case.patient_text)
            )
            context_task = asyncio.create_task(
                self._run_step("context_extraction", self._step_context_extraction, case.patient_text)
            )

            done, _ = await asyncio.wait([parse_task, context_task])
            for task in done:
                res = task.result()
                if isinstance(res, AgentStep):
                    yield res

            # Prerequisite Check: If parse failed, try recovering with normalized_context profile
            if self._state.patient_profile is None and self._state.normalized_context:
                logger.info("Recovering patient_profile from normalized_context after parse step failure.")
                self._state.patient_profile = self._state.normalized_context.patient_profile

            if self._state.patient_profile is None:
                for skipped in self._skip_remaining_steps("context_extraction"):
                    yield skipped
                self._state.completed_at = datetime.now(timezone.utc)
                return

            # ── Step 3: Clinical Reasoning ──
            yield self._mark_running("reason")
            step = await self._run_step("reason", self._step_reason)
            yield step

            if step.status == AgentStepStatus.FAILED or not self._state.clinical_reasoning:
                for skipped in self._skip_remaining_steps("reason"):
                    yield skipped
                self._state.completed_at = datetime.now(timezone.utc)
                return

            # ── Steps 4 & 5: Drugs + Evidence Retrieval (Parallel) ──
            parallel_tasks: List[asyncio.Task[AgentStep]] = []

            if case.include_drug_check:
                yield self._mark_running("drugs")
                parallel_tasks.append(
                    asyncio.create_task(self._run_step("drugs", self._step_drugs))
                )

            if case.include_guidelines:
                yield self._mark_running("evidence_retrieval")
                parallel_tasks.append(
                    asyncio.create_task(self._run_step("evidence_retrieval", self._step_evidence_retrieval))
                )

            if parallel_tasks:
                done, _ = await asyncio.wait(parallel_tasks)
                for task in done:
                    res = task.result()
                    if isinstance(res, AgentStep):
                        yield res

            # ── Step 6: Evidence Quality Grading ──
            if case.include_guidelines:
                yield self._mark_running("evidence_grading")
                yield await self._run_step("evidence_grading", self._step_evidence_grading)

            # ── Step 7: Conflict Detection ──
            if case.include_guidelines:
                yield self._mark_running("conflicts")
                yield await self._run_step("conflicts", self._step_conflicts)

            # ── Step 8: Synthesis ──
            yield self._mark_running("synthesize")
            yield await self._run_step("synthesize", self._step_synthesize)

            # ── Step 9: Verification ──
            yield self._mark_running("verify")
            yield await self._run_step("verify", self._step_verify)

            # ── Step 10: Uncertainty Assessment ──
            yield self._mark_running("uncertainty")
            yield await self._run_step("uncertainty", self._step_uncertainty)

            # Complete pipeline
            self._state.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            logger.error("Sequential pipeline aborted: %s", e, exc_info=True)
            for step in self._state.steps:
                if step.status in (AgentStepStatus.PENDING, AgentStepStatus.RUNNING):
                    step.status = AgentStepStatus.FAILED
                    step.error = f"Pipeline aborted: {str(e)}"
            self._state.completed_at = datetime.now(timezone.utc)
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # LangGraph StateGraph Engine
    # ──────────────────────────────────────────────────────────────────────────

    def _build_langgraph_workflow(self) -> Optional[Any]:
        """
        Build and compile the 10-step LangGraph StateGraph.

        Enforces prerequisite gates via conditional edges. Emits step events
        via an internal asyncio.Queue during execution.
        """
        if not LANGGRAPH_AVAILABLE:
            return None

        try:
            from typing import TypedDict

            class GraphState(TypedDict, total=False):
                completed: bool
                error: Optional[str]

            workflow = StateGraph(GraphState)

            # Node 1: Parse
            async def node_parse(s: GraphState) -> GraphState:
                yield_step = self._mark_running("parse")
                if self._step_queue:
                    await self._step_queue.put(yield_step)
                step = await self._run_step("parse", self._step_parse, self._current_case.patient_text)
                if self._step_queue:
                    await self._step_queue.put(step)
                return s

            # Node 2: Context Extraction
            async def node_context(s: GraphState) -> GraphState:
                yield_step = self._mark_running("context_extraction")
                if self._step_queue:
                    await self._step_queue.put(yield_step)
                step = await self._run_step("context_extraction", self._step_context_extraction, self._current_case.patient_text)
                if self._step_queue:
                    await self._step_queue.put(step)
                return s

            # Node 3: Clinical Reasoning
            async def node_reason(s: GraphState) -> GraphState:
                yield_step = self._mark_running("reason")
                if self._step_queue:
                    await self._step_queue.put(yield_step)
                step = await self._run_step("reason", self._step_reason)
                if self._step_queue:
                    await self._step_queue.put(step)
                return s

            # Node 4: Drug Interactions
            async def node_drugs(s: GraphState) -> GraphState:
                if self._current_case.include_drug_check:
                    yield_step = self._mark_running("drugs")
                    if self._step_queue:
                        await self._step_queue.put(yield_step)
                    step = await self._run_step("drugs", self._step_drugs)
                    if self._step_queue:
                        await self._step_queue.put(step)
                return s

            # Node 5: Evidence Retrieval
            async def node_evidence(s: GraphState) -> GraphState:
                if self._current_case.include_guidelines:
                    yield_step = self._mark_running("evidence_retrieval")
                    if self._step_queue:
                        await self._step_queue.put(yield_step)
                    step = await self._run_step("evidence_retrieval", self._step_evidence_retrieval)
                    if self._step_queue:
                        await self._step_queue.put(step)
                return s

            # Node 6: Evidence Grading
            async def node_grading(s: GraphState) -> GraphState:
                if self._current_case.include_guidelines:
                    yield_step = self._mark_running("evidence_grading")
                    if self._step_queue:
                        await self._step_queue.put(yield_step)
                    step = await self._run_step("evidence_grading", self._step_evidence_grading)
                    if self._step_queue:
                        await self._step_queue.put(step)
                return s

            # Node 7: Conflict Detection
            async def node_conflicts(s: GraphState) -> GraphState:
                if self._current_case.include_guidelines:
                    yield_step = self._mark_running("conflicts")
                    if self._step_queue:
                        await self._step_queue.put(yield_step)
                    step = await self._run_step("conflicts", self._step_conflicts)
                    if self._step_queue:
                        await self._step_queue.put(step)
                return s

            # Node 8: Synthesis
            async def node_synthesize(s: GraphState) -> GraphState:
                yield_step = self._mark_running("synthesize")
                if self._step_queue:
                    await self._step_queue.put(yield_step)
                step = await self._run_step("synthesize", self._step_synthesize)
                if self._step_queue:
                    await self._step_queue.put(step)
                return s

            # Node 9: Verification
            async def node_verify(s: GraphState) -> GraphState:
                yield_step = self._mark_running("verify")
                if self._step_queue:
                    await self._step_queue.put(yield_step)
                step = await self._run_step("verify", self._step_verify)
                if self._step_queue:
                    await self._step_queue.put(step)
                return s

            # Node 10: Uncertainty Assessment
            async def node_uncertainty(s: GraphState) -> GraphState:
                yield_step = self._mark_running("uncertainty")
                if self._step_queue:
                    await self._step_queue.put(yield_step)
                step = await self._run_step("uncertainty", self._step_uncertainty)
                if self._step_queue:
                    await self._step_queue.put(step)
                s["completed"] = True
                return s

            # Register Nodes
            workflow.add_node("parse", node_parse)
            workflow.add_node("context_extraction", node_context)
            workflow.add_node("reason", node_reason)
            workflow.add_node("drugs", node_drugs)
            workflow.add_node("evidence_retrieval", node_evidence)
            workflow.add_node("evidence_grading", node_grading)
            workflow.add_node("conflicts", node_conflicts)
            workflow.add_node("synthesize", node_synthesize)
            workflow.add_node("verify", node_verify)
            workflow.add_node("uncertainty", node_uncertainty)

            # Register Edges
            workflow.set_entry_point("parse")
            workflow.add_edge("parse", "context_extraction")

            # Conditional Edge 1: Prerequisite gate after parsing/context
            def gate_after_context(state: GraphState) -> str:
                if self._state.patient_profile is None and self._state.normalized_context:
                    self._state.patient_profile = self._state.normalized_context.patient_profile
                if self._state.patient_profile is None:
                    return END
                return "reason"

            workflow.add_conditional_edges(
                "context_extraction", gate_after_context, {END: END, "reason": "reason"}
            )

            # Conditional Edge 2: Prerequisite gate after clinical reasoning
            def gate_after_reason(state: GraphState) -> str:
                if not self._state.clinical_reasoning:
                    return END
                return "drugs"

            workflow.add_conditional_edges(
                "reason", gate_after_reason, {END: END, "drugs": "drugs"}
            )

            workflow.add_edge("drugs", "evidence_retrieval")
            workflow.add_edge("evidence_retrieval", "evidence_grading")
            workflow.add_edge("evidence_grading", "conflicts")
            workflow.add_edge("conflicts", "synthesize")
            workflow.add_edge("synthesize", "verify")
            workflow.add_edge("verify", "uncertainty")
            workflow.add_edge("uncertainty", END)

            return workflow.compile()

        except Exception as e:
            logger.warning("Failed to compile LangGraph workflow: %s — falling back to sequential", e)
            return None

    async def _run_langgraph(self, case: CaseSubmission) -> AsyncGenerator[AgentStep, None]:
        """Run the compiled LangGraph workflow, streaming step updates via a queue."""
        self._step_queue = asyncio.Queue()

        async def _graph_runner() -> None:
            try:
                initial_state = {"completed": False, "error": None}
                # ainvoke will run the full graph through compiled edges
                await self._compiled_graph.ainvoke(initial_state)
            except Exception as e:
                logger.error("LangGraph execution error: %s", e, exc_info=True)
                for step in self._state.steps:
                    if step.status == AgentStepStatus.PENDING:
                        step.status = AgentStepStatus.FAILED
                        step.error = f"Workflow aborted: {str(e)}"
            finally:
                self._state.completed_at = datetime.now(timezone.utc)
                if self._step_queue:
                    await self._step_queue.put(None)  # Sentinel for completion

        runner_task = asyncio.create_task(_graph_runner())

        while True:
            step = await self._step_queue.get()
            if step is None:
                break
            yield step

        await runner_task

    # ──────────────────────────────────────────────────────────────────────────
    # Step Implementations
    # ──────────────────────────────────────────────────────────────────────────

    async def _step_parse(self, patient_text: str) -> None:
        """Step 1: Parse patient text into structured PatientProfile."""
        profile = await self.patient_parser.run(patient_text)
        self._state.patient_profile = profile

        step = self._get_step("parse")
        step.input_summary = (
            patient_text[:100] + "..." if len(patient_text) > 100 else patient_text
        )
        step.output_summary = (
            f"Parsed: {profile.chief_complaint}, {len(profile.current_medications)} meds, "
            f"{len(profile.lab_results)} labs"
        )

    async def _step_context_extraction(self, patient_text: str) -> None:
        """Step 2: Normalize context, detect missing clinical fields & inconsistencies."""
        # ClinicalContextAgent exposes .run() or .extract()
        if hasattr(self.clinical_context, "extract"):
            context = await self.clinical_context.extract(patient_text)
        elif hasattr(self.clinical_context, "run"):
            context = await self.clinical_context.run(patient_text)
        else:
            raise RuntimeError("ClinicalContextAgent has neither .extract() nor .run() method")

        self._state.normalized_context = context

        step = self._get_step("context_extraction")
        step.output_summary = (
            f"Extracted context: {context.missing_fields.missing_count} missing fields "
            f"(critical={context.has_critical_missing}), {len(context.inconsistencies)} inconsistencies"
        )

    async def _step_reason(self) -> None:
        """Step 3: Clinical reasoning / differential diagnosis."""
        if not self._state.patient_profile:
            raise RuntimeError("Patient profile not available — parse step must run first")

        result = await self.clinical_reasoning.run(self._state.patient_profile)
        self._state.clinical_reasoning = result

        step = self._get_step("reason")
        step.output_summary = (
            f"{len(result.differential_diagnosis)} diagnoses, "
            f"{len(result.recommended_workup)} recommendations"
        )

    async def _step_drugs(self) -> None:
        """Step 4: Drug-drug interaction check for current and proposed medications."""
        if not self._state.patient_profile:
            raise RuntimeError("Patient profile not available for drug interaction check")

        meds = self._state.patient_profile.current_medications
        proposed_meds: List[str] = []
        if self._state.clinical_reasoning:
            for action in self._state.clinical_reasoning.recommended_workup:
                low = action.action.lower()
                if "medication" in low or "prescribe" in low or "administer" in low:
                    proposed_meds.append(action.action)

        result = await self.drug_interaction.run(meds, proposed_meds)
        self._state.drug_interactions = result

        step = self._get_step("drugs")
        step.output_summary = f"{len(result.interactions_found)} interactions found"

    async def _step_evidence_retrieval(self) -> None:
        """Step 5: Hybrid evidence retrieval (BM25 + ChromaDB + PubMed)."""
        # Determine query from top differential or chief complaint
        if self._state.clinical_reasoning and self._state.clinical_reasoning.differential_diagnosis:
            top_dx = self._state.clinical_reasoning.differential_diagnosis[0].diagnosis
            query = f"{top_dx} clinical guidelines management"
        elif self._state.patient_profile and self._state.patient_profile.chief_complaint:
            query = f"{self._state.patient_profile.chief_complaint} clinical guidelines"
        else:
            query = "clinical guidelines management"

        # Search PubMed with graceful fallback on network / quota errors
        pubmed_passages: List[EvidencePassage] = []
        try:
            pubmed_passages = await self.pubmed_client.search(
                query=query, max_results=settings.retrieval_top_k
            )
        except Exception as e:
            logger.warning("PubMed search failed: %s — continuing with local retrieval", e)

        # Retrieve via HybridRetriever
        bundle = await self.hybrid_retriever.retrieve(
            query=query,
            top_k=settings.retrieval_top_k,
            include_pubmed=bool(pubmed_passages),
            pubmed_passages=pubmed_passages,
        )

        self._state.evidence_bundle = bundle
        # Convert to GuidelineRetrievalResult for backward compatibility with conflict/synthesis tools
        self._state.guideline_retrieval = self._bundle_to_guidelines(bundle)

        step = self._get_step("evidence_retrieval")
        step.output_summary = (
            f"{len(bundle.passages)} passages retrieved "
            f"({len(pubmed_passages)} from PubMed, strategy={bundle.retrieval_strategy})"
        )

    async def _step_evidence_grading(self) -> None:
        """Step 6: Grade evidence passages using GRADE criteria (study design, recency, authority)."""
        bundle = self._state.evidence_bundle or EvidenceBundle(query="general clinical guidelines")

        # Grade bundle (handle sync or async grade_bundle)
        grade_call = self.evidence_grader.grade_bundle(bundle)
        if asyncio.iscoroutine(grade_call):
            graded = await grade_call
        else:
            graded = grade_call

        self._state.graded_evidence = graded

        step = self._get_step("evidence_grading")
        step.output_summary = f"{len(graded.graded_passages)} passages graded with quality annotations"

    async def _step_conflicts(self) -> None:
        """Step 7: Detect conflicts between retrieved guidelines and patient data."""
        result = await self.conflict_detection.run(
            patient_profile=self._state.patient_profile,
            clinical_reasoning=self._state.clinical_reasoning,
            drug_interactions=self._state.drug_interactions,
            guideline_retrieval=self._state.guideline_retrieval,
        )
        self._state.conflict_detection = result

        step = self._get_step("conflicts")
        n = len(result.conflicts)
        if n == 0:
            step.output_summary = "No conflicts detected"
        else:
            step.output_summary = f"{n} conflict(s) detected — {result.summary}"

    async def _step_synthesize(self) -> None:
        """Step 8: Synthesize all tool outputs into a final CDS report."""
        report = await self.synthesis.run(
            patient_profile=self._state.patient_profile,
            clinical_reasoning=self._state.clinical_reasoning,
            drug_interactions=self._state.drug_interactions,
            guideline_retrieval=self._state.guideline_retrieval,
            conflict_detection=self._state.conflict_detection,
        )
        self._state.final_report = report

        step = self._get_step("synthesize")
        step.output_summary = "CDS report synthesized"

    async def _step_verify(self) -> None:
        """Step 9: Extract atomic claims and verify entailment against evidence passages."""
        if not self._state.final_report:
            raise RuntimeError("Final report not available for verification step")

        report_sections = self._build_report_sections(self._state.final_report)
        report_text = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in report_sections.items() if v)
        bundle = self._state.evidence_bundle or EvidenceBundle(query="clinical guidelines")

        summary = await self.evidence_verifier.verify(
            report_text=report_text,
            evidence_bundle=bundle,
            report_sections=report_sections,
        )
        self._state.verification_summary = summary

        step = self._get_step("verify")
        step.output_summary = (
            f"Verified {summary.total_claims} claims: {summary.supported_count} supported, "
            f"{summary.refuted_count} refuted (precision={summary.citation_precision:.2f})"
        )

    async def _step_uncertainty(self) -> None:
        """Step 10: Assess uncertainty, evidence sufficiency, and abstention requirements."""
        uncertainty_report = await self._run_uncertainty_assessment()
        self._state.uncertainty_report = uncertainty_report

        # Construct final enhanced report
        enhanced_report = self._build_enhanced_report()
        self._state.enhanced_report = enhanced_report

        step = self._get_step("uncertainty")
        step.output_summary = (
            f"Overall uncertainty: {uncertainty_report.overall_uncertainty_level.upper()} "
            f"({len(uncertainty_report.uncertainty_items)} factors, abstain={uncertainty_report.abstention_decision.should_abstain})"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_uncertainty_assessment(self) -> UncertaintyReport:
        """Execute UncertaintyAgent.assess() dynamically matching its parameters."""
        agent = self.uncertainty_agent
        available_args = {
            "normalized_context": self._state.normalized_context,
            "patient_context": self._state.normalized_context,
            "patient_profile": self._state.patient_profile,
            "clinical_reasoning": self._state.clinical_reasoning,
            "evidence_bundle": self._state.evidence_bundle,
            "bundle": self._state.evidence_bundle,
            "graded_evidence": self._state.graded_evidence,
            "conflicts": self._state.conflict_detection,
            "verification_summary": self._state.verification_summary,
            "final_report": self._state.final_report,
            "report": self._state.final_report,
        }

        try:
            if hasattr(agent, "assess"):
                sig = inspect.signature(agent.assess)
                call_kwargs = {}
                has_var_keyword = False
                for p in sig.parameters.values():
                    if p.kind == p.VAR_KEYWORD:
                        has_var_keyword = True
                    elif p.name in available_args:
                        call_kwargs[p.name] = available_args[p.name]

                if has_var_keyword:
                    call_kwargs.update(available_args)

                res = agent.assess(**call_kwargs)
                if asyncio.iscoroutine(res):
                    return await res
                return res
        except Exception as e:
            logger.warning("UncertaintyAgent execution failed: %s — falling back to FallbackUncertaintyAgent", e)

        # Fallback to local deterministic assessor
        fallback = FallbackUncertaintyAgent()
        return await fallback.assess(**available_args)

    def _build_enhanced_report(self) -> EnhancedCDSReport:
        """Assemble the complete EnhancedCDSReport combining all 10-step findings."""
        base_report = self._state.final_report or CDSReport(patient_summary="Clinical decision support evaluation.")

        # Consolidate safety alerts
        safety_warnings: List[str] = []
        unsafe_flags: List[str] = []

        # Drug interaction warnings
        if self._state.drug_interactions:
            for ix in self._state.drug_interactions.interactions_found:
                if ix.severity in (Severity.CRITICAL, Severity.HIGH) or getattr(ix.severity, "value", "") in ("critical", "high"):
                    safety_warnings.append(
                        f"DRUG INTERACTION [{ix.severity.value.upper()}]: {ix.drug_a} + {ix.drug_b} — {ix.description}"
                    )

        # Guideline conflict alerts
        if self._state.conflict_detection:
            for c in self._state.conflict_detection.conflicts:
                if c.severity in (Severity.CRITICAL, Severity.HIGH) or getattr(c.severity, "value", "") in ("critical", "high"):
                    safety_warnings.append(
                        f"GUIDELINE CONFLICT [{c.severity.value.upper()}]: {c.description}"
                    )

        # Verification refuted claim alerts
        if self._state.verification_summary and self._state.verification_summary.refuted_count > 0:
            for v in self._state.verification_summary.claim_verifications:
                if v.status == ClaimSupportStatus.REFUTED:
                    unsafe_flags.append(f"REFUTED CLAIM: '{v.claim_text}' — {v.verification_note}")
                    safety_warnings.append(
                        f"VERIFICATION CONTRADICTION: Statement contradicted by evidence: '{v.claim_text[:90]}...'"
                    )

        # Abstention alert
        if self._state.uncertainty_report and self._state.uncertainty_report.abstention_decision.should_abstain:
            safety_warnings.append(
                f"ABSTENTION ALERT: AI recommendation withheld from automated action. Reason: "
                f"{self._state.uncertainty_report.abstention_decision.abstention_reason}"
            )

        # Determine human review status
        if safety_warnings or unsafe_flags:
            review_status = HumanReviewStatus.REQUIRES_CORRECTION
        elif self._state.verification_summary and self._state.verification_summary.unsupported_claim_rate > 0.3:
            review_status = HumanReviewStatus.PENDING
        else:
            review_status = HumanReviewStatus.PENDING

        missing_fields: List[str] = []
        if self._state.normalized_context and self._state.normalized_context.missing_fields:
            missing_fields = self._state.normalized_context.missing_fields.missing_field_names()

        return EnhancedCDSReport(
            patient_summary=base_report.patient_summary,
            differential_diagnosis=base_report.differential_diagnosis,
            drug_interaction_warnings=base_report.drug_interaction_warnings,
            guideline_recommendations=base_report.guideline_recommendations,
            suggested_next_steps=base_report.suggested_next_steps,
            caveats=base_report.caveats,
            conflicts=base_report.conflicts,
            sources_cited=base_report.sources_cited,
            generated_at=base_report.generated_at,
            # Phase 6 Verification
            verification_summary=self._state.verification_summary,
            verified_claims=(
                self._state.verification_summary.claim_verifications
                if self._state.verification_summary else []
            ),
            # Phase 4 & 5 Evidence
            evidence_bundle=self._state.evidence_bundle,
            evidence_grades=(
                self._state.graded_evidence.graded_passages
                if self._state.graded_evidence else []
            ),
            # Phase 7 Uncertainty
            uncertainty_report=self._state.uncertainty_report,
            missing_patient_information=missing_fields,
            # Safety and Review
            safety_warnings=safety_warnings,
            unsafe_recommendation_flags=unsafe_flags,
            human_review_status=review_status,
            pipeline_version="2.0.0-evidence-verified",
            experiment_id=self._state.experiment_id or "experiment_4",
        )

    @staticmethod
    def _bundle_to_guidelines(bundle: EvidenceBundle) -> GuidelineRetrievalResult:
        """Convert EvidenceBundle passages into GuidelineRetrievalResult for original tools."""
        excerpts = [
            GuidelineExcerpt(
                title=p.title or f"Clinical Evidence {p.passage_id}",
                excerpt=p.text,
                source=str(p.source.value if hasattr(p.source, "value") else p.source),
                url=p.url,
                relevance_score=p.reranker_score or p.hybrid_score or p.dense_score,
            )
            for p in bundle.passages
        ]
        return GuidelineRetrievalResult(query=bundle.query, excerpts=excerpts)

    @staticmethod
    def _build_report_sections(report: CDSReport) -> Dict[str, str]:
        """Deconstruct report into named sections for atomic claim extraction."""
        sections: Dict[str, str] = {}
        if report.patient_summary:
            sections["patient_summary"] = report.patient_summary

        if report.differential_diagnosis:
            dx_lines = []
            for d in report.differential_diagnosis:
                line = f"{d.diagnosis} (likelihood: {d.likelihood.value}): {d.reasoning}"
                if d.supporting_evidence:
                    line += f" Evidence: {' '.join(d.supporting_evidence)}"
                dx_lines.append(line)
            sections["differential_diagnosis"] = "\n".join(dx_lines)

        if report.guideline_recommendations:
            sections["guideline_recommendations"] = "\n".join(report.guideline_recommendations)

        if report.suggested_next_steps:
            sections["suggested_next_steps"] = "\n".join(
                f"{a.action} [{a.priority.value}]: {a.rationale}" for a in report.suggested_next_steps
            )

        if report.caveats:
            sections["caveats"] = "\n".join(report.caveats)

        return sections

    def _skip_remaining_steps(self, after_step_id: str) -> List[AgentStep]:
        """Mark subsequent pending steps as SKIPPED and return them for yielding."""
        skipped: List[AgentStep] = []
        found = False
        for step in self._state.steps:
            if step.step_id == after_step_id:
                found = True
                continue
            if found and step.status == AgentStepStatus.PENDING:
                step.status = AgentStepStatus.SKIPPED
                step.error = f"Skipped: prerequisite step '{after_step_id}' failed"
                skipped.append(step)
        return skipped

    def _mark_running(self, step_id: str) -> AgentStep:
        """Mark step as RUNNING and return it for immediate streaming."""
        step = self._get_step(step_id)
        step.status = AgentStepStatus.RUNNING
        return step

    async def _run_step(self, step_id: str, fn: Callable[..., Any], *args: Any) -> AgentStep:
        """Execute a single step function, logging timing and catching exceptions."""
        step = self._get_step(step_id)
        step.status = AgentStepStatus.RUNNING
        start = time.monotonic()

        try:
            res = fn(*args)
            if asyncio.iscoroutine(res):
                await res
            if step.status != AgentStepStatus.FAILED:
                step.status = AgentStepStatus.COMPLETED
        except Exception as e:
            logger.error("Step execution error in '%s': %s", step_id, e, exc_info=True)
            step.status = AgentStepStatus.FAILED
            step.error = str(e)
        finally:
            step.duration_ms = int((time.monotonic() - start) * 1000)

        return step

    def _get_step(self, step_id: str) -> AgentStep:
        """Retrieve step by step_id from the current state."""
        for step in self._state.steps:
            if step.step_id == step_id:
                return step
        raise ValueError(f"Unknown step_id: {step_id}")


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator Factory
# ──────────────────────────────────────────────────────────────────────────────

class OrchestratorFactory:
    """
    Factory for instantiating CDSS orchestrators.

    Allows seamless switching between the baseline 6-step orchestrator
    (Experiment 1) and the enhanced 10-step orchestrator (Experiments 2-4).
    """

    @staticmethod
    def create(use_langgraph: Optional[bool] = None) -> Union[Orchestrator, EnhancedOrchestrator]:
        """
        Create and return an Orchestrator or EnhancedOrchestrator instance.

        Args:
            use_langgraph: If None, reads from settings.use_langgraph.
                           If True, returns an EnhancedOrchestrator instance.
                           If False, returns an original Orchestrator instance.

        Returns:
            An instance of Orchestrator or EnhancedOrchestrator.
        """
        if use_langgraph is None:
            use_langgraph = getattr(settings, "use_langgraph", False)

        if use_langgraph:
            return EnhancedOrchestrator(use_langgraph=True)
        else:
            return Orchestrator()
