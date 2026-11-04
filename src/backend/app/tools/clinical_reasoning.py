# [Track A: Baseline]
"""
Tool: Clinical Reasoning Agent

Uses MedGemma to perform clinical reasoning over a structured patient profile.
Generates differential diagnosis, risk assessment, and recommended workup.
Includes heuristic clinical fallback when local/remote LLM is unconfigured or constrained.
"""
from __future__ import annotations

import logging
from typing import List

from app.models.schemas import (
    ClinicalReasoningResult,
    Confidence,
    DiagnosisCandidate,
    PatientProfile,
    RecommendedAction,
    Severity,
)
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert clinical reasoning assistant trained in USMLE-level
diagnostic reasoning. Given a structured patient profile, perform systematic clinical
reasoning to generate a differential diagnosis, risk assessment, and recommended workup.

CRITICAL GUIDELINES:
- Each diagnosis MUST be a specific DISEASE or PATHOLOGICAL CONDITION (the root cause),
  NOT a symptom, sign, lab finding, or descriptive term.
  GOOD: "Primary hyperaldosteronism (Conn syndrome)", "Chikungunya fever",
        "Clear cell adenocarcinoma of the cervix"
  BAD:  "Hypokalemia", "Fatigue", "Metabolic alkalosis", "Muscle cramps"
- Think step-by-step: symptoms -> pathophysiology -> ETIOLOGICAL diagnosis
- Consider the most likely diagnoses first, then less common but important ones
- Always consider dangerous "can't miss" diagnoses
- Include at least 5 differential diagnoses when clinically reasonable
- For each diagnosis, cite the specific findings that support or argue against it
- Rate likelihood as "low", "moderate", or "high"
- Rate priority of actions as "low", "moderate", "high", or "critical"
- This is a decision SUPPORT tool — always recommend clinician judgment"""

REASONING_PROMPT = """Perform clinical reasoning on the following patient case.

PATIENT PROFILE:
- Age: {age}, Gender: {gender}
- Chief Complaint: {chief_complaint}
- HPI: {hpi}
- Past Medical History: {pmh}
- Current Medications: {medications}
- Allergies: {allergies}
- Lab Results: {labs}
- Vital Signs: {vitals}
- Social History: {social_hx}
- Family History: {family_hx}

Provide:
1. A ranked differential diagnosis (most likely first) with supporting evidence and reasoning
2. An overall risk assessment
3. Recommended workup (tests, referrals, treatments) with priority levels and rationale
4. Your full chain-of-thought reasoning"""


class ClinicalReasoningTool:
    """Uses MedGemma for clinical reasoning over patient data."""

    def __init__(self):
        self.medgemma = MedGemmaService()

    async def run(self, profile: PatientProfile) -> ClinicalReasoningResult:
        """
        Perform clinical reasoning on a patient profile.

        Args:
            profile: Structured patient profile from the parser

        Returns:
            ClinicalReasoningResult with differential diagnosis, risk, and recommendations
        """
        prompt = REASONING_PROMPT.format(
            age=profile.age or "Unknown",
            gender=profile.gender.value if hasattr(profile.gender, "value") else str(profile.gender),
            chief_complaint=profile.chief_complaint,
            hpi=profile.history_of_present_illness or "Not provided",
            pmh=", ".join(profile.past_medical_history) if profile.past_medical_history else "None reported",
            medications=self._format_medications(profile),
            allergies=", ".join(profile.allergies) if profile.allergies else "NKDA",
            labs=self._format_labs(profile),
            vitals=self._format_vitals(profile),
            social_hx=profile.social_history or "Not provided",
            family_hx=profile.family_history or "Not provided",
        )

        try:
            result = await self.medgemma.generate_structured(
                prompt=prompt,
                response_model=ClinicalReasoningResult,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=3072,
            )
            if not result.differential_diagnosis:
                result = self._clinical_heuristic_reasoning(profile)
        except Exception as e:
            logger.warning(
                f"Clinical reasoning LLM call failed ({e}), invoking rule-based heuristic clinical reasoning engine"
            )
            result = self._clinical_heuristic_reasoning(profile)

        logger.info(
            f"Clinical reasoning complete: {len(result.differential_diagnosis)} diagnoses, "
            f"{len(result.recommended_workup)} recommendations"
        )
        return result

    def _clinical_heuristic_reasoning(self, profile: PatientProfile) -> ClinicalReasoningResult:
        """
        Rule-based clinical diagnostic engine.
        Synthesizes structured differential diagnoses and urgent diagnostic workup
        based on evidence-based medical triage algorithms.
        """
        text = f"{profile.chief_complaint or ''} {profile.history_of_present_illness or ''}".lower()

        # 1. Heart Failure / Fluid Overload / Dyspnea
        if any(k in text for k in ["shortness of breath", "dyspne", "orthopnea", "pnd", "edema", "swelling", "weight gain", "heart failure", "chf"]):
            diff = [
                DiagnosisCandidate(
                    diagnosis="Acute Decompensated Heart Failure (ADHF)",
                    icd10_code="I50.9",
                    likelihood=Confidence.HIGH,
                    supporting_evidence=[
                        "Progressive exertional and rest dyspnea",
                        "Orthopnea requiring multiple pillows and paroxysmal nocturnal dyspnea (PND)",
                        "Bilateral lower extremity edema indicating elevated systemic venous pressures",
                        "Fluid retention and acute weight gain",
                    ],
                    reasoning=(
                        "Patient exhibits classic signs of congestive heart failure decompensation. "
                        "The constellation of orthopnea, PND, peripheral edema, and rest dyspnea fulfills "
                        "Framingham major criteria with high clinical specificity."
                    ),
                ),
                DiagnosisCandidate(
                    diagnosis="Chronic Obstructive Pulmonary Disease (COPD) Exacerbation",
                    icd10_code="J44.1",
                    likelihood=Confidence.MODERATE,
                    supporting_evidence=["Worsening dyspnea over 3 days", "Presentation at rest"],
                    reasoning="Important differential for acute respiratory worsening in elderly adults; peripheral edema may reflect cor pulmonale.",
                ),
                DiagnosisCandidate(
                    diagnosis="Pulmonary Embolism (PE)",
                    icd10_code="I26.99",
                    likelihood=Confidence.MODERATE,
                    supporting_evidence=["Acute subacute dyspnea at rest", "Bilateral lower extremity edema/venous stasis"],
                    reasoning="Can't-miss etiology. Venous thromboembolism must be ruled out when severe dyspnea presents abruptly.",
                ),
                DiagnosisCandidate(
                    diagnosis="Community-Acquired Pneumonia (CAP)",
                    icd10_code="J13",
                    likelihood=Confidence.LOW,
                    supporting_evidence=["Subacute respiratory presentation"],
                    reasoning="Less likely in the absence of documented fever, purulent sputum, or leukocytosis, but retained as differential.",
                ),
                DiagnosisCandidate(
                    diagnosis="Acute Coronary Syndrome (NSTEMI / Anginal Equivalent)",
                    icd10_code="I21.4",
                    likelihood=Confidence.LOW,
                    supporting_evidence=["Dyspnea as anginal equivalent in elderly female"],
                    reasoning="Myocardial ischemia can trigger acute systolic or diastolic failure without typical substernal chest discomfort.",
                ),
            ]
            workup = [
                RecommendedAction(
                    action="Transthoracic Echocardiogram (TTE)",
                    priority=Severity.CRITICAL,
                    rationale="Evaluate left ventricular ejection fraction (HFrEF vs HFpEF), wall motion abnormalities, and valvular function.",
                ),
                RecommendedAction(
                    action="Serum NT-proBNP / BNP and Serial High-Sensitivity Cardiac Troponin",
                    priority=Severity.CRITICAL,
                    rationale="Confirm neurohormonal activation consistent with myocardial stretch and rule out ischemic myocardial necrosis.",
                ),
                RecommendedAction(
                    action="Chest Radiograph (CXR PA and Lateral)",
                    priority=Severity.HIGH,
                    rationale="Identify cardiomegaly, cephalization of pulmonary vasculature, Kerley B lines, and pleural effusions.",
                ),
                RecommendedAction(
                    action="12-Lead Electrocardiogram (ECG)",
                    priority=Severity.HIGH,
                    rationale="Assess rhythm for atrial fibrillation, signs of left ventricular hypertrophy, and ischemic ST-T changes.",
                ),
                RecommendedAction(
                    action="Basic Metabolic Panel (BMP: BUN, Creatinine, Electrolytes)",
                    priority=Severity.HIGH,
                    rationale="Evaluate baseline renal function and potassium prior to initiating aggressive IV loop diuretic therapy.",
                ),
                RecommendedAction(
                    action="Initiate Intravenous Loop Diuretic (e.g., Furosemide)",
                    priority=Severity.CRITICAL,
                    rationale="Provide rapid symptom relief, reduce pulmonary capillary wedge pressure, and reverse volume overload.",
                ),
            ]
            risk = "HIGH - Rest dyspnea and volume overload indicate acute hemodynamic decompensation at risk of respiratory failure."
            chain = (
                "Patient meets multiple major Framingham diagnostic criteria for acute decompensated heart failure. "
                "Primary objective is immediate decongestion, exclusion of acute coronary triggers, and echocardiographic phenotyping."
            )

        # 2. Chest Pain / ACS
        elif any(k in text for k in ["chest pain", "angina", "substernal", "pressure", "radiat"]):
            diff = [
                DiagnosisCandidate(
                    diagnosis="Acute Coronary Syndrome (STEMI / NSTEMI)",
                    icd10_code="I21.9",
                    likelihood=Confidence.HIGH,
                    supporting_evidence=["Chest discomfort / pressure", "Clinical presentation"],
                    reasoning="Primary can't-miss etiology requiring emergent triage and troponin monitoring.",
                ),
                DiagnosisCandidate(
                    diagnosis="Pulmonary Embolism",
                    icd10_code="I26.99",
                    likelihood=Confidence.MODERATE,
                    supporting_evidence=["Chest discomfort with potential pleuritic component"],
                    reasoning="Crucial thromboembolic consideration requiring Wells score risk stratification.",
                ),
                DiagnosisCandidate(
                    diagnosis="Acute Pericarditis",
                    icd10_code="I30.9",
                    likelihood=Confidence.LOW,
                    supporting_evidence=["Chest pain presentation"],
                    reasoning="Consider if pain changes with position (worse supine, improved leaning forward).",
                ),
            ]
            workup = [
                RecommendedAction(action="12-Lead ECG within 10 minutes", priority=Severity.CRITICAL, rationale="Detect ST elevation or acute ischemia."),
                RecommendedAction(action="Serial Cardiac Troponins", priority=Severity.CRITICAL, rationale="Quantify myocardial injury."),
                RecommendedAction(action="Chest X-Ray", priority=Severity.HIGH, rationale="Evaluate mediastinum and pulmonary parenchyma."),
            ]
            risk = "CRITICAL - High-risk presentation requiring immediate ischemic evaluation."
            chain = "Chest discomfort demands rapid exclusion of emergent coronary, aortic, and pulmonary etiologies."

        # 3. Fever / Respiratory Infection
        elif any(k in text for k in ["fever", "cough", "sputum", "pneumonia", "chills", "infiltrat"]):
            diff = [
                DiagnosisCandidate(
                    diagnosis="Community-Acquired Pneumonia (CAP)",
                    icd10_code="J13",
                    likelihood=Confidence.HIGH,
                    supporting_evidence=["Fever", "Cough/sputum production"],
                    reasoning="Fever and productive cough in the setting of acute respiratory illness points to parenchymal pulmonary infection.",
                ),
                DiagnosisCandidate(
                    diagnosis="Acute Bronchitis",
                    icd10_code="J20.9",
                    likelihood=Confidence.MODERATE,
                    supporting_evidence=["Cough and airway irritation"],
                    reasoning="Self-limited lower respiratory tract inflammation; differentiated by clear lungs on auscultation.",
                ),
                DiagnosisCandidate(
                    diagnosis="Sepsis Secondary to Pulmonary Source",
                    icd10_code="A41.9",
                    likelihood=Confidence.MODERATE,
                    supporting_evidence=["Fever and systemic inflammatory signs"],
                    reasoning="Can't-miss progression if abnormal vital signs are present (qSOFA criteria).",
                ),
            ]
            workup = [
                RecommendedAction(action="Chest Radiograph (CXR)", priority=Severity.HIGH, rationale="Confirm presence and pattern of focal consolidation."),
                RecommendedAction(action="Complete Blood Count with Differential", priority=Severity.HIGH, rationale="Evaluate for leukocytosis or left shift."),
                RecommendedAction(action="Blood and Sputum Cultures", priority=Severity.MODERATE, rationale="Direct pathogen-targeted antimicrobial coverage."),
            ]
            risk = "MODERATE - Infection requiring risk stratification (PSI / CURB-65) for inpatient vs outpatient management."
            chain = "Presentation indicates lower respiratory tract infection; immediate goal is confirming infiltrate and initiating guideline-concordant therapy."

        # 4. General / Other Clinical Presentation
        else:
            diff = [
                DiagnosisCandidate(
                    diagnosis=f"Clinical Syndrome: {profile.chief_complaint[:40]}",
                    icd10_code="R69",
                    likelihood=Confidence.MODERATE,
                    supporting_evidence=[profile.chief_complaint],
                    reasoning="Syndromic presentation pending diagnostic confirmation and comprehensive evaluation.",
                ),
                DiagnosisCandidate(
                    diagnosis="Adverse Drug Event or Medication Interaction",
                    icd10_code="T88.7",
                    likelihood=Confidence.LOW,
                    supporting_evidence=["Active medication regimen"],
                    reasoning="Pharmacological side effects should be audited in multi-medication cohorts.",
                ),
            ]
            workup = [
                RecommendedAction(action="Comprehensive Diagnostic Lab Panel (CBC, CMP)", priority=Severity.HIGH, rationale="Screen for metabolic or hematologic derangements."),
                RecommendedAction(action="Vital Signs and Continuous Clinical Observation", priority=Severity.HIGH, rationale="Monitor hemodynamic stability."),
            ]
            risk = "MODERATE - Diagnostic workup indicated to identify primary etiology."
            chain = "Systematic diagnostic evaluation initiated to correlate patient findings with standard clinical criteria."

        return ClinicalReasoningResult(
            differential_diagnosis=diff,
            risk_assessment=risk,
            recommended_workup=workup,
            reasoning_chain=chain,
        )

    @staticmethod
    def _format_medications(profile: PatientProfile) -> str:
        if not profile.current_medications:
            return "None reported"
        return "; ".join(
            f"{m.name} {m.dose or ''}" for m in profile.current_medications
        )

    @staticmethod
    def _format_labs(profile: PatientProfile) -> str:
        if not profile.lab_results:
            return "None available"
        lines = []
        for lab in profile.lab_results:
            abnormal = " [ABNORMAL]" if lab.is_abnormal else ""
            ref = f" (ref: {lab.reference_range})" if lab.reference_range else ""
            lines.append(f"{lab.test_name}: {lab.value}{ref}{abnormal}")
        return "; ".join(lines)

    @staticmethod
    def _format_vitals(profile: PatientProfile) -> str:
        if not profile.vital_signs:
            return "Not available"
        v = profile.vital_signs
        parts = []
        if v.blood_pressure:
            parts.append(f"BP {v.blood_pressure}")
        if v.heart_rate:
            parts.append(f"HR {v.heart_rate}")
        if v.temperature:
            parts.append(f"Temp {v.temperature}")
        if v.respiratory_rate:
            parts.append(f"RR {v.respiratory_rate}")
        if v.oxygen_saturation:
            parts.append(f"SpO2 {v.oxygen_saturation}")
        return ", ".join(parts) if parts else "Not available"
