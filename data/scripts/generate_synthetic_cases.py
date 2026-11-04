"""
Synthetic Clinical Case Generator.

Produces controlled synthetic patient cases for:
  - Missing-information experiments (deliberately omit fields)
  - Contradiction experiments (conflicting data)
  - Uncertainty experiments (ambiguous presentations)
  - Benchmark cases with known ground-truth labels

Output: data/processed/synthetic_cases.json
        data/processed/synthetic_cases_missing_info.json
        data/processed/synthetic_cases_contradictions.json

All records are clearly labelled synthetic=True and must never be presented
as real patient data.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Clinical scenario templates ───────────────────────────────────────────────
BASE_CASES = [
    {
        "case_id": "syn_001",
        "specialty": "Cardiology",
        "diagnosis_ground_truth": "Acute Coronary Syndrome / NSTEMI",
        "patient_text": (
            "65-year-old male with crushing substernal chest pain radiating to left arm for 2 hours. "
            "History of hypertension, hyperlipidemia, type 2 diabetes. "
            "Current medications: lisinopril 10mg daily, atorvastatin 40mg, metformin 500mg BID. "
            "No known drug allergies. "
            "Vitals: BP 158/94 mmHg, HR 98 bpm, RR 18/min, SpO2 96%, Temp 37.1°C. "
            "Labs: Troponin I 0.08 ng/mL (elevated), BNP 180 pg/mL, Glucose 210 mg/dL, "
            "Creatinine 1.1 mg/dL, HbA1c 8.2%. "
            "ECG: ST depression in V4-V6, no ST elevation."
        ),
        "missing_fields": [],
        "contradictions": [],
        "required_evidence": ["ACS guidelines", "antiplatelet therapy", "troponin interpretation"],
        "expected_uncertainty": "low",
    },
    {
        "case_id": "syn_002",
        "specialty": "Emergency Medicine",
        "diagnosis_ground_truth": "Sepsis / Bacterial Pneumonia",
        "patient_text": (
            "45-year-old female presenting with fever (39.8°C), productive cough with purulent sputum, "
            "shortness of breath, and altered mental status. "
            "Past medical history: COPD, alcohol use disorder. "
            "Current medications: tiotropium inhaler, albuterol PRN. "
            "NKDA. "
            "Vitals: BP 88/52 mmHg, HR 124 bpm, RR 28/min, SpO2 88% on room air. "
            "Labs: WBC 18.2 K/uL with 85% neutrophils, Lactate 3.8 mmol/L, "
            "Procalcitonin 8.4 ng/mL, Creatinine 2.1 mg/dL, Sodium 128 mEq/L. "
            "Chest X-ray: right lower lobe consolidation."
        ),
        "missing_fields": [],
        "contradictions": [],
        "required_evidence": ["Surviving Sepsis Campaign", "antibiotic selection", "fluid resuscitation"],
        "expected_uncertainty": "low",
    },
    {
        "case_id": "syn_003",
        "specialty": "Endocrinology",
        "diagnosis_ground_truth": "Diabetic Ketoacidosis",
        "patient_text": (
            "22-year-old male with type 1 diabetes presenting with nausea, vomiting, abdominal pain, "
            "polyuria, polydipsia for 2 days. Stopped insulin 3 days ago due to illness. "
            "Current medications: insulin glargine (when compliant). "
            "Vitals: BP 102/68 mmHg, HR 112 bpm, RR 24/min, SpO2 99%, Temp 37.4°C. "
            "Labs: Glucose 520 mg/dL, pH 7.18, Bicarbonate 10 mEq/L, "
            "Anion gap 26, Ketones 4+ on urinalysis, Potassium 5.8 mEq/L. "
            "No known allergies."
        ),
        "missing_fields": [],
        "contradictions": [],
        "required_evidence": ["DKA management", "insulin protocol", "electrolyte replacement"],
        "expected_uncertainty": "low",
    },
    {
        "case_id": "syn_004",
        "specialty": "Neurology",
        "diagnosis_ground_truth": "Ischemic Stroke",
        "patient_text": (
            "72-year-old female with sudden onset right arm weakness and expressive aphasia 90 minutes ago. "
            "History: atrial fibrillation (not on anticoagulation), hypertension. "
            "Current medications: metoprolol 25mg BID, amlodipine 5mg. "
            "Allergies: penicillin (rash). "
            "Vitals: BP 182/104 mmHg, HR 88 irregular, RR 16/min, SpO2 97%. "
            "NIHSS: 14. "
            "CT head: no hemorrhage. Last known well: 90 minutes ago."
        ),
        "missing_fields": [],
        "contradictions": [],
        "required_evidence": ["tPA eligibility", "stroke guidelines", "anticoagulation in AF"],
        "expected_uncertainty": "low",
    },
    {
        "case_id": "syn_005",
        "specialty": "Pulmonology",
        "diagnosis_ground_truth": "COPD Exacerbation",
        "patient_text": (
            "68-year-old male, 40 pack-year smoker, with worsening dyspnea, increased sputum production "
            "(yellow-green), and mild confusion over 3 days. "
            "Known COPD (FEV1/FVC 0.6, FEV1 42% predicted). "
            "Current medications: tiotropium, salmeterol/fluticasone, albuterol PRN. "
            "NKDA. "
            "Vitals: BP 136/82 mmHg, HR 106 bpm, RR 26/min, SpO2 84% on room air, Temp 37.9°C. "
            "ABG: pH 7.31, PaCO2 58 mmHg, PaO2 52 mmHg, HCO3 28 mEq/L."
        ),
        "missing_fields": [],
        "contradictions": [],
        "required_evidence": ["GOLD COPD guidelines", "NIV indications", "antibiotic use"],
        "expected_uncertainty": "low",
    },
]

# ── Missing-information variants ──────────────────────────────────────────────
def create_missing_info_variants(cases: list[dict]) -> list[dict]:
    """Create variants of cases with deliberate field omissions for missing-info experiments."""
    variants = []
    omission_scenarios = [
        ("no_vitals", "Vital signs omitted", ["BP", "HR", "RR", "SpO2", "Temp", "Vitals:"]),
        ("no_labs", "Laboratory results omitted", ["Labs:", "mg/dL", "mEq/L", "ng/mL", "K/uL", "mmol/L"]),
        ("no_medications", "Medication list omitted", ["medications:", "mg", "daily", "BID", "PRN"]),
        ("no_allergies", "Allergy status omitted", ["NKDA", "Allergies:", "allerg"]),
    ]

    for case in cases[:3]:  # First 3 cases get missing-info variants
        for omission_id, description, keywords in omission_scenarios:
            text = case["patient_text"]
            # Remove sentences containing the omitted keywords
            sentences = text.split(". ")
            filtered = [s for s in sentences if not any(kw.lower() in s.lower() for kw in keywords)]
            modified_text = ". ".join(filtered).strip()
            if not modified_text.endswith("."):
                modified_text += "."

            variant = {
                **case,
                "case_id": f"{case['case_id']}_{omission_id}",
                "patient_text": modified_text,
                "missing_fields": [omission_id.replace("no_", "")],
                "omission_description": description,
                "synthetic": True,
                "variant_type": "missing_information",
                "parent_case_id": case["case_id"],
            }
            variants.append(variant)

    return variants


# ── Contradiction variants ─────────────────────────────────────────────────────
def create_contradiction_variants(cases: list[dict]) -> list[dict]:
    """Create variants with deliberate contradictions for contradiction detection experiments."""
    variants = []
    contradictions = [
        {
            "case_id": "syn_con_001",
            "description": "Contradictory vital signs — bradycardia but tachycardia stated",
            "base_text": (
                "55-year-old male with chest palpitations. "
                "Vitals: HR 42 bpm on initial assessment. "
                "Nurse notes: patient tachycardic at HR 142 bpm. "
                "Medications: metoprolol 100mg BID. "
                "No allergies. No labs available."
            ),
            "contradiction_type": "vital_sign_discrepancy",
            "ground_truth_contradictions": ["heart_rate"],
            "specialty": "Cardiology",
        },
        {
            "case_id": "syn_con_002",
            "description": "Allergy conflict — penicillin allergy but amoxicillin prescribed",
            "base_text": (
                "34-year-old female with sinusitis symptoms for 10 days. "
                "Allergies: penicillin (anaphylaxis). "
                "Current treatment plan: amoxicillin 875mg BID for 10 days prescribed. "
                "Vitals stable. WBC 11.2 K/uL. No other medications."
            ),
            "contradiction_type": "allergy_medication_conflict",
            "ground_truth_contradictions": ["medication_allergy"],
            "specialty": "Internal Medicine",
        },
    ]

    for c in contradictions:
        c["synthetic"] = True
        c["variant_type"] = "contradiction"
        c["diagnosis_ground_truth"] = "varies"
        c["missing_fields"] = []
        c["required_evidence"] = []
        c["expected_uncertainty"] = "high"
        variants.append(c)

    return variants


def main():
    import random
    random.seed(42)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Mark all base cases as synthetic with provenance
    cases = []
    for case in BASE_CASES:
        case = {**case, "synthetic": True, "variant_type": "full_case",
                "data_source": "synthetic_hand_crafted_v1"}
        cases.append(case)

    missing_info = create_missing_info_variants(BASE_CASES)
    contradictions = create_contradiction_variants(BASE_CASES)

    # Save outputs
    for fname, data, desc in [
        ("synthetic_cases.json", cases, "base synthetic cases"),
        ("synthetic_cases_missing_info.json", missing_info, "missing-info variants"),
        ("synthetic_cases_contradictions.json", contradictions, "contradiction variants"),
    ]:
        path = PROCESSED_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("✓ Saved %d %s to %s", len(data), desc, path)

    logger.info("\n=== Synthetic Case Generation Complete ===")
    logger.info("Base cases:           %d", len(cases))
    logger.info("Missing-info variants: %d", len(missing_info))
    logger.info("Contradiction cases:   %d", len(contradictions))
    logger.info("\nAll records labelled synthetic=True.")
    logger.info("DO NOT present these as real patient data.")


if __name__ == "__main__":
    main()
