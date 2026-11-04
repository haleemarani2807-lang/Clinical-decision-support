"""
MTSamples Preprocessing Script.

Cleans the raw medical transcription dataset and outputs:
  data/processed/mtsamples_processed.csv  — cleaned, enriched records
  data/processed/mtsamples_train.csv      — 80% training split
  data/processed/mtsamples_test.csv       — 20% test split

Data schema:
  Raw:      note_id, description, medical_specialty, sample_name,
            transcription, keywords
  Processed: + cleaned_transcription, word_count, has_labs, has_meds,
               has_vitals, missing_fields, specialty_group

Usage:
  python data/scripts/preprocess_mtsamples.py
  python data/scripts/preprocess_mtsamples.py --input data/raw/mtsamples/mtsamples.csv
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "raw" / "mtsamples" / "mtsamples.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Specialties mapped to research-relevant groups
SPECIALTY_MAP = {
    "Cardiology": "Cardiology",
    "Emergency Room Reports": "Emergency Medicine",
    "Endocrinology": "Endocrinology",
    "Pulmonology": "Pulmonology",
    "Neurology": "Neurology",
    "Gastroenterology": "Gastroenterology",
    "Infectious Disease": "Infectious Disease",
    "Psychiatry / Psychology": "Psychiatry",
    "Pediatrics - Neonatal": "Pediatrics",
    "Nephrology": "Nephrology",
    "Hematology - Oncology": "Hematology",
    "Rheumatology": "Rheumatology",
    "Obstetrics / Gynecology": "OB/GYN",
    "Internal Medicine": "Internal Medicine",
    "General Medicine": "General Medicine",
    "Surgery": "Surgery",
    "Orthopedic": "Orthopedics",
    "Radiology": "Radiology",
}

# Patterns for clinical data presence detection
LAB_PATTERNS = re.compile(
    r'\b(hemoglobin|hgb|wbc|platelet|creatinine|bun|sodium|potassium|glucose|'
    r'hba1c|troponin|bnp|tsh|alt|ast|bili|lactate|pao2|pco2|inr|pt|aptt)\b',
    re.IGNORECASE
)
MED_PATTERNS = re.compile(
    r'\b(mg|mcg|units|tablet|capsule|injection|iv|po|bid|tid|qid|qd|prn|'
    r'metformin|lisinopril|atorvastatin|aspirin|warfarin|heparin|insulin|'
    r'metoprolol|amlodipine|furosemide|omeprazole|atorvastatin)\b',
    re.IGNORECASE
)
VITAL_PATTERNS = re.compile(
    r'\b(blood pressure|bp|heart rate|hr|pulse|temperature|temp|respiratory rate|'
    r'rr|oxygen saturation|o2 sat|spo2|weight|height|bmi)\b',
    re.IGNORECASE
)


def detect_missing_fields(row) -> list[str]:
    """Return list of clinically important fields that appear missing."""
    missing = []
    text = str(row.get("transcription", "")).lower()
    if not text or len(text) < 50:
        return ["transcription"]
    if not VITAL_PATTERNS.search(text):
        missing.append("vital_signs")
    if not LAB_PATTERNS.search(text):
        missing.append("laboratory_results")
    if not re.search(r'\b(allerg|allergic|nkda|no known)\b', text, re.IGNORECASE):
        missing.append("allergy_status")
    if not re.search(r'\b(chief complaint|presents with|presenting with|referred for)\b', text, re.IGNORECASE):
        missing.append("chief_complaint")
    return missing


def clean_text(text: str) -> str:
    """Basic clinical text cleaning."""
    if not isinstance(text, str):
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-ASCII control chars but keep medical symbols
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def preprocess(input_path: Path, output_dir: Path, test_size: float = 0.2, seed: int = 42):
    """Main preprocessing pipeline."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed. Run: pip install pandas")
        sys.exit(1)

    if not input_path.exists():
        logger.error(
            "Input file not found: %s\n"
            "Run: python data/scripts/download_kaggle.py --dataset mtsamples", input_path
        )
        sys.exit(1)

    logger.info("Loading MTSamples from %s...", input_path)
    df = pd.read_csv(input_path)
    logger.info("Raw records: %d", len(df))

    # ── Standardise columns ──────────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Add index if missing
    if "note_id" not in df.columns:
        df.insert(0, "note_id", range(1, len(df) + 1))

    # ── Clean text fields ────────────────────────────────────────────────────
    for col in ["transcription", "description", "keywords", "sample_name"]:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(clean_text)

    # ── Drop empty transcriptions ────────────────────────────────────────────
    initial_count = len(df)
    df = df[df["transcription"].str.len() > 100].copy()
    logger.info("After removing short/empty transcriptions: %d (removed %d)",
                len(df), initial_count - len(df))

    # ── Feature engineering ──────────────────────────────────────────────────
    df["word_count"] = df["transcription"].str.split().str.len()
    df["has_labs"] = df["transcription"].apply(
        lambda t: bool(LAB_PATTERNS.search(t))
    )
    df["has_medications"] = df["transcription"].apply(
        lambda t: bool(MED_PATTERNS.search(t))
    )
    df["has_vitals"] = df["transcription"].apply(
        lambda t: bool(VITAL_PATTERNS.search(t))
    )
    df["missing_fields"] = df.apply(
        lambda row: "|".join(detect_missing_fields(row)), axis=1
    )
    df["missing_field_count"] = df["missing_fields"].apply(
        lambda x: len(x.split("|")) if x else 0
    )

    # ── Specialty normalisation ───────────────────────────────────────────────
    if "medical_specialty" in df.columns:
        df["specialty_group"] = df["medical_specialty"].apply(
            lambda s: SPECIALTY_MAP.get(str(s).strip(), "Other")
        )

    # ── Data quality label ────────────────────────────────────────────────────
    df["data_source"] = "mtsamples_real_deidentified"
    df["synthetic"] = False

    # ── Train/test split (stratified by specialty) ────────────────────────────
    try:
        from sklearn.model_selection import train_test_split
        strat_col = "specialty_group" if "specialty_group" in df.columns else None
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df[strat_col] if strat_col else None
        )
    except Exception as e:
        logger.warning("Stratified split failed (%s) — using random split", e)
        train_df = df.sample(frac=1 - test_size, random_state=seed)
        test_df = df.drop(train_df.index)

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_path = output_dir / "mtsamples_processed.csv"
    train_path = output_dir / "mtsamples_train.csv"
    test_path = output_dir / "mtsamples_test.csv"

    df.to_csv(processed_path, index=False, encoding="utf-8")
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    logger.info("\n=== MTSamples Preprocessing Complete ===")
    logger.info("Total records:  %d", len(df))
    logger.info("Train records:  %d", len(train_df))
    logger.info("Test records:   %d", len(test_df))
    logger.info("Avg word count: %.0f", df["word_count"].mean())
    logger.info("With labs:      %d (%.1f%%)", df["has_labs"].sum(), 100 * df["has_labs"].mean())
    logger.info("With meds:      %d (%.1f%%)", df["has_medications"].sum(), 100 * df["has_medications"].mean())
    logger.info("With vitals:    %d (%.1f%%)", df["has_vitals"].sum(), 100 * df["has_vitals"].mean())
    if "specialty_group" in df.columns:
        logger.info("\nSpecialty distribution:\n%s", df["specialty_group"].value_counts().to_string())
    logger.info("\nOutputs saved:")
    logger.info("  %s", processed_path)
    logger.info("  %s", train_path)
    logger.info("  %s", test_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess MTSamples dataset")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess(args.input, args.output_dir, args.test_size, args.seed)


if __name__ == "__main__":
    main()
