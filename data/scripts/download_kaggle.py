"""
Dataset Download Script — Kaggle datasets.

Downloads:
  1. Medical Transcriptions (MTSamples) — tboyle10/medicaltranscriptions
  2. Diabetes 130-US Hospitals — brandao/diabetes
  3. (Optional) NBME Clinical Patient Notes — requires competition acceptance

Usage:
  python data/scripts/download_kaggle.py --all
  python data/scripts/download_kaggle.py --dataset mtsamples
  python data/scripts/download_kaggle.py --check       # check credentials only

Prerequisites:
  pip install kaggle
  Place ~/.kaggle/kaggle.json with your API credentials, OR set:
    KAGGLE_USERNAME and KAGGLE_KEY environment variables.

Datasets are saved to data/raw/ and never committed to Git.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Project root
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

DATASETS = {
    "mtsamples": {
        "slug": "tboyle10/medicaltranscriptions",
        "description": "Medical Transcriptions (MTSamples) — 4999 clinical transcription notes",
        "output_dir": RAW_DIR / "mtsamples",
        "expected_files": ["mtsamples.csv"],
        "competition": False,
    },
    "diabetes": {
        "slug": "brandao/diabetes",
        "description": "Diabetes 130-US Hospitals — structured EHR records",
        "output_dir": RAW_DIR / "diabetes",
        "expected_files": ["diabetic_data.csv"],
        "competition": False,
    },
    "nbme": {
        "slug": "nbme-score-clinical-patient-notes",
        "description": "NBME Clinical Patient Notes — requires competition acceptance",
        "output_dir": RAW_DIR / "nbme",
        "expected_files": ["train.csv", "features.csv", "patient_notes.csv"],
        "competition": True,
    },
}


def check_credentials() -> bool:
    """Verify Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_file = kaggle_json.exists()
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if has_file:
        logger.info("✓ Kaggle credentials found at %s", kaggle_json)
        return True
    if has_env:
        logger.info("✓ Kaggle credentials found in environment variables")
        return True

    logger.error(
        "✗ Kaggle credentials not found.\n"
        "  Option 1: Download kaggle.json from https://www.kaggle.com/settings/account\n"
        "             and place it at ~/.kaggle/kaggle.json\n"
        "  Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
    )
    return False


def setup_kaggle_env():
    """Set KAGGLE_USERNAME and KAGGLE_KEY from env if not already set via kaggle.json."""
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key


def download_dataset(name: str, force: bool = False) -> bool:
    """
    Download a Kaggle dataset.

    Args:
        name: Dataset key from DATASETS dict.
        force: Re-download even if files already exist.

    Returns:
        True if successful.
    """
    if name not in DATASETS:
        logger.error("Unknown dataset: %s. Available: %s", name, list(DATASETS.keys()))
        return False

    ds = DATASETS[name]
    out_dir: Path = ds["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if not force:
        existing = [out_dir / f for f in ds["expected_files"] if (out_dir / f).exists()]
        if len(existing) == len(ds["expected_files"]):
            logger.info("✓ %s already downloaded at %s", name, out_dir)
            return True

    logger.info("Downloading %s — %s", name, ds["description"])

    if ds["competition"]:
        logger.warning(
            "Dataset '%s' is from a Kaggle competition.\n"
            "You must have accepted the competition rules at:\n"
            "  https://www.kaggle.com/competitions/%s\n"
            "Attempting download...", name, ds["slug"]
        )

    try:
        import kaggle  # noqa: F401
        setup_kaggle_env()

        if ds["competition"]:
            # Competition dataset
            import subprocess
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", ds["slug"], "-p", str(out_dir)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error("Competition download failed:\n%s", result.stderr)
                logger.info("Skipping NBME — using MTSamples + synthetic fallback instead.")
                return False
        else:
            # Regular dataset
            import subprocess
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", ds["slug"], "-p", str(out_dir), "--unzip"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error("Dataset download failed:\n%s", result.stderr)
                return False

        logger.info("✓ Downloaded %s to %s", name, out_dir)
        _extract_zips(out_dir)
        return True

    except ImportError:
        logger.error(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Then set up credentials (see --check option)."
        )
        return False
    except Exception as e:
        logger.error("Download failed for %s: %s", name, e)
        return False


def _extract_zips(directory: Path):
    """Extract any zip files found in a directory."""
    for zip_path in directory.glob("*.zip"):
        logger.info("Extracting %s...", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(directory)
        zip_path.unlink()


def verify_downloads():
    """Check which datasets are available locally."""
    logger.info("\n=== Dataset Status ===")
    for name, ds in DATASETS.items():
        out_dir: Path = ds["output_dir"]
        files_present = [f for f in ds["expected_files"] if (out_dir / f).exists()]
        if len(files_present) == len(ds["expected_files"]):
            sizes = {f: f"{(out_dir / f).stat().st_size / 1024 / 1024:.1f} MB"
                     for f in ds["expected_files"]}
            logger.info("✓ %-12s %s", name, sizes)
        elif files_present:
            logger.warning("⚠ %-12s Partial: %s/%s files", name,
                           len(files_present), len(ds["expected_files"]))
        else:
            logger.warning("✗ %-12s Not downloaded", name)


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle datasets for Evidence-Verified CDSS"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Download specific dataset")
    parser.add_argument("--check", action="store_true", help="Check credentials and existing downloads")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    if args.check or args.all or args.dataset:
        if not check_credentials():
            if not (args.check):
                sys.exit(1)
            return

    if args.check:
        verify_downloads()
        return

    if args.all:
        results = {}
        for name in ["mtsamples", "diabetes"]:  # NBME skipped by default
            results[name] = download_dataset(name, force=args.force)

        logger.info("\n=== Download Summary ===")
        for name, success in results.items():
            status = "✓" if success else "✗"
            logger.info("%s %s", status, name)

        # NBME: check and warn
        logger.info(
            "\nNBME dataset: run with --dataset nbme after accepting competition rules at\n"
            "  https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes\n"
            "  Fallback: synthetic patient cases will be used if NBME is unavailable."
        )

    elif args.dataset:
        success = download_dataset(args.dataset, force=args.force)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
