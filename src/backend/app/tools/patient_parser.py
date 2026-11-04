# [Track A: Baseline]
"""
Tool: Patient Data Parser

Parses raw free-text patient case descriptions into a structured PatientProfile.
Uses MedGemma for intelligent extraction.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.models.schemas import PatientProfile
from app.services.medgemma import MedGemmaService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical data extraction assistant. Your job is to parse 
free-text patient case descriptions into structured data. Extract all available clinical 
information accurately. If a field is not mentioned, leave it empty or use defaults. 
Never fabricate information that isn't present in the input."""

EXTRACTION_PROMPT = """Parse the following patient case into structured clinical data.

Patient Case:
{patient_text}

Extract:
- Age, gender
- Chief complaint (primary reason for visit)
- History of present illness (HPI)
- Past medical history (list of conditions)
- Current medications (name and dose)
- Allergies
- Lab results (test name, value, reference range, abnormal flag)
- Vital signs
- Social history
- Family history
- Any additional relevant notes"""


class PatientParserTool:
    """Parses raw patient text into a structured PatientProfile."""

    def __init__(self):
        self.medgemma = MedGemmaService()

    async def run(self, patient_text: str) -> PatientProfile:
        """
        Parse free-text patient description into structured profile.

        Args:
            patient_text: Raw patient case description

        Returns:
            Structured PatientProfile
        """
        prompt = EXTRACTION_PROMPT.format(patient_text=patient_text)

        try:
            profile = await self.medgemma.generate_structured(
                prompt=prompt,
                response_model=PatientProfile,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,  # Low temp for factual extraction
                max_tokens=1500,
            )
            if not profile.chief_complaint or "[LLM unavailable" in profile.chief_complaint or len(profile.chief_complaint.strip()) < 5:
                profile.chief_complaint = patient_text[:200]
                profile.history_of_present_illness = patient_text

            if settings.privacy_mode:
                logger.info("Parsed patient profile (content redacted — privacy mode)")
            else:
                num_meds = len(profile.current_medications or [])
                num_allergies = len(profile.allergies or [])
                cc_preview = (profile.chief_complaint or "")[:40]
                logger.info(
                    f"Parsed patient profile: cc='{cc_preview}…' "
                    f"({num_meds} meds, {num_allergies} allergies)"
                )
            return profile

        except Exception as e:
            # Fallback: If any error occurs (API, parsing, etc.), do basic extraction
            logger.warning(f"Patient parsing failed ({type(e).__name__}: {e}), using basic extraction")
            return PatientProfile(
                chief_complaint=patient_text[:200],
                history_of_present_illness=patient_text,
                additional_notes=f"Auto-extracted from raw text (structured parsing failed: {type(e).__name__})",
            )
