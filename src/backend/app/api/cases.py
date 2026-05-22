"""
REST API for case submission and result retrieval.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException

from app.agent.orchestrator import Orchestrator
from app.config import settings
from app.models.schemas import (
    AgentState,
    CaseResponse,
    CaseResult,
    CaseSubmission,
)

router = APIRouter()

# In-memory store for active/completed cases
# In production, use Redis or a database
_cases: Dict[str, Orchestrator] = {}
_case_timestamps: Dict[str, float] = {}


def _evict_expired_cases() -> None:
    """Remove cases older than TTL (only enforced when privacy_mode=True)."""
    if not settings.privacy_mode:
        return
    now = time.time()
    ttl = settings.case_ttl_seconds
    expired = [cid for cid, ts in _case_timestamps.items() if now - ts > ttl]
    for cid in expired:
        _cases.pop(cid, None)
        _case_timestamps.pop(cid, None)


@router.post("/submit", response_model=CaseResponse)
async def submit_case(case: CaseSubmission):
    """
    Submit a patient case for analysis.

    The agent pipeline runs asynchronously. Use the WebSocket endpoint
    or poll /api/cases/{case_id} for real-time updates.
    """
    orchestrator = Orchestrator()

    # Generate a case_id upfront so we can return it immediately
    case_id = str(uuid.uuid4())[:8]

    async def _run_pipeline():
        async for _step in orchestrator.run(case):
            pass  # Steps are tracked in orchestrator state
        # Once run() creates state, store the orchestrator under the real case_id
        if orchestrator.state:
            _cases[orchestrator.state.case_id] = orchestrator

    asyncio.create_task(_run_pipeline())

    # Wait briefly for the orchestrator to initialise its state
    await asyncio.sleep(0.15)

    # Use the orchestrator's actual case_id if available, otherwise the pre-generated one
    actual_id = orchestrator.state.case_id if orchestrator.state else case_id
    _cases[actual_id] = orchestrator
    _case_timestamps[actual_id] = time.time()
    _evict_expired_cases()

    return CaseResponse(
        case_id=actual_id,
        status="running",
        message="Agent pipeline started. Connect to WebSocket for real-time updates.",
    )


@router.get("/{case_id}", response_model=CaseResult)
async def get_case(case_id: str):
    """Get the current state and results for a case."""
    if settings.privacy_mode:
        raise HTTPException(status_code=403, detail="Case retrieval is disabled in privacy mode")

    _evict_expired_cases()
    orchestrator = _cases.get(case_id)
    if not orchestrator or not orchestrator.state:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    return CaseResult(
        case_id=case_id,
        state=orchestrator.state,
        report=orchestrator.get_result(),
    )


@router.get("/", response_model=list[str])
async def list_cases():
    """List all case IDs."""
    if settings.privacy_mode:
        raise HTTPException(status_code=403, detail="Case listing is disabled in privacy mode")
    _evict_expired_cases()
    return list(_cases.keys())
