"""Full E2E test: submit case, poll until done, print report."""
import httpx
import asyncio
import json
import time

API = "http://localhost:8002"

async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        # Submit case
        body = {
            "patient_text": (
                "55-year-old male presenting with chest pain radiating to left arm "
                "for the past 2 hours. History of hypertension and type 2 diabetes. "
                "Current medications: metformin 1000mg BID, lisinopril 20mg daily, "
                "aspirin 81mg daily. Vitals: BP 160/95, HR 88, SpO2 97%."
            ),
            "include_drug_check": True,
            "include_guidelines": True,
        }
        r = await client.post(f"{API}/api/cases/submit", json=body)
        data = r.json()
        case_id = data["case_id"]
        print(f"Submitted case: {case_id}")

        # Poll until done
        steps: list = []
        result: dict = {}
        for i in range(60):  # up to 5 minutes
            await asyncio.sleep(5)
            r = await client.get(f"{API}/api/cases/{case_id}")
            result = r.json()
            state = result.get("state", {})
            steps = state.get("steps", [])

            # Print status
            statuses = [f"{s['step_id']}={s['status']}" for s in steps]
            print(f"  [{i*5}s] {', '.join(statuses)}")

            # Check if all done
            all_done = all(
                s["status"] in ("completed", "failed") for s in steps
            )
            if all_done:
                break

        # Print step details
        print("\n=== Step Results ===")
        for s in steps:
            err = s.get("error", "")
            dur = s.get("duration_ms", "?")
            print(f"  {s['step_id']:12s} {s['status']:10s} ({dur}ms) {err[:100] if err else 'OK'}")

        # --- Assertions ---
        # Verify all 6 pipeline steps are present
        step_ids = [s["step_id"] for s in steps]
        expected_steps = [
            "parse",
            "reason",
            "drugs",
            "guidelines",
            "conflicts",
            "synthesize",
        ]
        assert len(steps) == 6, f"Expected 6 steps, got {len(steps)}: {step_ids}"
        for exp in expected_steps:
            assert exp in step_ids, f"Missing expected step: {exp}"

        # Verify all steps completed (not failed)
        failed = [s["step_id"] for s in steps if s["status"] == "failed"]
        assert not failed, f"Steps failed: {failed}"

        completed = [s["step_id"] for s in steps if s["status"] == "completed"]
        print(f"\n✓ All {len(completed)}/6 pipeline steps completed successfully.")

        # Print report
        report = result.get("report")
        if report:
            print("\n=== CDS Report ===")
            print(json.dumps(report, indent=2, default=str)[:4000])
        else:
            print("\nNo report generated.")

asyncio.run(main())
