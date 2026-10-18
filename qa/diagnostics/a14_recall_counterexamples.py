"""Explicit synthetic recall counterexamples; no model or private data access.

Run as a module for aggregate diagnostics. The output describes synthetic
behavior and exposes risks; it is not evidence of real-video false negatives.
"""

from __future__ import annotations

import json

from forte.decision import DecisionBatch
from forte.events import analyze_rule_relevance, build_events
from forte.pipeline import SCREENING_FACT_IDS, _reconcile_batches, _with_pending
from forte.shared.schemas import CandidateEvent, ClipRequest, FactDecision, FrameRef
from rules.catalog import load_catalog


CATALOG = load_catalog()
FACTS = CATALOG.fact_definitions(trusted_scene_context="bank_cash_workstation")
RULE_ID = "RK_NBK231_P22_RECEIPT_DEPARTURE"


def frames(points):
    return [FrameRef(frame_id=f"synthetic-{point}", pts_sec=float(point),
                     path=f"synthetic/frame-{point}.jpg", width=32, height=32,
                     job_id="synthetic-recall") for point in points]


def batch(name, points, end, *, positive=False, screening=False):
    refs = frames(points)
    requested = [value for value in FACTS if not screening or value.fact_id in SCREENING_FACT_IDS]
    states = {value.fact_id: "absent" for value in requested}
    if screening:
        states["CASH_VISIBLE"] = "present"
    if positive:
        states.update(CATALOG.bindings[RULE_ID]["required_fact_states"])
    decisions = tuple(FactDecision(fact_id=key, state=state,
                                  supporting_frame_ids=[frame.frame_id for frame in refs])
                      for key, state in states.items())
    request = ClipRequest(job_id="synthetic-recall", clip_id=name, frame_refs=refs,
                          start_sec=0.0, end_sec=float(end), fact_definitions=requested)
    candidates = ()
    if positive:
        candidates = (CandidateEvent(candidate_id="synthetic-action", event_type_id="CASH_UNATTENDED",
            start_sec=5.0, end_sec=5.0, supporting_facts=list(decisions),
            supporting_frame_ids=["synthetic-5"], clip_id=name),)
    return DecisionBatch(request=request, facts=decisions, candidates=candidates)


def diagnose():
    screen = batch("synthetic-screen", [0, 2, 4, 6, 8], 8, screening=True)
    screen_relevance = analyze_rule_relevance(screen, CATALOG)
    coarse = batch("synthetic-coarse", list(range(9)), 8, positive=True)
    refined = batch("synthetic-refined", [0, 2, 4, 6, 8, 10, 12], 12)
    final, pending, requests = _reconcile_batches([coarse], [[refined]], CATALOG)
    frame_map = {frame.frame_id: frame for value in (coarse, refined) for frame in value.request.frame_refs}
    result = build_events(final, CATALOG, frame_map, {
        "duration_sec": 12.0, "complete": True, "gaps": [], "missing_window_indices": [],
    })
    result = _with_pending(result, pending, requests)
    original_support = set(coarse.candidates[0].supporting_frame_ids)
    new_refs = {frame.frame_id for frame in refined.request.frame_refs}
    return {
        "schema_version": "forte.synthetic-recall-counterexamples.v1",
        "private_media_or_model_used": False,
        "screen_requested_fact_count": len(screen.facts),
        "screen_requested_visibility_or_procedure_context": any(
            fact.fact_id in {"WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE", "ALL_COUNTING_PATHS_VISIBLE"}
            for fact in screen.facts),
        "screen_refuted_rule_count": len(screen_relevance.refuted_rule_ids),
        "screen_selects_full_primary": not screen_relevance.all_branches_refuted,
        "coarse_candidate_count": len(coarse.candidates),
        "refinement_reinspects_original_candidate_support": original_support <= new_refs,
        "remaining_candidate_count": sum(len(value.candidates) for value in final),
        "pending_interval_count": len(pending),
        "pending_refinement_request_count": len(requests),
        "reconciled_event_count": len(result.events),
        "reconciled_verdict": result.verdict,
    }


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2))

