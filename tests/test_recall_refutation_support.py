"""Original positive frames must survive negative resampling for adjudication.

All inputs are explicit synthetic fact states; no media or inference is used.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from forte.decision import DecisionBatch
from forte.events import build_events
from forte.pipeline import _reconcile_batches, _with_pending
from forte.shared.schemas import CandidateEvent, ClipRequest, FactDecision, FrameRef
from rules.catalog import load_catalog


CATALOG = load_catalog()
RULE_IDS = ("RK_NBK231_P22_RECEIPT_DEPARTURE", "RK_NBK231_P20_PREVIOUS_CASH_POCKET")


def make_batch(name, points, end, *, rule_id=None, propose=True, candidate_support=5):
    refs = [FrameRef(frame_id=f"synthetic-{point}", pts_sec=float(point),
                     path=f"synthetic/frame-{point}.jpg", width=32, height=32,
                     job_id="synthetic-recall-support") for point in points]
    states = {fact.fact_id: "absent" for fact in CATALOG.fact_definitions()}
    action_ids = set()
    if rule_id is not None:
        binding = CATALOG.bindings[rule_id]
        states.update(binding["required_fact_states"])
        action_ids = set(binding["violation_fact_ids"])
    facts = tuple(FactDecision(fact_id=key, state=state, supporting_frame_ids=(
        ["synthetic-5"] if key in action_ids else [frame.frame_id for frame in refs]
    )) for key, state in states.items())
    request = ClipRequest(job_id="synthetic-recall-support", clip_id=name, frame_refs=refs,
                          start_sec=0.0, end_sec=float(end),
                          fact_definitions=CATALOG.fact_definitions(trusted_scene_context="bank_cash_workstation"))
    candidates = ()
    if rule_id is not None and propose:
        candidates = (CandidateEvent(candidate_id="synthetic-action", event_type_id=binding["event_type_id"],
            start_sec=4.0, end_sec=6.0, supporting_facts=list(facts),
            supporting_frame_ids=[f"synthetic-{candidate_support}"], clip_id=name),)
    return DecisionBatch(request=request, facts=facts, candidates=candidates)


@pytest.mark.parametrize("rule_id", RULE_IDS)
@pytest.mark.parametrize("candidate_support", [4, 5])
def test_containing_negative_that_misses_positive_action_cannot_erase_candidate(rule_id, candidate_support):
    coarse = make_batch("coarse", range(9), 8, rule_id=rule_id, candidate_support=candidate_support)
    refined = make_batch("wide-resampled", [0, 2, 4, 6, 8, 10, 12], 12)
    final, pending, requests = _reconcile_batches([coarse], [[refined]], CATALOG)
    assert any(value.reason == "unresolved_coarse_candidate" for value in pending)
    frames = {frame.frame_id: frame for batch in (coarse, refined) for frame in batch.request.frame_refs}
    built = build_events(final, CATALOG, frames, {
        "duration_sec": 12.0, "complete": True, "gaps": [], "missing_window_indices": [],
    })
    built = _with_pending(built, pending, requests)
    assert not built.can_publish and built.verdict is None


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_negative_reinspecting_original_action_may_still_refute_it(rule_id):
    coarse = make_batch("coarse", range(9), 8, rule_id=rule_id)
    refined = make_batch("wide-reinspected", [0, 2, 4, 5, 6, 8, 10, 12], 12)
    final, pending, requests = _reconcile_batches([coarse], [[refined]], CATALOG)
    assert not pending and not requests
    assert not any(batch.candidates for batch in final)


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_positive_rule_facts_without_proposal_also_need_reinspection(rule_id):
    coarse = make_batch("coarse-facts-only", range(9), 8, rule_id=rule_id, propose=False)
    refined = make_batch("wide-resampled", [0, 2, 4, 6, 8, 10, 12], 12)
    _, pending, requests = _reconcile_batches([coarse], [[refined]], CATALOG)
    assert any(value.reason == "rule_facts_without_candidate" for value in pending)
    assert requests


def test_reinspection_of_one_candidate_cannot_clear_another_candidates_missing_support():
    coarse = make_batch("coarse", range(9), 8, rule_id=RULE_IDS[0])
    first = coarse.candidates[0]
    other_facts = [fact.model_copy(update={"supporting_frame_ids": ["synthetic-7"]})
                   if fact.fact_id == "CASH_HANDLER_LEAVES_WORKSTATION" else fact
                   for fact in first.supporting_facts]
    second = first.model_copy(update={"candidate_id": "synthetic-later-action", "start_sec": 7.0,
                                      "end_sec": 7.0, "supporting_frame_ids": ["synthetic-7"],
                                      "supporting_facts": other_facts})
    coarse = replace(coarse, candidates=(first, second))
    refined = make_batch("wide-first-only", [0, 2, 4, 5, 6, 8, 10, 12], 12)
    _, pending, _ = _reconcile_batches([coarse], [[refined]], CATALOG)
    assert len(pending) == 1
    assert pending[0].candidate_id == "synthetic-later-action"

