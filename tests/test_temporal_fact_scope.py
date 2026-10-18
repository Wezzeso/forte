"""Generated decisions only: a later absence can coexist with earlier presence."""

from dataclasses import replace

import pytest

from forte.decision import (DecisionConfig, DecisionOutputError, build_prompt,
    compact_frame_request, restore_frame_ids, validate_decisions)
from forte.events import EventBuildConfig, build_events
from forte.shared.schemas import ClipRequest, FactDecision, FrameRef
from rules.catalog import load_catalog


RULE = "RK_NBK231_P22_RECEIPT_DEPARTURE"
EVENT = "CASH_UNATTENDED"
HANDLER = "CASH_HANDLER_PRESENT"


def transition_fixture(compact=False):
    catalog = load_catalog()
    request = ClipRequest(job_id="synthetic-transition", clip_id="synthetic-eight-second-window",
        start_sec=0, end_sec=8,
        frame_refs=[FrameRef(frame_id=f"source-{index}", pts_sec=index,
            path=f"synthetic/frame-{index}.jpg", width=32, height=32) for index in range(9)],
        fact_definitions=catalog.fact_definitions(), rule_definitions=catalog.enabled_rules())
    aliases = compact_frame_request(request) if compact else None
    model_request = aliases.request if aliases else request
    ids = [frame.frame_id for frame in model_request.frame_refs]
    required = catalog.bindings[RULE]["required_fact_states"]
    raw = {"facts": [], "candidates": []}
    for definition in model_request.fact_definitions:
        state = "present" if definition.fact_id == HANDLER else required.get(definition.fact_id, "absent")
        support = ids[:2] if definition.fact_id == HANDLER else [ids[5]]
        raw["facts"].append({"fact_id": definition.fact_id, "state": state,
            "supporting_frame_ids": support, "notes": None})
    raw["candidates"].append({"candidate_id": "supported-later-departure", "event_type_id": EVENT,
        "start_frame_id": ids[5], "end_frame_id": ids[7],
        "supporting_fact_ids": list(required), "supporting_frame_ids": ids[5:8], "visibility_notes": [],
        "local_facts": [{"fact_id": HANDLER, "state": "absent", "supporting_frame_ids": ids[5:8], "notes": None}]})
    return catalog, request, model_request, aliases, raw


def assemble(batch, catalog):
    return build_events([batch], catalog, batch.request.frame_refs,
        {"duration_sec": 8, "complete": True, "gaps": [], "missing_window_indices": []},
        config=EventBuildConfig(event_types=catalog.event_types))


def test_direct_candidate_preserves_later_supported_absence_without_rewriting_batch():
    catalog, _, request, _, raw = transition_fixture()
    local = raw["candidates"][0].pop("local_facts")[0]
    batch = validate_decisions(raw, request, reviewed_event_ids=[EVENT])
    candidate = batch.candidates[0]
    candidate = candidate.model_copy(update={"supporting_facts": [
        FactDecision(**local) if fact.fact_id == HANDLER else fact for fact in candidate.supporting_facts]})
    result = assemble(replace(batch, candidates=(candidate,)), catalog)
    assert result.verdict == "violation"
    assert next(fact for fact in result.events[0].observed_facts if fact.fact_id == HANDLER).state == "absent"
    assert next(fact for fact in batch.facts if fact.fact_id == HANDLER).state == "present"


def test_raw_transition_can_be_expressed_with_explicit_local_source_support():
    catalog, _, request, _, raw = transition_fixture()
    batch = validate_decisions(raw, request, reviewed_event_ids=[EVENT])
    result = assemble(batch, catalog)
    assert result.verdict == "violation"
    assert result.events[0].anchor_frame_id == "source-5"
    assert next(fact for fact in batch.facts if fact.fact_id == HANDLER).supporting_frame_ids == ["source-0", "source-1"]


def test_compact_alias_restore_keeps_opposite_states_in_their_original_scopes():
    catalog, canonical, request, aliases, raw = transition_fixture(compact=True)
    batch = validate_decisions(raw, request, reviewed_event_ids=[EVENT])
    restored = restore_frame_ids(batch, aliases)
    assert restored.request is canonical
    assert next(fact for fact in restored.facts if fact.fact_id == HANDLER).state == "present"
    local = next(fact for fact in restored.candidates[0].supporting_facts if fact.fact_id == HANDLER)
    assert local.state == "absent" and local.supporting_frame_ids == ["source-5", "source-6", "source-7"]
    assert assemble(restored, catalog).verdict == "violation"


@pytest.mark.parametrize("prior_support,local_support,prior_state,local_state", [
    (["source-5"], ["source-5", "source-6"], "present", "absent"),
    (["source-0", "source-6"], ["source-5", "source-7"], "present", "absent"),
    (["source-0", "source-8"], ["source-5", "source-7"], "present", "absent"),
    (["source-7"], ["source-5"], "present", "absent"),
    (["source-0"], ["source-0"], "present", "absent"),
    (["source-0"], ["source-8"], "present", "absent"),
    (["source-0"], [], "present", "absent"),
    ([], ["source-5"], "unclear", "absent"),
    (["source-0"], [], "present", "unclear"),
    (["source-0"], ["source-5"], "present", "present"),
])
def test_raw_local_fact_rejects_overlap_out_of_scope_or_unsupported_state_changes(
        prior_support, local_support, prior_state, local_state):
    _, _, request, _, raw = transition_fixture()
    previous = next(fact for fact in raw["facts"] if fact["fact_id"] == HANDLER)
    previous.update(state=prior_state, supporting_frame_ids=prior_support)
    raw["candidates"][0]["local_facts"][0].update(state=local_state, supporting_frame_ids=local_support)
    with pytest.raises(DecisionOutputError, match="candidate_fact_scope_conflict"):
        validate_decisions(raw, request, reviewed_event_ids=[EVENT])


def test_local_fact_must_be_known_referenced_unique_and_explicitly_cited():
    for mutation in ("unknown_frame", "unreferenced_fact", "duplicate_fact", "uncited_local_frame"):
        _, _, request, _, raw = transition_fixture()
        candidate = raw["candidates"][0]
        local = candidate["local_facts"][0]
        if mutation == "unknown_frame":
            local["supporting_frame_ids"] = ["invented-frame"]
        elif mutation == "unreferenced_fact":
            local["fact_id"] = "CASH_HANDOVER_VISIBLE"
        elif mutation == "duplicate_fact":
            candidate["local_facts"].append(dict(local))
        else:
            candidate["supporting_frame_ids"] = ["source-5", "source-7"]
        with pytest.raises(DecisionOutputError):
            validate_decisions(raw, request, reviewed_event_ids=[EVENT])


def test_legacy_candidate_cannot_infer_absence_from_earlier_presence():
    catalog, _, request, _, raw = transition_fixture()
    raw["candidates"][0].pop("local_facts")
    batch = validate_decisions(raw, request, reviewed_event_ids=[EVENT])
    result = assemble(batch, catalog)
    assert not result.events and result.verdict is None
    assert any(item.reason == "required_visual_facts_not_established" for item in result.refinement_requests)


def test_presence_after_absence_is_explicit_and_keeps_both_batch_observations():
    catalog, _, request, _, raw = transition_fixture()
    cash = next(fact for fact in raw["facts"] if fact["fact_id"] == "CASH_VISIBLE")
    cash.update(state="absent", supporting_frame_ids=["source-0"])
    raw["candidates"][0]["local_facts"].append({"fact_id": "CASH_VISIBLE", "state": "present",
        "supporting_frame_ids": ["source-5"], "notes": None})
    batch = validate_decisions(raw, request, reviewed_event_ids=[EVENT])
    assert next(fact for fact in batch.facts if fact.fact_id == "CASH_VISIBLE").state == "absent"
    assert assemble(batch, catalog).verdict == "violation"


def test_explicit_transition_does_not_replace_unknown_context_or_missing_required_action():
    for fact_id, state in (("WORK_AREA_CONTINUOUSLY_VISIBLE", "unclear"),
                           ("CASH_HANDLER_LEAVES_WORKSTATION", "absent")):
        catalog, _, request, _, raw = transition_fixture()
        fact = next(fact for fact in raw["facts"] if fact["fact_id"] == fact_id)
        fact.update(state=state, supporting_frame_ids=[] if state == "unclear" else ["source-5"])
        batch = validate_decisions(raw, request, reviewed_event_ids=[EVENT])
        result = assemble(batch, catalog)
        assert not result.events and result.verdict is None


@pytest.mark.parametrize("compact", [False, True])
def test_published_schema_and_concise_contract_bind_local_fact_sources(compact):
    from jsonschema import Draft202012Validator
    _, _, request, _, raw = transition_fixture(compact=compact)
    config = DecisionConfig(concise_observations=True)
    bundle = build_prompt(request, reviewed_event_ids=[EVENT], config=config)
    Draft202012Validator(bundle.response_schema).validate(raw)
    assert bundle.prompt_version == "forte.visual-observation.v1.1.0-concise.1"
    assert "local_facts" in bundle.prompt
    raw["candidates"][0]["local_facts"][0]["notes"] = "An unrequested note."
    with pytest.raises(DecisionOutputError, match="concise_fact_notes_not_null"):
        validate_decisions(raw, request, reviewed_event_ids=[EVENT], config=config)


def test_alias_restore_rejects_overlapping_opposite_evidence_even_for_prebuilt_candidate():
    _, _, request, aliases, raw = transition_fixture(compact=True)
    raw["candidates"][0].pop("local_facts")
    batch = validate_decisions(raw, request, reviewed_event_ids=[EVENT])
    candidate = batch.candidates[0]
    changed = [fact.model_copy(update={"state": "absent", "supporting_frame_ids": ["f00"]})
        if fact.fact_id == HANDLER else fact for fact in candidate.supporting_facts]
    with pytest.raises(DecisionOutputError, match="inconsistent_aliased_candidate_facts"):
        restore_frame_ids(replace(batch, candidates=(candidate.model_copy(update={"supporting_facts": changed}),)), aliases)


@pytest.mark.parametrize("support", [["source-5"], ["source-0", "source-6"]])
def test_event_builder_rejects_overlapping_opposite_evidence_even_for_prebuilt_candidate(support):
    catalog, _, request, _, raw = transition_fixture()
    local = raw["candidates"][0].pop("local_facts")[0]
    next(fact for fact in raw["facts"] if fact["fact_id"] == HANDLER)["supporting_frame_ids"] = support
    batch = validate_decisions(raw, request, reviewed_event_ids=[EVENT])
    candidate = batch.candidates[0]
    changed = [FactDecision(**local) if fact.fact_id == HANDLER else fact for fact in candidate.supporting_facts]
    result = assemble(replace(batch, candidates=(candidate.model_copy(update={"supporting_facts": changed}),)), catalog)
    assert not result.events and result.verdict is None
    assert any(item.reason == "invalid_or_conflicting_visual_support" for item in result.refinement_requests)


def test_different_source_ids_at_same_pts_cannot_hide_a_contradiction():
    _, _, request, _, raw = transition_fixture()
    # A duplicate source timestamp still lies inside the candidate interval.
    request.frame_refs[4].pts_sec = 5
    next(fact for fact in raw["facts"] if fact["fact_id"] == HANDLER)["supporting_frame_ids"] = ["source-4"]
    with pytest.raises(DecisionOutputError, match="candidate_fact_scope_conflict"):
        validate_decisions(raw, request, reviewed_event_ids=[EVENT])
