"""Generated proof that internal scope supports are not public JPEG limits.

No media, private observations, model invocation, grammar compilation or GPU.
"""

import hashlib

import jsonschema
import pytest

from forte.decision import DecisionConfig, build_prompt, compact_frame_request, restore_frame_ids, validate_decisions
from forte.events import analyze_rule_relevance, build_events
from forte.shared.schemas import ClipRequest, FrameRef, VerifiedEvent
from rules.catalog import load_catalog
from serving.qwen38_grammar import checked_schema, trusted_fact_order_schema


CATALOG = load_catalog()
EVENT_IDS = tuple(sorted({binding["event_type_id"] for binding in CATALOG.bindings.values()}))


def request():
    frames = [FrameRef(frame_id=hashlib.sha256(f"generated-support-{index}".encode()).hexdigest(),
                      pts_sec=float(index), path=f"synthetic/support-{index}.jpg", width=32, height=32,
                      job_id="generated-support-contract") for index in range(16)]
    return ClipRequest(job_id="generated-support-contract", clip_id="generated-full-window",
                       start_sec=0.0, end_sec=15.0, frame_refs=frames,
                       fact_definitions=CATALOG.fact_definitions(trusted_scene_context="bank_cash_workstation"),
                       rule_definitions=CATALOG.enabled_rules())


def payload(clip, *, support_count, positive=False):
    states = {definition.fact_id: "absent" for definition in clip.fact_definitions}
    binding = CATALOG.bindings["RK_NBK231_P22_RECEIPT_DEPARTURE"]
    if positive:
        states.update(binding["required_fact_states"])
    support = [frame.frame_id for frame in clip.frame_refs[:support_count]]
    result = {"facts": [{"fact_id": definition.fact_id, "state": states[definition.fact_id],
                          "supporting_frame_ids": support, "notes": None}
                        for definition in clip.fact_definitions], "candidates": []}
    if positive:
        result["candidates"].append({"candidate_id": "generated-local-action", "event_type_id": "CASH_UNATTENDED",
            "start_frame_id": support[0], "end_frame_id": support[-1],
            "supporting_frame_ids": support, "supporting_fact_ids": list(binding["required_fact_states"]),
            "visibility_notes": []})
    return result


@pytest.mark.parametrize("concise", [False, True])
def test_sixteen_support_ids_pass_original_and_derived_grammar_schema_and_alias_restoration(concise):
    original = request()
    aliases = compact_frame_request(original)
    config = DecisionConfig(compact_frame_ids=True, concise_observations=concise)
    bundle = build_prompt(aliases.request, reviewed_event_ids=EVENT_IDS, config=config)
    support_schema = bundle.response_schema["$defs"]["_RawFact"]["properties"]["supporting_frame_ids"]
    assert support_schema["maxItems"] == 96
    raw = payload(aliases.request, support_count=16)
    jsonschema.Draft202012Validator(bundle.response_schema).validate(raw)
    derived, count = trusted_fact_order_schema(bundle.response_schema)
    assert count == 19
    for field in derived["properties"]["facts"]["prefixItems"]:
        assert field["properties"]["supporting_frame_ids"]["maxItems"] == 96
    jsonschema.Draft202012Validator(derived).validate(raw)
    assert checked_schema(bundle.response_schema)
    batch = validate_decisions(raw, aliases.request, reviewed_event_ids=EVENT_IDS, config=config)
    restored = restore_frame_ids(batch, aliases)
    assert all(len(fact.supporting_frame_ids) == 16 for fact in restored.facts)
    assert all(set(fact.supporting_frame_ids) == {frame.frame_id for frame in original.frame_refs}
               for fact in restored.facts)
    assert analyze_rule_relevance(restored, CATALOG).all_branches_refuted
    built = build_events([restored], CATALOG, original.frame_refs, {
        "duration_sec": 15.0, "complete": True, "gaps": [], "missing_window_indices": []})
    assert built.verdict == "no_violation"


@pytest.mark.parametrize("support_count", [1, 3, 4, 8])
def test_representative_point_support_does_not_assert_whole_window_refutation(support_count):
    clip = request()
    raw = payload(clip, support_count=support_count)
    batch = validate_decisions(raw, clip, reviewed_event_ids=EVENT_IDS)
    # Structurally representable, but smaller scope is intentionally insufficient
    # to establish the opposite state over all sixteen inspected sample frames.
    assert all(len(fact.supporting_frame_ids) == support_count for fact in batch.facts)
    assert not analyze_rule_relevance(batch, CATALOG).all_branches_refuted


@pytest.mark.parametrize("support_count", [1, 16])
def test_positive_action_need_not_cover_all_sixteen_frames_and_public_evidence_remains_three(support_count):
    clip = request()
    batch = validate_decisions(payload(clip, support_count=support_count, positive=True), clip,
                               reviewed_event_ids=EVENT_IDS)
    built = build_events([batch], CATALOG, clip.frame_refs, {
        "duration_sec": 15.0, "complete": True, "gaps": [], "missing_window_indices": []})
    assert built.verdict == "violation" and len(built.events) == 1
    event = built.events[0]
    assert 1 <= len(event.evidence_frame_ids) <= 3
    assert all(len(fact.supporting_frame_ids) == support_count for fact in event.observed_facts)
    # Internal 16-ID support is not copied into a public 16-JPEG finding.
    assert VerifiedEvent.model_json_schema()["properties"]["evidence_frame_ids"]["maxItems"] == 3


def test_supported_unclear_fact_is_valid_json_but_not_a_decisive_scope_assertion():
    clip = request()
    raw = payload(clip, support_count=16)
    for fact in raw["facts"]:
        fact["state"] = "unclear"
    batch = validate_decisions(raw, clip, reviewed_event_ids=EVENT_IDS)
    assert batch.complete and batch.has_uncertainty and batch.disposition == "unclear"
    assert analyze_rule_relevance(batch, CATALOG).requires_refinement
    assert not analyze_rule_relevance(batch, CATALOG).all_branches_refuted

