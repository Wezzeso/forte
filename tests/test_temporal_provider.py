"""Pure synthetic optional-provider tests; no model/torch/GPU/media execution."""
import asyncio
from types import SimpleNamespace
import time

import pytest

from forte.decision.head_contract import SourceFrame, TemporalHeadOutput
from forte.decision.temporal_backend import bind_feature_indices, native_window_coverage
from forte.decision.temporal_provider import (
    DECODER_SHA256, DiscriminativeDecisionProvider, HeadBackendResult,
    FACT_RUN_DECODER_SHA256, PhysicalFactRunPolicy, PhysicalFactRunRule,
    TemporalProviderError, decode_temporal_brackets, map_temporal_output,
)
from forte.decision.structured import validate_decisions
from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef
from test_machine_calibration_contract import manifest, SyntheticVerifier, digest


def fixture(count=5):
    calibration = manifest().model_copy(update={"decoder_sha256": DECODER_SHA256})
    identity = calibration.identity
    request = ClipRequest(job_id="synthetic-job", clip_id="synthetic-clip",
        frame_refs=[FrameRef(frame_id=digest(f"synthetic-frame-{index}"), pts_sec=float(index),
            source_pts=index, path=f"/synthetic-only/{index}.jpg", width=8, height=8)
            for index in range(count)],
        fact_definitions=[FactDefinition(fact_id=fact_id, description="Synthetic unit fixture")
                          for fact_id in identity.fact_ids], start_sec=0.0, end_sec=float(count))
    output = TemporalHeadOutput(identity=identity, job_id=request.job_id, clip_id=request.clip_id,
        frames=tuple(SourceFrame(frame_id=frame.frame_id, pts_sec=frame.pts_sec,
            source_pts=index, next_source_pts=index + 1 if index < count - 1 else None)
            for index, frame in enumerate(request.frame_refs)),
        window_logits=((0.0, 4.0, -4.0),) * 19,
        frame_logits=(((0.0, 4.0, -4.0),) * 19,) * count,
        support_logits=((4.0,) * 19,) * count,
        frame_valid=(True,) * count, global_view_available=(True,) * count,
        feature_coverage_complete=True, native_source_continuity_verified=True,
        trained_components={"window_fact_states": True, "frame_fact_states": True,
                            "frame_support": True, "event_boundaries": True}, boundary_logits=None)
    return request, output, calibration


def mapping(output, request, calibration, *, definitions=None, contexts=None, policy=None):
    return map_temporal_output(output, request, calibration, verifier=SyntheticVerifier(),
        event_definitions=definitions or {}, absence_context_fact_ids=contexts or {},
        physical_fact_run_policy=policy)


def test_pinned_nineteen_fact_outputs_cross_existing_contract_with_canonical_refs():
    request, output, calibration = fixture()
    raw = mapping(output, request, calibration)
    assert len(raw["facts"]) == 19 and raw["candidates"] == []
    assert all(fact["state"] == "present" and fact["notes"] is None for fact in raw["facts"])
    batch = validate_decisions(raw, request, reviewed_event_ids=())
    assert len(batch.facts) == 19
    assert all(set(fact.supporting_frame_ids) <= {frame.frame_id for frame in request.frame_refs}
               for fact in batch.facts)


def test_high_window_and_support_scores_cannot_override_unclear_frame_states():
    request, output, calibration = fixture()
    uncertain = output.model_copy(update={"frame_logits": (((0.0, -4.0, 4.0),) * 19,) * 5})
    assert all(fact["state"] == "unclear" and not fact["supporting_frame_ids"]
               for fact in mapping(uncertain, request, calibration)["facts"])


def test_unclear_window_state_and_missing_global_view_never_become_present():
    request, output, calibration = fixture()
    for update in ({"window_logits": ((0.0, -4.0, 4.0),) * 19},
                   {"global_view_available": (False,) * 5}, {"frame_valid": (False,) * 5}):
        raw = mapping(output.model_copy(update=update), request, calibration)
        assert all(fact["state"] == "unclear" for fact in raw["facts"])


@pytest.mark.parametrize("component", ["window_fact_states", "frame_fact_states", "frame_support"])
def test_untrained_head_branch_abstains_even_if_random_scores_are_high(component):
    request, output, calibration = fixture()
    components = {**output.trained_components, component: False}
    raw = mapping(output.model_copy(update={"trained_components": components}), request, calibration)
    assert all(fact["state"] == "unclear" for fact in raw["facts"])


def test_absence_needs_independent_present_context_and_native_feature_continuity():
    request, output, calibration = fixture()
    states = list(output.window_logits)
    states[0] = (4.0, 0.0, -4.0)
    frame_states = [list(frame) for frame in output.frame_logits]
    for frame in frame_states:
        frame[0] = states[0]
    output = output.model_copy(update={"window_logits": tuple(states),
        "frame_logits": tuple(tuple(frame) for frame in frame_states)})
    fact_id, context = output.identity.fact_ids[:2]
    assert mapping(output, request, calibration)["facts"][0]["state"] == "unclear"
    contexts = {fact_id: (context,)}
    assert mapping(output, request, calibration, contexts=contexts)["facts"][0]["state"] == "absent"
    incomplete = output.model_copy(update={"feature_coverage_complete": False})
    assert mapping(incomplete, request, calibration, contexts=contexts)["facts"][0]["state"] == "unclear"
    sparse = output.model_copy(update={"native_source_continuity_verified": False})
    assert mapping(sparse, request, calibration, contexts=contexts)["facts"][0]["state"] == "unclear"


def test_unseen_class_remains_unclear_while_other_classes_progress():
    request, output, calibration = fixture()
    records = dict(calibration.fact_calibration)
    first = output.identity.fact_ids[0]
    records[first] = records[first].model_copy(update={"present": None, "absent": None})
    raw = mapping(output, request, calibration.model_copy(update={"fact_calibration": records}))
    assert raw["facts"][0]["state"] == "unclear"
    assert raw["facts"][1]["state"] == "present"


def test_bank_scope_requires_explicit_trusted_operator_context_not_a_model_logit():
    request, output, calibration = fixture()
    old_id = output.identity.fact_ids[0]
    identity = output.identity.model_copy(update={"fact_ids": ("BANK_CASH_WORKSTATION", *output.identity.fact_ids[1:])})
    records = dict(calibration.fact_calibration)
    records["BANK_CASH_WORKSTATION"] = records.pop(old_id)
    calibration = calibration.model_copy(update={"identity": identity, "fact_calibration": records})
    output = output.model_copy(update={"identity": identity})
    request = request.model_copy(deep=True)
    request.fact_definitions[0] = request.fact_definitions[0].model_copy(update={"fact_id": "BANK_CASH_WORKSTATION"})
    raw = mapping(output, request, calibration)
    assert raw["facts"][0]["state"] == "unclear"
    raw = map_temporal_output(output, request, calibration, verifier=SyntheticVerifier(),
        event_definitions={}, absence_context_fact_ids={}, trusted_scene_context="bank_cash_workstation")
    assert raw["facts"][0]["state"] == "present"


def with_boundaries(output, starts, ends):
    return output.model_copy(update={"boundary_logits": tuple(
        ((4.0 if index in starts else -4.0, 4.0 if index in ends else -4.0),)
        for index in range(len(output.frames)))})


def test_distinct_repeated_source_occurrences_remain_distinct_and_bound_to_pts():
    request, output, calibration = fixture(7)
    output = with_boundaries(output, {0, 4}, {2, 6})
    definitions = {"SYNTHETIC_EVENT": {"required_fact_states": {output.identity.fact_ids[0]: "present"}}}
    raw = mapping(output, request, calibration, definitions=definitions)
    assert len(raw["candidates"]) == 2
    batch = validate_decisions(raw, request, reviewed_event_ids=tuple(definitions))
    assert [(candidate.start_sec, candidate.end_sec) for candidate in batch.candidates] == [(0.0, 2.0), (4.0, 6.0)]


def test_trained_calibrated_boundaries_own_axis_without_broad_third_fact_run():
    request, output, calibration = fixture(7)
    output = with_boundaries(output, {0, 4}, {2, 6})
    definitions = {"SYNTHETIC_EVENT": {"required_fact_states": {output.identity.fact_ids[0]: "present"}}}
    raw = mapping(output, request, calibration, definitions=definitions, policy=physical_policy(output))
    assert len(raw["candidates"]) == 2


@pytest.mark.parametrize("starts,ends", [({1}, set()), (set(), {2}), ({0, 2}, {4})])
def test_missing_or_overlapping_boundaries_fail_without_fabricated_endpoint(starts, ends):
    _, output, calibration = fixture()
    with pytest.raises(TemporalProviderError):
        decode_temporal_brackets(with_boundaries(output, starts, ends), calibration, ("SYNTHETIC_EVENT",))


def test_contiguous_boundary_run_uses_earliest_equal_peak_and_point_pair_is_allowed():
    _, output, calibration = fixture()
    result = decode_temporal_brackets(with_boundaries(output, {1, 2}, {1, 2}), calibration, ("SYNTHETIC_EVENT",))
    assert result == {"SYNTHETIC_EVENT": [(1, 1)]}


def test_each_required_fact_needs_source_support_inside_that_occurrence():
    request, output, calibration = fixture()
    output = with_boundaries(output, {2}, {4})
    rows = [list(row) for row in output.support_logits]
    for index in (2, 3, 4):
        rows[index][1] = -4.0
    output = output.model_copy(update={"support_logits": tuple(tuple(row) for row in rows)})
    definitions = {"SYNTHETIC_EVENT": {"required_fact_states": {
        output.identity.fact_ids[0]: "present", output.identity.fact_ids[1]: "present"}}}
    with pytest.raises(TemporalProviderError) as error:
        mapping(output, request, calibration, definitions=definitions)
    assert error.value.subcode == "localized_event_fact_support_outside_bracket"


def test_experimental_unknown_axis_fails_instead_of_mapping_to_nearest_baseline_fact():
    request, output, calibration = fixture()
    definitions = {"SYNTHETIC_EVENT": {"required_fact_states": {"EXP_UNKNOWN": "present"}}}
    with pytest.raises(TemporalProviderError) as error:
        mapping(output, request, calibration, definitions=definitions)
    assert error.value.subcode == "unsupported_event_fact_axis"


def physical_policy(output):
    return PhysicalFactRunPolicy(taxonomy_sha256=output.identity.taxonomy_sha256,
        rule_catalog_sha256=digest("synthetic-catalog"), decoder_sha256=FACT_RUN_DECODER_SHA256,
        rules=(PhysicalFactRunRule(rule_id="synthetic-rule", event_type_id="SYNTHETIC_EVENT",
            required_fact_states={output.identity.fact_ids[0]: "present",
                                  output.identity.fact_ids[1]: "present"}),))


def test_fact_run_proposals_work_without_any_trained_typed_boundary_class():
    request, output, calibration = fixture()
    calibration = calibration.model_copy(update={"event_calibration": {}, "decoder_sha256": None})
    policy = physical_policy(output)
    definitions = {"SYNTHETIC_EVENT": {"required_fact_states": {output.identity.fact_ids[0]: "present"}}}
    rows = [list(row) for row in output.support_logits]
    rows[2][0] = -4.0  # Unclear source gap must separate repeated occurrences.
    output = output.model_copy(update={"support_logits": tuple(tuple(row) for row in rows)})
    raw = mapping(output, request, calibration, definitions=definitions, policy=policy)
    batch = validate_decisions(raw, request, reviewed_event_ids=tuple(definitions))
    assert [(event.start_sec, event.end_sec) for event in batch.candidates] == [(0.0, 1.0), (3.0, 4.0)]
    assert all(len(event.supporting_facts) == 2 for event in batch.candidates)


def test_sparse_delivery_never_establishes_native_fact_run_continuity():
    request, output, calibration = fixture()
    calibration = calibration.model_copy(update={"event_calibration": {}, "decoder_sha256": None})
    definitions = {"SYNTHETIC_EVENT": {"required_fact_states": {output.identity.fact_ids[0]: "present"}}}
    policy = physical_policy(output)
    changed_frames = list(output.frames)
    changed_frames[0] = changed_frames[0].model_copy(update={"next_source_pts": 8})
    for update in ({"native_source_continuity_verified": False}, {"frames": tuple(changed_frames)}):
        raw = mapping(output.model_copy(update=update), request, calibration,
                      definitions=definitions, policy=policy)
        assert raw["candidates"] == []


def test_untrained_context_class_blocks_fact_run_without_fabricating_rule_context():
    request, output, calibration = fixture()
    calibration = calibration.model_copy(update={"event_calibration": {}, "decoder_sha256": None})
    records = dict(calibration.fact_calibration)
    context = output.identity.fact_ids[1]
    records[context] = records[context].model_copy(update={"present": None, "absent": None})
    calibration = calibration.model_copy(update={"fact_calibration": records})
    definitions = {"SYNTHETIC_EVENT": {"required_fact_states": {output.identity.fact_ids[0]: "present"}}}
    assert mapping(output, request, calibration, definitions=definitions,
                   policy=physical_policy(output))["candidates"] == []


def test_cache_index_binding_rejects_aliases_changed_pts_and_reordered_frames():
    request, _, _ = fixture()
    frames = tuple(SimpleNamespace(frame_id=frame.frame_id, pts_sec=frame.pts_sec,
        source_pts=frame.source_pts, next_source_pts=index + 1 if index < 4 else None,
        next_pts_sec=float(index + 1) if index < 4 else None, source_end_bound_sec=5.0 if index == 4 else None)
        for index, frame in enumerate(request.frame_refs))
    sequence = SimpleNamespace(frames=frames)
    assert bind_feature_indices(sequence, request) == tuple(range(5))
    assert native_window_coverage(sequence, request, tuple(range(5)))
    assert not native_window_coverage(sequence, request, (0, 2, 4))
    for field, value in (("frame_id", "f00"), ("pts_sec", 0.5), ("source_pts", 100)):
        changed = request.model_copy(deep=True)
        changed.frame_refs[0] = changed.frame_refs[0].model_copy(update={field: value})
        with pytest.raises(TemporalProviderError):
            bind_feature_indices(sequence, changed)
    reversed_request = request.model_copy(update={"frame_refs": list(reversed(request.frame_refs))})
    with pytest.raises(TemporalProviderError):
        bind_feature_indices(sequence, reversed_request)


class SyntheticBackend:
    """Explicit protocol test fixture; production has no fake fallback."""
    def __init__(self, output):
        self.output, self.identity, self.calls = output, output.identity, 0

    async def check_ready(self, *, deadline):
        return {"ready": True}

    async def predict(self, request, *, deadline, cancel_event=None):
        self.calls += 1
        return HeadBackendResult(self.output, {"input_frames": len(request.frame_refs),
            "elapsed_sec": 0.1, "notes": "SYNTHETIC_PRIVATE_SENTINEL", "prompt_tokens": float("nan")})


@pytest.mark.asyncio
async def test_optional_provider_runs_protocol_and_strips_non_numeric_unapproved_metrics():
    request, output, calibration = fixture()
    backend = SyntheticBackend(output)
    provider = DiscriminativeDecisionProvider(backend, calibration, verifier=SyntheticVerifier(),
        event_definitions={}, absence_context_fact_ids={})
    response = await provider.infer(request, prompt="untrusted synthetic prompt ignored by head",
        response_schema={}, deadline=time.monotonic() + 2)
    assert backend.calls == 1 and response.finish_reason == "stop"
    assert "notes" not in response.metrics and "prompt_tokens" not in response.metrics
    assert response.metrics["serving_attestation"]["provisional"] is True
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in repr(response)


@pytest.mark.asyncio
async def test_cancellation_and_unverified_calibration_do_not_run_backend():
    request, output, calibration = fixture()
    backend = SyntheticBackend(output)
    provider = DiscriminativeDecisionProvider(backend, calibration,
        verifier=SyntheticVerifier(threshold_selection_verified=False),
        event_definitions={}, absence_context_fact_ids={})
    with pytest.raises(TemporalProviderError):
        await provider.infer(request, prompt="", response_schema={}, deadline=time.monotonic() + 2)
    cancelled = asyncio.Event()
    cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        await provider.infer(request, prompt="", response_schema={}, deadline=time.monotonic() + 2,
            cancel_event=cancelled)
    assert backend.calls == 0
