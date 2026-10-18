"""Synthetic metadata-only grounding contracts; no models/GPU/media are read."""
from dataclasses import replace
from hashlib import sha256
import json

import pytest
from pydantic import ValidationError

from forte.shared.schemas import VideoAsset
from forte.video.batches import SourceFrameIdentity, source_frame_id
from training.grounding.contracts import (
    GroundingContractError, GroundingRawResult, PhysicalQuery, ProcessorClockProof,
    bind_intervals, expected_sampled_frame_indices, parse_intervals, query_set_sha256,
    validate_processor_clock, validate_queries, validate_source_ledger,
)


def digest(text):
    return sha256(text.encode()).hexdigest()


def fixture(*, vfr=False):
    offsets = (0, 30, 80, 120, 210) if vfr else tuple(range(0, 1040, 40))
    duration = 0.25 if vfr else 1.04
    asset = VideoAsset(job_id="synthetic-grounding", owned_path="/synthetic-only/source.mov",
        content_hash=digest("synthetic-vfr" if vfr else "synthetic-cfr"), duration_sec=duration,
        width=48, height=24, time_base="1/1000", source_start_sec=5.0, frame_count=len(offsets))
    frames = tuple(SourceFrameIdentity(frame_id=source_frame_id(asset.content_hash, 5000 + offset, 1, 1000),
        content_hash=asset.content_hash, job_id=asset.job_id, stream_index=0,
        source_pts=5000 + offset, time_base="1/1000", time_base_num=1, time_base_den=1000,
        source_origin_pts=5000, pts_sec=offset / 1000,
        next_source_pts=5000 + offsets[index + 1] if index + 1 < len(offsets) else None,
        next_pts_sec=offsets[index + 1] / 1000 if index + 1 < len(offsets) else None,
        source_end_bound_sec=duration if index + 1 == len(offsets) else None,
        width=48, height=24, applied_rotation_deg=0, key_frame=index == 0)
        for index, offset in enumerate(offsets))
    selected = (0, 4) if vfr else (0, 12, 25)
    times = [frames[index].pts_sec for index in selected]
    means = tuple((times[index] + times[min(index + 1, len(times) - 1)]) / 2 for index in range(0, len(times), 2))
    clock = ProcessorClockProof(kind="verified_native_pts_ticks_over_ticks_per_second" if vfr else
        "verified_cfr_native_indices_over_source_fps", sampled_frame_indices=selected,
        fps_num=None if vfr else 25, fps_den=None if vfr else 1,
        source_tick_indices=tuple(offsets[index] for index in selected) if vfr else None,
        ticks_per_second_num=1000 if vfr else None, ticks_per_second_den=1 if vfr else None,
        processor_clock_sha256=digest("inspected-processor-clock"), processor_clock_verified=True,
        processor_timestamps_sec=means)
    return asset, frames, tuple(frames[index].frame_id for index in selected), clock


def test_exact_json_proposals_preserve_order_duplicates_and_unrounded_seconds():
    values = parse_intervals("[[0.123456,0.9],[0,0],[0.123456,0.9]]", duration_sec=1.0)
    assert [(item.start_sec, item.end_sec) for item in values] == [(0.123456, 0.9), (0.0, 0.0), (0.123456, 0.9)]
    assert parse_intervals("[]", duration_sec=1.0) == ()


@pytest.mark.parametrize("raw,code", [
    ('{"intervals":[[0,1]]}', "interval_array_required"), ("null", "interval_array_required"),
    ("[[0]]", "interval_pair_required"), ("[[0,1,2]]", "interval_pair_required"),
    ('[["0",1]]', "interval_number_required"), ("[[true,1]]", "interval_number_required"),
    ("[[0,false]]", "interval_number_required"), ("[[NaN,1]]", "invalid_json"),
    ("[[0,Infinity]]", "invalid_json"), ("[[0,1e400]]", "interval_number_required"),
    ("[[-0.1,1]]", "interval_out_of_bounds"), ("[[1,0]]", "interval_out_of_bounds"),
    ("[[0,1.01]]", "interval_out_of_bounds"), ("```json\n[[0,1]]\n```", "invalid_json"),
    ("[[0,1]] trailing", "invalid_json"), ("[[0,", "invalid_json"),
])
def test_bad_or_incomplete_intervals_fail_without_repair(raw, code):
    with pytest.raises(GroundingContractError) as error:
        parse_intervals(raw, duration_sec=1.0)
    assert error.value.code == code


def test_integer_overflow_truncation_and_limits_are_static_failures():
    for raw in ("[[0," + "1" + "0" * 400 + "]]", json.dumps([[0, 1]] * 33)):
        with pytest.raises(GroundingContractError):
            parse_intervals(raw, duration_sec=1.0)
    with pytest.raises(GroundingContractError, match="incomplete_model_completion"):
        parse_intervals("[]", duration_sec=1.0, finish_reason="length")
    for duration in (True, "1", 0.0, float("nan"), float("inf")):
        with pytest.raises(GroundingContractError, match="invalid_duration"):
            parse_intervals("[]", duration_sec=duration)


def test_complete_nonzero_source_clock_and_odd_processor_padding_are_explicit():
    asset, frames, sampled, clock = fixture()
    source = validate_source_ledger(asset, frames)
    assert source.source_origin_pts == 5000 and source.native_frame_count == 26
    assert expected_sampled_frame_indices(asset, frames) == (0, 12, 25)
    assert clock.processor_timestamps_sec == (0.24, 1.0)
    validate_processor_clock(asset, frames, sampled, clock)


def test_snap_keeps_original_seconds_and_can_point_to_an_unsampled_native_frame():
    asset, frames, sampled, clock = fixture()
    result = bind_intervals(parse_intervals("[[0.03,1.04]]", duration_sec=asset.duration_sec),
        asset=asset, source_index=frames, sampled_frame_ids=sampled, clock=clock)[0]
    assert result.proposed_start_sec == 0.03 and result.proposed_end_sec == 1.04
    assert result.start.nearest_source_frame_id == frames[1].frame_id and result.start.source_pts == 5040
    assert result.start.pts_sec == 0.04 and result.start.pts_milliseconds == 40.0
    assert result.start.proposed_milliseconds == 30.0 and result.start.snap_distance_sec == 0.01
    assert not result.start.sampled_by_model
    assert result.end.pts_sec == 1.0 and result.end.snap_distance_sec == 0.04
    assert result.end.source_frame_end_sec == 1.04


def test_exact_midpoint_tie_snaps_to_earlier_actual_source_frame():
    asset, frames, sampled, clock = fixture()
    result = bind_intervals(parse_intervals("[[0.02,0.02]]", duration_sec=asset.duration_sec),
        asset=asset, source_index=frames, sampled_frame_ids=sampled, clock=clock)[0]
    assert result.start.nearest_source_frame_id == frames[0].frame_id


def test_vfr_rejects_cfr_clock_and_accepts_only_separate_verified_native_tick_coordinate():
    asset, frames, sampled, clock = fixture(vfr=True)
    validate_processor_clock(asset, frames, sampled, clock)
    assert clock.processor_timestamps_sec == (0.105,)
    cfr = ProcessorClockProof(kind="verified_cfr_native_indices_over_source_fps", sampled_frame_indices=(0, 4),
        fps_num=25, fps_den=1, processor_clock_sha256=clock.processor_clock_sha256,
        processor_clock_verified=True, processor_timestamps_sec=(0.08,))
    with pytest.raises(GroundingContractError, match="cfr_clock_required"):
        validate_processor_clock(asset, frames, sampled, cfr)
    result = bind_intervals(parse_intervals("[[0.11,0.2]]", duration_sec=asset.duration_sec),
        asset=asset, source_index=frames, sampled_frame_ids=sampled, clock=clock)[0]
    assert result.start.source_pts == 5120 and result.end.source_pts == 5210
    assert not result.start.sampled_by_model


@pytest.mark.parametrize("field,value", [("source_pts", True), ("pts_sec", float("nan")),
    ("next_pts_sec", float("nan")), ("next_source_pts", 9999), ("frame_id", digest("invented")),
    ("content_hash", digest("other-source")), ("width", 256), ("source_origin_pts", True)])
def test_source_binding_rejects_forged_pts_identity_geometry_and_nonfinite_metadata(field, value):
    asset, frames, _, _ = fixture()
    changed = (replace(frames[0], **{field: value}), *frames[1:])
    with pytest.raises(GroundingContractError):
        validate_source_ledger(asset, changed)


def test_tail_missing_or_invented_sample_clock_pairmean_cannot_bind():
    asset, frames, sampled, clock = fixture()
    with pytest.raises(GroundingContractError):
        validate_source_ledger(asset, frames[:-1])
    with pytest.raises(GroundingContractError):
        validate_source_ledger(asset, (*frames[:-1], replace(frames[-1], source_end_bound_sec=float("nan"))))
    bad = clock.model_copy(update={"processor_timestamps_sec": (0.25, 1.0)})
    with pytest.raises(GroundingContractError, match="processor_pair_timestamps_mismatch"):
        validate_processor_clock(asset, frames, sampled, bad)
    bad = clock.model_copy(update={"fps_num": 30})
    with pytest.raises(GroundingContractError, match="processor_clock_mismatch"):
        validate_processor_clock(asset, frames, sampled, bad)


def test_pilot_source_budget_is_bounded_before_sample_target_iteration():
    asset, frames, _, _ = fixture()
    with pytest.raises(GroundingContractError, match="source_budget_exceeded"):
        expected_sampled_frame_indices(asset.model_copy(update={"duration_sec": 1e300}), frames)


@pytest.mark.parametrize("field,value", [("processor_clock_verified", 1),
    ("timestamp_precision_decimals", True), ("temporal_patch_size", 2.0), ("fps_num", True)])
def test_clock_numeric_and_verification_fields_never_coerce_bool_or_float(field, value):
    _, _, _, clock = fixture()
    with pytest.raises(ValidationError):
        ProcessorClockProof.model_validate({**clock.model_dump(), field: value})


def raw_result(raw="[]"):
    asset, frames, sampled, clock = fixture()
    intervals = bind_intervals(parse_intervals(raw, duration_sec=asset.duration_sec),
        asset=asset, source_index=frames, sampled_frame_ids=sampled, clock=clock)
    return GroundingRawResult(**validate_source_ledger(asset, frames).model_dump(),
        query=PhysicalQuery(query_id="CASH_COUNT_VISIBLE", text="Synthetic physical counting query"),
        model_id="synthetic-pilot", model_revision="a" * 40, processor_revision="a" * 40,
        model_manifest_sha256=digest("model"), input_recipe_sha256=digest("input"), clock=clock,
        sampled_frame_ids=sampled, intervals=intervals, raw_response=raw,
        generated_token_ids=(100, 101), metrics={"generated_tokens": 2.0})


def test_empty_raw_result_is_no_proposal_never_a_legal_negative_or_accuracy_claim():
    result = raw_result()
    assert result.proposal_status == "no_grounding_proposal"
    assert result.legal_verdict is None and result.proposal_only
    assert result.true_onset_accuracy == "unmeasured" and result.frame_accuracy == "unmeasured"
    assert not result.millisecond_accuracy_claimed and "unsampled" in result.coverage_reason
    assert GroundingRawResult.model_validate_json(result.model_dump_json()) == result


def test_raw_text_token_and_interval_semantics_are_separate_and_not_silently_corrected():
    result = raw_result("[[0.03,1.0]]")
    assert result.raw_response == "[[0.03,1.0]]" and result.generated_token_ids == (100, 101)
    assert result.proposal_status == "proposal_intervals"
    for update in ({"raw_response": "[]"}, {"legal_verdict": "no_violation"},
                   {"proposal_status": "no_grounding_proposal"}, {"frame_accuracy": "verified"}):
        with pytest.raises(ValidationError):
            GroundingRawResult.model_validate({**result.model_dump(), **update})


def test_query_set_is_operator_pinned_without_an_invented_model_or_legal_catalog():
    queries = validate_queries([{"query_id": "CASH_COUNT_VISIBLE", "text": "Synthetic generic action"}])
    pin = query_set_sha256(queries)
    assert validate_queries([query.model_dump() for query in queries], expected_queries_sha256=pin) == queries
    assert len(queries[0].query_sha256) == 64
    with pytest.raises(GroundingContractError, match="query_set_changed"):
        validate_queries([{"query_id": "CASH_COUNT_VISIBLE", "text": "Changed"}], expected_queries_sha256=pin)
    with pytest.raises(GroundingContractError, match="query_set_invalid"):
        validate_queries([queries[0], queries[0]])
