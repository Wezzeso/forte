"""Strict proposal-only temporal grounding and actual source-clock binding.

No media, model, label, filename lookup or GPU operation occurs here. The pilot
must supply the actual complete A05 native ledger and an independently inspected
processor-clock receipt. Sparse grounding is neither legal classification nor
verification of a true action onset, even when milliseconds are displayed.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from forte.shared.schemas import VideoAsset

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$", strict=True)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Integer = Annotated[int, Field(strict=True)]
Nonnegative = Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
MAX_INTERVALS = 32
MAX_MODEL_TEXT_BYTES = 65536
MAX_PILOT_DURATION_SEC = 32.0
COVERAGE_REASON = (
    "2fps samples plus original tail provide temporal proposals only; unsampled "
    "frames and true action onset are unverified. Processor pair averaging and "
    "one-decimal input timestamps do not establish frame or millisecond accuracy."
)
GROUNDING_ERROR_CODES = frozenset({
    "invalid_duration", "incomplete_model_completion", "invalid_json", "interval_array_required",
    "interval_budget_exceeded", "interval_pair_required", "interval_number_required",
    "interval_out_of_bounds", "source_ledger_missing", "source_identity_mismatch",
    "source_timeline_mismatch", "source_coverage_incomplete", "source_geometry_mismatch",
    "sample_binding_mismatch", "processor_clock_mismatch", "cfr_clock_required",
    "processor_pair_timestamps_mismatch", "query_set_invalid", "query_set_changed",
    "source_budget_exceeded",
})


class GroundingContractError(ValueError):
    def __init__(self, code: str):
        self.code = code if code in GROUNDING_ERROR_CODES else "source_identity_mismatch"
        super().__init__("grounding_pilot_contract_invalid:" + self.code)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class PhysicalQuery(_Contract):
    query_id: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{1,127}$", strict=True)]
    text: Annotated[str, Field(min_length=1, max_length=4096, strict=True)]

    @property
    def query_sha256(self) -> str:
        return _digest(self.model_dump(mode="json"))


def query_set_sha256(queries: Sequence[PhysicalQuery]) -> str:
    return _digest([query.model_dump(mode="json") for query in queries])


def validate_queries(raw_rows: Any, *, expected_queries_sha256: str | None = None) -> tuple[PhysicalQuery, ...]:
    try:
        if not isinstance(raw_rows, (list, tuple)) or not raw_rows or len(raw_rows) > 32:
            raise ValueError
        queries = tuple(PhysicalQuery.model_validate(row) if not isinstance(row, PhysicalQuery) else
            PhysicalQuery.model_validate_json(row.model_dump_json()) for row in raw_rows)
        if len({query.query_id for query in queries}) != len(queries):
            raise ValueError
    except Exception:
        raise GroundingContractError("query_set_invalid") from None
    if expected_queries_sha256 is not None and query_set_sha256(queries) != expected_queries_sha256:
        raise GroundingContractError("query_set_changed")
    return queries


class ProposedInterval(_Contract):
    start_sec: Nonnegative
    end_sec: Nonnegative

    @model_validator(mode="after")
    def ordered(self):
        if self.start_sec > self.end_sec:
            raise ValueError("interval endpoints must be ordered")
        return self


def parse_intervals(raw_json: str | bytes, *, duration_sec: float,
                    finish_reason: str = "stop") -> tuple[ProposedInterval, ...]:
    """Preserve exact proposed seconds/order; never repair, merge, snap or infer.

    Only a JSON array of exact two-number arrays is accepted. Empty means no
    grounding proposal, never a confirmed absence or a no_violation verdict.
    """
    if not _finite_number(duration_sec) or duration_sec <= 0:
        raise GroundingContractError("invalid_duration")
    if finish_reason != "stop":
        raise GroundingContractError("incomplete_model_completion")
    try:
        if not isinstance(raw_json, (str, bytes)):
            raise ValueError
        raw = raw_json.encode("utf-8") if isinstance(raw_json, str) else raw_json
        if len(raw) > MAX_MODEL_TEXT_BYTES:
            raise ValueError
        values = json.loads(raw.decode("utf-8"),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except Exception:
        raise GroundingContractError("invalid_json") from None
    if not isinstance(values, list):
        raise GroundingContractError("interval_array_required")
    if len(values) > MAX_INTERVALS:
        raise GroundingContractError("interval_budget_exceeded")
    intervals = []
    for pair in values:
        if not isinstance(pair, list) or len(pair) != 2:
            raise GroundingContractError("interval_pair_required")
        if any(not _finite_number(value) for value in pair):
            raise GroundingContractError("interval_number_required")
        start, end = pair
        if not 0 <= start <= end <= duration_sec:
            raise GroundingContractError("interval_out_of_bounds")
        intervals.append(ProposedInterval(start_sec=float(start), end_sec=float(end)))
    return tuple(intervals)


class ProcessorClockProof(_Contract):
    kind: Literal["verified_cfr_native_indices_over_source_fps",
                  "verified_native_pts_ticks_over_ticks_per_second"]
    sampled_frame_indices: Annotated[tuple[Integer, ...], Field(min_length=1, max_length=66)]
    fps_num: Annotated[int, Field(gt=0, strict=True)] | None = None
    fps_den: Annotated[int, Field(gt=0, strict=True)] | None = None
    source_tick_indices: tuple[Integer, ...] | None = None
    ticks_per_second_num: Annotated[int, Field(gt=0, strict=True)] | None = None
    ticks_per_second_den: Annotated[int, Field(gt=0, strict=True)] | None = None
    processor_clock_sha256: Digest
    processor_clock_verified: Literal[True]
    temporal_patch_size: Literal[2] = 2
    timestamp_precision_decimals: Literal[1] = 1
    processor_timestamps_sec: Annotated[tuple[Nonnegative, ...], Field(min_length=1, max_length=33)]

    @field_validator("temporal_patch_size", "timestamp_precision_decimals", mode="before")
    @classmethod
    def integer_policy_values(cls, value):
        if type(value) is not int:
            raise ValueError("processor policy integers cannot be bool or float")
        return value

    @field_validator("processor_clock_verified", mode="before")
    @classmethod
    def boolean_receipt(cls, value):
        if type(value) is not bool:
            raise ValueError("processor verification receipt must be boolean")
        return value

    @model_validator(mode="after")
    def explicit_clock(self):
        if (any(index < 0 for index in self.sampled_frame_indices)
                or any(first >= second for first, second in zip(self.sampled_frame_indices, self.sampled_frame_indices[1:]))):
            raise ValueError("sample indices must be actual increasing native ordinals")
        if self.kind == "verified_cfr_native_indices_over_source_fps":
            if (self.fps_num is None or self.fps_den is None or math.gcd(self.fps_num, self.fps_den) != 1
                    or any(value is not None for value in (self.source_tick_indices, self.ticks_per_second_num, self.ticks_per_second_den))):
                raise ValueError("CFR profile requires only exact reduced source FPS")
        elif (self.fps_num is not None or self.fps_den is not None or self.source_tick_indices is None
                or len(self.source_tick_indices) != len(self.sampled_frame_indices)
                or any(tick < 0 for tick in self.source_tick_indices)
                or any(first >= second for first, second in zip(self.source_tick_indices, self.source_tick_indices[1:]))
                or self.ticks_per_second_num is None or self.ticks_per_second_den is None
                or math.gcd(self.ticks_per_second_num, self.ticks_per_second_den) != 1):
            raise ValueError("native-tick profile requires its own exact rational tick clock")
        return self


class SourceLedgerBinding(_Contract):
    source_job_id: str
    content_sha256: Digest
    source_time_base: str
    source_origin_pts: Integer
    duration_sec: Annotated[float, Field(gt=0, le=32, strict=True, allow_inf_nan=False)]
    native_frame_count: Annotated[int, Field(gt=0, strict=True)]
    source_ledger_sha256: Digest


def validate_source_ledger(asset: VideoAsset, source_index: Sequence[Any]) -> SourceLedgerBinding:
    """Validate actual A05 identities without reading/decoding any source bytes."""
    from forte.video.batches import SourceFrameIdentity, source_frame_id
    if not _finite_number(asset.duration_sec) or not 0 < asset.duration_sec <= MAX_PILOT_DURATION_SEC:
        raise GroundingContractError("source_budget_exceeded")
    if not source_index or len(source_index) > 10000:
        raise GroundingContractError("source_ledger_missing")
    try:
        base = Fraction(asset.time_base)
    except (TypeError, ValueError, ZeroDivisionError):
        raise GroundingContractError("source_timeline_mismatch") from None
    if any(not isinstance(frame, SourceFrameIdentity) for frame in source_index):
        raise GroundingContractError("source_identity_mismatch")
    origin = source_index[0].source_origin_pts
    if (base <= 0 or type(origin) is not int
            or abs(float(base * origin) - asset.source_start_sec) > 1e-6):
        raise GroundingContractError("source_timeline_mismatch")
    normalized = []
    seen = set()
    for index, frame in enumerate(source_index):
        if (not isinstance(frame, SourceFrameIdentity) or type(frame.source_pts) is not int
                or frame.content_hash != asset.content_hash or frame.job_id != asset.job_id
                or type(frame.stream_index) is not int or frame.stream_index != asset.stream_index or frame.time_base != asset.time_base
                or type(frame.time_base_num) is not int or type(frame.time_base_den) is not int
                or (frame.time_base_num, frame.time_base_den) != (base.numerator, base.denominator)
                or type(frame.source_origin_pts) is not int or frame.source_origin_pts != origin or frame.frame_id in seen
                or frame.frame_id != source_frame_id(asset.content_hash, frame.source_pts, base.numerator, base.denominator)):
            raise GroundingContractError("source_identity_mismatch")
        if (not _finite_number(frame.pts_sec)
                or abs(frame.pts_sec - float((frame.source_pts - origin) * base)) > 1e-6
                or not 0 <= frame.pts_sec < asset.duration_sec
                or index and source_index[index - 1].source_pts >= frame.source_pts):
            raise GroundingContractError("source_timeline_mismatch")
        if (any(type(value) is not int for value in (frame.width, frame.height, frame.applied_rotation_deg))
                or type(frame.key_frame) is not bool
                or (frame.width, frame.height, frame.applied_rotation_deg) != (asset.width, asset.height, asset.rotation_deg)):
            raise GroundingContractError("source_geometry_mismatch")
        if index + 1 < len(source_index):
            following = source_index[index + 1]
            if (type(frame.next_source_pts) is not int or frame.next_source_pts != following.source_pts
                    or not _finite_number(frame.next_pts_sec)
                    or abs(frame.next_pts_sec - following.pts_sec) > 1e-6 or frame.source_end_bound_sec is not None):
                raise GroundingContractError("source_coverage_incomplete")
        elif (frame.next_source_pts is not None or frame.next_pts_sec is not None
                or not _finite_number(frame.source_end_bound_sec)
                or abs(frame.source_end_bound_sec - asset.duration_sec) > 1e-6):
            raise GroundingContractError("source_coverage_incomplete")
        seen.add(frame.frame_id)
        normalized.append({"frame_id": frame.frame_id, "source_pts": frame.source_pts,
            "pts_sec": frame.pts_sec, "next_source_pts": frame.next_source_pts,
            "next_pts_sec": frame.next_pts_sec, "source_end_bound_sec": frame.source_end_bound_sec,
            "key_frame": frame.key_frame})
    if (source_index[0].source_pts != origin or source_index[0].pts_sec != 0.0
            or asset.frame_count is not None and asset.frame_count != len(source_index)):
        raise GroundingContractError("source_coverage_incomplete")
    binding = {"source_job_id": asset.job_id, "content_sha256": asset.content_hash,
        "source_time_base": asset.time_base, "source_origin_pts": origin,
        "duration_sec": asset.duration_sec, "native_frame_count": len(source_index)}
    return SourceLedgerBinding(**binding, source_ledger_sha256=_digest({**binding, "frames": normalized}))


def validate_processor_clock(asset: VideoAsset, source_index: Sequence[Any],
        sampled_frame_ids: Sequence[str], clock: ProcessorClockProof) -> None:
    validate_source_ledger(asset, source_index)
    try:
        clock = ProcessorClockProof.model_validate_json(clock.model_dump_json())
    except Exception:
        raise GroundingContractError("processor_clock_mismatch") from None
    if (not sampled_frame_ids or len(sampled_frame_ids) != len(clock.sampled_frame_indices)
            or any(not isinstance(frame_id, str) for frame_id in sampled_frame_ids)
            or sampled_frame_ids[0] != source_index[0].frame_id or sampled_frame_ids[-1] != source_index[-1].frame_id
            or len(set(sampled_frame_ids)) != len(sampled_frame_ids)
            or any(index >= len(source_index) for index in clock.sampled_frame_indices)
            or tuple(source_index[index].frame_id for index in clock.sampled_frame_indices) != tuple(sampled_frame_ids)):
        raise GroundingContractError("sample_binding_mismatch")
    if clock.sampled_frame_indices != expected_sampled_frame_indices(asset, source_index):
        raise GroundingContractError("sample_binding_mismatch")
    base = Fraction(asset.time_base)
    if clock.kind == "verified_cfr_native_indices_over_source_fps":
        gaps = {second.source_pts - first.source_pts for first, second in zip(source_index, source_index[1:])}
        if len(gaps) != 1:
            raise GroundingContractError("cfr_clock_required")
        source_fps = 1 / (base * next(iter(gaps)))
        if Fraction(clock.fps_num, clock.fps_den) != source_fps:
            raise GroundingContractError("processor_clock_mismatch")
        times = [Fraction(index * clock.fps_den, clock.fps_num) for index in clock.sampled_frame_indices]
    else:
        ticks = tuple(source_index[index].source_pts - source_index[0].source_origin_pts for index in clock.sampled_frame_indices)
        if (ticks != clock.source_tick_indices or Fraction(clock.ticks_per_second_num, clock.ticks_per_second_den) != 1 / base):
            raise GroundingContractError("processor_clock_mismatch")
        times = [tick * base for tick in ticks]
    for index, value in zip(clock.sampled_frame_indices, times):
        if abs(float(value) - source_index[index].pts_sec) > 1e-6:
            raise GroundingContractError("processor_clock_mismatch")
    # Qwen3-VL repeats the last index for an odd final temporal patch, averages
    # each pair, then renders one decimal. Preserve the unrounded means here.
    expected = tuple(float((times[index] + times[min(index + 1, len(times) - 1)]) / 2)
                     for index in range(0, len(times), 2))
    if (len(expected) != len(clock.processor_timestamps_sec)
            or any(abs(actual - wanted) > 1e-6 for actual, wanted in zip(clock.processor_timestamps_sec, expected))):
        raise GroundingContractError("processor_pair_timestamps_mismatch")


def expected_sampled_frame_indices(asset: VideoAsset, source_index: Sequence[Any]) -> tuple[int, ...]:
    """Exact pinned2fps target selection: nearest native PTS, earlier ties, tail."""
    validate_source_ledger(asset, source_index)
    base, origin = Fraction(asset.time_base), source_index[0].source_origin_pts
    times = tuple((frame.source_pts - origin) * base for frame in source_index)
    selected = []
    for index in range(math.ceil(Fraction(str(asset.duration_sec)) * 2)):
        target = Fraction(index, 2)
        ordinal = min(range(len(source_index)), key=lambda item: (abs(times[item] - target), times[item]))
        if ordinal not in selected:
            selected.append(ordinal)
    if len(source_index) - 1 not in selected:
        selected.append(len(source_index) - 1)
    return tuple(selected)


class BoundarySnap(_Contract):
    proposed_sec: Nonnegative
    proposed_milliseconds: Nonnegative
    nearest_source_frame_id: Digest
    source_pts: Integer
    pts_sec: Nonnegative
    pts_milliseconds: Nonnegative
    snap_distance_sec: Nonnegative
    sampled_by_model: bool
    source_frame_end_sec: Nonnegative
    source_frame_interval_sec: Nonnegative

    @model_validator(mode="after")
    def consistent_units(self):
        if (abs(self.proposed_milliseconds - self.proposed_sec * 1000) > 1e-6
                or abs(self.pts_milliseconds - self.pts_sec * 1000) > 1e-6
                or abs(self.snap_distance_sec - abs(self.pts_sec - self.proposed_sec)) > 1e-6
                or self.source_frame_end_sec <= self.pts_sec
                or abs(self.source_frame_interval_sec - (self.source_frame_end_sec - self.pts_sec)) > 1e-6):
            raise ValueError("boundary units/source interval inconsistent")
        return self


class BoundInterval(_Contract):
    proposed_start_sec: Nonnegative
    proposed_end_sec: Nonnegative
    start: BoundarySnap
    end: BoundarySnap
    source_ledger_sha256: Digest
    processor_clock_sha256: Digest

    @model_validator(mode="after")
    def unchanged_proposal(self):
        if (self.proposed_start_sec > self.proposed_end_sec or self.start.proposed_sec != self.proposed_start_sec
                or self.end.proposed_sec != self.proposed_end_sec):
            raise ValueError("bound interval must preserve model proposal")
        return self


def bind_intervals(intervals: Sequence[ProposedInterval], *, asset: VideoAsset,
        source_index: Sequence[Any], sampled_frame_ids: Sequence[str],
        clock: ProcessorClockProof) -> tuple[BoundInterval, ...]:
    """Bind unchanged proposals to nearest actual source PTS; no accuracy claim."""
    source = validate_source_ledger(asset, source_index)
    validate_processor_clock(asset, source_index, sampled_frame_ids, clock)
    if len(intervals) > MAX_INTERVALS:
        raise GroundingContractError("interval_budget_exceeded")
    base, origin = Fraction(asset.time_base), source.source_origin_pts
    actual_times = tuple((frame.source_pts - origin) * base for frame in source_index)
    sampled = set(sampled_frame_ids)
    def snap(proposed: float) -> BoundarySnap:
        exact = Fraction(str(proposed))
        position = min(range(len(source_index)), key=lambda index: (abs(actual_times[index] - exact), actual_times[index]))
        frame = source_index[position]
        ending = frame.next_pts_sec if frame.next_pts_sec is not None else frame.source_end_bound_sec
        return BoundarySnap(proposed_sec=proposed, proposed_milliseconds=proposed * 1000,
            nearest_source_frame_id=frame.frame_id, source_pts=frame.source_pts,
            pts_sec=frame.pts_sec, pts_milliseconds=frame.pts_sec * 1000,
            snap_distance_sec=float(abs(actual_times[position] - exact)), sampled_by_model=frame.frame_id in sampled,
            source_frame_end_sec=ending, source_frame_interval_sec=ending - frame.pts_sec)
    result = []
    for interval in intervals:
        if (not isinstance(interval, ProposedInterval) or not 0 <= interval.start_sec <= interval.end_sec <= asset.duration_sec
                or not _finite_number(interval.start_sec) or not _finite_number(interval.end_sec)):
            raise GroundingContractError("interval_out_of_bounds")
        result.append(BoundInterval(proposed_start_sec=interval.start_sec, proposed_end_sec=interval.end_sec,
            start=snap(interval.start_sec), end=snap(interval.end_sec), source_ledger_sha256=source.source_ledger_sha256,
            processor_clock_sha256=clock.processor_clock_sha256))
    return tuple(result)


class GroundingRawResult(_Contract):
    schema_version: Literal["forte.grounding-pilot-result.v1"] = "forte.grounding-pilot-result.v1"
    source_job_id: str
    content_sha256: Digest
    source_time_base: str
    source_origin_pts: Integer
    duration_sec: Annotated[float, Field(gt=0, le=32, strict=True, allow_inf_nan=False)]
    source_ledger_sha256: Digest
    native_frame_count: Annotated[int, Field(gt=0, strict=True)]
    query: PhysicalQuery
    query_sha256: Digest | None = None
    model_id: str
    model_revision: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$", strict=True)]
    processor_revision: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$", strict=True)]
    model_manifest_sha256: Digest
    input_recipe_sha256: Digest
    clock: ProcessorClockProof
    sampled_frame_ids: Annotated[tuple[Digest, ...], Field(min_length=1, max_length=66)]
    intervals: Annotated[tuple[BoundInterval, ...], Field(max_length=32)]
    raw_response: Annotated[str, Field(max_length=65536)] = Field(repr=False)
    generated_token_ids: Annotated[tuple[Annotated[int, Field(ge=0, strict=True)], ...], Field(max_length=256)] = Field(repr=False)
    finish_reason: Literal["stop"] = "stop"
    metrics: dict[str, Nonnegative]
    proposal_status: Literal["proposal_intervals", "no_grounding_proposal"] | None = None
    proposal_only: Literal[True] = True
    legal_verdict: Literal[None] = None
    true_onset_accuracy: Literal["unmeasured"] = "unmeasured"
    frame_accuracy: Literal["unmeasured"] = "unmeasured"
    millisecond_accuracy_claimed: Literal[False] = False
    inference_coverage: Literal["sparse_2fps_plus_original_tail"] = "sparse_2fps_plus_original_tail"
    coverage_reason: Literal[COVERAGE_REASON] = COVERAGE_REASON

    @field_validator("proposal_only", "millisecond_accuracy_claimed", mode="before")
    @classmethod
    def boolean_scope_flags(cls, value):
        if type(value) is not bool:
            raise ValueError("scope flags must be explicit booleans")
        return value

    @model_validator(mode="after")
    def consistent_proposal(self):
        from forte.video.batches import source_frame_id
        expected_query_hash = self.query.query_sha256
        if self.query_sha256 is not None and self.query_sha256 != expected_query_hash:
            raise ValueError("result query hash inconsistent")
        object.__setattr__(self, "query_sha256", expected_query_hash)
        parsed = parse_intervals(self.raw_response, duration_sec=self.duration_sec, finish_reason=self.finish_reason)
        if tuple((item.start_sec, item.end_sec) for item in parsed) != tuple(
                (item.proposed_start_sec, item.proposed_end_sec) for item in self.intervals):
            raise ValueError("result must retain exact raw model interval proposals")
        expected = "proposal_intervals" if self.intervals else "no_grounding_proposal"
        if self.proposal_status is not None and self.proposal_status != expected:
            raise ValueError("proposal status must match proposal presence")
        object.__setattr__(self, "proposal_status", expected)
        if (len(set(self.sampled_frame_ids)) != len(self.sampled_frame_ids)
                or len(self.sampled_frame_ids) != len(self.clock.sampled_frame_indices)
                or self.model_revision != self.processor_revision
                or any(interval.source_ledger_sha256 != self.source_ledger_sha256
                    or interval.processor_clock_sha256 != self.clock.processor_clock_sha256
                    or not 0 <= interval.proposed_start_sec <= interval.proposed_end_sec <= self.duration_sec
                    for interval in self.intervals)):
            raise ValueError("result source/processor binding inconsistent")
        base = Fraction(self.source_time_base)
        for interval in self.intervals:
            for boundary in (interval.start, interval.end):
                if (boundary.nearest_source_frame_id != source_frame_id(self.content_sha256, boundary.source_pts,
                        base.numerator, base.denominator)
                        or abs(boundary.pts_sec - float((boundary.source_pts - self.source_origin_pts) * base)) > 1e-6
                        or not boundary.pts_sec < self.duration_sec or boundary.source_frame_end_sec > self.duration_sec
                        or boundary.sampled_by_model != (boundary.nearest_source_frame_id in self.sampled_frame_ids)):
                    raise ValueError("result boundary must bind actual canonical source clock")
        return self
