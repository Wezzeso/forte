"""CPU native localization and bounded evidence projection, without fake refs.

Full native source/logit coverage is validated before selecting a small set of
actual original JPEGs. A05 must materialize those JPEGs from the source; only
then does the existing strict observation contract produce a DecisionBatch.
The plan itself is private provisional observations, never a public verdict.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef, RuleRecord, VideoAsset
from forte.observability.model_metrics import number

from .machine_contract import MachineCalibrationManifest, MachineEvidenceVerifier
from .temporal_native import NativeHeadResult
from .temporal_provider import PhysicalFactRunPolicy, TemporalProviderError, _map_source_bound_output
from .structured import DecisionConfig, validate_decisions


def _digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, repr=False)
class NativeObservationPlan:
    result: NativeHeadResult = field(repr=False)
    asset: VideoAsset = field(repr=False)
    fact_definitions: tuple[FactDefinition, ...] = field(repr=False)
    rule_definitions: tuple[RuleRecord, ...] = field(repr=False)
    data: dict[str, Any] = field(repr=False)
    event_type_ids: tuple[str, ...]
    requested_frame_ids: tuple[str, ...] = field(repr=False)
    _data_sha256: str = field(repr=False)
    provisional: bool = True

    def bind_evidence_frames(self, frames: Sequence[FrameRef]):
        """Validate actual A05 JPEGs and return the ordinary strict DecisionBatch.

        The <=96 bound remains. No full-sequence FrameRef with a fake image path
        is ever constructed. Do not truncate the requested set if over budget.
        """
        if _digest(self.data) != self._data_sha256:
            raise TemporalProviderError("native_observation_plan_changed")
        self.result.context.validate_asset(self.asset, self.result.output.identity)
        if len(self.requested_frame_ids) > 96:
            raise TemporalProviderError("native_evidence_projection_budget_exceeded")
        source = {frame.frame_id: frame for frame in self.result.context.frames}
        if (len(frames) != len(self.requested_frame_ids)
                or {frame.frame_id for frame in frames} != set(self.requested_frame_ids)):
            raise TemporalProviderError("native_materialized_frame_set_mismatch")
        by_id = {frame.frame_id: frame for frame in frames}
        for frame in frames:
            expected = source[frame.frame_id]
            path = Path(frame.path)
            try:
                valid = (frame.pts_sec == expected.pts_sec and frame.source_pts == expected.source_pts
                    and frame.job_id in {None, self.asset.job_id}
                    and frame.width == expected.width and frame.height == expected.height
                    and bool(frame.jpeg_sha256) and not path.is_symlink() and path.is_file()
                    and path.stat().st_size <= 20 * 1024 * 1024
                    and hashlib.sha256(path.read_bytes()).hexdigest() == frame.jpeg_sha256)
            except OSError:
                valid = False
            if not valid:
                raise TemporalProviderError("native_materialized_frame_binding_invalid")
        request = ClipRequest(job_id=self.asset.job_id, clip_id=self.result.output.clip_id,
            frame_refs=[by_id[frame_id] for frame_id in self.requested_frame_ids],
            fact_definitions=list(self.fact_definitions), rule_definitions=list(self.rule_definitions),
            start_sec=0.0, end_sec=self.asset.duration_sec)
        batch = validate_decisions(self.data, request, reviewed_event_ids=self.event_type_ids,
            config=DecisionConfig(concise_observations=True), metrics={
                "elapsed_sec": number(self.result.metrics.get("total_elapsed_sec",
                    self.result.metrics.get("head_elapsed_sec", 0.0))) or 0.0,
                "input_frames": len(self.result.context.frames), "generated_tokens": 0,
                "serving_attestation": {"source": "provisional_native_temporal_head",
                    "model_revision": self.result.output.identity.head_sha256,
                    "processor_revision": self.result.output.identity.encoder_revision,
                    "provisional": True, "human_fact_event_gold": False,
                    "independent_fact_event_accuracy_measured": False}})
        return batch


def map_native_observations(result: NativeHeadResult, asset: VideoAsset,
    manifest: MachineCalibrationManifest, *, verifier: MachineEvidenceVerifier,
    fact_definitions: Sequence[FactDefinition], rule_definitions: Sequence[RuleRecord] = (),
    event_definitions: Mapping[str, Mapping[str, Any]],
    absence_context_fact_ids: Mapping[str, tuple[str, ...]],
    physical_fact_run_policy: PhysicalFactRunPolicy | None = None,
    trusted_scene_context: str | None = None,
) -> NativeObservationPlan:
    """Bind every actual native frame, then localize honestly provisional facts.

    This is the live-upload counterpart to the offline bounded cache backend.
    It does not look up a known recording, label, anchor or previous prediction.
    Missing fact calibration/head supervision yields unclear, not a clean label.
    """
    try:
        checked = NativeHeadResult.model_validate_json(result.model_dump_json(), strict=True)
    except Exception:
        raise TemporalProviderError("native_observation_result_contract_invalid") from None
    checked.context.validate_asset(asset, manifest.identity)
    if (checked.output.identity != manifest.identity
            or tuple(fact.fact_id for fact in fact_definitions) != manifest.identity.fact_ids):
        raise TemporalProviderError("native_observation_axis_or_model_mismatch")
    data = _map_source_bound_output(checked.output, manifest, verifier=verifier,
        event_definitions=event_definitions, absence_context_fact_ids=absence_context_fact_ids,
        physical_fact_run_policy=physical_fact_run_policy, trusted_scene_context=trusted_scene_context)
    required = {frame_id for fact in data["facts"] for frame_id in fact["supporting_frame_ids"]}
    for candidate in data["candidates"]:
        required.update((candidate["start_frame_id"], candidate["end_frame_id"], *candidate["supporting_frame_ids"]))
    # A01's request requires an actual frame even if every fact is unclear. The
    # first original frame may be materialized for audit, without inventing any
    # support or promoting an observation. Callers can fail incomplete before it.
    if not required:
        required.add(checked.context.frames[0].frame_id)
    ordered = tuple(frame.frame_id for frame in checked.context.frames if frame.frame_id in required)
    if len(ordered) != len(required) or len(ordered) > 96:
        raise TemporalProviderError("native_evidence_projection_budget_exceeded")
    return NativeObservationPlan(checked, asset.model_copy(deep=True),
        tuple(fact.model_copy(deep=True) for fact in fact_definitions),
        tuple(rule.model_copy(deep=True) for rule in rule_definitions), data,
        tuple(event_definitions), ordered, _digest(data))
