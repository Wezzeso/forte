"""Arbitrary-source native head runner for the optional isolated GPU service.

This has no source-name/label lookup or training split. A05/A15 supply verified
native source metadata and encoder features for the actual uploaded VideoAsset.
The CPU worker verifies calibration and builds bounded source evidence later.
No ML packages are imported until an explicitly requested load/predict call.
"""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forte.shared.schemas import VideoAsset

from .head_contract import Digest, Finite, HeadIdentity, Identifier, TemporalHeadOutput
from .temporal_provider import TemporalProviderError, _checkpoint


class _NativeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class NativeFeatureFrame(_NativeContract):
    frame_id: Digest
    source_pts: Annotated[int, Field(strict=True)]
    next_source_pts: Annotated[int, Field(strict=True)] | None
    pts_sec: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
    next_pts_sec: Finite | None
    source_end_bound_sec: Finite | None
    width: Annotated[int, Field(gt=0, le=65536, strict=True)]
    height: Annotated[int, Field(gt=0, le=65536, strict=True)]
    applied_rotation_deg: Literal[0, 90, 180, 270]
    key_frame: bool
    view_mask: tuple[bool, ...]
    view_roles: tuple[Literal["global", "roi"], ...]
    normalized_boxes: tuple[tuple[Finite, Finite, Finite, Finite], ...]


class NativeSourceContext(_NativeContract):
    schema_version: Literal["forte.native-head-source.v1"] = "forte.native-head-source.v1"
    job_id: Identifier
    clip_id: Identifier
    content_sha256: Digest
    duration_sec: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]
    time_base_num: Annotated[int, Field(gt=0, strict=True)]
    time_base_den: Annotated[int, Field(gt=0, strict=True)]
    origin_pts: Annotated[int, Field(strict=True)]
    stream_index: Annotated[int, Field(ge=0, strict=True)]
    feature_spec_sha256: Digest
    encoder_manifest_sha256: Digest
    source_complete: bool
    tail_included: bool
    frames: Annotated[tuple[NativeFeatureFrame, ...], Field(min_length=1, max_length=10000)]

    @model_validator(mode="after")
    def native_binding(self):
        from forte.video.batches import source_frame_id

        if not self.source_complete or not self.tail_included:
            raise ValueError("native source must be complete, including original tail")
        if self.frames[0].source_pts != self.origin_pts or self.frames[0].pts_sec != 0.0:
            raise ValueError("native source beginning missing")
        previous = None
        ids = set()
        views = len(self.frames[0].view_mask)
        if not 1 <= views <= 8:
            raise ValueError("native view budget invalid")
        for index, frame in enumerate(self.frames):
            expected_id = source_frame_id(self.content_sha256, frame.source_pts,
                                          self.time_base_num, self.time_base_den)
            expected_pts = (frame.source_pts - self.origin_pts) * self.time_base_num / self.time_base_den
            if (frame.frame_id != expected_id or frame.frame_id in ids
                    or not math.isclose(frame.pts_sec, expected_pts, rel_tol=0, abs_tol=1e-6)
                    or not 0 <= frame.pts_sec < self.duration_sec
                    or previous is not None and frame.source_pts <= previous):
                raise ValueError("native canonical frame/PTS binding invalid")
            ids.add(frame.frame_id)
            if index + 1 < len(self.frames):
                next_frame = self.frames[index + 1]
                if (frame.next_source_pts != next_frame.source_pts or frame.next_pts_sec is None
                        or not math.isclose(frame.next_pts_sec, next_frame.pts_sec, rel_tol=0, abs_tol=1e-6)
                        or frame.source_end_bound_sec is not None):
                    raise ValueError("native adjacency chain invalid")
            elif (frame.next_source_pts is not None or frame.next_pts_sec is not None
                    or frame.source_end_bound_sec is None
                    or not math.isclose(frame.source_end_bound_sec, self.duration_sec, rel_tol=0, abs_tol=1e-6)):
                raise ValueError("native original tail binding invalid")
            if not len(frame.view_mask) == len(frame.view_roles) == len(frame.normalized_boxes) == views:
                raise ValueError("native view shape invalid")
            if not any(valid and role == "global" for valid, role in zip(frame.view_mask, frame.view_roles)):
                raise ValueError("native global view missing")
            for valid, role, box in zip(frame.view_mask, frame.view_roles, frame.normalized_boxes):
                if (not all(0 <= value <= 1 for value in box) or box[0] >= box[2] or box[1] >= box[3]
                        or valid and role == "global" and box != (0.0, 0.0, 1.0, 1.0)):
                    raise ValueError("native view geometry invalid")
            previous = frame.source_pts
        return self

    def validate_asset(self, asset: VideoAsset, identity: HeadIdentity) -> None:
        try:
            base = Fraction(asset.time_base)
        except (TypeError, ValueError, ZeroDivisionError):
            raise TemporalProviderError("native_asset_time_base_missing") from None
        if (self.job_id != asset.job_id or self.content_sha256 != asset.content_hash
                or self.duration_sec != asset.duration_sec or self.stream_index != asset.stream_index
                or Fraction(self.time_base_num, self.time_base_den) != base
                or not math.isclose(float(base * self.origin_pts), asset.source_start_sec, rel_tol=0, abs_tol=1e-6)
                or self.feature_spec_sha256 != identity.feature_spec_sha256
                or self.encoder_manifest_sha256 != identity.encoder_artifacts_sha256
                or any(frame.width != asset.width or frame.height != asset.height
                       or frame.applied_rotation_deg != asset.rotation_deg for frame in self.frames)
                or asset.frame_count is not None and len(self.frames) != asset.frame_count):
            raise TemporalProviderError("native_asset_or_feature_identity_mismatch")

    def source_index(self):
        """Reconstruct A05 identities from retained actual decoder metadata.

        These are source ledger records, never JPEG FrameRefs. The evidence
        materializer still verifies the original source and exact selected PTS.
        """
        from forte.video.batches import SourceFrameIdentity
        return tuple(SourceFrameIdentity(frame_id=frame.frame_id, content_hash=self.content_sha256,
            job_id=self.job_id, stream_index=self.stream_index, source_pts=frame.source_pts,
            time_base=f"{self.time_base_num}/{self.time_base_den}", time_base_num=self.time_base_num,
            time_base_den=self.time_base_den, source_origin_pts=self.origin_pts,
            pts_sec=frame.pts_sec, next_source_pts=frame.next_source_pts, next_pts_sec=frame.next_pts_sec,
            source_end_bound_sec=frame.source_end_bound_sec, width=frame.width, height=frame.height,
            applied_rotation_deg=frame.applied_rotation_deg, key_frame=frame.key_frame) for frame in self.frames)


class NativeHeadResult(_NativeContract):
    schema_version: Literal["forte.native-head-result.v1"] = "forte.native-head-result.v1"
    context: NativeSourceContext = Field(repr=False)
    output: TemporalHeadOutput = Field(repr=False)
    video_logit: Finite
    # No notes, paths, per-frame observations, labels or identities in telemetry.
    metrics: dict[str, Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]]

    @model_validator(mode="after")
    def output_source_binding(self):
        if set(self.metrics) - {"head_elapsed_sec", "native_frames", "decode_elapsed_sec", "encoder_elapsed_sec",
                               "source_verification_sec", "total_elapsed_sec"}:
            raise ValueError("native metric keys must be static")
        if (self.output.job_id != self.context.job_id or self.output.clip_id != self.context.clip_id
                or len(self.output.frames) != len(self.context.frames)
                or not self.output.feature_coverage_complete or not self.output.native_source_continuity_verified
                or any((frame.frame_id, frame.pts_sec, frame.source_pts, frame.next_source_pts) !=
                       (source.frame_id, source.pts_sec, source.source_pts, source.next_source_pts)
                       for frame, source in zip(self.output.frames, self.context.frames))):
            raise ValueError("native head output/source binding invalid")
        return self


class NativeHeadRunner:
    """Persistent real A09 checkpoint; A15 owns the persistent warm encoder.

    The containing GPU sidecar serializes this runner. It must verify the actual
    uploaded source bytes before producing NativeSourceContext. This class does
    not authorize a checkpoint, select thresholds or publish a legal verdict.
    """
    def __init__(self, checkpoint_dir: Path, *, checkpoint_manifest_sha256: str,
                 identity: HeadIdentity, device: str = "cpu"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_manifest_sha256 = checkpoint_manifest_sha256
        self.identity, self.device = identity, device
        self._model = None
        self._trained_components: dict[str, bool] = {}

    def load(self) -> dict[str, Any]:
        from training.temporal.checkpoint import load_checkpoint
        try:
            model, identity, metadata = load_checkpoint(self.checkpoint_dir,
                expected_manifest_sha256=self.checkpoint_manifest_sha256, device=self.device)
            components = metadata.get("provenance", {}).get("trained_components", {})
            if (identity != self.identity or not isinstance(components, dict)
                    or any(type(value) is not bool for value in components.values())):
                raise TemporalProviderError("native_checkpoint_identity_or_training_mask_invalid")
            self._model, self._trained_components = model, dict(components)
            return self.check_ready()
        except TemporalProviderError:
            raise
        except Exception:
            raise TemporalProviderError("native_checkpoint_load_failed") from None

    def check_ready(self) -> dict[str, Any]:
        return {"ready": self._model is not None, "head_sha256": self.identity.head_sha256,
            "feature_spec_sha256": self.identity.feature_spec_sha256,
            "trained_components": dict(self._trained_components), "candidate_calibration_verified": False}

    def predict_native(self, context: NativeSourceContext, features: Any, *,
                       deadline: float, cancel_event: Any = None) -> NativeHeadResult:
        _checkpoint(deadline, cancel_event)
        if self._model is None:
            raise TemporalProviderError("native_checkpoint_not_loaded")
        if (context.feature_spec_sha256 != self.identity.feature_spec_sha256
                or context.encoder_manifest_sha256 != self.identity.encoder_artifacts_sha256):
            raise TemporalProviderError("native_feature_recipe_mismatch")
        from training.temporal.checkpoint import sequence_inputs
        import torch
        started = time.monotonic()
        try:
            frames, facts, events = len(context.frames), len(self.identity.fact_ids), len(self.identity.event_type_ids)
            if (not isinstance(features, torch.Tensor) or features.device.type != "cpu"
                    or tuple(features.shape) != (frames, len(context.frames[0].view_mask), self._model.project[0].in_features)
                    or features.dtype not in {torch.float16, torch.float32}
                    or not bool(torch.isfinite(features).all())):
                raise TemporalProviderError("native_feature_tensor_invalid")
            inputs = sequence_inputs(context, features, device=self.device)
            _checkpoint(deadline, cancel_event)
            with torch.inference_mode():
                values = self._model(**inputs)

            def tensor(name: str, shape: tuple[int, ...]):
                value = values.get(name)
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                        or not bool(torch.isfinite(value).all())):
                    raise TemporalProviderError("native_head_tensor_invalid")
                return value.detach().to(device="cpu", dtype=torch.float32).tolist()[0]

            output = TemporalHeadOutput.model_validate({"identity": self.identity,
                "job_id": context.job_id, "clip_id": context.clip_id,
                "frames": [{"frame_id": frame.frame_id, "pts_sec": frame.pts_sec,
                    "source_pts": frame.source_pts, "next_source_pts": frame.next_source_pts} for frame in context.frames],
                "window_logits": tensor("window_logits", (1, facts, 3)),
                "frame_logits": tensor("frame_logits", (1, frames, facts, 3)),
                "support_logits": tensor("support_logits", (1, frames, facts)),
                "boundary_logits": tensor("boundary_logits", (1, frames, events, 2)) if events else None,
                "frame_valid": [any(frame.view_mask) for frame in context.frames],
                "global_view_available": [any(valid and role == "global" for valid, role in
                    zip(frame.view_mask, frame.view_roles)) for frame in context.frames],
                "feature_coverage_complete": True, "native_source_continuity_verified": True,
                "trained_components": self._trained_components}, strict=False)
            video_logit = tensor("video_logit", (1,))
            _checkpoint(deadline, cancel_event)
            return NativeHeadResult(context=context, output=output, video_logit=video_logit,
                metrics={"head_elapsed_sec": time.monotonic() - started, "native_frames": float(frames)})
        except TemporalProviderError:
            raise
        except Exception:
            raise TemporalProviderError("native_head_execution_failed") from None
