"""Real opt-in checkpoint/cache backend; no model is loaded at import time.

The A09 loaders verify safetensors and immutable feature/source identities. This
adapter supplies only feature/time/view tensors to the head. It never looks up
provided video labels or teacher decisions while predicting. The bounded A08
request remains canonical, and no synthetic path or FrameRef is constructed.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from pathlib import Path
import re
import time
from typing import Any

from forte.shared.schemas import ClipRequest

from .head_contract import HeadIdentity, SourceFrame, TemporalHeadOutput
from .temporal_provider import HeadBackendResult, TemporalProviderError, _checkpoint


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def bind_feature_indices(sequence: Any, request: ClipRequest) -> tuple[int, ...]:
    """Bind actual request IDs, integer PTS and normalized PTS to cache entries."""
    by_id = {frame.frame_id: index for index, frame in enumerate(sequence.frames)}
    indices = []
    for reference in request.frame_refs:
        index = by_id.get(reference.frame_id)
        if index is None:
            raise TemporalProviderError("request_source_frame_missing_from_cache")
        frame = sequence.frames[index]
        if (reference.pts_sec != frame.pts_sec
                or reference.source_pts is not None and reference.source_pts != frame.source_pts):
            raise TemporalProviderError("request_source_pts_mismatch")
        indices.append(index)
    if not indices or any(first >= second for first, second in zip(indices, indices[1:])):
        raise TemporalProviderError("request_source_order_mismatch")
    return tuple(indices)


def native_window_coverage(sequence: Any, request: ClipRequest, indices: tuple[int, ...]) -> bool:
    """Native continuity is feature coverage, never visibility or procedure proof."""
    if indices != tuple(range(indices[0], indices[-1] + 1)):
        return False
    frames = [sequence.frames[index] for index in indices]
    if any(first.next_source_pts != second.source_pts for first, second in zip(frames, frames[1:])):
        return False
    if request.start_sec is None or request.end_sec is None:
        return False
    if frames[0].pts_sec > request.start_sec:
        return False
    tail = frames[-1].next_pts_sec
    if tail is None:
        tail = frames[-1].source_end_bound_sec
    return tail is not None and tail >= request.end_sec


class CachedTemporalHeadBackend:
    """Execute a genuine pinned A09 head over one source-bound private cache.

    Loading and inference run in one guarded thread. A cancelled/timed-out call
    cannot free the model for a second overlapping call while torch still runs.
    There is no model/mock fallback. This initial adapter takes at most96 actual
    FrameRefs through A08; full dense caches stay intact and may supply separate
    windows. A whole-record MIL evaluation uses A09's separate executable flow.
    """

    def __init__(self, checkpoint_dir: Path, *, checkpoint_manifest_sha256: str,
                 feature_manifest: Path, feature_manifest_sha256: str,
                 identity: HeadIdentity, device: str = "cpu"):
        if (not re.fullmatch(r"[a-f0-9]{64}", checkpoint_manifest_sha256)
                or not re.fullmatch(r"[a-f0-9]{64}", feature_manifest_sha256)
                or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device)):
            raise TemporalProviderError("backend_configuration_invalid")
        self.identity = identity
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_manifest_sha256 = checkpoint_manifest_sha256
        self.feature_manifest = Path(feature_manifest)
        self.feature_manifest_sha256 = feature_manifest_sha256
        self.device = device
        self._model = self._sequence = self._features = self._cache_manifest = None
        self._trained_components: dict[str, bool] = {}
        self._running: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    def _load_sync(self) -> None:
        from training.temporal.cache import load_feature_cache
        from training.temporal.checkpoint import load_checkpoint

        try:
            if (self.feature_manifest.is_symlink() or not self.feature_manifest.is_file()
                    or self.feature_manifest.stat().st_size > 32 * 1024 * 1024
                    or _file_sha256(self.feature_manifest) != self.feature_manifest_sha256):
                raise TemporalProviderError("feature_manifest_pin_mismatch")
            model, identity, metadata = load_checkpoint(self.checkpoint_dir,
                expected_manifest_sha256=self.checkpoint_manifest_sha256, device=self.device)
            sequence, features, payload = load_feature_cache(self.feature_manifest)
            if (identity != self.identity or payload.get("feature_spec_sha256") != identity.feature_spec_sha256
                    or sequence.encoder_manifest_sha256 != identity.encoder_artifacts_sha256):
                raise TemporalProviderError("checkpoint_feature_identity_mismatch")
            components = metadata.get("provenance", {}).get("trained_components", {})
            if not isinstance(components, dict) or any(type(value) is not bool for value in components.values()):
                raise TemporalProviderError("checkpoint_trained_components_invalid")
            self._model, self._sequence, self._features, self._cache_manifest = model, sequence, features, payload
            self._trained_components = dict(components)
        except TemporalProviderError:
            raise
        except Exception:
            # Loader exceptions may contain paths or private metadata. Surface a
            # static technical class and leave details in the private loader.
            raise TemporalProviderError("checkpoint_or_feature_load_failed") from None

    async def _run(self, function: Any, *, deadline: float) -> Any:
        _checkpoint(deadline, None)
        async with self._lock:
            if self._closed:
                raise TemporalProviderError("backend_closed")
            if self._running is not None:
                if not self._running.done():
                    raise TemporalProviderError("backend_previous_call_still_running")
                # Consume a previously timed-out task's exception without
                # printing values, then allow a new explicit request.
                try:
                    self._running.result()
                except BaseException:
                    pass
                self._running = None
            self._running = asyncio.create_task(asyncio.to_thread(function))
            current = self._running
        async with asyncio.timeout_at(deadline):
            return await asyncio.shield(current)

    async def check_ready(self, *, deadline: float) -> dict[str, Any]:
        _checkpoint(deadline, None)
        if self._closed:
            raise TemporalProviderError("backend_closed")
        if self._model is None:
            await self._run(self._load_sync, deadline=deadline)
        return {"ready": True, "head_sha256": self.identity.head_sha256,
                "feature_spec_sha256": self.identity.feature_spec_sha256}

    def _predict_sync(self, request: ClipRequest) -> HeadBackendResult:
        from training.temporal.checkpoint import sequence_inputs
        import torch

        started = time.monotonic()
        indices = bind_feature_indices(self._sequence, request)
        if tuple(fact.fact_id for fact in request.fact_definitions) != self.identity.fact_ids:
            raise TemporalProviderError("request_fact_axis_mismatch")
        try:
            inputs = sequence_inputs(self._sequence, self._features, device=self.device, indices=indices)
            with torch.inference_mode():
                tensors = self._model(**inputs)

            def tensor(name: str, shape: tuple[int, ...]):
                value = tensors.get(name)
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                        or not bool(torch.isfinite(value).all())):
                    raise TemporalProviderError("head_tensor_shape_or_finiteness_invalid")
                return value.detach().to(device="cpu", dtype=torch.float32).tolist()[0]

            count, facts, events = len(indices), len(self.identity.fact_ids), len(self.identity.event_type_ids)
            boundary = tensor("boundary_logits", (1, count, events, 2)) if events else None
            native = [self._sequence.frames[index] for index in indices]
            output = TemporalHeadOutput.model_validate({
                "identity": self.identity, "job_id": request.job_id, "clip_id": request.clip_id,
                "frames": [{"frame_id": frame.frame_id, "pts_sec": frame.pts_sec,
                    "source_pts": frame.source_pts, "next_source_pts": frame.next_source_pts} for frame in native],
                "window_logits": tensor("window_logits", (1, facts, 3)),
                "frame_logits": tensor("frame_logits", (1, count, facts, 3)),
                "support_logits": tensor("support_logits", (1, count, facts)),
                "frame_valid": [any(frame.view_mask) for frame in native],
                "global_view_available": [any(valid and role == "global" for valid, role in
                    zip(frame.view_mask, frame.view_roles)) for frame in native],
                "feature_coverage_complete": native_window_coverage(self._sequence, request, indices),
                "native_source_continuity_verified": native_window_coverage(self._sequence, request, indices),
                "trained_components": self._trained_components,
                "boundary_logits": boundary,
            }, strict=False)
            # This score is a video-binary prioritization signal only. It does
            # not enter the observation mapper and never labels facts absent.
            video_score = tensor("video_logit", (1,))
            if not isinstance(video_score, float) or not math.isfinite(video_score):
                raise TemporalProviderError("head_video_score_invalid")
            return HeadBackendResult(output, {"elapsed_sec": time.monotonic() - started,
                "input_frames": count, "generated_tokens": 0})
        except TemporalProviderError:
            raise
        except Exception:
            raise TemporalProviderError("head_execution_failed") from None

    async def predict(self, request: ClipRequest, *, deadline: float,
                      cancel_event: Any = None) -> HeadBackendResult:
        _checkpoint(deadline, cancel_event)
        if len(request.frame_refs) > 96:
            raise TemporalProviderError("bounded_observation_reference_budget_exceeded")
        await self.check_ready(deadline=deadline)
        immutable_request = ClipRequest.model_validate_json(request.model_dump_json())
        result = await self._run(lambda: self._predict_sync(immutable_request), deadline=deadline)
        _checkpoint(deadline, cancel_event)
        return result

    async def aclose(self) -> None:
        self._closed = True
        if self._running is not None:
            try:
                await asyncio.shield(self._running)
            except BaseException:
                pass
        self._model = self._features = self._sequence = self._cache_manifest = None
