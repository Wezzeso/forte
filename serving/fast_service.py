"""Optional private, warm native-video encoder/head service.

No ML, decoder, web framework or model is imported at module import. This is
an isolated candidate backend, never a source-name lookup or legal decision
publisher. The CPU provider separately verifies fact calibration. Start only
with explicit pinned artifacts in the separate feature runtime.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import platform
import threading
import time
from typing import Any


class FastServiceError(RuntimeError):
    """Static diagnostics only; never embed a path or model/decoder exception."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sidecar_packages() -> dict[str, str | None]:
    versions = {}
    for name in ("av", "safetensors", "pydantic", "uvicorn", "httpx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _checkpoint(deadline: float, cancel: threading.Event) -> None:
    if cancel.is_set():
        raise FastServiceError("native_request_cancelled")
    if time.monotonic() >= deadline:
        raise FastServiceError("native_request_deadline_exceeded")


def source_code_bundle(root: Path | None = None) -> dict[str, Any]:
    """Hash the complete packaged runtime Python source, without private data."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = {}
    for directory in ("src", "serving", "training"):
        for path in sorted((root / directory).rglob("*.py")):
            relative = path.relative_to(root)
            if any(part in {"private", "__pycache__", ".venv", "runs"} for part in relative.parts):
                continue
            if path.is_symlink():
                raise FastServiceError("native_source_code_symlink_invalid")
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files or not all(any(name.startswith(prefix) for name in files) for prefix in
                            ("src/", "serving/", "training/")):
        raise FastServiceError("native_source_code_bundle_incomplete")
    return {"schema_version": "forte.packaged-python-source.v1", "files": files}


@dataclass(frozen=True)
class FastServiceConfig:
    source_root: Path
    snapshot: Path
    encoder_manifest: Path
    encoder_manifest_sha256: str
    feature_spec: Path
    feature_spec_sha256: str
    head_identity: Path
    head_identity_sha256: str
    checkpoint_dir: Path
    checkpoint_manifest_sha256: str
    source_code_bundle_sha256: str
    batch_size: int = 32
    device: str = "cuda:0"
    decoder_threads: int = 4
    prefetch_batches: int = 2
    max_source_bytes: int = 256 * 1024 * 1024
    max_source_frames: int = 10000
    max_duration_sec: float = 143.0
    max_budget_sec: float = 120.0

    def __post_init__(self):
        if (any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in (
                self.encoder_manifest_sha256, self.feature_spec_sha256,
                self.head_identity_sha256, self.checkpoint_manifest_sha256, self.source_code_bundle_sha256))
                or type(self.batch_size) is not int or self.batch_size != 32
                or type(self.decoder_threads) is not int or not 1 <= self.decoder_threads <= 8
                or type(self.prefetch_batches) is not int or not 1 <= self.prefetch_batches <= 2
                or type(self.max_source_bytes) is not int or not 0 < self.max_source_bytes <= 256 * 1024 * 1024
                or type(self.max_source_frames) is not int or not 0 < self.max_source_frames <= 10000
                or not math.isfinite(self.max_duration_sec) or not 0 < self.max_duration_sec <= 143
                or not math.isfinite(self.max_budget_sec) or not 0 < self.max_budget_sec <= 120
                or not re.fullmatch(r"cuda:[0-7]", self.device)):
            raise FastServiceError("native_service_configuration_invalid")


def _read_json(path: Path, *, file_sha256: str | None = None) -> Any:
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 1024 * 1024 or file_sha256 and hashlib.sha256(raw).hexdigest() != file_sha256:
            raise FastServiceError("native_artifact_identity_invalid")
        return json.loads(raw)
    except FastServiceError:
        raise
    except Exception:
        raise FastServiceError("native_artifact_unavailable") from None


def verify_source(asset: Any, config: FastServiceConfig, *, deadline: float,
                  cancel_event: threading.Event) -> tuple[Path, tuple[int, int, int, int]]:
    """Permit the API's exact uploads/<job_id>.mov layout, never client paths.

    The shared upload mount must be read-only in this service. Size/stat checks
    supplement the full content digest; the decoder independently verifies it.
    """
    try:
        _checkpoint(deadline, cancel_event)
        if (not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", asset.job_id)
                or not 0 < asset.duration_sec < config.max_duration_sec
                or asset.frame_count is not None and asset.frame_count > config.max_source_frames
                or asset.width * asset.height > 33554432):
            raise FastServiceError("native_source_budget_invalid")
        root = Path(config.source_root).absolute()
        if root.resolve(strict=True) != root or not root.is_dir():
            raise FastServiceError("native_source_root_invalid")
        expected = root / (asset.job_id + ".mov")
        supplied = Path(asset.owned_path)
        if (supplied != expected or not supplied.is_absolute() or supplied.is_symlink()
                or supplied.resolve(strict=True) != expected or not supplied.is_file()):
            raise FastServiceError("native_source_path_not_owned")
        before = supplied.stat()
        if not 0 < before.st_size <= config.max_source_bytes:
            raise FastServiceError("native_source_size_invalid")
        digest = hashlib.sha256()
        with supplied.open("rb") as source:
            while data := source.read(1024 * 1024):
                _checkpoint(deadline, cancel_event)
                digest.update(data)
        fingerprint = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after = supplied.stat()
        if (fingerprint != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or digest.hexdigest() != asset.content_hash):
            raise FastServiceError("native_source_integrity_failed")
        return supplied, fingerprint
    except FastServiceError:
        raise
    except Exception:
        raise FastServiceError("native_source_unavailable") from None


class NativeFeatureBackend:
    """One explicitly loaded encoder and real head, serialized by the ASGI app."""
    def __init__(self, config: FastServiceConfig):
        self.config = config
        self.encoder = self.runner = self.identity = None
        self.execution_manifest: dict[str, Any] | None = None
        self._ready = False

    def load(self) -> dict[str, Any]:
        if self._ready:
            return self.check_ready()
        from forte.decision.head_contract import HeadIdentity
        from forte.decision.temporal_native import NativeHeadRunner
        from forte.video.batches import rgb_transform_recipe
        from serving.fast_encoder import FastEncoder, FastEncoderConfig
        from training.temporal.feature_spec import build_feature_spec

        code = source_code_bundle()
        if _canonical_sha256(code) != self.config.source_code_bundle_sha256:
            raise FastServiceError("native_source_code_bundle_mismatch")
        identity = HeadIdentity.model_validate(_read_json(self.config.head_identity,
            file_sha256=self.config.head_identity_sha256), strict=False)
        spec = _read_json(self.config.feature_spec)
        if (_canonical_sha256(spec) != self.config.feature_spec_sha256
                or identity.feature_spec_sha256 != self.config.feature_spec_sha256
                or identity.encoder_artifacts_sha256 != self.config.encoder_manifest_sha256):
            raise FastServiceError("native_head_feature_identity_mismatch")
        decoder = rgb_transform_recipe(target_size=(256, 256), aspect_policy="letterbox")
        sampling = {"mode": "native", "all_native_frames": True, "tail_required": True,
                    "time": "original_integer_pts_and_canonical_rational_base"}
        crop = {"views": "global_only", "source_box_xyxy": [0., 0., 1., 1.]}
        encoder = FastEncoder(FastEncoderConfig(self.config.snapshot, self.config.encoder_manifest,
            self.config.encoder_manifest_sha256, batch_size=self.config.batch_size,
            device=self.config.device, preprocessing_mode="decoder_rgb_256",
            input_transform_sha256=decoder["sha256"]))
        self.encoder = encoder
        try:
            verification = encoder.load()
            actual, actual_sha = build_feature_spec(encoder.recipe,
                decoder_recipe_sha256=_canonical_sha256(decoder),
                sampling_recipe_sha256=_canonical_sha256(sampling), crop_recipe_sha256=_canonical_sha256(crop))
            if actual_sha != self.config.feature_spec_sha256 or actual != spec:
                raise FastServiceError("native_executed_feature_recipe_mismatch")
            runner = NativeHeadRunner(self.config.checkpoint_dir,
                checkpoint_manifest_sha256=self.config.checkpoint_manifest_sha256,
                identity=identity, device=self.config.device)
            if runner.load().get("ready") is not True:
                raise FastServiceError("native_head_not_ready")
            execution = {
                "schema_version": "forte.native-execution.v1",
                "source_code_bundle_sha256": self.config.source_code_bundle_sha256,
                "encoder_manifest_sha256": self.config.encoder_manifest_sha256,
                "feature_spec_sha256": self.config.feature_spec_sha256,
                "encoder_recipe_sha256": _canonical_sha256(encoder.recipe),
                "head_sha256": identity.head_sha256,
                "checkpoint_manifest_sha256": self.config.checkpoint_manifest_sha256,
                "head_identity_file_sha256": self.config.head_identity_sha256,
                "device": self.config.device, "gpu_name": verification.get("gpu_name"),
                "gpu_compute_capability": verification.get("gpu_compute_capability"),
                "batch_size": self.config.batch_size, "decoder_threads": self.config.decoder_threads,
                "decoder_thread_type": "FRAME", "prefetch_batches": self.config.prefetch_batches,
                "decoder_subprocess": True, "max_batch_bytes": 32 * 1024 * 1024,
                "source_policy": "all_native_frames_and_original_tail",
                "views": "global_only", "roi_enabled": False,
                "target_size": [256, 256], "aspect_policy": "letterbox",
                "max_source_bytes": self.config.max_source_bytes,
                "max_source_frames": self.config.max_source_frames,
                "max_duration_sec_exclusive": self.config.max_duration_sec,
                "max_source_pixels": 33554432, "max_budget_sec": self.config.max_budget_sec,
                "max_request_bytes": 64 * 1024, "max_response_bytes": 64 * 1024 * 1024,
                "max_inflight_requests": 1, "hidden_request_queue": False,
                "determinism": {key: encoder.recipe.get(key) for key in (
                    "deterministic_algorithms", "cublas_workspace_config", "cudnn_benchmark",
                    "cudnn_deterministic", "allow_tf32", "seed", "attention", "model_dtype")},
                "runtime_packages": encoder.recipe.get("packages"),
                "sidecar_packages": _sidecar_packages(),
                "python_version": platform.python_version(),
                "repeatability_measured": False, "candidate_calibration_verified": False}
            _canonical_sha256(execution)  # Reject non-JSON runtime attestation before readiness.
            self.execution_manifest = execution
            self.runner, self.identity, self._ready = runner, identity, True
            return self.check_ready()
        except Exception:
            self._ready, self.runner, self.identity, self.execution_manifest = False, None, None, None
            encoder.close()
            raise FastServiceError("native_pinned_runtime_load_failed") from None

    def check_ready(self) -> dict[str, Any]:
        runner_ready = self.runner.check_ready() if self.runner is not None else {}
        return {"ready": self._ready and runner_ready.get("ready") is True,
            "head_sha256": self.identity.head_sha256 if self.identity is not None else None,
            "feature_spec_sha256": self.config.feature_spec_sha256,
            "encoder_manifest_sha256": self.config.encoder_manifest_sha256,
            "source_code_bundle_sha256": self.config.source_code_bundle_sha256,
            "execution_manifest_sha256": _canonical_sha256(self.execution_manifest) if self.execution_manifest else None,
            "batch_size": self.config.batch_size,
            "candidate_calibration_verified": False}

    def predict(self, asset: Any, *, clip_id: str, deadline: float,
                cancel_event: threading.Event) -> Any:
        if not self.check_ready()["ready"]:
            raise FastServiceError("native_backend_not_ready")
        from forte.decision.temporal_native import NativeFeatureFrame, NativeHeadResult, NativeSourceContext
        from forte.video.batches import iter_rgb_batches
        from serving.feature_stream import iter_encoded_batches
        import torch

        started = time.monotonic()
        source, fingerprint = verify_source(asset, self.config, deadline=deadline, cancel_event=cancel_event)
        verified = time.monotonic()

        def source_factory(cancel):
            return iter_rgb_batches(asset, sample_fps=None, cancel=cancel, use_subprocess=True,
                verify_source_hash=True, batch_size=self.config.batch_size, target_size=(256, 256),
                aspect_policy="letterbox", deadline_monotonic=deadline,
                max_runtime_sec=self.config.max_budget_sec, max_source_frames=self.config.max_source_frames,
                max_output_frames=self.config.max_source_frames, thread_count=self.config.decoder_threads,
                max_pixels=33554432, max_batch_bytes=32 * 1024 * 1024)

        frames, chunks, encoded_sec, origin = [], [], 0.0, None
        with closing(iter_encoded_batches(source_factory, self.encoder, deadline=deadline,
                cancel_event=cancel_event, max_source_frames=self.config.max_source_frames,
                prefetch_batches=self.config.prefetch_batches)) as batches:
            for batch in batches:
                _checkpoint(deadline, cancel_event)
                if batch.views_per_frame != 1:
                    raise FastServiceError("native_global_only_feature_recipe_required")
                chunks.append(batch.features)
                encoded_sec += batch.metrics["elapsed_sec"]
                for frame, transform in zip(batch.frames, batch.transforms):
                    if (frame.job_id != asset.job_id or frame.stream_index != asset.stream_index
                            or transform.view_id != "global"):
                        raise FastServiceError("native_decoded_source_binding_failed")
                    if origin is None:
                        origin = frame.source_origin_pts
                    frames.append(NativeFeatureFrame(frame_id=frame.frame_id, source_pts=frame.source_pts,
                        next_source_pts=frame.next_source_pts, pts_sec=float(frame.pts_sec),
                        next_pts_sec=frame.next_pts_sec, source_end_bound_sec=frame.source_end_bound_sec,
                        width=frame.width, height=frame.height, applied_rotation_deg=frame.applied_rotation_deg,
                        key_frame=frame.key_frame, view_mask=(True,), view_roles=("global",),
                        normalized_boxes=(tuple(float(value) for value in transform.source_box_xyxy),)))
        _checkpoint(deadline, cancel_event)
        if not frames:
            raise FastServiceError("native_source_empty")
        after = source.stat()
        if fingerprint != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise FastServiceError("native_source_changed_during_inference")
        base = Fraction(asset.time_base)
        context = NativeSourceContext(job_id=asset.job_id, clip_id=clip_id,
            content_sha256=asset.content_hash, duration_sec=float(asset.duration_sec),
            time_base_num=base.numerator, time_base_den=base.denominator, origin_pts=origin,
            stream_index=asset.stream_index, feature_spec_sha256=self.config.feature_spec_sha256,
            encoder_manifest_sha256=self.config.encoder_manifest_sha256,
            source_complete=True, tail_included=True, frames=tuple(frames))
        context.validate_asset(asset, self.identity)
        features = torch.cat(chunks, dim=0).unsqueeze(1)
        del chunks
        result = self.runner.predict_native(context, features, deadline=deadline, cancel_event=cancel_event)
        _checkpoint(deadline, cancel_event)
        # Decode and encode overlap; report measured encoder wall sum rather
        # than pretending the remaining wall time is pure decoder CPU time.
        return NativeHeadResult(context=result.context, output=result.output, video_logit=result.video_logit,
            metrics={**result.metrics, "source_verification_sec": verified - started,
                "encoder_elapsed_sec": encoded_sec, "total_elapsed_sec": time.monotonic() - started})

    def close(self):
        self._ready = False
        self.runner = None
        if self.encoder is not None:
            self.encoder.close()
        self.encoder = None


class NativeServiceApp:
    """Small ASGI boundary with one GPU admission and retained cancellation.

    A disconnected/timed-out caller signals the native decoder, but the GPU
    slot remains occupied until the actual worker thread and decoder drain.
    No request body, native result, source path or exception text is logged.
    """
    def __init__(self, backend: Any, *, max_body_bytes: int = 64 * 1024,
                 max_response_bytes: int = 64 * 1024 * 1024, max_budget_sec: float = 120.):
        if (not 1 <= max_body_bytes <= 64 * 1024 or not 1 <= max_response_bytes <= 64 * 1024 * 1024
                or not math.isfinite(max_budget_sec) or not 0 < max_budget_sec <= 120):
            raise FastServiceError("native_http_limits_invalid")
        self.backend = backend
        self.max_body_bytes, self.max_response_bytes, self.max_budget_sec = max_body_bytes, max_response_bytes, max_budget_sec
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forte-native")
        self._active: dict[asyncio.Future, threading.Event] = {}
        self._busy = False

    async def _json(self, send, status: int, value: Any):
        raw = value if isinstance(value, bytes) else json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        await send({"type": "http.response.start", "status": status,
            "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store"),
                        (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw})

    def _done(self, future):
        self._active.pop(future, None)
        self._busy = False
        if not future.cancelled():
            future.exception()  # Retain/consume worker failures even after disconnect.

    def _execute(self, asset, clip_id, deadline, cancel):
        result = self.backend.predict(asset, clip_id=clip_id, deadline=deadline, cancel_event=cancel)
        raw = (result.model_dump_json() if hasattr(result, "model_dump_json") else
               json.dumps(result, separators=(",", ":"), allow_nan=False)).encode()
        _checkpoint(deadline, cancel)
        if len(raw) > self.max_response_bytes:
            raise FastServiceError("native_result_size_exceeded")
        return raw

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        path, method = scope.get("path"), scope.get("method")
        if method == "GET" and path == "/health":
            await self._json(send, 200, {"alive": True})
            return
        if method == "GET" and path == "/ready":
            ready = self.backend.check_ready()
            await self._json(send, 200 if ready.get("ready") is True else 503, ready)
            return
        if method == "GET" and path == "/execution-manifest":
            manifest = getattr(self.backend, "execution_manifest", None)
            await self._json(send, 200 if manifest else 503,
                manifest if manifest else {"error": "native_execution_manifest_not_ready"})
            return
        if method != "POST" or path != "/v1/temporal/predict":
            await self._json(send, 404, {"error": "native_route_not_found"})
            return
        started = time.monotonic()
        try:
            body = bytearray()
            async with asyncio.timeout(2):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    if message["type"] != "http.request":
                        raise FastServiceError("native_request_invalid")
                    piece = message.get("body", b"")
                    if len(body) + len(piece) > self.max_body_bytes:
                        await self._json(send, 413, {"error": "native_request_size_exceeded"})
                        return
                    body.extend(piece)
                    if not message.get("more_body", False):
                        break
            request = json.loads(body)
            if (not isinstance(request, dict) or set(request) - {"asset", "deadline_budget_sec", "clip_id"}
                    or not {"asset", "deadline_budget_sec"} <= set(request)):
                raise FastServiceError("native_request_invalid")
            budget = request["deadline_budget_sec"]
            if type(budget) not in (float, int) or not math.isfinite(budget) or not 0 < budget <= self.max_budget_sec:
                raise FastServiceError("native_request_budget_invalid")
            from forte.shared.schemas import VideoAsset
            asset = VideoAsset.model_validate(request["asset"])
            clip_id = request.get("clip_id", asset.job_id + "-native")
            if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", clip_id):
                raise FastServiceError("native_clip_identity_invalid")
            deadline = started + budget
            if deadline <= time.monotonic():
                await self._json(send, 504, {"error": "native_request_deadline_exceeded"})
                return
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._json(send, 400, {"error": "native_request_invalid"})
            return
        if self.backend.check_ready().get("ready") is not True:
            await self._json(send, 503, {"error": "native_backend_not_ready"})
            return
        # No await between checking and setting; this is one event-loop admission.
        if self._busy:
            await self._json(send, 503, {"error": "native_backend_busy"})
            return
        self._busy = True
        cancel = threading.Event()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, self._execute, asset, clip_id, deadline, cancel)
        self._active[future] = cancel
        future.add_done_callback(self._done)

        async def disconnected():
            while True:
                if (await receive())["type"] == "http.disconnect":
                    return

        watcher = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait((future, watcher), timeout=max(0, deadline - time.monotonic()),
                                         return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                cancel.set()
                return
            if future not in done:
                cancel.set()
                await self._json(send, 504, {"error": "native_request_deadline_exceeded"})
                return
            try:
                raw = future.result()
            except Exception:
                status = 504 if time.monotonic() >= deadline else 422
                await self._json(send, status, {"error": "native_prediction_failed"})
                return
            await self._json(send, 200, raw)
        finally:
            if not future.done():
                cancel.set()
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    async def _lifespan(self, receive, send):
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await asyncio.get_running_loop().run_in_executor(self._executor, self.backend.load)
                    await send({"type": "lifespan.startup.complete"})
                except Exception:
                    await send({"type": "lifespan.startup.failed", "message": "native_pinned_runtime_load_failed"})
                    return
            elif message["type"] == "lifespan.shutdown":
                for cancel in self._active.values():
                    cancel.set()
                if self._active:
                    await asyncio.wait(tuple(self._active), timeout=5)
                if self._active:
                    await send({"type": "lifespan.shutdown.failed", "message": "native_worker_not_drained"})
                else:
                    await asyncio.get_running_loop().run_in_executor(self._executor, self.backend.close)
                    await send({"type": "lifespan.shutdown.complete"})
                self._executor.shutdown(wait=False, cancel_futures=True)
                return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "snapshot", "encoder-manifest", "feature-spec", "head-identity", "checkpoint-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("encoder-manifest-sha256", "feature-spec-sha256", "head-identity-sha256", "checkpoint-manifest-sha256",
                 "source-code-bundle-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decoder-threads", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = vars(parser.parse_args())
    host, port = args.pop("host"), args.pop("port")
    if not 1024 <= port <= 65535:
        parser.error("port must be between 1024 and 65535")
    try:
        app = NativeServiceApp(NativeFeatureBackend(FastServiceConfig(**args)))
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, access_log=False,
            log_config=None, log_level="critical", limit_concurrency=8,
            h11_max_incomplete_event_size=64 * 1024))
        server.run()
        return 0 if server.started else 2
    except Exception:
        print(json.dumps({"status": "failed", "error": "native_service_start_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
