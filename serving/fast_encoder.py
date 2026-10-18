"""Optional offline SigLIP vision features and generated-only CUDA benchmark.

Importing this module imports no ML/image libraries and initializes no device.
Loading and benchmarking are explicit actions for a coordinated GPU window.
Pixels and feature values are never included in diagnostics or object reprs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any


class FastEncoderError(RuntimeError):
    """Only static, content-independent error codes belong in this exception."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FastEncoderConfig:
    snapshot: Path
    manifest: Path
    manifest_sha256: str
    batch_size: int = 32
    device: str = "cuda:0"
    preprocessing_mode: str = "official_processor"
    input_transform_sha256: str | None = None
    max_input_bytes: int = 256 * 1024 * 1024
    expected_torch: str = "2.13.0+cu129"
    expected_transformers: str = "5.17.0"
    expected_torchvision: str = "0.28.0+cu129"

    def __post_init__(self):
        if not re.fullmatch(r"[a-f0-9]{64}", self.manifest_sha256):
            raise FastEncoderError("encoder_manifest_sha256_required")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 128:
            raise FastEncoderError("encoder_batch_limit_invalid")
        if not re.fullmatch(r"cuda:[0-7]", self.device):
            raise FastEncoderError("encoder_explicit_cuda_device_required")
        if type(self.max_input_bytes) is not int or not 1 <= self.max_input_bytes <= 512 * 1024 * 1024:
            raise FastEncoderError("encoder_input_byte_limit_invalid")
        if self.preprocessing_mode not in {"official_processor", "decoder_rgb_256"}:
            raise FastEncoderError("encoder_preprocessing_mode_invalid")
        if self.preprocessing_mode == "decoder_rgb_256":
            if not re.fullmatch(r"[a-f0-9]{64}", self.input_transform_sha256 or ""):
                raise FastEncoderError("encoder_decoder_transform_identity_required")
        elif self.input_transform_sha256 is not None:
            raise FastEncoderError("encoder_unexpected_decoder_transform_identity")


def _expected_rgb_bytes(shape: tuple[int, int, int, int], config: FastEncoderConfig) -> int:
    if (not isinstance(shape, tuple) or len(shape) != 4
            or any(type(value) is not int for value in shape)):
        raise FastEncoderError("encoder_rgb_shape_invalid")
    batch, height, width, channels = shape
    if not (1 <= batch <= config.batch_size and 1 <= height <= 8192
            and 1 <= width <= 8192 and channels == 3):
        raise FastEncoderError("encoder_rgb_shape_or_batch_limit")
    if config.preprocessing_mode == "decoder_rgb_256" and (height, width) != (256, 256):
        raise FastEncoderError("encoder_decoder_geometry_mismatch")
    expected = batch * height * width * channels
    if expected > config.max_input_bytes:
        raise FastEncoderError("encoder_rgb_byte_or_contiguity_limit")
    return expected


def validate_rgb_buffer(pixels: bytes | memoryview, shape: tuple[int, int, int, int],
                        config: FastEncoderConfig) -> tuple[int, int, int, int]:
    """Check a tight CPU uint8 BHWC RGB buffer before allocating/importing ML."""
    expected = _expected_rgb_bytes(shape, config)
    if not isinstance(pixels, (bytes, memoryview)):
        raise FastEncoderError("encoder_tight_rgb_bytes_required")
    view = memoryview(pixels)
    if (not view.c_contiguous or view.itemsize != 1 or view.format not in {"B", "b", "c"}
            or view.nbytes != expected or expected > config.max_input_bytes):
        raise FastEncoderError("encoder_rgb_byte_or_contiguity_limit")
    return shape


def _checkpoint(deadline: float | None, cancelled: Any) -> None:
    if cancelled is not None and cancelled.is_set():
        raise FastEncoderError("encoder_cancelled")
    if deadline is not None and (not math.isfinite(deadline) or time.monotonic() >= deadline):
        raise FastEncoderError("encoder_deadline_exceeded")


@dataclass(frozen=True)
class EncodedBatch:
    # CPU FP16 features are private; shape/timing metadata alone is printable.
    features: Any = field(repr=False)
    shape: tuple[int, int]
    metrics: dict[str, float]
    encoder_recipe_sha256: str


def encoder_recipe(config: FastEncoderConfig, manifest: dict, versions: dict[str, str]) -> dict:
    processor_entry = next(entry for entry in manifest["files"]
                           if entry["path"] == "preprocessor_config.json")
    return {
        "schema_version": "forte.encoder-recipe.v1",
        "encoder_manifest_sha256": config.manifest_sha256,
        "model_id": manifest["model_id"], "model_revision": manifest["revision"],
        "processor_revision": manifest["processor_revision"],
        "processor_config_sha256": processor_entry["sha256"],
        "model_class": "SiglipVisionModel", "processor_class": "SiglipImageProcessor",
        "processor_backend": "torchvision_cpu", "packages": dict(versions),
        "input_layout": "BHWC", "input_dtype": "uint8", "input_color": "RGB",
        "preprocessing_mode": config.preprocessing_mode,
        "input_transform_sha256": config.input_transform_sha256,
        "processor_do_resize": config.preprocessing_mode == "official_processor",
        "processor_target_size": [256, 256], "resample_pil_enum": 2,
        "rescale_factor": 1 / 255, "mean": [.5, .5, .5], "std": [.5, .5, .5],
        "attention": "sdpa", "model_dtype": "bfloat16", "inference_mode": True,
        "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False, "cudnn_deterministic": True, "allow_tf32": False,
        "seed": 20261017,
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("fast_encoder.py", "feature_stream.py")
        },
        "feature_kind": "pooler_output", "feature_dim": 768,
        "l2_normalization": False, "output_layout": "BD", "output_dtype": "float16",
    }


class FastEncoder:
    """One explicitly loaded vision-only CUDA model; no baseline registration.

    ``encode_rgb_bytes`` also accepts A05 RGBFrameBatch.pixels/.shape when the
    config declares decoder_rgb_256 and binds that exact decoder-transform hash.
    The caller owns source IDs/PTS/view geometry and builds A09's full feature
    spec. This class has no source-label, filename or verdict input.
    """

    def __init__(self, config: FastEncoderConfig):
        self.config = config
        self._lock = threading.Lock()
        self._torch = self._numpy = self._model = self._processor = None
        self._recipe = self._verification = None
        self.load_seconds: float | None = None

    @property
    def recipe(self) -> dict:
        if self._recipe is None:
            raise FastEncoderError("encoder_not_loaded")
        return json.loads(json.dumps(self._recipe))

    @property
    def recipe_sha256(self) -> str:
        return canonical_sha256(self.recipe)

    def load(self) -> dict:
        with self._lock:
            if self._model is not None:
                return dict(self._verification)
            started = time.perf_counter()
            # Full local hashes must pass before any model/processor/device load.
            from forte.model.encoder_staging import load_encoder_manifest, verify_encoder_snapshot
            manifest = load_encoder_manifest(self.config.manifest,
                expected_manifest_sha256=self.config.manifest_sha256)
            verification = verify_encoder_snapshot(self.config.snapshot, self.config.manifest,
                expected_manifest_sha256=self.config.manifest_sha256)
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
                os.environ[name] = "1"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                versions = {name: importlib.metadata.version(name) for name in
                            ("torch", "torchvision", "transformers", "pillow", "numpy")}
                if any(versions[key] != expected for key, expected in (
                    ("torch", self.config.expected_torch),
                    ("torchvision", self.config.expected_torchvision),
                    ("transformers", self.config.expected_transformers))):
                    raise FastEncoderError("encoder_runtime_version_mismatch")
                import numpy as np
                import torch
                from transformers import SiglipImageProcessor, SiglipVisionModel
                from transformers.utils import logging as transformer_logging
                transformer_logging.disable_progress_bar()
                transformer_logging.set_verbosity_error()
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.manual_seed(20261017)
                if not torch.cuda.is_available():
                    raise FastEncoderError("encoder_cuda_unavailable")
                with torch.cuda.device(self.config.device):
                    if not torch.cuda.is_bf16_supported():
                        raise FastEncoderError("encoder_cuda_bfloat16_unavailable")
                    processor = SiglipImageProcessor.from_pretrained(str(self.config.snapshot),
                        local_files_only=True)
                    if (processor.size != {"height": 256, "width": 256}
                            or list(processor.image_mean) != [.5, .5, .5]
                            or list(processor.image_std) != [.5, .5, .5]
                            or int(processor.resample) != 2
                            or not processor.do_resize or not processor.do_rescale or not processor.do_normalize
                            or not math.isclose(processor.rescale_factor, 1 / 255, rel_tol=0, abs_tol=1e-16)):
                        raise FastEncoderError("encoder_effective_processor_mismatch")
                    if {path.name for path in Path(self.config.snapshot).iterdir()} != {
                        entry["path"] for entry in manifest["files"]}:
                        raise FastEncoderError("encoder_unexpected_snapshot_artifact")
                    model, loading = SiglipVisionModel.from_pretrained(str(self.config.snapshot),
                        local_files_only=True, use_safetensors=True,
                        dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
                    ignored = loading.get("unexpected_keys", [])
                    if (loading.get("missing_keys") or loading.get("mismatched_keys")
                            or loading.get("error_msgs") or any(not (
                                key.startswith("text_model.") or key in {"logit_scale", "logit_bias"}
                            ) for key in ignored)):
                        raise FastEncoderError("encoder_vision_weights_not_fully_loaded")
                    if (model.config.hidden_size != 768 or model.config.image_size != 256
                            or (256 // model.config.patch_size) ** 2 != manifest["loading_contract"]["patch_tokens"]):
                        raise FastEncoderError("encoder_model_shape_mismatch")
                    model = model.eval().requires_grad_(False).to(self.config.device)
                    torch.cuda.synchronize(self.config.device)
                self._numpy, self._torch = np, torch
                self._processor, self._model = processor, model
                self._recipe = encoder_recipe(self.config, manifest, versions)
                self._verification = {**verification, "local_model_loaded": True,
                    "model_class": type(model).__name__, "processor_class": type(processor).__name__,
                    "encoder_recipe_sha256": canonical_sha256(self._recipe),
                    "packages": versions, "selected_for_production": False,
                    "ignored_text_tower_keys": len(ignored),
                    "gpu_name": torch.cuda.get_device_name(self.config.device),
                    "gpu_compute_capability": list(torch.cuda.get_device_capability(self.config.device))}
                self.load_seconds = time.perf_counter() - started
                return dict(self._verification)
            except FastEncoderError:
                raise
            except Exception as error:
                failure = FastEncoderError("encoder_optional_runtime_or_local_load_failed")
                failure.safe_exception_type = type(error).__name__
                raise failure from None

    def encode_rgb_bytes(self, pixels: bytes | memoryview, shape: tuple[int, int, int, int], *,
                         input_transform_sha256: str | None = None,
                         deadline: float | None = None, cancel_event: Any = None) -> EncodedBatch:
        validate_rgb_buffer(pixels, shape, self.config)
        if input_transform_sha256 != self.config.input_transform_sha256:
            raise FastEncoderError("encoder_batch_transform_identity_mismatch")
        _checkpoint(deadline, cancel_event)
        with self._lock:
            if self._model is None:
                raise FastEncoderError("encoder_not_loaded")
            torch, np = self._torch, self._numpy
            try:
                started = time.perf_counter()
                # Own a bounded writable copy; processor tensor conversion must
                # not expose an immutable caller buffer to in-place operations.
                images = np.frombuffer(pixels, dtype=np.uint8).reshape(shape).copy()
                inputs = self._processor(images=list(images), return_tensors="pt", device="cpu",
                    do_resize=self.config.preprocessing_mode == "official_processor")
                values = inputs["pixel_values"]
                if tuple(values.shape) != (shape[0], 3, 256, 256):
                    raise FastEncoderError("encoder_processed_tensor_shape_invalid")
                preprocessed = time.perf_counter()
                _checkpoint(deadline, cancel_event)
                torch.cuda.synchronize(self.config.device)
                device_started = time.perf_counter()
                values = values.to(device=self.config.device, dtype=torch.bfloat16)
                forward_start, forward_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.inference_mode(), torch.cuda.device(self.config.device):
                    forward_start.record()
                    output = self._model(pixel_values=values).pooler_output
                    forward_end.record()
                    if output is None or tuple(output.shape) != (shape[0], 768):
                        raise FastEncoderError("encoder_feature_shape_invalid")
                    features = output.detach().to(device="cpu", dtype=torch.float16)
                torch.cuda.synchronize(self.config.device)
                completed = time.perf_counter()
                if not bool(torch.isfinite(features).all().item()):
                    raise FastEncoderError("encoder_nonfinite_features")
                _checkpoint(deadline, cancel_event)
                return EncodedBatch(features, (shape[0], 768), {
                    "elapsed_sec": completed - started,
                    "preprocessing_sec": preprocessed - started,
                    "h2d_forward_d2h_sec": completed - device_started,
                    "cuda_forward_sec": forward_start.elapsed_time(forward_end) / 1000,
                }, self.recipe_sha256)
            except FastEncoderError:
                raise
            except Exception:
                raise FastEncoderError("encoder_batch_execution_failed") from None

    def encode_frame_batch(self, batch: Any, *, deadline: float | None = None,
                           cancel_event: Any = None) -> EncodedBatch:
        """Consume A05's RGBFrameBatch without importing its decoder module."""
        if self.config.preprocessing_mode != "decoder_rgb_256":
            raise FastEncoderError("encoder_decoder_preprocessing_mode_required")
        return self.encode_rgb_bytes(batch.pixels, batch.shape,
            input_transform_sha256=batch.transform_sha256,
            deadline=deadline, cancel_event=cancel_event)

    def memory_stats(self, *, reset_peak: bool = False) -> dict:
        if self._model is None:
            raise FastEncoderError("encoder_not_loaded")
        cuda = self._torch.cuda
        cuda.synchronize(self.config.device)
        if reset_peak:
            cuda.reset_peak_memory_stats(self.config.device)
        free, total = cuda.mem_get_info(self.config.device)
        return {"device_free_bytes": free, "device_total_bytes": total,
                "process_allocated_bytes": cuda.memory_allocated(self.config.device),
                "process_reserved_bytes": cuda.memory_reserved(self.config.device),
                "process_peak_allocated_bytes": cuda.max_memory_allocated(self.config.device),
                "process_peak_reserved_bytes": cuda.max_memory_reserved(self.config.device)}

    def close(self) -> None:
        with self._lock:
            self._model = self._processor = None
            self._recipe = self._verification = None
            if self._torch is not None:
                self._torch.cuda.synchronize(self.config.device)
                self._torch.cuda.empty_cache()


def _synthetic_rgb(np: Any, count: int, width: int, height: int, offset: int) -> bytes:
    # Bounded generated content only; accepts no paths, URLs or existing media.
    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]
    base = rows + columns
    output = np.empty((count, height, width, 3), dtype=np.uint8)
    for index in range(count):
        output[index, :, :, 0] = (base + offset + index) % 256
        output[index, :, :, 1] = (base * 3 + index) % 256
        output[index, :, :, 2] = (base * 7 + offset) % 256
    return output.tobytes()


def benchmark(args: Any) -> dict:
    config = FastEncoderConfig(Path(args.snapshot), Path(args.manifest), args.manifest_sha256,
        batch_size=args.batch_size, device=args.device)
    # Validate size before generating/allocating any synthetic pixel buffer.
    _expected_rgb_bytes((args.batch_size, args.input_height, args.input_width, 3), config)
    runtime = FastEncoder(config)
    try:
        verification = runtime.load()
        np = runtime._numpy
        for index in range(args.warmup_batches):
            payload = _synthetic_rgb(np, args.batch_size, args.input_width, args.input_height, index)
            batch = runtime.encode_rgb_bytes(payload,
                (args.batch_size, args.input_height, args.input_width, 3))
            del batch, payload
        trials = []
        for trial in range(args.trials):
            before = runtime.memory_stats(reset_peak=True)
            started = time.perf_counter()
            sums = {"synthetic_generation_sec": 0.0, "preprocessing_sec": 0.0,
                    "h2d_forward_d2h_sec": 0.0, "cuda_forward_sec": 0.0}
            batches = 0
            for offset in range(0, args.frames, args.batch_size):
                count = min(args.batch_size, args.frames - offset)
                generation_started = time.perf_counter()
                payload = _synthetic_rgb(np, count, args.input_width, args.input_height,
                    offset + trial * args.frames + 1000)
                sums["synthetic_generation_sec"] += time.perf_counter() - generation_started
                batch = runtime.encode_rgb_bytes(payload, (count, args.input_height, args.input_width, 3))
                for key in batch.metrics:
                    if key in sums:
                        sums[key] += batch.metrics[key]
                batches += 1
                del batch, payload
            after = runtime.memory_stats()
            elapsed = time.perf_counter() - started
            trials.append({"frames": args.frames, "batches": batches, "wall_sec": elapsed,
                "frames_per_second": args.frames / elapsed, **sums,
                "memory_before": before, "memory_after": after,
                "all_output_shapes_and_finite_values_validated": True})
        return {"recorded_at": datetime.now(timezone.utc).isoformat(),
            "synthetic_only": True, "private_media_opened": False, "encoder_only": True,
            "synthetic_encoder_inference_verified": True,
            "upload_decode_head_evidence_measured": False, "event_accuracy_measured": False,
            "selected_for_production": False, "verification": verification,
            "encoder_recipe": runtime.recipe, "encoder_recipe_sha256": runtime.recipe_sha256,
            "load_and_full_verification_sec": runtime.load_seconds,
            "warmup_batches": args.warmup_batches, "batch_size": args.batch_size,
            "input_shape": ["batch", args.input_height, args.input_width, 3],
            "output_shape": ["batch", 768], "trials": trials}
    finally:
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=4273)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=2)
    args = parser.parse_args()
    if not (1 <= args.frames <= 10000 and 1 <= args.trials <= 3
            and 1 <= args.warmup_batches <= 4 and 32 <= args.input_width <= 2048
            and 32 <= args.input_height <= 2048):
        parser.error("invalid_bounded_synthetic_workload")
    try:
        print(json.dumps(benchmark(args), allow_nan=False))
        return 0
    except Exception as error:
        safe = isinstance(error, FastEncoderError) or type(error).__module__ == "forte.model.encoder_staging"
        code = str(error) if safe else "encoder_benchmark_failed"
        print(json.dumps({"synthetic_only": True, "benchmark_completed": False, "error_code": code,
                          "error_type": type(error).__name__,
                          "runtime_cause_type": getattr(error, "safe_exception_type", None)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
