"""Isolated offline TimeLens2 diagnostic pilot; never a legal verdict provider.

No ML/decoder import or model load occurs at import. Every run freshly decodes
the accepted source, keeps its full native PTS ledger, and presents sparse 2fps
plus the original tail as video. Private responses/tokens are never printed.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

MODEL_ID = "MCG-NJU/TimeLens2-2B"
REVISION = "10a779b6ef6fc8ad41367db2d1eb6779ca84995a"
MANIFEST_SHA256 = "721f00a6cba0705755627f3f139b02cba570f680b03c43e54ab32e5fac2c660f"


class GroundingPilotError(RuntimeError):
    """Static codes only; no private path, prompt, response or exception text."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _checkpoint(deadline: float, cancel_event: Any = None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GroundingPilotError("grounding_cancelled")
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise GroundingPilotError("grounding_deadline_exceeded")


@dataclass(frozen=True)
class GroundingPilotConfig:
    snapshot: Path
    manifest: Path
    manifest_sha256: str = MANIFEST_SHA256
    device: str = "cuda:0"
    target_size: int = 224
    sample_fps: float = 2.0
    max_sampled_frames: int = 66
    max_new_tokens: int = 128
    max_native_frames: int = 10000
    max_source_duration_sec: float = 32.0
    max_source_bytes: int = 256 * 1024 * 1024
    process_memory_fraction: float = .10
    min_free_bytes: int = 9 * 1024**3
    clock_policy: str = "strict_cfr"
    clock_audit_path: Path | None = None
    clock_audit_sha256: str | None = None
    seed: int = 20261017

    def __post_init__(self):
        if (self.manifest_sha256 != MANIFEST_SHA256 or not re.fullmatch(r"cuda:[0-7]", self.device)
                or self.target_size != 224 or self.sample_fps != 2.0
                or type(self.max_sampled_frames) is not int or not 2 <= self.max_sampled_frames <= 66
                or type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 256
                or type(self.max_native_frames) is not int or not 2 <= self.max_native_frames <= 10000
                or not 0 < self.max_source_duration_sec <= 32
                or type(self.max_source_bytes) is not int or not 0 < self.max_source_bytes <= 256 * 1024 * 1024
                or self.process_memory_fraction != .10 or self.min_free_bytes < 9 * 1024**3
                or self.clock_policy not in {"strict_cfr", "native_pts_ticks"}
                or type(self.seed) is not int):
            raise GroundingPilotError("grounding_profile_invalid")
        if (self.clock_audit_path is None) != (self.clock_audit_sha256 is None) or (
                self.clock_audit_sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", self.clock_audit_sha256)):
            raise GroundingPilotError("grounding_native_tick_cpu_audit_required")


@dataclass(frozen=True)
class GroundingAttempt:
    status: str
    error_code: str | None
    raw_result: Any = field(repr=False)
    source_ledger: tuple[Any, ...] = field(repr=False)
    timings: dict[str, float]
    attestation: dict[str, Any]
    gpu_stats: dict[str, int]
    raw_response: str = field(repr=False)
    generated_token_ids: tuple[int, ...] = field(repr=False)
    finish_reason: str

    def private_dict(self) -> dict[str, Any]:
        return {"status": self.status, "error_code": self.error_code,
            "raw_result": self.raw_result.model_dump(mode="json") if self.raw_result is not None else None,
            "source_ledger": [asdict(frame) for frame in self.source_ledger],
            "timings": dict(self.timings), "attestation": self.attestation, "gpu_stats": self.gpu_stats,
            "raw_response": self.raw_response, "generated_token_ids": list(self.generated_token_ids),
            "finish_reason": self.finish_reason}

    def safe_summary(self) -> dict[str, Any]:
        return {"status": self.status, "error_code": self.error_code,
            "native_frames": len(self.source_ledger), "generated_tokens": len(self.generated_token_ids),
            "interval_count": len(self.raw_result.intervals) if self.raw_result is not None else 0,
            "finish_reason": self.finish_reason, "timings": dict(self.timings), "gpu_stats": self.gpu_stats,
            "proposal_only": True, "accuracy_measured": False}


def validate_native_ledger(asset: Any, frames: tuple[Any, ...]) -> None:
    from forte.video.batches import validate_source_identity
    if len(frames) < 2 or frames[0].pts_sec != 0.0:
        raise GroundingPilotError("grounding_complete_native_ledger_required")
    for index, frame in enumerate(frames):
        validate_source_identity(asset, frame)
        if index + 1 < len(frames):
            if frame.next_source_pts != frames[index + 1].source_pts:
                raise GroundingPilotError("grounding_native_ledger_gap")
        elif (frame.next_source_pts is not None or frame.source_end_bound_sec is None
                or abs(frame.source_end_bound_sec - asset.duration_sec) > 1e-6):
            raise GroundingPilotError("grounding_original_tail_missing")
    if asset.frame_count is not None and asset.frame_count != len(frames):
        raise GroundingPilotError("grounding_native_frame_count_mismatch")


def source_clock_values(frames: tuple[Any, ...], sampled_indices: tuple[int, ...], policy: str) -> dict[str, Any]:
    """Explicit encodings: true CFR ordinals, or original source clock ticks."""
    if not frames or not sampled_indices or any(type(index) is not int or not 0 <= index < len(frames)
                                               for index in sampled_indices):
        raise GroundingPilotError("grounding_clock_indices_invalid")
    base = Fraction(frames[0].time_base_num, frames[0].time_base_den)
    origin = frames[0].source_origin_pts
    if policy == "strict_cfr":
        if len(frames) < 2:
            raise GroundingPilotError("grounding_cfr_proof_requires_two_frames")
        step = frames[1].source_pts - frames[0].source_pts
        if step <= 0 or any(frame.source_pts - origin != index * step for index, frame in enumerate(frames)):
            raise GroundingPilotError("grounding_vfr_not_supported_by_cfr_profile")
        rate = 1 / (base * step)
        indices = sampled_indices
        fields = {"kind": "verified_cfr_native_indices_over_source_fps", "fps_num": rate.numerator,
                  "fps_den": rate.denominator}
    elif policy == "native_pts_ticks":
        rate = 1 / base
        indices = tuple(frames[index].source_pts - origin for index in sampled_indices)
        fields = {"kind": "verified_native_pts_ticks_over_ticks_per_second",
                  "source_tick_indices": indices, "ticks_per_second_num": rate.numerator,
                  "ticks_per_second_den": rate.denominator}
    else:
        raise GroundingPilotError("grounding_clock_profile_invalid")
    for index, value in zip(sampled_indices, indices):
        if not math.isclose(float(Fraction(value) / rate), frames[index].pts_sec, rel_tol=0, abs_tol=1e-7):
            raise GroundingPilotError("grounding_processor_clock_ratio_mismatch")
    return {"indices": indices, "rate": rate, "proof_fields": fields}


def expected_pair_timestamps(indices: tuple[int, ...], rate: Fraction) -> tuple[float, ...]:
    if not indices or rate <= 0:
        raise GroundingPilotError("grounding_processor_clock_invalid")
    values = indices + (indices[-1],) if len(indices) % 2 else indices
    return tuple(float(Fraction(values[index] + values[index + 1], 2) / rate)
                 for index in range(0, len(values), 2))


def processor_clock_recipe(processor: Any, policy: str) -> dict[str, Any]:
    """Bind the actual installed clock/pixel/model code, without private inputs."""
    import transformers.processing_utils as processing
    import transformers.video_processing_utils as video_processing
    import transformers.video_utils as video_utils
    import transformers.models.qwen3_vl.modeling_qwen3_vl as model_module
    modules = {type(processor).__module__: inspect.getfile(type(processor)),
        type(processor.video_processor).__module__: inspect.getfile(type(processor.video_processor)),
        processing.__name__: processing.__file__, video_processing.__name__: video_processing.__file__,
        video_utils.__name__: video_utils.__file__, model_module.__name__: model_module.__file__}
    return {"schema_version": "forte.timelens-clock.v1", "clock_policy": policy,
        "transformers_version": importlib.metadata.version("transformers"),
        "implementation_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for name, path in sorted(modules.items())},
        "decoded_array_input": True, "do_sample_frames": False, "do_resize": False,
        "metadata_removed_before_model": True, "temporal_patch_size": 2,
        "timestamp_rule": "pair_mean_of_clock_index_divided_by_clock_rate_repeat_odd_tail",
        "timestamp_precision_decimals": 1,
        "encoding": "source_pts_minus_origin_and_ticks_per_second" if policy == "native_pts_ticks"
                    else "native_frame_ordinal_and_verified_constant_source_fps"}


def _verify_clock_audit(path: Path, digest: str, clock_sha: str, *,
                        expected_processor_assets: dict[str, str], expected_runtime_packages: dict[str, str]) -> None:
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError
        audit = json.loads(raw)
        checks = ("generated_only", "cpu_only", "pixel_values_identical_under_clock_reencoding",
            "video_grid_identical_under_clock_reencoding", "nonzero_origin_verified", "vfr_verified",
            "odd_tail_verified", "processor_timestamps_verified", "rendered_timestamp_tokens_verified")
        if (audit.get("schema_version") != "forte.grounding-processor-clock-audit.v1"
                or audit.get("transformers_version") != "5.17.0"
                or audit.get("clock_policy") != "native_pts_ticks"
                or audit.get("processor_clock_sha256") != clock_sha
                or audit.get("model_id") != MODEL_ID or audit.get("processor_revision") != REVISION
                or audit.get("model_manifest_sha256") != MANIFEST_SHA256
                or audit.get("public_processor_assets_sha256") != expected_processor_assets
                or audit.get("target_runtime_verified") is not True
                or any(audit.get("packages", {}).get(name) != expected_runtime_packages.get(name)
                       for name in ("transformers", "torch", "torchvision", "numpy", "pillow", "tokenizers"))
                or audit.get("runner_source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or audit.get("cuda_initialized") is not False or audit.get("model_weights_loaded") is not False
                or audit.get("model_forward_calls") != 0 or audit.get("private_source_files_read") != 0
                or any(audit.get(key) is not True for key in checks)):
            raise ValueError
    except Exception:
        raise GroundingPilotError("grounding_native_tick_cpu_audit_invalid") from None


def decode_sampled_source(asset: Any, config: GroundingPilotConfig, *, deadline: float,
                          cancel_event: Any = None) -> tuple[tuple[Any, ...], tuple[int, ...], bytes]:
    """Decode native frames once; retain at most 2fps+tail bounded RGB buffers."""
    from forte.video.batches import iter_rgb_batches
    if (not 0 < asset.duration_sec <= config.max_source_duration_sec or asset.width * asset.height > 33554432
            or not 0 < Path(asset.owned_path).stat().st_size <= config.max_source_bytes):
        raise GroundingPilotError("grounding_source_budget_exceeded")
    targets = [Fraction(index, 2) for index in range(math.ceil(asset.duration_sec * config.sample_fps))]
    selected, ledger, position, previous = {}, [], 0, None
    with closing(iter_rgb_batches(asset, sample_fps=None, cancel=cancel_event, use_subprocess=True,
            verify_source_hash=True, batch_size=16, target_size=(224, 224), aspect_policy="letterbox",
            deadline_monotonic=deadline, max_runtime_sec=max(.001, deadline - time.monotonic()),
            max_source_frames=config.max_native_frames, max_output_frames=config.max_native_frames,
            thread_count=4)) as batches:
        last_complete = last_tail = False
        for batch in batches:
            for index, frame in enumerate(batch.frames):
                _checkpoint(deadline, cancel_event)
                ordinal = len(ledger)
                ledger.append(frame)
                current = (ordinal, frame, bytes(batch.frame_pixels(index)))
                current_time = Fraction(frame.source_pts - frame.source_origin_pts) * Fraction(frame.time_base_num, frame.time_base_den)
                while position < len(targets) and current_time >= targets[position]:
                    candidate = current
                    if previous is not None:
                        previous_time = Fraction(previous[1].source_pts - previous[1].source_origin_pts) * Fraction(
                            previous[1].time_base_num, previous[1].time_base_den)
                        if abs(previous_time - targets[position]) <= abs(current_time - targets[position]):
                            candidate = previous
                    selected[candidate[0]] = candidate[2]
                    position += 1
                previous = current
            last_complete, last_tail = batch.source_complete, batch.tail_included
        if previous is None or not last_complete or not last_tail:
            raise GroundingPilotError("grounding_decode_incomplete")
        selected[previous[0]] = previous[2]
    if len(selected) > config.max_sampled_frames or len(selected) < 2:
        raise GroundingPilotError("grounding_sample_frame_budget_invalid")
    frames, indices = tuple(ledger), tuple(sorted(selected))
    validate_native_ledger(asset, frames)
    from training.grounding.contracts import expected_sampled_frame_indices
    if indices != expected_sampled_frame_indices(asset, frames):
        raise GroundingPilotError("grounding_sampling_contract_mismatch")
    return frames, indices, b"".join(selected[index] for index in indices)


def audit_source_clock(asset: Any, config: GroundingPilotConfig, *, deadline: float,
                       cancel_event: Any = None) -> dict[str, Any]:
    """CPU-only fresh decode/PTS preflight. This does not certify the processor.

    A18 pins this private receipt before GPU calls. A01's separate generated
    CPU processor audit is still required for the native-tick model profile.
    No Torch, Transformers or CUDA import occurs on this execution path.
    """
    from training.grounding.contracts import validate_source_ledger
    started = time.perf_counter()
    ledger, ordinals, pixels = decode_sampled_source(asset, config, deadline=deadline, cancel_event=cancel_event)
    del pixels
    binding = validate_source_ledger(asset, ledger)
    values = source_clock_values(ledger, ordinals, config.clock_policy)
    gaps = sorted({second.source_pts - first.source_pts for first, second in zip(ledger, ledger[1:])})
    return {"schema_version": "forte.grounding-source-clock-audit.v1", **binding.model_dump(mode="json"),
        "clock_policy": config.clock_policy, "source_ledger": [asdict(frame) for frame in ledger],
        "sampled_frame_indices": list(ordinals), "sampled_frame_ids": [ledger[index].frame_id for index in ordinals],
        "clock_encoding_indices": list(values["indices"]),
        "clock_rate_num": values["rate"].numerator, "clock_rate_den": values["rate"].denominator,
        "clock_proof_fields": values["proof_fields"],
        "expected_processor_timestamps_sec": list(expected_pair_timestamps(values["indices"], values["rate"])),
        "actual_native_tick_gaps": gaps, "is_cfr": len(gaps) == 1,
        "sampling_policy": "nearest_native_to_half_second_grid_tie_earlier_plus_original_tail",
        "sample_fps": 2., "target_size": 224, "aspect_policy": "letterbox",
        "processor_clock_verified": False, "model_inference_performed": False,
        "private_pixels_persisted": False, "decode_calls": 1,
        "elapsed_sec": time.perf_counter() - started}


class GroundingPilot:
    def __init__(self, config: GroundingPilotConfig):
        self.config = config
        self._model = self._processor = self._torch = None
        self._attestation: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.cold_load_sec: float | None = None
        self._model_load_calls = 0

    @property
    def attestation(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._attestation))

    def load(self) -> dict[str, Any]:
        with self._lock:
            if self._model is not None:
                return self.attestation
            started = time.perf_counter()
            model = None
            try:
                from forte.model.manifest import load_manifest, verify_local_snapshot
                manifest = load_manifest(self.config.manifest)
                verification = verify_local_snapshot(self.config.snapshot, self.config.manifest)
                if (verification.manifest_sha256 != self.config.manifest_sha256
                        or verification.model_id != MODEL_ID or verification.revision != REVISION
                        or verification.processor_revision != REVISION):
                    raise GroundingPilotError("grounding_checkpoint_pin_mismatch")
                expected = {entry["path"] for entry in manifest["files"]}
                actual = {str(path.relative_to(self.config.snapshot)).replace("\\", "/")
                          for path in self.config.snapshot.rglob("*") if path.is_file()}
                if actual != expected or any(name.endswith((".py", ".bin")) for name in actual):
                    raise GroundingPilotError("grounding_unexpected_snapshot_artifact")
                model_config = json.loads((self.config.snapshot / "config.json").read_text())
                if (model_config.get("model_type") != "qwen3_vl" or model_config.get("auto_map")
                        or model_config.get("architectures") != ["Qwen3VLForConditionalGeneration"]):
                    raise GroundingPilotError("grounding_standard_model_required")
                for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
                    os.environ[name] = "1"
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "torchvision", "av", "numpy", "pillow", "safetensors", "tokenizers")}
                if any(packages[name] != version for name, version in (
                        ("torch", "2.13.0+cu129"), ("transformers", "5.17.0"),
                        ("torchvision", "0.28.0+cu129"), ("av", "19.0.1"))):
                    raise GroundingPilotError("grounding_runtime_version_mismatch")
                import torch
                self._torch = torch
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
                from transformers.utils import logging
                logging.set_verbosity_error()
                logging.disable_progress_bar()
                processor = AutoProcessor.from_pretrained(str(self.config.snapshot), local_files_only=True,
                    trust_remote_code=False, use_fast=True)
                if (type(processor).__name__ != "Qwen3VLProcessor"
                        or type(processor.video_processor).__name__ != "Qwen3VLVideoProcessor"
                        or processor.video_processor.patch_size != 16
                        or processor.video_processor.temporal_patch_size != 2
                        or processor.video_processor.merge_size != 2):
                    raise GroundingPilotError("grounding_processor_class_or_patch_mismatch")
                clock_recipe = processor_clock_recipe(processor, self.config.clock_policy)
                clock_sha = canonical_sha256(clock_recipe)
                if self.config.clock_policy == "native_pts_ticks":
                    if self.config.clock_audit_path is None:
                        raise GroundingPilotError("grounding_native_tick_cpu_audit_required")
                    _verify_clock_audit(self.config.clock_audit_path, self.config.clock_audit_sha256, clock_sha,
                        expected_processor_assets={row["path"]: row["sha256"] for row in manifest["files"]
                                                   if row.get("kind") == "processor_or_config"},
                        expected_runtime_packages=packages)
                if not torch.cuda.is_available():
                    raise GroundingPilotError("grounding_cuda_unavailable")
                torch.cuda.set_per_process_memory_fraction(self.config.process_memory_fraction, self.config.device)
                free_before, total = torch.cuda.mem_get_info(self.config.device)
                if free_before < self.config.min_free_bytes:
                    raise GroundingPilotError("grounding_cuda_free_memory_below_minimum")
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.manual_seed(self.config.seed)
                torch.cuda.reset_peak_memory_stats(self.config.device)
                model, loading = Qwen3VLForConditionalGeneration.from_pretrained(str(self.config.snapshot),
                    local_files_only=True, trust_remote_code=False, use_safetensors=True,
                    dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
                if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                    raise GroundingPilotError("grounding_weights_not_fully_loaded")
                model = model.eval().requires_grad_(False).to(self.config.device)
                torch.cuda.synchronize(self.config.device)
                self._model, self._processor, self._torch = model, processor, torch
                self._model_load_calls += 1
                self.cold_load_sec = time.perf_counter() - started
                decoder_sha = {name: hashlib.sha256((Path(__file__).resolve().parents[1] / "src/forte/video" / name).read_bytes()).hexdigest()
                               for name in ("batches.py", "decode.py", "probe.py")}
                self._attestation = {"model_id": MODEL_ID, "model_revision": REVISION,
                    "processor_revision": REVISION, "model_manifest_sha256": verification.manifest_sha256,
                    "files_verified": verification.files_verified, "bytes_verified": verification.bytes_verified,
                    "model_class": type(model).__name__, "processor_class": type(processor).__name__,
                    "image_processor_class": type(processor.image_processor).__name__,
                    "video_processor_class": type(processor.video_processor).__name__, "packages": packages,
                    "processor_clock_recipe": clock_recipe, "processor_clock_sha256": clock_sha,
                    "clock_policy": self.config.clock_policy, "clock_audit_sha256": self.config.clock_audit_sha256,
                    "cold_load_sec": self.cold_load_sec, "process_memory_fraction": .10,
                    "memory_cap_bytes": int(total * .10), "gpu_free_before_load_bytes": free_before,
                    "gpu_total_bytes": total, "gpu_name": torch.cuda.get_device_name(self.config.device),
                    "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "decoder_source_sha256": decoder_sha, "target_size": 224, "aspect_policy": "letterbox",
                    "sample_fps": 2., "original_tail_required": True, "max_new_tokens": self.config.max_new_tokens,
                    "max_sampled_frames": self.config.max_sampled_frames,
                    "max_native_frames": self.config.max_native_frames,
                    "max_source_duration_sec": self.config.max_source_duration_sec,
                    "decoder_threads": 4, "decoder_thread_type": "FRAME", "decoder_batch_size": 16,
                    "processor_temporal_padding": "repeat_original_last_pixel_and_clock_index_to_multiple_two",
                    "do_sample": False, "seed": self.config.seed, "deterministic_algorithms": True,
                    "allow_tf32": False, "attention": "sdpa", "model_dtype": "bfloat16",
                    "proposal_only": True, "official_publisher_pixel_profile": False,
                    "accuracy_measured": False, "no_audio": True,
                    "model_load_calls": self._model_load_calls,
                    "memory_limit_kind": "pytorch_cuda_allocator_fraction_not_external_process_cap",
                    "sampling_policy": "nearest_native_to_half_second_grid_tie_earlier_plus_original_tail"}
                self._attestation["input_recipe_sha256"] = canonical_sha256({key: value for key, value in
                    self._attestation.items() if key not in {"cold_load_sec", "gpu_free_before_load_bytes"}})
                return self.attestation
            except GroundingPilotError:
                model = None
                self.close()
                raise
            except Exception:
                model = None
                self.close()
                raise GroundingPilotError("grounding_offline_load_failed") from None

    def memory_stats(self, *, reset_peak: bool = False) -> dict[str, int]:
        if self._torch is None:
            return {}
        cuda = self._torch.cuda
        cuda.synchronize(self.config.device)
        if reset_peak:
            cuda.reset_peak_memory_stats(self.config.device)
        free, total = cuda.mem_get_info(self.config.device)
        return {"free_bytes": free, "total_bytes": total,
            "allocated_bytes": cuda.memory_allocated(self.config.device),
            "reserved_bytes": cuda.memory_reserved(self.config.device),
            "peak_allocated_bytes": cuda.max_memory_allocated(self.config.device),
            "peak_reserved_bytes": cuda.max_memory_reserved(self.config.device)}

    def run(self, asset: Any, query: Any, *, deadline: float, cancel_event: Any = None,
            expected_source_ledger_sha256: str | None = None) -> GroundingAttempt:
        """Synchronous one-request admission; all per-attempt failures are retained."""
        if not self._lock.acquire(blocking=False):
            return GroundingAttempt("failed", "grounding_busy", None, (), {}, {**self.attestation,
                "decode_calls": 0, "processor_calls": 0, "generation_calls": 0,
                "source_result_cache_hits": 0, "model_load_calls": self._model_load_calls}, {}, "", (), "not_started")
        started, ledger, raw, tokens, finish, timings, result = time.perf_counter(), (), "", (), "not_started", {}, None
        error = None
        active_stage = None
        counts = {"decode_calls": 0, "processor_calls": 0, "generation_calls": 0,
                  "source_result_cache_hits": 0, "model_load_calls": self._model_load_calls}
        try:
            _checkpoint(deadline, cancel_event)
            if self._model is None:
                raise GroundingPilotError("grounding_not_loaded")
            from training.grounding.contracts import (PhysicalQuery, ProcessorClockProof, GroundingRawResult,
                                                       parse_intervals, bind_intervals, validate_source_ledger)
            from forte.video.probe import content_sha256
            if not isinstance(query, PhysicalQuery):
                raise GroundingPilotError("grounding_physical_query_contract_required")
            source = Path(asset.owned_path)
            if (not source.is_file() or not 0 < source.stat().st_size <= self.config.max_source_bytes
                    or not 0 < asset.duration_sec <= self.config.max_source_duration_sec):
                raise GroundingPilotError("grounding_source_budget_exceeded")
            verifying = time.perf_counter()
            active_stage = ("source_verification_sec", verifying)
            if content_sha256(Path(asset.owned_path), deadline_monotonic=deadline, cancel=cancel_event) != asset.content_hash:
                raise GroundingPilotError("grounding_source_hash_mismatch")
            timings["source_verification_sec"] = time.perf_counter() - verifying
            active_stage = None
            decoding = time.perf_counter()
            active_stage = ("decode_sec", decoding)
            counts["decode_calls"] += 1
            ledger, ordinals, pixels = decode_sampled_source(asset, self.config, deadline=deadline, cancel_event=cancel_event)
            timings["decode_sec"] = time.perf_counter() - decoding
            active_stage = None
            source_binding = validate_source_ledger(asset, ledger)
            if (expected_source_ledger_sha256 is not None
                    and expected_source_ledger_sha256 != source_binding.source_ledger_sha256):
                raise GroundingPilotError("grounding_preflight_source_ledger_changed")
            clock_values = source_clock_values(ledger, ordinals, self.config.clock_policy)
            expected = expected_pair_timestamps(clock_values["indices"], clock_values["rate"])
            processor = self._processor
            actual = tuple(float(value) for value in processor._calculate_timestamps(
                list(clock_values["indices"]), float(clock_values["rate"]), 2))
            if len(actual) != len(expected) or any(abs(a - b) > 1e-7 for a, b in zip(actual, expected)):
                raise GroundingPilotError("grounding_live_processor_clock_mismatch")
            clock = ProcessorClockProof(**clock_values["proof_fields"], sampled_frame_indices=ordinals,
                processor_clock_sha256=self._attestation["processor_clock_sha256"], processor_clock_verified=True,
                temporal_patch_size=2, timestamp_precision_decimals=1, processor_timestamps_sec=actual)
            import numpy as np
            from transformers import StoppingCriteria, StoppingCriteriaList
            from transformers.video_utils import VideoMetadata
            processing = time.perf_counter()
            active_stage = ("processor_sec", processing)
            video = np.frombuffer(pixels, dtype=np.uint8).reshape(len(ordinals), 224, 224, 3).copy()
            metadata = VideoMetadata(total_num_frames=len(ledger), fps=float(clock_values["rate"]),
                width=224, height=224, duration=asset.duration_sec,
                frames_indices=list(clock_values["indices"]), video_backend="forte_native_ledger")
            prompt = (f"Given the query: {json.dumps(query.text, ensure_ascii=False)}, return ALL time spans "
                      "(in seconds) where the query is relevant.\nOutput format MUST be a JSON array of [start, end] pairs.\n")
            messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            counts["processor_calls"] += 1
            inputs = processor(text=[text], videos=[video], video_metadata=[metadata],
                do_sample_frames=False, do_resize=False, return_tensors="pt", input_data_format="channels_last",
                cap_pixels_per_frame=False)
            allowed = {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids"}
            if set(inputs) - allowed or not {"input_ids", "pixel_values_videos", "video_grid_thw"} <= set(inputs):
                raise GroundingPilotError("grounding_processor_model_inputs_invalid")
            grid = inputs["video_grid_thw"].tolist()
            if grid != [[math.ceil(len(ordinals) / 2), 14, 14]]:
                raise GroundingPilotError("grounding_video_grid_mismatch")
            # Compare the actual expanded text timestamp sequence, not an assumed
            # source-FPS label. Decoding these tokens stays strictly in memory.
            expanded = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
            observed_stamps = re.findall(r"<([0-9]+\.[0-9]) seconds>", expanded)
            if observed_stamps != [f"{value:.1f}" for value in actual]:
                raise GroundingPilotError("grounding_rendered_processor_clock_mismatch")
            timings["processor_sec"] = time.perf_counter() - processing
            active_stage = None
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            if prompt_tokens > 12000:
                raise GroundingPilotError("grounding_prompt_token_budget_exceeded")
            self.memory_stats(reset_peak=True)
            transferring = time.perf_counter()
            active_stage = ("h2d_sec", transferring)
            inputs = inputs.to(self.config.device)
            self._torch.cuda.synchronize(self.config.device)
            timings["h2d_sec"] = time.perf_counter() - transferring
            active_stage = None
            _checkpoint(deadline, cancel_event)
            first_token = []
            torch = self._torch

            class Stop(StoppingCriteria):
                def __call__(inner, input_ids, scores, **kwargs):
                    if input_ids.shape[-1] > prompt_tokens and not first_token:
                        torch.cuda.synchronize(self.config.device)
                        first_token.append(time.perf_counter())
                    stop = time.monotonic() >= deadline or cancel_event is not None and cancel_event.is_set()
                    return torch.full((input_ids.shape[0],), bool(stop), dtype=torch.bool, device=input_ids.device)

            generating = time.perf_counter()
            active_stage = ("generate_sec", generating)
            counts["generation_calls"] += 1
            with torch.inference_mode():
                output = self._model.generate(**inputs, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                    num_beams=1, use_cache=True, return_dict_in_generate=False, output_scores=False,
                    repetition_penalty=1.0, stopping_criteria=StoppingCriteriaList([Stop()]))
            torch.cuda.synchronize(self.config.device)
            timings["generate_sec"] = time.perf_counter() - generating
            active_stage = None
            if first_token:
                timings["first_generated_token_sec"] = first_token[0] - generating
            tokens = tuple(int(value) for value in output[0, prompt_tokens:].detach().cpu().tolist())
            raw = processor.tokenizer.decode(list(tokens), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            eos = self._model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            finish = "stop" if tokens and tokens[-1] in eos else "length"
            if cancel_event is not None and cancel_event.is_set():
                finish = "cancelled"
            elif time.monotonic() >= deadline:
                finish = "deadline"
            timings["prompt_tokens"] = float(prompt_tokens)
            timings["generated_tokens"] = float(len(tokens))
            parsing = time.perf_counter()
            active_stage = ("parse_bind_sec", parsing)
            intervals = parse_intervals(raw, duration_sec=asset.duration_sec, finish_reason=finish)
            bound = bind_intervals(intervals, asset=asset, source_index=ledger,
                sampled_frame_ids=tuple(ledger[index].frame_id for index in ordinals), clock=clock)
            timings["parse_bind_sec"] = time.perf_counter() - parsing
            active_stage = None
            timings["total_sec"] = time.perf_counter() - started
            result = GroundingRawResult(**source_binding.model_dump(), query=query, model_id=MODEL_ID,
                model_revision=REVISION, processor_revision=REVISION,
                model_manifest_sha256=self.config.manifest_sha256,
                input_recipe_sha256=self._attestation["input_recipe_sha256"], clock=clock,
                sampled_frame_ids=tuple(ledger[index].frame_id for index in ordinals), intervals=bound,
                raw_response=raw, generated_token_ids=tokens, finish_reason=finish,
                metrics={key: float(value) for key, value in timings.items()})
        except GroundingPilotError as exc:
            error = str(exc)
        except Exception as exc:
            from training.grounding.contracts import GroundingContractError
            error = ("grounding_output_" + exc.code if isinstance(exc, GroundingContractError)
                     else "grounding_cuda_oom" if type(exc).__name__ == "OutOfMemoryError"
                     else "grounding_" + active_stage[0].removesuffix("_sec") + "_failed" if active_stage
                     else "grounding_decode_process_generate_or_output_invalid")
        finally:
            if active_stage is not None:
                timings.setdefault(active_stage[0], time.perf_counter() - active_stage[1])
            timings["total_sec"] = time.perf_counter() - started
            try:
                stats = self.memory_stats()
            except Exception:
                stats = {}
            self._lock.release()
        return GroundingAttempt("done" if result is not None else "failed", error, result, ledger,
            timings, {**self.attestation, **counts}, stats, raw, tokens, finish)

    def close(self) -> None:
        self._model = self._processor = None
        if self._torch is not None:
            self._torch.cuda.empty_cache()
        self._torch = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--asset-json", required=True, type=Path)
    parser.add_argument("--query-json", type=Path)
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--clock-policy", choices=("strict_cfr", "native_pts_ticks"), default="strict_cfr")
    parser.add_argument("--clock-audit-path", type=Path)
    parser.add_argument("--clock-audit-sha256")
    parser.add_argument("--source-ledger-sha256")
    parser.add_argument("--timeout-sec", type=float, default=30.)
    args = parser.parse_args()
    pilot = None
    try:
        if not 0 < args.timeout_sec <= 30 or args.private_output.exists():
            raise GroundingPilotError("grounding_cli_limits_invalid")
        from forte.shared.schemas import VideoAsset
        from training.grounding.contracts import PhysicalQuery
        asset = VideoAsset.model_validate_json(args.asset_json.read_text())
        config = GroundingPilotConfig(args.snapshot or Path("unused"), args.manifest or Path("unused"),
            clock_policy=args.clock_policy, clock_audit_path=args.clock_audit_path,
            clock_audit_sha256=args.clock_audit_sha256)
        if args.audit_source:
            receipt = audit_source_clock(asset, config, deadline=time.monotonic() + args.timeout_sec)
            safe = {"status": "audited", "native_frame_count": receipt["native_frame_count"],
                "sampled_frame_count": len(receipt["sampled_frame_indices"]),
                "clock_policy": receipt["clock_policy"], "is_cfr": receipt["is_cfr"],
                "model_inference_performed": False}
            exit_code = 0
        else:
            if args.snapshot is None or args.manifest is None or args.query_json is None:
                raise GroundingPilotError("grounding_model_and_query_inputs_required")
            if not re.fullmatch(r"[a-f0-9]{64}", args.source_ledger_sha256 or ""):
                raise GroundingPilotError("grounding_source_preflight_pin_required")
            query = PhysicalQuery.model_validate_json(args.query_json.read_text())
            pilot = GroundingPilot(config)
            pilot.load()
            attempt = pilot.run(asset, query, deadline=time.monotonic() + args.timeout_sec,
                expected_source_ledger_sha256=args.source_ledger_sha256)
            receipt, safe, exit_code = attempt.private_dict(), attempt.safe_summary(), 0 if attempt.status == "done" else 2
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.private_output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(receipt, target, ensure_ascii=False, allow_nan=False)
            target.flush()
            os.fsync(target.fileno())
        print(json.dumps(safe, sort_keys=True))
        return exit_code
    except Exception:
        print(json.dumps({"status": "failed", "error_code": "grounding_pilot_failed"}))
        return 2
    finally:
        if pilot is not None:
            pilot.close()


if __name__ == "__main__":
    raise SystemExit(main())
