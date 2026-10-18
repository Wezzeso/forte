"""Generated-only, CPU-only audit of the pinned TimeLens video processor.

No weights, source videos, labels, remote code or model forward are used. A17
can stage this file, the NEW runner, the public manifest and its nonweight
assets into a network-disabled container. Only aggregate JSON is emitted.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

MANIFEST_SHA = "721f00a6cba0705755627f3f139b02cba570f680b03c43e54ab32e5fac2c660f"
REVISION = "10a779b6ef6fc8ad41367db2d1eb6779ca84995a"
TARGET_PACKAGES = {"transformers": "5.17.0", "torch": "2.13.0+cu129", "torchvision": "0.28.0+cu129"}


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_assets(snapshot, manifest_path):
    """Read only public nonweight artifacts and require their exact pinned set."""
    raw = Path(manifest_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "audit_manifest_pin_invalid")
    manifest = json.loads(raw)
    require(manifest["revision"] == manifest["processor_revision"] == REVISION, "audit_revision_invalid")
    entries = {row["path"]: row for row in manifest["files"] if row["kind"] == "processor_or_config"}
    require(len(entries) == 12, "audit_asset_set_invalid")
    actual = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()}
    require(actual == set(entries), "audit_asset_set_invalid")
    receipts = {}
    for name, row in sorted(entries.items()):
        path = snapshot / name
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == row["size_bytes"],
                "audit_asset_size_invalid")
        digest = sha(path)
        require(digest == row["sha256"], "audit_asset_digest_invalid")
        receipts[name] = digest
    return receipts


def generated_cases():
    # These are invented integer source clocks. No corpus metadata is read.
    return (
        ("cfr_even", Fraction(1, 90000), 0, tuple(3000 * index for index in range(61)), (0, 15, 30, 60)),
        ("cfr_odd_nonzero", Fraction(1, 90000), 450000, tuple(3000 * index for index in range(61)), (0, 15, 30, 45, 60)),
        ("vfr_even_nonzero", Fraction(1, 90000), 900000, (0, 1700, 4400, 9100, 15000, 22500, 31500, 45000, 67500), (0, 3, 5, 8)),
        ("vfr_odd_nonzero", Fraction(1, 90000), 630000, (0, 1700, 4400, 9100, 15000, 22500, 31500, 45000, 67500), (0, 3, 5, 7, 8)),
        ("rational_clock_odd", Fraction(1001, 90000), 12345, (0, 1, 3, 6, 10, 15, 21), (0, 2, 3, 5, 6)),
    )


def independent_pair_means(offsets, ordinals, base):
    timestamps = tuple(Fraction(offsets[index]) * base for index in ordinals)
    if len(timestamps) % 2:
        timestamps += (timestamps[-1],)
    return tuple(float((timestamps[index] + timestamps[index + 1]) / 2)
                 for index in range(0, len(timestamps), 2))


def load_runner(path):
    spec = importlib.util.spec_from_file_location("forte_clock_audit_runner", path)
    require(spec is not None and spec.loader is not None, "audit_runner_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit(snapshot, manifest_path, runner_path, *, require_target_runtime=False):
    # Set offline controls before importing HF. No CUDA API that initializes a
    # context is called. Any accidental initialization fails closed below.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    assets = verify_assets(snapshot, manifest_path)
    packages = {name: importlib.metadata.version(name) for name in
                ("transformers", "torch", "torchvision", "numpy", "pillow", "tokenizers")}
    target = all(packages[name] == version for name, version in TARGET_PACKAGES.items())
    require(packages["transformers"] == "5.17.0", "audit_transformers_version_invalid")
    require(not require_target_runtime or target, "audit_target_runtime_invalid")
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from transformers.utils import logging
    from transformers.video_utils import VideoMetadata
    require(not torch.cuda.is_initialized(), "audit_cuda_was_initialized")

    def forbidden_cuda_init(*args, **kwargs):
        raise RuntimeError("audit_cuda_initialization_forbidden")

    torch.cuda._lazy_init = forbidden_cuda_init
    torch.set_num_threads(2)
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True,
                                            trust_remote_code=False, use_fast=True)
    require(type(processor).__name__ == "Qwen3VLProcessor"
            and type(processor.video_processor).__name__ == "Qwen3VLVideoProcessor"
            and processor.video_processor.patch_size == 16
            and processor.video_processor.temporal_patch_size == 2
            and processor.video_processor.merge_size == 2, "audit_processor_type_invalid")
    runner = load_runner(runner_path)
    recipe = runner.processor_clock_recipe(processor, "native_pts_ticks")
    require(recipe["transformers_version"] == "5.17.0", "audit_recipe_version_invalid")
    # Generated text has no labels, filenames, paths or corpus identifiers.
    messages = [{"role": "user", "content": [{"type": "video"},
                {"type": "text", "text": "Return an empty JSON array for this synthetic processor audit."}]}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def process(video, indices, rate, native_count, duration):
        metadata = VideoMetadata(total_num_frames=native_count, fps=float(rate), width=224, height=224,
            duration=float(duration), frames_indices=list(indices), video_backend="forte_generated_clock_audit")
        inputs = processor(text=[template], videos=[video], video_metadata=[metadata],
            do_sample_frames=False, do_resize=False, return_tensors="pt", input_data_format="channels_last",
            cap_pixels_per_frame=False)
        require(not {"fps", "second_per_grid_ts", "video_metadata"}.intersection(inputs),
                "audit_metadata_leaked_to_model_inputs")
        require(all(not hasattr(value, "device") or value.device.type == "cpu" for value in inputs.values()),
                "audit_non_cpu_tensor")
        text = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        stamps = re.findall(r"<([0-9]+\.[0-9]) seconds>", text)
        return inputs, stamps

    results = []
    differing_controls = 0
    for name, base, origin, offsets, ordinals in generated_cases():
        frames = tuple(SimpleNamespace(time_base_num=base.numerator, time_base_den=base.denominator,
            source_origin_pts=origin, source_pts=origin + offset, pts_sec=float(base * offset)) for offset in offsets)
        encoded = runner.source_clock_values(frames, ordinals, "native_pts_ticks")
        require(encoded["indices"] == tuple(offsets[index] for index in ordinals)
                and encoded["rate"] == 1 / base, "audit_clock_encoder_invalid")
        expected = independent_pair_means(offsets, ordinals, base)
        actual = tuple(processor._calculate_timestamps(list(encoded["indices"]), float(encoded["rate"]), 2))
        require(len(actual) == len(expected) and all(math.isfinite(value) and abs(value - truth) <= 1e-7
                for value, truth in zip(actual, expected)), "audit_pair_mean_invalid")
        video = ((np.arange(len(ordinals) * 224 * 224 * 3, dtype=np.uint32).reshape(len(ordinals), 224, 224, 3)
                  * 13 + np.arange(len(ordinals), dtype=np.uint32)[:, None, None, None] * 29) % 256).astype(np.uint8)
        original_indices = tuple(encoded["indices"])
        tick_inputs, tick_stamps = process(video, original_indices, encoded["rate"], len(frames), base * (offsets[-1] + 3000))
        control_inputs, control_stamps = process(video, ordinals, Fraction(30), len(frames), base * (offsets[-1] + 3000))
        require(original_indices == encoded["indices"], "audit_input_clock_mutated")
        require(torch.equal(tick_inputs["pixel_values_videos"], control_inputs["pixel_values_videos"]),
                "audit_pixels_depend_on_clock_encoding")
        require(torch.equal(tick_inputs["video_grid_thw"], control_inputs["video_grid_thw"]),
                "audit_grid_depends_on_clock_encoding")
        require(tick_inputs["video_grid_thw"].tolist() == [[math.ceil(len(ordinals) / 2), 14, 14]],
                "audit_grid_invalid")
        require(tick_stamps == [f"{value:.1f}" for value in expected], "audit_rendered_timestamp_invalid")
        padded_verified = None
        if len(ordinals) % 2:
            padded_video = np.concatenate((video, video[-1:]), axis=0)
            padded, padded_stamps = process(padded_video, original_indices + (original_indices[-1],),
                encoded["rate"], len(frames), base * (offsets[-1] + 3000))
            padded_verified = torch.equal(tick_inputs["pixel_values_videos"], padded["pixel_values_videos"]) \
                and torch.equal(tick_inputs["video_grid_thw"], padded["video_grid_thw"]) and padded_stamps == tick_stamps
            require(padded_verified, "audit_odd_tail_padding_invalid")
        is_vfr = len({second - first for first, second in zip(offsets, offsets[1:])}) > 1
        if is_vfr:
            differing_controls += tick_stamps != control_stamps
            try:
                runner.source_clock_values(frames, ordinals, "strict_cfr")
            except runner.GroundingPilotError:
                pass
            else:
                raise RuntimeError("audit_vfr_accepted_by_cfr_profile")
        results.append({"generated_case": name, "native_frame_count": len(frames), "sampled_frame_count": len(ordinals),
            "vfr": is_vfr, "nonzero_origin": origin != 0, "odd_tail": len(ordinals) % 2 == 1,
            "pixel_values_equal": True, "video_grid_equal": True, "pair_mean_verified": True,
            "timestamp_tokens_verified": True, "explicit_tail_padding_equal": padded_verified,
            "control_timestamp_text_differs": tick_stamps != control_stamps})
    require(differing_controls >= 2, "audit_clock_control_not_discriminating")
    require(not torch.cuda.is_initialized(), "audit_cuda_was_initialized")
    return {"schema_version": "forte.grounding-processor-clock-audit.v1", "transformers_version": packages["transformers"],
        "clock_policy": "native_pts_ticks", "processor_clock_sha256": runner.canonical_sha256(recipe),
        "processor_clock_recipe": recipe, "model_id": "MCG-NJU/TimeLens2-2B", "processor_revision": REVISION,
        "model_manifest_sha256": MANIFEST_SHA, "public_processor_assets_sha256": assets,
        "audit_source_sha256": sha(__file__), "runner_source_sha256": sha(runner_path),
        "packages": packages, "target_runtime_verified": target, "generated_only": True, "cpu_only": True,
        "cuda_initialized": False, "model_weights_loaded": False, "model_forward_calls": 0,
        "private_source_files_read": 0, "pixel_values_identical_under_clock_reencoding": True,
        "video_grid_identical_under_clock_reencoding": True, "nonzero_origin_verified": True, "vfr_verified": True,
        "odd_tail_verified": True, "processor_timestamps_verified": True, "rendered_timestamp_tokens_verified": True,
        "one_decimal_rounding_max_error_sec": .05, "cases": results,
        "scope": "processor_clock_and_pixels_only_no_model_quality_or_legal_accuracy_claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-target-runtime", action="store_true")
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "audit_output_exists")
        receipt = audit(args.snapshot, args.manifest, args.runner, require_target_runtime=args.require_target_runtime)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "passed", "generated_cases": len(receipt["cases"]),
            "target_runtime_verified": receipt["target_runtime_verified"], "cpu_only": True,
            "processor_clock_sha256": receipt["processor_clock_sha256"], "receipt_sha256": sha(args.output)}))
        return 0
    except Exception:
        # No traceback, paths, token text or asset content in command output.
        print(json.dumps({"status": "failed", "error_code": "processor_clock_audit_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
