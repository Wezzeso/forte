"""Optional persistent-model, fresh-source weak-head repeatability experiment.

No import initializes CUDA. Execution requires a real pinned checkpoint, a
root-released GPU window and an exposure reservation made on the central host.
Use --help; all source identities, predictions and tensors stay private.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

from benchmarks.performance.repeatability import (RepeatabilityError, canonical_semantics,
    digest, file_digest, freeze_plan, require, summarize, validate_plan, write_new)

CODE_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("torch", "torchvision", "transformers", "pillow", "numpy", "av", "safetensors")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def private_path(path):
    path = Path(path).resolve()
    require("private" in path.parts, "repeat_private_artifact_location_required")
    return path


def runtime_identity():
    """Metadata-only hardware/runtime query; no model or input is opened."""
    info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,uuid",
        "--format=csv,noheader,nounits"], capture_output=True, check=True, text=True,
        timeout=10).stdout.strip().splitlines()
    require(len(info) == 1 and len(info[0].split(",")) == 3, "repeat_single_gpu_required")
    name, driver, gpu_uuid = (part.strip() for part in info[0].split(","))
    return {"packages": {key: importlib.metadata.version(key) for key in PACKAGES},
        "gpu_name": name, "gpu_driver": driver, "gpu_uuid": gpu_uuid,
        "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
        "allow_tf32": False, "seed": 20261017}


def code_pins():
    paths = {Path(__file__), Path(__file__).with_name("repeatability.py")}
    for folder in ("src", "serving", "training"):
        for path in (CODE_ROOT / folder).rglob("*.py"):
            if not any(part in {"private", "runs", "__pycache__", ".venv"}
                       for part in path.relative_to(CODE_ROOT).parts):
                require(not path.is_symlink(), "repeat_source_code_symlink")
                paths.add(path)
    paths.update(CODE_ROOT / name for name in ("pyproject.toml", "uv.lock")
                 if (CODE_ROOT / name).is_file())
    return {"code:" + path.relative_to(CODE_ROOT).as_posix():
        {"path": str(path), "sha256": file_digest(path)} for path in sorted(paths)}


def prepare(args):
    from training.temporal.extract_features import validate_prepared_inputs
    inputs = read(args.inputs)
    rows = validate_prepared_inputs(inputs)
    require(0 <= args.source_index < len(rows), "repeat_source_index_invalid")
    source = rows[args.source_index]
    media = Path(source["asset"]["owned_path"])
    require(media.is_file() and not media.is_symlink() and media.stat().st_size == source["source_bytes"]
        and file_digest(media) == source["content_sha256"], "repeat_source_bytes_mismatch")
    checkpoint = read(args.checkpoint / "checkpoint.json")
    require(checkpoint.get("schema_version") == "forte.temporal-head-checkpoint.v1"
        and checkpoint["head_identity"]["taxonomy_sha256"] == file_digest(args.taxonomy)
        and checkpoint["head_identity"]["encoder_artifacts_sha256"] == file_digest(args.encoder_manifest),
        "repeat_checkpoint_binding_invalid")
    require(file_digest(args.checkpoint / "head.safetensors") == checkpoint["weights_sha256"],
        "repeat_checkpoint_weights_mismatch")
    binding_paths = {"inputs": args.inputs, "checkpoint_manifest": args.checkpoint / "checkpoint.json",
        "head_weights": args.checkpoint / "head.safetensors", "encoder_manifest": args.encoder_manifest,
        "taxonomy": args.taxonomy, "rules": args.rules, "runtime": args.runtime,
        "deployment_proof": args.deployment_proof}
    bindings = {key: {"path": str(path.resolve()), "sha256": file_digest(path)}
                for key, path in binding_paths.items()}
    bindings.update(code_pins())
    root = private_path(args.private_root)
    require(not root.exists(), "repeat_output_root_exists")
    require(args.device == "cuda:0" and 1 <= args.batch_size <= 128
        and 1 <= args.source_timeout_sec <= 600, "repeat_limits_invalid")
    plan = freeze_plan(private_path(args.plan), source=source, bindings=bindings,
        settings={"snapshot": str(args.snapshot.resolve()), "device": args.device,
            "batch_size": args.batch_size, "concurrency": 1, "cache_reuse": False,
            "model_lifetime": "persistent_three_calls",
            "threshold": 0., "head_device": "cuda:0", "source_timeout_sec": args.source_timeout_sec,
            "process_timeout_sec": args.source_timeout_sec + 120.,
            "expected_feature_spec_sha256": checkpoint["head_identity"]["feature_spec_sha256"],
            "sampling": "all_native_frames_original_pts_tail", "target_size": [256, 256],
            "aspect_policy": "letterbox", "views": "global_only"},
        runtime=read(args.runtime), private_root=root)
    return {"status": "prepared_not_launched", "expected_attempts": 3,
        "independent_sources": 1, "conditions_sha256": plan["conditions_sha256"],
        "model_calls_made": 0, "jury_legal_completion_eligible": False}


def verify_bindings(plan):
    validate_plan(plan)
    require(all(Path(item["path"]).is_file() and file_digest(item["path"]) == item["sha256"]
        for item in plan["bindings"].values()), "repeat_pinned_artifact_changed")
    require(runtime_identity() == plan["expected_runtime"], "repeat_runtime_changed")
    source = plan["source"]
    media = Path(source["asset"]["owned_path"])
    require(not media.is_symlink() and media.stat().st_size == source["source_bytes"]
        and file_digest(media) == source["content_sha256"], "repeat_source_bytes_mismatch")


def reserve(args):
    """Run on the central registry host before copying the receipt to CUDA host."""
    from scripts.exposure_registry import append_exposures, read_registry
    plan = validate_plan(read(args.plan))
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    content = plan["source"]["content_sha256"]
    for row in plan["attempts"]:
        append_exposures(args.registry, rows, {content},
            trial_id=row["attempt_id"].replace("_", "-"), phase="local-trial", purpose="local_model_attempt")
    state = read_registry(args.registry, rows)
    receipt = {"schema_version": "forte.repeatability-exposure-reservation.v1",
        "conditions_sha256": plan["conditions_sha256"], "content_sha256": content,
        "attempt_ids": [row["attempt_id"] for row in plan["attempts"]],
        "registry_tip_sha256": state["tip_sha256"], "registry_event_count": state["event_count"],
        "registered": True}
    write_new(private_path(args.receipt), receipt)
    return {"status": "reserved_not_launched", "expected_attempts": 3,
        "reservation_sha256": file_digest(args.receipt), "model_calls_made": 0}


def tensor_digest(value):
    tensor = value.detach().cpu().contiguous()
    return digest({"dtype": str(tensor.dtype), "shape": list(tensor.shape),
        "bytes_sha256": hashlib.sha256(tensor.view(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()})


def run_attempt(plan, attempt, shared_runtime):
    """One fresh source pass; the same verified encoder/head persist for all three."""
    started = time.perf_counter()
    authorization = read(private_path(plan["private_root"]) / "execution-authorization.json")
    require(authorization.get("conditions_sha256") == plan["conditions_sha256"]
        and authorization.get("gpu_window_released") is True
        and attempt["attempt_id"] in authorization.get("attempt_ids", []),
        "repeat_parent_execution_authorization_required")
    directory = private_path(plan["private_root"]) / attempt["attempt_id"]
    require(not directory.exists(), "repeat_attempt_directory_exists")
    directory.mkdir(parents=True)
    record = {"attempt_id": attempt["attempt_id"], "status": "failed", "phase": "binding",
        "process_id": os.getpid(),
        "timings": {}, "fresh_stages_verified": False,
        "cache_hits": {"decode": 0, "encoder": 0, "head": 0},
        "stage_calls": {"decode": 0, "encoder_batches": 0, "head": 0}}
    timings = record["timings"]
    try:
        verify_bindings(plan)
        from forte.shared.schemas import VideoAsset
        from forte.video.batches import iter_rgb_batches, rgb_transform_recipe
        from serving.fast_encoder import FastEncoder, FastEncoderConfig
        from serving.feature_stream import iter_encoded_batches
        from training.temporal.cache import write_feature_cache, sequence_from_source_frames
        from training.temporal.checkpoint import load_checkpoint, sequence_inputs
        import torch
        from safetensors.torch import save_file

        settings, bindings, row = plan["settings"], plan["bindings"], plan["source"]
        decoder = rgb_transform_recipe(target_size=(256, 256), aspect_policy="letterbox")
        sampling = {"mode": "native", "all_native_frames": True, "tail_required": True,
                    "time": "original_integer_pts_and_canonical_rational_base"}
        crops = {"views": "global_only", "source_box_xyxy": [0., 0., 1., 1.]}
        record["phase"] = "persistent_runtime"
        if shared_runtime.get("load_attempted"):
            require(shared_runtime.get("ready") is True, "repeat_persistent_runtime_unavailable")
            encoder, head, identity, verification = (shared_runtime[key] for key in
                ("encoder", "head", "identity", "verification"))
            timings["encoder_load_sec"] = timings["head_load_sec"] = 0.
        else:
            shared_runtime["load_attempted"] = True
            shared_runtime["encoder_load_calls"] = 1
            record["phase"] = "encoder_load"
            encoder = FastEncoder(FastEncoderConfig(Path(settings["snapshot"]),
                Path(bindings["encoder_manifest"]["path"]), bindings["encoder_manifest"]["sha256"],
                batch_size=settings["batch_size"], device=settings["device"],
                preprocessing_mode="decoder_rgb_256", input_transform_sha256=decoder["sha256"]))
            shared_runtime["encoder"] = encoder
            stage = time.perf_counter()
            verification = encoder.load()
            timings["encoder_load_sec"] = time.perf_counter() - stage
            record["phase"] = "head_load"
            stage = time.perf_counter()
            shared_runtime["head_load_calls"] = 1
            head, identity, _ = load_checkpoint(Path(bindings["checkpoint_manifest"]["path"]).parent,
                expected_manifest_sha256=bindings["checkpoint_manifest"]["sha256"], device=settings["head_device"])
            torch.cuda.synchronize(settings["device"])
            timings["head_load_sec"] = time.perf_counter() - stage
            shared_runtime.update(head=head, identity=identity, verification=verification, ready=True)
        require(torch.are_deterministic_algorithms_enabled()
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
            and torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark,
            "repeat_effective_determinism_mismatch")
        record["runtime"] = runtime_identity()
        record["encoder_recipe"] = encoder.recipe

        record["phase"] = "fresh_features"
        asset = VideoAsset.model_validate(row["asset"])
        source_started = time.perf_counter()
        deadline = time.monotonic() + settings["source_timeout_sec"]
        rgb_hashes = []
        timings["decode_active_sec"] = timings["encoder_active_sec"] = 0.

        def source_factory(cancel):
            record["stage_calls"]["decode"] += 1
            iterator = iter(iter_rgb_batches(asset, sample_fps=None, cancel=cancel,
                verify_source_hash=True, batch_size=settings["batch_size"], target_size=(256, 256),
                aspect_policy="letterbox", deadline_monotonic=deadline,
                max_runtime_sec=settings["source_timeout_sec"], max_source_frames=10000,
                max_output_frames=10000, use_subprocess=True))
            try:
                while True:
                    stage = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        timings["decode_active_sec"] += time.perf_counter() - stage
                        return
                    timings["decode_active_sec"] += time.perf_counter() - stage
                    rgb_hashes.append({"frame_ids": [frame.frame_id for frame in batch.frames],
                        "shape": list(batch.shape), "rgb_sha256": hashlib.sha256(batch.pixels).hexdigest()})
                    yield batch
            finally:
                iterator.close()

        def measured_stream():
            stream = iter_encoded_batches(source_factory, encoder, deadline=deadline)
            try:
                for batch in stream:
                    record["stage_calls"]["encoder_batches"] += 1
                    timings["encoder_active_sec"] += batch.metrics["elapsed_sec"]
                    yield batch
            finally:
                stream.close()

        # The timed path consumes freshly encoded in-memory features. Artifact
        # serialization happens after the binary output, not inside its latency.
        batches = list(measured_stream())
        frames = [frame for batch in batches for frame in batch.frames]
        require(batches and batches[-1].source_complete and batches[-1].segment_complete
            and batches[-1].tail_included and batches[-1].decoded_source_frame_count == len(frames)
            and batches[-1].delivered_count == len(frames)
            and all(batch.views_per_frame == 1 and all(transform.view_id == "global"
                and tuple(transform.source_box_xyxy) == (0., 0., 1., 1.) for transform in batch.transforms)
                for batch in batches), "repeat_incomplete_or_non_global_source")
        sequence, _, spec_sha256 = sequence_from_source_frames(asset, frames, group_id=row["group_id"],
            split=row["split"], encoder_recipe=encoder.recipe, decoder_recipe=decoder,
            sampling_recipe=sampling, crop_recipe=crops)
        features = torch.cat([batch.features for batch in batches], dim=0).unsqueeze(1).contiguous()
        timings["fresh_feature_pipeline_sec"] = time.perf_counter() - source_started
        require(spec_sha256 == identity.feature_spec_sha256
            == settings["expected_feature_spec_sha256"], "repeat_feature_spec_checkpoint_mismatch")
        record["phase"] = "head_forward"
        stage = time.perf_counter()
        inputs = sequence_inputs(sequence, features, device=settings["head_device"])
        torch.cuda.synchronize(settings["device"])
        with torch.inference_mode():
            record["stage_calls"]["head"] += 1
            raw_outputs = head(**inputs)
        torch.cuda.synchronize(settings["device"])
        timings["head_forward_sec"] = time.perf_counter() - stage
        timings["source_start_to_binary_output_sec"] = time.perf_counter() - source_started
        require(all(value is None or (isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()))
            for value in raw_outputs.values()), "repeat_nonfinite_head_output")
        record["phase"] = "artifact_write"
        stage = time.perf_counter()
        entry = write_feature_cache(directory / "fresh_features", asset, iter(batches),
            encoder.recipe, decoder, sampling, crops, row["group_id"], row["split"])
        require(entry["feature_spec_sha256"] == spec_sha256, "repeat_saved_feature_binding_mismatch")
        outputs = {key: value.detach().cpu().contiguous() for key, value in raw_outputs.items() if value is not None}
        save_file(outputs, str(directory / "head-output.safetensors"))
        semantics = canonical_semantics({"scope": "weak_binary_output_only",
            "binary_prediction": bool(outputs["video_logit"].item() >= settings["threshold"]),
            "threshold": settings["threshold"], "fact_predictions_eligible": False,
            "event_candidates_eligible": False, "jury_legal_result": None,
            "evidence_jpeg_sha256": None, "evidence_unavailable_reason": "binary_only_path"})
        source_identity = {"content_sha256": sequence.content_sha256,
            "time_base_num": sequence.time_base_num, "time_base_den": sequence.time_base_den,
            "origin_pts": sequence.origin_pts, "frames": [asdict(frame) for frame in sequence.frames]}
        write_new(directory / "canonical-private.json", {"source_identity": source_identity,
            "decoded_rgb": rgb_hashes, "semantics": semantics})
        timings["artifact_write_sec"] = time.perf_counter() - stage
        record.update(status="done", phase="complete", fresh_stages_verified=True,
            feature_sha256=tensor_digest(features), output_sha256=digest({key:
                None if value is None else tensor_digest(value) for key, value in raw_outputs.items()}),
            source_identity_sha256=digest(source_identity),
            decoded_rgb_sha256=digest(rgb_hashes), semantic_sha256=digest(semantics),
            frame_count=len(sequence.frames), feature_shape=list(features.shape),
            actual_head_weights_sha256=identity.head_sha256,
            encoder_files_verified=verification.get("local_model_loaded") is True)
    except Exception as error:
        # No decoder/ML exception text, source path, tensor or prediction goes to stdout.
        record["error_code"] = str(error) if isinstance(error, RepeatabilityError) else "repeat_stage_failed"
    finally:
        timings["whole_attempt_sec"] = time.perf_counter() - started
        write_new(directory / "attempt.json", record)
    return record


def compare_records(plan, records):
    good = [row for row in records if row.get("status") == "done"]
    if len(good) < 2:
        return []
    from safetensors.torch import load_file
    import torch
    root = private_path(plan["private_root"])
    comparisons = []
    for first, second in itertools.combinations(good, 2):
        row = {key + "_equal": first[key + "_sha256"] == second[key + "_sha256"]
            for key in ("source_identity", "decoded_rgb", "semantic")}
        row.update(feature_bits_equal=first["feature_sha256"] == second["feature_sha256"],
            output_bits_equal=first["output_sha256"] == second["output_sha256"],
            first_attempt_id=first["attempt_id"], second_attempt_id=second["attempt_id"])
        for kind, relative in (("feature", "fresh_features/features.safetensors"),
                               ("output", "head-output.safetensors")):
            left, right = (load_file(str(root / record["attempt_id"] / relative), device="cpu")
                           for record in (first, second))
            shape_match = set(left) == set(right) and all(left[key].shape == right[key].shape for key in left)
            row[kind + "_max_abs_difference"] = max((float((left[key].to(torch.float64)
                - right[key].to(torch.float64)).abs().max()) for key in left), default=0.) if shape_match else None
        comparisons.append(row)
    return comparisons


def execute(args):
    plan = validate_plan(read(args.plan))
    require(args.gpu_window_released, "repeat_gpu_release_required")
    verify_bindings(plan)
    receipt = read(private_path(args.receipt))
    require(file_digest(args.receipt) == args.receipt_sha256
        and receipt.get("registered") is True and receipt.get("conditions_sha256") == plan["conditions_sha256"]
        and receipt.get("content_sha256") == plan["source"]["content_sha256"]
        and receipt.get("attempt_ids") == [row["attempt_id"] for row in plan["attempts"]],
        "repeat_exposure_receipt_mismatch")
    root = private_path(plan["private_root"])
    require(not root.exists() and not args.report.exists(), "repeat_execution_already_exists")
    root.mkdir(parents=True)
    write_new(root / "execution-authorization.json", {"conditions_sha256": plan["conditions_sha256"],
        "gpu_window_released": True, "attempt_ids": receipt["attempt_ids"],
        "exposure_receipt_sha256": args.receipt_sha256})
    records, shared_runtime = [], {}
    try:
        for attempt in plan["attempts"]:
            records.append(run_attempt(plan, attempt, shared_runtime))
    finally:
        encoder = shared_runtime.get("encoder")
        if encoder is not None:
            encoder.close()
    comparisons = compare_records(plan, records)
    write_new(root / "private-results.json", {"records": records, "comparisons": comparisons})
    report = summarize(plan, records, comparisons)
    report["model_load_calls"] = {"encoder": shared_runtime.get("encoder_load_calls", 0),
        "head": shared_runtime.get("head_load_calls", 0)}
    report["observed_process_count"] = len({row["process_id"] for row in records})
    report["persistent_runtime_verified"] = report["model_load_calls"] == {"encoder": 1, "head": 1} \
        and report["observed_process_count"] == 1
    if not report["persistent_runtime_verified"]:
        report["status"], report["complete_three_attempt_evidence"] = "incomplete", False
    write_new(args.report, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runtime = sub.add_parser("runtime", help="write metadata-only runtime identity; no inference")
    runtime.add_argument("--output", required=True, type=Path)
    prep = sub.add_parser("prepare", help="freeze source/artifacts/three attempts; no inference")
    for name in ("inputs", "checkpoint", "snapshot", "encoder-manifest", "taxonomy", "rules",
                 "runtime", "deployment-proof", "plan", "private-root"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--source-index", type=int, required=True)
    prep.add_argument("--batch-size", type=int, default=32)
    prep.add_argument("--source-timeout-sec", type=float, default=120.)
    prep.add_argument("--device", default="cuda:0")
    reservation = sub.add_parser("reserve", help="reserve all three attempts on central exposure registry")
    reservation.add_argument("--plan", required=True, type=Path)
    reservation.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    reservation.add_argument("--registry", type=Path, default=Path("benchmarks/private/model-exposure.jsonl"))
    reservation.add_argument("--receipt", required=True, type=Path)
    launch = sub.add_parser("execute", help="requires actual checkpoint, exposure reservation and explicit released window")
    launch.add_argument("--plan", required=True, type=Path)
    launch.add_argument("--receipt", required=True, type=Path)
    launch.add_argument("--receipt-sha256", required=True)
    launch.add_argument("--report", required=True, type=Path)
    launch.add_argument("--gpu-window-released", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "runtime":
            write_new(private_path(args.output), runtime_identity())
            report = {"status": "runtime_metadata_captured", "model_calls_made": 0}
        elif args.command == "prepare":
            report = prepare(args)
        elif args.command == "reserve":
            report = reserve(args)
        elif args.command == "execute":
            report = execute(args)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        code = str(error) if isinstance(error, RepeatabilityError) else "repeat_operation_failed"
        print(json.dumps({"status": "failed", "error_code": code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
