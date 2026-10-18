"""Preserve selected public-training weights and sanitized receipts locally.

No dataset, image, log, base-model snapshot, or remote deletion is supported.
Transfers require at least 4 GiB of local free space after the planned copy.
The remote run must already be successful and pinned by its exact run.json SHA.
The frozen training receipt does not establish whether a later, separate sealed
evaluation occurred; this helper neither reads test-access markers nor test data.
The fixed SSH alias is reached through Ubuntu WSL on Windows, directly on Linux.
An externally reviewed index can instead be supplied with its own SHA256 pin.
Extra JSON receipts use --extra-json /absolute/authorized/path.json=sha256.
Source archives and dependency environments are intentionally left to the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

HOST_ALIAS = "forte-a100-smoke-20261017"
REMOTE_ROOT = PurePosixPath("/data/claude/training/public-20261018")
REMOTE_RUNS = REMOTE_ROOT / "runs"
LOCAL_ROOT = Path(__file__).resolve().parents[1] / "training/private/public_runs/20261018"
SHA = re.compile(r"[a-f0-9]{64}\Z")
REMOTE_SAFE = re.compile(r"/[a-zA-Z0-9_./-]+\Z")
SAFE_NAME = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,159}\Z")
RESERVE_BYTES = 4 * 1024**3
MAX_ARTIFACT_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 4 * 1024**3
MAX_JSON_BYTES = 8 * 1024**2
SCHEMA = "forte.public-training-preservation-index.v1"
RECEIPTS = {
    "rf": ("config.json", "run.json", "effective-recipe.json", "training-data-policy.json",
           "evaluation-protocol.json", "baseline-dev.json", "progress.json"),
    "crop": ("config.json", "run.json", "evaluation-protocol.json", "development-baseline.json",
             "development-adapter.json", "step-proof.json"),
}
# Unknown top-level fields never cross the metadata channel. Nested values are
# bounded JSON with credential-shaped keys removed, rather than arbitrary text.
TOP_FIELDS = set("""schema_version state scope mode serving_eligible promotion_eligible approved_types
training_completed memory_smoke_only memory_recipe model_manifest_sha256 crop_manifest_sha256
sealed_test_used_for_training_or_selection audio_used rfdetr_version model_class model_id revision class_names classes
dataset_lock_sha256 dataset_sha256 config_sha256 runner_sha256 pretrain_weights_sha256
evaluation_protocol_sha256 recipe_reference baseline_scope test_evaluation_count defer_sealed_test
optimizer_steps physical_batches epoch recent_losses elapsed_s max_memory_allocated_mib
max_memory_reserved_mib fit_wall_s checkpoint checkpoint_sha256 final_epoch_dev_metrics
selected_checkpoint_dev_metric final_trainable_parameters_sha256 final_frozen_parameters_sha256
initial_trainable_parameters_sha256 initial_frozen_parameters_sha256 metrics model training
class_balanced_sampling small_object_zoomout train_data_policy runtime_versions recipe_predeclared
verified_dataset_import_reference expected_dataset_lock_sha256 expected_classes expected_pretrain_weights_sha256
freeze_encoder gradient_checkpointing compile cuda_graphs amp_dtype batch_size eval_batch_size grad_accum_steps
multi_scale use_ema epochs resolution seed lr gpu_memory_budget_mib minimum_free_margin_mib max_fit_seconds
evaluation_reserve_seconds stop_at_utc early_stopping_patience device train_short_side_curriculum
evaluation_short_sides optimizer_steps lora_rank lora_alpha max_length max_pixels train_per_class
validation_per_class test_per_class learning_rate runtime_image_digest expected_model_manifest_sha256
expected_memory_smoke_run_sha256 development_baseline_accuracy development_adapter_accuracy development_improved
fit_walltime_seconds total_command_walltime_seconds development_inference_timing peak_allocated_mib
peak_reserved_mib adapter_files step_proof_sha256 adapter_after_sha256 adapter_before_sha256 frozen_before_sha256
frozen_after_sha256 trainable_parameters target_modules gpu cuda training_sampling unique_training_crop_support
planned_training_draw_counts planned_unique_drawn_crops conditions macro_accuracy pooled_accuracy accuracy
correct total per_class timing selection_scope data_policy max_steps progress
phase passed failure image_id source network sealed_test_deferred limits gpu_samples maximum_sampled_training_mib
minimum_sampled_global_free_mib cpu_preflight_exit_code gpu_lock_held production_before production_after
production_unchanged readiness_before readiness_after queue_before queue_after gpu_before gpu_after container_id
training_exit_code final_progress owned_containers_retained_stopped immutable_inputs_unchanged elapsed_seconds
postcheck_failures files file_count manifest_sha256 versions constraints_sha256 requirements_sha256
python torch torchvision transformers peft accelerate bitsandbytes supervision pytorch_lightning
protocol support units primary_metric development_models sealed_test_models accepted_output invalid_output
derived_stress_views_are_independent_examples bank_video_false_alarm_or_accuracy_claim source_sha256
""".split())
PRIVATE_KEY = re.compile(r"secret|password|credential|authorization|api[_-]?key|access[_-]?token|refresh[_-]?token"
                         r"|(?:^|[_-])(?:token|key|passphrase)(?:$|[_-])", re.I)
SAFE_KEY = re.compile(r"[a-zA-Z0-9_./:+() -]{1,160}\Z")


class PreservationError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise PreservationError(code)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_path(value, *, run=False):
    require(isinstance(value, str) and REMOTE_SAFE.fullmatch(value), "unsafe_remote_path")
    path = PurePosixPath(value)
    base = REMOTE_RUNS if run else REMOTE_ROOT
    require(str(path) == value and ".." not in path.parts and path != base
            and path.is_relative_to(base), "remote_path_outside_authorized_scope")
    return path


def read_json(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON_BYTES,
            "receipt_file_invalid_or_oversize")
    try:
        return json.loads(path.read_text())
    except (ValueError, UnicodeError):
        raise PreservationError("receipt_json_invalid") from None


def sanitize(value, *, depth=0):
    require(depth <= 12, "receipt_nesting_exceeds_limit")
    if value is None or type(value) is bool:
        return value
    if type(value) in (float, int):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if len(value) > 1024 or any(x in value for x in ("\x00", "\r", "\n")):
            return None
        # URLs/user-info and bearer strings do not belong in compact receipts.
        if "://" in value or re.search(r"bearer\s|(?:sk|hf)_[A-Za-z0-9]{12,}", value, re.I):
            return None
        return value
    if isinstance(value, list):
        require(len(value) <= 2048, "receipt_collection_exceeds_limit")
        return [sanitize(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        require(len(value) <= 4096, "receipt_collection_exceeds_limit")
        return {key: sanitize(item, depth=depth + 1) for key, item in value.items()
                if isinstance(key, str) and SAFE_KEY.fullmatch(key) and not PRIVATE_KEY.search(key)}
    raise PreservationError("receipt_value_invalid")


def sanitized_receipt(data):
    if isinstance(data, list):  # Numeric optimizer step proofs only.
        return sanitize(data)
    require(isinstance(data, dict), "receipt_object_required")
    return sanitize({key: value for key, value in data.items() if key in TOP_FIELDS})


def verify_success(run, kind):
    require(isinstance(run, dict) and kind in RECEIPTS, "run_kind_invalid")
    schema, state = (("forte.specialists.public-run.v1", "trained_unselected_pending_test") if kind == "rf"
                     else ("forte.public-crop-lora.run.v1", "trained_unselected"))
    require(run.get("schema_version") == schema and run.get("state") == state,
            "successful_deferred_run_required")
    require(type(run.get("optimizer_steps")) is int and run["optimizer_steps"] > 0,
            "real_optimizer_update_required")
    require(not run.get("memory_smoke_only") and run.get("approved_types", []) == []
            and not run.get("promotion_eligible", False) and not run.get("serving_eligible", False),
            "unpromoted_full_fit_required")
    require(run.get("test_evaluation_count", 0) == 0, "frozen_training_receipt_contains_test_results")
    if kind == "rf":
        require(run.get("defer_sealed_test") is True and run.get("mode") == "fixed_recipe_train",
                "rf_sealed_test_must_be_deferred")
    else:
        require(run.get("training_completed") is True, "crop_full_fit_required")


def artifact_names(run, kind):
    if kind == "rf":
        require(run.get("checkpoint") == "checkpoint_best_total.pth", "selected_rf_checkpoint_invalid")
        return {run["checkpoint"]: run.get("checkpoint_sha256")}
    files = run.get("adapter_files")
    require(isinstance(files, dict) and {"adapter_model.safetensors", "adapter_config.json"} <= set(files),
            "compact_adapter_files_missing")
    result = {}
    for name, digest in files.items():
        require(isinstance(name, str) and SAFE_NAME.fullmatch(name), "adapter_file_name_invalid")
        if name == "README.md":
            continue
        require(PurePosixPath(name).suffix in {".safetensors", ".json", ".txt"},
                "adapter_payload_extension_forbidden")
        result["adapter/" + name] = digest
    return result


def build_remote_index(run_root, kind, expected_run_sha256, extras=()):
    root = Path(str(remote_path(run_root, run=True)))
    require(root.is_dir() and root == root.resolve() and not root.is_symlink(), "canonical_remote_run_required")
    run_path = root / "run.json"
    require(SHA.fullmatch(expected_run_sha256) and file_sha256(run_path) == expected_run_sha256,
            "remote_run_hash_mismatch")
    run = read_json(run_path)
    verify_success(run, kind)
    artifacts = []
    for name, digest in artifact_names(run, kind).items():
        require(isinstance(digest, str) and SHA.fullmatch(digest), "artifact_hash_pin_required")
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                "artifact_file_invalid")
        size = path.stat().st_size
        require(0 < size <= MAX_ARTIFACT_BYTES and file_sha256(path) == digest,
                "artifact_size_or_hash_invalid")
        artifacts.append({"relative_path": name, "remote_path": str(path), "bytes": size, "sha256": digest})
    receipts = []
    for name in RECEIPTS[kind]:
        path = root / name
        if path.is_file():
            receipts.append({"relative_path": "receipts/" + name, "remote_sha256": file_sha256(path),
                             "content": sanitized_receipt(read_json(path))})
    require({"receipts/run.json", "receipts/config.json", "receipts/evaluation-protocol.json"}
            <= {row["relative_path"] for row in receipts}, "required_training_receipts_missing")
    for number, specification in enumerate(extras):
        raw_path, separator, digest = specification.rpartition("=")
        require(separator and SHA.fullmatch(digest), "extra_receipt_pin_required")
        path = Path(str(remote_path(raw_path)))
        require(path.suffix == ".json" and path == path.resolve() and file_sha256(path) == digest,
                "extra_receipt_hash_or_path_invalid")
        receipts.append({"relative_path": f"receipts/extra-{number:02d}-{path.name}",
                         "remote_sha256": digest, "content": sanitized_receipt(read_json(path))})
    require(file_sha256(run_path) == expected_run_sha256, "remote_run_changed_during_index")
    return {"schema_version": SCHEMA, "kind": kind, "host_alias": HOST_ALIAS,
            "remote_run_root": str(root), "remote_run_sha256": expected_run_sha256,
            "run_state": run["state"], "artifacts": artifacts, "receipts": receipts,
            "receipt_scope": "frozen_training_metadata_only",
            "remote_originals_retained": True, "images_or_datasets_included": False,
            "promotion_eligible": False}


def validate_index(index, *, run_root, kind, expected_run_sha256):
    require(isinstance(index, dict) and index.get("schema_version") == SCHEMA, "index_schema_invalid")
    require(index.get("host_alias") == HOST_ALIAS and index.get("kind") == kind
            and index.get("remote_run_root") == str(remote_path(run_root, run=True))
            and index.get("remote_run_sha256") == expected_run_sha256, "index_identity_mismatch")
    expected_state = "trained_unselected_pending_test" if kind == "rf" else "trained_unselected"
    require(index.get("run_state") == expected_state and index.get("images_or_datasets_included") is False
            and index.get("remote_originals_retained") is True and index.get("promotion_eligible") is False,
            "index_scope_invalid")
    require(index.get("receipt_scope") == "frozen_training_metadata_only", "index_receipt_scope_invalid")
    rows = index.get("artifacts")
    require(isinstance(rows, list) and 1 <= len(rows) <= 32, "index_artifacts_invalid")
    names = set()
    total = 0
    for row in rows:
        require(isinstance(row, dict), "index_artifact_invalid")
        name = row.get("relative_path")
        require(isinstance(name, str) and name not in names, "duplicate_artifact_path")
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and str(rel) == name and ".." not in rel.parts
                and all(SAFE_NAME.fullmatch(part) for part in rel.parts), "artifact_relative_path_invalid")
        if kind == "rf":
            require(name == "checkpoint_best_total.pth", "selected_rf_checkpoint_invalid")
        else:
            require(len(rel.parts) == 2 and rel.parts[0] == "adapter"
                    and rel.suffix in {".safetensors", ".json", ".txt"}, "adapter_payload_extension_forbidden")
        require(row.get("remote_path") == str(PurePosixPath(run_root) / rel), "artifact_remote_path_mismatch")
        remote_path(row["remote_path"], run=True)
        require(type(row.get("bytes")) is int and 0 < row["bytes"] <= MAX_ARTIFACT_BYTES
                and isinstance(row.get("sha256"), str) and SHA.fullmatch(row["sha256"]),
                "artifact_size_or_hash_invalid")
        total += row["bytes"]
        names.add(name)
    require(total <= MAX_TOTAL_BYTES, "artifact_total_exceeds_limit")
    if kind == "crop":
        require({"adapter/adapter_model.safetensors", "adapter/adapter_config.json"} <= names,
                "compact_adapter_files_missing")
    receipt_names = set()
    receipts = index.get("receipts")
    require(isinstance(receipts, list) and 3 <= len(receipts) <= 32, "index_receipts_invalid")
    for row in receipts:
        require(isinstance(row, dict), "index_receipt_invalid")
        name = row.get("relative_path")
        require(isinstance(name, str) and name.startswith("receipts/") and name.endswith(".json")
                and len(PurePosixPath(name).parts) == 2 and SAFE_NAME.fullmatch(PurePosixPath(name).name)
                and name not in receipt_names, "receipt_relative_path_invalid")
        require(isinstance(row.get("remote_sha256"), str) and SHA.fullmatch(row["remote_sha256"]),
                "receipt_hash_invalid")
        require(sanitized_receipt(row.get("content")) == row.get("content"), "receipt_not_sanitized")
        receipt_names.add(name)
    require({"receipts/run.json", "receipts/config.json", "receipts/evaluation-protocol.json"} <= receipt_names,
            "required_training_receipts_missing")
    by_name = {row["relative_path"]: row for row in receipts}
    selected_run = by_name["receipts/run.json"]
    require(selected_run["remote_sha256"] == expected_run_sha256, "run_receipt_hash_mismatch")
    verify_success(selected_run["content"], kind)
    require({row["relative_path"]: row["sha256"] for row in rows}
            == artifact_names(selected_run["content"], kind), "selected_artifact_pins_mismatch")
    for name, pin in (("config.json", "config_sha256"),
                      ("evaluation-protocol.json", "evaluation_protocol_sha256")):
        require(selected_run["content"].get(pin) == by_name["receipts/" + name]["remote_sha256"],
                "training_receipt_pin_mismatch")
    return total


def transport_prefix():
    return ["wsl", "-d", "Ubuntu", "--"] if os.name == "nt" else []


def run_command(args, *, input_text=None, timeout=120):
    try:
        result = subprocess.run(args, input=input_text, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise PreservationError("preservation_transport_unavailable_or_timed_out") from None
    require(result.returncode == 0, "preservation_transport_failed")
    return result.stdout.strip()


def fetch_remote_index(args):
    source = Path(__file__).read_text()
    # Compile the trusted helper as its own module so its future import remains
    # legal. User path values are Python literals, never shell fragments.
    script = ("namespace={'__name__':'preservation_index_library',"
              "'__file__':'/tmp/preserve_public_training_run.py'}\nexec(compile(" + repr(source)
              + ", '<trusted-preservation-helper>', 'exec'), namespace)\n"
              + "print(namespace['json'].dumps(namespace['build_remote_index']("
              + repr(args.run_root) + "," + repr(args.kind) + "," + repr(args.expected_run_sha256)
              + "," + repr(args.extra_json) + "),sort_keys=True,allow_nan=False))\n")
    raw = run_command(transport_prefix() + ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                                           HOST_ALIAS, "python3", "-"], input_text=script, timeout=300)
    require(len(raw.encode()) <= 16 * 1024**2, "remote_index_exceeds_limit")
    try:
        return json.loads(raw)
    except ValueError:
        raise PreservationError("remote_index_json_invalid") from None


def writable_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as target:
        json.dump(value, target, sort_keys=True, indent=2, allow_nan=False)
        target.write("\n")


def preserve(args):
    remote_path(args.run_root, run=True)
    require(SHA.fullmatch(args.expected_run_sha256), "run_hash_pin_required")
    if args.remote_index:
        require(args.expected_index_sha256 and SHA.fullmatch(args.expected_index_sha256)
                and file_sha256(args.remote_index) == args.expected_index_sha256, "reviewed_index_hash_mismatch")
        index = read_json(args.remote_index)
    else:
        require(not args.expected_index_sha256, "index_pin_without_file")
        index = fetch_remote_index(args)
    total = validate_index(index, run_root=args.run_root, kind=args.kind,
                           expected_run_sha256=args.expected_run_sha256)
    default_name = PurePosixPath(args.run_root).parent.name + "-" + args.kind
    destination = Path(args.destination) if args.destination else LOCAL_ROOT / default_name
    destination = destination.absolute()
    require(destination == destination.resolve() and destination != LOCAL_ROOT
            and destination.is_relative_to(LOCAL_ROOT) and not destination.exists(),
            "fresh_authorized_local_destination_required")
    require(not any(part.lower() in {"videos", "images", "datasets", "secrets"}
                    for part in destination.relative_to(LOCAL_ROOT).parts), "local_destination_scope_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata_bytes = len(json.dumps(index).encode()) * 2 + 16 * 1024**2
    require(shutil.disk_usage(destination.parent).free - total - metadata_bytes >= RESERVE_BYTES,
            "local_disk_reserve_would_be_violated")
    destination.mkdir(mode=0o700)
    writable_json(destination / "preservation-index.json", index)
    results = []
    for row in index["artifacts"]:
        require(shutil.disk_usage(destination).free - row["bytes"] >= RESERVE_BYTES,
                "local_disk_reserve_would_be_violated")
        target = destination / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        incoming = target.with_name(target.name + ".incoming")
        require(not target.exists() and not incoming.exists(), "fresh_local_artifact_required")
        if os.name == "nt":
            # WSL's Windows command bridge treats backslashes as shell escapes.
            # Forward slashes preserve the absolute Windows path through it.
            scp_destination = run_command(["wsl", "-d", "Ubuntu", "--", "wslpath", "-u", "-a", incoming.as_posix()])
            require(scp_destination.startswith("/mnt/") and "\n" not in scp_destination,
                    "wsl_destination_conversion_invalid")
        else:
            scp_destination = str(incoming)
        run_command(transport_prefix() + ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                                          HOST_ALIAS + ":" + row["remote_path"], scp_destination], timeout=1800)
        require(incoming.is_file() and not incoming.is_symlink() and incoming.stat().st_size == row["bytes"]
                and file_sha256(incoming) == row["sha256"], "copied_artifact_size_or_hash_mismatch")
        os.chmod(incoming, 0o600)
        incoming.rename(target)
        results.append({key: row[key] for key in ("relative_path", "bytes", "sha256")})
    for row in index["receipts"]:
        target = destination / row["relative_path"]
        target.parent.mkdir(exist_ok=True, mode=0o700)
        writable_json(target, row["content"])
    report = {"schema_version": "forte.public-training-preservation.v1", "verified": True,
              "kind": args.kind, "remote_run_sha256": args.expected_run_sha256,
              "preservation_index_sha256": file_sha256(destination / "preservation-index.json"),
              "artifacts": results, "sanitized_receipts": len(index["receipts"]),
              "bytes": total, "local_free_after_bytes": shutil.disk_usage(destination).free,
              "minimum_reserve_bytes": RESERVE_BYTES, "remote_originals_retained": True,
              "receipt_scope": "frozen_training_metadata_only",
              "images_or_datasets_included": False, "promotion_eligible": False,
              "destination": str(destination)}
    require(report["local_free_after_bytes"] >= RESERVE_BYTES, "local_disk_reserve_would_be_violated")
    writable_json(destination / "preservation-report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", required=True)
    result.add_argument("--kind", choices=RECEIPTS, required=True)
    result.add_argument("--expected-run-sha256", required=True)
    result.add_argument("--remote-index", type=Path)
    result.add_argument("--expected-index-sha256")
    result.add_argument("--extra-json", action="append", default=[])
    result.add_argument("--destination", type=Path)
    result.add_argument("--remote-index-only", action="store_true", help=argparse.SUPPRESS)
    return result


def main():
    args = parser().parse_args()
    try:
        report = (build_remote_index(args.run_root, args.kind, args.expected_run_sha256, args.extra_json)
                  if args.remote_index_only else preserve(args))
        print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except PreservationError as exc:
        print(json.dumps({"verified": False, "failure": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
