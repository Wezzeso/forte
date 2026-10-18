"""CPU-only validation of the independently trained text procedural Q30 LoRA.

Reuses the pinned Q30 physical tensor/base validator, without its DESK serving
metadata. No model loading, inference, media, bank labels or merging occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import qwen30_adapter_export as physical

PHYSICAL_SHA256 = "c2d4919416003a7bdebb3025f410816628aa42fb8a199e284906d43c543b4c88"


def validate(run_root: Path, base: Path, base_receipt: Path) -> dict:
    require = physical.require
    require(physical.file_sha256(Path(physical.__file__)) == PHYSICAL_SHA256,
            "physical_validator_source_mismatch")
    run = physical._read_json(run_root / "run.json")
    require(run.get("schema_version") == "forte.qwen30-procedure-lora.run.v1",
            "procedure_run_schema_mismatch")
    require(run.get("state") in {"trained_not_qualified", "stopped_budget", "stopped_throughput_pilot"}
            and run.get("mode") == "train",
            "procedure_training_not_complete")
    require(type(run.get("optimizer_steps")) is int and 0 < run["optimizer_steps"] <= 100,
            "procedure_positive_actual_steps_required")
    require(run["optimizer_steps"] != 100 or run["state"] == "trained_not_qualified",
            "procedure_terminal_state_mismatch")
    require(run["optimizer_steps"] == 100 or run["state"] in {"stopped_budget", "stopped_throughput_pilot"},
            "procedure_partial_terminal_state_mismatch")
    require(run.get("qualification_state") == "not_evaluated" and
            run.get("production_enabled") is False and run.get("bank_video_labels_used") is False,
            "procedure_candidate_policy_mismatch")
    require(run.get("model_id") == physical.BASE_REPO and run.get("revision") == physical.BASE_REVISION,
            "procedure_training_base_mismatch")
    require(run.get("quantization") == "bf16" and run.get("base_dtype") == "bfloat16",
            "procedure_training_dtype_mismatch")
    require(run.get("adapter_scope") == "text_only_procedure_reasoning" and
            run.get("target_loss_policy") == "procedural_response_values_only" and
            run.get("training_data_policy") == "independent_synthetic_typed_timelines",
            "procedure_training_scope_mismatch")
    for key in ("frozen_before_sha256", "frozen_after_sha256", "initial_adapter_sha256",
                "final_adapter_sha256", "weight_manifest_sha256", "source_sha256",
                "config_sha256", "manifest_sha256"):
        require(isinstance(run.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", run[key]),
                "procedure_training_hash_missing")
    require(run["frozen_before_sha256"] == run["frozen_after_sha256"] and
            run["initial_adapter_sha256"] != run["final_adapter_sha256"],
            "procedure_frozen_or_adapter_change_proof_failed")
    require(run["weight_manifest_sha256"] == physical.BASE_WEIGHT_MANIFEST_SHA256 and
            run.get("trainable_parameters") == 3342336,
            "procedure_training_physical_scope_mismatch")
    metrics = run.get("steps")
    require(isinstance(metrics, list) and len(metrics) == run["optimizer_steps"],
            "procedure_actual_step_metrics_missing")
    for number, metric in enumerate(metrics, 1):
        require(isinstance(metric, dict) and metric.get("step") == number,
                "procedure_step_sequence_invalid")
        for key in ("mean_loss", "gradient_norm", "learning_rate", "seconds",
                    "peak_allocated_mib", "peak_reserved_mib"):
            value = metric.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "procedure_step_metric_invalid")
    directory = run_root / "adapter"
    require(directory.is_dir() and not directory.is_symlink(), "procedure_adapter_directory_invalid")
    files = run.get("adapter_files")
    require(isinstance(files, dict) and {"adapter_config.json", "adapter_model.safetensors"} <= set(files),
            "procedure_adapter_inventory_missing")
    require(set(files) == {p.name for p in directory.iterdir() if p.is_file()},
            "procedure_adapter_inventory_mismatch")
    for name, expected in files.items():
        require(isinstance(name, str) and Path(name).name == name and "/" not in name and "\\" not in name,
                "procedure_adapter_filename_invalid")
        path = directory / name
        require(path.is_file() and not path.is_symlink() and physical.file_sha256(path) == expected,
                "procedure_adapter_file_hash_mismatch")
    text, base_result = physical.validate_base(base, base_receipt)
    adapter, tensors = physical.validate_adapter(directory, text)
    require(adapter["tensor_count"] == 192 and adapter["parameter_count"] == 3342336,
            "procedure_adapter_tensor_scope_mismatch")
    return {"schema": "forte.qwen30-procedure-adapter-validation.v1", "status": "validated",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "base": base_result,
        "adapter": adapter, "tensor_metadata": tensors,
        "adapter_sha256": physical.canonical_sha256({"adapter_config.json": adapter["config_sha256"],
            "adapter_model.safetensors": adapter["weights_sha256"]}),
        "training_run_receipt_sha256": physical.file_sha256(run_root / "run.json"),
        "training_source_sha256": run["source_sha256"], "training_config_sha256": run["config_sha256"],
        "training_manifest_sha256": run["manifest_sha256"], "optimizer_steps": run["optimizer_steps"],
        "training_state": run["state"],
        "training_completion_status": "completed_100_not_qualified" if run["optimizer_steps"] == 100
            else "partial_not_qualified",
        "frozen_base_unchanged": True, "trained_adapter_changed": True,
        "adapter_scope": "text_only_procedure_reasoning", "visual_perception_improvement_claim": False,
        "default_enabled": False, "qualification_status": "unqualified" if run["optimizer_steps"] == 100
            else "partial_not_qualified",
        "runtime_load_verified": False, "inference_executed": False,
        "physical_validator_sha256": PHYSICAL_SHA256}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Output belongs beside the run, never inside the adapter or base snapshot.
    if args.output.resolve().parent != args.run_root.resolve().parent or args.output.resolve() in {
            args.base_receipt.resolve(), (args.run_root / "run.json").resolve()}:
        print(json.dumps({"status": "failed", "failure_code": "procedure_receipt_location_invalid"}))
        return 2
    try:
        result = validate(args.run_root, args.base_dir, args.base_receipt)
    except physical.AdapterExportError as error:
        result = {"schema": "forte.qwen30-procedure-adapter-validation.v1", "status": "failed",
                  "failure_code": str(error), "default_enabled": False}
    except Exception:
        result = {"schema": "forte.qwen30-procedure-adapter-validation.v1", "status": "failed",
                  "failure_code": "procedure_validation_operation_failed", "default_enabled": False}
    physical.write_receipt(args.output, result)
    print(json.dumps({key: result[key] for key in
        ("status", "failure_code", "adapter_sha256", "optimizer_steps") if key in result}, sort_keys=True))
    return 0 if result["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
