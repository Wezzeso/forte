"""Rehearse selected-checkpoint APIs on development data without opening sealed pixels.

RF-DETR 1.11.2 documents evaluate(split="val") as the valid/ dataset via
trainer.validate: https://github.com/roboflow/rf-detr/blob/1.11.2/src/rfdetr/detr.py
This separate runner does not change the frozen fitting or final-test code.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import time

from training.guards import TrainingGateError
from .common import cli_error, file_sha256, load_json, number, pinned_json, relative_file, require, write_json
from .public_train import (RFDETR_VERSION, cli_deadline_watchdog,
                           fixed_detection_evaluation, preflight,
                           remaining_seconds, scalar_metrics)


def pending_inputs(run_path: Path, expected_run_sha256: str) -> tuple[dict, dict, Path]:
    """Pin existing metadata and weights; do not inspect any dataset pixels."""
    run_path = run_path.resolve()
    run = pinned_json(run_path, expected_run_sha256)
    require(run.get("schema_version") == "forte.specialists.public-run.v1"
            and run.get("state") == "trained_unselected_pending_test"
            and run.get("defer_sealed_test") is True and run.get("test_evaluation_count") == 0
            and run.get("serving_eligible") is False and run.get("rfdetr_version") == RFDETR_VERSION,
            "public_dev_run_not_pending")
    config_path = relative_file(run_path.parent, "config.json")
    config = pinned_json(config_path, run.get("config_sha256", ""))
    checkpoint = relative_file(run_path.parent, run.get("checkpoint"))
    require(file_sha256(checkpoint) == run.get("checkpoint_sha256"), "public_dev_checkpoint_changed")
    remaining_seconds(config)
    return run, config, checkpoint


def development_view(lock: dict) -> dict:
    """Copy only included validation rows, retaining their valid/ image paths.

    The fixed scorer's internal split name is sealed_test. This temporary
    in-memory adapter uses that name only for development records; its results
    are explicitly relabeled validation before being written.
    """
    rows = [copy.deepcopy(row) for row in lock.get("records", [])
            if row.get("split") == "validation" and row.get("included") is True]
    require(bool(rows), "public_dev_validation_records_missing")
    for row in rows:
        path = PurePosixPath(row.get("export_path", ""))
        require(not path.is_absolute() and path.parts and path.parts[0] == "valid"
                and ".." not in path.parts and "\\" not in str(path), "public_dev_image_path_not_validation")
        row["split"] = "sealed_test"
    result = {key: copy.deepcopy(lock[key]) for key in ("schema_version", "state", "origin", "classes", "class_names")}
    result.update(records=rows, evaluation_scope="development_only",
                  files={path: digest for path, digest in lock.get("files", {}).items()
                         if PurePosixPath(path).parts and PurePosixPath(path).parts[0] == "valid"})
    return result


def configure_gpu(config: dict):
    """Use the frozen run's package, allocator budget, headroom, and deadline."""
    require(importlib.metadata.version("rfdetr") == RFDETR_VERSION, "public_dev_installed_package_mismatch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "public_dev_bf16_cuda_required")
    remaining_seconds(config)
    free, total = torch.cuda.mem_get_info(0)
    cap = config["gpu_memory_budget_mib"] * 1024**2
    require(free >= cap + config["minimum_free_margin_mib"] * 1024**2,
            "public_dev_shared_gpu_headroom_insufficient")
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    return torch


def selected_model(checkpoint: Path):
    from rfdetr import RFDETR
    return RFDETR.from_checkpoint(str(checkpoint), device="cuda:0")


def model_device_snapshot(selected, phase: str) -> dict:
    """Inspect the wrapper target and real model tensors without moving them."""
    context = getattr(selected, "model", None)
    module = getattr(context, "model", None)
    require(context is not None and module is not None and callable(getattr(module, "parameters", None)),
            "public_dev_model_device_shape_invalid")
    parameters = list(module.parameters())
    buffers = list(module.buffers()) if callable(getattr(module, "buffers", None)) else []
    require(bool(parameters), "public_dev_model_device_shape_invalid")
    return {"phase": phase, "wrapper_model_device": str(getattr(context, "device", None)),
            "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
            "buffer_devices": sorted({str(buffer.device) for buffer in buffers}),
            "parameter_tensors": len(parameters), "buffer_tensors": len(buffers)}


def phase_timing(wall_seconds: float, development_images: int) -> dict:
    require(number(wall_seconds, 0) and type(development_images) is int and development_images >= 0,
            "public_dev_phase_timing_invalid")
    throughput = development_images / wall_seconds if wall_seconds else 0.0
    require(number(throughput, 0), "public_dev_phase_throughput_invalid")
    return {"wall_seconds": wall_seconds, "development_images": development_images,
            "images_per_second": throughput}


def validate_development(run_path: Path, expected_run_sha256: str, output: Path) -> dict:
    run_path, output = run_path.resolve(), output.resolve()
    run, config, checkpoint = pending_inputs(run_path, expected_run_sha256)
    root, lock = preflight(config)  # verify_fit_bundle never opens sealed image bytes.
    require(run.get("dataset_lock_sha256") == config["expected_dataset_lock_sha256"]
            and run.get("class_names") == lock["class_names"], "public_dev_dataset_binding_mismatch")
    protocol = relative_file(run_path.parent, "evaluation-protocol.json")
    require(file_sha256(protocol) == run.get("evaluation_protocol_sha256"), "public_dev_protocol_changed")
    view = development_view(lock)
    require(not output.exists() and not output.is_relative_to(root.resolve()), "public_dev_output_not_new_or_inside_dataset")
    output.mkdir(parents=True)
    receipt = {"schema_version": "forte.specialists.public-dev-rehearsal.v1", "state": "started",
               "scope": "development_only", "serving_eligible": False, "approved_types": [],
               "test_evaluation_count": 0, "sealed_pixels_accessed": False,
               "run_sha256": expected_run_sha256, "checkpoint_sha256": run["checkpoint_sha256"],
               "config_sha256": run["config_sha256"], "dataset_lock_sha256": run["dataset_lock_sha256"],
               "evaluation_protocol_sha256": run["evaluation_protocol_sha256"],
               "rfdetr_version": RFDETR_VERSION, "source_split": "validation",
               "development_images": len(view["records"]), "checkpoint_selection_changed": False,
               "intended_prediction_device": "cuda:0", "gpu_device_snapshots": [], "phase_timings": {}}
    write_json(output / "receipt.json", receipt)
    started, stage = time.monotonic(), "gpu_setup"
    try:
        torch = configure_gpu(config)
        stage = "from_checkpoint"
        selected = selected_model(checkpoint)
        require(list(selected.class_names) == lock["classes"], "public_dev_selected_class_mapping_mismatch")
        before = model_device_snapshot(selected, "before_native_evaluate")
        receipt["gpu_device_snapshots"].append(before)
        require(before["wrapper_model_device"] in {"cuda", "cuda:0"}, "public_dev_wrapper_target_not_cuda_zero")
        remaining_seconds(config)
        stage = "native_evaluate_val"
        torch.cuda.synchronize()
        native_started = time.monotonic()
        native = selected.evaluate(dataset_dir=str(root), output_dir=str(output / "native-dev"),
                                   split="val", device="cuda:0", batch_size=1, eval_batch_size=1,
                                   amp_dtype="bf16", num_workers=0, multi_scale=False,
                                   eval_max_dets=500, log_per_class_metrics=True)
        torch.cuda.synchronize()
        receipt["phase_timings"][stage] = phase_timing(time.monotonic() - native_started, len(view["records"]))
        receipt["gpu_device_snapshots"].append(model_device_snapshot(selected, "after_native_evaluate"))
        require(isinstance(native, dict) and bool(native) and all(isinstance(key, str) for key in native),
                "public_dev_native_metrics_shape_invalid")
        native_metrics = scalar_metrics(native)
        require(len(native_metrics) == len(native) and any(key.startswith("val/") for key in native_metrics),
                "public_dev_native_metrics_values_invalid")
        write_json(output / "native-selected-dev-metrics.json", native_metrics)
        remaining_seconds(config)
        stage = "fixed_detection_evaluation"

        class FirstPredictionAudit:
            def __init__(self):
                self.seen = False

            def predict(self, *args, **kwargs):
                prediction = selected.predict(*args, **kwargs)
                if not self.seen:
                    self.seen = True
                    snapshot = model_device_snapshot(selected, "after_first_fixed_predict")
                    receipt["gpu_device_snapshots"].append(snapshot)
                    devices = snapshot["parameter_devices"] + snapshot["buffer_devices"]
                    require(snapshot["wrapper_model_device"] in {"cuda", "cuda:0"}
                            and bool(devices) and all(device in {"cuda", "cuda:0"} for device in devices),
                            "public_dev_first_prediction_not_cuda_zero")
                return prediction

        prediction_audit = FirstPredictionAudit()
        torch.cuda.synchronize()
        fixed_started = time.monotonic()
        fixed, predictions = fixed_detection_evaluation(prediction_audit, root, view)
        torch.cuda.synchronize()
        receipt["phase_timings"][stage] = phase_timing(time.monotonic() - fixed_started, len(view["records"]))
        require(prediction_audit.seen, "public_dev_first_prediction_missing")
        require(isinstance(fixed, dict) and isinstance(predictions, list)
                and len(predictions) == len(view["records"]), "public_dev_fixed_metrics_shape_invalid")
        fixed = {**fixed, "scope": "development_only", "task": "public_object_detection_not_bank_violation",
                 "split": "validation", "scorer_internal_split": "sealed_test"}
        write_json(output / "fixed-development-predictions.json", predictions)
        write_json(output / "fixed-development-metrics.json", fixed)
        stage = "immutable_artifact_recheck"
        pending_inputs(run_path, expected_run_sha256)
        preflight(config)
        require(file_sha256(protocol) == run["evaluation_protocol_sha256"], "public_dev_protocol_changed")
        remaining_seconds(config)
        receipt.update(state="complete", native_selected_dev_metrics=native_metrics,
                       fixed_development_metrics=fixed, api_shape_errors=[],
                       fixed_predictions_sha256=file_sha256(output / "fixed-development-predictions.json"),
                       native_metrics_sha256=file_sha256(output / "native-selected-dev-metrics.json"),
                       fixed_metrics_sha256=file_sha256(output / "fixed-development-metrics.json"),
                       max_memory_allocated_mib=torch.cuda.max_memory_allocated(0) / 1024**2,
                       max_memory_reserved_mib=torch.cuda.max_memory_reserved(0) / 1024**2,
                       elapsed_s=time.monotonic() - started)
        write_json(output / "receipt.json", receipt)
        return {"ok": True, "state": "complete", "scope": "development_only", "serving_eligible": False,
                "test_evaluation_count": 0, "receipt_sha256": file_sha256(output / "receipt.json")}
    except Exception as exc:
        error = str(exc) if isinstance(exc, TrainingGateError) else "public_dev_api_execution_failed"
        receipt.update(state="failed", failure_stage=stage, error=error, error_type=type(exc).__name__,
                       api_shape_errors=[{"stage": stage, "code": error, "error_type": type(exc).__name__}],
                       elapsed_s=time.monotonic() - started)
        write_json(output / "receipt.json", receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-run-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        _, config, _ = pending_inputs(args.run, args.expected_run_sha256)
        # Redirect only the watchdog's optional failure receipt, preserving the
        # pinned config and original run even when the absolute deadline fires.
        watchdog = cli_deadline_watchdog({**config, "output_dir": str(args.output.resolve())})
        try:
            result = validate_development(args.run, args.expected_run_sha256, args.output)
        finally:
            watchdog.cancel()
        print(json.dumps(result))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
