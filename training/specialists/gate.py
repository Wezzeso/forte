"""Issue a class-level approval only after recomputing hash-bound holdout metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (cli_error, file_sha256, load_json, relative_file,
                     require, write_json)
from .evaluate import score_predictions
from .prepare import verify_dataset
from .train import preflight, verify_run


def gate(run_path: Path, metrics_path: Path, *, video_validation: Path | None = None,
         baseline_video_validation: Path | None = None, requested_types: list[str] | None = None) -> dict:
    run, _ = verify_run(run_path)
    config = load_json(relative_file(run_path.parent, "config.json"))
    _, lock = preflight(config)
    require(run["dataset_lock_sha256"] == config["expected_dataset_lock_sha256"]
            and run["class_names"] == lock["class_names"]
            and run["thresholds"] == config["thresholds"]
            and run["gate_policy"] == config["gate_policy"], "specialist_run_identity_mismatch")
    metrics = load_json(metrics_path)
    require(metrics.get("schema_version") == "forte.specialists.metrics.v1"
            and metrics.get("state") == "complete", "specialist_metrics_incomplete")
    for field in ("checkpoint_sha256", "dataset_lock_sha256", "config_sha256", "class_names", "model_class", "thresholds"):
        require(metrics.get(field) == run.get(field), "specialist_metrics_identity_mismatch")
    require(metrics.get("run_sha256") == file_sha256(run_path), "specialist_metrics_run_hash_mismatch")
    predictions_path = relative_file(metrics_path.parent, metrics.get("predictions"))
    require(file_sha256(predictions_path) == metrics.get("predictions_sha256"), "specialist_predictions_hash_mismatch")
    scores = score_predictions(lock, load_json(predictions_path), run["thresholds"])
    require(all(metrics.get(field) == value for field, value in scores.items()), "specialist_metrics_recompute_mismatch")
    policy = config["gate_policy"]
    approved = [name for name, result in scores["per_class"].items()
                if result["positive_instances"] >= policy["minimum_positive_instances"]
                and result["recall"] is not None and result["recall"] >= policy["minimum_recall"]
                and result["fp"] == 0]
    reasons = []
    if not {"id_card", "key_in_lock"} <= set(approved):
        reasons.append("required_classes_recall_or_support_failed")
    if scores["total_false_positives"] != 0:
        reasons.append("heldout_false_positives")
    if scores["clean_images"] < policy["minimum_clean_images"]:
        reasons.append("heldout_clean_support_insufficient")
    eligible = not reasons
    result = {"schema_version": "forte.specialists.gate.v1", "eligible": eligible,
            "scope": "object_classes_only", "approved_types": [],
            "approved_classes": sorted(approved) if eligible else [], "reasons": reasons,
            "checkpoint_sha256": run["checkpoint_sha256"], "model_class": run["model_class"],
            "class_names": run["class_names"], "rfdetr_version": run["rfdetr_version"],
            "resolution": run["resolution"], "dataset_lock_sha256": run["dataset_lock_sha256"],
            "config_sha256": run["config_sha256"], "run_sha256": file_sha256(run_path),
            "metrics_sha256": file_sha256(metrics_path), "thresholds": run["thresholds"],
            "heldout_data_sha256": scores["heldout_data_sha256"],
            "evaluation_protocol_sha256": scores["evaluation_protocol_sha256"],
            "per_class": scores["per_class"], "total_false_positives": scores["total_false_positives"],
            "clean_images": scores["clean_images"],
            "limitation": "Class detection approval does not establish whole-video violation accuracy."}
    if video_validation or baseline_video_validation or requested_types:
        require(eligible, "specialist_video_class_gate_failed")
        require(video_validation is not None and baseline_video_validation is not None and requested_types,
                "specialist_video_promotion_inputs_required")
        from .video_gate import approve_video_types
        result.update(approve_video_types(video_validation, baseline_video_validation, requested_types, run))
        result["scope"] = "object_classes_and_validated_video_types"
        result["limitation"] = "Video type approval covers the bound forty-video evaluation only."
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Path to the trained run.json")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-validation", type=Path)
    parser.add_argument("--baseline-video-validation", type=Path)
    parser.add_argument("--approve-type", action="append", default=[])
    args = parser.parse_args()
    try:
        result = gate(args.run.resolve(), args.metrics.resolve(), video_validation=args.video_validation,
                      baseline_video_validation=args.baseline_video_validation, requested_types=args.approve_type)
        require(not args.output.exists(), "specialist_gate_output_already_exists")
        write_json(args.output, result)
        print(json.dumps({"ok": result["eligible"], "eligible": result["eligible"], "reasons": result["reasons"]}))
        return 0 if result["eligible"] else 1
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
