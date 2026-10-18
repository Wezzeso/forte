"""Metadata-only W2 target/source join; never a compute authorization."""
import argparse
import json
from pathlib import Path

from training.guards import TrainingGateError, file_sha256, load_annotations, load_json, require
from training.temporal.axes import load_axes
from training.temporal.protocol import validate_protocol_config, require_protocol_supervision, validate_question_policy
from training.temporal.supervision import class_coverage
from training.temporal.train_weak import prepare_supervision
from training.temporal.weak_preflight import resolve_config


def inspect(config: dict) -> dict:
    config = resolve_config(config)
    protocol = validate_protocol_config(config)
    question_policy = validate_question_policy(config)
    path = Path(config["provided_train_labels"])
    require(file_sha256(path) == config["experiment_authorization"]["expected_provided_label_manifests_sha256"]["train"],
            "weak_supervision_inspection_train_role_pin_mismatch")
    labels = load_annotations(path)
    facts, events = load_axes(config)
    data, coverage, conflicts = prepare_supervision(config, labels, facts, events, metadata_only=True)
    blocker = None
    try:
        require_protocol_supervision(protocol, coverage)
    except TrainingGateError as exc:
        blocker = str(exc)
    return {"schema_version": "forte.weak-supervision-inspection.v1", "experiment_id": config["experiment_id"],
        "gpu_used": False, "feature_tensors_opened": False, "model_weights_opened": False,
        "fresh_exposure_training_gate_run": False, "training_authorized_by_this_report": False,
        "validation_or_final_labels_opened": False, "teacher_export_sha256": config["expected_machine_annotations_sha256"],
        "feature_cache_index_sha256": config["expected_feature_cache_index_sha256"],
        "protocol_manifest_sha256": config["expected_protocol_manifest_sha256"],
        "anchor_question_policy_file_sha256": config.get("expected_anchor_question_policy_sha256"),
        "anchor_question_semantic_policy_sha256": question_policy.get("policy_sha256") if question_policy else None,
        "declared_supervision_available": blocker is None, "declared_supervision_blocker_code": blocker,
        "coverage": coverage, "dataset_conflict_mask": conflicts,
        "class_coverage": class_coverage(data, fact_ids=facts, event_ids=events)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inspect(load_json(args.config))
        require(not args.output.exists(), "weak_supervision_inspection_output_exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"gpu_used": False, "teacher_export_sha256": report["teacher_export_sha256"],
                          "coverage": report["coverage"], "inspection_sha256": file_sha256(args.output)}, sort_keys=True))
        return 0 if report["declared_supervision_available"] else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "private_values_printed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
