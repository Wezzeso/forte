"""Execute frozen-head validation, exporting private scores for A10 evaluation.

This measures model-only binary predictions. It makes no event/fact accuracy
or upload-to-complete-result latency claim. Final-check data cannot enter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from training.guards import TrainingGateError, canonical_sha256, file_sha256, load_annotations, load_json, require
from training.temporal.weak_preflight import resolve_config, validate_metadata
from training.temporal.train_weak import load_cache_entries, require_compute_release, configure_determinism


def run(config: dict) -> dict:
    config = resolve_config(config, label_split="validation")
    labels, gate = validate_metadata(config, label_split="validation")
    require_compute_release(config)
    deterministic_policy = configure_determinism()
    # Freeze all inputs and the decision rule before any validation forward.
    threshold_path = Path(config["video_threshold"]).resolve()
    require(file_sha256(threshold_path) == config.get("expected_video_threshold_sha256"), "weak_video_threshold_pin_mismatch")
    threshold = load_json(threshold_path)
    require(threshold.get("schema_version") == "forte.weak_video_threshold.v1"
            and threshold.get("score_axis") == "violation_log_odds"
            and threshold.get("comparison") == "greater_than_or_equal"
            and threshold.get("selection_method") == "predeclared_fixed_zero"
            and threshold.get("minimum_video_score") == 0.0
            and threshold.get("validation_labels_used_for_selection") is False,
            "weak_unapproved_video_threshold")
    require(threshold.get("provided_validation_labels_sha256") == config["experiment_authorization"]
            ["expected_provided_label_manifests_sha256"]["validation"], "weak_threshold_validation_role_mismatch")
    plan_path = Path(config["expected_attempts"]).resolve()
    require(file_sha256(plan_path) == config.get("expected_attempts_sha256"), "weak_validation_plan_pin_mismatch")
    plans_raw = load_json(plan_path)
    plans = plans_raw.get("attempts", plans_raw.get("records", [])) if isinstance(plans_raw, dict) else plans_raw
    require(isinstance(plans, list) and bool(plans)
            and len({row["attempt_id"] for row in plans}) == len(plans), "weak_validation_plan_invalid")
    split_rows = load_json(Path(config["automated_split"]))["records"]
    val_map = {row.get("sample_id", row.get("recording_id")): row.get("content_sha256", row.get("content_hash"))
               for row in split_rows if row.get("split") == "validation"}
    label_contents = {row.get("content_sha256", row.get("content_hash")) for row in labels}
    require(all(row["sample_id"] in val_map for row in plans)
            and {val_map[row["sample_id"]] for row in plans} == label_contents,
            "weak_validation_plan_population_mismatch")
    entries = load_cache_entries(config, split="validation", labels=labels)
    entry_map = {row["content_sha256"]: row for row in entries}
    destination = Path(config["validation_output_dir"]).resolve()
    require(not destination.exists(), "weak_validation_output_exists")
    from training.runtime_check import verify_runtime
    verify_runtime(Path(config.get("runtime_lock", "training/runtime-lock.json")), gpu=True)
    import torch
    from training.temporal.cache import load_feature_cache
    from training.temporal.checkpoint import load_checkpoint, sequence_inputs
    from evaluation.weak_supervision import evaluate_weak_binary, provided_label_digest

    device = config.get("device", "cuda:0")
    model, identity, metadata = load_checkpoint(Path(config["checkpoint_dir"]),
        expected_manifest_sha256=config["expected_checkpoint_manifest_sha256"], device=device)
    require(threshold.get("head_sha256") == identity.head_sha256
            and threshold.get("head_identity_sha256") == canonical_sha256(identity.model_dump(mode="json"))
            and threshold.get("feature_spec_sha256") == identity.feature_spec_sha256,
            "weak_threshold_checkpoint_binding_mismatch")
    predictions, temporal_predictions = [], []
    with torch.inference_mode():
        for plan in plans:
            content = val_map[plan["sample_id"]]
            # Every attempt produces an output row, including failures. Private
            # exception messages are never exported into cloud/tool logs.
            row = {"attempt_id": plan["attempt_id"], "sample_id": plan["sample_id"],
                   "status": "failed", "prediction_validated": False, "public_contract_validated": False}
            try:
                sequence, features, manifest = load_feature_cache(Path(entry_map[content]["manifest_path"]))
                require(sequence.content_sha256 == content and sequence.split == "validation"
                        and manifest["feature_spec_sha256"] == identity.feature_spec_sha256,
                        "weak_validation_feature_binding_mismatch")
                inputs = sequence_inputs(sequence, features, device=device)
                torch.cuda.synchronize()
                start = time.monotonic()
                output = model(**inputs)
                score = float(output["video_logit"][0].cpu())
                torch.cuda.synchronize()
                elapsed = (time.monotonic() - start) * 1000
                require(bool(torch.isfinite(output["video_logit"]).all()), "weak_validation_nonfinite_score")
                import hashlib
                def output_digest(values):
                    digest = hashlib.sha256()
                    for name in sorted(values):
                        tensor = values[name]
                        if tensor is not None:
                            digest.update(name.encode())
                            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                    return digest.hexdigest()
                hashes, maximum_delta = [output_digest(output)], 0.0
                for _ in range(2):
                    repeated = model(**inputs)
                    torch.cuda.synchronize()
                    hashes.append(output_digest(repeated))
                    for name, value in output.items():
                        if value is not None and value.numel():
                            maximum_delta = max(maximum_delta, float((value - repeated[name]).abs().max().cpu()))
                row.update(status="done", prediction_validated=True, video_logit=score,
                           verdict="violation" if score >= 0 else "no_violation",
                           head_forward_ms=elapsed,
                           repeatability={"independent_model_forwards": 3, "output_sha256": hashes,
                                          "bitwise_equal_this_runtime": len(set(hashes)) == 1,
                                          "maximum_absolute_delta": maximum_delta},
                           timing_origin="cached_features_head_only", native_frame_count=len(sequence.frames))
                instance_scores = output["video_instance_logits"][0].detach().cpu().tolist()
                temporal_predictions.append({"sample_id": plan["sample_id"], "attempt_id": plan["attempt_id"],
                    "score_axis": "violation_log_odds", "threshold": 0.0,
                    "head_sha256": identity.head_sha256, "feature_spec_sha256": identity.feature_spec_sha256,
                    "time_base": f"{sequence.time_base_num}/{sequence.time_base_den}",
                    "source_start_sec": sequence.origin_pts * sequence.time_base_num / sequence.time_base_den,
                    "frames": [{"frame_id": frame.frame_id, "source_pts": frame.source_pts,
                                "pts_sec": frame.pts_sec, "eventness_logit": value}
                               for frame, value in zip(sequence.frames, instance_scores)]})
            except Exception as exc:
                row.update(error_code="head_validation_attempt_failed", error_type=type(exc).__name__)
            predictions.append(row)
    destination.mkdir(parents=True)
    prediction_path = destination / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                                       for row in predictions), encoding="utf-8")
    (destination / "temporal-predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                  for row in temporal_predictions), encoding="utf-8")
    evaluation_metadata = {"model_id": "forte-frozen-feature-weak-mil", "model_revision": identity.head_sha256,
        "head_identity": identity.model_dump(mode="json"), "conditions_sha256": canonical_sha256({
            "feature_spec_sha256": identity.feature_spec_sha256, "architecture": metadata["architecture"],
            "minimum_video_score": 0.0, "comparison_scope": "weak_model_only"}),
        "expected_attempts_sha256": file_sha256(plan_path),
        "validation_labels_used_for_threshold_selection": False,
        "validation_anchors_used_for_training": False,
        "validation_anchors_used_for_threshold_selection": False,
        "validation_anchors_used_for_prompting": False,
        "deterministic_policy": deterministic_policy,
        "threshold_selection_sha256": file_sha256(threshold_path), "comparison_scope": "weak_model_only",
        "inference_kind": "real_local", "execution_verified": True, "expected_attempts_frozen_before_run": True,
        "full_service_latency_measured": False, "human_fact_event_truth": "UNKNOWN"}
    proof = gate["gate_proof"]
    proof["exposure_registry_file_sha256"] = file_sha256(Path(config["exposure_registry"]))
    report = evaluate_weak_binary(labels, plans, predictions, split_manifest=load_json(Path(config["automated_split"])),
        experiment_config=config["experiment_authorization"], weak_gate_proof=proof, metadata=evaluation_metadata,
        provided_labels_sha256=provided_label_digest(labels), predictions_sha256=file_sha256(prediction_path))
    for name, payload in (("metadata.json", evaluation_metadata), ("gate-proof.json", proof), ("evaluation.json", report)):
        (destination / name).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": report["status"], "attempts": len(plans), "classification_completed": sum(row["status"] == "done" for row in predictions),
            "report_sha256": file_sha256(destination / "evaluation.json"), "predictions_sha256": file_sha256(prediction_path),
            "model_selection_eligible": report["model_selection_eligible"], "human_fact_event_accuracy_measured": False,
            "full_service_latency_measured": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(load_json(args.config)), sort_keys=True))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "weak_validation_failed", "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
