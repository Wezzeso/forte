"""Train a masked tracked-object Transformer on reviewed own/synthetic intervals.

CPU-only preflight never imports torch or reads private video. Training has no
download path, never uses the camera holdout, and cannot promote a runtime type.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

from forte.fast.tracked_head import (STEP_SECONDS, build_transformer,
    featurize_sequence, feature_names, validate_feature_schema, validate_model_config)
from training.specialists.common import (canonical_sha256, cli_error, file_sha256,
    load_json, number, pinned_json, relative_file, require, sha, write_json)
from training.specialists.prepare import verify_dataset

LINK_FIELDS = ("source_recording_sha256", "scenario_group", "camera_id", "split", "origin")
PATH_FIELDS = ("dataset_dir", "feature_manifest", "feature_schema", "output_dir")


def resolve_config(path: Path) -> dict:
    config = load_json(path)
    require(isinstance(config, dict), "tracked_train_config_object_required")
    for key in PATH_FIELDS:
        value = config.get(key)
        require(isinstance(value, str) and bool(value) and "://" not in value,
            "tracked_train_local_paths_required")
        config[key] = str((path.resolve().parent / value).resolve())
    return config


def interval_targets(intervals: Any, times: tuple[float, ...], types: list[str], duration: float) -> list[list[int]]:
    """Only explicitly reviewed positive/negative intervals supervise the head."""
    require(isinstance(intervals, list), "tracked_reviewed_intervals_required")
    labels = [[-100] * len(types) for _ in times]
    seen: dict[str, list[tuple[float, float]]] = {}
    for interval in intervals:
        require(isinstance(interval, dict) and interval.get("type_id") in types
            and type(interval.get("label")) is int and interval["label"] in {0, 1}
            and interval.get("human_reviewed") is True and interval.get("reviewer")
            and interval.get("reviewed_at"), "tracked_interval_review_required")
        start, end = interval.get("start_sec"), interval.get("end_sec")
        require(number(start, 0, duration) and number(end, 0, duration) and start < end,
            "tracked_interval_time_invalid")
        type_id = interval["type_id"]
        require(not any(start < b and end > a for a, b in seen.setdefault(type_id, [])),
            "tracked_interval_overlap")
        seen[type_id].append((start, end))
        index = types.index(type_id)
        points = [i for i, t in enumerate(times) if start <= t < end]
        require(bool(points), "tracked_interval_has_no_grid_observation")
        for i in points:
            labels[i][index] = interval["label"]
    return labels


def validate_feature_manifest(manifest: dict, lock: dict, expected_lock: str,
        schema: dict, schema_file_sha256: str, base: Path, model: dict) -> list[dict]:
    require(manifest.get("schema_version") == "forte.tracked-head.data.v1"
        and manifest.get("frozen") is True and manifest.get("reviewed_by")
        and manifest.get("reviewed_at"), "tracked_feature_data_review_required")
    require(manifest.get("dataset_lock_sha256") == expected_lock
        and manifest.get("feature_schema_sha256") == schema_file_sha256,
        "tracked_feature_dataset_link_mismatch")
    sources = {r["sample_id"]: r for r in lock["records"]}
    source_roles, group_roles, content_roles = {}, {}, {}
    banned = {r.get("source_recording_sha256") for r in lock["records"] if r.get("origin") == "organizer"}
    for source in lock["records"]:
        role = source.get("split")
        for key, mapping in (("source_recording_sha256", source_roles),
                ("scenario_group", group_roles), ("content_sha256", content_roles)):
            value = source.get(key)
            require(value and mapping.get(value, role) == role, "tracked_partition_leakage")
            mapping[value] = role
    for source in lock["records"]:
        if source.get("origin") not in {"own", "synthetic"}:
            continue
        provenance = source.get("provenance", {})
        require(source.get("source_recording_sha256") not in banned
            and provenance.get("commercial_use_allowed") is True and provenance.get("reference"),
            "tracked_source_provenance_required")
        if source["origin"] == "own":
            require(provenance.get("participants_consented") is True
                and provenance.get("recording_authorized") is True, "tracked_source_consent_required")
        else:
            ancestors = provenance.get("input_source_hashes")
            require(provenance.get("synthetic_inputs_reviewed") is True and provenance.get("generator")
                and isinstance(ancestors, list) and all(sha(value) for value in ancestors)
                and not set(ancestors) & banned, "tracked_synthetic_ancestry_invalid")
            require(all(source_roles.get(value, source["split"]) == source["split"]
                and content_roles.get(value, source["split"]) == source["split"] for value in ancestors),
                "tracked_synthetic_ancestry_leakage")
    rows = manifest.get("records")
    require(isinstance(rows, list) and bool(rows), "tracked_feature_records_required")
    seen, output, roles, cameras, seen_recordings = set(), [], {}, {}, set()
    for row in rows:
        require(isinstance(row, dict), "tracked_feature_record_invalid")
        identifier = row.get("sequence_id")
        require(isinstance(identifier, str) and identifier and identifier not in seen,
            "tracked_sequence_id_invalid")
        seen.add(identifier)
        source = sources.get(row.get("source_sample_id"))
        require(source is not None and source.get("origin") in {"own", "synthetic"}
            and source.get("split") in {"train", "validation", "holdout"},
            "tracked_organizer_or_unknown_source_forbidden")
        require(all(row.get(k) == source.get(k) for k in LINK_FIELDS), "tracked_source_lineage_mismatch")
        require(row["source_recording_sha256"] not in seen_recordings, "tracked_duplicate_complete_recording")
        seen_recordings.add(row["source_recording_sha256"])
        require(source.get("labels_reviewed") is True and source.get("exhaustive_labels") is True
            and source.get("label_reviewer") and source.get("label_reviewed_at")
            and row.get("human_reviewed") is True and row.get("reviewer") and row.get("reviewed_at"),
            "tracked_sequence_review_required")
        for key in ("source_recording_sha256", "scenario_group"):
            value = row[key]
            require(roles.get((key, value), row["split"]) == row["split"], "tracked_partition_leakage")
            roles[key, value] = row["split"]
        cameras.setdefault(row["camera_id"], set()).add(row["split"])
        require(sha(row.get("feature_sha256")), "tracked_feature_hash_required")
        # Holdout identities are checked, but its feature bytes/labels are not opened.
        if row["split"] == "holdout":
            require(row["origin"] == "own", "tracked_camera_holdout_requires_own")
            continue
        path = relative_file(base, row.get("feature_path"))
        payload = pinned_json(path, row["feature_sha256"])
        require(payload.get("schema_version") == "forte.tracked-head.sequence.v1"
            and payload.get("feature_schema_sha256") == schema_file_sha256
            and payload.get("source_recording_sha256") == row["source_recording_sha256"]
            and payload.get("complete_recording") is True,
            "tracked_sequence_source_or_schema_mismatch")
        duration = payload.get("duration_sec")
        require(number(duration, .001, model["max_steps"] * STEP_SECONDS), "tracked_sequence_duration_invalid")
        try:
            sequence = featurize_sequence(payload.get("frames", []), schema)
        except ValueError as exc:
            require(False, str(exc))
        require(len(sequence.times) == math.ceil(duration / STEP_SECONDS)
            and len(sequence.times) <= model["max_steps"], "tracked_full_recording_grid_required")
        targets = interval_targets(payload.get("label_intervals"), sequence.times, model["types"], duration)
        require(any(value != -100 for target in targets for value in target), "tracked_sequence_no_supervision")
        output.append(dict(row) | {"sequence": sequence, "targets": targets})
    require({r["split"] for r in rows} >= {"train", "validation", "holdout"},
        "tracked_train_validation_camera_holdout_required")
    require(all("holdout" not in splits or splits == {"holdout"} for splits in cameras.values()),
        "tracked_camera_holdout_leakage")
    # Check the whole pinned lock too: a manifest subset cannot hide a leaked holdout camera.
    for source in lock["records"]:
        if source.get("camera_id") in cameras and "holdout" in cameras[source["camera_id"]]:
            require(source.get("split") == "holdout", "tracked_camera_holdout_leakage")
    return output


def preflight(config: dict) -> dict:
    require(config.get("schema_version") == "forte.tracked-head.train.v1", "tracked_train_config_schema")
    require(isinstance(config.get("runtime_versions"), dict)
        and isinstance(config["runtime_versions"].get("torch"), str)
        and config["runtime_versions"]["torch"], "tracked_torch_version_pin_required")
    for field, high in (("optimizer_steps", 100000), ("seed", 2**31 - 1)):
        require(type(config.get(field)) is int and (0 if field == "seed" else 1) <= config[field] <= high,
            "tracked_training_parameter_invalid")
    require(number(config.get("learning_rate"), 1e-9, .01)
        and number(config.get("weight_decay"), 0, 1), "tracked_training_parameter_invalid")
    require(config.get("device") in {"cpu", "cuda"}
        and type(config.get("allow_cpu_training")) is bool
        and (config["device"] != "cpu" or config["allow_cpu_training"]), "tracked_explicit_training_device_required")
    model = config.get("model", {})
    try:
        validate_model_config(model)
    except (TypeError, ValueError) as exc:
        require(False, str(exc))
    schema = pinned_json(Path(config["feature_schema"]), config.get("expected_feature_schema_sha256", ""))
    try:
        validate_feature_schema(schema)
    except (TypeError, ValueError) as exc:
        require(False, str(exc))
    lock = verify_dataset(Path(config["dataset_dir"]), config.get("expected_dataset_lock_sha256", ""))
    manifest_path = Path(config["feature_manifest"])
    manifest = pinned_json(manifest_path, config.get("expected_feature_manifest_sha256", ""))
    rows = validate_feature_manifest(manifest, lock, config["expected_dataset_lock_sha256"], schema,
        config["expected_feature_schema_sha256"], manifest_path.resolve().parent, model)
    training_support = {}
    for column, type_id in enumerate(model["types"]):
        labels = [target[column] for row in rows if row["split"] == "train" for target in row["targets"]]
        counts = {"positive_grid_points": labels.count(1), "negative_grid_points": labels.count(0)}
        require(counts["positive_grid_points"] > 0 and counts["negative_grid_points"] > 0,
            "tracked_per_type_reviewed_training_support_required")
        training_support[type_id] = counts
    return {"rows": rows, "schema": schema, "lock": lock,
        "summary": {"sequences": len(rows), "split_counts": dict(Counter(r["split"] for r in rows)),
            "training_support_per_type": training_support,
            "holdout_used_for_training_or_thresholds": False,
            "known_targets": sum(v != -100 for r in rows for row in r["targets"] for v in row),
            "unknown_targets": sum(v == -100 for r in rows for row in r["targets"] for v in row)}}


def calibrate_thresholds(predictions: list[dict], types: list[str]) -> tuple[dict, dict]:
    """Use only fully reviewed development videos; zero FP precedes recall."""
    per_type, metrics = {}, {}
    for column, type_id in enumerate(types):
        scored = []
        known_pairs = []
        for row in predictions:
            require(len(row["probabilities"]) == len(row["targets"]) == len(row["times"])
                and bool(row["times"]) and all(len(values) == len(types)
                    and all(number(v, 0, 1) for v in values) for values in row["probabilities"]),
                "tracked_validation_prediction_invalid")
            labels = [target[column] for target in row["targets"]]
            probs = [value[column] for value in row["probabilities"]]
            known_pairs.extend((p, label) for p, label in zip(probs, labels) if label != -100)
            if all(label != -100 for label in labels):
                scored.append((row, max(probs), int(any(labels))))
        clean = [p for _, p, label in scored if label == 0]
        positive = [p for _, p, label in scored if label == 1]
        threshold = max(.5, math.nextafter(max(clean), math.inf)) if clean else None
        supported = bool(clean and positive and threshold is not None and threshold <= 1)
        per_type[type_id] = {"supported": supported, "threshold": threshold if supported else None,
            "selection": "validation_full_video_zero_false_positives",
            "positive_videos": len(positive), "clean_videos": len(clean)}
        if supported:
            tp = sum(p >= threshold for p in positive)
            fp = sum(p >= threshold for p in clean)
            frame_tp = sum(p >= threshold and label == 1 for p, label in known_pairs)
            frame_fp = sum(p >= threshold and label == 0 for p, label in known_pairs)
            frame_positive = sum(label == 1 for _, label in known_pairs)
            errors = []
            for row, peak, label in scored:
                if label == 1 and peak >= threshold:
                    peak_index = max(range(len(row["probabilities"])), key=lambda i: row["probabilities"][i][column])
                    peak_t = row["times"][peak_index]
                    errors.append(min(abs(peak_t - t) for t, targets in zip(row["times"], row["targets"])
                        if targets[column] == 1))
            metrics[type_id] = {"video_true_positive": tp, "video_false_positive": fp,
                "video_recall": tp / len(positive), "video_precision": tp / (tp + fp) if tp + fp else None,
                "known_frame_recall": frame_tp / frame_positive if frame_positive else None,
                "known_frame_precision": frame_tp / (frame_tp + frame_fp) if frame_tp + frame_fp else None,
                "mean_peak_distance_to_positive_grid_sec": sum(errors) / len(errors) if errors else None,
                "unknown_frames_excluded": sum(target[column] == -100 for r in predictions for target in r["targets"])}
        else:
            metrics[type_id] = {"status": "insufficient_fully_reviewed_validation_support"}
    return per_type, metrics


def train(config: dict) -> dict:
    prepared = preflight(config)
    output = Path(config["output_dir"])
    require(not output.exists(), "tracked_output_already_exists")
    try:
        installed = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        require(False, "tracked_torch_not_installed")
    require(installed == config["runtime_versions"]["torch"], "tracked_torch_version_mismatch")
    if config["device"] == "cuda":
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {None, ":4096:8", ":16:8"},
            "tracked_cublas_determinism_configuration_invalid")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch.nn import functional as F
    if config["device"] == "cuda":
        require(torch.cuda.is_available(), "tracked_cuda_unavailable")
    random.seed(config["seed"]); torch.manual_seed(config["seed"])
    if config["device"] == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    model = build_transformer(config["model"], len(feature_names(prepared["schema"]))).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    training = [r for r in prepared["rows"] if r["split"] == "train"]
    validation = [r for r in prepared["rows"] if r["split"] == "validation"]
    require(bool(training) and bool(validation), "tracked_training_partitions_empty")

    def tensors(row):
        sequence = row["sequence"]
        return (torch.tensor(sequence.values, dtype=torch.float32, device=config["device"])[None],
            torch.tensor(sequence.observed, dtype=torch.bool, device=config["device"])[None],
            torch.tensor(row["targets"], dtype=torch.float32, device=config["device"])[None])

    losses = []
    order = []
    for step in range(config["optimizer_steps"]):
        if not order:
            order = list(training); random.shuffle(order)
        row = order.pop()
        values, observed, targets = tensors(row)
        known = targets != -100
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(values, observed)
        loss = F.binary_cross_entropy_with_logits(logits[known], targets[known])
        require(bool(torch.isfinite(loss)), "tracked_nonfinite_training_loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    predictions = []
    with torch.inference_mode():
        for row in validation:
            values, observed, _ = tensors(row)
            probabilities = model(values, observed).sigmoid()[0].cpu().tolist()
            predictions.append({"sequence_id": row["sequence_id"], "times": list(row["sequence"].times),
                "probabilities": probabilities, "targets": row["targets"]})
    per_type, validation_metrics = calibrate_thresholds(predictions, config["model"]["types"])
    output.mkdir(parents=True)
    checkpoint = output / "head.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
    write_json(output / "model-config.json", config["model"])
    write_json(output / "feature-schema.json", prepared["schema"])
    schema_sha = file_sha256(output / "feature-schema.json")
    write_json(output / "thresholds.json", {"schema_version": "forte.tracked-head.thresholds.v1",
        "feature_schema_sha256": schema_sha, "per_type": per_type})
    write_json(output / "validation-metrics.json", {"schema_version": "forte.tracked-head.validation.v1",
        "scope": "own_synthetic_validation_only", "full_video_promotion": False,
        "per_type": validation_metrics, "predictions": predictions, "training": prepared["summary"],
        "optimizer_steps": config["optimizer_steps"], "final_training_loss": losses[-1]})
    entries = {"checkpoint": "head.pt", "model_config": "model-config.json", "feature_schema": "feature-schema.json",
        "thresholds": "thresholds.json", "validation_metrics": "validation-metrics.json"}
    artifact = {"schema_version": "forte.tracked-head.artifact.v1", "state": "trained_unselected",
        "approved_types": [], "requires_independent_verification": True,
        "training_config_sha256": canonical_sha256(config), "runtime_versions": config["runtime_versions"],
        "dataset_lock_sha256": config["expected_dataset_lock_sha256"],
        "feature_manifest_sha256": config["expected_feature_manifest_sha256"],
        "input_feature_schema_sha256": config["expected_feature_schema_sha256"],
        "files": {k: {"path": name, "sha256": file_sha256(output / name)} for k, name in entries.items()}}
    write_json(output / "artifact.json", artifact)
    write_json(output / "training-config.json", config)
    return {"ok": True, "state": "trained_unselected", "approved_types": [],
        "artifact_sha256": file_sha256(output / "artifact.json"), "validation": validation_metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        config = resolve_config(args.config)
        result = ({"ok": True, "state": "preflight_only", **preflight(config)["summary"]}
            if args.preflight else train(config))
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        return cli_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
