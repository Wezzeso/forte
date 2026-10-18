"""Executable train-only organizer-binary MIL with optional provenance-bound silver.

One complete native-frame recording is one bag. The frozen encoder is never
loaded here. Validation/final labels are never opened by this command.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

from training.guards import TrainingGateError, canonical_sha256, file_sha256, load_annotations, load_json, require
from training.temporal.weak_preflight import resolve_config, validate_metadata


def require_compute_release(config: dict) -> None:
    release = config.get("compute_release_authorization", {})
    require(release.get("released_by_A18_or_root") is True
            and isinstance(release.get("release_reference"), str) and bool(release["release_reference"].strip()),
            "weak_compute_window_not_released")


def configure_determinism() -> dict:
    # Must run BEFORE runtime_check initializes CUDA/cuBLAS.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"deterministic_algorithms": True, "unsupported_operation_policy": "fail",
            "cublas_workspace_config": ":4096:8", "cudnn_benchmark": False,
            "tf32": False, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cross_hardware_bitwise_equivalence_claimed": False}


def source_code_binding() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in {".venv", "private", "runs", "__pycache__"})
        files.extend(Path(directory) / name for name in sorted(names) if name.endswith(".py"))
    hashes = {str(path.relative_to(root)).replace("\\", "/"): file_sha256(path) for path in sorted(files)}
    return {"training_source_tree_sha256": canonical_sha256(hashes), "file_count": len(hashes),
            "files": hashes, "git_commit_available": False,
            "scope": "training_python_sources_including_temporal_head_and_checkpoint_io"}


def load_cache_entries(config: dict, *, split: str, labels: list[dict]) -> list[dict]:
    """Select only the requested role; each entry must have an independent hash."""
    index = Path(config["feature_cache_index"]).resolve()
    require(index.is_file() and file_sha256(index) == config.get("expected_feature_cache_index_sha256"),
            "weak_cache_index_pin_mismatch")
    payload = load_json(index)
    require(isinstance(payload, dict) and payload.get("schema_version") == "forte.private-feature-cache-index.v1"
            and payload.get("all_native_frames") is True and payload.get("final_check_excluded") is True
            and payload.get("assignments_sha256") == config["experiment_authorization"]["expected_assignments_sha256"],
            "weak_cache_index_assignment_or_coverage_mismatch")
    entries = payload.get("entries", payload.get("records", [])) if isinstance(payload, dict) else payload
    require(isinstance(entries, list), "weak_cache_index_invalid")
    expected = {row.get("content_sha256", row.get("content_hash")): row for row in labels}
    selected = [row for row in entries if row.get("split") == split]
    require(len(selected) == len(expected) and len({row.get("content_sha256") for row in selected}) == len(selected)
            and {row.get("content_sha256") for row in selected} == set(expected), "weak_cache_population_mismatch")
    for row in selected:
        path = Path(row["manifest_path"])
        path = path.resolve() if path.is_absolute() else (index.parent / path).resolve()
        require(path.is_file() and file_sha256(path) == row.get("manifest_sha256"), "weak_cache_manifest_pin_mismatch")
        require(row.get("cache_complete") is True and row.get("group_id") == expected[row["content_sha256"]]["group_id"],
                "weak_cache_assignment_mismatch")
        row["manifest_path"] = str(path)
    return selected


def load_training_data(config: dict, labels: list[dict], fact_ids: tuple, event_ids: tuple, *, metadata_only: bool = False):
    from training.temporal.cache import load_feature_cache
    from training.temporal.weak_targets import build_weak_supervision, teacher_windows_from_machine_record
    from training.temporal.coarse_targets import load_coarse_anchors, anchor_frame_indices

    entries = load_cache_entries(config, split="train", labels=labels)
    by_content = {row.get("content_sha256", row.get("content_hash")): row for row in labels}
    coarse = load_coarse_anchors(config, split="train", labels=labels)
    teacher_path = Path(config["machine_annotations"]).resolve() if config.get("machine_annotations") else None
    teachers = {}
    if teacher_path is not None and teacher_path.is_file():
        # Teacher labels are optional and must be separately pinned.
        require(file_sha256(teacher_path) == config.get("expected_machine_annotations_sha256"),
                "weak_teacher_export_pin_mismatch")
        for row in load_annotations(teacher_path):
            content = row.get("content_sha256")
            if content in by_content and row.get("split") == "train":
                teachers[content] = row
    data, counts = [], Counter()
    spec_hash = None
    for entry in entries:
        if metadata_only:
            from training.temporal.cache import _verify_manifest
            manifest = load_json(Path(entry["manifest_path"]))
            sequence, features = _verify_manifest(manifest), None
        else:
            sequence, features, manifest = load_feature_cache(Path(entry["manifest_path"]))
        content = sequence.content_sha256
        require(sequence.split == "train" and content in by_content
                and sequence.group_id == by_content[content]["group_id"], "weak_loaded_cache_assignment_mismatch")
        require(spec_hash is None or manifest["feature_spec_sha256"] == spec_hash, "weak_mixed_feature_recipes_forbidden")
        spec_hash = manifest["feature_spec_sha256"]
        base = build_weak_supervision(sequence, by_content[content], teacher_windows=[], fact_ids=fact_ids,
                    event_type_ids=event_ids, expected_label_source_sha256=config["expected_provided_label_source_sha256"])
        windows = []
        if content in teachers:
            try:
                normalized = teacher_windows_from_machine_record(sequence, teachers[content])
            except TrainingGateError:
                counts["teacher_records_rejected"] += 1
                normalized = []
            for window in normalized:
                try:
                    silver = build_weak_supervision(sequence, by_content[content], teacher_windows=[window],
                        fact_ids=fact_ids, event_type_ids=event_ids,
                        expected_label_source_sha256=config["expected_provided_label_source_sha256"])
                    for target_window in silver.teacher_windows:
                        target_counts = target_window.targets.coverage()
                        if not any(target_counts.values()):
                            counts["teacher_windows_fully_masked"] += 1
                            continue
                        windows.append(target_window)
                        for name, count in target_counts.items():
                            name = "event_brackets" if name == "reviewed_event_brackets" else name
                            counts["silver_" + name] += count
                    counts["masked_teacher_event_conflicts"] += silver.masked_teacher_event_conflicts
                    counts["masked_teacher_support_conflicts"] += int(window.get("masked_support_conflicts", 0))
                except TrainingGateError as exc:
                    # Missing prompt/revision/support provenance never blocks
                    # independent organizer MIL, and never becomes silver loss.
                    counts["teacher_windows_rejected"] += 1
                    counts["teacher_window_rejection_" + str(exc)] += 1
        counts["teacher_windows_eligible"] += len(windows)
        counts["native_frames"] += len(sequence.frames)
        counts["recording_bags"] += 1
        anchors = tuple(anchor_frame_indices(sequence, point, float(config.get("coarse_anchor_tolerance_sec", 1.0)))
                        for point in coarse.get(content, ()))
        counts["provided_coarse_positive_anchor_bags"] += len(anchors)
        data.append((sequence, features, manifest, base.provided_video_label, windows, anchors))
    require(bool(data), "weak_training_cache_empty")
    return data, dict(counts)


def prepare_supervision(config: dict, labels: list[dict], fact_ids: tuple, event_ids: tuple,
                        *, metadata_only: bool = False):
    from training.temporal.supervision import mask_cross_window_conflicts
    data, coverage = load_training_data(config, labels, fact_ids, event_ids, metadata_only=metadata_only)
    mask = None
    if config.get("mask_cross_window_target_conflicts", False):
        data, mask = mask_cross_window_conflicts(data)
        coverage = {key: value for key, value in coverage.items() if not key.startswith("silver_")}
        coverage["teacher_windows_before_dataset_conflict_mask"] = coverage.get("teacher_windows_eligible", 0)
        coverage["teacher_windows_eligible"] = sum(len(record[4]) for record in data)
        coverage["teacher_windows_fully_masked"] = coverage.get("teacher_windows_fully_masked", 0) + mask["fully_masked_teacher_windows"]
        for record in data:
            for window in record[4]:
                for key, value in window.targets.coverage().items():
                    key = "event_brackets" if key == "reviewed_event_brackets" else key
                    coverage["silver_" + key] = coverage.get("silver_" + key, 0) + value
    return data, coverage, mask


def fit(model, data, *, device: str, epochs: int, max_steps: int, learning_rate: float,
        weight_decay: float, teacher_weight: float, seed: int, coarse_anchor_weight: float = .5,
        component_loss_weights: dict | None = None) -> dict:
    """Real optimizer loop, reusable by synthetic CPU tests without data gates."""
    import torch
    from training.temporal.checkpoint import sequence_inputs
    from training.temporal.head import sparse_loss, weak_video_mil_loss
    from training.temporal.coarse_targets import coarse_positive_anchor_loss

    require(1 <= epochs <= 30 and 1 <= max_steps <= 1000 and 0 < learning_rate <= .1
            and 0 <= weight_decay <= 1 and 0 <= teacher_weight <= 1 and 0 <= coarse_anchor_weight <= 1,
            "weak_optimizer_bounds_invalid")
    model.to(device).train()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=learning_rate, weight_decay=weight_decay)
    rng, steps, losses = random.Random(seed), 0, []
    started = time.monotonic()
    for _ in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for index in order:
            sequence, features, _, label, silver_windows = data[index][:5]
            anchor_indices = data[index][5] if len(data[index]) > 5 else ()
            inputs = sequence_inputs(sequence, features, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs)
            loss, _ = weak_video_mil_loss(output, torch.tensor([label], device=device), inputs["frame_mask"])
            if anchor_indices and coarse_anchor_weight:
                loss = loss + coarse_anchor_weight * coarse_positive_anchor_loss(
                    output, anchor_indices, temperature=model.mil_temperature)
            require(bool(torch.isfinite(loss)), "weak_nonfinite_loss")
            # Backpropagate whole-bag loss first, freeing long-recording graph.
            loss.backward()
            value = float(loss.detach().cpu())
            del inputs, output, loss
            if teacher_weight and silver_windows:
                for silver in silver_windows:
                    inputs = sequence_inputs(sequence, features, device=device, indices=silver.sequence_frame_indices)
                    output = model(**inputs)
                    target = silver.targets
                    tensors = {name: torch.tensor([getattr(target, name)], device=device, dtype=torch.long)
                               for name in ("window_states", "frame_states", "frame_support", "event_boundaries")}
                    loss, _ = sparse_loss(output, tensors, component_weights=component_loss_weights)
                    loss = loss * teacher_weight / len(silver_windows)
                    require(bool(torch.isfinite(loss)), "weak_nonfinite_silver_loss")
                    loss.backward()
                    value += float(loss.detach().cpu())
                    del inputs, output, tensors, loss
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            require(bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
                    "weak_nonfinite_or_missing_gradients")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            require(bool(torch.isfinite(norm)) and float(norm) > 0, "weak_zero_or_nonfinite_gradient_norm")
            optimizer.step()
            steps += 1
            losses.append(value)
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
    model.eval()
    return {"optimizer_steps": steps, "initial_loss": losses[0], "final_loss": losses[-1],
            "mean_training_loss": sum(losses) / len(losses), "training_wall_sec": time.monotonic() - started,
            "optimizer": "AdamW", "gradient_clip_norm": 1.0, "negative_instance_weight": 1.0,
            "batch_unit": "one_complete_recording_mil_bag"}


def write_fixed_threshold(destination: Path, receipt: dict, *, feature_spec_sha256: str,
                          provided_validation_labels_sha256: str) -> str:
    """Freeze this rule before validation is opened; no empirical tuning."""
    threshold = {"schema_version": "forte.weak_video_threshold.v1", "score_axis": "violation_log_odds",
                 "minimum_video_score": 0.0, "comparison": "greater_than_or_equal",
                 "selection_method": "predeclared_fixed_zero", "validation_labels_used_for_selection": False,
                 "head_sha256": receipt["weights_sha256"], "head_identity_sha256": receipt["head_identity_sha256"],
                 "feature_spec_sha256": feature_spec_sha256,
                 "provided_validation_labels_sha256": provided_validation_labels_sha256}
    path = destination / "video-threshold.json"
    require(not path.exists(), "weak_threshold_already_exists")
    path.write_text(json.dumps(threshold, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return file_sha256(path)


def load_parent_checkpoint(config: dict, *, architecture: dict, identity: dict):
    from training.temporal.checkpoint import load_checkpoint
    model, bound, metadata = load_checkpoint(Path(config["initial_checkpoint_dir"]),
        expected_manifest_sha256=config["expected_initial_checkpoint_manifest_sha256"], device="cpu")
    require(metadata["architecture"] == architecture
            and {key: value for key, value in bound.model_dump(mode="json").items() if key != "head_sha256"}
                == {key: list(value) if isinstance(value, tuple) else value for key, value in identity.items()},
            "weak_initial_checkpoint_binding_mismatch")
    old = metadata["provenance"]
    return model, {"manifest_sha256": config["expected_initial_checkpoint_manifest_sha256"],
        "head_sha256": bound.head_sha256,
        "optimizer_steps": old.get("cumulative_optimizer_steps", old.get("metrics", {}).get("optimizer_steps", 0)),
        "trained_components": old.get("trained_components", {}), "source_code_binding": old.get("source_code_binding")}


def run(config: dict) -> dict:
    config = resolve_config(config)
    from training.temporal.protocol import validate_protocol_config, require_protocol_supervision, validate_question_policy
    protocol = validate_protocol_config(config)
    question_policy = validate_question_policy(config)
    labels, gate = validate_metadata(config)
    require_compute_release(config)
    deterministic_policy = configure_determinism()
    from training.runtime_check import verify_runtime
    runtime = verify_runtime(Path(config.get("runtime_lock", "training/runtime-lock.json")), gpu=False)
    import torch
    from training.temporal.checkpoint import save_checkpoint
    from training.temporal.head import FrozenFeatureWeakMILHead
    from training.temporal.supervision import component_weights, freeze_unsupervised_branches, class_coverage

    taxonomy_path = Path(config["taxonomy_path"]).resolve()
    require(file_sha256(taxonomy_path) == config.get("expected_taxonomy_sha256"), "weak_taxonomy_pin_mismatch")
    from training.temporal.axes import load_axes
    fact_ids, event_ids = load_axes(config)
    destination = Path(config["output_dir"]).resolve()
    require(not destination.exists(), "weak_run_destination_exists")
    seed = int(config.get("seed", 20261017))
    random.seed(seed)
    torch.manual_seed(seed)
    data, coverage, conflict_mask = prepare_supervision(config, labels, fact_ids, event_ids)
    require_protocol_supervision(protocol, coverage)
    architecture = {"kind": "FrozenFeatureWeakMILHead", "embedding_dim": data[0][0].embedding_dim,
                    "hidden_dim": int(config.get("hidden_dim", 128)), "event_count": len(event_ids),
                    "mil_temperature": float(config.get("mil_temperature", .25))}
    model = FrozenFeatureWeakMILHead(**{key: value for key, value in architecture.items() if key != "kind"})
    manifest = data[0][2]
    recipe = manifest["feature_spec"]["encoder_recipe"]
    identity = {"encoder_id": recipe["model_id"], "encoder_revision": recipe["model_revision"],
                "encoder_artifacts_sha256": data[0][0].encoder_manifest_sha256,
                "feature_spec_sha256": manifest["feature_spec_sha256"], "taxonomy_sha256": file_sha256(taxonomy_path),
                "fact_ids": fact_ids, "event_type_ids": event_ids}
    parent = None
    if config.get("initial_checkpoint_dir"):
        model, parent = load_parent_checkpoint(config, architecture=architecture, identity=identity)
        if protocol is not None:
            require(parent["head_sha256"] == protocol.get("initial_checkpoint_expected_weights_sha256"),
                    "weak_protocol_parent_weights_mismatch")
    loss_weights = component_weights(config)
    observed_classes = class_coverage(data, fact_ids=fact_ids, event_ids=event_ids)
    frozen_branches = (freeze_unsupervised_branches(model, coverage, weights=loss_weights,
                        teacher_weight=float(config.get("teacher_weight", .1)))
                       if config.get("freeze_unsupervised_branches", False) else {})
    # No CUDA allocation until actual source/target/parent bindings have passed.
    runtime = verify_runtime(Path(config.get("runtime_lock", "training/runtime-lock.json")), gpu=True)
    torch.cuda.manual_seed_all(seed)
    initial_destination = Path(config.get("reference_output_dir", config.get("untrained_output_dir", str(destination) + "-untrained"))).resolve()
    initial_receipt = save_checkpoint(model, initial_destination, identity=identity, architecture=architecture,
        provenance={"initial_untrained_reference": parent is None, "initial_reference_kind": "parent_checkpoint" if parent else "random_initialization",
                    "parent_checkpoint": parent, "optimizer_steps": 0, "seed": seed,
                    "metadata_gate": gate, "runtime": runtime, "human_verified": False,
                    "deterministic_policy": deterministic_policy, "source_code_binding": source_code_binding(),
                    "fact_axis_manifest_sha256": config["expected_fact_axis_manifest_sha256"],
                    "code_archive_sha256": config.get("code_archive_sha256"),
                    "trained_components": parent["trained_components"] if parent else {"video_binary_mil": False, "sparse_silver": False}})
    validation_pin = config["experiment_authorization"]["expected_provided_label_manifests_sha256"]["validation"]
    write_fixed_threshold(initial_destination, initial_receipt, feature_spec_sha256=identity["feature_spec_sha256"],
                          provided_validation_labels_sha256=validation_pin)
    metrics = fit(model, data, device=config.get("device", "cuda:0"), epochs=int(config.get("epochs", 12)),
                  max_steps=int(config.get("max_steps", 336)), learning_rate=float(config.get("learning_rate", .001)),
                  weight_decay=float(config.get("weight_decay", .01)), teacher_weight=float(config.get("teacher_weight", .1)), seed=seed,
                  coarse_anchor_weight=float(config.get("coarse_anchor_weight", .5)), component_loss_weights=loss_weights)
    teacher_used = float(config.get("teacher_weight", .1)) > 0
    trained_components = {"video_binary_mil": True,
        "sparse_silver": teacher_used and coverage.get("teacher_windows_eligible", 0) > 0,
        "window_fact_states": teacher_used and loss_weights["window"] > 0 and coverage.get("silver_window_fact_labels", 0) > 0,
        "frame_fact_states": teacher_used and loss_weights["frame"] > 0 and coverage.get("silver_frame_fact_labels", 0) > 0,
        "frame_support": teacher_used and loss_weights["support"] > 0 and coverage.get("silver_support_positive_labels", 0)
            + coverage.get("silver_support_negative_labels", 0) > 0,
        "event_boundaries": teacher_used and loss_weights["boundary"] > 0 and coverage.get("silver_boundary_positive_labels", 0)
            + coverage.get("silver_boundary_negative_labels", 0) > 0,
        "provided_coarse_eventness": float(config.get("coarse_anchor_weight", .5)) > 0
            and coverage.get("provided_coarse_positive_anchor_bags", 0) > 0}
    provenance = {"run_config_sha256": canonical_sha256(config), "metadata_gate": gate,
                  "deterministic_policy": deterministic_policy,
                  "source_code_binding": source_code_binding(),
                  "code_archive_sha256": config.get("code_archive_sha256"),
                  "fact_axis_manifest_sha256": config["expected_fact_axis_manifest_sha256"],
                  "parent_checkpoint": parent, "parent_optimizer_steps": parent["optimizer_steps"] if parent else 0,
                  "cumulative_optimizer_steps": metrics["optimizer_steps"] + (parent["optimizer_steps"] if parent else 0),
                  "class_coverage": observed_classes, "frozen_branches": frozen_branches,
                  "dataset_conflict_mask": conflict_mask,
                  "component_loss_weights": loss_weights,
                  "effective_silver_loss_weights": {key: float(config.get("teacher_weight", .1)) * value
                                                     for key, value in loss_weights.items()},
                  "validation_reuse_disclosure": config.get("validation_reuse_disclosure", "initial_predeclared_provided_label_validation"),
                  "protocol_manifest_sha256": config.get("expected_protocol_manifest_sha256"),
                  "anchor_question_policy_file_sha256": config.get("expected_anchor_question_policy_sha256"),
                  "anchor_question_semantic_policy_sha256": question_policy.get("policy_sha256") if question_policy else None,
                  "teacher_question_factory_source_sha256": question_policy.get("factory_source_sha256") if question_policy else None,
                  "hyperparameters": {name: config.get(name, default) for name, default in
                      (("epochs", 12), ("max_steps", 336), ("learning_rate", .001), ("weight_decay", .01),
                       ("teacher_weight", .1), ("coarse_anchor_weight", .5), ("hidden_dim", 128), ("mil_temperature", .25))},
                  "runtime": runtime, "coverage": coverage, "metrics": metrics, "seed": seed,
                  "feature_cache_index_sha256": config["expected_feature_cache_index_sha256"],
                  "teacher_export_sha256": config.get("expected_machine_annotations_sha256"),
                  "provided_train_anchors_sha256": config.get("expected_provided_train_anchors_sha256"),
                  "coarse_anchor_source_sha256": config.get("expected_coarse_anchor_source_sha256"),
                  "coarse_anchor_tolerance_sec": float(config.get("coarse_anchor_tolerance_sec", 1.0)),
                  "coarse_tolerance_origin": "predeclared_modeling_assumption_not_observed_event_duration",
                  "fact_event_truth": "silver_or_unknown", "binary_truth": "organizer_workbook_train_only",
                  "calibration_established": False, "validation_labels_opened": False, "final_check_labels_opened": False,
                  "initial_checkpoint_receipt": initial_receipt,
                  "trained_components": trained_components}
    if parent is None:
        provenance["untrained_checkpoint_receipt"] = initial_receipt
    result = save_checkpoint(model, destination, identity=identity, architecture=architecture, provenance=provenance)
    # Fixed threshold frozen before validation is read. This is a binary-head
    # decision rule, never a fact-state or evidence calibration threshold.
    write_fixed_threshold(destination, result, feature_spec_sha256=identity["feature_spec_sha256"],
                          provided_validation_labels_sha256=validation_pin)
    summary = {"status": "trained_candidate_unselected", **result, "recording_bags": coverage["recording_bags"],
            "initial_checkpoint_receipt": initial_receipt,
            "optimizer_steps": metrics["optimizer_steps"], "human_verified": False, "quality_measured": False,
            "fact_event_calibrated": False, "full_service_latency_measured": False}
    if parent is None:
        summary["untrained_checkpoint_receipt"] = initial_receipt
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        if args.preflight:
            from training.temporal.protocol import validate_protocol_config, validate_question_policy
            validate_protocol_config(resolve_config(config))
            validate_question_policy(resolve_config(config))
            _, result = validate_metadata(config)
            result.pop("gate_proof", None)
        else:
            result = run(config)
        print(json.dumps(result, sort_keys=True))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "weak_training_failed", "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
