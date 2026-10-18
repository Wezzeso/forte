"""CPU-only post-fit train diagnosis; no optimization or validation-label access."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.guards import file_sha256, load_annotations, load_json, require


def run(config: dict, *, output: Path) -> dict:
    from training.temporal.weak_preflight import resolve_config
    from training.temporal.train_weak import load_cache_entries
    from training.temporal.cache import load_feature_cache
    from training.temporal.checkpoint import load_checkpoint, sequence_inputs
    from training.temporal.coarse_targets import load_coarse_anchors, anchor_frame_indices
    import torch
    from torch.nn import functional as F

    started = time.monotonic()
    config = resolve_config(config)
    labels = load_annotations(Path(config["provided_train_labels"]))
    require(file_sha256(Path(config["provided_train_labels"])) == config["experiment_authorization"]
            ["expected_provided_label_manifests_sha256"]["train"], "train_diagnostic_role_pin_mismatch")
    entries = load_cache_entries(config, split="train", labels=labels)
    coarse = load_coarse_anchors(config, split="train", labels=labels)
    by_content = {row.get("content_sha256", row.get("content_hash")): row for row in labels}
    torch.set_num_threads(2)
    torch.manual_seed(20261017)
    torch.use_deterministic_algorithms(True)
    checkpoints = (("initial_untrained", Path(config["untrained_output_dir"])),
                   ("trained_W1", Path(config["output_dir"])))
    models = {}
    for name, path in checkpoints:
        sha = file_sha256(path / "checkpoint.json")
        model, identity, metadata = load_checkpoint(path, expected_manifest_sha256=sha, device="cpu")
        models[name] = (model, identity)
    results = {name: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "bags": 0, "bag_bce": [],
                       "coarse_scores": [], "coarse_bce": [], "positive_bag_scores": [], "negative_bag_scores": []}
               for name in models}
    with torch.inference_mode():
        for entry in entries:
            sequence, features, manifest = load_feature_cache(Path(entry["manifest_path"]))
            require(sequence.split == "train" and sequence.content_sha256 in by_content,
                    "train_diagnostic_forbidden_role")
            label = by_content[sequence.content_sha256]["provided_video_label"]
            require(type(label) is bool, "train_diagnostic_binary_label_invalid")
            inputs = sequence_inputs(sequence, features, device="cpu")
            indices = [anchor_frame_indices(sequence, point, float(config.get("coarse_anchor_tolerance_sec", 1.0)))
                       for point in coarse.get(sequence.content_sha256, ())]
            for name, (model, identity) in models.items():
                require(manifest["feature_spec_sha256"] == identity.feature_spec_sha256,
                        "train_diagnostic_feature_binding_mismatch")
                prediction = model(**inputs)
                score = float(prediction["video_logit"][0])
                require(math.isfinite(score), "train_diagnostic_nonfinite_score")
                result = results[name]
                positive = score >= 0
                result[("tp" if positive else "fn") if label else ("fp" if positive else "tn")] += 1
                result["bags"] += 1
                result["positive_bag_scores" if label else "negative_bag_scores"].append(score)
                result["bag_bce"].append(float(F.binary_cross_entropy_with_logits(prediction["video_logit"], torch.tensor([float(label)]))))
                for anchor in indices:
                    values = prediction["video_instance_logits"][0, list(anchor)]
                    point_score = model.mil_temperature * (torch.logsumexp(values / model.mil_temperature, dim=0) - math.log(len(anchor)))
                    result["coarse_scores"].append(float(point_score))
                    result["coarse_bce"].append(float(F.binary_cross_entropy_with_logits(point_score, torch.ones_like(point_score))))
    def mean(values):
        return sum(values) / len(values) if values else None
    summaries = {}
    for name, values in results.items():
        summaries[name] = {key: values[key] for key in ("tp", "tn", "fp", "fn", "bags")}
        summaries[name].update(accuracy=(values["tp"] + values["tn"]) / values["bags"],
            mean_binary_bag_bce=mean(values["bag_bce"]), mean_positive_bag_score=mean(values["positive_bag_scores"]),
            mean_negative_bag_score=mean(values["negative_bag_scores"]), coarse_positive_bags=len(values["coarse_scores"]),
            mean_coarse_point_bag_logit=mean(values["coarse_scores"]), mean_coarse_point_bag_bce=mean(values["coarse_bce"]),
            coarse_bags_above_fixed_zero=sum(value >= 0 for value in values["coarse_scores"]))
    report = {"schema_version": "forte.weak-train-fit-diagnostic.v1", "device": "cpu", "gpu_used": False,
              "role": "train_only", "validation_or_final_labels_opened": False, "optimizer_steps": 0,
              "threshold_tuned": False, "threshold": 0.0, "input_cache_index_sha256": config["expected_feature_cache_index_sha256"],
              "comparison": summaries, "wall_sec": time.monotonic() - started,
              "scope": "post_fit_binary_and_coarse_training_fit_not_independent_accuracy"}
    require(not output.exists(), "train_diagnostic_output_exists")
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(load_json(args.config), output=args.output), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "private_values_printed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
