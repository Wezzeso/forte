"""Prepare pinned private runner configs without opening media or label rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.guards import TrainingGateError, HEX64, file_sha256, load_json, require

AXIS_SHA = "a5b62ded9208bc7112d8996dbb7475df1e715aba5f34415786cf0e87f1f35f8e"
TAXONOMY_SHA = "817b24496dc55d060e944491f877747dde52cb5195ffd1edeea2cd7ffffbe9da"


def write_config(path: Path, payload: dict) -> dict:
    require(not path.exists(), "weak_runtime_config_already_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return {"config_path": str(path.resolve()), "config_sha256": file_sha256(path), "model_or_media_opened": False}


def training_config(mapping_path: Path, *, expected_mapping_sha256: str,
                    cache_index: str, expected_cache_index_sha256: str,
                    code_archive_sha256: str | None = None) -> dict:
    require(file_sha256(mapping_path) == expected_mapping_sha256, "weak_private_mapping_pin_mismatch")
    require(HEX64.fullmatch(expected_cache_index_sha256) is not None, "weak_cache_index_pin_invalid")
    mapping = load_json(mapping_path)
    config = dict(mapping["runtime_overrides"])
    for key in ("provided_validation_labels", "provided_validation_anchors", "expected_provided_validation_anchors_sha256"):
        config.pop(key, None)
    config.update(experiment_id="W1", track="weak_candidate_unselected", human_verified=False,
                  feature_cache_index=cache_index, expected_feature_cache_index_sha256=expected_cache_index_sha256,
                  fact_axis_manifest="/app/training/temporal/fact-axis.v1.json", expected_fact_axis_manifest_sha256=AXIS_SHA,
                  taxonomy_path="/app/annotations/taxonomy.v1.json", expected_taxonomy_sha256=TAXONOMY_SHA,
                  runtime_lock="/app/training/runtime-lock.json", device="cuda:0", seed=20261017,
                  epochs=12, max_steps=336, learning_rate=.001, weight_decay=.01, teacher_weight=.1,
                  coarse_anchor_weight=.5, coarse_anchor_tolerance_sec=1.0, hidden_dim=128, mil_temperature=.25,
                  output_dir="/runs/W1", untrained_output_dir="/runs/W1-untrained",
                  code_archive_sha256=code_archive_sha256)
    return config


def evaluation_config(training: dict, *, mapping_path: Path, expected_mapping_sha256: str,
                      checkpoint_dir: Path, output_dir: str) -> dict:
    require(file_sha256(mapping_path) == expected_mapping_sha256, "weak_private_mapping_pin_mismatch")
    mapping = load_json(mapping_path)["runtime_overrides"]
    # A fresh immutable mapping may replace only the exposure snapshot paths
    # while retaining identical authority/label/teacher/plan bytes.
    config = {**training, **mapping}
    for key in ("provided_train_labels", "provided_train_anchors", "machine_annotations"):
        config.pop(key, None)
    for key in ("provided_validation_labels", "expected_attempts", "expected_attempts_sha256"):
        config[key] = mapping[key]
    # Read only saved model metadata and the already frozen threshold; this
    # command does not inspect validation labels or prediction values.
    config.update(checkpoint_dir=str(checkpoint_dir), expected_checkpoint_manifest_sha256=file_sha256(checkpoint_dir / "checkpoint.json"),
                  video_threshold=str(checkpoint_dir / "video-threshold.json"),
                  expected_video_threshold_sha256=file_sha256(checkpoint_dir / "video-threshold.json"),
                  validation_output_dir=output_dir)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    train = modes.add_parser("train")
    train.add_argument("--cache-index", required=True)
    train.add_argument("--expected-cache-index-sha256", required=True)
    train.add_argument("--code-archive-sha256")
    evaluate = modes.add_parser("evaluate")
    evaluate.add_argument("--training-config", type=Path, required=True)
    evaluate.add_argument("--checkpoint-dir", type=Path, required=True)
    evaluate.add_argument("--validation-output-dir", required=True)
    for mode in (train, evaluate):
        mode.add_argument("--mapping", type=Path, required=True)
        mode.add_argument("--expected-mapping-sha256", required=True)
        mode.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "train":
            config = training_config(args.mapping, expected_mapping_sha256=args.expected_mapping_sha256,
                cache_index=args.cache_index, expected_cache_index_sha256=args.expected_cache_index_sha256,
                code_archive_sha256=args.code_archive_sha256)
        else:
            config = evaluation_config(load_json(args.training_config), mapping_path=args.mapping,
                expected_mapping_sha256=args.expected_mapping_sha256, checkpoint_dir=args.checkpoint_dir,
                output_dir=args.validation_output_dir)
        print(json.dumps(write_config(args.output, config), sort_keys=True))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "weak_config_preparation_failed", "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
