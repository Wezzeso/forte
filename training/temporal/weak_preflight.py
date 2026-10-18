"""Metadata-only weak experiment gate; never load a model or allocate a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.split_gate import validate_automated_experiment_inputs
from training.guards import TrainingGateError, file_sha256, load_annotations, load_json, require


def resolve_config(config: dict, *, label_split: str = "train") -> dict:
    """Overlay runtime paths on the independently pinned A02 experiment config."""
    result = dict(config)
    if "experiment_config" in config:
        path = Path(config["experiment_config"]).resolve()
        require(path.is_file() and file_sha256(path) == config.get("expected_experiment_config_sha256"),
                "weak_experiment_config_pin_mismatch")
        trusted = load_json(path)
        # Runtime settings may relocate paths, but cannot override the authority.
        result["experiment_authorization"] = trusted["experiment_authorization"]
        result.setdefault("automated_split", trusted.get("automated_split", trusted.get("split_path")))
        result.setdefault("scenario_groups", trusted.get("scenario_groups", trusted.get("scenario_path")))
        result.setdefault("exposure_registry", trusted.get("exposure_registry", trusted.get("registry_path")))
        result.setdefault("exposure_inventory_status", trusted.get("exposure_inventory_status", trusted.get("coverage_status_path")))
        result.setdefault("provided_" + label_split + "_labels", trusted["provided_label_paths"][label_split])
    authority = result.get("experiment_authorization", {})
    result.setdefault("expected_provided_label_source_sha256", authority.get("expected_provided_label_source_sha256"))
    result.setdefault("source_manifest", "data/manifests/recordings.jsonl")
    require(label_split in {"train", "validation"}, "weak_final_check_access_forbidden")
    return result


def validate_metadata(config: dict, *, label_split: str = "train") -> tuple[list[dict], dict]:
    config = resolve_config(config, label_split=label_split)
    label_key = "provided_" + label_split + "_labels"
    paths = {name: Path(config[name]).resolve() for name in
             ("automated_split", "source_manifest", "scenario_groups", label_key,
              "exposure_registry", "exposure_inventory_status")}
    require(all(path.is_file() for path in paths.values()), "weak_data_artifacts_not_ready")
    labels = load_annotations(paths[label_key])
    override = config.get("source_inventory_path_override")
    result = validate_automated_experiment_inputs(
        load_json(paths["automated_split"]), load_annotations(paths["source_manifest"]),
        load_json(paths["scenario_groups"]), experiment_authorization=config.get("experiment_authorization", {}),
        registry_path=paths["exposure_registry"], coverage_status_path=paths["exposure_inventory_status"],
        provided_label_records=labels,
        expected_provided_label_source_sha256=config.get("expected_provided_label_source_sha256", ""),
        label_split=label_split, requested_content_hashes={row.get("content_sha256", row.get("content_hash")) for row in labels},
        requested_recording_ids={row["recording_id"] for row in labels},
        source_inventory_path_override=Path(override).resolve() if override else None,
    )
    require(result.get("ok") is True, "weak_automated_split_exposure_or_label_gate_failed")
    return labels, {"track": "weak_candidate_unselected", "human_verified": False,
                    "label_split": label_split, "contents": len(labels), "metadata_gate_passed": True,
                    "gate_proof": result,
                    "input_checksums": {name: file_sha256(path) for name, path in paths.items()},
                    "gpu_used": False, "training_executed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        _, result = validate_metadata(load_json(args.config))
        print(json.dumps(result))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "weak_metadata_preflight_failed",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
