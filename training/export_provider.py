"""Optionally export private provider prompts/targets while preserving canonical inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forte.decision import DecisionConfig

from .alignment import provider_records
from .guards import TrainingGateError, canonical_sha256, file_sha256, load_json, require
from .train import load_inputs, write_json


def export(config_path: Path, output: Path) -> dict:
    config = load_json(config_path)
    records, input_hashes = load_inputs(config)
    decision_path = Path(config.get("decision_config", "configs/decision/default.json"))
    decision_config = DecisionConfig(**load_json(decision_path))
    aligned, aggregate = provider_records(records, config=decision_config,
                                           media_root=Path(config["media_root"]).resolve())
    destination = output.resolve()
    private_root = Path(config["train_manifest"]).resolve().parent
    require(destination.is_relative_to(private_root), "provider_export_must_stay_private")
    require(not destination.exists(), "provider_export_already_exists")
    require(not destination.with_suffix(destination.suffix + ".tmp").exists(), "provider_export_temporary_already_exists")
    payload = {"schema_version": "forte.training.provider-export.v1", **input_hashes, **aggregate,
               "decision_config_sha256": file_sha256(decision_path),
               "canonical_manifest_preserved": True,
               "records": [{"canonical_record_sha256": canonical_sha256(canonical),
                            "provider_input": {field: provider[field] for field in
                                               ("frames", "prompt", "prompt_version", "config_version", "target")}}
                           for canonical, provider in zip(records, aligned)]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, payload)
    return {"status": "exported_private", **aggregate, "canonical_manifest_preserved": True,
            "gpu_used": False, "model_inference_executed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.config, args.output)))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "private_provider_export_failed",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
