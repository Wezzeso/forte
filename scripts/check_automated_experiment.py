#!/usr/bin/env python3
"""Recheck a separately pinned machine experiment without media/network calls."""
import argparse
import json
from pathlib import Path
import re

try:
    from .audit_data import read_jsonl, sha256_file, write_json
    from .split_gate import validate_automated_experiment_inputs
except ImportError:
    from audit_data import read_jsonl, sha256_file, write_json
    from split_gate import validate_automated_experiment_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/splits/automated_experiment.run-config.json"))
    parser.add_argument("--expected-config-sha256", required=True, help="Independent trusted orchestrator pin; never inferred from the supplied file")
    parser.add_argument("--role", choices=["train", "validation"], required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    parser.add_argument("--split", type=Path)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--provided-labels", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--coverage-status", type=Path)
    parser.add_argument("--source-inventory-override", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config_digest = sha256_file(args.config)
        if config_digest != args.expected_config_sha256:
            raise ValueError("independent_machine_experiment_config_pin_mismatch")
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        split_path = args.split or Path(config["split_path"])
        scenario_path = args.scenario or Path(config["scenario_path"])
        label_path = args.provided_labels or Path(config["provided_label_paths"][args.role])
        if sha256_file(split_path) != config["split_manifest_sha256"] or sha256_file(scenario_path) != config["scenario_manifest_sha256"]:
            raise ValueError("frozen_experiment_artifact_file_binding_mismatch")
        split = json.loads(split_path.read_text(encoding="utf-8-sig"))
        if sha256_file(args.manifest) != split["source_manifest_sha256"]:
            raise ValueError("machine_experiment_source_manifest_mismatch")
        authorization = config["experiment_authorization"]
        if sha256_file(label_path) != authorization["expected_provided_label_manifests_sha256"][args.role]:
            raise ValueError("frozen_binary_label_file_binding_mismatch")
        report = validate_automated_experiment_inputs(
            split, read_jsonl(args.manifest), json.loads(scenario_path.read_text(encoding="utf-8-sig")),
            experiment_authorization=authorization,
            registry_path=args.registry or Path(config["registry_path"]),
            coverage_status_path=args.coverage_status or Path(config["coverage_status_path"]),
            provided_label_records=read_jsonl(label_path),
            expected_provided_label_source_sha256=authorization["expected_provided_label_source_sha256"],
            label_split=args.role, source_inventory_path_override=args.source_inventory_override,
        )
        report.update(config_sha256=config_digest, split_manifest_sha256=config["split_manifest_sha256"],
                      scenario_manifest_sha256=config["scenario_manifest_sha256"])
        if args.output:
            if not args.output.resolve().is_relative_to(Path("data/splits").resolve()) and "private" not in args.output.resolve().parts:
                raise ValueError("experiment_gate_proof_must_remain_private")
            if args.output.resolve() in {args.config.resolve(), split_path.resolve(), scenario_path.resolve(), label_path.resolve()}:
                raise ValueError("gate_proof_cannot_overwrite_frozen_inputs")
            write_json(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ok"] else 2
    except Exception as exc:
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)) else "automated_experiment_check_failed"
        print(json.dumps({"ok": False, "error_code": code, "human_verified_training_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
