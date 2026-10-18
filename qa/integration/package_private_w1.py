"""Freeze private W1/W2 metadata for relocation; never print private records.

This is separate from the public source bundle. It copies immutable bytes,
checks both relocated role gates and writes path/hash mappings. It performs
no media access, inference, transfer, training or service action.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_w2_approval(approval: dict, entries: dict, teacher_sha256: str | None) -> None:
    """A new W2 release cannot inherit W1's compute authorization or pins."""
    require(approval.get("experiment_id") == "W2"
            and approval.get("compute_released") is True
            and isinstance(approval.get("compute_release_authorization"), str)
            and "W2" in approval["compute_release_authorization"].upper(),
            "w2_specific_compute_approval_missing")
    required = {"machine_annotations", "runtime_configuration", "question_policy",
                "teacher_terminal_receipt", "proposal_selection_index"}
    require(required.issubset(entries), "w2_required_metadata_missing")
    expected = approval.get("expected_artifact_sha256", {})
    require(isinstance(teacher_sha256, str) and teacher_sha256 == expected.get("machine_annotations"),
            "w2_independent_teacher_pin_missing")
    for key, entry in entries.items():
        if key != "a20_approval":
            require(expected.get(key) == entry["sha256"], "w2_independent_artifact_pin_mismatch")


def freeze(config_path: Path, expected_config_sha256: str,
           approval_path: Path, expected_approval_sha256: str,
           machine_annotations: Path, expected_attempts: Path, *,
           allow_stale_preparation: bool = False,
           extras: dict[str, Path] | None = None,
           expected_extra_sha256: dict[str, str] | None = None,
           bundle_name: str | None = None,
           use_reserved_directory: bool = False,
           expected_machine_annotations_sha256: str | None = None) -> dict:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    require(digest(config_path) == expected_config_sha256, "base_config_pin_mismatch")
    require(digest(approval_path) == expected_approval_sha256, "independent_approval_pin_mismatch")
    if expected_machine_annotations_sha256 is not None:
        require(digest(machine_annotations) == expected_machine_annotations_sha256,
                "independent_teacher_pin_mismatch")
    config, approval = load(config_path), load(approval_path)
    require(approval.get("base_run_config_sha256") == expected_config_sha256
            and approval.get("machine_experiment_authorized") is True,
            "machine_experiment_approval_missing")
    require(digest(Path(config["split_path"])) == config["split_manifest_sha256"]
            == approval["split_manifest_sha256"], "frozen_split_pin_mismatch")
    require(digest(Path(config["scenario_path"])) == config["scenario_manifest_sha256"],
            "frozen_scenario_pin_mismatch")
    coverage = load(Path(config["coverage_status_path"]))
    inventory = Path(coverage["source_inventory_path"])
    require(digest(inventory) == coverage["source_inventory_sha256"], "inventory_pin_mismatch")
    labels = config["provided_label_paths"]
    auth = config["experiment_authorization"]
    for role in ("train", "validation"):
        require(digest(Path(labels[role])) == auth["expected_provided_label_manifests_sha256"][role],
                "role_label_pin_mismatch")
    plan = load(expected_attempts)
    require(plan.get("created_before_prediction_execution") is True
            and plan.get("split_assignments_sha256") == approval["expected_assignments_sha256"]
            and plan.get("provided_labels_sha256") == auth["expected_provided_label_manifests_sha256"]["validation"]
            and len(plan.get("records", [])) == 6, "validation_plan_binding_mismatch")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = bundle_name or "w1-metadata-" + stamp
    require(re.fullmatch(r"w[12]-metadata-[0-9]{8}T[0-9]{12}Z", name) is not None,
            "private_bundle_name_invalid")
    destination = root / "qa" / "integration" / "private" / name
    require("private" in destination.parts and not destination.is_symlink(), "private_bundle_destination_invalid")
    from deploy.private_backup import restrict_directory
    if use_reserved_directory:
        require(destination.is_dir() and {p.name for p in destination.iterdir()} == {"reservation.json"},
                "private_bundle_reservation_invalid")
        reservation = load(destination / "reservation.json")
        require(reservation.get("name") == name
                and reservation.get("status") == "paths_reserved_only_no_inputs_snapshotted"
                and reservation.get("compute_released") is False,
                "private_bundle_reservation_invalid")
    else:
        require(not destination.exists(), "private_bundle_destination_invalid")
        restrict_directory(destination)
    files = {
        "experiment_config": (config_path, "experiment.run-config.json"),
        "automated_split": (Path(config["split_path"]), "automated-split.json"),
        "scenario_groups": (Path(config["scenario_path"]), "scenario-groups.json"),
        "source_manifest": (root / "data/manifests/recordings.jsonl", "recordings.jsonl"),
        "provided_train_labels": (Path(labels["train"]), "provided-binary-labels.train.jsonl"),
        "provided_validation_labels": (Path(labels["validation"]), "provided-binary-labels.validation.jsonl"),
        "exposure_registry": (Path(config["registry_path"]), "model-exposure.jsonl"),
        "exposure_inventory_status": (Path(config["coverage_status_path"]), "exposure-inventory-status.json"),
        "source_inventory_path_override": (inventory, "accepted-inventory.json"),
        "expected_attempts": (expected_attempts, "validation.expected.json"),
        "machine_annotations": (machine_annotations, "machine-annotations.jsonl"),
        "a20_approval": (approval_path, "a20-approval.json"),
    }
    for key, path in (extras or {}).items():
        require(key.isidentifier() and key not in files and path.suffix in {".json", ".jsonl"},
                "extra_private_metadata_invalid")
        require(digest(path) == (expected_extra_sha256 or {}).get(key),
                "extra_private_metadata_pin_mismatch")
        filenames = {"question_policy": "question-policy.json",
                     "runtime_configuration": "runtime-configuration.json",
                     "teacher_terminal_receipt": "teacher-terminal.json",
                     "proposal_selection_index": "proposal-selection-index.json",
                     "export_proof": "export-proof.aggregate.json"}
        files[key] = (path, filenames.get(key, key + path.suffix))
    entries = {}
    captured = {}
    for key, (source, filename) in files.items():
        require(source.is_file() and not source.is_symlink(), "private_metadata_source_invalid")
        before = digest(source)
        content = source.read_bytes()
        require(sha256(content).hexdigest() == before == digest(source),
                "private_metadata_changed_during_snapshot")
        captured[key] = content
        entries[key] = {"relative_path": filename, "sha256": before, "bytes": len(content)}

    if name.startswith("w2-"):
        validate_w2_approval(approval, entries, expected_machine_annotations_sha256)
        runtime = json.loads(captured["runtime_configuration"].decode("utf-8-sig"))
        release = runtime.get("compute_release_authorization", {})
        require(release.get("released_by_A18_or_root") is True
                and release.get("release_reference") == approval["compute_release_authorization"],
                "w2_runtime_compute_approval_mismatch")
        require(runtime.get("expected_machine_annotations_sha256") == expected_machine_annotations_sha256,
                "w2_runtime_teacher_pin_mismatch")
    require(all(digest(source) == entries[key]["sha256"] for key, (source, _) in files.items()),
            "private_metadata_changed_during_snapshot")
    for key, entry in entries.items():
        with (destination / entry["relative_path"]).open("xb") as stream:
            stream.write(captured[key])

    copied = lambda key: destination / entries[key]["relative_path"]
    gate_results = {}
    for role in ("train", "validation"):
        command = [sys.executable, str(root / "scripts/check_automated_experiment.py"),
            "--config", str(copied("experiment_config")),
            "--expected-config-sha256", expected_config_sha256, "--role", role,
            "--manifest", str(copied("source_manifest")),
            "--split", str(copied("automated_split")),
            "--scenario", str(copied("scenario_groups")),
            "--provided-labels", str(copied("provided_" + role + "_labels")),
            "--registry", str(copied("exposure_registry")),
            "--coverage-status", str(copied("exposure_inventory_status")),
            "--source-inventory-override", str(copied("source_inventory_path_override"))]
        outcome = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
        proof = json.loads(outcome.stdout)
        stale_only = set(proof.get("errors", {})) == {"exposure_inventory_snapshot_stale"}
        require((outcome.returncode == 0 and proof.get("ok") is True)
                or (allow_stale_preparation and outcome.returncode == 2 and stale_only),
                "relocated_experiment_gate_failed")
        gate_results[role] = {"ok": proof.get("ok"), "errors": proof.get("errors", {})}

    require(all(digest(source) == entries[key]["sha256"] for key, (source, _) in files.items()),
            "private_metadata_changed_during_gate")

    container = PurePosixPath("/private") / name
    remote = PurePosixPath("/data/forte-ml/private") / name
    overrides = {key: str(container / value["relative_path"]) for key, value in entries.items()}
    overrides.update(expected_experiment_config_sha256=expected_config_sha256,
        expected_machine_annotations_sha256=entries["machine_annotations"]["sha256"],
        expected_attempts_sha256=entries["expected_attempts"]["sha256"],
        compute_release_authorization={"released_by_A18_or_root": approval.get("compute_released") is True,
            "release_reference": approval.get("compute_release_authorization")})
    for key in extras or {}:
        overrides["expected_" + key + "_sha256"] = entries[key]["sha256"]
    if "question_policy" in entries:
        overrides["anchor_question_policy"] = overrides["question_policy"]
        overrides["expected_anchor_question_policy_sha256"] = entries["question_policy"]["sha256"]
    if "coarse_anchor_pins" in entries:
        pins = load(copied("coarse_anchor_pins"))
        require(pins.get("assignments_sha256") == approval["expected_assignments_sha256"]
                and pins.get("split_file_sha256") == approval["split_manifest_sha256"],
                "coarse_anchor_assignment_pin_mismatch")
        overrides["expected_coarse_anchor_source_sha256"] = pins["anchor_source_sha256"]
    mapping = {"schema_version": "forte.private-machine-relocation.v1" if name.startswith("w2-") else "forte.private-w1-relocation.v1", "name": name,
        "experiment_id": "W2" if name.startswith("w2-") else "W1",
        "status": "immutable_metadata_only", "private": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_directory": str(remote), "container_directory": str(container),
        "mount": "/data/forte-ml/private:/private:ro", "files": entries,
        "runtime_overrides": overrides, "relocated_gate_results": gate_results,
        "snapshot_fresh_at_package_time": all(value["ok"] for value in gate_results.values()),
        "fresh_gate_must_be_rechecked_before_execution": True,
        "compute_release_recorded": approval.get("compute_released") is True,
        "final_check_role_label_file_included": False,
        "base_config_bytes_modified": False, "machine_export_bytes_modified": False,
        "media_or_weights_included": False, "contents_must_not_be_logged_or_publicly_shared": True}
    mapping_path = destination / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    archive = destination / "metadata.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for filename in [value["relative_path"] for value in entries.values()] + ["mapping.json"]:
            bundle.add(destination / filename, arcname=name + "/" + filename, recursive=False)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        require(len(members) == len(entries) + 1 and all(member.isfile() and not member.name.startswith("/")
                and ".." not in PurePosixPath(member.name).parts for member in members),
                "private_bundle_archive_invalid")
        for key, value in entries.items():
            stream = bundle.extractfile(name + "/" + value["relative_path"])
            require(stream is not None and sha256(stream.read()).hexdigest() == value["sha256"],
                    "private_bundle_archived_digest_mismatch")
    require(all(digest(source) == entries[key]["sha256"] for key, (source, _) in files.items()),
            "private_metadata_changed_during_archive")
    return {"ok": True, "archive": str(archive), "archive_sha256": digest(archive),
        "archive_bytes": archive.stat().st_size, "mapping": str(mapping_path),
        "mapping_sha256": digest(mapping_path), "remote_directory": str(remote),
        "container_directory": str(container), "metadata_files": len(entries),
        "snapshot_fresh_at_package_time": mapping["snapshot_fresh_at_package_time"],
        "private_contents_printed": False, "gpu_or_network_calls": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data/splits/automated_experiment.run-config.json"))
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--approval", type=Path, default=Path("qa/integration/private/automated-experiment-pin.json"))
    parser.add_argument("--expected-approval-sha256", required=True)
    parser.add_argument("--machine-annotations", type=Path, default=Path("annotations/private/automatic/machine_annotations.jsonl"))
    parser.add_argument("--expected-attempts", type=Path, default=Path("evaluation/private/weak-W1-validation.expected.json"))
    parser.add_argument("--allow-stale-preparation", action="store_true")
    parser.add_argument("--bundle-name")
    parser.add_argument("--use-reserved-directory", action="store_true")
    parser.add_argument("--expected-machine-annotations-sha256")
    parser.add_argument("--extra-private", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--expected-extra-sha256", action="append", default=[], metavar="KEY=SHA256")
    args = parser.parse_args()
    try:
        extras = dict(item.split("=", 1) for item in args.extra_private)
        result = freeze(args.config, args.expected_config_sha256, args.approval,
            args.expected_approval_sha256, args.machine_annotations, args.expected_attempts,
            allow_stale_preparation=args.allow_stale_preparation,
            extras={key: Path(value) for key, value in extras.items()},
            expected_extra_sha256=dict(item.split("=", 1) for item in args.expected_extra_sha256),
            bundle_name=args.bundle_name, use_reserved_directory=args.use_reserved_directory,
            expected_machine_annotations_sha256=args.expected_machine_annotations_sha256)
        print(json.dumps(result))
        return 0
    except Exception as error:
        code = str(error) if isinstance(error, ValueError) and str(error).replace("_", "").isalpha() else "private_metadata_packaging_failed"
        print(json.dumps({"ok": False, "error_code": code, "error_type": type(error).__name__, "private_contents_printed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
