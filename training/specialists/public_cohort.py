"""One predeclared comparison on a shared public holdout inventory.

Registration occupies the *same* global filename as ``claim_sealed_test``.
Both selected models, their configurations and their protocols must be frozen
before registration. Each fixed member then gets one attempt, including failed
attempts. Registration and attempt files are immutable: write evaluation results
elsewhere. This is an audit/coordination boundary, not a filesystem sandbox.

All paths are local absolute paths. Hash verification reads file bytes; this
module never decodes images, imports model runtimes, or computes test metrics.
The caller must claim before opening heldout examples or invoking inference.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from .common import (canonical_sha256, file_sha256, load_json, pinned_json,
                     relative_file, require, sha)

SCHEMA = "forte.public-eval-cohort.v1"
AUTHORIZATION_SCHEMA = "forte.public-eval-cohort-authorization.v1"
ATTEMPT_SCHEMA = "forte.public-eval-cohort-attempt.v1"
MEMBER_IDS = frozenset({"detector", "crop_vlm"})
_PIN_FIELDS = frozenset({"run", "checkpoint", "config", "protocol"})
_ARTIFACT_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}$")


def _path(value: Any) -> Path:
    require(isinstance(value, (str, Path)) and bool(str(value)) and "://" not in str(value),
            "public_cohort_local_path_required")
    path = Path(value)
    require(path.is_absolute(), "public_cohort_absolute_path_required")
    return path.resolve()


def _pin(value: Any) -> dict:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and sha(value.get("sha256")), "public_cohort_artifact_pin_invalid")
    path = _path(value["path"])
    require(path.is_file() and file_sha256(path) == value["sha256"],
            "public_cohort_artifact_hash_mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def sealed_inventory(lock: dict) -> dict[str, str]:
    """The legacy full-inventory identity, never a task's sampled subset."""
    require(isinstance(lock, dict) and isinstance(lock.get("files"), dict),
            "public_cohort_dataset_inventory_missing")
    files = {name: digest for name, digest in lock["files"].items()
             if isinstance(name, str) and name.startswith("sealed_test/")}
    require(bool(files) and all(sha(value) for value in files.values()),
            "public_cohort_test_inventory_invalid")
    for name in files:
        parts = name.split("/")
        require(len(parts) >= 2 and all(part not in {"", ".", ".."} for part in parts)
                and "\\" not in name and ":" not in name,
                "public_cohort_test_inventory_path_invalid")
    return files


def _dataset(dataset_pin: dict) -> tuple[dict, str]:
    dataset_pin = _pin(dataset_pin)
    path = Path(dataset_pin["path"])
    require(path.name == "dataset-lock.json", "public_cohort_dataset_lock_name_invalid")
    lock = load_json(path)
    require(isinstance(lock, dict)
            and lock.get("schema_version") == "forte.specialists.public-dataset-lock.v1"
            and lock.get("state") == "frozen" and lock.get("origin") == "public",
            "public_cohort_public_frozen_dataset_required")
    require(lock.get("dataset_sha256") == canonical_sha256(
        {key: value for key, value in lock.items() if key != "dataset_sha256"}),
        "public_cohort_dataset_digest_mismatch")
    files = sealed_inventory(lock)
    for name, digest in files.items():
        require(file_sha256(relative_file(path.parent, name)) == digest,
                "public_cohort_test_file_hash_mismatch")
    return lock, canonical_sha256(files)


def _authorization(pin: dict, registry: Path, dataset_sha256: str, inventory_sha256: str) -> dict:
    pin = _pin(pin)
    auth = load_json(Path(pin["path"]))
    require(isinstance(auth, dict) and auth.get("schema_version") == AUTHORIZATION_SCHEMA
            and auth.get("authorized_by") == "root"
            and isinstance(auth.get("authorization_reference"), str)
            and bool(auth["authorization_reference"].strip()),
            "public_cohort_root_authorization_required")
    require(_path(auth.get("registry_dir")) == registry
            and auth.get("dataset_lock_sha256") == dataset_sha256
            and auth.get("sealed_inventory_sha256") == inventory_sha256
            and auth.get("member_ids") == sorted(MEMBER_IDS),
            "public_cohort_authorization_scope_mismatch")
    return pin


def _members(value: Any, registry: Path, dataset_pin: dict, inventory: str) -> dict:
    require(isinstance(value, dict) and set(value) == MEMBER_IDS,
            "public_cohort_exact_members_required")
    result = {}
    for member_id in sorted(MEMBER_IDS):
        member = value[member_id]
        require(isinstance(member, dict) and set(member) == _PIN_FIELDS | {"artifacts"},
                "public_cohort_member_fields_invalid")
        pins = {key: _pin(member[key]) for key in sorted(_PIN_FIELDS)}
        artifacts = member["artifacts"]
        require(isinstance(artifacts, dict) and "evaluation_source" in artifacts
                and all(isinstance(name, str) and _ARTIFACT_NAME.fullmatch(name) for name in artifacts),
                "public_cohort_source_artifacts_required")
        pins["artifacts"] = {name: _pin(pin) for name, pin in sorted(artifacts.items())}
        config = load_json(Path(pins["config"]["path"]))
        require(isinstance(config, dict)
                and _path(config.get("test_exposure_registry_dir")) == registry
                and _path(config.get("dataset_dir")) == Path(dataset_pin["path"]).parent
                and config.get("expected_dataset_lock_sha256") == dataset_pin["sha256"],
                "public_cohort_frozen_config_scope_mismatch")
        run = load_json(Path(pins["run"]["path"]))
        require(isinstance(run, dict)
                and run.get("state") in {"trained_unselected", "trained_unselected_pending_test"}
                and run.get("config_sha256") == pins["config"]["sha256"]
                and run.get("evaluation_protocol_sha256") == pins["protocol"]["sha256"]
                and run.get("dataset_lock_sha256") == dataset_pin["sha256"]
                and type(run.get("optimizer_steps")) is int and run["optimizer_steps"] > 0,
                "public_cohort_selected_run_mismatch")
        require(type(run.get("test_evaluation_count", 0)) is int
                and run.get("test_evaluation_count", 0) == 0
                and not run.get("test_state") and not run.get("test_exposure_receipt")
                and run.get("sealed_test_used_for_training_or_selection", False) is False
                and not run.get("memory_smoke_only", False),
                "public_cohort_run_already_exposed_or_smoke")
        if member_id == "detector":
            require(run.get("schema_version") == "forte.specialists.public-run.v1"
                    and run.get("checkpoint_sha256") == pins["checkpoint"]["sha256"],
                    "public_cohort_selected_checkpoint_mismatch")
        else:
            require(run.get("schema_version") == "forte.public-crop-lora.run.v1"
                    and run.get("adapter_files", {}).get("adapter_model.safetensors")
                    == pins["checkpoint"]["sha256"], "public_cohort_selected_checkpoint_mismatch")
        protocol = load_json(Path(pins["protocol"]["path"]))
        require(isinstance(protocol, dict) and protocol.get("member_id") == member_id
                and protocol.get("dataset_lock_sha256") == dataset_pin["sha256"]
                and protocol.get("sealed_inventory_sha256") == inventory
                and protocol.get("selection_scope") == "public_train_dev_only"
                and protocol.get("test_reuse_for_tuning_allowed") is False,
                "public_cohort_protocol_scope_mismatch")
        source_field = "runner_sha256" if member_id == "detector" else "source_sha256"
        require(protocol.get(source_field) == pins["artifacts"]["evaluation_source"]["sha256"],
                "public_cohort_protocol_source_mismatch")
        result[member_id] = pins
    return result


def _exclusive_json(path: Path, value: dict, error: str) -> None:
    """Existence consumes the reservation, even if writing or later work fails."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        require(False, error)


def register_cohort(registry_dir: Path, dataset_lock_path: Path,
                    expected_dataset_lock_sha256: str, members: dict, *,
                    authorization_path: Path, expected_authorization_sha256: str) -> Path:
    """Reserve the only joint comparison after both models are selected on dev.

    The root orchestrator supplies an independently pinned authorization naming
    the central registry, full dataset and inventory. This helper cannot create
    that authorization. Existing legacy/cohort/corrupt reservations all block.
    """
    registry = _path(registry_dir)
    dataset_pin = _pin({"path": str(_path(dataset_lock_path)), "sha256": expected_dataset_lock_sha256})
    _, inventory = _dataset(dataset_pin)
    auth = _authorization({"path": str(_path(authorization_path)), "sha256": expected_authorization_sha256},
                          registry, dataset_pin["sha256"], inventory)
    frozen = _members(members, registry, dataset_pin, inventory)
    registry.mkdir(parents=True, exist_ok=True)
    receipt = registry / f"public-sealed-{inventory}.json"
    require(not receipt.is_symlink(), "public_cohort_registry_symlink_rejected")
    value = {"schema_version": SCHEMA, "state": "registered", "member_ids": sorted(MEMBER_IDS),
             "registry_dir": str(registry), "dataset_lock": dataset_pin,
             "sealed_inventory_sha256": inventory, "members": frozen, "authorization": auth,
             "protocol_implementation_sha256": file_sha256(Path(__file__)),
             "registered_at_utc": datetime.now(timezone.utc).isoformat(),
             "interpretation": "one_joint_comparison_not_independent_replications",
             "attempt_policy": "one_per_declared_member_failure_consumes_no_reset"}
    _exclusive_json(receipt, value, "public_cohort_inventory_already_reserved_or_exposed")
    return receipt


def verify_cohort(cohort_receipt: Path, expected_cohort_sha256: str) -> dict:
    """Rehash the shared inventory and *all* members, without claiming access."""
    supplied = Path(cohort_receipt)
    require(not supplied.is_symlink(), "public_cohort_registry_symlink_rejected")
    receipt = _path(supplied)
    value = pinned_json(receipt, expected_cohort_sha256)
    require(value.get("schema_version") == SCHEMA and value.get("state") == "registered"
            and value.get("member_ids") == sorted(MEMBER_IDS)
            and value.get("protocol_implementation_sha256") == file_sha256(Path(__file__)),
            "public_cohort_receipt_schema_or_source_mismatch")
    registry = _path(value.get("registry_dir"))
    require(receipt.parent == registry, "public_cohort_receipt_registry_mismatch")
    dataset_pin = _pin(value.get("dataset_lock"))
    _, inventory = _dataset(dataset_pin)
    require(value.get("sealed_inventory_sha256") == inventory
            and receipt.name == f"public-sealed-{inventory}.json", "public_cohort_inventory_mismatch")
    _authorization(value.get("authorization"), registry, dataset_pin["sha256"], inventory)
    require(_members(value.get("members"), registry, dataset_pin, inventory) == value["members"],
            "public_cohort_noncanonical_member_pins")
    return value


def claim_cohort_member(config: dict, lock: dict, checkpoint_sha256: str, *,
                        cohort_receipt: Path, expected_cohort_sha256: str,
                        member_id: str, run_path: Path, config_path: Path,
                        protocol_path: Path) -> Path:
    """Consume one member attempt before any heldout decoding/model execution.

    Do not rewrite the returned marker on completion. Save metrics and completion
    separately. Call ``verify_cohort`` immediately before loading frozen model
    files too; run final models sequentially under the host's compute deadline.
    """
    require(isinstance(member_id, str) and member_id in MEMBER_IDS,
            "public_cohort_member_id_invalid")
    cohort = verify_cohort(cohort_receipt, expected_cohort_sha256)
    member = cohort["members"][member_id]
    require(_path(run_path) == Path(member["run"]["path"])
            and _path(config_path) == Path(member["config"]["path"])
            and _path(protocol_path) == Path(member["protocol"]["path"])
            and checkpoint_sha256 == member["checkpoint"]["sha256"],
            "public_cohort_claim_artifact_mismatch")
    require(config == load_json(Path(member["config"]["path"]))
            and lock == load_json(Path(cohort["dataset_lock"]["path"])),
            "public_cohort_claim_config_or_dataset_mismatch")
    receipt = _path(cohort_receipt)
    attempt = receipt.with_name(f"{receipt.stem}.member-{member_id}.attempt.json")
    require(not attempt.is_symlink(), "public_cohort_registry_symlink_rejected")
    _exclusive_json(attempt, {"schema_version": ATTEMPT_SCHEMA, "state": "attempted",
        "member_id": member_id, "cohort_sha256": expected_cohort_sha256,
        "sealed_inventory_sha256": cohort["sealed_inventory_sha256"],
        "dataset_lock_sha256": cohort["dataset_lock"]["sha256"],
        "run_sha256": member["run"]["sha256"], "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": member["config"]["sha256"], "protocol_sha256": member["protocol"]["sha256"],
        "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure_consumes_attempt": True}, "public_cohort_member_already_attempted")
    return attempt
