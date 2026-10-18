"""Local-only recipe binding helper; no model load, upload or remote mutation.

First bind a source archive and successful tiny GPU proof to new local/remote
recipes. Run again against those recipes with the exact root release and idle
handoff receipt only when root explicitly releases the reservation. Existing
recipes and all historical receipts remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from training.guards import TrainingGateError, file_sha256, load_json, require
from training.qwen38_preflight import (reserve_attempt_plan, semantic_recipe_sha256,
                                       validate_runtime_relocation)
from training.qwen38_silver import _verify_smoke_proof, collect_release_bindings, load_inputs


def _write_new(path: Path, value: dict) -> None:
    require("private" in path.resolve().parts and not path.exists(), "qwen38_bind_new_private_output_required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _copy_new(source: Path, destination: Path) -> None:
    require(source.is_file(), "qwen38_bind_source_artifact_missing")
    if destination.exists():
        require(file_sha256(destination) == file_sha256(source), "qwen38_bind_existing_artifact_conflict")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as origin, destination.open("xb") as target:
        shutil.copyfileobj(origin, target)
    require(file_sha256(source) == file_sha256(destination), "qwen38_bind_copy_hash_mismatch")


def validate_tiny_source_archive(proof: dict, archive_path: Path) -> None:
    """Join the tiny proof's actual code hashes to the immutable source tar."""
    required = {"check_gradient.py", "architectures.py", "data.py", "train.py"}
    pins = proof.get("source_sha256", {})
    require(isinstance(pins, dict) and required <= pins.keys(), "qwen38_bind_tiny_source_pins_missing")
    with tarfile.open(archive_path, "r:*") as archive:
        for name in sorted(required):
            member = archive.getmember("training/" + name)
            require(member.isfile(), "qwen38_bind_tiny_source_archive_entry_invalid")
            source = archive.extractfile(member)
            require(source is not None, "qwen38_bind_tiny_source_archive_entry_missing")
            with source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            require(actual == pins[name], "qwen38_bind_tiny_source_archive_pin_mismatch")


def bind_recipes(local_config: dict, runtime_config: dict, *, source_archive: Path,
                 tiny_proof: Path | None, output_local: Path, output_runtime: Path,
                 staged_tiny: Path | None = None, runtime_tiny_path: str | None = None,
                 inventory: Path | None = None, exposure_status: Path | None = None,
                 central_registry: Path | None = None, release: Path | None = None,
                 handoff: Path | None = None, reserve: bool = False,
                 reservation_output: Path | None = None) -> dict:
    """Bind immutable artifacts, validate local data and optionally reserve.

    Root release creation is intentionally external. Its semantic recipe hash
    must match the exact generated remote recipe before reservation is allowed.
    """
    require(not output_local.exists() and not output_runtime.exists(), "qwen38_bind_outputs_already_exist")
    require(source_archive.is_file(), "qwen38_bind_source_archive_missing")
    local, remote = dict(local_config), dict(runtime_config)
    validate_runtime_relocation(local, remote)
    source_hash = file_sha256(source_archive)
    local["code_archive_sha256"] = remote["code_archive_sha256"] = source_hash
    if central_registry:
        local["exposure_registry"] = str(central_registry.resolve())
    if exposure_status:
        local["exposure_inventory_status"] = str(exposure_status.resolve())
    if inventory:
        local["source_inventory_path_override"] = str(inventory.resolve())
    if tiny_proof:
        require(staged_tiny is not None and runtime_tiny_path is not None
                and PurePosixPath(runtime_tiny_path).is_absolute(), "qwen38_bind_tiny_paths_missing")
        _copy_new(tiny_proof, staged_tiny)
        local["tiny_smoke_receipt"] = str(staged_tiny.resolve())
        remote["tiny_smoke_receipt"] = runtime_tiny_path
        local["expected_tiny_smoke_receipt_sha256"] = remote["expected_tiny_smoke_receipt_sha256"] = file_sha256(staged_tiny)
    if release:
        require(release.is_file(), "qwen38_bind_root_release_missing")
        local["compute_release"] = str(release.resolve())
        local["expected_compute_release_sha256"] = remote["expected_compute_release_sha256"] = file_sha256(release)
    validate_runtime_relocation(local, remote)
    rows, bindings, audit = load_inputs(local)
    bindings = collect_release_bindings(local, bindings, audit)
    if local.get("quantization") == "nf4":
        with tarfile.open(source_archive, "r:*") as archive:
            member = archive.getmember("training/qwen38_proof.py")
            require(member.isfile(), "qwen38_bind_nf4_helper_archive_entry_invalid")
            source = archive.extractfile(member)
            require(source is not None, "qwen38_bind_nf4_helper_archive_entry_missing")
            with source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            require(actual == bindings["nf4_proof_helper_source_sha256"], "qwen38_bind_nf4_helper_archive_pin_mismatch")
    proof_ready = tiny_proof is not None or bool(local.get("expected_tiny_smoke_receipt_sha256"))
    if proof_ready:
        _verify_smoke_proof(local, bindings)
        validate_tiny_source_archive(load_json(Path(local["tiny_smoke_receipt"])), source_archive)
    if reserve:
        require(release is not None and handoff is not None and handoff.is_file() and proof_ready,
                "qwen38_bind_reservation_authority_missing")
        authorization = load_json(release)
        require(authorization.get("idle_handoff_receipt_sha256") == file_sha256(handoff)
                and authorization.get("input_bindings") == bindings
                and authorization.get("config_sha256") == semantic_recipe_sha256(remote),
                "qwen38_bind_root_release_or_handoff_mismatch")
        require(reservation_output is not None, "qwen38_bind_reservation_output_missing")
        local["attempt_reservation"] = str(reservation_output.resolve())
        # Only paths can change; the exact runtime reservation path was predeclared.
        validate_runtime_relocation(local, remote)
    _write_new(output_local, local)
    _write_new(output_runtime, remote)
    reservation = None
    if reserve:
        reservation = reserve_attempt_plan(local, config_sha256=file_sha256(output_local),
            plan_path=Path(local["attempt_plan"]), release_path=release,
            output_path=reservation_output, runtime_config=remote)
    return {"schema_version": "forte.qwen38-local-bind.v1", "status": "bound",
            "source_archive_sha256": source_hash, "local_config_sha256": file_sha256(output_local),
            "runtime_config_sha256": file_sha256(output_runtime),
            "runtime_semantic_recipe_sha256": semantic_recipe_sha256(remote),
            "local_semantic_recipe_sha256": semantic_recipe_sha256(local), "input_bindings": bindings,
            "coverage": audit["coverage"], "tiny_gpu_proof_ready": proof_ready,
            "reservation_created": reservation is not None, "reservation": reservation,
            "model_calls_made": False, "remote_mutations_made": False,
            "helper_sha256": file_sha256(Path(__file__))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("local-config", "runtime-config", "source-archive", "output-local", "output-runtime"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("tiny-proof", "staged-tiny", "inventory", "exposure-status", "central-registry",
                "release", "handoff", "reservation-output", "receipt"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--runtime-tiny-path")
    parser.add_argument("--reserve", action="store_true", help="requires supplied exact fresh root release and idle handoff")
    args = parser.parse_args()
    try:
        result = bind_recipes(load_json(args.local_config), load_json(args.runtime_config),
            source_archive=args.source_archive, tiny_proof=args.tiny_proof,
            output_local=args.output_local, output_runtime=args.output_runtime,
            staged_tiny=args.staged_tiny, runtime_tiny_path=args.runtime_tiny_path,
            inventory=args.inventory, exposure_status=args.exposure_status, central_registry=args.central_registry,
            release=args.release, handoff=args.handoff, reserve=args.reserve,
            reservation_output=args.reservation_output)
        if args.receipt:
            _write_new(args.receipt, result)
        # Private identities stay in the written receipt; terminal output is aggregate.
        print(json.dumps({key: result[key] for key in ("schema_version", "status", "source_archive_sha256",
            "runtime_semantic_recipe_sha256", "tiny_gpu_proof_ready", "reservation_created", "model_calls_made", "remote_mutations_made")}))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc), "model_calls_made": False}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "qwen38_bind_failed", "error_type": type(exc).__name__, "model_calls_made": False}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
