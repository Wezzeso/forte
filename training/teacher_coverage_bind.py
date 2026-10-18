"""Local-only central reservation for the exact 34-call TRAIN expansion.

Consumes a real, fresh root release; never creates compute authority or calls
the teacher. Uses the established teacher-exposure receipt contract unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.exposure_registry import (read_registry, register_content_manifest, valid_hash)
from training.guards import TrainingGateError, canonical_sha256, file_sha256, load_json, require
from training.qwen38_preflight import FIXED_STOP_UTC, validate_role_metadata
from training.teacher_sampling import validate_teacher_plan


def validate_release(release: dict, plan: dict, attempts: dict, *, plan_file_sha256: str,
                     attempt_file_sha256: str, native_sha256: str, handoff_sha256: str,
                     now: float | None = None) -> None:
    now = time.time() if now is None else now
    require(release.get("schema_version") == "forte.teacher-execution-release.v1"
            and release.get("released") is True and release.get("released_by") == "root"
            and isinstance(release.get("reference"), str) and bool(release["reference"].strip())
            and release.get("purpose") == "source_bound_teacher_annotation"
            and release.get("plan_sha256") == plan.get("plan_sha256")
            and release.get("teacher_plan_file_sha256") == plan_file_sha256
            and release.get("attempt_plan_file_sha256") == attempt_file_sha256,
            "coverage_exact_root_release_missing")
    require(release.get("max_model_requests") == 34 and release.get("max_records") == 17
            and release.get("record_offset") == 0 and release.get("anchor_focus_only") is True
            and len(plan.get("records", [])) == 17 and len(attempts.get("attempts", [])) == 34,
            "coverage_root_scope_changed")
    require(release.get("model_manifest_sha256") == "249977e15be3d72425155475e59adf2385747b16e28654c9d13d45761abf75cc"
            and release.get("native_pts_ledger_verified") is True
            and release.get("native_pts_ledger_sha256") == native_sha256
            and release.get("idle_handoff_receipt_sha256") == handoff_sha256
            and release.get("exclusive_idle_gpu_handoff_verified") is True
            and release.get("fresh_role_exposure_verified") is True,
            "coverage_native_teacher_or_handoff_pin_missing")
    require(all(valid_hash(release.get(key)) for key in
                ("producer_source_archive_sha256", "producer_runtime_image_sha256", "coverage_guard_sha256")),
            "coverage_producer_pins_missing")
    wall, issued, expires = release.get("max_runtime_sec"), release.get("issued_at_unix"), release.get("expires_at_unix")
    require(type(wall) is int and 0 < wall <= 1800 and type(issued) in {int, float}
            and type(expires) in {int, float} and issued <= now and now + wall + 120 < expires
            <= FIXED_STOP_UTC.timestamp(), "coverage_release_wall_or_stop_invalid")


def reserve(config: dict, *, plan_path: Path, attempt_path: Path, worklist_path: Path,
            release_path: Path, native_path: Path, handoff_path: Path, output_path: Path) -> dict:
    for path in (plan_path, attempt_path, worklist_path, release_path, native_path, output_path):
        require("private" in path.resolve().parts, "coverage_metadata_must_stay_private")
    require(not output_path.exists(), "coverage_exposure_receipt_already_exists")
    plan, attempts, release = map(load_json, (plan_path, attempt_path, release_path))
    worklist = load_json(worklist_path)
    require(release.get("worklist_file_sha256") == file_sha256(worklist_path)
            and plan.get("worklist_sha256") in {file_sha256(worklist_path), canonical_sha256(worklist)},
            "coverage_exact_worklist_pin_missing")
    validate_teacher_plan(plan, load_json(Path(config["automated_split"])),
                          split_sha256=file_sha256(Path(config["automated_split"])))
    validate_release(release, plan, attempts, plan_file_sha256=file_sha256(plan_path),
        attempt_file_sha256=file_sha256(attempt_path), native_sha256=file_sha256(native_path),
        handoff_sha256=file_sha256(handoff_path))
    require(release["coverage_guard_sha256"] == file_sha256(Path(__file__).with_name("coverage_expansion.py")),
            "coverage_guard_local_source_pin_mismatch")
    contents = {row["content_sha256"] for row in plan["records"]}
    require(len(contents) == 17 and all(row.get("split") == "train"
            and len(row.get("priority_second_indices", [])) == 2 for row in plan["records"]),
            "coverage_exact_train_plan_invalid")
    require(len(worklist.get("records", [])) == 17
            and {row.get("content_sha256") for row in worklist["records"]} == contents
            and all(row.get("split") == "train" for row in worklist["records"]),
            "coverage_worklist_population_mismatch")
    slot_rows = attempts["attempts"]
    require(len({row.get("attempt_id") for row in slot_rows}) == 34
            and [row.get("content_sha256") for row in slot_rows]
            == [row["content_sha256"] for row in plan["records"] for _ in range(2)]
            and all(row.get("split") == "train" for row in slot_rows), "coverage_attempt_order_invalid")
    require([(row.get("content_sha256"), row.get("second_index")) for row in slot_rows]
            == [(row["content_sha256"], index) for row in plan["records"] for index in row["priority_second_indices"]]
            and attempts.get("teacher_plan_sha256") == plan["plan_sha256"], "coverage_exact_window_slots_changed")
    gate = validate_role_metadata(config, requested_contents=contents, allowed_splits={"train"})
    manifest_path, registry_path = Path(config["source_manifest"]), Path(config["exposure_registry"])
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    trial = "teacher-" + canonical_sha256({"plan": plan["plan_sha256"], "release": release["reference"]})[:24]
    require(not any(event["trial_id"] == trial for event in read_registry(registry_path, records)["events"]),
            "coverage_release_already_reserved")
    registration = register_content_manifest(worklist_path, registry_path, manifest_path,
        Path(config["automated_split"]), expected_assignments_sha256=config["expected_assignments_sha256"],
        trial_id=trial, purpose="bounded_teacher_attempt", allowed_splits={"train"})
    receipt = {"schema_version": "forte.teacher-exposure-receipt.v1", "registration_origin": "authoritative_central_exposure_registry",
        "ok": True, "plan_sha256": plan["plan_sha256"], "assignments_sha256": plan["assignments_sha256"],
        "split_file_sha256": plan["split_file_sha256"], "source_manifest_sha256": file_sha256(manifest_path),
        "execution_release_reference": release["reference"], "execution_release_sha256": canonical_sha256(release),
        "execution_release_file_sha256": file_sha256(release_path), "registered_content_sha256": sorted(contents),
        "registered_content_set_sha256": canonical_sha256(sorted(contents)), "allowed_splits": ["train"],
        "registry_tip_sha256": registration["registry_tip_sha256"], "trial_id": trial,
        "maximum_attempts": 34, "attempts": slot_rows, "native_pts_ledger_sha256": file_sha256(native_path),
        "role_exposure_gate": gate, "registration": registration,
        "all_reserved_attempts_consume_slots_including_failures": True, "model_calls_made": False}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return {"ok": True, "registered_train_contents": 17, "reserved_teacher_attempts": 34,
            "receipt_sha256": file_sha256(output_path), "model_calls_made": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("role-config", "plan", "attempt-plan", "worklist", "release", "native-ledger", "handoff", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--inventory", type=Path, help="fresh actual accepted-job snapshot; source/split pins remain unchanged")
    parser.add_argument("--exposure-status", type=Path)
    args = parser.parse_args()
    try:
        config = load_json(args.role_config)
        if args.inventory:
            config["source_inventory_path_override"] = str(args.inventory.resolve())
        if args.exposure_status:
            config["exposure_inventory_status"] = str(args.exposure_status.resolve())
        result = reserve(config, plan_path=args.plan, attempt_path=args.attempt_plan,
            worklist_path=args.worklist, release_path=args.release, native_path=args.native_ledger,
            handoff_path=args.handoff, output_path=args.output)
        print(json.dumps(result))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc), "model_calls_made": False}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "coverage_local_reservation_failed",
                          "error_type": type(exc).__name__, "model_calls_made": False}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
