#!/usr/bin/env python3
"""Append-only local inventory of model-attempted/reserved source contents.

Only technical identities are stored; no labels, predictions, media, or
descriptions are read. CLI output is aggregate JSON. Reserving an attempt
before submission protects failed/uncertain requests conservatively too.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator

SCHEMA_VERSION = "forte.model-exposure.v1"
DEFAULT_REGISTRY = Path("benchmarks/private/model-exposure.jsonl")
ZERO_HASH = "0" * 64


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def manifest_identity(records: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    by_id: dict[str, str] = {}
    by_hash: dict[str, list[str]] = {}
    for record in records:
        recording_id, content_hash = record.get("recording_id"), record.get("content_hash")
        if not isinstance(recording_id, str) or not recording_id or recording_id in by_id or not valid_hash(content_hash):
            raise ValueError("invalid_exposure_source_manifest")
        if record.get("content_sha256", content_hash) != content_hash:
            raise ValueError("manifest_content_alias_mismatch")
        by_id[recording_id] = content_hash
        by_hash.setdefault(content_hash, []).append(recording_id)
    return by_id, {key: sorted(values) for key, values in by_hash.items()}


def read_registry(path: Path, records: list[dict]) -> dict:
    _, by_hash = manifest_identity(records)
    if not path.is_file():
        raise ValueError("exposure_inventory_missing")
    events = []
    seen_ids = set()
    previous = ZERO_HASH
    hashes = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                raise ValueError("exposure_registry_blank_record")
            event = json.loads(line)
            signed = {key: value for key, value in event.items() if key != "event_sha256"}
            expected = hashlib.sha256(canonical(signed)).hexdigest()
            content_hash = event.get("content_hash")
            if event.get("schema_version") != SCHEMA_VERSION or event.get("previous_event_sha256") != previous or event.get("event_sha256") != expected:
                raise ValueError("exposure_registry_chain_invalid")
            if not isinstance(event.get("event_id"), str) or event["event_id"] in seen_ids:
                raise ValueError("exposure_registry_event_conflict")
            if content_hash not in by_hash or event.get("recording_ids") != by_hash[content_hash]:
                raise ValueError("exposure_registry_unknown_content")
            hashes.add(content_hash)
            previous = expected
            seen_ids.add(event["event_id"])
            events.append(event)
    return {"events": events, "content_hashes": hashes, "event_count": len(events), "tip_sha256": previous,
            "content_set_sha256": hashlib.sha256(canonical(sorted(hashes))).hexdigest()}


@contextmanager
def registry_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def append_exposures(registry_path: Path, records: list[dict], content_hashes: set[str], *, trial_id: str, phase: str, purpose: str) -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,95}", trial_id) or phase not in {"baseline", "burst", "paired", "all", "local-trial"}:
        raise ValueError("invalid_exposure_attempt_identity")
    if purpose not in {"benchmark_attempt", "benchmark_historical", "local_model_attempt", "feature_extraction_attempt", "bounded_teacher_attempt"}:
        raise ValueError("invalid_exposure_purpose")
    if "private" not in registry_path.resolve().parts:
        raise ValueError("exposure_registry_must_remain_private")
    _, by_hash = manifest_identity(records)
    if not content_hashes or not content_hashes <= set(by_hash):
        raise ValueError("exposure_attempt_unknown_content")
    with registry_lock(registry_path):
        if not registry_path.exists():
            registry_path.touch()
        registry = read_registry(registry_path, records)
        by_event = {event["event_id"]: event for event in registry["events"]}
        previous = registry["tip_sha256"]
        appended = 0
        with registry_path.open("ab") as target:
            for content_hash in sorted(content_hashes):
                identity = {"trial_id": trial_id, "phase": phase, "purpose": purpose, "content_hash": content_hash}
                event_id = "exposure_" + hashlib.sha256(canonical(identity)).hexdigest()[:32]
                if event_id in by_event:
                    continue
                event = {"schema_version": SCHEMA_VERSION, "event_id": event_id, **identity,
                         "recording_ids": by_hash[content_hash], "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "exclusion_state": "exclude_from_final_check", "previous_event_sha256": previous}
                event["event_sha256"] = hashlib.sha256(canonical(event)).hexdigest()
                target.write(canonical(event) + b"\n")
                previous = event["event_sha256"]
                appended += 1
            target.flush()
            os.fsync(target.fileno())
        final = read_registry(registry_path, records)
    return {"ok": True, "events_appended": appended, "exposure_events": final["event_count"],
            "distinct_exposed_contents": len(final["content_hashes"]), "training_authorized": False}


def register_selection(selection_path: Path, registry_path: Path, manifest_path: Path, *, trial_id: str, phase: str, purpose: str = "benchmark_attempt") -> dict:
    """A18 calls before its first request; failure must prevent submission."""
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    if selection.get("selection_scope") != "proposed_train_benchmark_only":
        raise ValueError("unexpected_exposure_selection_scope")
    if selection.get("source_manifest_sha256") != file_digest(manifest_path):
        raise ValueError("exposure_selection_manifest_digest_mismatch")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_id, _ = manifest_identity(records)
    items = [selection["baseline"]] if phase == "baseline" else selection["burst"] if phase == "burst" else [selection["baseline"], *selection["burst"]]
    hashes = set()
    for item in items:
        content_hash = item.get("content_hash")
        if by_id.get(item.get("recording_id")) != content_hash or item.get("proposed_split") != "train":
            raise ValueError("exposure_selection_identity_mismatch")
        hashes.add(content_hash)
    return append_exposures(registry_path, records, hashes, trial_id=trial_id, phase=phase, purpose=purpose)


def accepted_inventory_hashes(inventory: dict) -> set[str]:
    if inventory.get("schema_version") != "forte.service-content-exposure-inventory.v1" or inventory.get("coverage_state") != "complete_snapshot" or inventory.get("scope") != "all_accepted_jobs_since_service_start":
        raise ValueError("accepted_inventory_coverage_incomplete")
    if "records" in inventory:
        rows = inventory["records"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != {"content_hash"} for row in rows) or inventory.get("accepted_job_count") != len(rows):
            raise ValueError("accepted_inventory_records_invalid")
        hashes = [row["content_hash"] for row in rows]
    else:
        # Explicit compatibility with the first unique-hash-only snapshot.
        hashes = inventory.get("content_hashes")
        if isinstance(hashes, list) and len(set(hashes)) != len(hashes):
            raise ValueError("accepted_inventory_hashes_invalid")
    if not isinstance(hashes, list) or any(not valid_hash(value) for value in hashes):
        raise ValueError("accepted_inventory_hashes_invalid")
    if not isinstance(inventory.get("accepted_job_count"), int) or inventory["accepted_job_count"] < len(hashes) or inventory.get("unique_content_count") != len(set(hashes)):
        raise ValueError("accepted_inventory_counts_invalid")
    return set(hashes)


def register_accepted_inventory(inventory_path: Path, registry_path: Path, manifest_path: Path, status_path: Path) -> dict:
    """Backfill manual/other accepted uploads from a private service snapshot."""
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    hashes = accepted_inventory_hashes(inventory)
    if "private" not in inventory_path.resolve().parts or "private" not in status_path.resolve().parts:
        raise ValueError("accepted_inventory_must_remain_private")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    _, by_hash = manifest_identity(records)
    known = hashes & set(by_hash)
    digest = file_digest(inventory_path)
    if known:
        append_exposures(registry_path, records, known, trial_id="service-snapshot-" + digest[:16], phase="local-trial", purpose="local_model_attempt")
    registry = read_registry(registry_path, records)
    status = {"schema_version": "forte.exposure-coverage.v1", "coverage_state": "complete_snapshot",
              "captured_at_utc": inventory["captured_at_utc"], "scope": "all_accepted_jobs_since_service_start",
              "registry_tip_sha256": registry["tip_sha256"], "registry_event_count": registry["event_count"],
              "source_inventory_sha256": digest, "source_inventory_path": str(inventory_path.resolve()),
              "accepted_job_count": inventory["accepted_job_count"], "known_corpus_contents": len(known),
              "active_job_count_at_snapshot": inventory.get("active_job_count"),
              "limitation": "Uploads accepted after captured_at_utc are outside this snapshot; refresh before split approval."}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, status_path)
    return {"ok": True, "accepted_jobs_in_snapshot": inventory["accepted_job_count"], "known_corpus_contents_in_snapshot": len(known),
            "distinct_exposed_contents": len(registry["content_hashes"]), "exposure_events": registry["event_count"],
            "training_authorized": False, "locked": False}


def register_content_manifest(input_path: Path, registry_path: Path, manifest_path: Path, split_path: Path, *,
                              expected_assignments_sha256: str, trial_id: str, purpose: str,
                              allowed_splits: set[str] | None = None) -> dict:
    """Reserve frozen train/validation contents before non-API model calls.

    Root/A17/A06 call locally before launching extraction/teacher work. The
    registry is central and append-only; accepted-job snapshots only add to
    it and cannot erase direct feature/teacher exposures. No source media is
    opened. An input containing even one disallowed role fails as a whole.
    """
    try:
        from .audit_data import check_split, json_bytes
    except ImportError:
        from audit_data import check_split, json_bytes
    allowed_splits = {"train"} if allowed_splits is None else set(allowed_splits)
    if not allowed_splits or not allowed_splits <= {"train", "validation"}:
        raise ValueError("non_api_exposure_roles_invalid")
    if purpose not in {"feature_extraction_attempt", "bounded_teacher_attempt", "local_model_attempt"}:
        raise ValueError("non_api_exposure_purpose_invalid")
    if purpose == "bounded_teacher_attempt" and allowed_splits != {"train"}:
        raise ValueError("bounded_teacher_exposure_must_be_train_only")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_id, by_hash = manifest_identity(records)
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    if not valid_hash(expected_assignments_sha256) or split.get("assignments_sha256") != expected_assignments_sha256 or hashlib.sha256(json_bytes(split.get("records", []))).hexdigest() != expected_assignments_sha256:
        raise ValueError("non_api_exposure_frozen_assignment_mismatch")
    if split.get("source_manifest_sha256") != file_digest(manifest_path):
        raise ValueError("non_api_exposure_source_manifest_mismatch")
    assignments = {}
    scenario_members = {}
    for row in split.get("records", []):
        assignments.setdefault(row["content_hash"], []).append(row)
        scenario_members.setdefault(row["scenario_group_id"], []).append(row["recording_id"])
    scenario = {"groups": [{"scenario_group_id": group_id, "recording_ids": members} for group_id, members in sorted(scenario_members.items())]}
    registry = read_registry(registry_path, records)
    integrity = check_split(split, records, scenario=scenario, excluded_final_hashes=registry["content_hashes"])
    if not integrity["ok"]:
        raise ValueError("non_api_exposure_existing_split_integrity_failed")
    if input_path.suffix.casefold() == ".jsonl":
        inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
        inputs = payload if isinstance(payload, list) else payload.get("records")
        if isinstance(payload, dict):
            if payload.get("assignments_sha256", expected_assignments_sha256) != expected_assignments_sha256:
                raise ValueError("non_api_worklist_assignment_mismatch")
            if payload.get("split_file_sha256", file_digest(split_path)) != file_digest(split_path):
                raise ValueError("non_api_worklist_split_file_mismatch")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("non_api_exposure_inputs_missing")
    selected, split_counts = set(), {}
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError("non_api_exposure_input_invalid")
        content_hash = item.get("content_sha256", item.get("content_hash"))
        if content_hash not in by_hash or item.get("content_hash", content_hash) != content_hash:
            raise ValueError("non_api_exposure_unknown_content")
        content_assignments = assignments.get(content_hash, [])
        roles = {row["split"] for row in content_assignments}
        if len(roles) != 1 or not roles <= allowed_splits:
            raise ValueError("non_api_exposure_input_outside_allowed_roles")
        role = next(iter(roles))
        if item.get("split", role) != role:
            raise ValueError("non_api_exposure_input_role_mismatch")
        if item.get("recording_id") is not None and by_id.get(item["recording_id"]) != content_hash:
            raise ValueError("non_api_exposure_input_recording_mismatch")
        aliases = item.get("recording_ids")
        if aliases is not None and (not isinstance(aliases, list) or not aliases or not set(aliases) <= set(by_hash[content_hash])):
            raise ValueError("non_api_exposure_alias_mismatch")
        if content_hash not in selected:
            split_counts[role] = split_counts.get(role, 0) + 1
        selected.add(content_hash)
    result = append_exposures(registry_path, records, selected, trial_id=trial_id, phase="local-trial", purpose=purpose)
    result.update(registered_input_contents=len(selected), allowed_split_counts=split_counts,
                  assignments_sha256=expected_assignments_sha256, registry_tip_sha256=read_registry(registry_path, records)["tip_sha256"],
                  media_opened=False, model_calls_made=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register-selection")
    register.add_argument("--selection", type=Path, default=Path("benchmarks/private/a18-inputs.json"))
    register.add_argument("--trial-id", required=True)
    register.add_argument("--phase", choices=["baseline", "burst", "paired", "all"], required=True)
    register.add_argument("--purpose", choices=["benchmark_attempt", "benchmark_historical"], default="benchmark_attempt")
    accepted = sub.add_parser("register-accepted-inventory")
    accepted.add_argument("--inventory", type=Path, default=Path("benchmarks/private/service-accepted-content-hashes.json"))
    accepted.add_argument("--status-output", type=Path, default=Path("benchmarks/private/exposure-inventory-status.json"))
    non_api = sub.add_parser("register-content-manifest")
    non_api.add_argument("--input", type=Path, required=True)
    non_api.add_argument("--split", type=Path, default=Path("data/splits/automated_provisional_split.json"))
    non_api.add_argument("--expected-assignments-sha256", required=True)
    non_api.add_argument("--trial-id", required=True)
    non_api.add_argument("--purpose", choices=["feature_extraction_attempt", "bounded_teacher_attempt", "local_model_attempt"], required=True)
    non_api.add_argument("--allowed-split", choices=["train", "validation"], action="append")
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "register-selection":
            result = register_selection(args.selection, args.registry, args.manifest, trial_id=args.trial_id, phase=args.phase, purpose=args.purpose)
        elif args.command == "register-accepted-inventory":
            result = register_accepted_inventory(args.inventory, args.registry, args.manifest, args.status_output)
        elif args.command == "register-content-manifest":
            result = register_content_manifest(args.input, args.registry, args.manifest, args.split,
                                               expected_assignments_sha256=args.expected_assignments_sha256,
                                               trial_id=args.trial_id, purpose=args.purpose,
                                               allowed_splits=set(args.allowed_split) if args.allowed_split else None)
        else:
            records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            registry = read_registry(args.registry, records)
            result = {"ok": True, "exposure_events": registry["event_count"], "distinct_exposed_contents": len(registry["content_hashes"])}
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)) else "exposure_registry_operation_failed"
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error_code": code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
