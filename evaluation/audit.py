"""Audit exact duplicates, scenario groups and derived-window inheritance."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .io import digest

SPLITS = {"train", "validation", "final_check"}


def sample_id(row: dict) -> str:
    value = row.get("sample_id", row.get("recording_id"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("record has no nonempty sample_id")
    return value


def records_digest(records: list[dict]) -> str:
    """Canonical SHA256: UTF-8, sorted keys, compact JSON, records sorted by ID."""
    return digest(sorted(records, key=lambda row: sample_id(row)))


def audit_split(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("records", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("split records must be an array of objects")
    errors: Counter[str] = Counter()
    identities: dict[str, dict[str, set[str]]] = {name: defaultdict(set) for name in ("sample", "content", "group", "scenario")}
    labels: dict[str, set[bool]] = defaultdict(set)
    sample_splits: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        sid = sample_id(row)
        split = row.get("split")
        if split not in SPLITS:
            errors["invalid_split"] += 1
            continue
        counts[split] += 1
        if sid in sample_splits:
            errors["duplicate_sample_record"] += 1
        sample_splits[sid] = split
        identities["sample"][sid].add(split)
        content = row.get("content_sha256", row.get("content_hash"))
        group = row.get("group_id")
        if not isinstance(content, str) or len(content) != 64 or any(c not in "0123456789abcdef" for c in content.lower()):
            errors["missing_or_invalid_content_hash"] += 1
        else:
            content = content.lower()
            identities["content"][content].add(split)
            if isinstance(row.get("has_violation"), bool):
                labels[content].add(row["has_violation"])
        if not isinstance(group, str) or not group.strip():
            errors["missing_group_id"] += 1
        else:
            identities["group"][group].add(split)
        scenario = row.get("scenario_group_id")
        if scenario:
            identities["scenario"][str(scenario)].add(split)
    for kind, values in identities.items():
        errors[kind + "_leakage"] = sum(len(splits) > 1 for splits in values.values())
    errors["conflicting_duplicate_labels"] = sum(len(values) > 1 for values in labels.values())
    for row in rows:
        parent = row.get("parent_sample_id", row.get("source_recording_id"))
        if parent:
            if parent not in sample_splits:
                errors["missing_parent_sample"] += 1
            elif row.get("split") != sample_splits[parent]:
                errors["derived_split_leakage"] += 1
    fingerprint = records_digest(rows)
    digest_matches = manifest.get("records_sha256") == fingerprint
    if manifest.get("records_sha256") and not digest_matches:
        errors["records_digest_mismatch"] += 1
    locked = (
        manifest.get("state") == "locked"
        and manifest.get("scenario_groups_reviewed") is True
        and bool(manifest.get("approved_by"))
        and bool(manifest.get("locked_at"))
        and bool(manifest.get("source_manifest_sha256"))
        and digest_matches
    )
    findings = {key: value for key, value in sorted(errors.items()) if value}
    return {
        "schema_version": "forte.split_audit.v1",
        "record_count": len(rows),
        "unique_content_count": len(identities["content"]),
        "unique_group_count": len(identities["group"]),
        "counts_by_split": dict(sorted(counts.items())),
        "error_counts": findings,
        "integrity_passed": bool(rows) and not findings,
        "locked": locked,
        "training_gate_passed": bool(rows) and not findings and locked,
        "records_sha256": fingerprint,
        "scenario_relationships_reviewed": manifest.get("scenario_groups_reviewed") is True,
        "note": "Group integrity precedes exact split counts; proposed assignments are not a training lock.",
    }
