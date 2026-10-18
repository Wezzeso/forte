"""Transactional, local-only review records with explicit negative coverage.

The database and exports contain private data. Callers must never log record
bodies, exception values, video names, reviewer notes, or model proposals.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import SCHEMA_VERSION
from .catalog import event_definitions, fact_definitions

LOCKED_SPLITS = {"train", "validation", "final_check", "test"}


class ReviewError(ValueError):
    """A deliberately generic error safe for aggregate terminal diagnostics."""


class RevisionConflict(ReviewError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def read_manifest(path: Path) -> Iterable[dict[str, Any]]:
    """Read JSON/JSONL locally; never echo private rows to a tool result."""
    with path.open(encoding="utf-8-sig") as handle:
        if path.suffix.lower() == ".jsonl":
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ReviewError("manifest rows must be objects")
                    yield value
        else:
            payload = json.load(handle)
            records = payload if isinstance(payload, list) else payload.get("records", payload.get("recordings", []))
            if not isinstance(records, list):
                raise ReviewError("manifest requires a record list")
            yield from records


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReviewError("timestamps must be finite numeric seconds")
    return float(value)


def _interval(row: dict[str, Any], duration: float) -> tuple[float, float]:
    start, end = _number(row.get("start_sec")), _number(row.get("end_sec"))
    if not 0 <= start < end <= duration:
        raise ReviewError("interval lies outside source timeline or is empty")
    return start, end


def _covers_video(windows: list[dict[str, Any]], duration: float) -> bool:
    cursor = 0.0
    for start, end in sorted((float(row["start_sec"]), float(row["end_sec"])) for row in windows):
        if start > cursor + 1e-6:
            return False
        cursor = max(cursor, end)
    return cursor >= duration - 1e-6


def validate_annotation(record: dict[str, Any]) -> None:
    """Validate reviewed claims against existing frames and A01 fact contracts.

    Structural validation cannot establish what happened in a private video.
    That judgment is explicitly attributed to the local human reviewer.
    """
    try:
        from forte.shared.schemas import FactDecision, FrameRef
    except ImportError:
        raise ReviewError("shared contracts unavailable; install the project before review") from None

    if record.get("review_state") not in {"pending", "in_progress", "reviewed", "needs_adjudication"}:
        raise ReviewError("unknown review state")
    if record.get("review_state") != "reviewed":
        return
    if record.get("review_origin") != "human" or not str(record.get("reviewer_id", "")).strip():
        raise ReviewError("review requires an identified local human reviewer")
    if record.get("source_role") == "jury_test":
        raise ReviewError("jury test media cannot become training annotations")
    duration = _number(record.get("duration_sec"))
    if duration <= 0:
        raise ReviewError("review requires a probed positive duration")
    frame_map: dict[str, Any] = {}
    for raw in record.get("frames", []):
        try:
            frame = FrameRef.model_validate(raw)
        except Exception:
            raise ReviewError("invalid frame reference") from None
        if frame.frame_id in frame_map or not 0 <= frame.pts_sec <= duration:
            raise ReviewError("duplicate frame or frame outside timeline")
        if not Path(frame.path).is_file():
            raise ReviewError("reviewed evidence frame is not available locally")
        frame_map[frame.frame_id] = frame
    taxonomy_facts, taxonomy_events = fact_definitions(), event_definitions()
    event_ids: set[str] = set()
    for event in record.get("events", []):
        start, end = _interval(event, duration)
        anchor = _number(event.get("anchor_sec"))
        if not start <= anchor <= end:
            raise ReviewError("event anchor outside interval")
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in event_ids:
            raise ReviewError("event IDs must be nonempty and unique")
        event_ids.add(event_id)
        definition = taxonomy_events.get(event.get("event_type_id"))
        if definition is None:
            raise ReviewError("event type is not in the frozen taxonomy")
        evidence_ids = event.get("evidence_frame_ids", [])
        if not 1 <= len(evidence_ids) <= 3 or len(set(evidence_ids)) != len(evidence_ids):
            raise ReviewError("events require one to three distinct source frame IDs")
        if set(evidence_ids) - frame_map.keys():
            raise ReviewError("event references unknown source frame IDs")
        if not any(abs(frame_map[fid].pts_sec - anchor) < 1e-6 for fid in evidence_ids):
            raise ReviewError("anchor must match a source evidence frame timestamp")
        fact_states: dict[str, str] = {}
        for raw in event.get("facts", []):
            try:
                fact = FactDecision.model_validate(raw)
            except Exception:
                raise ReviewError("invalid shared FactDecision") from None
            if fact.fact_id not in taxonomy_facts or fact.fact_id in fact_states:
                raise ReviewError("unknown or duplicate fact ID")
            if set(fact.supporting_frame_ids) - frame_map.keys():
                raise ReviewError("fact references unknown source frame IDs")
            if fact.state in {"present", "absent"} and not fact.supporting_frame_ids:
                raise ReviewError("reviewed clear facts require source evidence")
            fact_states[fact.fact_id] = fact.state
        if any(fact_states.get(fid) != state for fid, state in definition["required_fact_states"].items()):
            raise ReviewError("event does not support the taxonomy-required facts")
        if event.get("rule_grounding_status") not in {"unresolved", "source_verified", "verified_applicable"}:
            raise ReviewError("explicit rule grounding status is required")
        if event.get("rule_grounding_status") == "verified_applicable" and not event.get("rule_id"):
            raise ReviewError("verified grounding requires a catalog rule ID")
        if event.get("rule_grounding_status") == "verified_applicable":
            try:
                from rules.catalog import load_catalog
                catalog = load_catalog()
                rule = catalog.get_rule(event["rule_id"])
                binding = catalog.bindings[rule.rule_id]
            except Exception:
                raise ReviewError("event requires an existing enabled checksum-verified catalog rule") from None
            if binding["event_type_id"] != event["event_type_id"] or any(fact_states.get(fid) != state for fid, state in binding["required_fact_states"].items()):
                raise ReviewError("reviewed event does not support the exact verified rule binding")
            if event.get("temporal_verified") is not True or event.get("exceptions_verified") is not True:
                raise ReviewError("grounded events require explicit temporal and exception review")
    negative_windows = record.get("negative_windows", [])
    for window in negative_windows:
        start, end = _interval(window, duration)
        if window.get("negative_verified") is not True or window.get("context_complete") is not True or window.get("visibility") != "clear":
            raise ReviewError("negative windows require explicit clear complete-context verification")
        if not str(window.get("criteria_version", "")).strip():
            raise ReviewError("negative window requires the reviewed criteria version")
        if any(start < event["end_sec"] and event["start_sec"] < end for event in record.get("events", [])):
            raise ReviewError("negative windows overlap a reviewed event")
    for window in record.get("visibility_regions", []):
        _interval(window, duration)
        if window.get("visibility") not in {"clear", "partial", "unobservable"}:
            raise ReviewError("unknown visibility state")
    verdict = record.get("verdict")
    if verdict not in {None, "violation", "no_violation"}:
        raise ReviewError("invalid reviewed verdict")
    if verdict == "violation" and not any(event.get("rule_grounding_status") == "verified_applicable" for event in record.get("events", [])):
        raise ReviewError("positive legal verdict requires a grounded reviewed event")
    if verdict == "no_violation":
        if record.get("events") or record.get("unresolved_items") or not record.get("full_review_complete") or not _covers_video(negative_windows, duration):
            raise ReviewError("negative whole-video verdict requires complete verified coverage without unresolved items")


EDITABLE_FIELDS = {
    "review_state", "reviewer_id", "review_origin", "full_review_complete", "verdict",
    "events", "negative_windows", "visibility_regions", "unresolved_items", "frames",
    "review_notes", "corrections", "local_model_provenance",
}


class ReviewStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        try:
            self.workspace.chmod(0o700)
        except OSError:
            pass
        self.database_path = self.workspace / "review.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS records (sample_id TEXT PRIMARY KEY, content_hash TEXT UNIQUE NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS audit (audit_id INTEGER PRIMARY KEY, sample_id TEXT NOT NULL, revision INTEGER NOT NULL, changed_at TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, before_payload TEXT, after_payload TEXT NOT NULL)")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database_path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def seed(self, rows: Iterable[dict[str, Any]], media_root: Path) -> dict[str, int]:
        added = linked = skipped = 0
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in rows:
                content_hash = row.get("content_hash", row.get("content_sha256", row.get("sha256")))
                if not isinstance(content_hash, str) or len(content_hash) != 64 or any(c not in "0123456789abcdef" for c in content_hash):
                    raise ReviewError("manifest row lacks a valid stable SHA-256")
                recording_id = str(row.get("recording_id", row.get("sample_id", "")))
                if not recording_id:
                    raise ReviewError("manifest row lacks an opaque recording ID")
                path = row.get("owned_path", row.get("source_path", row.get("absolute_path", row.get("relative_path", row.get("path")))))
                if not isinstance(path, str) or not path:
                    raise ReviewError("manifest row lacks a media reference")
                media_path = Path(path)
                if not media_path.is_absolute():
                    media_path = media_root / media_path
                media_path = media_path.resolve()
                if not media_path.is_file() or not media_path.is_relative_to(media_root.resolve()):
                    raise ReviewError("manifest media must be a file within the supplied media root")
                existing = db.execute("SELECT payload,revision FROM records WHERE content_hash=?", (content_hash,)).fetchone()
                if existing:
                    record = json.loads(existing["payload"])
                    if recording_id in record["recording_ids"]:
                        skipped += 1
                        continue
                    before = existing["payload"]
                    group_id = row.get("scenario_group_id") or row.get("recording_group_id") or row.get("group_id") or content_hash
                    split = row.get("split", "unassigned")
                    if group_id != record["group_id"] or split != record["split"]:
                        raise ReviewError("duplicate content has conflicting group or split")
                    record["recording_ids"].append(recording_id)
                    record["source_records"].append(row)
                    record["media_paths"].append(str(media_path))
                    record["revision"] = existing["revision"] + 1
                    db.execute("UPDATE records SET revision=?,payload=? WHERE content_hash=?", (record["revision"], canonical(record), content_hash))
                    self._audit(db, record, "manifest_import", "link_duplicate", before)
                    linked += 1
                    continue
                sample_id = "sample_" + content_hash[:24]
                source_role = row.get("source_role", row.get("dataset_role", "organizer_development"))
                if source_role not in {"organizer_development", "team_training", "jury_test", "synthetic"}:
                    raise ReviewError("manifest source_role is unknown")
                record = {
                    "schema_version": SCHEMA_VERSION, "sample_id": sample_id,
                    "content_hash": content_hash, "content_sha256": content_hash,
                    "recording_ids": [recording_id], "group_id": row.get("scenario_group_id") or row.get("recording_group_id") or row.get("group_id") or content_hash,
                    "split": row.get("split", "unassigned"), "source_role": source_role,
                    "split_lock_verified": False, "split_manifest_sha256": None,
                    "duration_sec": row.get("duration_sec", row.get("metadata", {}).get("duration_sec")),
                    "media_paths": [str(media_path)], "source_records": [row],
                    "review_state": "pending", "review_status": "pending", "review_origin": None,
                    "reviewer_id": None, "reviewed_at": None, "full_review_complete": False,
                    "verdict": None, "events": [], "negative_windows": [], "frames": [],
                    "visibility_regions": [], "corrections": [], "review_notes": "",
                    "unresolved_items": ["human_local_review_required"], "revision": 1,
                    "created_at": utc_now(), "taxonomy_version": "forte.taxonomy.v1",
                }
                if record["split"] not in LOCKED_SPLITS | {"unassigned"}:
                    raise ReviewError("manifest split is unknown")
                db.execute("INSERT INTO records VALUES(?,?,?,?)", (sample_id, content_hash, 1, canonical(record)))
                self._audit(db, record, "manifest_import", "seed_pending", None)
                added += 1
        return {"unique_added": added, "duplicate_recordings_linked": linked, "already_present": skipped}

    @staticmethod
    def _audit(db: sqlite3.Connection, record: dict[str, Any], actor: str, action: str, before: str | None) -> None:
        db.execute("INSERT INTO audit(sample_id,revision,changed_at,actor,action,before_payload,after_payload) VALUES(?,?,?,?,?,?,?)", (record["sample_id"], record["revision"], utc_now(), actor, action, before, canonical(record)))

    def get(self, sample_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM records WHERE sample_id=?", (sample_id,)).fetchone()
        if not row:
            raise ReviewError("unknown sample ID")
        return json.loads(row["payload"])

    def list_records(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [json.loads(row["payload"]) for row in db.execute("SELECT payload FROM records ORDER BY sample_id")]

    def save(self, sample_id: str, patch: dict[str, Any], *, expected_revision: int) -> dict[str, Any]:
        if set(patch) - EDITABLE_FIELDS:
            raise ReviewError("only review fields may be edited; source identity is immutable")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload,revision FROM records WHERE sample_id=?", (sample_id,)).fetchone()
            if not row:
                raise ReviewError("unknown sample ID")
            if expected_revision != row["revision"]:
                raise RevisionConflict("record changed; reload before saving")
            record = json.loads(row["payload"])
            record.update(patch)
            validate_annotation(record)
            record["revision"] += 1
            record["updated_at"] = utc_now()
            record["reviewed_at"] = utc_now() if record["review_state"] == "reviewed" else None
            record["review_status"] = "human_reviewed" if record["review_state"] == "reviewed" else record["review_state"]
            db.execute("UPDATE records SET payload=?,revision=? WHERE sample_id=?", (canonical(record), record["revision"], sample_id))
            self._audit(db, record, str(record.get("reviewer_id") or "local_reviewer"), "review_update", row["payload"])
        return record

    def append_local_proposal(self, sample_id: str, proposal: dict[str, Any], *, expected_revision: int) -> dict[str, int]:
        """Append validated model output without replacing human annotation.

        Use annotations.proposals.import_proposal for file/path/schema checks.
        This transaction independently protects content identity, review state,
        prior human edits and concurrent browser changes.
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload,revision FROM records WHERE sample_id=?", (sample_id,)).fetchone()
            if not row:
                raise ReviewError("unknown proposal review unit")
            record = json.loads(row["payload"])
            if proposal.get("schema_version") != "forte.local-proposal.v1" or proposal.get("reviewed") is not False or proposal.get("asset", {}).get("content_hash") != record["content_hash"]:
                raise ReviewError("proposal is not bound to this unreviewed content identity")
            if any(existing["audit_sha256"] == proposal["audit_sha256"] for existing in record.get("local_model_proposals", [])):
                return {"proposals_imported": 0, "already_imported": 1}
            if expected_revision != row["revision"]:
                raise RevisionConflict("record changed; reload before importing proposal")
            if record["source_role"] == "jury_test" or record["review_state"] not in {"pending", "in_progress"} or record.get("review_origin") not in {None, "local_model"} or record.get("reviewer_id"):
                raise ReviewError("proposal import cannot replace a human review or jury test record")
            if any(record.get(key) for key in ("events", "negative_windows", "frames", "corrections", "review_notes")):
                raise ReviewError("proposal import cannot alter existing human annotation fields")
            entry = {**proposal, "proposal_id": "proposal_" + proposal["audit_sha256"][:24], "imported_at": utc_now()}
            record.setdefault("local_model_proposals", []).append(entry)
            record.update(review_state="in_progress", review_status="in_progress", review_origin="local_model", reviewed_at=None, reviewer_id=None)
            record["local_model_provenance"] = proposal["provenance"]
            if "local_model_proposal_requires_human_review" not in record["unresolved_items"]:
                record["unresolved_items"].append("local_model_proposal_requires_human_review")
            if proposal["provenance"].get("actual_snapshot_verified") is not True and "local_model_runtime_revision_unverified" not in record["unresolved_items"]:
                record["unresolved_items"].append("local_model_runtime_revision_unverified")
            record["revision"] += 1
            record["updated_at"] = utc_now()
            db.execute("UPDATE records SET payload=?,revision=? WHERE sample_id=?", (canonical(record), record["revision"], sample_id))
            self._audit(db, record, "local_model", "import_unreviewed_proposal", row["payload"])
        return {"proposals_imported": 1, "already_imported": 0}

    def publish_machine_summary(self, sample_id: str, summary: dict[str, Any], *, expected_revision: int) -> dict[str, int]:
        """Update a separate read-only machine track, preserving human fields."""
        if summary.get("schema_version") != "forte.machine-annotations.v1" or summary.get("human_reviewed") is not False or summary.get("training_authorized") is not False or summary.get("review_status") != "machine_consistency_checked":
            raise ReviewError("invalid_machine_summary_authority")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload,revision FROM records WHERE sample_id=?", (sample_id,)).fetchone()
            if not row:
                raise ReviewError("unknown_machine_summary_source")
            record = json.loads(row["payload"])
            if summary.get("content_sha256") != record["content_hash"] or summary.get("sample_id") != sample_id or record["source_role"] == "jury_test":
                raise ReviewError("machine_summary_source_binding_invalid")
            if record.get("machine_first_pass", {}).get("summary_sha256") == summary.get("summary_sha256"):
                return {"machine_summaries_updated": 0, "machine_summaries_unchanged": 1}
            if row["revision"] != expected_revision:
                raise RevisionConflict("machine_summary_revision_conflict")
            record["machine_first_pass"] = summary
            record["revision"] += 1
            record["updated_at"] = utc_now()
            db.execute("UPDATE records SET payload=?,revision=? WHERE sample_id=?", (canonical(record), record["revision"], sample_id))
            self._audit(db, record, "local_machine_automation", "machine_first_pass_summary", row["payload"])
        return {"machine_summaries_updated": 1, "machine_summaries_unchanged": 0}

    def aggregate(self) -> dict[str, Any]:
        records = self.list_records()
        states = Counter(record["review_state"] for record in records)
        return {
            "unique_clips": len(records), "recordings": sum(len(row["recording_ids"]) for row in records),
            "review_states": dict(sorted(states.items())),
            "reviewed_with_unresolved_items": sum(row["review_state"] == "reviewed" and bool(row["unresolved_items"]) for row in records),
            "export_ready_with_locked_split": sum(row["review_state"] == "reviewed" and row["split"] in LOCKED_SPLITS and row.get("split_lock_verified") and row["source_role"] != "jury_test" for row in records),
            "review_complete": bool(records) and all(row["review_state"] == "reviewed" for row in records),
            "unique_clips_with_local_proposals": sum(bool(row.get("local_model_proposals")) for row in records),
            "local_proposal_count": sum(len(row.get("local_model_proposals", [])) for row in records),
            "machine_first_pass_count": sum(bool(row.get("machine_first_pass")) for row in records),
        }

    def sync_splits(self, manifest: Path) -> dict[str, int]:
        """Read A02/A10's approved immutable split; never infer a lock."""
        raw = manifest.read_bytes()
        locked = json.loads(raw.decode("utf-8-sig"))
        review_flags = [locked[key] for key in ("scenario_review_complete", "scenario_groups_reviewed") if key in locked]
        if locked.get("locked") is not True or not review_flags or any(flag is not True for flag in review_flags) or not locked.get("approved_by"):
            raise ReviewError("split requires explicit approval, scenario review completion, and a lock")
        desired: dict[str, tuple[str, str]] = {}
        for row in locked.get("records", []):
            content_hash = row.get("content_hash", row.get("content_sha256"))
            group_id = row.get("scenario_group_id") or row.get("group_id")
            split = row.get("split")
            if not content_hash or not group_id or split not in LOCKED_SPLITS:
                raise ReviewError("locked split row is incomplete")
            if content_hash in desired and desired[content_hash] != (group_id, split):
                raise ReviewError("duplicate content crosses locked groups or splits")
            desired[content_hash] = (group_id, split)
        groups: dict[str, set[str]] = {}
        for group_id, split in desired.values():
            groups.setdefault(group_id, set()).add(split)
        if any(len(values) != 1 for values in groups.values()):
            raise ReviewError("scenario group crosses splits")
        digest = hashlib.sha256(raw).hexdigest()
        updated = 0
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT payload FROM records").fetchall()
            for item in rows:
                record = json.loads(item["payload"])
                if record["content_hash"] not in desired:
                    raise ReviewError("locked split does not cover every seeded unique recording")
                group_id, split = desired[record["content_hash"]]
                if split == "train" and locked.get("training_authorized") is not True:
                    raise ReviewError("training split is not explicitly authorized")
                record.update(group_id=group_id, split=split, split_lock_verified=True, split_manifest_sha256=digest)
                record["revision"] += 1
                db.execute("UPDATE records SET payload=?,revision=? WHERE sample_id=?", (canonical(record), record["revision"], record["sample_id"]))
                self._audit(db, record, str(locked["approved_by"]), "approved_split_sync", item["payload"])
                updated += 1
        return {"unique_records_assigned_to_locked_split": updated}

    def export_reviewed(self, output: Path, split: str | None = None) -> dict[str, int]:
        records = self.list_records()
        export = []
        for record in records:
            if record["review_state"] != "reviewed" or record["source_role"] == "jury_test" or record["split"] == "unassigned" or not record.get("split_lock_verified"):
                continue
            if split and record["split"] != split:
                continue
            validate_annotation(record)
            row = {key: value for key, value in record.items() if key not in {"source_records", "media_paths", "review_notes", "local_model_provenance", "local_model_proposals", "machine_first_pass"}}
            row["negative_verified"] = row.get("verdict") == "no_violation" and row.get("full_review_complete") is True
            export.append(row)
        output = output.resolve()
        if not output.is_relative_to(self.workspace):
            raise ReviewError("private annotation exports must remain inside the private workspace")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in export:
                handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        return {"exported_reviewed_records": len(export), "excluded_records": len(records) - len(export)}
