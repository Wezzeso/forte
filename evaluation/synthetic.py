"""Generate tiny public synthetic fixtures, never examples of supplied media."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .audit import records_digest
from .io import write_json
from .metrics import evaluate


def fixtures() -> tuple[list[dict], list[dict], dict, dict]:
    review = {"review_status": "human_reviewed", "reviewer_id": "synthetic_fixture_author", "reviewed_at": "2026-10-17T00:00:00Z"}

    def event(kind: str, time: float, rule: str = "synthetic_rule_a") -> dict:
        return {"event_type_id": kind, "anchor_sec": time, "rule_id": rule, "evidence_frame_ids": ["synthetic_frame"], "local_review": {**review, "evidence_supported": True}}

    truth, predictions, records = [], [], []
    for index in range(1, 9):
        sid = f"synthetic_{index:02d}"
        events = [event("synthetic_a", 1.0), event("synthetic_a", 2.0)] if index == 1 else [event("synthetic_a", 3.5)] if index in {2, 5, 6} else []
        truth.append({"sample_id": sid, **review, "verdict": "violation" if events else "no_violation", "events": events, "duration_sec": 10.0, "scenario_tags": ["multiple_events"] if index == 1 else ["window_boundary"] if index == 2 else ["normal_lookalike"] if index == 4 else []})
        records.append({"sample_id": sid, "group_id": f"synthetic_group_{index}", "content_sha256": hashlib.sha256(sid.encode()).hexdigest(), "split": "validation", "has_violation": bool(events)})
        if index == 6:
            continue
        if index in {2, 7}:
            predictions.append({"sample_id": sid, "status": "failed"})
            continue
        predicted_events = [event("synthetic_a", 1.4), event("synthetic_a", 2.4)] if index == 1 else [event("synthetic_a", 2)] if index == 4 else [event("synthetic_b", 3.5)] if index == 5 else []
        predictions.append({"sample_id": sid, "status": "done", "verdict": "violation" if predicted_events else "no_violation", "events": predicted_events, "elapsed_sec": 301 if index == 8 else 12})
    manifest = {"state": "locked", "scenario_groups_reviewed": True, "approved_by": "synthetic_fixture_author", "locked_at": "2026-10-17T00:00:00Z", "source_manifest_sha256": hashlib.sha256(b"synthetic-only").hexdigest(), "records": records, "records_sha256": records_digest(records)}
    metadata = {"model_id": "synthetic_fixture_no_model", "model_revision": "synthetic", "inference_kind": "synthetic", "conditions_sha256": "synthetic", "rules_sha256": "synthetic", "taxonomy_sha256": "synthetic", "final_check_used_for_tuning": False}
    return truth, predictions, manifest, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    truth, predictions, manifest, metadata = fixtures()
    for name, rows in (("truth.jsonl", truth), ("predictions.jsonl", predictions)):
        (output / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    write_json(output / "splits.json", manifest)
    write_json(output / "metadata.json", metadata)
    write_json(output / "verified-rules.json", {"verified_rule_ids": ["synthetic_rule_a"]})
    report = evaluate(truth, predictions, metadata=metadata, split_manifest=manifest, verified_rule_ids={"synthetic_rule_a"})
    write_json(output / "report.json", report)
    print(json.dumps({"fixture_only": True, "samples": len(truth), "completed": report["completion"]["completed"], "failed_or_missing": report["completion"]["unprocessed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
