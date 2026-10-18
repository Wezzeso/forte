"""All record-level operations stay in local files. stdout is aggregate only."""

from __future__ import annotations

import argparse
import json
import sys

from .audit import audit_split
from .io import read_json, read_records, write_json
from .metrics import evaluate
from .selection import select_candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-splits")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--output", required=True)
    run = commands.add_parser("evaluate")
    for name in ("truth", "predictions", "output"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--metadata")
    run.add_argument("--splits")
    run.add_argument("--split", default="validation", choices=("train", "validation", "final_check"))
    run.add_argument("--rules", help="JSON with verified_rule_ids array; only independently verified applicable IDs")
    run.add_argument("--tolerance", type=float, action="append", help="Internal diagnostic seconds; may repeat")
    run.add_argument("--deadline-sec", type=float, default=300.0)
    select = commands.add_parser("select")
    for name in ("baseline", "candidate", "output"):
        select.add_argument("--" + name, required=True)
    select.add_argument("--minimum-reviewed", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-splits":
            report = audit_split(read_json(args.manifest))
            success = report["training_gate_passed"]
        elif args.command == "evaluate":
            rules = read_json(args.rules) if args.rules else None
            rule_ids = set(rules["verified_rule_ids"]) if rules is not None else None
            report = evaluate(read_records(args.truth), read_records(args.predictions), metadata=read_json(args.metadata) if args.metadata else None, split_manifest=read_json(args.splits) if args.splits else None, split=args.split, tolerances_sec=tuple(args.tolerance or (0.5, 1.0, 2.0)), deadline_sec=args.deadline_sec, verified_rule_ids=rule_ids)
            success = report["ground_truth"]["required_samples"] > 0
        else:
            if args.minimum_reviewed < 1:
                raise ValueError("minimum-reviewed must be positive")
            report = select_candidate(read_json(args.baseline), read_json(args.candidate), minimum_reviewed=args.minimum_reviewed)
            success = report["candidate_earned_selection"]
        write_json(args.output, report)
        if args.command == "evaluate":
            summary = {"schema_version": report["schema_version"], "claim_scope": report["claim_scope"], "required": report["completion"]["required"], "completed": report["completion"]["completed"], "human_reviewed_resolved": report["ground_truth"]["human_reviewed_resolved"]}
        elif args.command == "select":
            summary = {"selected": report["selected"], "candidate_earned_selection": success, "blocker_count": len(report["blockers"])}
        else:
            summary = {"integrity_passed": report["integrity_passed"], "locked": report["locked"], "training_gate_passed": success, "error_count": sum(report["error_counts"].values())}
        print(json.dumps(summary, sort_keys=True))
        return 0 if success else 2
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        # Exception strings can contain source paths/record contents; don't echo.
        print(json.dumps({"error": "evaluation_input_or_report_invalid", "details": "Check local files against evaluation/README.md; no private record values were printed."}), file=sys.stderr)
        return 1
