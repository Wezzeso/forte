"""Run ``python -m annotations.cli --help`` from the repository root."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .server import make_server
from .store import ReviewError, ReviewStore, read_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Private local review. Terminal output is aggregate only.")
    parser.add_argument("--workspace", type=Path, default=Path("annotations/private"))
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser("seed", help="Create pending records from A02 metadata; does not review videos")
    seed.add_argument("--manifest", type=Path, required=True)
    seed.add_argument("--media-root", type=Path, default=Path.cwd())
    seed.add_argument("--review-queue", type=Path, help="Optional A02 content-level unresolved review tasks")
    seed.add_argument("--workbook-rows", type=Path, help="Optional immutable original weak labels; copied locally only")
    commands.add_parser("status", help="Show aggregate review status only")
    serve = commands.add_parser("serve", help="Start the private loopback workbench")
    serve.add_argument("--port", type=int, default=8765)
    export = commands.add_parser("export", help="Export human-reviewed records with locked splits only")
    export.add_argument("--output", type=Path, default=Path("reviewed.jsonl"))
    export.add_argument("--split", choices=["train", "validation", "final_check", "test"])
    locked = commands.add_parser("sync-splits", help="Import an explicitly approved, scenario-reviewed A02 split lock")
    locked.add_argument("--manifest", type=Path, required=True)
    apply = commands.add_parser("apply", help="Apply a local private JSON review patch without printing its content")
    apply.add_argument("--input", type=Path, required=True)
    proposal = commands.add_parser("import-proposal", help="Import a private team-local model audit as an unreviewed proposal only")
    proposal.add_argument("--input", type=Path, required=True)
    proposal.add_argument("--proposal-root", type=Path, action="append", required=True, help="Explicit allowed local audit root; repeatable")
    proposal.add_argument("--source-root", type=Path, action="append", required=True, help="Explicit allowed local source media root; repeatable")
    proposal.add_argument("--frame-root", type=Path, action="append", required=True, help="Explicit allowed local decoded frame root; repeatable")
    proposal.add_argument("--sample-id", help="Optional opaque target ID; must match source SHA-256")
    proposal.add_argument("--expected-revision", type=int, help="Optional optimistic revision guard")
    proposal.add_argument("--source-map", type=Path, help="Private JSON map from content SHA-256 to absolute local input path")
    proposal.add_argument("--path-map", type=Path, help="Private JSON {remote_frame_root, local_frame_root} for unchanged downloaded JPEGs")
    fetch = commands.add_parser("fetch-proposal", help="After terminal handoff, privately download/hash-check/rebase/import team-owned audit and JPEGs")
    fetch.add_argument("--handoff", type=Path, required=True)
    fetch.add_argument("--source-root", type=Path, action="append", help="Optional narrower source roots; defaults to already seeded input parents")
    automatic = commands.add_parser("auto-first-pass", help="Derive an explicitly machine-generated track from existing local model proposals; no inference")
    automatic.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    automatic.add_argument("--publish-workbench", action="store_true", help="Update separate machine summaries with revision fencing; preserve all human fields")
    automatic.add_argument("--split-manifest", type=Path, help="Pin an A02 frozen split for a train/validation-only autonomous worklist")
    automatic.add_argument("--provided-anchor-manifest", type=Path, action="append", help="Optional private train/validation coarse anchors; separate focus/evaluation metadata only, never final-check")
    automatic.add_argument("--export-track", help="Separate private version directory, e.g. w2; preserves an already consumed default export")
    automatic.add_argument("--proposal-index", type=Path, help="Restrict a versioned export to the exact SHA-bound terminal handoff set")
    proxy = commands.add_parser("proxy", help="Create optional silent local browser proxies; originals stay unchanged")
    proxy.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = ReviewStore(args.workspace)
        if args.command == "seed":
            queue = {row["content_hash"]: row for row in read_manifest(args.review_queue)} if args.review_queue else {}
            weak_labels: dict[str, list[dict]] = {}
            if args.workbook_rows:
                for row in read_manifest(args.workbook_rows):
                    weak_labels.setdefault(row.get("recording_id", ""), []).append(row)
            def enriched_rows():
                for row in read_manifest(args.manifest):
                    if row.get("content_hash") in queue:
                        row["original_review_task"] = queue[row["content_hash"]]
                    if row.get("recording_id") in weak_labels:
                        row["original_weak_labels"] = weak_labels[row["recording_id"]]
                    yield row
            result = store.seed(enriched_rows(), args.media_root.resolve())
        elif args.command == "status":
            result = store.aggregate()
        elif args.command == "export":
            destination = args.output if args.output.is_absolute() else store.workspace / args.output
            result = store.export_reviewed(destination, args.split)
        elif args.command == "sync-splits":
            result = store.sync_splits(args.manifest)
        elif args.command == "apply":
            payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
            store.save(payload["sample_id"], payload["patch"], expected_revision=payload["expected_revision"])
            result = {"review_updates_saved": 1}
        elif args.command == "import-proposal":
            from .proposals import import_proposal
            mapping = {}
            if args.source_map is not None or args.path_map is not None:
                if args.source_map is None or args.path_map is None:
                    raise ReviewError("source-map and path-map must be supplied together")
                sources = json.loads(args.source_map.read_text(encoding="utf-8-sig"))
                paths = json.loads(args.path_map.read_text(encoding="utf-8-sig"))
                if not isinstance(sources, dict) or not sources or not isinstance(paths, dict) or set(paths) != {"remote_frame_root", "local_frame_root"}:
                    raise ReviewError("invalid explicit private source/path mapping objects")
                if any(not isinstance(key, str) or len(key) != 64 or not isinstance(value, str) for key, value in sources.items()):
                    raise ReviewError("source-map must map stable SHA-256 strings to local path strings")
                mapping = {"local_source_by_hash": sources, "remote_frame_root": paths["remote_frame_root"], "local_frame_root": paths["local_frame_root"]}
            result = import_proposal(store, args.input.resolve(), proposal_roots=args.proposal_root, source_roots=args.source_root, frame_roots=args.frame_root, sample_id=args.sample_id, expected_revision=args.expected_revision, **mapping)
        elif args.command == "fetch-proposal":
            from .fetch import fetch_proposal
            result = fetch_proposal(store, args.handoff.resolve(), source_roots=args.source_root)
        elif args.command == "auto-first-pass":
            from .machine import automatic_first_pass
            result = automatic_first_pass(store, args.manifest.resolve(), publish_workbench=args.publish_workbench, split_manifest=args.split_manifest.resolve() if args.split_manifest else None, provided_anchor_manifests=args.provided_anchor_manifest, export_track=args.export_track, proposal_index=args.proposal_index)
        elif args.command == "proxy":
            if shutil.which(args.ffmpeg) is None:
                raise ReviewError("ffmpeg is unavailable; original video can still be opened in a local player")
            directory = store.workspace / "proxies"
            directory.mkdir(exist_ok=True)
            created = skipped = failed = 0
            for record in store.list_records():
                target = directory / (record["sample_id"] + ".mp4")
                if target.exists():
                    skipped += 1
                    continue
                temporary = target.with_name(target.stem + ".tmp.mp4")
                process = subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-i", record["media_paths"][0], "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-movflags", "+faststart", "-y", str(temporary)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if process.returncode == 0:
                    temporary.replace(target)
                    created += 1
                else:
                    temporary.unlink(missing_ok=True)
                    failed += 1
            result = {"proxies_created": created, "already_present": skipped, "failed": failed, "note": "Proxies are navigation aids; register original decoded source PTS for annotation evidence."}
        else:
            with make_server(store, args.port) as server:
                print(f"Local human review: http://127.0.0.1:{server.server_port} (private; no cloud calls)", flush=True)
                server.serve_forever()
            return 0
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except ReviewError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    except Exception:
        print(json.dumps({"error": "local_operation_failed; inspect privately without sharing record contents"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

