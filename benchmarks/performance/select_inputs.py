"""Prepare metadata-only proposed-train inputs; keep paths/hashes in private output."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def select(project, manifest_path, split_path, private_output, aggregate_output):
    project = project.resolve()
    private_root = (project / "benchmarks/private").resolve()
    if not private_output.resolve().is_relative_to(private_root):
        raise ValueError("selection must remain in gitignored benchmarks/private")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    canonical = (json.dumps(split["records"], ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != split["assignments_sha256"]:
        raise ValueError("split assignment integrity mismatch")
    if split.get("source_manifest_sha256") != digest(manifest_path):
        raise ValueError("source manifest integrity mismatch")
    # Only whitelist technical identity/metadata fields; never use labels.
    records = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        records[raw["recording_id"]] = {key: raw.get(key) for key in (
            "recording_id", "content_hash", "relative_path", "duration_sec", "width", "height")}
    assignments = [{key: raw.get(key) for key in ("recording_id", "content_hash", "split")}
                   for raw in split["records"]]
    final_hashes = {item["content_hash"] for item in assignments if item["split"] == "final_check"}
    train = {}
    for item in assignments:
        if item["split"] != "train":
            continue
        raw = records[item["recording_id"]]
        if raw["content_hash"] != item["content_hash"] or raw["content_hash"] in final_hashes:
            raise ValueError("cross-split content or hash mismatch")
        duration = raw["duration_sec"]
        if not isinstance(duration, (int, float)) or duration <= 0:
            continue
        train.setdefault(raw["content_hash"], raw)
    ordered = sorted(train.values(), key=lambda item: (item["duration_sec"], item["content_hash"]))
    if len(ordered) < 6:
        raise ValueError("six distinct train contents required")
    baseline, remaining = ordered[0], ordered[1:]
    # Include the proposed-train maximum and span duration quantiles. The
    # baseline content is excluded so that all five burst contents remain novel.
    indices = [round(index * (len(remaining) - 1) / 4) for index in range(5)]
    selected = [baseline] + [remaining[index] for index in indices]
    items = []
    for index, item in enumerate(selected):
        path = (project / item["relative_path"]).resolve(strict=True)
        if not path.is_relative_to((project / "videos").resolve()) or not path.is_file():
            raise ValueError("source path outside preserved local videos")
        if digest(path) != item["content_hash"]:
            raise ValueError("selected source bytes changed")
        items.append({"sample_id": "baseline" if index == 0 else f"burst{index:02d}",
                      "path": str(path), "content_hash": item["content_hash"],
                      "recording_id": item["recording_id"], "proposed_split": "train",
                      "duration_sec": item["duration_sec"], "width": item["width"], "height": item["height"]})
    private = {"schema_version": "1.0", "selection_scope": "proposed_train_benchmark_only",
               "training_authorized": False, "split_locked": bool(split.get("locked")),
               "source_manifest_sha256": digest(manifest_path), "source_split_sha256": digest(split_path),
               "baseline": items[0], "burst": items[1:]}
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_text(json.dumps(private, ensure_ascii=False, indent=2), encoding="utf-8")
    durations = [item["duration_sec"] for item in selected]
    public = {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection_scope": "proposed_train_benchmark_only", "training_authorized": False,
              "split_locked": bool(split.get("locked")), "selected_distinct_contents": len(items),
              "baseline_count": 1, "burst_count": 5, "baseline_disjoint_from_burst": True,
              "labels_used_for_selection": False, "media_pixels_or_audio_inspected": False,
              "final_check_media_opened": False, "source_hashes_verified": True,
              "duration_seconds_min": min(durations), "duration_seconds_max": max(durations),
              "resolution_counts": [{"width": width, "height": height,
                                     "count": sum(item["width"] == width and item["height"] == height for item in selected)}
                                    for width, height in sorted({(item["width"], item["height"]) for item in selected})],
              "selection_rule": "shortest train baseline; five disjoint train duration quantiles including train maximum",
              "quality_review": "unmeasured; benchmark selection is not reviewed ground truth"}
    aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    aggregate_output.write_text(json.dumps(public, indent=2), encoding="utf-8")
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/candidate_split.json"))
    parser.add_argument("--private-output", type=Path, default=Path("benchmarks/private/a18-inputs.json"))
    parser.add_argument("--aggregate-output", type=Path, default=Path("benchmarks/performance/input-selection-aggregate.json"))
    args = parser.parse_args()
    try:
        report = select(args.project, args.manifest, args.split, args.private_output, args.aggregate_output)
        print(json.dumps({"selected_distinct_contents": report["selected_distinct_contents"],
                          "baseline_count": report["baseline_count"], "burst_count": report["burst_count"],
                          "final_check_media_opened": report["final_check_media_opened"]}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
