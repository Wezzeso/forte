"""Append immutable machine exports and compile original TRAIN JPEGs separately.

No teacher inference or source-media decode occurs here. Raw teacher records
and images are never rewritten. Compact aliases are an explicitly versioned
derived training representation, applied after canonical balancing/validation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from typing import Any

from .guards import HEX64, TrainingGateError, canonical_sha256, file_sha256, load_annotations, load_json, local_file, require


def _validate_schema(rows: list[dict], schema_path: Path) -> None:
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(load_json(schema_path))
    require(not any(next(validator.iter_errors(row), None) is not None for row in rows), "silver_combine_machine_schema_invalid")


def combine_exports(config: dict) -> dict:
    inputs = config.get("input_exports", [])
    require(len(inputs) == 2, "silver_combine_requires_old_and_new_exports")
    destination, proof = Path(config["output"]), Path(config["proof_output"])
    require(not destination.exists() and not proof.exists() and destination != proof, "silver_combine_output_exists")
    rows, chunks, seen = [], [], set()
    expected = config["expected_teacher"]
    for item in inputs:
        path = Path(item["path"])
        require(file_sha256(path) == item["sha256"], "silver_combine_input_digest_mismatch")
        raw = path.read_bytes()
        require(raw.endswith(b"\n") and bool(raw.strip()), "silver_combine_input_must_end_newline")
        current = load_annotations(path)
        for row in current:
            content = row.get("content_sha256")
            require(HEX64.fullmatch(str(content)) is not None and content not in seen
                and row.get("split") == "train" and row.get("human_reviewed") is False,
                "silver_combine_duplicate_or_forbidden_content")
            seen.add(content)
            for provenance in [row.get("provenance", {}), *[window.get("provenance", {}) for window in row.get("windows", [])]]:
                require(all(provenance.get(key) == expected.get(key) for key in
                    ("model_id", "model_revision", "processor_revision", "model_manifest_sha256")),
                    "silver_combine_teacher_identity_mismatch")
            require(row.get("source_pts_identity_checked") is True, "silver_combine_source_clock_unverified")
        rows.extend(current)
        chunks.append(raw)
    schema_path = Path(config["machine_schema"])
    require(file_sha256(schema_path) == config["expected_machine_schema_sha256"], "silver_combine_schema_pin_mismatch")
    _validate_schema(rows, schema_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        for chunk in chunks:
            stream.write(chunk)
    result = {"schema_version": "forte.append-only-silver-export.v1", "file_sha256": file_sha256(destination),
        "all_train": True, "human_reviewed": 0, "records": len(rows), "schema_errors": 0,
        "windows": sum(len(row.get("windows", [])) for row in rows),
        "old_export_preserved_as_exact_byte_prefix": destination.read_bytes().startswith(chunks[0]),
        "input_exports": inputs, "teacher_identity": expected,
        "mixed_window_prompt_hashes_preserved": True, "raw_sources_modified": False,
        "whole_video_gold_created": False, "legal_or_boundary_truth_created": False}
    proof.parent.mkdir(parents=True, exist_ok=True)
    with proof.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def resolve_original_frame(path: str, mappings: list[dict], *, content: str | None = None, frame_id: str | None = None) -> Path:
    matches = []
    for mapping in mappings:
        pure = PureWindowsPath if mapping["path_style"] == "windows" else PurePosixPath
        try:
            relative = pure(path).relative_to(pure(mapping["original_prefix"]))
        except ValueError:
            continue
        require(".." not in relative.parts and not relative.is_absolute(), "silver_combine_frame_path_escape")
        if mapping.get("content_addressed") is True:
            require(HEX64.fullmatch(str(content)) is not None and HEX64.fullmatch(str(frame_id)) is not None,
                    "silver_combine_content_addressed_identity_missing")
            relative_path = f"{content}/{frame_id}.jpg"
        else:
            relative_path = str(Path(*relative.parts))
        matches.append(local_file(relative_path, Path(mapping["actual_root"]).resolve()))
    require(len(matches) == 1, "silver_combine_frame_requires_exactly_one_root_mapping")
    return matches[0]


def compile_media(config: dict) -> dict:
    annotations = Path(config["annotations"])
    require(file_sha256(annotations) == config["expected_annotations_sha256"], "silver_combine_annotations_pin_mismatch")
    rows = load_annotations(annotations)
    root, manifest_path = Path(config["media_root"]).resolve(), Path(config["manifest_output"]).resolve()
    require(not root.exists() and not manifest_path.exists() and not manifest_path.is_relative_to(root),
            "silver_combine_media_requires_new_destination")
    mappings = config["source_mappings"]
    require(bool(mappings) and all(item.get("path_style") in {"windows", "posix"} for item in mappings),
            "silver_combine_source_mappings_invalid")
    require(not any(root.is_relative_to(Path(item["actual_root"]).resolve())
                    or Path(item["actual_root"]).resolve().is_relative_to(root) for item in mappings),
            "silver_combine_destination_overlaps_source")
    from forte.video.batches import source_frame_id
    entries, identities = [], set()
    root.mkdir(parents=True)
    for row in rows:
        content = row["content_sha256"]
        require(row.get("split") == "train" and row.get("human_reviewed") is False, "silver_combine_media_role_forbidden")
        require(row.get("partition_binding", {}).get("assignments_sha256") == config["assignments_sha256"],
                "silver_combine_media_split_changed")
        base = Fraction(row["source_time_base"])
        require(base > 0, "silver_combine_time_base_invalid")
        for frame in row["frames"]:
            fid, pts = frame["frame_id"], frame["source_pts"]
            require(type(pts) is int and fid == source_frame_id(content, pts, base.numerator, base.denominator)
                and abs(float(pts * base) - row.get("source_start_sec", 0) - frame["pts_sec"]) <= 1e-6,
                "silver_combine_frame_identity_invalid")
            require((content, fid) not in identities, "silver_combine_frame_duplicate")
            identities.add((content, fid))
            source = resolve_original_frame(frame["path"], mappings, content=content, frame_id=fid)
            require(file_sha256(source) == frame["jpeg_sha256"], "silver_combine_original_jpeg_digest_mismatch")
            relative = f"{content}/{fid}.jpg"
            destination = root / relative
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            require(file_sha256(destination) == frame["jpeg_sha256"], "silver_combine_copy_digest_mismatch")
            entries.append({"content_sha256": content, "frame_id": fid, "source_pts": pts,
                "pts_sec": frame["pts_sec"], "sha256": frame["jpeg_sha256"], "path": relative, "split": "train"})
    manifest = {"schema_version": "forte.qwen38.train-media.v1", "split": "train", "human_verified": False,
        "assignments_sha256": config["assignments_sha256"], "machine_annotations_sha256": config["expected_annotations_sha256"],
        "frames": sorted(entries, key=lambda item: (item["content_sha256"], item["pts_sec"])),
        "source_mappings_sha256": canonical_sha256(mappings), "original_jpeg_bytes_preserved": True}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {"frames": len(entries), "contents": len({item["content_sha256"] for item in entries}),
            "manifest_sha256": file_sha256(manifest_path), "source_frames_modified": False}


def compact_training_aliases(rows: list[dict]) -> tuple[list[dict], dict]:
    """New derived representation matching serving f00 aliases; never raw edits.

    Canonical balance weights must be computed before this transformation:
    request-local aliases are not global source-anchor identities.
    """
    result, mappings = [], []
    for original in rows:
        row = deepcopy(original)
        require(row.get("split") == "train" and row.get("target_scope") == "exact_anchor_frame",
                "silver_alias_scope_forbidden")
        originals = [frame["frame_id"] for frame in row["frames"]]
        require(len(originals) == len(set(originals)), "silver_alias_source_frame_duplicate")
        aliases = {fid: f"f{index:02d}" for index, fid in enumerate(originals)}
        require(row["focal_frame_id"] in aliases, "silver_alias_focal_frame_missing")
        row["canonical_focal_frame_id"] = row["focal_frame_id"]
        row["focal_frame_id"] = aliases[row["focal_frame_id"]]
        for frame in row["frames"]:
            frame["canonical_frame_id"], frame["frame_id"] = frame["frame_id"], aliases[frame["frame_id"]]
        for fact in row["target"]["facts"]:
            require(set(fact["supporting_frame_ids"]) <= aliases.keys(), "silver_alias_foreign_support")
            fact["supporting_frame_ids"] = [aliases[fid] for fid in fact["supporting_frame_ids"]]
        for fid, alias in aliases.items():
            row["prompt"] = row["prompt"].replace(fid, alias)
        require(not any(fid in row["prompt"] for fid in aliases), "silver_alias_prompt_leaks_canonical_id")
        row["frame_alias_version"] = "forte.training.compact-f00.v1"
        result.append(row)
        mappings.append({"sample_id": row["sample_id"], "canonical_to_alias": aliases})
    return result, {"schema_version": "forte.training.alias-receipt.v1", "input_rows_sha256": canonical_sha256(rows),
        "derived_rows_sha256": canonical_sha256(result), "mappings": mappings,
        "teacher_bytes_modified": False, "source_pixels_modified": False,
        "balance_must_precede_aliasing": True, "serving_alias_scheme": "compact_frame_request_f00"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("combine", "media"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        function = combine_exports if args.operation == "combine" else compile_media
        print(json.dumps(function(load_json(args.config)), sort_keys=True))
        return 0
    except TrainingGateError as error:
        print(json.dumps({"status": "blocked", "reason_code": str(error)}))
        return 2
    except Exception as error:
        print(json.dumps({"status": "failed", "reason_code": "silver_combine_failed", "error_type": type(error).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
