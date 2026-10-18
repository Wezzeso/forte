"""Run generated-only ML source plumbing in a CPU image with no private mounts.

This is an operational dependency check, not a model, dataset or quality test.
All generated inputs and outputs live in one temporary directory and disappear.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import tempfile


def smoke() -> dict:
    from PIL import Image
    from jsonschema import Draft202012Validator
    from rules.catalog import load_catalog
    from annotations.machine import summarize_record, canonical_receipt_sha256
    from forte.shared.schemas import VideoAsset
    from forte.video.decode import frame_identity
    from training.coverage_export import export_coverage
    from training.guards import TrainingGateError, canonical_sha256, file_sha256, load_annotations
    from training.silver_combine import combine_exports, compile_media
    from training.teacher_sampling import TeacherSamplingConfig, PLAN_SCHEMA

    root = Path(__file__).resolve().parents[1]
    imports = ("annotations.catalog", "annotations.machine", "annotations.proposals",
               "annotations.rebase", "annotations.store", "training.coverage_export",
               "training.silver_combine", "training.silver_balance", "training.qwen38_silver",
               "training.prepare_native_ledgers", "training.qwen38_preflight",
               "deploy.qwen38_bind_runtime", "deploy.qwen38_acceptance_host")
    for name in imports:
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("ml_smoke_import_did_not_use_packaged_source")
    catalog = load_catalog(root / "rules/catalog.json", verify_archives=True)
    schema = root / "annotations/machine.schema.json"
    Draft202012Validator.check_schema(json.loads(schema.read_text(encoding="utf-8")))
    with tempfile.TemporaryDirectory(prefix="forte-generated-ml-smoke-") as temporary:
        work = Path(temporary)

        def write(name, value):
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            return path

        records, sources, proposals, slots = [], [], [], []
        identity = {"model_id": "synthetic/generated-no-model", "model_revision": "a" * 40,
                    "processor_revision": "a" * 40, "model_manifest_sha256": "b" * 64}
        raw_root = work / "generated-audits"
        for index in range(17):
            source = work / f"generated-source-{index}.txt"
            source.write_bytes(f"generated metadata source {index}".encode())
            content = file_sha256(source)
            job = f"generated-job-{index}"
            asset = VideoAsset(job_id=job, owned_path=str(source), content_hash=content,
                               duration_sec=1.0, width=2, height=2, time_base="1/10")
            frame = work / "generated-frames" / f"frame-{index}.jpg"
            frame.parent.mkdir(exist_ok=True)
            Image.new("RGB", (2, 2), (index, 30, 60)).save(frame, "JPEG")
            fid = frame_identity(asset, 5)
            prompt = hashlib.sha256(f"generated prompt {index}".encode()).hexdigest()
            definition = {"fact_id": "ANCHOR__CASH_VISIBLE", "description": (
                "INDEPENDENT EXACT-FRAME QUESTION: evaluate physical fact CASH_VISIBLE "
                "at frame f00, timestamp 0.500000000 seconds ONLY. Generated criterion."),
                "required": True}
            target = {"frame_id": fid, "pts_sec": .5, "fact_id": "CASH_VISIBLE",
                "state": "present", "supporting_frame_ids": [fid], "context_frame_ids": [fid],
                "scope": "exact_anchor_frame", "derivation": "direct_anchor_teacher_question",
                "localizable_category": "physical_anchor_atom", "human_verified": False,
                "eligible": True, "question_fact_id": definition["fact_id"],
                "question_definition": definition, "provenance": {"prompt_sha256": prompt,
                    "focal_question_sha256": canonical_receipt_sha256(definition)}}
            window = {"clip_id": f"generated-window-{index}", "second_index": 0,
                "start_sec": 0.0, "end_sec": 1.0, "focal_start_sec": 0.0,
                "focal_end_sec": 1.0, "anchor_frame_ids": [fid], "anchor_pts_sec": [.5],
                "frame_ids": [fid], "facts": [{"fact_id": "CASH_VISIBLE", "state": "present",
                    "supporting_frame_ids": [fid]}], "candidate_events": [],
                "complete": True, "prompt_sha256": prompt, "frame_fact_targets": [target]}
            slots.append({"content_sha256": content, "second_index": 0,
                          "frames": [{"frame_id": fid}], "anchor_frame_id": fid,
                          "anchor_pts_sec": .5})
            proposal = {"schema_version": "forte.local-proposal.v1", "asset": asset.model_dump(),
                "phase": "partial", "annotation_origin": "local_model", "human_reviewed": False,
                "organizer_labels_used_for_facts": False, "window_decisions": [window],
                "frames": [{"frame_id": fid, "source_pts": 5, "pts_sec": .5,
                    "width": 2, "height": 2, "job_id": job, "path": str(frame),
                    "jpeg_sha256": file_sha256(frame)}],
                "coverage": {"attempted_model_requests": 1, "complete": False},
                "provenance": {**identity, "actual_snapshot_verified": True,
                    "serving_attestation": {**identity, "model_files_verified": True}}}
            proposals.append((content, proposal))
            records.append({"content_sha256": content, "split": "train",
                            "scenario_group_id": f"generated-{index}", "duration_sec": 1.0,
                            "second_count": 1})
            sources.append({"content_sha256": content, "recording_id": f"generated-{index}"})
        assignments = [{key: row[key] for key in ("content_sha256", "split", "scenario_group_id")}
                       for row in records]
        digest = hashlib.sha256((json.dumps(assignments, ensure_ascii=False, sort_keys=True,
                                            indent=2) + "\n").encode()).hexdigest()
        split = write("generated-split.json", {"records": assignments, "assignments_sha256": digest})
        plan = {"schema_version": PLAN_SCHEMA, "partition": "train", "records": records,
                "sampling": asdict(TeacherSamplingConfig()), "split_file_sha256": file_sha256(split),
                "assignments_sha256": digest}
        plan["plan_sha256"] = canonical_sha256(plan)
        plan_path = write("generated-plan.json", plan)
        ledger = write("generated-ledger.json", {"attempts": slots})
        manifest = work / "generated-sources.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
        for content, proposal in proposals:
            proposal["provenance"]["plan_sha256"] = plan["plan_sha256"]
            path = raw_root / "train" / content / "teacher.private.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(proposal), encoding="utf-8")
        paths = {"teacher_plan": plan_path, "split": split, "native_ledger": ledger,
                 "source_manifest": manifest, "machine_schema": schema}
        config = {key: str(path) for key, path in paths.items()}
        config.update({"expected_" + key + "_sha256": file_sha256(path)
                       for key, path in paths.items()})
        config.update(raw_teacher_root=str(raw_root), expected_teacher=identity,
                      output=str(work / "generated-export.jsonl"),
                      proof_output=str(work / "generated-export-proof.json"))
        exported = export_coverage(config)
        rows = load_annotations(Path(config["output"]))
        for row in rows:
            target = row["windows"][0]["frame_fact_targets"][0]
            review = row["windows"][0]["support_reviews"][0]
            assert target["state"] == "present" and target["supervision_mask"] is True
            assert review["supports_state"] is True and review["provenance"] == target["provenance"]
        inputs = []
        for index, subset in enumerate((rows[:8], rows[8:])):
            path = work / f"generated-export-{index}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in subset), encoding="utf-8")
            inputs.append({"path": str(path), "sha256": file_sha256(path)})
        combined_path = work / "generated-combined.jsonl"
        combined = combine_exports({"input_exports": inputs, "expected_teacher": identity,
            "output": str(combined_path), "proof_output": str(work / "generated-combine-proof.json"),
            "machine_schema": str(schema), "expected_machine_schema_sha256": file_sha256(schema)})
        media = compile_media({"annotations": str(combined_path),
            "expected_annotations_sha256": file_sha256(combined_path),
            "media_root": str(work / "generated-compiled-media"),
            "manifest_output": str(work / "generated-media-manifest.json"),
            "assignments_sha256": digest, "source_mappings": [{
                "path_style": "windows" if work.drive else "posix",
                "original_prefix": str(work / "generated-frames"),
                "actual_root": str(work / "generated-frames")} ]})
        assert exported["records"] == combined["records"] == media["frames"] == 17
        assert exported["attempted_teacher_calls"] == exported["windows"] == 17
        assert summarize_record({"local_model_proposals": []}, work) is None
        # Exercise the newer failure-inclusive exporter with the real normalizer.
        # Every input below is generated; no live teacher audit is modified.
        audits = sorted(raw_root.glob("train/*/teacher.private.json"))
        failed = json.loads(audits[0].read_text(encoding="utf-8"))
        failed["window_decisions"] = []
        failed["invalid_teacher_windows"] = [{"state": "invalid_teacher_output"}]
        audits[0].write_text(json.dumps(failed), encoding="utf-8")
        partial_config = {**config, "output": str(work / "generated-partial-export.jsonl"),
                          "proof_output": str(work / "generated-partial-proof.json")}
        partial = export_coverage(partial_config)
        assert partial["records"] == partial["windows"] == 16
        assert partial["excluded_source_count"] == partial["invalid_or_incomplete_call_count"] == 1
        for audit in audits:
            document = json.loads(audit.read_text(encoding="utf-8"))
            document["window_decisions"] = []
            audit.write_text(json.dumps(document), encoding="utf-8")
        empty_config = {**config, "output": str(work / "generated-forbidden-empty-export.jsonl"),
                        "proof_output": str(work / "generated-all-invalid-proof.json")}
        try:
            export_coverage(empty_config)
        except TrainingGateError as error:
            assert str(error) == "coverage_export_no_eligible_targets"
        else:
            raise AssertionError("generated_all_invalid_export_was_not_rejected")
        assert not Path(empty_config["output"]).exists()
    return {"schema_version": "forte.ml-source-dependency-smoke.v1", "status": "passed",
        "public_module_imports": len(imports), "catalog_rules": len(catalog.rules),
        "catalog_archives_checksum_verified": sum(len(s["archives"]) for s in catalog.sources.values()),
        "machine_schema_validated": True, "generated_coverage_export_records": 17,
        "generated_combined_records": 17, "generated_original_jpegs_copied": 17,
        "generated_attested_exact_frame_targets": 17, "generated_window_receipts": 17,
        "generated_one_invalid_source_retained_records": 16,
        "generated_all_invalid_export_blocked": True,
        "gpu_calls": 0, "model_calls": 0, "private_inputs_read": False,
        "source_media_decoded": False, "generated_outputs_removed": True}


if __name__ == "__main__":
    print(json.dumps(smoke(), sort_keys=True))
