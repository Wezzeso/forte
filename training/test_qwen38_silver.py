"""Synthetic SILVER provenance, split, leakage, conflict and budget checks."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.qwen38_silver import (MEDIA_SCHEMA, PHYSICAL_FACTS, SCHEMA, build_silver_rows,
    MASK_WHOLE_SUPPORT_FACT, prepare_media, select_context, validate_compute_release, validate_media, validate_recipe)
from training.qwen38_preflight import semantic_recipe_sha256
from forte.video.batches import source_frame_id


def fixtures():
    content = "a" * 64
    config = {"schema_version": SCHEMA, "experiment_id": "QWEN38-SILVER", "human_verified": False,
        "dataset_origin": "local_teacher_silver", "model_id": "Qwen/Qwen3.8-27B", "model_type": "qwen3_5",
        "epochs": 1, "max_steps": 100, "microbatch_size": 1, "gradient_accumulation_steps": 8,
        "max_frames": 4, "max_length": 4096, "max_wall_seconds": 1800, "seed": 20261017,
        "sequence_packing": False, "max_pixels_per_image": 65536, "learning_rate": 1e-5,
        "media_format": "image_sequence", "quantization": "nf4", "stop_utc": "2026-10-18T06:01:25Z",
        "nf4_bias_initialization_policy": "exact_bf16_before_freeze.v1",
        "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
        "expected_assignments_sha256": "b" * 64, "expected_automated_split_sha256": "c" * 64,
        "runtime_image_sha256": "e" * 64, "code_archive_sha256": "f" * 64,
        "expected_attempt_plan_sha256": "1" * 64}
    expected = {"model_id": "private/synthetic", "model_revision": "d" * 40,
                "processor_revision": "d" * 40, "model_manifest_sha256": "d" * 64}
    config["expected_teacher"] = expected
    provenance = {**expected, "prompt_sha256": "e" * 64, "source_proposal_sha256": "f" * 64,
        "inference_location": "team_local_service", "external_ai_used": False, "actual_snapshot_verified": True,
        "serving_attestation": {**expected, "model_files_verified": True}}
    frames = [{"frame_id": source_frame_id(content, pts, 1, 10), "source_pts": pts, "pts_sec": pts / 10,
               "jpeg_sha256": str(index + 1) * 64, "path": "never-prompted/private-answer-name.jpg"}
              for index, pts in enumerate((0, 10, 20, 30))]
    context = [item["frame_id"] for item in frames]
    targets, reviews = [], []
    for fact, state in (("CASH_VISIBLE", "present"), ("CASH_HANDOVER_VISIBLE", "unclear")):
        definition = {"fact_id": "ANCHOR__" + fact, "description": "Exact physical focal question."}
        prov = {**deepcopy(provenance), "focal_question_sha256": canonical_sha256(definition),
                "explicit_teacher_target_sha256": "2" * 64}
        target = {"frame_id": context[1], "pts_sec": 1.0, "fact_id": fact, "state": state,
            "supporting_frame_ids": [context[1]], "context_frame_ids": context,
            "scope": "exact_anchor_frame", "derivation": "direct_anchor_teacher_question",
            "localizable_category": "physical_anchor_atom", "human_verified": False,
            "review_status": "machine_consistency_checked", "supervision_mask": True,
            "question_fact_id": "ANCHOR__" + fact, "question_definition": definition, "provenance": prov}
        targets.append(target)
        reviews.append({**deepcopy(target), "teacher_state": state, "supports_state": state != "unclear"})
    window = {"window_id": "synthetic-window", "frame_ids": context, "human_reviewed": False,
        "model_output_complete": True, "provenance": provenance, "frame_fact_targets": targets,
        "support_reviews": reviews, "target": {"facts": [], "candidates": [{"invented": "illegal"}]}}
    teacher = {"schema_version": "forte.machine-annotations.v1", "annotation_origin": "local_model",
        "review_status": "machine_consistency_checked", "human_reviewed": False, "organizer_labels_used": False,
        "source_pts_identity_checked": True, "content_sha256": content, "group_id": "synthetic-group",
        "recording_ids": ["synthetic-recording"], "split": "train", "source_time_base": "1/10",
        "source_start_sec": 0.0, "provenance": provenance, "frames": frames, "windows": [window],
        "provided_focus": {"coarse_anchor_sec": 991.0}, "explanation_ru": "workbook violation filename"}
    teacher["partition_binding"] = {"assignments_sha256": config["expected_assignments_sha256"],
        "split_file_sha256": config["expected_automated_split_sha256"]}
    split = {"records": [{"content_hash": content, "recording_id": "synthetic-recording",
                           "scenario_group_id": "synthetic-group", "split": "train"}]}
    sources = [{"content_sha256": content, "recording_id": "synthetic-recording"}]
    taxonomy = {"facts": [{"fact_id": fact, "question_ru": "Physical question " + fact,
                           "definition_ru": "Visible physical predicate."} for fact in sorted(PHYSICAL_FACTS)]}
    media = {(content, frame["frame_id"]): {"content_sha256": content, "frame_id": frame["frame_id"],
        "pts_sec": frame["pts_sec"], "source_pts": frame["source_pts"], "sha256": frame["jpeg_sha256"],
        "path": f"{content}/{frame['frame_id']}.jpg", "split": "train"} for frame in frames}
    return config, [teacher], split, sources, taxonomy, media


class SilverTrainingTests(unittest.TestCase):
    def rows(self, values):
        return build_silver_rows(*values)

    def test_partial_physical_only_and_no_answer_leakage(self):
        rows, coverage = self.rows(fixtures())
        row = rows[0]
        self.assertEqual(row["target"]["candidates"], [])
        self.assertTrue(row["fact_supervision_mask"]["CASH_VISIBLE"])
        self.assertFalse(row["fact_supervision_mask"]["CASH_HANDOVER_VISIBLE"])
        self.assertEqual(row["target"]["facts"][1]["state"], "unclear")
        self.assertNotIn("991", row["prompt"])
        self.assertNotIn("workbook", row["prompt"])
        self.assertNotIn("private-answer-name", json.dumps(row))
        self.assertEqual(coverage["legal_targets"], 0)
        self.assertEqual(coverage["boundary_targets"], 0)
        self.assertFalse(coverage["human_verified"])

    def test_validation_and_duplicate_alias_roles_are_forbidden(self):
        values = list(fixtures())
        values[2]["records"].append({**values[2]["records"][0], "split": "validation"})
        with self.assertRaisesRegex(TrainingGateError, "partition_forbidden"):
            self.rows(values)

    def test_wrong_teacher_and_external_inference_fail(self):
        for key, value in (("model_revision", "9" * 40), ("external_ai_used", True)):
            values = list(fixtures())
            values[1][0]["provenance"][key] = value
            with self.assertRaises(TrainingGateError):
                self.rows(values)

    def test_raw_support_join_mismatch_is_rejected(self):
        values = list(fixtures())
        values[1][0]["windows"][0]["support_reviews"][0]["teacher_state"] = "absent"
        with self.assertRaisesRegex(TrainingGateError, "support_review_join"):
            self.rows(values)

    def test_wrong_source_clock_is_rejected(self):
        values = list(fixtures())
        values[1][0]["frames"][0]["source_pts"] = 99
        with self.assertRaisesRegex(TrainingGateError, "source_frame_identity"):
            self.rows(values)

    def test_cross_window_conflicts_mask_only_exact_pair(self):
        values = list(fixtures())
        original = values[1][0]["windows"][0]
        # Retain an independent supported predicate so both rows remain usable.
        extra = deepcopy(original["frame_fact_targets"][1])
        extra.update(state="present", fact_id="CASH_HANDLER_PRESENT", question_fact_id="ANCHOR__CASH_HANDLER_PRESENT")
        extra["question_definition"]["fact_id"] = "ANCHOR__CASH_HANDLER_PRESENT"
        extra["provenance"]["focal_question_sha256"] = canonical_sha256(extra["question_definition"])
        original["frame_fact_targets"].append(extra)
        original["support_reviews"].append({**deepcopy(extra), "teacher_state": "present", "supports_state": True})
        conflicting = deepcopy(original)
        conflicting["window_id"] = "conflicting"
        conflicting["frame_fact_targets"][0]["state"] = "absent"
        conflicting["support_reviews"][0].update(state="absent", teacher_state="absent")
        values[1][0]["windows"].append(conflicting)
        rows, coverage = self.rows(values)
        self.assertEqual(coverage["cross_window_conflicting_pairs"], 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not row["fact_supervision_mask"]["CASH_VISIBLE"] for row in rows))
        self.assertTrue(all(row["fact_supervision_mask"]["CASH_HANDLER_PRESENT"] for row in rows))
        self.assertEqual(original["frame_fact_targets"][0]["state"], "present")

    def test_context_budget_is_deterministic_and_keeps_support(self):
        self.assertEqual(select_context(["a", "b", "c", "d"], {"c"}, 2), ["b", "c"])
        with self.assertRaisesRegex(TrainingGateError, "support_exceeds"):
            select_context(["a", "b", "c"], {"a", "b", "c"}, 2)

    def test_whole_over_budget_fact_is_masked_only_with_explicit_policy(self):
        values = list(fixtures())
        values[0]["max_frames"] = 4
        window = values[1][0]["windows"][0]
        content = values[1][0]["content_sha256"]
        for index in range(4, 9):
            pts = index * 10
            fid = source_frame_id(content, pts, 1, 10)
            frame = {"frame_id": fid, "source_pts": pts, "pts_sec": float(index),
                     "jpeg_sha256": str(index) * 64, "path": "private-source-unused-name.jpg"}
            values[1][0]["frames"].append(frame)
            window["frame_ids"].append(fid)
            values[5][(content, fid)] = {"content_sha256": content, "frame_id": fid, "source_pts": pts,
                "pts_sec": float(index), "sha256": frame["jpeg_sha256"], "path": f"{content}/{fid}.jpg", "split": "train"}
        for target in window["frame_fact_targets"] + window["support_reviews"]:
            target["context_frame_ids"] = window["frame_ids"].copy()
        # Keep one independently supported physical fact in the same row.
        target, review = window["frame_fact_targets"][1], window["support_reviews"][1]
        target["state"] = "absent"
        review.update(state="absent", teacher_state="absent", supports_state=True)
        original_support = list(window["frame_ids"])
        window["frame_fact_targets"][0]["supporting_frame_ids"] = original_support.copy()
        window["support_reviews"][0]["supporting_frame_ids"] = original_support.copy()
        original_sha = canonical_sha256(values[1])
        with self.assertRaisesRegex(TrainingGateError, "support_exceeds_frame_budget"):
            self.rows(values)
        values[0]["support_budget_policy"] = MASK_WHOLE_SUPPORT_FACT
        rows, coverage = self.rows(values)
        self.assertEqual(canonical_sha256(values[1]), original_sha)
        row = rows[0]
        self.assertFalse(row["fact_supervision_mask"]["CASH_VISIBLE"])
        self.assertTrue(row["fact_supervision_mask"]["CASH_HANDOVER_VISIBLE"])
        self.assertEqual(row["target"]["facts"][0]["state"], "unclear")
        self.assertEqual(row["target"]["facts"][0]["supporting_frame_ids"], [])
        exclusion = row["support_budget_exclusions"][0]
        self.assertEqual(exclusion["original_state"], "present")
        self.assertEqual(exclusion["original_supporting_frame_ids"], original_support)
        self.assertEqual(exclusion["original_support_count"], 9)
        self.assertEqual(exclusion["reason_code"], "full_original_support_exceeds_frame_budget")
        self.assertFalse(exclusion["source_or_citations_modified"])
        self.assertEqual(coverage["support_budget_excluded_fact_count"], 1)
        self.assertEqual(len(row["frames"]), 4)

    def test_multiple_individually_fitting_fact_supports_cannot_silently_truncate_union(self):
        values = list(fixtures())
        values[0].update(max_frames=2, support_budget_policy=MASK_WHOLE_SUPPORT_FACT)
        window = values[1][0]["windows"][0]
        for index, extra in enumerate((window["frame_ids"][0], window["frame_ids"][2])):
            target, review = window["frame_fact_targets"][index], window["support_reviews"][index]
            supports = [target["frame_id"], extra]
            target.update(state="present", supporting_frame_ids=supports)
            review.update(state="present", teacher_state="present", supports_state=True,
                          supporting_frame_ids=supports.copy())
        with self.assertRaisesRegex(TrainingGateError, "support_exceeds_frame_budget"):
            self.rows(values)

    def test_recipe_cannot_exceed_one_epoch_100_steps_or_stop(self):
        for key, value in (("epochs", 2), ("max_steps", 101), ("max_steps", True),
                           ("stop_utc", "2026-10-18T06:01:26Z"), ("max_pixels_per_image", 1024)):
            config = fixtures()[0]
            config[key] = value
            with self.assertRaises(TrainingGateError):
                validate_recipe(config)

    def test_media_rejects_filename_clues_and_unindexed_siblings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content, fid = "a" * 64, "b" * 64
            relative = f"{content}/{fid}.jpg"
            file = root / relative
            file.parent.mkdir()
            file.write_bytes(b"synthetic-jpeg-bytes")
            entry = {"split": "train", "content_sha256": content, "frame_id": fid,
                     "sha256": file_sha256(file), "path": relative}
            manifest = {"schema_version": MEDIA_SCHEMA, "split": "train", "human_verified": False, "frames": [entry]}
            self.assertEqual(len(validate_media(manifest, root)), 1)
            (root / "validation.mov").write_bytes(b"not-allowed")
            with self.assertRaisesRegex(TrainingGateError, "unindexed_media"):
                validate_media(manifest, root)
            (root / "validation.mov").unlink()
            entry["path"] = "answer-positive.jpg"
            with self.assertRaisesRegex(TrainingGateError, "content_names"):
                validate_media(manifest, root)

    def test_release_is_exact_mode_recipe_inputs_and_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = fixtures()[0]
            path = Path(temporary) / "release.json"
            config["compute_release"] = str(path)
            bindings = {"target_rows_sha256": "a" * 64}
            release = {"schema_version": "forte.qwen38.compute-release.v1", "released_by": "root",
                "authorized": True, "experiment_id": "QWEN38-SILVER", "mode": "smoke",
                "config_sha256": semantic_recipe_sha256(config), "input_bindings": bindings,
                "runtime_image_sha256": config["runtime_image_sha256"], "code_archive_sha256": config["code_archive_sha256"],
                "issued_at": "2026-10-17T12:00:00Z", "expires_at": "2026-10-17T14:00:00Z",
                "max_wall_seconds": 1800, "exclusive_idle_gpu_handoff_verified": True,
                "train_only_mount_verified": True, "idle_handoff_receipt_sha256": "2" * 64,
                "maximum_optimizer_steps": 2, "attempt_plan_sha256": config["expected_attempt_plan_sha256"]}
            path.write_text(json.dumps(release), encoding="utf-8")
            config["expected_compute_release_sha256"] = file_sha256(path)
            now = datetime(2026, 10, 17, 13, 0, tzinfo=timezone.utc)
            validate_compute_release(config, bindings, smoke=True, now=now)
            with self.assertRaisesRegex(TrainingGateError, "binding_mismatch"):
                validate_compute_release(config, bindings, smoke=False, now=now)
            config["learning_rate"] = 2e-5
            with self.assertRaisesRegex(TrainingGateError, "binding_mismatch"):
                validate_compute_release(config, bindings, smoke=True, now=now)

    def test_prepare_copies_original_bytes_and_does_not_mutate_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, teachers, split, sources, taxonomy, _ = fixtures()
            source_root = root / "originals"
            source_root.mkdir()
            for index, frame in enumerate(teachers[0]["frames"]):
                source = source_root / f"{index}.jpg"
                source.write_bytes(b"synthetic-jpeg-" + str(index).encode())
                frame["path"], frame["jpeg_sha256"] = str(source), file_sha256(source)
            for key, value, jsonl in (("machine_annotations", teachers, True), ("automated_split", split, False),
                                     ("source_manifest", sources, True), ("taxonomy_path", taxonomy, False)):
                path = root / (key + ".json")
                path.write_text("\n".join(json.dumps(row) for row in value) + "\n" if jsonl else json.dumps(value), encoding="utf-8")
                config[key] = str(path)
                pin_key = "expected_taxonomy_sha256" if key == "taxonomy_path" else "expected_" + key + "_sha256"
                config[pin_key] = file_sha256(path)
            teachers[0]["partition_binding"]["split_file_sha256"] = config["expected_automated_split_sha256"]
            teacher_path = Path(config["machine_annotations"])
            teacher_path.write_text(json.dumps(teachers[0]) + "\n", encoding="utf-8")
            config["expected_machine_annotations_sha256"] = file_sha256(teacher_path)
            config.update(source_frames_root=str(source_root), media_root=str(root / "derived-media"),
                          training_media_manifest=str(root / "derived-manifest.json"))
            before = file_sha256(teacher_path)
            with patch("training.qwen38_silver.validate_question_policy", return_value={}):
                result = prepare_media(config)
            self.assertEqual(result["frames"], 4)
            self.assertFalse(result["gpu_training_executed"])
            self.assertEqual(file_sha256(teacher_path), before)
            manifest = json.loads(Path(config["training_media_manifest"]).read_text())
            validate_media(manifest, Path(config["media_root"]))
            with self.assertRaisesRegex(TrainingGateError, "new_separate_destination"):
                with patch("training.qwen38_silver.validate_question_policy", return_value={}):
                    prepare_media(config)


if __name__ == "__main__":
    unittest.main()
