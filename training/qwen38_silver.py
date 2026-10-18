"""Separate partial machine-SILVER Qwen3.8 experiment; never an E1 gate bypass.

Run --preflight to validate metadata and original JPEG hashes without a GPU.
Actual --smoke / training require an independently pinned root compute release,
an actual-image tiny visual smoke proof, and a read-only TRAIN-only media mount.
No download, teacher call, serving selection, or external telemetry is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path, PureWindowsPath
import random
import re
import shutil
import sys
import time
from typing import Any

from .guards import (HEX64, TrainingGateError, canonical_sha256, file_sha256,
                     load_annotations, load_json, local_file, require)

SCHEMA = "forte.qwen38.silver-config.v1"
RUN_SCHEMA = "forte.qwen38.silver-run.v1"
MEDIA_SCHEMA = "forte.qwen38.train-media.v1"
PHYSICAL_FACTS = frozenset({"CASH_VISIBLE", "CASH_ON_WORK_SURFACE", "CASH_HANDLER_PRESENT",
    "CASH_HANDLER_LEAVES_WORKSTATION", "CASH_HANDOVER_VISIBLE", "CASH_COUNT_VISIBLE",
    "CASH_PLACED_IN_CLOTHING_POCKET", "CASH_PLACED_IN_PERSONAL_CONTAINER"})
FIXED_STOP = datetime(2026, 10, 18, 6, 1, 25, tzinfo=timezone.utc)
UNWEIGHTED_LOSS = "unweighted_partial_fact_tokens.v1"
BALANCED_LOSS = "forte.silver.fact-state-content-anchor.v1"
CANONICAL_FRAMES = "canonical_source_frame_ids.v1"
COMPACT_FRAMES = "forte.training.compact-f00.v1"
STRICT_SUPPORT_BUDGET = "require_all_support_within_frame_budget.v1"
MASK_WHOLE_SUPPORT_FACT = "mask_whole_fact_if_full_support_exceeds_budget.v1"


def utc(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        require(parsed.tzinfo is not None, "silver_deadline_timezone_missing")
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        raise TrainingGateError("silver_deadline_invalid") from None


def validate_recipe(config: dict[str, Any]) -> None:
    require(config.get("schema_version") == SCHEMA and config.get("experiment_id") == "QWEN38-SILVER",
            "silver_recipe_schema_invalid")
    require(config.get("human_verified") is False and config.get("dataset_origin") == "local_teacher_silver",
            "silver_origin_invalid")
    require(config.get("model_id") == "Qwen/Qwen3.8-27B" and config.get("model_type") == "qwen3_5",
            "silver_main_model_identity_invalid")
    for key, low, high in (("epochs", 1, 1), ("max_steps", 2, 100), ("microbatch_size", 1, 1),
                          ("gradient_accumulation_steps", 1, 64), ("max_frames", 2, 8),
                          ("max_length", 128, 4096), ("max_wall_seconds", 1, 3600)):
        value = config.get(key)
        require(type(value) is int and low <= value <= high, "silver_" + key + "_out_of_bounds")
    require(config.get("seed") == 20261017 and config.get("sequence_packing") is False,
            "silver_seed_or_packing_invalid")
    require(type(config.get("max_pixels_per_image")) is int
            and 65536 <= config["max_pixels_per_image"] <= 1048576, "silver_pixel_budget_invalid")
    require(type(config.get("learning_rate")) in {int, float}
            and math.isfinite(config["learning_rate"]) and 0 < config["learning_rate"] <= 1e-4,
            "silver_learning_rate_invalid")
    require(config.get("media_format") == "image_sequence" and config.get("quantization") in {"none", "nf4"},
            "silver_input_or_quantization_invalid")
    if config["quantization"] == "nf4":
        require(config.get("nf4_bias_initialization_policy") == "exact_bf16_before_freeze.v1",
                "silver_nf4_bias_initialization_policy_missing")
    require(utc(config.get("stop_utc")) <= FIXED_STOP, "silver_fixed_stop_moved")
    require(type(config.get("lora_rank")) is int and config["lora_rank"] in {8, 16}
            and type(config.get("lora_alpha")) is int and config["lora_alpha"] in {16, 32}
            and type(config.get("lora_dropout")) in {int, float} and config["lora_dropout"] == 0.05,
            "silver_lora_recipe_invalid")
    require(config.get("silver_loss_policy", UNWEIGHTED_LOSS) in {UNWEIGHTED_LOSS, BALANCED_LOSS},
            "silver_loss_policy_invalid")
    require(config.get("frame_alias_policy", CANONICAL_FRAMES) in {CANONICAL_FRAMES, COMPACT_FRAMES},
            "silver_frame_alias_policy_invalid")
    require(config.get("support_budget_policy", STRICT_SUPPORT_BUDGET) in {STRICT_SUPPORT_BUDGET, MASK_WHOLE_SUPPORT_FACT},
            "silver_support_budget_policy_invalid")
    if config.get("silver_loss_policy") == BALANCED_LOSS:
        require(type(config.get("silver_balance_minimum_class_contents")) is int
                and config["silver_balance_minimum_class_contents"] >= 2, "silver_balance_content_minimum_invalid")


def select_context(frame_ids: list[str], required: set[str], budget: int) -> list[str]:
    """Frozen selection: exact support first, then nearest temporal neighbours."""
    require(bool(frame_ids) and len(frame_ids) == len(set(frame_ids)) and required <= set(frame_ids),
            "silver_context_binding_invalid")
    require(bool(required) and len(required) <= budget, "silver_support_exceeds_frame_budget")
    positions = {key: index for index, key in enumerate(frame_ids)}
    ordered = sorted(frame_ids, key=lambda key: (min(abs(positions[key] - positions[item])
                                                  for item in required), positions[key]))
    selected = set(ordered[:budget]) | required
    require(len(selected) <= budget and len(selected) >= 2, "silver_context_frame_budget_invalid")
    return [key for key in frame_ids if key in selected]


def _teacher_identity(provenance: Any, expected: dict[str, Any], *, require_prompt: bool = True) -> None:
    require(isinstance(provenance, dict) and bool(expected), "silver_teacher_identity_missing")
    require(all(provenance.get(key) == expected.get(key) for key in
                ("model_id", "model_revision", "processor_revision", "model_manifest_sha256")),
            "silver_teacher_identity_mismatch")
    require(provenance.get("inference_location") == "team_local_service"
            and provenance.get("external_ai_used") is False and provenance.get("actual_snapshot_verified") is True,
            "silver_teacher_not_private_attested")
    for key in (("prompt_sha256",) if require_prompt else ()) + ("source_proposal_sha256", "model_manifest_sha256"):
        require(HEX64.fullmatch(str(provenance.get(key, ""))) is not None, "silver_teacher_pin_missing")
    attestation = provenance.get("serving_attestation", {})
    require(attestation.get("model_files_verified") is True
            and all(attestation.get(key) == expected.get(key) for key in
                    ("model_revision", "processor_revision", "model_manifest_sha256")),
            "silver_teacher_attestation_mismatch")


def validate_media(manifest: dict[str, Any], root: Path, *, verify_bytes: bool = True) -> dict[tuple[str, str], dict]:
    """No validation/final media, filename clues, or unindexed sibling bytes."""
    require(manifest.get("schema_version") == MEDIA_SCHEMA and manifest.get("split") == "train"
            and manifest.get("human_verified") is False, "silver_media_manifest_invalid")
    entries = manifest.get("frames")
    require(isinstance(entries, list) and bool(entries), "silver_media_empty")
    indexed, paths = {}, set()
    for item in entries:
        require(isinstance(item, dict) and item.get("split") == "train", "silver_media_partition_forbidden")
        content, fid = item.get("content_sha256"), item.get("frame_id")
        require(all(HEX64.fullmatch(str(key)) for key in (content, fid, item.get("sha256"))),
                "silver_media_hash_invalid")
        require(item.get("path") == f"{content}/{fid}.jpg", "silver_media_requires_content_names")
        path = local_file(item["path"], root)
        require((content, fid) not in indexed and path not in paths, "silver_media_duplicate")
        if verify_bytes:
            require(file_sha256(path) == item["sha256"], "silver_media_checksum_mismatch")
        indexed[(content, fid)] = item
        paths.add(path)
    require(paths == {item.resolve() for item in root.rglob("*") if item.is_file()},
            "silver_unindexed_media_mounted")
    return indexed


def build_silver_rows(config: dict, teachers: list[dict], split: dict, sources: list[dict],
                      taxonomy: dict, media: dict[tuple[str, str], dict]) -> tuple[list[dict], dict]:
    """Join original exact-frame teacher statements; never consume workbook targets."""
    assignments = defaultdict(list)
    for row in split.get("records", []):
        assignments[row.get("content_hash", row.get("content_sha256"))].append(row)
    source_ids = defaultdict(set)
    for row in sources:
        source_ids[row.get("content_sha256", row.get("content_hash"))].add(row.get("recording_id"))
    definitions = {item["fact_id"]: item for item in taxonomy.get("facts", [])}
    require(PHYSICAL_FACTS <= definitions.keys(), "silver_physical_taxonomy_incomplete")
    rows, observations, count = [], defaultdict(set), Counter()
    support_budget_exclusions = []
    budget_policy = config.get("support_budget_policy", STRICT_SUPPORT_BUDGET)
    require(budget_policy in {STRICT_SUPPORT_BUDGET, MASK_WHOLE_SUPPORT_FACT}, "silver_support_budget_policy_invalid")
    seen_contents = set()
    from forte.video.batches import source_frame_id

    for record in teachers:
        content = record.get("content_sha256")
        require(HEX64.fullmatch(str(content)) is not None and content not in seen_contents,
                "silver_teacher_content_duplicate_or_invalid")
        seen_contents.add(content)
        assignment = assignments.get(content, [])
        require(bool(assignment) and all(row.get("split") == "train" for row in assignment),
                "silver_training_partition_forbidden")
        require(record.get("schema_version") == "forte.machine-annotations.v1"
                and record.get("annotation_origin") == "local_model" and record.get("human_reviewed") is False
                and record.get("review_status") == "machine_consistency_checked"
                and record.get("organizer_labels_used") is False and record.get("source_pts_identity_checked") is True
                and record.get("split") == "train", "silver_teacher_origin_invalid")
        group = assignment[0].get("scenario_group_id", assignment[0].get("group_id"))
        require(record.get("group_id") == group and all(row.get("scenario_group_id", row.get("group_id")) == group
                for row in assignment), "silver_group_binding_mismatch")
        require(bool(record.get("recording_ids")) and set(record["recording_ids"]) <= source_ids[content]
                and set(record["recording_ids"]) <= {row.get("recording_id") for row in assignment},
                "silver_source_recording_mismatch")
        binding = record.get("partition_binding", {})
        require(binding.get("assignments_sha256") == config["expected_assignments_sha256"]
                and binding.get("split_file_sha256") == config["expected_automated_split_sha256"],
                "silver_teacher_split_pin_mismatch")
        _teacher_identity(record.get("provenance"), config["expected_teacher"], require_prompt=False)
        try:
            base = Fraction(record["source_time_base"])
        except (ValueError, KeyError, ZeroDivisionError):
            raise TrainingGateError("silver_source_time_base_invalid") from None
        frames = {}
        for frame in record.get("frames", []):
            fid = frame.get("frame_id")
            require(type(frame.get("source_pts")) is int and fid == source_frame_id(content, frame["source_pts"],
                    base.numerator, base.denominator), "silver_source_frame_identity_mismatch")
            require(type(frame.get("pts_sec")) in {int, float} and math.isfinite(frame["pts_sec"])
                    and abs(float(frame["source_pts"] * base) - record.get("source_start_sec", 0)
                            - frame["pts_sec"]) <= 1e-6, "silver_source_pts_mismatch")
            require(fid not in frames, "silver_source_frame_duplicate")
            frames[fid] = frame
        for window in record.get("windows", []):
            count["teacher_windows"] += 1
            require(window.get("human_reviewed") is False and window.get("model_output_complete") is True,
                    "silver_teacher_window_incomplete")
            _teacher_identity(window.get("provenance"), config["expected_teacher"])
            context = window.get("frame_ids", [])
            require(isinstance(context, list) and set(context) <= frames.keys() and len(context) >= 2,
                    "silver_window_context_invalid")
            require(all(frames[a]["pts_sec"] < frames[b]["pts_sec"] for a, b in zip(context, context[1:])),
                    "silver_window_pts_not_ordered")
            support_reviews = {(item.get("frame_id"), item.get("fact_id")): item
                               for item in window.get("support_reviews", [])}
            targets = defaultdict(list)
            seen_pairs = set()
            for target in window.get("frame_fact_targets", []):
                focal, fact, state = target.get("frame_id"), target.get("fact_id"), target.get("state")
                require(fact in PHYSICAL_FACTS and focal in context and state in {"present", "absent", "unclear"}
                        and (focal, fact) not in seen_pairs, "silver_exact_target_invalid")
                seen_pairs.add((focal, fact))
                require(target.get("scope") == "exact_anchor_frame"
                        and target.get("derivation") == "direct_anchor_teacher_question"
                        and target.get("localizable_category") == "physical_anchor_atom"
                        and target.get("human_verified") is False and target.get("review_status") == "machine_consistency_checked"
                        and target.get("pts_sec") == frames[focal]["pts_sec"]
                        and target.get("context_frame_ids") == context, "silver_exact_target_scope_invalid")
                provenance = target.get("provenance", {})
                _teacher_identity(provenance, config["expected_teacher"])
                definition = target.get("question_definition", {})
                require(provenance.get("prompt_sha256") == window["provenance"]["prompt_sha256"]
                        and provenance.get("focal_question_sha256") == canonical_sha256(definition)
                        and HEX64.fullmatch(str(provenance.get("explicit_teacher_target_sha256", ""))) is not None
                        and definition.get("fact_id") == "ANCHOR__" + fact
                        and target.get("question_fact_id") == "ANCHOR__" + fact,
                        "silver_exact_question_pin_mismatch")
                supports = target.get("supporting_frame_ids")
                require(isinstance(supports, list) and set(supports) <= set(context),
                        "silver_exact_support_invalid")
                review = support_reviews.get((focal, fact))
                require(review is not None and review.get("state") == state and review.get("teacher_state") == state
                        and review.get("supporting_frame_ids") == supports
                        and review.get("provenance") == provenance, "silver_support_review_join_mismatch")
                enabled = (target.get("supervision_mask") is True and review.get("supervision_mask") is True
                           and state != "unclear" and focal in supports and review.get("supports_state") is True)
                if enabled:
                    observations[(content, focal, fact)].add(state)
                targets[focal].append({"fact_id": fact, "state": state, "supporting_frame_ids": list(supports),
                                       "enabled": enabled})
                count["direct_frame_observations"] += 1
            for focal, facts in targets.items():
                exclusions = []
                for item in facts:
                    if item["enabled"] and len(set(item["supporting_frame_ids"]) | {focal}) > config["max_frames"]:
                        if budget_policy == MASK_WHOLE_SUPPORT_FACT:
                            # This complete physical target is omitted from loss.
                            # Its original state/citations stay in provenance; no
                            # subset is represented as sufficient source support.
                            exclusion = {"policy": budget_policy, "reason_code": "full_original_support_exceeds_frame_budget",
                                "content_sha256": content, "window_id": window.get("window_id"),
                                "focal_frame_id": focal, "fact_id": item["fact_id"], "original_state": item["state"],
                                "original_supporting_frame_ids": list(item["supporting_frame_ids"]),
                                "original_support_count": len(set(item["supporting_frame_ids"])),
                                "required_frame_count": len(set(item["supporting_frame_ids"]) | {focal}),
                                "max_frames": config["max_frames"], "teacher_window_sha256": canonical_sha256(window),
                                "training_target_omitted_whole": True, "source_or_citations_modified": False}
                            exclusions.append(exclusion)
                            support_budget_exclusions.append(exclusion)
                            item["enabled"] = False
                required = {focal} | {fid for item in facts if item["enabled"] for fid in item["supporting_frame_ids"]}
                selected = select_context(context, required, config["max_frames"])
                selected_frames = []
                for fid in selected:
                    item = media.get((content, fid))
                    require(item is not None and item["sha256"] == frames[fid].get("jpeg_sha256")
                            and item.get("source_pts") == frames[fid]["source_pts"]
                            and item.get("pts_sec") == frames[fid]["pts_sec"], "silver_media_source_join_mismatch")
                    selected_frames.append({key: item[key] for key in
                                            ("frame_id", "pts_sec", "source_pts", "path", "sha256")})
                row = {"content_sha256": content, "group_id": group, "split": "train",
                    "source_recording_ids": list(record["recording_ids"]), "frames": selected_frames,
                    "focal_frame_id": focal, "target_scope": "exact_anchor_frame",
                    "annotation_origin": "local_model", "human_reviewed": False,
                    "sample_id": canonical_sha256([content, window.get("window_id"), focal]),
                    "raw_teacher_window_sha256": canonical_sha256(window), "_facts": facts}
                if exclusions:
                    row["support_budget_exclusions"] = exclusions
                rows.append(row)
    conflicts = {key for key, states in observations.items() if len(states) > 1}
    retained = []
    for row in rows:
        facts = row.pop("_facts")
        mask = {item["fact_id"]: item["enabled"] and (row["content_sha256"], row["focal_frame_id"], item["fact_id"])
                not in conflicts for item in facts}
        for item in facts:
            if not mask[item["fact_id"]]:
                item["state"], item["supporting_frame_ids"] = "unclear", []
        if not any(mask.values()):
            count["fully_masked_windows"] += 1
            continue
        row["target"] = {"facts": [{key: item[key] for key in ("fact_id", "state", "supporting_frame_ids")}
                                    for item in facts], "candidates": []}
        row["fact_supervision_mask"] = mask
        # Only frozen public definitions and source timing reach the model. Raw
        # teacher prompts, filenames, point labels, explanations and notes do not.
        definitions_text = [{"fact_id": item["fact_id"], "question": definitions[item["fact_id"]]["question_ru"],
                             "definition": definitions[item["fact_id"]]["definition_ru"]} for item in facts]
        row["prompt"] = ("Evaluate the requested visible physical predicates at focal frame " + row["focal_frame_id"]
            + " only. Other frames are temporal context. Present/absent apply only at this source frame; "
            "uncertain or unobserved facts remain unclear. Return facts with supporting_frame_ids; candidates is empty. "
            "Do not infer legal violations, video-level verdicts, procedure omissions, or event boundaries. "
            + json.dumps(definitions_text, ensure_ascii=False, separators=(",", ":")))
        count["supervised_fact_targets"] += sum(mask.values())
        count["masked_fact_targets"] += len(mask) - sum(mask.values())
        retained.append(row)
    require(bool(retained), "silver_no_eligible_supported_targets")
    return retained, {**dict(count), "eligible_windows": len(retained), "teacher_contents": len(teachers),
        "eligible_contents": len({row["content_sha256"] for row in retained}),
        "cross_window_conflicting_pairs": len(conflicts), "human_verified": False,
        "target_scope": "exact_anchor_frame_physical_facts_only", "boundary_targets": 0,
        "legal_targets": 0, "workbook_targets": 0, "original_teacher_bytes_modified": False,
        "support_budget_policy": budget_policy, "support_budget_excluded_fact_count": len(support_budget_exclusions),
        "support_budget_exclusions_sha256": canonical_sha256(support_budget_exclusions),
        "target_rows_sha256": canonical_sha256(retained)}


def validate_question_policy(config: dict, teachers: list[dict]) -> dict:
    """Regenerate exact questions from independently pinned historical factory."""
    policy_path, catalog_path = Path(config["anchor_question_policy"]), Path(config["question_catalog"])
    require(policy_path.is_file() and file_sha256(policy_path) == config.get("expected_anchor_question_policy_sha256")
            and catalog_path.is_file() and file_sha256(catalog_path) == config.get("expected_question_catalog_sha256"),
            "silver_question_policy_or_catalog_pin_mismatch")
    policy = load_json(policy_path)
    payload = {key: value for key, value in policy.items() if key != "policy_sha256"}
    from . import teacher_sampling
    from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef
    require(policy.get("schema_version") == "forte.anchor-question-policy.v1"
            and policy.get("scope") == "exact_anchor_frame" and policy.get("derivation") == "direct_anchor_teacher_question"
            and policy.get("policy_sha256") == canonical_sha256(payload)
            and set(policy.get("fact_ids", [])) == PHYSICAL_FACTS and policy.get("question_count") == 8
            and policy.get("catalog_sha256") == file_sha256(catalog_path)
            and policy.get("teacher_module_sha256") == file_sha256(Path(teacher_sampling.__file__))
            and policy.get("factory_source_sha256") == hashlib.sha256(inspect.getsource(
                teacher_sampling.with_anchor_questions).encode()).hexdigest(), "silver_question_factory_pin_mismatch")
    catalog = load_json(catalog_path)["facts"]
    definitions = [FactDefinition(fact_id=fact, description=catalog[fact]["question_ru"] + " "
                                 + catalog[fact]["definition_ru"]) for fact in sorted(PHYSICAL_FACTS)]
    checked = 0
    for record in teachers:
        frames = {item["frame_id"]: item for item in record["frames"]}
        for window in record["windows"]:
            refs = [FrameRef(frame_id=fid, pts_sec=frames[fid]["pts_sec"], path="/private/source/" + fid + ".jpg",
                             width=1, height=1) for fid in window["frame_ids"]]
            request = ClipRequest(job_id="silver-policy", clip_id="silver-policy", frame_refs=refs,
                                  fact_definitions=definitions)
            by_focal = {ref.frame_id: ref for ref in refs}
            for target in window.get("frame_fact_targets", []):
                augmented = teacher_sampling.with_anchor_questions(request, by_focal[target["frame_id"]])
                exact = {item.fact_id: item.model_dump(mode="json") for item in augmented.fact_definitions}
                require(target.get("question_definition") == exact.get(target.get("question_fact_id")),
                        "silver_exact_historical_question_mismatch")
                checked += 1
    return {"question_policy_sha256": file_sha256(policy_path), "question_catalog_sha256": file_sha256(catalog_path),
            "exact_questions_checked": checked}


def apply_supervision_policy(config: dict, canonical_rows: list[dict], *, require_pins: bool = True
                             ) -> tuple[list[dict], dict]:
    """Derive a pinned optional objective after original source/target validation."""
    policy = config.get("silver_loss_policy", UNWEIGHTED_LOSS)
    require(policy in {UNWEIGHTED_LOSS, BALANCED_LOSS}, "silver_loss_policy_invalid")
    rows, plan = canonical_rows, None
    if policy == BALANCED_LOSS:
        from .silver_balance import plan_balanced_silver, balanced_rows
        minimum = config.get("silver_balance_minimum_class_contents")
        require(type(minimum) is int and minimum >= 2, "silver_balance_content_minimum_invalid")
        plan = plan_balanced_silver(canonical_rows, minimum_class_contents=minimum)
        if require_pins:
            plan_path = Path(config.get("silver_balance_plan", ""))
            require(plan_path.is_file() and file_sha256(plan_path) == config.get("expected_silver_balance_plan_sha256")
                    and load_json(plan_path) == plan
                    and plan["plan_sha256"] == config.get("expected_silver_balance_plan_semantic_sha256"),
                    "silver_balance_exact_plan_pin_mismatch")
        rows = balanced_rows(canonical_rows, plan)
    alias_policy = config.get("frame_alias_policy", CANONICAL_FRAMES)
    require(alias_policy in {CANONICAL_FRAMES, COMPACT_FRAMES}, "silver_frame_alias_policy_invalid")
    alias_receipt = None
    if alias_policy == COMPACT_FRAMES:
        from .silver_combine import compact_training_aliases
        rows, alias_receipt = compact_training_aliases(rows)
    weights = [{"sample_id": row["sample_id"], "fact_loss_weights": row.get("fact_loss_weights")}
               for row in rows]
    objective = {"schema_version": "forte.silver-objective.v1", "loss_policy": policy,
        "frame_alias_policy": alias_policy, "minimum_class_contents": config.get("silver_balance_minimum_class_contents")
            if plan is not None else None,
        "support_budget_policy": config.get("support_budget_policy", STRICT_SUPPORT_BUDGET),
        "normalization": "global_fact_state_content_unique_anchor_mass_times_effective_row_count" if plan is not None
            else "model_mean_cross_entropy_over_enabled_fact_tokens",
        "unknown_legal_candidate_boundary_tokens_masked": True,
        "balance_precedes_aliasing": True, "example_count": len(rows)}
    report = {"canonical_target_rows_sha256": canonical_sha256(canonical_rows),
        "effective_rows_sha256": canonical_sha256(rows), "effective_fact_weights_sha256": canonical_sha256(weights),
        "supervision_objective_sha256": canonical_sha256(objective), "objective": objective,
        "effective_rows": len(rows), "human_verified": False, "balance_plan": plan,
        "alias_receipt": alias_receipt}
    if plan is not None:
        report["balance_plan_semantic_sha256"] = plan["plan_sha256"]
        if require_pins:
            for key in ("effective_rows_sha256", "effective_fact_weights_sha256"):
                require(report[key] == config.get("expected_" + key), "silver_balanced_effective_input_pin_mismatch")
            report["balance_plan_file_sha256"] = file_sha256(Path(config["silver_balance_plan"]))
    return rows, report


def load_inputs(config: dict, *, require_supervision_pins: bool = True) -> tuple[list[dict], dict, dict]:
    validate_recipe(config)
    from .qwen38_preflight import validate_silver_metadata
    audit = validate_silver_metadata(config)
    require(audit.get("ok") is True, "silver_metadata_exposure_gate_failed")
    keys = ("source_manifest", "automated_split", "scenario_groups", "machine_annotations", "export_proof",
            "taxonomy_path", "training_media_manifest")
    hashes = {}
    for key in keys:
        digest = file_sha256(Path(config[key]))
        pin_key = "expected_taxonomy_sha256" if key == "taxonomy_path" else "expected_" + key + "_sha256"
        require(digest == config.get(pin_key), "silver_input_file_pin_mismatch")
        hashes[key + "_sha256"] = digest
    media_root = Path(config["media_root"]).resolve()
    media = validate_media(load_json(Path(config["training_media_manifest"])), media_root)
    teachers = load_annotations(Path(config["machine_annotations"]))
    policy = validate_question_policy(config, teachers)
    hashes.update({key: value for key, value in policy.items() if key.endswith("sha256")})
    rows, coverage = build_silver_rows(config, teachers,
        load_json(Path(config["automated_split"])), load_annotations(Path(config["source_manifest"])),
        load_json(Path(config["taxonomy_path"])), media)
    rows, supervision = apply_supervision_policy(config, rows, require_pins=require_supervision_pins)
    return rows, hashes, {"coverage": coverage, "exposure": audit, "question_policy": policy,
                         "supervision": supervision}


def prepare_supervision(config: dict) -> dict:
    """Create one new private balancing plan; never fit, reserve or mutate raw labels."""
    require(config.get("silver_loss_policy") == BALANCED_LOSS, "silver_prepare_requires_balance_policy")
    rows, _, audit = load_inputs(config, require_supervision_pins=False)
    supervision = audit["supervision"]
    path = Path(config["silver_balance_plan"]).resolve()
    require("private" in path.parts and not path.exists(), "silver_balance_plan_new_private_path_required")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, supervision["balance_plan"])
    return {"status": "supervision_prepared", "gpu_training_executed": False, "human_verified": False,
        "training_authorized_by_this_preparation": False, "effective_rows": len(rows),
        "expected_silver_balance_plan_sha256": file_sha256(path),
        "expected_silver_balance_plan_semantic_sha256": supervision["balance_plan_semantic_sha256"],
        "expected_effective_rows_sha256": supervision["effective_rows_sha256"],
        "expected_effective_fact_weights_sha256": supervision["effective_fact_weights_sha256"],
        "supervision_objective_sha256": supervision["supervision_objective_sha256"],
        "aggregate": supervision["balance_plan"]["aggregate"]}


def collect_release_bindings(config: dict, hashes: dict, audit: dict) -> dict:
    """One metadata-only binding graph shared by the executor and local binder."""
    result = dict(hashes)
    for key in ("model_manifest", "runtime_lock"):
        path = Path(config[key])
        require(path.is_file(), "silver_binding_artifact_missing")
        result[key + "_sha256"] = file_sha256(path)
    require(result["runtime_lock_sha256"] == config.get("expected_runtime_lock_sha256"),
            "silver_runtime_lock_pin_mismatch")
    result["target_rows_sha256"] = audit["coverage"]["target_rows_sha256"]
    if "support_budget_exclusions_sha256" in audit["coverage"]:
        result["support_budget_exclusions_sha256"] = audit["coverage"]["support_budget_exclusions_sha256"]
    supervision = audit.get("supervision")
    require(isinstance(supervision, dict), "silver_effective_supervision_binding_missing")
    for key in ("canonical_target_rows_sha256", "effective_rows_sha256", "effective_fact_weights_sha256",
                "supervision_objective_sha256", "balance_plan_semantic_sha256", "balance_plan_file_sha256"):
        if key in supervision:
            result[key] = supervision[key]
    nf4_proof = verify_nf4_initialization_proof(config, base_manifest_sha256=result["model_manifest_sha256"])
    if nf4_proof is not None:
        result["nf4_bias_initialization_proof_sha256"] = nf4_proof["proof_sha256"]
        result["nf4_proof_helper_source_sha256"] = nf4_proof["helper_source_sha256"]
    return result


def supervision_summary(report: dict) -> dict:
    result = {key: value for key, value in report.items() if key not in {"balance_plan", "alias_receipt"}}
    if report.get("balance_plan") is not None:
        result["balance_aggregate"] = report["balance_plan"]["aggregate"]
        result["active_fact_ids"] = report["balance_plan"]["active_fact_ids"]
        result["deferred_fact_ids"] = report["balance_plan"]["deferred_fact_ids"]
    if report.get("alias_receipt") is not None:
        result["alias_receipt_sha256"] = canonical_sha256(report["alias_receipt"])
    return result


def collate_objective_batch(collator: Any, row: dict, *, device: str, balanced: bool) -> tuple[dict, Any]:
    """Keep verified fact-span metadata outside the model's forward kwargs."""
    if not balanced:
        return {key: tensor.to(device) for key, tensor in collator([row]).items()}, None
    import torch
    from .silver_balance import fact_token_loss_weights
    batch, positions = collator.collate_with_fact_spans([row])
    weights = fact_token_loss_weights(batch["labels"][0].tolist(), positions, row["fact_loss_weights"])
    return ({key: tensor.to(device) for key, tensor in batch.items()},
            torch.tensor([weights], dtype=torch.float32, device=device))


def objective_loss(model: Any, batch: dict, weights: Any, *, example_count: int) -> Any:
    if weights is None:
        return model(**batch).loss
    from .silver_balance import weighted_causal_fact_loss
    kwargs = {key: value for key, value in batch.items() if key != "labels"}
    output = model(**kwargs, labels=None)
    return weighted_causal_fact_loss(output.logits, batch["labels"], weights, example_count=example_count)


def prepare_media(config: dict) -> dict:
    """Copy verified original JPEG bytes into a new mount; preserve raw sources.

    This preparation has no inference or GPU side effects. Only explicit TRAIN
    source contents are copied, under content hashes rather than source names.
    The generated manifest is private and its hash must be frozen in a new run
    config before the exposure preflight or compute release can pass.
    """
    validate_recipe(config)
    paths = {key: Path(config[key]) for key in ("machine_annotations", "automated_split", "source_manifest", "taxonomy_path")}
    for key, path in paths.items():
        pin_key = "expected_taxonomy_sha256" if key == "taxonomy_path" else "expected_" + key + "_sha256"
        require(path.is_file() and file_sha256(path) == config.get(pin_key), "silver_prepare_input_pin_mismatch")
    teachers = load_annotations(paths["machine_annotations"])
    validate_question_policy(config, teachers)
    source_root = Path(config["source_frames_root"]).resolve()
    root, manifest_path = Path(config["media_root"]).resolve(), Path(config["training_media_manifest"]).resolve()
    require(not root.exists() and not manifest_path.exists() and not manifest_path.is_relative_to(root)
            and not root.is_relative_to(source_root) and not source_root.is_relative_to(root),
            "silver_prepare_requires_new_separate_destination")
    entries, source_paths = {}, {}
    for record in teachers:
        content = record.get("content_sha256")
        for frame in record.get("frames", []):
            fid = frame.get("frame_id")
            original_path = frame.get("path")
            original_root = config.get("source_frames_original_root")
            if original_root:
                try:
                    # Relocate only the exact configured original prefix. This
                    # accommodates immutable Windows teacher receipts on Linux.
                    original_path = str(Path(*PureWindowsPath(original_path).relative_to(
                        PureWindowsPath(original_root)).parts))
                except (ValueError, TypeError):
                    raise TrainingGateError("silver_source_frame_relocation_invalid") from None
            source = local_file(original_path, source_root)
            require(HEX64.fullmatch(str(content)) is not None and HEX64.fullmatch(str(fid)) is not None
                    and file_sha256(source) == frame.get("jpeg_sha256"), "silver_prepare_original_frame_hash_mismatch")
            entry = {"content_sha256": content, "frame_id": fid, "source_pts": frame["source_pts"],
                     "pts_sec": frame["pts_sec"], "sha256": frame["jpeg_sha256"], "split": "train",
                     "path": f"{content}/{fid}.jpg"}
            require((content, fid) not in entries or entries[(content, fid)] == entry,
                    "silver_prepare_frame_conflict")
            entries[(content, fid)], source_paths[(content, fid)] = entry, source
    # Full exact-frame provenance/partition joins run before creating any file.
    _, coverage = build_silver_rows(config, teachers, load_json(paths["automated_split"]),
        load_annotations(paths["source_manifest"]), load_json(paths["taxonomy_path"]), entries)
    root.mkdir(parents=True)
    for key, item in sorted(entries.items()):
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_paths[key], destination)
        require(file_sha256(destination) == item["sha256"], "silver_prepare_copy_hash_mismatch")
    manifest = {"schema_version": MEDIA_SCHEMA, "split": "train", "human_verified": False,
                "assignments_sha256": config["expected_assignments_sha256"],
                "machine_annotations_sha256": config["expected_machine_annotations_sha256"],
                "frames": [item for _, item in sorted(entries.items())]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    validate_media(manifest, root)
    return {"status": "train_media_prepared", "frames": len(entries),
            "training_media_manifest_sha256": file_sha256(manifest_path), "human_verified": False,
            "gpu_training_executed": False, "coverage": coverage,
            "training_authorized_by_this_preparation": False}


def validate_compute_release(config: dict, bindings: dict, *, smoke: bool,
                             now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    release_path = Path(config.get("compute_release", ""))
    require(release_path.is_file() and file_sha256(release_path) == config.get("expected_compute_release_sha256"),
            "silver_fresh_root_release_missing")
    release = load_json(release_path)
    from .qwen38_preflight import semantic_recipe_sha256
    recipe_hash = semantic_recipe_sha256(config)
    require(release.get("schema_version") == "forte.qwen38.compute-release.v1"
            and release.get("released_by") == "root" and release.get("authorized") is True
            and release.get("experiment_id") == "QWEN38-SILVER"
            and release.get("mode") == ("smoke" if smoke else "train")
            and release.get("config_sha256") == recipe_hash
            and release.get("input_bindings") == bindings, "silver_root_release_binding_mismatch")
    for key in ("runtime_image_sha256", "code_archive_sha256"):
        require(HEX64.fullmatch(str(config.get(key, ""))) is not None and release.get(key) == config[key],
                "silver_runtime_release_pin_mismatch")
    require(release.get("exclusive_idle_gpu_handoff_verified") is True
            and release.get("train_only_mount_verified") is True
            and HEX64.fullmatch(str(release.get("idle_handoff_receipt_sha256", ""))) is not None,
            "silver_exclusive_gpu_or_mount_release_missing")
    require(release.get("attempt_plan_sha256") == config.get("expected_attempt_plan_sha256")
            and type(release.get("maximum_optimizer_steps")) is int
            and 2 <= release["maximum_optimizer_steps"] <= (2 if smoke else 100)
            and (2 if smoke else config["max_steps"]) <= release["maximum_optimizer_steps"],
            "silver_release_attempt_or_step_bound_invalid")
    require(utc(release.get("issued_at")) <= now < utc(release.get("expires_at"))
            <= utc(config["stop_utc"]) <= FIXED_STOP, "silver_compute_release_expired")
    require(type(release.get("max_wall_seconds")) is int
            and config["max_wall_seconds"] <= release["max_wall_seconds"], "silver_release_wall_ceiling_exceeded")
    require(now.timestamp() + config["max_wall_seconds"] + 120 <= utc(config["stop_utc"]).timestamp(),
            "silver_stop_headroom_insufficient")
    return release


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_hash(tensor: Any) -> str:
    from .qwen38_proof import tensor_sha256
    return tensor_sha256(tensor)


def parameter_hashes(model: Any, *, trainable: bool) -> dict[str, str]:
    return {name: tensor_hash(parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad is trainable}


def verify_nf4_initialization_proof(config: dict, *, base_manifest_sha256: str) -> dict | None:
    """Require the actual full-model generated proof for this exact helper/image."""
    if config["quantization"] != "nf4":
        return None
    from . import qwen38_proof
    path = Path(config.get("nf4_bias_initialization_proof", ""))
    require(path.is_file() and file_sha256(path) == config.get("expected_nf4_bias_initialization_proof_sha256"),
            "silver_nf4_full_generated_proof_missing")
    proof = load_json(path)
    require(proof.get("schema_version") == "forte.qwen38.nf4-initialization-proof.v1"
            and proof.get("status") == "passed" and proof.get("private_input_used") is False
            and proof.get("model_id") == config["model_id"] and proof.get("base_revision") == config["base_revision"]
            and proof.get("base_manifest_sha256") == base_manifest_sha256
            and proof.get("runtime_image_sha256") == config["runtime_image_sha256"]
            and proof.get("helper_source_sha256") == file_sha256(Path(qwen38_proof.__file__))
            and type(proof.get("optimizer_steps")) is int and proof["optimizer_steps"] >= 2
            and proof.get("all_initialization_values_exactly_preserved") is True
            and proof.get("frozen_post_initialization_bytes_unchanged") is True
            and proof.get("quantization_state_unchanged") is True
            and set(proof.get("changed_adapter_groups", [])) == {"self_attn", "linear_attn", "mlp"},
            "silver_nf4_full_generated_proof_invalid")
    return {"proof_sha256": file_sha256(path), "helper_source_sha256": proof["helper_source_sha256"],
            "optimizer_steps": proof["optimizer_steps"], "private_input_used": False}


def safe_failure_code(error: Exception) -> str:
    if isinstance(error, TrainingGateError) and re.fullmatch(r"[a-z][a-z0-9_]{0,159}", str(error)):
        return str(error)
    return "silver_local_training_error"


def _verify_smoke_proof(config: dict, bindings: dict, *, full: bool = False) -> None:
    key = "two_step_smoke" if full else "tiny_smoke"
    path = Path(config.get(key + "_receipt", ""))
    require(path.is_file() and file_sha256(path) == config.get("expected_" + key + "_receipt_sha256"),
            "silver_" + key + "_proof_missing")
    proof = load_json(path)
    require(proof.get("runtime_image_sha256") == config["runtime_image_sha256"]
            and proof.get("gpu_executed") is True and proof.get("finite_loss_verified") is True
            and proof.get("gradients_verified") is True and proof.get("peft_load_verified") is True
            and proof.get("reload_forward_verified") is True
            and proof.get("frozen_tensors_unchanged") is True and proof.get("changed_adapter_tensors", 0) > 0,
            "silver_" + key + "_proof_invalid")
    if full:
        require(proof.get("schema_version") == RUN_SCHEMA and proof.get("mode") == "smoke"
                and proof.get("status") == "candidate_unselected" and proof.get("human_verified") is False
                and proof.get("optimizer_steps") == 2 and proof.get("input_bindings") == bindings
                and proof.get("base_revision") == config["base_revision"]
                and proof.get("code_archive_sha256") == config["code_archive_sha256"], "silver_two_step_data_binding_invalid")
    else:
        require(proof.get("schema_version") == "forte.training.tiny-visual-smoke.v1"
                and proof.get("status") == "passed" and proof.get("model_type") == "qwen3_5"
                and proof.get("optimizer_steps") == 2, "silver_tiny_visual_architecture_proof_invalid")


def validate_attempt_accounting(config: dict, rows: list[dict], *, smoke: bool,
                                now: datetime | None = None) -> dict:
    """Pre-reserve every private forward/backward, including two reload forwards."""
    from .qwen38_preflight import validate_attempt_reservation
    plan_path = Path(config.get("attempt_plan", ""))
    reservation_path = Path(config.get("attempt_reservation", ""))
    require(plan_path.is_file() and file_sha256(plan_path) == config.get("expected_attempt_plan_sha256")
            and reservation_path.is_file(), "silver_attempt_accounting_pin_missing")
    plan, reservation = load_json(plan_path), load_json(reservation_path)
    validate_attempt_reservation(reservation, plan, plan_sha256=config["expected_attempt_plan_sha256"],
                                 phase="smoke" if smoke else "fit", now=now)
    from .qwen38_preflight import semantic_recipe_sha256
    recipe_hash = semantic_recipe_sha256(config)
    require(reservation.get("release_sha256") == config.get("expected_compute_release_sha256")
            and reservation.get("config_sha256") == recipe_hash
            and reservation.get("runtime_image_sha256") == config.get("runtime_image_sha256")
            and reservation.get("source_archive_sha256") == config.get("code_archive_sha256")
            and reservation.get("base_manifest_sha256") == file_sha256(Path(config["model_manifest"])),
            "silver_attempt_runtime_binding_mismatch")
    budget = min(len(rows), (2 if smoke else config["max_steps"]) * (1 if smoke else config["gradient_accumulation_steps"]))
    order = list(range(len(rows)))
    random.Random(config["seed"]).shuffle(order)
    expected = [rows[index]["content_sha256"] for index in order[:budget]]
    expected.extend([rows[order[0]]["content_sha256"]] * 2)
    require([item.get("content_sha256") for item in reservation.get("attempts", [])] == expected
            and all(item.get("split") == "train" for item in reservation["attempts"]),
            "silver_attempt_plan_does_not_cover_exact_forwards")
    release = load_json(Path(config["compute_release"]))
    require(len(expected) == release.get("maximum_attempts"),
            "silver_release_attempt_count_mismatch")
    if config.get("silver_loss_policy") == BALANCED_LOSS:
        require(reservation.get("input_bindings") == release.get("input_bindings")
                and all(HEX64.fullmatch(str(reservation.get("input_bindings", {}).get(key, ""))) is not None
                        for key in ("effective_rows_sha256", "effective_fact_weights_sha256", "supervision_objective_sha256",
                                    "balance_plan_semantic_sha256", "balance_plan_file_sha256")),
                "silver_balanced_attempt_objective_binding_mismatch")
    return reservation


def train(config: dict, *, preflight_only: bool = False, smoke: bool = False) -> dict:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false")
    rows, hashes, audit = load_inputs(config)
    from .train import check_local_model
    model_dir, processor_dir = check_local_model(config)
    hashes = collect_release_bindings(config, hashes, audit)
    runtime_path = Path(config["runtime_lock"])
    require(runtime_path.is_file() and file_sha256(runtime_path) == config.get("expected_runtime_lock_sha256"),
            "silver_runtime_lock_pin_mismatch")
    runtime_lock = load_json(runtime_path)
    if config["quantization"] == "nf4":
        require("bitsandbytes" in {**runtime_lock.get("core_packages", {}), **runtime_lock.get("addon_packages", {})},
                "silver_nf4_runtime_package_unpinned")
    nf4_proof = verify_nf4_initialization_proof(config, base_manifest_sha256=hashes["model_manifest_sha256"])
    output = Path(config["output_dir"]).resolve()
    require(not output.exists() and not any(output.is_relative_to(root) for root in
                (model_dir, processor_dir, Path(config["media_root"]).resolve())), "silver_output_would_modify_inputs")
    if preflight_only:
        return {"status": "preflight_passed", "gpu_training_executed": False, "human_verified": False,
                "input_bindings": hashes, **{key: value for key, value in audit.items() if key != "supervision"},
                "supervision": supervision_summary(audit["supervision"])}
    release = validate_compute_release(config, hashes, smoke=smoke)
    reservation = validate_attempt_accounting(config, rows, smoke=smoke)
    require(sys.platform == "linux" and bool(os.statvfs(config["media_root"]).f_flag & os.ST_RDONLY),
            "silver_train_media_mount_not_readonly")
    _verify_smoke_proof(config, hashes)
    if not smoke:
        _verify_smoke_proof(config, hashes, full=True)

    from .train import configure_adapter, load_model_and_processor
    from .data import SilverCompletionCollator
    from .runtime_check import verify_runtime
    from .qwen38_proof import (canonicalize_nf4_bias_storage, compare_frozen_snapshots,
                               frozen_snapshot, save_quarantined_adapter)
    runtime_verified = verify_runtime(runtime_path, gpu=False)
    import numpy as np
    import torch
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.monotonic()
    def check_time() -> None:
        require(time.monotonic() - started < config["max_wall_seconds"]
                and datetime.now(timezone.utc).timestamp() + 120 < min(utc(config["stop_utc"]).timestamp(),
                    utc(release["expires_at"]).timestamp()),
                "silver_wall_or_stop_ceiling_reached")
    output.mkdir(parents=True)
    proof_dir = output / "proof"
    proof_dir.mkdir()
    balanced = config.get("silver_loss_policy", UNWEIGHTED_LOSS) == BALANCED_LOSS
    supervision = audit["supervision"]
    if supervision.get("balance_plan") is not None:
        _write_json(proof_dir / "balance-plan.private.json", supervision["balance_plan"])
    if supervision.get("alias_receipt") is not None:
        _write_json(proof_dir / "frame-aliases.private.json", supervision["alias_receipt"])
    _write_json(proof_dir / "support-budget-exclusions.private.json", {
        "schema_version": "forte.silver-support-budget-exclusions.v1",
        "policy": config.get("support_budget_policy", STRICT_SUPPORT_BUDGET),
        "exclusions": [item for row in rows for item in row.get("support_budget_exclusions", [])],
        "canonical_exclusions_sha256": audit["coverage"].get("support_budget_exclusions_sha256")})
    run = {"schema_version": RUN_SCHEMA, "status": "running", "mode": "smoke" if smoke else "train",
        "dataset_origin": "local_teacher_silver", "human_verified": False, "selection": "unselected",
        "gpu_executed": False, "optimizer_steps": 0, "microsteps": 0,
        "phase": "phase4_optional_balanced_physical_facts" if balanced else "phase3_exact_nf4_bias_initialization_and_byte_proof",
        "failure_stage": None, "reason_code": None, "stage": "base_load",
        "peak_vram_bytes": None, "peak_reserved_vram_bytes": None,
        "finite_loss_verified": False, "gradients_verified": False, "frozen_tensors_unchanged": False,
        "losses": [], "optimizer_step_metrics": [],
        "started_at": datetime.now(timezone.utc).isoformat(), "input_bindings": hashes,
        "base_revision": config["base_revision"], "processor_revision": config["processor_revision"],
        "config_sha256": canonical_sha256(config), "runtime_image_sha256": config["runtime_image_sha256"],
        "code_archive_sha256": config["code_archive_sha256"], "coverage": audit["coverage"],
        "supervision": supervision_summary(supervision),
        "supervision_objective_sha256": hashes["supervision_objective_sha256"],
        "base_manifest_sha256": hashes["model_manifest_sha256"],
        "quantization": {"format": config["quantization"], "compute_dtype": "bfloat16",
                         "double_quant": config["quantization"] == "nf4"},
        "processor_limits": {"max_pixels_per_image": config["max_pixels_per_image"], "max_frames": config["max_frames"],
                             "max_length": config["max_length"]},
        "attempt_plan_sha256": config["expected_attempt_plan_sha256"],
        "attempt_reservation_sha256": file_sha256(Path(config["attempt_reservation"])),
        "attempted_forwards": 0, "reserved_forwards": reservation["requested_attempts"],
        "attempt_statuses": [{**item, "status": "reserved"} for item in reservation["attempts"]],
        "runtime_package_verification": runtime_verified,
        "nf4_initialization_generated_proof": nf4_proof,
        "seed": seed, "validation_comparison": "pending", "serving_load_verified": False}
    _write_json(output / "run.json", run)
    _write_json(output / "config.json", config)
    def begin_forward() -> None:
        index = run["attempted_forwards"]
        require(index < len(run["attempt_statuses"]), "silver_unreserved_forward_forbidden")
        run["attempt_statuses"][index]["status"] = "attempted"
        run["attempted_forwards"] += 1
        _write_json(output / "run.json", run)
    def complete_forward() -> None:
        run["attempt_statuses"][run["attempted_forwards"] - 1]["status"] = "completed"
    def measure_runtime() -> None:
        run["elapsed_seconds"] = time.monotonic() - started
        run["observed_at"] = datetime.now(timezone.utc).isoformat()
        try:
            if torch.cuda.is_initialized():
                run["gpu_initialized"] = True
                run["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
                run["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
        except Exception:
            run["vram_measurement_reason_code"] = "silver_cuda_memory_measurement_unavailable"
    def stage(name: str) -> None:
        run["stage"] = name
        measure_runtime()
        _write_json(output / "run.json", run)
    def persist_frozen_difference(before: dict, after: dict) -> dict:
        difference = compare_frozen_snapshots(before, after)
        _write_json(proof_dir / "frozen.difference.json", difference)
        run["frozen_proof"] = {"scope": before["scope"], "parameters_checked": difference["parameters_checked"],
            "before_parameters_sha256": before["parameters_sha256"], "after_parameters_sha256": after["parameters_sha256"],
            "changed_parameter_count": difference["changed_parameter_count"],
            "missing_parameter_count": len(difference["missing_names"]), "added_parameter_count": len(difference["added_names"]),
            "before_receipt_sha256": file_sha256(proof_dir / "frozen.before.json"),
            "after_receipt_sha256": file_sha256(proof_dir / "frozen.after.json"),
            "difference_receipt_sha256": file_sha256(proof_dir / "frozen.difference.json"),
            "difference_path": "proof/frozen.difference.json"}
        return difference
    model = None
    frozen_before = frozen_after = None
    try:
        torch.cuda.reset_peak_memory_stats()
        stage("base_load")
        model, processor = load_model_and_processor(config, model_dir, processor_dir)
        runtime_load = model._forte_runtime_load_receipt
        check_time()
        stage("adapter_setup")
        model, trainable = configure_adapter(model, config)
        require(all("lora_" in name for name, _ in trainable), "silver_non_adapter_optimizer_parameter")
        if config["quantization"] == "nf4":
            stage("nf4_bias_initialization")
            initialization = canonicalize_nf4_bias_storage(model, policy=config["nf4_bias_initialization_policy"])
            _write_json(proof_dir / "nf4-bias.initialization.json", initialization)
            run["nf4_bias_initialization"] = {"policy": initialization["policy"],
                "biases_examined": initialization["biases_examined"],
                "biases_storage_changed": initialization["biases_storage_changed"],
                "all_values_exactly_preserved": initialization["all_values_exactly_preserved"],
                "receipt_sha256": file_sha256(proof_dir / "nf4-bias.initialization.json")}
        stage("frozen_before_snapshot")
        initial_adapter = parameter_hashes(model, trainable=True)
        frozen_before = frozen_snapshot(model)
        _write_json(proof_dir / "frozen.before.json", frozen_before)
        stage("collator_optimizer_setup")
        collator = SilverCompletionCollator(processor, Path(config["media_root"]).resolve(),
            config["max_length"], max_pixels=config["max_pixels_per_image"], model_type="qwen3_5")
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable],
                                     lr=config["learning_rate"], weight_decay=0.01)
        accumulation = 1 if smoke else config["gradient_accumulation_steps"]
        microstep_budget = min(len(rows), (2 if smoke else config["max_steps"]) * accumulation)
        require(math.ceil(microstep_budget / accumulation) >= 2, "silver_two_steps_unavailable")
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        run.update(gpu_executed=True, trainable_parameters=sum(p.numel() for _, p in trainable),
            adapted_module_inventory=model._forte_lora_target_inventory, runtime_load=runtime_load,
            versions={package: importlib.metadata.version(package) for package in ("torch", "transformers", "peft")},
            gpu_name=torch.cuda.get_device_name(0), frozen_parameter_hashes_sha256=frozen_before["parameters_sha256"])
        check_time()
        with (output / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
            optimizer_group_started = time.monotonic()
            for microstep, index in enumerate(order[:microstep_budget], start=1):
                check_time()
                if (microstep - 1) % accumulation == 0:
                    optimizer_group_started = time.monotonic()
                stage("training_batch_prepare")
                batch, token_weights = collate_objective_batch(collator, rows[index], device="cuda", balanced=balanced)
                group_start = ((microstep - 1) // accumulation) * accumulation
                group_size = min(accumulation, microstep_budget - group_start)
                stage("training_forward")
                begin_forward()
                forward_started = time.monotonic()
                loss = objective_loss(model, batch, token_weights, example_count=len(rows))
                require(bool(torch.isfinite(loss).item()), "silver_loss_nonfinite")
                run["losses"].append({"microstep": microstep, "loss": float(loss.item()),
                                      "forward_wall_seconds": time.monotonic() - forward_started})
                run["finite_loss_verified"] = True
                complete_forward()
                stage("training_backward")
                (loss / group_size).backward()
                run["microsteps"] = microstep
                if microstep % accumulation == 0 or microstep == microstep_budget:
                    stage("training_gradient_check")
                    gradients = [parameter.grad for _, parameter in trainable if parameter.grad is not None]
                    require(bool(gradients) and all(bool(torch.isfinite(g).all().item()) for g in gradients)
                            and any(bool(torch.count_nonzero(g).item()) for g in gradients),
                            "silver_gradient_missing_zero_or_nonfinite")
                    require(not any(p.grad is not None for p in model.parameters() if not p.requires_grad),
                            "silver_frozen_parameter_received_gradient")
                    from .architectures import validate_adapter_gradients
                    validate_adapter_gradients(model, inventory=model._forte_lora_target_inventory)
                    run["gradients_verified"] = True
                    norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                    require(bool(torch.isfinite(norm).item()), "silver_gradient_norm_nonfinite")
                    stage("optimizer_step")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    run["optimizer_steps"] += 1
                    step_metrics = {"step": run["optimizer_steps"], "microsteps": microstep,
                        "loss_last_microbatch": float(loss.item()), "gradient_norm": float(norm.item()),
                        "elapsed_seconds": time.monotonic() - started,
                        "optimizer_group_wall_seconds": time.monotonic() - optimizer_group_started,
                        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()}
                    run["optimizer_step_metrics"].append(step_metrics)
                    metrics.write(json.dumps(step_metrics) + "\n")
                    metrics.flush()
                    _write_json(output / "run.json", run)
        check_time()
        require(run["optimizer_steps"] >= 2 and (not smoke or run["optimizer_steps"] == 2),
                "silver_real_optimizer_steps_missing")
        stage("adapter_change_proof")
        changed_names = [name for name, digest in parameter_hashes(model, trainable=True).items()
                         if initial_adapter[name] != digest]
        changed = len(changed_names)
        changed_groups = {group: any("." + group + "." in name for name in changed_names)
                          for group in ("self_attn", "linear_attn", "mlp")}
        require(changed > 0 and all(changed_groups.values()), "silver_adapter_group_tensors_unchanged")
        stage("frozen_after_snapshot")
        frozen_after = frozen_snapshot(model)
        _write_json(proof_dir / "frozen.after.json", frozen_after)
        difference = persist_frozen_difference(frozen_before, frozen_after)
        stage("frozen_after_compare")
        require(difference["unchanged"], "silver_frozen_tensors_changed")
        run["frozen_tensors_unchanged"] = True
        stage("adapter_serialize")
        adapter = output / "adapter"
        for peft_config in model.peft_config.values():
            peft_config.revision = config["base_revision"]
            peft_config.base_model_name_or_path = config["model_id"]
        model.save_pretrained(adapter, safe_serialization=True)
        stage("serialized_adapter_verify")
        from peft import PeftConfig, load_peft_weights, get_peft_model_state_dict
        PeftConfig.from_pretrained(adapter, local_files_only=True)
        reloaded = load_peft_weights(str(adapter), device="cpu", local_files_only=True)
        require(bool(reloaded) and all(bool(torch.isfinite(tensor).all().item()) for tensor in reloaded.values()),
                "silver_saved_adapter_invalid")
        def exact_state(actual: dict, expected: dict) -> bool:
            return actual.keys() == expected.keys() and all(actual[key].shape == expected[key].shape
                and actual[key].dtype == expected[key].dtype
                and torch.equal(actual[key].detach().cpu(), expected[key].detach().cpu()) for key in actual)
        require(exact_state(get_peft_model_state_dict(model, adapter_name="default"), reloaded),
                "silver_serialized_adapter_tensor_mismatch")
        model.eval()
        stage("roundtrip_before_forward")
        batch, token_weights = collate_objective_batch(collator, rows[order[0]], device="cuda", balanced=balanced)
        with torch.no_grad():
            begin_forward()
            expected_loss = objective_loss(model, batch, token_weights, example_count=len(rows)).detach().float()
            require(bool(torch.isfinite(expected_loss).item()), "silver_presave_reload_loss_nonfinite")
            complete_forward()
        stage("roundtrip_adapter_load")
        model.load_adapter(str(adapter), adapter_name="silver_roundtrip", is_trainable=False, local_files_only=True)
        model.set_adapter("silver_roundtrip")
        model.eval()
        require(exact_state(get_peft_model_state_dict(model, adapter_name="silver_roundtrip"), reloaded),
                "silver_reloaded_adapter_tensor_mismatch")
        stage("roundtrip_after_forward")
        with torch.no_grad():
            begin_forward()
            actual_loss = objective_loss(model, batch, token_weights, example_count=len(rows)).detach().float()
        require(bool(torch.isfinite(actual_loss).item()) and bool(torch.allclose(expected_loss, actual_loss, atol=1e-4, rtol=1e-4)),
                "silver_reload_forward_mismatch")
        complete_forward()
        stage("receipt_finalize")
        checksums = {path.relative_to(adapter).as_posix(): file_sha256(path)
                     for path in sorted(adapter.rglob("*")) if path.is_file()}
        check_time()
        run.update(status="candidate_unselected", finite_loss_verified=True, gradients_verified=True,
            changed_adapter_tensors=changed, frozen_tensors_unchanged=True, peft_load_verified=True,
            changed_adapter_groups=changed_groups, serialized_adapter_tensors_exact=True,
            reload_forward_verified=True, adapter_checksums=checksums,
            adapter_tree_sha256=canonical_sha256(checksums), peak_vram_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic() - started, completed_at=datetime.now(timezone.utc).isoformat())
        stage("completed")
        _write_json(output / "run.json", run)
        return {key: run[key] for key in ("status", "mode", "optimizer_steps", "microsteps", "adapter_tree_sha256",
                                         "human_verified", "selection")}
    except Exception as error:
        failure_stage = run["stage"]
        reason_code = safe_failure_code(error)
        run.update(status="failed", reason_code=reason_code, failure_stage=failure_stage,
                   selection="unselected", serving_load_verified=False)
        if reason_code in {"silver_loss_nonfinite", "silver_presave_reload_loss_nonfinite", "silver_reload_forward_mismatch"}:
            run["finite_loss_verified"] = False
        if model is not None and frozen_before is not None and frozen_after is None:
            try:
                frozen_after = frozen_snapshot(model)
                _write_json(proof_dir / "frozen.after.json", frozen_after)
                persist_frozen_difference(frozen_before, frozen_after)
            except Exception as diagnostic_error:
                run["failure_frozen_snapshot_reason_code"] = safe_failure_code(diagnostic_error)
        if model is not None and run["optimizer_steps"] > 0:
            try:
                quarantine = save_quarantined_adapter(model, output, config, reason_code=reason_code,
                    failure_stage=failure_stage, optimizer_steps=run["optimizer_steps"])
                _write_json(output / "quarantine" / "receipt.json", quarantine)
                run["quarantine"] = {"status": "quarantined", "serving_eligible": False,
                    "receipt_sha256": file_sha256(output / "quarantine" / "receipt.json"),
                    "adapter_tree_sha256": quarantine["adapter_tree_sha256"]}
            except Exception as quarantine_error:
                run["quarantine"] = {"status": "save_failed", "serving_eligible": False,
                                     "reason_code": safe_failure_code(quarantine_error)}
        for item in run["attempt_statuses"]:
            if item["status"] == "attempted":
                item["status"] = "failed"
            elif item["status"] == "reserved":
                item["status"] = "cancelled_before_attempt"
        run.update(status="failed", elapsed_seconds=time.monotonic() - started,
                   completed_at=datetime.now(timezone.utc).isoformat())
        _write_json(output / "run.json", run)
        raise
    finally:
        measure_runtime()
        _write_json(output / "run.json", run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="Exactly two actual optimizer steps and save/reload")
    mode.add_argument("--prepare-media", action="store_true", help="Copy hash-verified TRAIN JPEG bytes into a new private mount")
    mode.add_argument("--prepare-supervision", action="store_true", help="Create one new private balanced objective plan; no GPU")
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        result = (prepare_media(config) if args.prepare_media else prepare_supervision(config) if args.prepare_supervision
                  else train(config, preflight_only=args.preflight, smoke=args.smoke))
        print(json.dumps(result, allow_nan=False))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "silver_local_training_error",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    sys.exit(main())
