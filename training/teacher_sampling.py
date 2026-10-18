"""Optional private, source-bound physical-fact teacher annotation.

Planning reads metadata only. ``run`` requires a plan-bound execution release
after the shared GPU owner grants the annotation slot. The production API is
untouched. Outputs are unreviewed local-model proposals, never ground truth.

Each second gets native-PTS focal samples and surrounding procedure context.
Fact semantics retain the entire request interval; focal metadata does NOT
turn these observations into exclusive one-second or dense frame labels.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any

from training.guards import canonical_sha256, file_sha256, load_json, local_file, require

PLAN_SCHEMA = "forte.second-teacher-plan.v1"
WORKLIST_SCHEMA = "forte.autonomous-annotation-worklist.v1"
RELEASE_SCHEMA = "forte.teacher-execution-release.v1"
ANCHOR_PHYSICAL_FACT_IDS = (
    "CASH_VISIBLE", "CASH_ON_WORK_SURFACE", "CASH_HANDLER_PRESENT",
    "CASH_HANDLER_LEAVES_WORKSTATION", "CASH_HANDOVER_VISIBLE", "CASH_COUNT_VISIBLE",
    "CASH_PLACED_IN_CLOTHING_POCKET", "CASH_PLACED_IN_PERSONAL_CONTAINER",
)


@dataclass(frozen=True)
class TeacherSamplingConfig:
    context_before_sec: float = 4.0
    context_after_sec: float = 4.0
    context_fps: float = 1.0
    focal_fps: float = 4.0
    max_frames: int = 16
    max_seconds_per_record: int = 600
    max_decode_frames: int = 8192
    anchor_frame_questions: bool = True
    trusted_scene_context: str | None = None

    def __post_init__(self) -> None:
        require(all(math.isfinite(value) and value >= 0 for value in (
            self.context_before_sec, self.context_after_sec)), "teacher_context_invalid")
        require(all(math.isfinite(value) and value > 0 for value in (
            self.context_fps, self.focal_fps)), "teacher_sampling_rate_invalid")
        require(all(type(value) is int and value > 0 for value in (
            self.max_frames, self.max_seconds_per_record, self.max_decode_frames)), "teacher_budget_invalid")
        require(self.max_frames <= 64, "teacher_adapter_frame_limit")
        require(type(self.anchor_frame_questions) is bool, "teacher_anchor_question_flag_invalid")
        require(self.trusted_scene_context in {None, "bank_cash_workstation"}, "teacher_scene_context_invalid")


@dataclass(frozen=True)
class TeacherSecondWindow:
    index: int
    focal_start_sec: float
    focal_end_sec: float
    anchor_target_sec: float
    start_sec: float
    end_sec: float
    timestamps_sec: tuple[float, ...]


def _span_samples(start: float, end: float, fps: float) -> list[float]:
    count = max(1, math.ceil((end - start) * fps))
    return [start + (end - start) * index / count for index in range(count + 1)]


def plan_second_windows(duration_sec: float, config: TeacherSamplingConfig | None = None) -> list[TeacherSecondWindow]:
    """Cover every source second and the final partial second, without pruning."""
    config = config or TeacherSamplingConfig()
    require(isinstance(duration_sec, (int, float)) and not isinstance(duration_sec, bool)
            and math.isfinite(duration_sec) and duration_sec > 0, "teacher_duration_invalid")
    count = math.ceil(duration_sec)
    require(count <= config.max_seconds_per_record, "teacher_second_budget_insufficient")
    result = []
    for second in range(count):
        focal_start, focal_end = float(second), min(float(second + 1), duration_sec)
        start = max(0.0, focal_start - config.context_before_sec)
        end = min(duration_sec, focal_end + config.context_after_sec)
        anchor = (focal_start + focal_end) / 2
        points = sorted(set(_span_samples(start, end, config.context_fps)
                            + _span_samples(focal_start, focal_end, config.focal_fps) + [anchor]))
        require(len(points) <= config.max_frames, "teacher_window_frame_budget_insufficient")
        result.append(TeacherSecondWindow(second, focal_start, focal_end, anchor, start, end, tuple(points)))
    require(len({point for item in result for point in item.timestamps_sec}) <= config.max_decode_frames,
            "teacher_decode_budget_insufficient")
    return result


def _assignments(split: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = split.get("records")
    require(isinstance(records, list) and bool(records), "teacher_split_records_missing")
    # A02's digest deliberately hashes its exact sorted/indented JSON form.
    expected = hashlib.sha256((json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    require(split.get("assignments_sha256") == expected, "teacher_assignments_hash_mismatch")
    indexed: dict[str, dict[str, Any]] = {}
    groups: dict[str, str] = {}
    for row in records:
        content = row.get("content_sha256", row.get("content_hash"))
        partition = row.get("split")
        group = row.get("scenario_group_id") or row.get("group_id")
        require(isinstance(content, str) and re.fullmatch(r"[a-f0-9]{64}", content) is not None,
                "teacher_content_identity_invalid")
        require(partition in {"train", "validation", "final_check"} and isinstance(group, str) and bool(group),
                "teacher_partition_or_group_invalid")
        require(group not in groups or groups[group] == partition, "teacher_scenario_split_leakage")
        require(content not in indexed or indexed[content]["split"] == partition,
                "teacher_duplicate_split_leakage")
        groups[group] = partition
        indexed[content] = {"split": partition, "scenario_group_id": group}
    return indexed


def build_teacher_plan(worklist: dict[str, Any], split: dict[str, Any], *, split_sha256: str,
                       worklist_sha256: str, partition: str = "train",
                       config: TeacherSamplingConfig | None = None) -> dict[str, Any]:
    """Project a metadata worklist onto exactly one allowed frozen partition.

    Binary labels, original filenames, and annotations are never projected into
    the observation request. Source paths are private I/O metadata only.
    """
    config = config or TeacherSamplingConfig()
    require(partition in {"train", "validation"}, "teacher_final_or_unknown_partition_forbidden")
    require(worklist.get("schema_version") == WORKLIST_SCHEMA, "teacher_worklist_schema_invalid")
    require(worklist.get("split_file_sha256") == split_sha256, "teacher_split_file_binding_changed")
    require(worklist.get("assignments_sha256") == split.get("assignments_sha256"),
            "teacher_worklist_assignments_changed")
    assignments = _assignments(split)
    selected = []
    seen = set()
    for row in worklist.get("records", []):
        content = row.get("content_sha256")
        frozen = assignments.get(content)
        require(frozen is not None and row.get("split") == frozen["split"], "teacher_worklist_partition_mismatch")
        require(row.get("scenario_group_id") == frozen["scenario_group_id"], "teacher_worklist_group_mismatch")
        if frozen["split"] != partition:
            continue
        require(content not in seen, "teacher_duplicate_worklist_content")
        seen.add(content)
        duration = row.get("duration_sec")
        windows = plan_second_windows(duration, config)
        require(isinstance(row.get("source_path"), str) and bool(row["source_path"]), "teacher_source_path_missing")
        selected.append({"sample_id": row["sample_id"], "content_sha256": content,
                         "scenario_group_id": frozen["scenario_group_id"], "split": partition,
                         "source_path": row["source_path"], "duration_sec": duration,
                         "second_count": len(windows)})
    require(bool(selected), "teacher_partition_worklist_empty")
    result = {"schema_version": PLAN_SCHEMA, "partition": partition, "records": selected,
              "split_file_sha256": split_sha256, "worklist_sha256": worklist_sha256,
              "assignments_sha256": split["assignments_sha256"], "sampling": asdict(config),
              "organizer_labels_used_for_facts": False, "training_authorized": False,
              "annotation_origin": "local_model", "human_reviewed": False,
              "fact_scope": "full_request_context_with_second_focal_samples",
              "native_frame_timing_claim": "actual_sampled_source_pts_only_not_dense_labels"}
    result["plan_sha256"] = canonical_sha256(result)
    return result


def validate_teacher_plan(plan: dict[str, Any], split: dict[str, Any], *, split_sha256: str) -> None:
    require(plan.get("schema_version") == PLAN_SCHEMA, "teacher_plan_schema_invalid")
    require(plan.get("plan_sha256") == canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"}),
            "teacher_plan_hash_changed")
    require(plan.get("split_file_sha256") == split_sha256
            and plan.get("assignments_sha256") == split.get("assignments_sha256"), "teacher_frozen_split_changed")
    require(plan.get("partition") in {"train", "validation"}, "teacher_final_partition_forbidden")
    require(set(plan.get("sampling", {})) == set(asdict(TeacherSamplingConfig())), "teacher_sampling_contract_changed")
    assignments = _assignments(split)
    require(bool(plan.get("records")), "teacher_plan_records_missing")
    seen = set()
    for record in plan["records"]:
        content = record.get("content_sha256")
        frozen = assignments.get(content)
        require(content not in seen and frozen is not None and frozen["split"] == plan["partition"]
                and record.get("split") == frozen["split"]
                and record.get("scenario_group_id") == frozen["scenario_group_id"], "teacher_plan_source_partition_changed")
        seen.add(content)
        require(record.get("second_count") == len(plan_second_windows(record.get("duration_sec"),
                TeacherSamplingConfig(**plan["sampling"]))), "teacher_plan_second_count_changed")


def validate_execution_release(release: dict[str, Any], plan: dict[str, Any]) -> None:
    """The orchestrator grants this only while it owns the serialized GPU slot."""
    require(release.get("schema_version") == RELEASE_SCHEMA and release.get("released") is True,
            "teacher_execution_not_released")
    require(release.get("plan_sha256") == plan.get("plan_sha256"), "teacher_release_plan_mismatch")
    require(isinstance(release.get("reference"), str) and bool(release["reference"].strip())
            and release.get("purpose") == "source_bound_teacher_annotation", "teacher_release_reference_missing")
    expires = release.get("expires_at_unix")
    require(isinstance(expires, (int, float)) and not isinstance(expires, bool)
            and math.isfinite(expires) and expires > time.time(), "teacher_execution_release_expired")


def validate_exposure_receipt(receipt: dict[str, Any], plan: dict[str, Any], records: list[dict[str, Any]],
                              release: dict[str, Any], *, source_manifest_sha256: str) -> None:
    """Allow remote execution after the orchestrator reserved CENTRAL exposures.

    The private receipt is prepared from the authoritative registry's successful
    append, before uploading this run's bundle. The remote runner must not
    mutate an isolated registry copy and pretend that it is the central union.
    """
    require(receipt.get("schema_version") == "forte.teacher-exposure-receipt.v1"
            and receipt.get("registration_origin") == "authoritative_central_exposure_registry"
            and receipt.get("ok") is True, "teacher_central_exposure_receipt_invalid")
    require(receipt.get("plan_sha256") == plan["plan_sha256"]
            and receipt.get("assignments_sha256") == plan["assignments_sha256"]
            and receipt.get("split_file_sha256") == plan["split_file_sha256"]
            and receipt.get("source_manifest_sha256") == source_manifest_sha256
            and receipt.get("execution_release_reference") == release["reference"]
            and receipt.get("execution_release_sha256") == canonical_sha256(release),
            "teacher_central_exposure_receipt_binding_changed")
    required = {record["content_sha256"] for record in records}
    registered = receipt.get("registered_content_sha256", [])
    require(isinstance(registered, list) and required <= set(registered)
            and receipt.get("registered_content_set_sha256") == canonical_sha256(sorted(registered))
            and receipt.get("allowed_splits") == ["train"]
            and re.fullmatch(r"[a-f0-9]{64}", str(receipt.get("registry_tip_sha256", ""))) is not None,
            "teacher_central_exposure_receipt_contents_missing")


def prioritize_train_anchors(plan: dict[str, Any], anchors: dict[str, Any], *, anchors_sha256: str) -> dict[str, Any]:
    """Use supplied coarse timing only to schedule train observations first.

    No supplied binary label, claimed event type or legal answer is retained.
    The teacher still receives only original images and generic fact questions.
    Validation/final timing is forbidden as an inference-conditioning input.
    """
    require(plan.get("partition") == "train", "teacher_validation_anchor_conditioning_forbidden")
    require(anchors.get("split_file_sha256") == plan["split_file_sha256"]
            and anchors.get("assignments_sha256") == plan["assignments_sha256"], "teacher_anchor_split_binding_changed")
    by_content = {}
    for row in anchors.get("records", []):
        require(row.get("split") == "train", "teacher_anchor_partition_forbidden")
        content = row.get("content_sha256")
        require(content not in by_content, "teacher_duplicate_anchor_content")
        values = row.get("coarse_anchor_sec", [])
        require(isinstance(values, list) and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0 for value in values), "teacher_coarse_anchor_invalid")
        by_content[content] = values
    updated = {**plan, "records": [dict(record) for record in plan["records"]]}
    for record in updated["records"]:
        values = by_content.get(record["content_sha256"], [])
        require(all(value <= record["duration_sec"] for value in values), "teacher_anchor_outside_source")
        priority = []
        for anchor in values:
            center = min(record["second_count"] - 1, math.floor(anchor))
            for index in (center, center - 1, center + 1):
                if 0 <= index < record["second_count"] and index not in priority:
                    priority.append(index)
        record["priority_second_indices"] = priority
        record["inspection_focus"] = {"origin": "provided_coarse_event_anchor" if values else "uniform_seconds",
            "supplied_anchor_sec": values, "independent_localization": False,
            "fact_scope": "full_request_context_with_cited_frame_support", "focal_exclusive_labels": False}
    # Anchor-bearing train records are inspected before unlabeled timing records.
    updated["records"].sort(key=lambda record: not bool(record["priority_second_indices"]))
    updated["anchor_manifest_sha256"] = anchors_sha256
    updated["plan_sha256"] = canonical_sha256({key: value for key, value in updated.items() if key != "plan_sha256"})
    return updated


def load_train_anchors(path: Path, pins_path: Path) -> dict[str, Any]:
    """Read only the pinned train export; validation/final files never condition requests."""
    pins = load_json(pins_path)
    require(pins.get("schema_version") == "forte.user-coarse-anchor-pins.v1"
            and pins.get("target_scope") == "class_agnostic_approximate_event_anchor", "teacher_anchor_pins_invalid")
    expected = pins.get("role_manifests", {}).get("train", {})
    require(file_sha256(path) == expected.get("sha256"), "teacher_train_anchor_manifest_hash_changed")
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, ValueError):
        require(False, "teacher_train_anchor_manifest_invalid")
    require(len(records) == expected.get("records"), "teacher_train_anchor_count_changed")
    return {"split_file_sha256": pins["split_file_sha256"],
            "assignments_sha256": pins["assignments_sha256"], "records": records}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bind_teacher_requests(job_id: str, frames: list, windows: list[TeacherSecondWindow], facts: list) -> list:
    """Use only A05 references; full-scene source frames are never fabricated."""
    from forte.video.sampling import CoverageWindow, ProcessorBudget, make_clip_requests
    planned = [CoverageWindow(window.index, window.start_sec, window.end_sec,
                              window.timestamps_sec, "teacher-second") for window in windows]
    maximum = max(len(window.timestamps_sec) for window in windows)
    return make_clip_requests(job_id, frames, planned, fact_definitions=facts, rule_definitions=[],
                              budget=ProcessorBudget(max_frames=maximum))


def with_anchor_questions(request: Any, anchor: Any) -> Any:
    """Ask separate point-localized questions; never reuse a window state.

    The alias is produced by A08's authoritative request-local mapping, not by
    a filename or supplied event answer. Source IDs are restored before storage.
    """
    from forte.decision.aliasing import compact_frame_request
    from forte.shared.schemas import ClipRequest, FactDefinition
    aliases = compact_frame_request(request)
    alias = aliases.canonical_to_alias[anchor.frame_id]
    full_definitions = {fact.fact_id: fact for fact in request.fact_definitions}
    additions = []
    for fact_id in ANCHOR_PHYSICAL_FACT_IDS:
        require(fact_id in full_definitions, "teacher_anchor_fact_definition_missing")
        description = (
            f"INDEPENDENT EXACT-FRAME QUESTION: evaluate physical fact {fact_id} at frame {alias}, "
            f"timestamp {anchor.pts_sec:.9f} seconds ONLY. Other images are temporal context. "
            "For actions, present means the action is visibly in progress at this exact frame; "
            "an action occurring elsewhere in the context is not present at the anchor. "
            "For static facts evaluate visible objects/persons at the anchor. "
            "Return unclear if the exact moment, object, action or visibility cannot be established. "
            "A decisive state must cite the anchor frame; additional contextual frame supports are allowed. "
            "Absence here concerns only this visible frame, never an omitted procedure or the whole window. "
            "Do not copy the full-context fact's state. Generic criterion: " + full_definitions[fact_id].description
        )
        additions.append(FactDefinition(fact_id="ANCHOR__" + fact_id, description=description))
    values = request.model_dump(mode="python")
    values["fact_definitions"] = list(request.fact_definitions) + additions
    return ClipRequest.model_validate(values)


def extract_anchor_targets(batch: Any, anchor: Any) -> list[dict[str, Any]]:
    """Preserve teacher answers and masks; no point state is inferred from support."""
    targets = []
    questions = {question.fact_id: question for question in batch.request.fact_definitions}
    for fact in batch.facts:
        if not fact.fact_id.startswith("ANCHOR__"):
            continue
        fact_id = fact.fact_id.removeprefix("ANCHOR__")
        require(fact_id in ANCHOR_PHYSICAL_FACT_IDS, "teacher_unknown_anchor_fact")
        require(fact.fact_id in questions, "teacher_anchor_question_definition_missing")
        question_definition = questions[fact.fact_id].model_dump(mode="json")
        supported = fact.state == "unclear" or anchor.frame_id in fact.supporting_frame_ids
        targets.append({"frame_id": anchor.frame_id, "pts_sec": anchor.pts_sec, "fact_id": fact_id,
            "state": fact.state, "supporting_frame_ids": list(fact.supporting_frame_ids),
            "context_frame_ids": [frame.frame_id for frame in batch.request.frame_refs],
            "derivation": "direct_anchor_teacher_question", "scope": "exact_anchor_frame",
            "question_fact_id": fact.fact_id, "annotation_origin": "local_model",
            "question_definition": question_definition,
            "review_status": "machine_consistency_checked", "localizable_category": "physical_anchor_atom",
            "provenance": {"prompt_sha256": batch.prompt_hash, "prompt_version": batch.prompt_version,
                           "config_version": batch.config_version,
                           "focal_question_sha256": canonical_sha256(question_definition)},
            "human_verified": False, "eligible": supported,
            "mask_reason": None if supported else "decisive_anchor_answer_did_not_cite_anchor"})
    return targets


def summarize_teacher_windows(windows: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate-only diagnostics suitable for the orchestration chat."""
    result = {"valid_teacher_windows": len(windows), "window_fact_decisions": 0,
              "window_unclear_facts": 0, "frame_fact_targets": 0, "frame_target_present": 0,
              "frame_target_absent": 0, "frame_target_unclear": 0, "eligible_frame_targets": 0,
              "masked_frame_targets": 0, "runtime_attested_windows": 0}
    for window in windows:
        result["window_fact_decisions"] += len(window["facts"])
        result["window_unclear_facts"] += sum(fact["state"] == "unclear" for fact in window["facts"])
        result["runtime_attested_windows"] += int(window.get("runtime_snapshot_verified") is True)
        for target in window.get("frame_fact_targets", []):
            result["frame_fact_targets"] += 1
            result["frame_target_" + target["state"]] += 1
            result["eligible_frame_targets"] += int(target["eligible"] is True)
            result["masked_frame_targets"] += int(target["eligible"] is not True)
    return result


async def run_teacher_record(record: dict[str, Any], plan: dict[str, Any], *, source_root: Path,
                             output_root: Path, provider: Any, manifest: dict[str, Any],
                             manifest_sha256: str, release: dict[str, Any], deadline: float,
                             cancel: threading.Event | None = None,
                             max_model_requests: int | None = None,
                             source_by_hash: dict[str, str] | None = None,
                             anchor_focus_only: bool = False) -> dict[str, int]:
    """Run one private record, checkpointing validated observations per second.

    Caller validates the whole plan against the frozen split before this entry.
    A05 probes/hash-checks source media and performs one union sparse decode.
    The provider is A07's private client; injection is only for synthetic tests.
    """
    from forte.decision import DecisionConfig, DecisionEngine
    from forte.decision.structured import DecisionOutputError
    from forte.model import InvalidModelOutput
    from forte.video.decode import decode_frames, validate_frame_ref
    from forte.video.probe import probe_video
    from rules.catalog import load_catalog

    validate_execution_release(release, plan)
    require(record in plan["records"] and record["split"] == plan["partition"], "teacher_record_not_in_plan")
    require(plan["partition"] == "train", "teacher_runner_train_only")
    require(max_model_requests is None or type(max_model_requests) is int and max_model_requests > 0,
            "teacher_model_request_limit_invalid")
    cancel = cancel or threading.Event()
    config = TeacherSamplingConfig(**plan["sampling"])
    source_path = (source_by_hash or {}).get(record["content_sha256"], record["source_path"])
    source = local_file(source_path, source_root.resolve())
    job_id = "teacher-" + canonical_sha256({"content": record["content_sha256"], "plan": plan["plan_sha256"],
                                          "anchor_focus_only": anchor_focus_only})[:24]
    job_root = output_root.resolve() / plan["partition"] / job_id
    audit_path = job_root / "teacher.private.json"
    asset = await asyncio.to_thread(probe_video, source, job_id=job_id,
                                     deadline_monotonic=deadline, cancel=cancel)
    require(asset.content_hash == record["content_sha256"], "teacher_source_hash_changed")
    require(abs(asset.duration_sec - record["duration_sec"]) <= max(.1, 1 / (asset.fps or 30)),
            "teacher_source_duration_changed")
    windows = plan_second_windows(asset.duration_sec, config)
    priority = record.get("priority_second_indices", [])
    require(isinstance(priority, list) and len(priority) == len(set(priority))
            and all(type(index) is int and 0 <= index < len(windows) for index in priority),
            "teacher_priority_indices_invalid")
    selected_windows = [windows[index] for index in priority] if anchor_focus_only else windows
    require(bool(selected_windows), "teacher_anchor_focus_has_no_supplied_train_anchors")
    targets = sorted({point for window in selected_windows for point in window.timestamps_sec})
    frames = await asyncio.to_thread(decode_frames, asset, job_root / "frames", timestamps_sec=targets,
                                      max_frames=config.max_decode_frames,
                                      deadline_monotonic=deadline, cancel=cancel)
    for frame in frames:
        validate_frame_ref(asset, frame)
    # These are generic physical/task definitions only. No rule or event catalog
    # is sent to the teacher, and no organizer labels are loaded by this runner.
    facts = load_catalog().fact_definitions(trusted_scene_context=config.trusted_scene_context)
    requests = bind_teacher_requests(job_id, frames, selected_windows, facts)
    pairs = dict(zip((window.index for window in selected_windows), zip(selected_windows, requests, strict=True), strict=True))
    ordered_indices = priority + [index for index in pairs if index not in priority]
    engine = DecisionEngine(provider, reviewed_event_ids=(), event_definitions={},
                            config=DecisionConfig(compact_frame_ids=True, concise_observations=True))
    completed: list[dict[str, Any]] = []
    invalid_windows: list[dict[str, Any]] = []
    checkpoints = job_root / "windows"
    provenance = {"model_id": manifest["model_id"], "model_revision": manifest["revision"],
                  "processor_revision": manifest["processor_revision"], "model_manifest_sha256": manifest_sha256,
                  "execution_location": "team_deployed_service", "external_ai_used": False,
                  "local_run_id": job_id, "plan_sha256": plan["plan_sha256"],
                  "execution_release_reference": release["reference"], "actual_snapshot_verified": False}

    def audit(phase: str, error_code: str | None = None) -> None:
        provenance["actual_snapshot_verified"] = bool(completed) and all(
            window.get("runtime_snapshot_verified") is True for window in completed)
        completed_indices = {window["second_index"] for window in completed}
        missing = [window.index for window in windows if window.index not in completed_indices]
        _write(audit_path, {"schema_version": "forte.local-proposal.v1", "phase": phase,
            "job_id": job_id, "sample_id": record["sample_id"], "split": record["split"],
            "scenario_group_id": record["scenario_group_id"], "asset": asset.model_dump(mode="json"),
            "frames": [frame.model_dump(mode="json") for frame in frames], "window_decisions": completed,
            "decisions": [fact for window in completed for fact in window["facts"]],
            "candidate_events": [], "verified_events": [], "events": [], "publication_ready": False,
            "annotation_origin": "local_model", "human_reviewed": False, "training_authorized": False,
            "invalid_teacher_windows": invalid_windows,
            "organizer_labels_used_for_facts": False, "provenance": provenance, "error_code": error_code,
            "coverage": {"duration_sec": asset.duration_sec, "complete": not missing,
                         "policy": "per_second_focal_samples_plus_full_context",
                         "planned_second_count": len(windows), "completed_second_count": len(completed),
                         "missing_second_indices": missing, "anchor_focus_only": anchor_focus_only,
                         "attempted_model_requests": model_requests},
            "limitations": ["teacher_proposals_are_not_independent_ground_truth",
                            "focal_samples_do_not_establish_dense_native_frame_labels",
                            "request_coverage_does_not_establish_continuous_visibility"]})

    resumed = 0
    model_requests = 0
    try:
        for index in ordered_indices:
            window, request = pairs[index]
            validate_execution_release(release, plan)
            require(not cancel.is_set() and time.monotonic() < deadline, "teacher_deadline_or_cancellation")
            checkpoint = checkpoints / f"{window.index:06d}.json"
            anchor_candidates = [frame for frame in request.frame_refs
                                 if window.focal_start_sec <= frame.pts_sec < window.focal_end_sec
                                 or window.focal_end_sec == asset.duration_sec and frame.pts_sec == window.focal_end_sec]
            require(bool(anchor_candidates), "teacher_focal_bin_has_no_native_anchor")
            anchor = min(anchor_candidates, key=lambda frame: abs(frame.pts_sec - window.anchor_target_sec))
            # A supplied train anchor can choose which real frame receives the
            # independent point question. It never supplies the answer or
            # establishes that an action actually occurred there.
            supplied = record.get("inspection_focus", {}).get("supplied_anchor_sec", [])
            focal_supplied = [point for point in supplied
                              if window.focal_start_sec <= point < window.focal_end_sec
                              or point == window.focal_end_sec == asset.duration_sec]
            anchor_selection = "native_second_midpoint"
            if focal_supplied:
                anchor = min(anchor_candidates, key=lambda frame: min(abs(frame.pts_sec - point) for point in focal_supplied))
                anchor_selection = "nearest_native_frame_to_supplied_coarse_train_anchor"
            if config.anchor_frame_questions:
                request = with_anchor_questions(request, anchor)
            request_hash = canonical_sha256(request.model_dump(mode="json"))
            binding = {"plan_sha256": plan["plan_sha256"], "model_manifest_sha256": manifest_sha256,
                       "request_sha256": request_hash}
            if checkpoint.exists():
                saved = load_json(checkpoint)
                require(saved.get("binding") == binding, "teacher_resume_binding_changed")
                body = saved.get("window")
                require(isinstance(body, dict) and saved.get("window_sha256") == canonical_sha256(body),
                        "teacher_resume_window_changed")
                require(body.get("second_index") == window.index and body.get("complete") is True,
                        "teacher_resume_window_incomplete")
                completed.append(body)
                previous_attestation = body.get("serving_attestation")
                if isinstance(previous_attestation, dict):
                    provenance["serving_attestation"] = previous_attestation
                resumed += 1
                continue
            if max_model_requests is not None and model_requests >= max_model_requests:
                break
            model_requests += 1
            try:
                batch = await engine.analyze(request, deadline=deadline, cancel_event=cancel)
            except (DecisionOutputError, InvalidModelOutput) as error:
                # Invalid completions count against the bounded call budget.
                # They remain missing observations and produce no training target.
                invalid_windows.append({"second_index": window.index,
                                        "error_type": type(error).__name__, "state": "invalid_teacher_output"})
                audit("partial", "teacher_invalid_output")
                continue
            attestation = batch.metrics.get("serving_attestation") or {}
            verified = (attestation.get("model_files_verified") is True
                        and attestation.get("model_revision") == manifest["revision"]
                        and attestation.get("model_manifest_sha256") == manifest_sha256)
            if verified:
                provenance.update(actual_snapshot_verified=True, serving_attestation=dict(attestation))
            body = {"clip_id": request.clip_id, "second_index": window.index,
                    "start_sec": request.start_sec, "end_sec": request.end_sec,
                    "focal_start_sec": window.focal_start_sec, "focal_end_sec": window.focal_end_sec,
                    "anchor_frame_ids": [anchor.frame_id], "anchor_pts_sec": [anchor.pts_sec],
                    "anchor_selection": anchor_selection,
                    "frame_ids": [frame.frame_id for frame in request.frame_refs],
                    "requested_fact_ids": [fact.fact_id for fact in facts], "requested_rule_ids": [],
                    "teacher_requested_fact_ids": [fact.fact_id for fact in request.fact_definitions],
                    "facts": [fact.model_dump(mode="json") for fact in batch.facts
                              if not fact.fact_id.startswith("ANCHOR__")], "candidate_events": [],
                    "frame_fact_targets": extract_anchor_targets(batch, anchor),
                    "generation": 0, "complete": True, "disposition": batch.disposition,
                    "prompt_hash": batch.prompt_hash, "prompt_version": batch.prompt_version,
                    "config_version": batch.config_version, "human_reviewed": False,
                    "fact_scope": "full_request_context", "runtime_snapshot_verified": verified,
                    "serving_attestation": dict(attestation) if verified else None,
                    "inspection_focus": record.get("inspection_focus", {
                        "origin": "uniform_seconds", "independent_localization": False,
                        "fact_scope": "full_request_context_with_cited_frame_support", "focal_exclusive_labels": False})}
            _write(checkpoint, {"binding": binding, "window_sha256": canonical_sha256(body), "window": body})
            completed.append(body)
        audit("done" if len(completed) == len(windows) else "partial")
    except BaseException:
        audit("failed", "teacher_record_incomplete")
        raise
    return {"completed_records": int(len(completed) == len(windows)),
            "partial_records": int(len(completed) != len(windows)),
            "completed_seconds": len(completed), "resumed_seconds": resumed,
            "native_source_refs": len(frames), "model_requests": model_requests,
            "invalid_output_requests": len(invalid_windows), "human_reviewed_records": 0,
            **summarize_teacher_windows(completed)}


async def _run_cli(args: Any) -> dict[str, Any]:
    from forte.model import LocalModelConfig, LocalModelProvider
    from forte.model.manifest import load_manifest

    plan, split, release = load_json(args.plan), load_json(args.split), load_json(args.execution_release)
    validate_teacher_plan(plan, split, split_sha256=file_sha256(args.split))
    validate_execution_release(release, plan)
    require(plan["partition"] == "train", "teacher_runner_train_only")
    require(type(args.max_records) is int and args.max_records > 0, "teacher_record_limit_invalid")
    require(type(args.max_model_requests) is int and args.max_model_requests > 0, "teacher_model_request_limit_invalid")
    manifest = load_manifest(args.manifest)
    # Register the exact bounded source set BEFORE constructing any inference
    # work. Registry failures prevent every model call. Registration is an
    # append-only exposure record, not a replacement for the GPU owner's lease.
    from scripts.exposure_registry import register_content_manifest
    eligible_records = [record for record in plan["records"]
                        if not args.anchor_focus_only or record.get("priority_second_indices")]
    selected_records = eligible_records[args.record_offset:args.record_offset + args.max_records]
    require(bool(selected_records), "teacher_record_selection_empty")
    selected_path = args.output_root.resolve() / "execution-inputs" / (plan["plan_sha256"] + ".json")
    _write(selected_path, {"records": [{"content_sha256": record["content_sha256"], "split": "train"}
                                     for record in selected_records]})
    if getattr(args, "exposure_receipt", None):
        validate_exposure_receipt(load_json(args.exposure_receipt), plan, selected_records, release,
                                  source_manifest_sha256=file_sha256(args.source_manifest))
    else:
        registry_receipt = register_content_manifest(selected_path, args.exposure_registry, args.source_manifest,
            args.split, expected_assignments_sha256=plan["assignments_sha256"],
            trial_id="teacher-" + canonical_sha256({"plan": plan["plan_sha256"], "release": release["reference"]})[:24],
            purpose="bounded_teacher_attempt", allowed_splits={"train"})
        require(registry_receipt.get("ok") is True, "teacher_exposure_registration_failed")
    model_root = args.output_root.resolve() / plan["partition"]
    provider = LocalModelProvider(LocalModelConfig(media_root=model_root, base_url=args.endpoint,
        model=manifest["served_model_name"], expected_revision=manifest["revision"],
        api_key=os.environ.get("FORTE_MODEL_API_KEY"),
        max_frames=plan["sampling"]["max_frames"], max_tokens=4096, max_attempts=1))
    totals = {"completed_records": 0, "partial_records": 0, "completed_seconds": 0, "resumed_seconds": 0,
              "native_source_refs": 0, "model_requests": 0, "invalid_output_requests": 0,
              "human_reviewed_records": 0, **summarize_teacher_windows([])}
    deadline = time.monotonic() + args.max_runtime_sec
    try:
        for record in selected_records:
            remaining = args.max_model_requests - totals["model_requests"]
            if remaining <= 0:
                break
            summary = await run_teacher_record(record, plan, source_root=args.source_root,
                output_root=args.output_root, provider=provider, manifest=manifest,
                manifest_sha256=file_sha256(args.manifest), release=release, deadline=deadline,
                max_model_requests=remaining,
                source_by_hash=load_json(args.source_map) if args.source_map else None,
                anchor_focus_only=args.anchor_focus_only)
            for key, value in summary.items():
                totals[key] += value
    finally:
        await provider.aclose()
    return {"status": "teacher_proposals_written_private", "partition": plan["partition"], **totals,
            "planned_records": len(plan["records"]), "training_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_command = commands.add_parser("plan", help="Metadata-only; never performs inference")
    plan_command.add_argument("--worklist", type=Path, required=True)
    plan_command.add_argument("--split", type=Path, required=True)
    plan_command.add_argument("--partition", choices=("train", "validation"), default="train")
    plan_command.add_argument("--output", type=Path, required=True)
    plan_command.add_argument("--train-anchors", type=Path)
    plan_command.add_argument("--anchor-pins", type=Path, default=Path("data/splits/user_coarse_anchors.pins.json"))
    plan_command.add_argument("--trusted-scene-context", choices=("bank_cash_workstation",))
    run_command = commands.add_parser("run", help="Run only after the serialized GPU slot is released")
    for flag in ("plan", "split", "execution-release", "source-root", "output-root", "manifest"):
        run_command.add_argument("--" + flag, type=Path, required=True)
    run_command.add_argument("--endpoint", required=True)
    run_command.add_argument("--max-records", type=int, default=1)
    run_command.add_argument("--record-offset", type=int, default=0)
    run_command.add_argument("--max-model-requests", type=int, default=8)
    run_command.add_argument("--source-map", type=Path)
    run_command.add_argument("--anchor-focus-only", action="store_true",
                             help="Bounded first pass over train coarse-anchor neighborhoods only")
    run_command.add_argument("--exposure-registry", type=Path, default=Path("benchmarks/private/model-exposure.jsonl"))
    run_command.add_argument("--exposure-receipt", type=Path,
                             help="Pinned private receipt of a prior authoritative central registration")
    run_command.add_argument("--source-manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    run_command.add_argument("--max-runtime-sec", type=float, default=3600)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            require(not args.output.exists(), "teacher_plan_output_already_exists")
            result = build_teacher_plan(load_json(args.worklist), load_json(args.split),
                split_sha256=file_sha256(args.split), worklist_sha256=file_sha256(args.worklist),
                partition=args.partition, config=TeacherSamplingConfig(trusted_scene_context=args.trusted_scene_context))
            if args.train_anchors:
                result = prioritize_train_anchors(result, load_train_anchors(args.train_anchors, args.anchor_pins),
                                                  anchors_sha256=file_sha256(args.train_anchors))
            _write(args.output, result)
            summary = {"status": "metadata_plan_created", "partition": args.partition,
                       "records": len(result["records"]),
                       "second_requests": sum(record["second_count"] for record in result["records"]),
                       "model_inference_executed": False, "gpu_used": False}
        else:
            require(math.isfinite(args.max_runtime_sec) and args.max_runtime_sec > 0,
                    "teacher_runtime_limit_invalid")
            require(args.record_offset >= 0, "teacher_record_offset_invalid")
            summary = asyncio.run(_run_cli(args))
        print(json.dumps(summary))
        return 0
    except Exception as error:
        # Messages may contain media paths or private observations. Expose only
        # our fixed guard codes, never arbitrary provider/decoder exception text.
        from training.guards import TrainingGateError
        code = str(error) if isinstance(error, TrainingGateError) else "teacher_execution_failed"
        print(json.dumps({"status": "blocked_or_failed", "reason_code": code,
                          "error_type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
