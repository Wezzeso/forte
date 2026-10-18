"""Automated first pass from existing team-local model outputs.

No inference, external service, organizer-label lookup or human annotation is
performed here. Machine consistency checks are not independent visual review.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from .catalog import event_definitions, fact_definitions
from .proposals import _sha256, _under_roots
from .store import ReviewError, ReviewStore, canonical, utc_now

CONFIG = {
    "method": "local_model_structured_first_pass.v1",
    "group_method": "event_fact_signature_geometry_duration.v1",
    "duration_bucket_sec": 10,
    "organizer_labels_used": False,
    "absent_requires_complete_visible_context": True,
    "candidate_endpoints": "exact_unambiguous_original_pts",
    "unmodeled_groups": "exact_content_only",
    "frame_target_method": "direct_anchor_question_bound.v2",
    "explicit_anchor_unclear": "class_unclear_and_no_decisive_support",
    "support_conflicts": "retain_receipts_mask_both_directions",
}
CONFIG_SHA256 = hashlib.sha256(canonical(CONFIG).encode()).hexdigest()
ANCHOR_PHYSICAL_FACT_IDS = frozenset({"CASH_VISIBLE", "CASH_ON_WORK_SURFACE", "CASH_HANDLER_PRESENT", "CASH_HANDLER_LEAVES_WORKSTATION", "CASH_HANDOVER_VISIBLE", "CASH_COUNT_VISIBLE", "CASH_PLACED_IN_CLOTHING_POCKET", "CASH_PLACED_IN_PERSONAL_CONTAINER"})


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def canonical_receipt_sha256(value: Any) -> str:
    """Shared training/calibration digest, distinct from legacy summary hashes."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _normalize_frame_targets(window: dict[str, Any], original: dict[str, Any], frames: dict[str, Any], *, source_bound: bool) -> None:
    raw_targets = original.get("frame_fact_targets", [])
    if not isinstance(raw_targets, list):
        window["quality_flags"].append("invalid_explicit_frame_target_omitted")
        return
    targets, reviews, seen = [], [], set()
    prompt = window["provenance"]["prompt_sha256"]
    for raw in raw_targets:
        if not isinstance(raw, dict) or any(not isinstance(raw.get(key), str) for key in ("fact_id", "frame_id", "state")):
            window["quality_flags"].append("invalid_explicit_frame_target_omitted")
            continue
        fact_id, frame_id, state = raw["fact_id"], raw["frame_id"], raw["state"]
        support, context = raw.get("supporting_frame_ids", []), raw.get("context_frame_ids", [])
        definition, raw_provenance = raw.get("question_definition"), raw.get("provenance", {})
        base_valid = (fact_id in ANCHOR_PHYSICAL_FACT_IDS and frame_id in window["frame_ids"] and state in {"present", "absent", "unclear"}
                      and raw.get("scope") == "exact_anchor_frame" and raw.get("derivation") == "direct_anchor_teacher_question"
                      and raw.get("localizable_category") == "physical_anchor_atom" and raw.get("human_verified") is False
                      and isinstance(support, list) and isinstance(context, list) and all(isinstance(key, str) for key in support + context)
                      and set(support + context) <= set(window["frame_ids"]))
        if not base_valid:
            window["quality_flags"].append("invalid_explicit_frame_target_omitted")
            continue
        expected_question = "ANCHOR__" + fact_id
        alias = f"f{window['frame_ids'].index(frame_id):02d}"
        prefix = f"INDEPENDENT EXACT-FRAME QUESTION: evaluate physical fact {fact_id} at frame {alias}, timestamp {frames[frame_id].pts_sec:.9f} seconds ONLY."
        question_bound = (isinstance(definition, dict) and definition.get("fact_id") == expected_question
                          and isinstance(definition.get("description"), str) and definition["description"].startswith(prefix)
                          and raw.get("question_fact_id") == expected_question and isinstance(raw_provenance, dict)
                          and prompt is not None and raw_provenance.get("prompt_sha256") == prompt
                          and raw_provenance.get("focal_question_sha256") == canonical_receipt_sha256(definition)
                          and frame_id in original.get("anchor_frame_ids", []) and raw.get("pts_sec") == frames[frame_id].pts_sec
                          and original.get("complete") is True)
        if not question_bound:
            window["quality_flags"].append("unbound_focal_question_target_omitted")
            continue
        identity = (frame_id, fact_id)
        if identity in seen:
            window["quality_flags"].append("duplicate_focal_target_pair_omitted")
            targets = [item for item in targets if (item["frame_id"], item["fact_id"]) != identity]
            reviews = [item for item in reviews if (item["frame_id"], item["fact_id"]) != identity]
            continue
        seen.add(identity)
        decisive = state in {"present", "absent"} and frame_id in support
        eligible = source_bound and (state == "unclear" or decisive) and raw.get("eligible") is True
        target_provenance = {**deepcopy(window["provenance"]), "focal_question_sha256": raw_provenance["focal_question_sha256"], "original_target_provenance": deepcopy(raw_provenance), "explicit_teacher_target_sha256": canonical_receipt_sha256(raw)}
        item = {"frame_id": frame_id, "pts_sec": frames[frame_id].pts_sec, "fact_id": fact_id, "state": state, "supporting_frame_ids": list(dict.fromkeys(support)), "context_frame_ids": list(dict.fromkeys(context)), "review_status": "machine_consistency_checked", "scope": "exact_anchor_frame", "derivation": "direct_anchor_teacher_question", "localizable_category": "physical_anchor_atom", "human_verified": False, "supervision_mask": eligible, "question_fact_id": expected_question, "question_definition": deepcopy(definition), "provenance": target_provenance}
        targets.append(item)
        if not eligible:
            continue
        review = {**deepcopy(item), "focal_frame_id": frame_id, "teacher_state": state, "supports_state": decisive, "reference_kind": "focal_frame_explicit_decisive" if decisive else "focal_frame_explicit_unclear"}
        if not decisive:
            review["negative_reference_kind"] = "focal_frame_explicit_unclear"
            conflict = any(fact["fact_id"] == fact_id and window["fact_supervision_mask"].get(fact_id) and frame_id in fact["supporting_frame_ids"] for fact in window["target"]["facts"])
            if conflict:
                window["quality_flags"].append("direct_anchor_window_support_conflict")
                review["supervision_mask"] = False
                review["mask_reason"] = "direct_anchor_window_support_conflict"
        reviews.append(review)
    if targets:
        window["frame_fact_targets"] = targets
    if reviews:
        window["support_reviews"] = reviews


def _latest_proposal(record: dict[str, Any]) -> dict[str, Any] | None:
    candidates = record.get("local_model_proposals", [])
    if not candidates:
        return None
    # Summaries represent the current imported attempt, including its actual
    # failures/abstentions. Older longer attempts remain separately available.
    return max(candidates, key=lambda value: (value.get("imported_at", ""), value["audit_sha256"]))


def _readable_explanation(row: dict[str, Any], definitions: dict[str, Any], catalog: Any) -> dict[str, Any]:
    """Deterministic local WHAT/WHY; no free model notes or invented rule match."""
    observations, seen = [], set()
    for window in row["windows"]:
        for fact in window["target"]["facts"]:
            if not window["fact_supervision_mask"].get(fact["fact_id"]):
                continue
            identity = (fact["fact_id"], fact["state"], tuple(fact["supporting_frame_ids"]))
            if identity in seen:
                continue
            seen.add(identity)
            definition = definitions[fact["fact_id"]]
            observations.append({"fact_id": fact["fact_id"], "state": fact["state"], "text_ru": definition["observable_phrase_ru" if fact["state"] == "present" else "absence_phrase_ru"], "supporting_frame_ids": fact["supporting_frame_ids"], "window_id": window["window_id"], "origin": "machine_observation"})
    event_explanations = []
    for event in row["events"]:
        rules = []
        for match in event.get("rule_matches", []):
            rule = catalog.get_rule(match["rule_id"])
            rules.append({"rule_id": rule.rule_id, "title_ru": rule.title, "clause": rule.clause, "required_states_match": match["required_states_match"], "unsatisfied_fact_ids": match["unsatisfied_fact_ids"], "grounding_status": "requires_temporal_and_exception_verification"})
        event_explanations.append({"event_type_id": event["event_type_id"], "what_ru": "Машинный кандидат: " + catalog.event_type_label(event["event_type_id"]), "why_ru": "Совпадение физических признаков с нормой само по себе не подтверждает последовательность действий, применимость и исключения.", "rule_candidates": rules, "evidence_frame_ids": event.get("evidence_frame_ids", []), "human_reviewed": False})
    return {"status": "machine_candidates" if event_explanations else "physical_observations_only" if observations else "abstain", "what_ru": "Есть машинные кандидаты событий; проверяйте исходные подтверждающие кадры." if event_explanations else "Тип события пока не определен по доступному машинному выводу.", "why_ru": "Ни бинарная метка записи, ни приблизительная секунда не устанавливают тип события или нарушенную норму. Отсутствие машинного кандидата не доказывает отсутствие нарушения.", "physical_observations": observations, "events": event_explanations, "human_reviewed": False, "organizer_labels_used_for_explanation": False}


def summarize_record(record: dict[str, Any], workspace: Path) -> dict[str, Any] | None:
    """Derive scope-bound weak targets; never promote failures to negatives."""
    from forte.shared.schemas import FactDecision, FrameRef, VideoAsset
    from forte.video.decode import validate_frame_ref
    from rules.catalog import load_catalog

    proposal = _latest_proposal(record)
    if proposal is None:
        return None
    asset = VideoAsset.model_validate(proposal["asset"])
    if asset.content_hash != record["content_hash"] or record["source_role"] == "jury_test":
        raise ReviewError("machine_source_identity_forbidden")
    flags: set[str] = set()
    if not Path(asset.owned_path).is_file() or _sha256(Path(asset.owned_path)) != asset.content_hash:
        flags.add("source_hash_unavailable_or_changed")
    frames, valid_frames = {}, set()
    for raw in proposal.get("frames", []):
        try:
            frame = FrameRef.model_validate(raw)
            frame.validate_timeline(asset)
            validate_frame_ref(asset, frame)
            if frame.job_id != asset.job_id or frame.frame_id in frames:
                raise ValueError("source binding")
            path = Path(frame.path)
            if not path.is_absolute() or not path.is_file():
                raise ValueError("registered local frame unavailable")
            if frame.jpeg_sha256 is None or _sha256(path) != frame.jpeg_sha256:
                raise ValueError("checksum")
            valid_frames.add(frame.frame_id)
            frames[frame.frame_id] = frame
        except Exception:
            flags.add("source_frame_missing_changed_or_unbound")
    catalog = load_catalog()
    expected_facts = {fact.fact_id for fact in catalog.fact_definitions()}
    known_facts, known_events = fact_definitions(), event_definitions()
    windows, events = [], []
    for index, raw_window in enumerate(proposal.get("window_decisions", [])):
        window_flags: set[str] = set()
        ids = list(dict.fromkeys(raw_window.get("frame_ids", [])))
        valid_ids = [frame_id for frame_id in ids if frame_id in valid_frames]
        if len(valid_ids) != len(ids):
            window_flags.add("window_frame_binding_incomplete")
        start, end = raw_window["start_sec"], raw_window["end_sec"]
        observed = []
        for raw_fact in raw_window.get("facts", []):
            try:
                fact = FactDecision.model_validate(raw_fact)
                if fact.fact_id not in known_facts or set(fact.supporting_frame_ids) - set(ids):
                    raise ValueError("fact binding")
                observed.append(fact.model_dump(mode="json"))
            except Exception:
                window_flags.add("invalid_fact_observation_masked")
        states = {fact["fact_id"]: fact["state"] for fact in observed}
        context = {"continuous_visibility": states.get("WORK_AREA_CONTINUOUSLY_VISIBLE") == "present", "procedure_complete": states.get("PROCEDURE_CONTEXT_COMPLETE") == "present", "counting_paths_visible": states.get("ALL_COUNTING_PATHS_VISIBLE") == "present"}
        targets, masks = [], {}
        for fact in observed:
            support = [frame_id for frame_id in fact["supporting_frame_ids"] if frame_id in valid_ids]
            state = fact["state"]
            supported = state in {"present", "absent"} and bool(support) and not flags
            if state == "absent" and not (context["continuous_visibility"] and context["procedure_complete"]):
                supported = False
                window_flags.add("absent_without_complete_context_masked")
            if state == "absent" and fact["fact_id"] == "CASH_COUNT_VISIBLE" and not context["counting_paths_visible"]:
                supported = False
                window_flags.add("count_absence_without_all_paths_masked")
            masks[fact["fact_id"]] = supported
            targets.append({"fact_id": fact["fact_id"], "state": state if supported else "unclear", "supporting_frame_ids": support, "notes": None})
        missing = sorted(expected_facts - states.keys())
        if missing:
            window_flags.add("partial_fact_vocabulary_not_filled")
        raw_targets = []
        for event in raw_window.get("candidate_events", []):
            if event.get("event_type_id") not in known_events or not start <= event["start_sec"] <= event["end_sec"] <= end:
                window_flags.add("candidate_scope_invalid_masked")
                continue
            support = [frame_id for frame_id in event.get("supporting_frame_ids", []) if frame_id in valid_ids]
            bounds = [[frame_id for frame_id in valid_ids if abs(frames[frame_id].pts_sec - event[key]) < 1e-6] for key in ("start_sec", "end_sec")]
            fact_ids = [fact["fact_id"] for fact in event.get("supporting_facts", []) if fact.get("fact_id") in states]
            exact = all(len(values) == 1 for values in bounds) and bool(support) and bool(fact_ids) and not flags
            if exact:
                raw_targets.append({"candidate_id": "machine_" + _digest({"proposal": proposal["audit_sha256"], "window": index, "candidate": event})[:24], "event_type_id": event["event_type_id"], "start_frame_id": bounds[0][0], "end_frame_id": bounds[1][0], "supporting_fact_ids": fact_ids, "supporting_frame_ids": support, "visibility_notes": []})
            else:
                window_flags.add("candidate_endpoint_or_support_ambiguous_masked")
            within = [frame_id for frame_id in support if event["start_sec"] <= frames[frame_id].pts_sec <= event["end_sec"]]
            anchor = min(within, key=lambda frame_id: abs(frames[frame_id].pts_sec - (event["start_sec"] + event["end_sec"]) / 2)) if within else None
            event_states = {fact["fact_id"]: fact["state"] for fact in event.get("supporting_facts", [])}
            matches = []
            for rule in catalog.enabled_rules():
                binding = catalog.bindings[rule.rule_id]
                if binding["event_type_id"] == event["event_type_id"]:
                    missing_rule = [fact_id for fact_id, state in binding["required_fact_states"].items() if event_states.get(fact_id) != state]
                    matches.append({"rule_id": rule.rule_id, "required_states_match": not missing_rule, "unsatisfied_fact_ids": missing_rule, "temporal_independently_verified": False, "exceptions_independently_verified": False})
            events.append({"event_id": "machine_event_" + _digest({"proposal": proposal["audit_sha256"], "window": index, "event": event})[:24], "event_type_id": event["event_type_id"], "start_sec": event["start_sec"], "end_sec": event["end_sec"], "anchor_frame_id": anchor, "anchor_sec": frames[anchor].pts_sec if anchor else None, "anchor_selection": "nearest_supporting_original_pts" if anchor else "unavailable", "evidence_frame_ids": ([anchor] + [frame_id for frame_id in support if frame_id != anchor])[:3] if anchor else [], "facts": deepcopy(event.get("supporting_facts", [])), "rule_matches": matches, "status": "machine_candidate", "human_reviewed": False, "window_id": raw_window.get("clip_id"), "exact_target_endpoints": exact})
        windows.append({"window_id": "machine_window_" + _digest({"proposal": proposal["audit_sha256"], "index": index})[:24], "clip_id": raw_window.get("clip_id"), "start_sec": start, "end_sec": end, "frame_ids": valid_ids, "requested_fact_ids": [fact["fact_id"] for fact in observed], "observed_facts": observed, "target": {"facts": targets, "candidates": raw_targets}, "fact_supervision_mask": masks, "context_masks": context, "quality_flags": sorted(window_flags), "missing_full_vocabulary_fact_ids": missing, "generation": raw_window.get("generation", 0), "model_output_complete": raw_window.get("complete") is True, "human_reviewed": False})
        if any(key in raw_window for key in ("focal_start_sec", "focal_end_sec", "anchor_frame_ids")):
            focal_start, focal_end = raw_window.get("focal_start_sec"), raw_window.get("focal_end_sec")
            anchors = raw_window.get("anchor_frame_ids", [])
            numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            if numeric(focal_start) and numeric(focal_end) and start <= focal_start <= focal_end <= end and isinstance(anchors, list) and anchors and set(anchors) <= set(valid_ids) and all(focal_start <= frames[key].pts_sec <= focal_end for key in anchors):
                windows[-1]["inspection_focus"] = {"start_sec": focal_start, "end_sec": focal_end, "anchor_frame_ids": anchors, "fact_scope": "full_request_context_with_cited_frame_support", "focal_exclusive_labels": False}
            else:
                windows[-1]["quality_flags"].append("invalid_inspection_focus_metadata_omitted")
    for event in proposal.get("verified_events", []):
        anchor = event.get("anchor_frame_id")
        if anchor not in valid_frames:
            flags.add("pipeline_verified_event_source_unavailable")
            continue
        events.append({**deepcopy(event), "anchor_sec": frames[anchor].pts_sec, "anchor_selection": "original_pipeline_anchor", "status": "pipeline_verified_machine_only", "human_reviewed": False, "rule_grounding_independently_reviewed": False})
    coverage = proposal.get("coverage", {})
    if not coverage.get("complete"):
        flags.add("technical_coverage_incomplete")
    if proposal.get("phase") != "done":
        flags.add("attempt_not_successfully_completed")
    if not windows:
        flags.add("no_completed_model_window_outputs")
    if proposal.get("provenance", {}).get("actual_snapshot_verified") is not True:
        flags.add("runtime_snapshot_not_independently_attested")
    verdict = None
    if not flags and proposal.get("publication_ready") is True and proposal.get("phase") == "done":
        verdict = "violation" if proposal.get("verified_events") else "no_violation" if not proposal.get("candidate_events") else None
    provenance = deepcopy(proposal["provenance"])
    attestation = provenance.get("serving_attestation", {})
    provenance.update(source_proposal_sha256=proposal["audit_sha256"], model_manifest_sha256=attestation.get("model_manifest_sha256"), prompt_sha256=provenance.get("prompt_sha256"))
    if provenance.get("execution_location") == "team_deployed_service" and provenance.get("external_ai_used") is False:
        provenance["inference_location"] = "team_local_service"
        provenance["inference_location_source"] = "validated_imported_proposal_execution_location_and_external_ai_used"
    for window, original in zip(windows, proposal.get("window_decisions", []), strict=True):
        prompt = original.get("prompt_sha256", original.get("prompt_hash"))
        exact_prompt = prompt if isinstance(prompt, str) and re.fullmatch(r"[a-f0-9]{64}", prompt) else None
        window["provenance"] = {**deepcopy(provenance), "prompt_sha256": exact_prompt, "prompt_hash_source": "original_audit_window" if exact_prompt else None, "prompt_version": original.get("prompt_version"), "config_version": original.get("config_version"), "proposal_config_sha256": CONFIG_SHA256}
        if exact_prompt is None:
            window["quality_flags"].append("exact_window_prompt_hash_unavailable")
        _normalize_frame_targets(window, original, frames, source_bound=not flags.intersection({"source_hash_unavailable_or_changed", "source_frame_missing_changed_or_unbound"}))
    row = {"schema_version": "forte.machine-annotations.v1", "annotation_origin": "local_model", "review_status": "machine_consistency_checked", "machine_review_scope": "structural_source_and_rule_consistency_only", "human_reviewed": False, "training_authorized": False, "organizer_labels_used": False, "sample_id": record["sample_id"], "recording_ids": record["recording_ids"], "content_sha256": asset.content_hash, "protected_group_id": record["group_id"], "split": record["split"], "source_role": record["source_role"], "duration_sec": asset.duration_sec, "source_time_base": asset.time_base, "source_start_sec": asset.source_start_sec, "frames": [frame.model_dump(mode="json") for frame in frames.values()], "windows": windows, "events": events, "machine_verdict": verdict, "machine_presence": "candidate_present" if events else "unresolved" if flags else "no_candidate", "coverage": coverage, "limitations": sorted(flags | {"machine_observations_are_not_independent_ground_truth"}), "provenance": provenance, "config_sha256": CONFIG_SHA256}
    row["source_pts_identity_checked"] = len(valid_frames) == len(proposal.get("frames", [])) and bool(valid_frames)
    row["per_second"] = []
    for second in range(math.ceil(asset.duration_sec)):
        start, end = float(second), min(float(second + 1), asset.duration_sec)
        frame_ids = [frame_id for frame_id, frame in frames.items() if start <= frame.pts_sec < end or end == asset.duration_sec and frame.pts_sec == end]
        overlapping = [window for window in windows if window["start_sec"] < end and start <= window["end_sec"] and set(window["frame_ids"]) & set(frame_ids)]
        state = "unknown_no_sampled_frame" if not frame_ids else "source_frames_available_unanalyzed" if not overlapping else "machine_observed" if any(window["model_output_complete"] for window in overlapping) else "partial_machine_observation"
        machine_events = [event for event in events if event.get("anchor_sec") is not None and start <= event["anchor_sec"] < end]
        row["per_second"].append({"start_sec": start, "end_sec": end, "state": state, "source_frame_ids": frame_ids, "source_pts_sec": [frames[frame_id].pts_sec for frame_id in frame_ids], "window_ids": [window["window_id"] for window in overlapping], "machine_event_ids": [event["event_id"] for event in machine_events], "event_state": "machine_candidate_present" if machine_events else "abstain", "human_reviewed": False})
    row["timestamp_note"] = "Second boundaries are coverage bins; event/frame times retain actual source PTS. Numeric milliseconds do not establish one-millisecond accuracy."
    row["explanation_ru"] = _readable_explanation(row, known_facts, catalog)
    row["summary_sha256"] = _digest(row)
    row["generated_at"] = utc_now()
    return row


def build_group_proposals(records: list[dict[str, Any]], summaries: list[dict[str, Any]], source_manifest_sha256: str) -> dict[str, Any]:
    """Full alias coverage; heuristic groups are explicitly unvalidated."""
    by_hash = {row["content_sha256"]: row for row in summaries}
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        row = by_hash.get(record["content_hash"])
        if row is None:
            group_id, method = "auto_exact_" + record["content_hash"][:24], "exact_content_only_no_model_observation"
            references = []
        else:
            facts = sorted({(fact["fact_id"], fact["state"]) for window in row["windows"] for fact in window["observed_facts"] if fact["state"] == "present"})
            first = row["frames"][0] if row["frames"] else {}
            signature = {"events": sorted({event["event_type_id"] for event in row["events"]}), "present_facts": facts, "geometry": [first.get("width"), first.get("height"), first.get("rotation_deg")], "duration_bucket": int(row["duration_sec"] // CONFIG["duration_bucket_sec"])}
            group_id, method = "auto_model_" + _digest(signature)[:24], CONFIG["group_method"]
            references = [row["provenance"]["source_proposal_sha256"]]
        item = groups.setdefault(group_id, {"proposed_group_id": group_id, "recording_ids": [], "content_sha256": [], "method": method, "confidence_status": "heuristic_unvalidated", "provenance_refs": [], "human_reviewed": False})
        item["recording_ids"].extend(record["recording_ids"])
        item["content_sha256"].append(record["content_hash"])
        item["provenance_refs"].extend(references)
    for item in groups.values():
        for key in ("recording_ids", "content_sha256", "provenance_refs"):
            item[key] = sorted(set(item[key]))
    return {"schema_version": "forte.automatic-group-proposals.v1", "proposal_origin": "local_model_and_metadata", "method": CONFIG["group_method"], "config_sha256": CONFIG_SHA256, "source_manifest_sha256": source_manifest_sha256, "groups": [groups[key] for key in sorted(groups)], "provenance": {"config": CONFIG, "model_source_count": len(summaries), "exact_content_only_source_count": len(records)-len(summaries), "organizer_labels_used": False, "human_reviewed": False, "scenario_independence_validated": False}, "limitations": ["model_signature_similarity_is_not_proof_of_same_scenario", "unmodeled_sources_have_exact_duplicate_grouping_only", "automatic_proposals_do_not_authorize_split_lock_or_training"], "training_authorized": False}


def _write(path: Path, raw: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_annotation_worklist(records: list[dict[str, Any]], summaries: list[dict[str, Any]], split: dict[str, Any] | None = None, split_file_sha256: str | None = None) -> dict[str, Any]:
    indexed = {}
    if split is not None:
        for source in split.get("records", []):
            content = source.get("content_sha256", source.get("content_hash"))
            entry = {"split": source["split"], "scenario_group_id": source.get("scenario_group_id") or source["group_id"]}
            if content in indexed and indexed[content] != entry:
                raise ReviewError("machine_worklist_split_conflict")
            indexed[content] = entry
    by_hash = {row["content_sha256"]: row for row in summaries}
    work, excluded = [], 0
    for record in records:
        partition = indexed.get(record["content_hash"])
        if partition is not None and partition["split"] not in {"train", "validation"}:
            excluded += 1
            continue
        if split is not None and partition is None:
            raise ReviewError("machine_worklist_split_source_missing")
        summary = by_hash.get(record["content_hash"])
        pending_seconds = [item["start_sec"] for item in summary["per_second"] if item["state"] != "machine_observed"] if summary else list(range(math.ceil(record.get("duration_sec") or 0)))
        work.append({"sample_id": record["sample_id"], "content_sha256": record["content_hash"], "recording_ids": record["recording_ids"], "source_path": record["media_paths"][0], "duration_sec": record.get("duration_sec"), "split": partition["split"] if partition else "unassigned", "scenario_group_id": partition["scenario_group_id"] if partition else None, "has_machine_first_pass": summary is not None, "pending_second_bins": pending_seconds, "requested_sampling": "decode_original_source_pts_cover_each_second", "queue_state": "waiting_execution_release" if partition else "waiting_automatic_split_assignment", "execution_authorized": False, "human_review_required": False})
    return {"schema_version": "forte.autonomous-annotation-worklist.v1", "records": work, "split_file_sha256": split_file_sha256, "assignments_sha256": split.get("assignments_sha256") if split else None, "split_origin": split.get("state") if split else None, "excluded_final_check_or_test_sources": excluded, "execution_authorized": False, "training_authorized": False, "organizer_labels_used_for_facts": False}


def attach_provided_focus(summaries: list[dict[str, Any]], worklist: dict[str, Any], manifests: list[Path], records: list[dict[str, Any]]) -> dict[str, int]:
    """Keep coarse provided metadata separate; validation never conditions input."""
    if not worklist.get("assignments_sha256"):
        raise ReviewError("provided_anchor_requires_pinned_assignments")
    source_by_hash = {row["content_hash"]: row for row in records}
    focus = {}
    for manifest in manifests:
        manifest_hash = _sha256(manifest)
        for raw in manifest.read_text(encoding="utf-8-sig").splitlines():
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ReviewError("provided_anchor_record_invalid")
            content, split = value.get("content_sha256"), value.get("split")
            source = source_by_hash.get(content)
            if value.get("schema_version") != "forte.user-coarse-anchor.v1" or value.get("anchor_origin") != "user_provided_coarse_table" or value.get("target_scope") != "class_agnostic_approximate_event_anchor" or value.get("event_type_known") is not False or value.get("boundaries_known") is not False or value.get("legal_ground_truth") is not False or split not in {"train", "validation"} or value.get("assignments_sha256") != worklist["assignments_sha256"] or not isinstance(value.get("anchor_source_sha256"), str) or re.fullmatch("[a-f0-9]{64}", value["anchor_source_sha256"]) is None or source is None or set(value.get("recording_ids", [])) != set(source["recording_ids"]):
                raise ReviewError("provided_anchor_source_or_partition_binding_invalid")
            points = value.get("coarse_anchor_sec")
            if not isinstance(points, list) or any(not isinstance(point, (int, float)) or isinstance(point, bool) or not math.isfinite(point) or not 0 <= point <= source["duration_sec"] for point in points):
                raise ReviewError("provided_anchor_point_invalid")
            item = {"coarse_anchor_sec": points, "anchor_origin": "user_provided_coarse_table", "anchor_source_sha256": value["anchor_source_sha256"], "manifest_sha256": manifest_hash, "assignments_sha256": value["assignments_sha256"], "target_scope": "class_agnostic_approximate_event_anchor", "anchor_semantics": "approximate_point_not_boundary", "uncertainty_sec": None, "outside_anchor_interval_label": "unknown", "event_type_known": False, "boundaries_known": False, "legal_ground_truth": False, "split": split, "allowed_use": "train_only_context_prior_or_eventness_bag" if split == "train" else "evaluation_only_do_not_condition_inference"}
            if content in focus and focus[content] != item:
                raise ReviewError("provided_anchor_duplicate_conflict")
            focus[content] = item
    joined = 0
    for row in summaries:
        if row["content_sha256"] in focus:
            item = focus[row["content_sha256"]]
            if item["split"] != row["split"]:
                raise ReviewError("provided_anchor_summary_split_mismatch")
            row["provided_focus" if row["split"] == "train" else "provided_evaluation_focus"] = deepcopy(item)
            row["summary_sha256"] = _digest({key: value for key, value in row.items() if key not in {"generated_at", "summary_sha256"}})
            joined += 1
    work_joined = 0
    for row in worklist["records"]:
        item = focus.get(row["content_sha256"])
        if item is not None and item["split"] != row["split"]:
            raise ReviewError("provided_anchor_worklist_split_mismatch")
        if item is not None and row["split"] == "train":
            row["provided_focus"] = deepcopy(item)
            work_joined += 1
    return {"provided_focus_summaries_joined": joined, "train_worklist_focus_joined": work_joined}


def automatic_first_pass(store: ReviewStore, source_manifest: Path, *, publish_workbench: bool = False, split_manifest: Path | None = None, provided_anchor_manifests: list[Path] | None = None, export_track: str | None = None, proposal_index: Path | None = None) -> dict[str, int]:
    if export_track is not None and re.fullmatch("[a-z][a-z0-9_-]{0,31}", export_track) is None:
        raise ReviewError("machine_export_track_invalid")
    records = store.list_records()
    selected_hashes, selection_hash = None, None
    if proposal_index is not None:
        if export_track is None:
            raise ReviewError("machine_proposal_selection_requires_versioned_export")
        selection = json.loads(proposal_index.read_text(encoding="utf-8-sig"))
        entries = selection.get("entries") if isinstance(selection, dict) else None
        if not isinstance(selection, dict) or selection.get("schema_version") != "forte.proposal-artifact-handoff-index.v1" or not isinstance(entries, list) or not entries:
            raise ReviewError("machine_proposal_selection_index_invalid")
        selected_hashes = set()
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("terminal_verified") is not True or entry.get("artifact_available") is not True or entry.get("terminal_status") not in {"done", "failed"} or not isinstance(entry.get("audit_sha256"), str) or re.fullmatch("[a-f0-9]{64}", entry["audit_sha256"]) is None or entry["audit_sha256"] in selected_hashes:
                raise ReviewError("machine_proposal_selection_receipt_invalid")
            selected_hashes.add(entry["audit_sha256"])
        selection_hash = _sha256(proposal_index)
    eligible_records = [record for record in records if selected_hashes is None or (_latest_proposal(record) or {}).get("audit_sha256") in selected_hashes]
    summaries = [row for record in eligible_records if (row := summarize_record(record, store.workspace)) is not None]
    if selected_hashes is not None:
        if {row["provenance"]["source_proposal_sha256"] for row in summaries} != selected_hashes:
            raise ReviewError("machine_proposal_selection_not_fully_imported_or_current")
        for row in summaries:
            row["proposal_selection_index_sha256"] = selection_hash
            row["summary_sha256"] = _digest({key: value for key, value in row.items() if key not in {"generated_at", "summary_sha256"}})
    split = json.loads(split_manifest.read_text(encoding="utf-8-sig")) if split_manifest else None
    split_hash = _sha256(split_manifest) if split_manifest else None
    if split is not None:
        assignments = {}
        for source in split.get("records", []):
            content = source.get("content_sha256", source.get("content_hash"))
            value = (source["split"], source.get("scenario_group_id") or source["group_id"])
            if content in assignments and assignments[content] != value:
                raise ReviewError("machine_export_split_binding_conflict")
            assignments[content] = value
        for row in summaries:
            if row["content_sha256"] not in assignments:
                raise ReviewError("machine_export_split_source_missing")
            row["split"], row["group_id"] = assignments[row["content_sha256"]]
            row["partition_binding"] = {"split_file_sha256": split_hash, "assignments_sha256": split.get("assignments_sha256"), "origin": split.get("state"), "scenario_independence_validated": False}
            row["summary_sha256"] = _digest({key: value for key, value in row.items() if key not in {"generated_at", "summary_sha256"}})
    groups = build_group_proposals(records, summaries, _sha256(source_manifest))
    output = store.workspace / "automatic"
    if export_track is not None:
        output = output / export_track
    if not output.resolve().is_relative_to(store.workspace):
        raise ReviewError("machine_export_private_root_escape")
    output.mkdir(parents=True, exist_ok=True)
    worklist = build_annotation_worklist(records, summaries, split, split_hash)
    focus_counts = attach_provided_focus(summaries, worklist, provided_anchor_manifests, records) if provided_anchor_manifests else {}
    _write(output / "machine_annotations.jsonl", "".join(canonical(row)+"\n" for row in summaries))
    _write(output / "automatic_group_proposals.json", canonical(groups)+"\n")
    _write(output / "autonomous_annotation_worklist.json", canonical(worklist)+"\n")
    published = unchanged = conflicts = 0
    if publish_workbench:
        for row in summaries:
            record = next(value for value in records if value["sample_id"] == row["sample_id"])
            try:
                result = store.publish_machine_summary(row["sample_id"], row, expected_revision=record["revision"])
                published += result["machine_summaries_updated"]
                unchanged += result["machine_summaries_unchanged"]
            except ReviewError:
                conflicts += 1
    return {"source_contents": len(records), "recording_aliases": sum(len(row["recording_ids"]) for row in records), "machine_summaries": len(summaries), "machine_windows": sum(len(row["windows"]) for row in summaries), "source_second_bins": sum(len(row["per_second"]) for row in summaries), "model_derived_group_sources": len(summaries), "exact_content_only_group_sources": len(records)-len(summaries), "proposed_groups": len(groups["groups"]), "autonomous_worklist_sources": len(worklist["records"]), "workbench_summaries_updated": published, "workbench_summaries_unchanged": unchanged, "workbench_revision_conflicts": conflicts, "human_reviews_added": 0, "training_authorizations_added": 0, **focus_counts}
