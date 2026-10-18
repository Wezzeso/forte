"""Offline, failure-inclusive comparison of three pinned Qwen development arms.

This module performs no inference, training, media access or network operation.
It consumes a frozen development-only attempt plan and independently supplied
organizer binary labels. Coarse points are optional post-prediction diagnostics,
never invented event/type/fact/boundary or legal ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .io import digest, read_json, read_records, write_json
from .metrics import anchor, number, ratio
from .weak_supervision import provided_label_digest

ARMS = ("baseline_8b", "qwen38_untuned", "qwen38_tuned")
DEV_ROLES = {"train", "validation"}
IDENTITY_FIELDS = ("arm", "sample_id", "content_sha256", "split", "repeat_index")
MODEL_FIELDS = ("model_id", "model_revision", "adapter_sha256")
FAILURE_STATUSES = {"failed", "timeout", "upload_failed", "admission_rejected", "invalid_json", "insufficient_analysis", "not_attempted", "cancelled", "processing", "queued", "running", "model_identity_unverified"}
FULL_RESULT_ORIGIN = "client_upload_start_to_full_result_byte"
UI_RESULT_ORIGIN = "client_upload_start_to_usable_rendered_result"
PLAN_SCHEMA = "forte.qwen38_comparison_plan.v1"
REPORT_SCHEMA = "forte.qwen38_development_comparison.v1"


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise ValueError("record population must be a list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("record must be an object")
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or value in indexed:
            raise ValueError("missing or duplicate record identity")
        indexed[value] = row
    return indexed


def _role(row: dict) -> str:
    # Explicit role is mandatory. A filename, model response or caller default
    # cannot turn an unknown/held-out source into a development source.
    role = row.get("split")
    if role not in DEV_ROLES:
        raise ValueError("only explicit train/validation roles are permitted")
    return role


def _content(row: dict) -> str:
    content = row.get("content_sha256", row.get("content_hash"))
    if not _sha(content) or ("content_hash" in row and row["content_hash"] != content):
        raise ValueError("missing or conflicting content identity")
    return content


def validate_plan(plan: dict) -> tuple[dict, dict]:
    """Validate development scope before opening result/reference artifacts."""
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("invalid comparison plan schema")
    records = _index(plan.get("records"), "sample_id")
    if not records:
        raise ValueError("comparison population must not be empty")
    seen_contents = set()
    for record in records.values():
        _role(record)
        content = _content(record)
        if content in seen_contents:
            raise ValueError("aliases cannot increase independent sample size")
        seen_contents.add(content)
        duration = record.get("duration_sec")
        if duration is not None and (number(duration) is None or duration <= 0):
            raise ValueError("invalid source duration")
    arms = plan.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("exactly baseline, untuned and tuned arms are required")
    for name, identity in arms.items():
        if not isinstance(identity, dict) or any(not isinstance(identity.get(k), str) or not identity[k].strip() for k in MODEL_FIELDS[:2]):
            raise ValueError("arm needs actual model identifier and immutable revision")
        if re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64}|sha256:[a-f0-9]{64})", identity["model_revision"]) is None:
            raise ValueError("model revision must be an immutable commit or artifact digest")
        if "adapter_sha256" not in identity:
            raise ValueError("adapter identity must be explicit")
        if name == "qwen38_tuned":
            if not _sha(identity["adapter_sha256"]):
                raise ValueError("tuned arm needs a real adapter artifact hash")
        elif identity["adapter_sha256"] is not None:
            raise ValueError("baseline and untuned arms cannot carry an adapter")
    if any(arms["qwen38_untuned"][key] != arms["qwen38_tuned"][key] for key in MODEL_FIELDS[:2]):
        raise ValueError("untuned and tuned arms must use the same base revision")
    recipe = plan.get("recipe")
    required = {"sampling", "prompt_sha256", "generation", "evidence", "timing", "concurrency", "supported_envelope"}
    if not isinstance(recipe, dict) or not required.issubset(recipe) or any(recipe[key] is None for key in required):
        raise ValueError("shared recipe must declare sampling, prompt, generation, evidence, timing and envelope")
    if not _sha(recipe["prompt_sha256"]) or plan.get("recipe_sha256") != digest(recipe):
        raise ValueError("shared recipe hash mismatch")
    if not _sha(plan.get("provided_labels_sha256")):
        raise ValueError("binary reference artifact must be pinned")
    if plan.get("coarse_anchors_sha256") is not None and not _sha(plan["coarse_anchors_sha256"]):
        raise ValueError("invalid coarse reference pin")
    expected = _index(plan.get("attempts"), "attempt_id")
    populations: dict[str, set[tuple[str, int]]] = {name: set() for name in ARMS}
    for attempt in expected.values():
        sid, arm, repeat = attempt.get("sample_id"), attempt.get("arm"), attempt.get("repeat_index")
        if sid not in records or arm not in ARMS or isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise ValueError("invalid planned attempt population")
        if _role(attempt) != records[sid]["split"] or _content(attempt) != _content(records[sid]):
            raise ValueError("planned attempt source/role mismatch")
        pair = (sid, repeat)
        if pair in populations[arm]:
            raise ValueError("duplicate arm/sample/repeat cannot be silently merged")
        populations[arm].add(pair)
    if not populations[ARMS[0]] or any(populations[name] != populations[ARMS[0]] for name in ARMS[1:]):
        raise ValueError("all three arms must have exactly the same frozen attempt population")
    if {sid for sid, _ in populations[ARMS[0]]} != set(records):
        raise ValueError("every declared source must be planned in every arm")
    for sid in records:
        indices = sorted(repeat for item, repeat in populations[ARMS[0]] if item == sid)
        if indices != list(range(len(indices))):
            raise ValueError("repeat indices must start at zero with no gaps")
    return records, expected


def _references(rows: list[dict], records: dict, *, coarse: bool = False) -> dict[str, dict]:
    by_content = {_content(record): record for record in records.values()}
    indexed = {}
    for row in rows:
        content, role = _content(row), _role(row)
        if content not in by_content or role != by_content[content]["split"] or content in indexed:
            raise ValueError("reference population/role or duplicate-content mismatch")
        if not isinstance(row.get("provided_video_label"), bool):
            raise ValueError("provided binary reference must be a boolean")
        if coarse:
            if row.get("anchor_origin") != "user_provided_coarse_table" or row.get("anchor_semantics") != "approximate_point_not_boundary" or row.get("target_scope") != "class_agnostic_approximate_event_anchor":
                raise ValueError("coarse references cannot masquerade as exact event gold")
            if row.get("allowed_use") != "evaluation_only_do_not_condition_inference":
                raise ValueError("coarse reference allowed-use mismatch")
            if any(row.get(key) is not False for key in ("boundaries_known", "event_type_known", "legal_ground_truth")):
                raise ValueError("coarse references must disclose unknown scopes")
            points = row.get("coarse_anchor_sec")
            if not isinstance(points, list) or any(number(point) is None or point < 0 for point in points) or len(set(points)) != len(points) or bool(points) != row["provided_video_label"]:
                raise ValueError("invalid coarse points")
            duration = number(by_content[content].get("duration_sec"))
            if duration is not None and any(point > duration for point in points):
                raise ValueError("coarse point outside source timeline")
        elif row.get("provided_label_origin") != "organizer_workbook" or row.get("target_scope") != "video_binary_mil_only" or row.get("teacher_used_as_binary_ground_truth") is not False:
            raise ValueError("binary reference provenance or scope mismatch")
        indexed[content] = row
    return indexed


def _confusion() -> Counter:
    return Counter({key: 0 for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")})


def _count(values: Counter, label: bool, result: dict) -> None:
    if not result["completed"]:
        values["unprocessed_positive" if label else "unprocessed_negative"] += 1
    else:
        predicted = result["verdict"] == "violation"
        values[("tp" if predicted else "fn") if label else ("fp" if predicted else "tn")] += 1


def _summary(values: Counter) -> dict:
    tp, fp, fn, tn, up, un = (values[key] for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative"))
    return {**dict(values), "completed_labeled_denominator": tp + fp + fn + tn,
            "all_labeled_denominator_including_failures": tp + fp + fn + tn + up + un,
            "effective_positive_recall_including_failures": ratio(tp, tp + fn + up),
            "effective_correct_rate_including_failures": ratio(tp + tn, tp + fp + fn + tn + up + un)}


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    # This intentionally does not manufacture a p95 from six reused contents.
    return {"count": len(ordered), "p50": ordered[(len(ordered) - 1) // 2] if ordered else None,
            "max": ordered[-1] if ordered else None, "p95": None,
            "p95_status": "NOT_REPORTED_SMALL_REUSED_DEVELOPMENT_POPULATION"}


def _normalize(row: dict | None, record: dict) -> dict:
    if row is None:
        return {"completed": False, "verdict": None, "points": [], "failure": "missing_attempt", "full_ms": None, "ui_ms": None}
    payload = row.get("complete_result", row)
    completed = row.get("status") == "done" and row.get("public_contract_validated") is True and isinstance(payload, dict)
    payload = payload if isinstance(payload, dict) else {}
    verdict, points = payload.get("verdict"), []
    events = payload.get("violations", payload.get("events"))
    completed = completed and isinstance(verdict, str) and verdict in {"violation", "no_violation"} and isinstance(events, list) and payload.get("status", "done") == "done"
    if completed:
        completed = bool(events) == (verdict == "violation")
        duration = number(record.get("duration_sec"))
        for event in events:
            point = anchor(event) if isinstance(event, dict) else None
            kind = event.get("type", event.get("event_type_id")) if isinstance(event, dict) else None
            description = event.get("description") if isinstance(event, dict) else None
            if point is None or point < 0 or (duration is not None and point > duration) or not isinstance(kind, str) or not kind.strip() or not isinstance(description, str) or not description.strip():
                completed = False
                break
            points.append(point)
    # Trusted upstream public-contract/JPEG validation is required in addition
    # to these coherence checks. No booleans below constitute a media audit.
    full_ms, ui_ms = number(row.get("upload_start_to_full_result_ms")), number(row.get("upload_start_to_usable_result_ms"))
    full_ms = full_ms if completed and row.get("timing_origin") == FULL_RESULT_ORIGIN and full_ms is not None and full_ms >= 0 else None
    ui_ms = ui_ms if completed and row.get("ui_timing_origin") == UI_RESULT_ORIGIN and ui_ms is not None and full_ms is not None and ui_ms >= full_ms else None
    status = row.get("status")
    failure = status if isinstance(status, str) and status in FAILURE_STATUSES else "missing_status" if status is None else "other_failure_status"
    return {"completed": bool(completed), "verdict": verdict if completed else None,
            "points": points if completed else [], "failure": None if completed else "invalid_success_payload" if status == "done" else failure, "full_ms": full_ms, "ui_ms": ui_ms}


def compare_qwen38(plan: dict, results: list[dict], labels: list[dict], *, metadata: dict,
                   coarse_anchors: list[dict] | None = None, coarse_tolerances_sec: tuple[float, ...] = (1.0, 2.0, 5.0)) -> dict:
    records, expected = validate_plan(plan)
    actual = _index(results, "attempt_id")
    if actual.keys() - expected.keys():
        raise ValueError("unplanned attempts cannot be discarded")
    references = _references(labels, records)
    if plan["provided_labels_sha256"] != provided_label_digest(labels):
        raise ValueError("provided binary reference hash mismatch")
    coarse = _references(coarse_anchors or [], records, coarse=True)
    if coarse_anchors is not None:
        if plan.get("coarse_anchors_sha256") != provided_label_digest(coarse_anchors):
            raise ValueError("coarse reference artifact hash mismatch")
        if any(content not in references or row["provided_video_label"] != references[content]["provided_video_label"] for content, row in coarse.items()):
            raise ValueError("coarse and binary reference disagreement")
    if not coarse_tolerances_sec or any(number(value) is None or value < 0 for value in coarse_tolerances_sec):
        raise ValueError("invalid coarse diagnostic tolerance")
    plan_sha = digest(plan)
    actual_model_verified = set()
    for aid, row in actual.items():
        planned = expected[aid]
        if any(row.get(key) != planned[key] for key in IDENTITY_FIELDS) or isinstance(row.get("repeat_index"), bool):
            raise ValueError("result arm/sample/content/role/repeat attribution mismatch")
        _role(row)
        if row.get("comparison_plan_sha256") != plan_sha or row.get("recipe_sha256") != plan["recipe_sha256"]:
            raise ValueError("result frozen plan/shared recipe mismatch")
        identity = row.get("model_identity", row if all(key in row for key in MODEL_FIELDS) else None)
        if identity is None:
            # Upload/admission failures may precede any inference. Bind their
            # intended arm without inventing a loaded model identity.
            intended = row.get("expected_model_identity")
            if not isinstance(intended, dict) or any(key not in intended or intended[key] != plan["arms"][planned["arm"]][key] for key in MODEL_FIELDS):
                raise ValueError("failure needs explicit intended model attribution")
        elif not isinstance(identity, dict) or any(key not in identity or identity[key] != plan["arms"][planned["arm"]][key] for key in MODEL_FIELDS):
            raise ValueError("result model/base/adapter attribution mismatch")
        else:
            actual_model_verified.add(aid)
    blockers = Counter()
    for key, required in (("attempt_plan_frozen_before_run", True), ("validation_reused_for_development", True), ("final_check_used", False), ("execution_verified", True), ("exposure_audit_passed", True), ("fresh_upload_decode_inference_verified", True)):
        if metadata.get(key) is not required:
            blockers[key + "_unverified"] += 1
    if metadata.get("inference_kind") != "real_local":
        blockers["real_local_inference_unverified"] += 1
    if metadata.get("comparison_plan_sha256") != plan_sha or not _sha(metadata.get("exposure_audit_sha256")):
        blockers["frozen_plan_or_fresh_exposure_binding_unverified"] += 1
    verified_arms = metadata.get("verified_model_identity_receipts", {})
    if not isinstance(verified_arms, dict) or any(not _sha(verified_arms.get(name)) for name in ARMS):
        blockers["actual_model_identity_receipts_unverified"] += 1
    if set(_content(row) for row in records.values()) != set(references):
        blockers["binary_reference_coverage_incomplete"] += 1
    if len(actual) < len(expected):
        blockers["planned_attempts_missing"] += len(expected) - len(actual)
    claimed_success_without_model = sum(row.get("status") == "done" and aid not in actual_model_verified for aid, row in actual.items())
    if claimed_success_without_model:
        blockers["successful_result_actual_model_identity_unverified"] += claimed_success_without_model
    if any(row.get("status") in ("processing", "queued", "running") for row in actual.values()):
        blockers["nonterminal_attempts"] += 1
    if coarse and metadata.get("coarse_anchors_used_to_condition_inference") is not False:
        blockers["coarse_anchor_post_prediction_use_unverified"] += 1

    reports = {}
    for arm in ARMS:
        arm_expected = sorted((row for row in expected.values() if row["arm"] == arm), key=lambda row: (row["sample_id"], row["repeat_index"]))
        per_attempt, unique = _confusion(), _confusion()
        per_attempt_roles = {role: _confusion() for role in sorted(DEV_ROLES)}
        unique_roles = {role: _confusion() for role in sorted(DEV_ROLES)}
        failures, by_sample = Counter(), defaultdict(list)
        full_times, ui_times = [], []
        role_counts = Counter()
        counts = Counter()
        for planned in arm_expected:
            sid = planned["sample_id"]
            observed = actual.get(planned["attempt_id"])
            if observed is not None and observed.get("status") == "done" and planned["attempt_id"] not in actual_model_verified:
                observed = {**observed, "status": "model_identity_unverified"}
            normalized = _normalize(observed, records[sid])
            by_sample[sid].append(normalized)
            role_counts[planned["split"]] += 1
            counts["completed"] += normalized["completed"]
            counts["missing_attempts"] += planned["attempt_id"] not in actual
            counts["actual_model_identity_unknown_attempts"] += planned["attempt_id"] not in actual_model_verified
            if not normalized["completed"]:
                failures[str(normalized["failure"])] += 1
            for field, values in (("full_ms", full_times), ("ui_ms", ui_times)):
                if normalized[field] is not None:
                    values.append(normalized[field])
                counts[field + "_under_10s"] += normalized[field] is not None and normalized[field] < 10_000
            reference = references.get(_content(records[sid]))
            if reference is not None:
                _count(per_attempt, reference["provided_video_label"], normalized)
                _count(per_attempt_roles[planned["split"]], reference["provided_video_label"], normalized)
        coarse_distances, bands = [], Counter()
        points_total, predicted_points, failed_points = 0, 0, 0
        disagreements = 0
        for sid, attempts in by_sample.items():
            first = attempts[0]  # repeat_index zero; no best-of-repeats policy.
            content = _content(records[sid])
            if content in references:
                _count(unique, references[content]["provided_video_label"], first)
                _count(unique_roles[records[sid]["split"]], references[content]["provided_video_label"], first)
            disagreements += len({item["verdict"] for item in attempts if item["completed"]}) > 1
            if content in coarse:
                points = coarse[content]["coarse_anchor_sec"]
                points_total += len(points)
                predicted_points += len(first["points"])
                failed_points += len(points) if not first["completed"] else 0
                for point in points:
                    if first["points"]:
                        distance = min(abs(point - prediction) for prediction in first["points"])
                        coarse_distances.append(distance)
                        for tolerance in coarse_tolerances_sec:
                            bands[tolerance] += distance <= tolerance
        n = len(arm_expected)
        reports[arm] = {
            "model_identity": plan["arms"][arm],
            "completion": {"required_attempts": n, "completed_public_results": counts["completed"], "failed_or_unprocessed": n - counts["completed"], "missing_attempts": counts["missing_attempts"], "actual_model_identity_unknown_attempts": counts["actual_model_identity_unknown_attempts"], "failure_counts": dict(sorted(failures.items())), "rate": ratio(counts["completed"], n), "planned_attempts_by_role": dict(role_counts)},
            "binary_per_attempt": _summary(per_attempt),
            "binary_per_attempt_by_role": {role: _summary(values) for role, values in per_attempt_roles.items()},
            "binary_per_unique_content": {**_summary(unique), "policy": "first frozen attempt (repeat_index zero) per content", "repeat_verdict_disagreement_contents": disagreements},
            "binary_per_unique_content_by_role": {role: _summary(values) for role, values in unique_roles.items()},
            "timing": {"required_attempts": n, "full_result_bytes": {"origin": FULL_RESULT_ORIGIN, "latency_ms_completed_with_valid_origin": _distribution(full_times), "strictly_under_10000_ms": counts["full_ms_under_10s"], "failed_late_or_unmeasured": n - counts["full_ms_under_10s"], "all_attempt_rate": ratio(counts["full_ms_under_10s"], n)}, "usable_rendered_result": {"origin": UI_RESULT_ORIGIN, "latency_ms_completed_with_valid_origin": _distribution(ui_times), "strictly_under_10000_ms": counts["ui_ms_under_10s"], "failed_late_or_unmeasured": n - counts["ui_ms_under_10s"], "all_attempt_rate": ratio(counts["ui_ms_under_10s"], n)}, "all_required_usable_results_under_10s_observed": not blockers and counts["ui_ms_under_10s"] == n},
            "coarse_point_diagnostics": {"status": "APPROXIMATE_POINT_DIAGNOSTIC_ONLY" if coarse else "UNMEASURED", "provided_points": points_total, "points_with_failed_first_attempt": failed_points, "returned_prediction_points_on_reference_contents": predicted_points, "bands": [{"tolerance_sec": tolerance, "hit_points": bands[tolerance], "failure_inclusive_hit_rate": ratio(bands[tolerance], points_total)} for tolerance in coarse_tolerances_sec], "nearest_prediction_to_provided_point_sec_not_onset_error": _distribution(coarse_distances), "exact_event_precision_or_f1_available": False},
        }
    deltas = {}
    baseline = reports["baseline_8b"]
    for arm in ARMS[1:]:
        candidate = reports[arm]
        deltas[arm] = {key: candidate["binary_per_unique_content"][key] - baseline["binary_per_unique_content"][key] for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")}
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "SYNTHETIC_ONLY" if metadata.get("inference_kind") == "synthetic" else "UNMEASURED" if not actual else "PROVENANCE_INCOMPLETE_OR_PARTIAL" if blockers else "DEVELOPMENT_DIAGNOSTIC_READY",
        "scope": "same_frozen_recipe_reused_development_contents_three_pinned_arms",
        "arms": reports, "unique_content_confusion_deltas_from_baseline": deltas,
        "reference_scopes": {"video_binary": "organizer_workbook_provided_not_human_reverified", "development_roles_present": sorted({row["split"] for row in records.values()}), "binary_reference_contents": len(references), "coarse_reference_contents": len(coarse), "validation_is_reused_development": True, "scenario_independence": "NOT_ESTABLISHED_BY_THIS_REPORT", "event_types": "UNMEASURED_NO_GENUINE_REFERENCE", "visible_facts": "UNMEASURED_NO_GENUINE_REFERENCE", "evidence_support": "UNMEASURED_NO_GENUINE_REFERENCE", "legal_applicability": "UNMEASURED_NO_GENUINE_REFERENCE", "event_boundaries_or_true_onset": "UNMEASURED_NO_GENUINE_REFERENCE"},
        "bindings": {"comparison_plan_sha256": plan_sha, "recipe_sha256": plan["recipe_sha256"], "provided_labels_sha256": plan["provided_labels_sha256"], "coarse_anchors_sha256": provided_label_digest(coarse_anchors) if coarse_anchors is not None else None, "results_sha256": digest(results), "metadata_sha256": digest(metadata)},
        "provenance_blocker_counts": dict(sorted(blockers.items())),
        "claim_limits": {"model_promotion_authorized": False, "training_authorized": False, "heldout_accuracy_claim_permitted": False, "universal_100_percent_claim_permitted": False, "millisecond_semantic_onset_claim_permitted": False, "teacher_agreement_is_independent_accuracy": False, "final_check_read_or_evaluated": False},
        "limitations": ["Every frozen attempt, including failed uploads, admission rejection, invalid JSON, insufficient analysis, timeouts and missing terminals, remains in completion and applicable labeled denominators.", "Attribution/role/model/recipe mismatches invalidate the input; they are never silently dropped or converted to clean negatives.", "Repeated attempts do not create independent videos; unique-content metrics use repeat zero even when a later repeat succeeds.", "Shared recipe hashes and upstream verification flags bind supplied artifacts; this offline report cannot independently prove actual model execution, source/JPEG provenance or the absence of undisclosed attempts.", "Validation is reused development data. Automatic groups and the small number of contents do not establish scenario independence or generalization.", "Coarse timing points are approximate positive cues with unknown uncertainty, class, complete-event inventory and boundaries. Distance/hit bands are not exact localization error, event precision/F1 or legal accuracy.", "No genuine fact/type/support/legal/boundary reference export is consumed here. Use the separate human-reviewed evaluator if those references are later independently established.", "Full JSON/evidence bytes precede usable rendering. A bytes-only HTTP run cannot establish the user's usable-result <10-second target; load/warmup and human-review clocks require separate reporting.", "No candidate is promoted by this report; a larger model name or an adapter file does not establish improvement. Final-check is outside this command's accepted scope."],
    }
    report["report_sha256"] = digest(report)
    return report


def _dev_input_path(value: str) -> str:
    # Catch accidental held-out filename selection *before* reading any bytes.
    # Role checks remain mandatory because filenames cannot establish identity.
    compact = re.sub(r"[^a-z0-9]", "", str(value).lower())
    if "finalcheck" in compact:
        raise ValueError("held-out input paths are forbidden")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--coarse-anchors", action="append")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        all_inputs = [args.plan, args.metadata, *args.results, *args.labels, *(args.coarse_anchors or [])]
        for path in all_inputs:
            _dev_input_path(path)
        if Path(args.output).resolve() in {Path(path).resolve() for path in all_inputs}:
            raise ValueError("comparison output must not overwrite source artifacts")
        plan = read_json(args.plan)
        validate_plan(plan)
        labels = [row for path in args.labels for row in read_records(path)]
        results = [row for path in args.results for row in read_records(path)]
        anchors = [row for path in args.coarse_anchors for row in read_records(path)] if args.coarse_anchors else None
        report = compare_qwen38(plan, results, labels, metadata=read_json(args.metadata), coarse_anchors=anchors)
        write_json(args.output, report)
        print(json.dumps({"status": report["status"], "required_attempts": sum(row["completion"]["required_attempts"] for row in report["arms"].values()), "private_values_printed": False, "model_promotion_authorized": False}, sort_keys=True))
        return 0 if report["status"] == "DEVELOPMENT_DIAGNOSTIC_READY" else 2
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"error": "qwen38_comparison_input_invalid", "private_values_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
