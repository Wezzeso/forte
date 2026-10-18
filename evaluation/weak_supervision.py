"""Operational weak-supervision evaluation against independent provided labels.

This track deliberately does not require human event annotations. It evaluates
only organizer-provided whole-video binary labels on frozen, exposure-audited
automatic groups. Teacher facts/events remain silver targets, never accuracy
ground truth. No API, model, media or GPU operation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .audit import audit_split, sample_id
from .io import digest, read_json, read_records, write_json
from .metrics import distribution, number, ratio

SCHEMA = "forte.weak_operational_evaluation.v1"


def provided_label_digest(rows: list[dict]) -> str:
    """A02 label_records_digest format; newline-delimited sorted-key records."""
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    return hashlib.sha256(payload).hexdigest()


def weak_assignment_digest(rows: list[dict]) -> str:
    """A02 assignments_sha256 format; separate from human-track records_digest."""
    return hashlib.sha256((json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")).hexdigest()


def _content(row: dict) -> str:
    value = row.get("content_sha256", row.get("content_hash"))
    if not isinstance(value, str) or not value:
        raise ValueError("content identity missing")
    return value


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("missing or duplicate evaluation identity")
        result[identifier] = row
    return result


def _counts(expected: bool, row: dict | None, result: Counter) -> None:
    if not row or not row["classification_completed"]:
        result["unprocessed_positive" if expected else "unprocessed_negative"] += 1
    else:
        predicted = row["verdict"] == "violation"
        result[("tp" if predicted else "fn") if expected else ("fp" if predicted else "tn")] += 1


def _confusion() -> Counter:
    return Counter({key: 0 for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")})


def _summary(values: Counter) -> dict:
    tp, fp, fn, tn, up, un = (values[key] for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative"))
    return {**dict(values), "completed_labeled_denominator": tp + fp + fn + tn, "all_provided_label_denominator": tp + fp + fn + tn + up + un, "precision_completed": ratio(tp, tp + fp), "recall_completed": ratio(tp, tp + fn), "effective_positive_recall_including_failures": ratio(tp, tp + fn + up), "effective_correct_rate_including_failures": ratio(tp + tn, tp + fp + fn + tn + up + un)}


def evaluate_weak_binary(
    label_records: list[dict], expected_attempts: list[dict], predictions: list[dict], *,
    split_manifest: dict, experiment_config: dict, weak_gate_proof: dict, metadata: dict,
    provided_labels_sha256: str, predictions_sha256: str | None = None,
) -> dict:
    """Aggregate all attempts; unique-content confusion uses first planned try.

    Head-only classification may be selected provisionally without pretending
    that a public result, grounded event or end-to-end latency has been checked.
    A02 owns weak-group authorization/exposure checks; genuine human-gold review
    remains an independent, stricter track.
    """
    expected, actual = _index(expected_attempts, "attempt_id"), _index(predictions, "attempt_id")
    if actual.keys() - expected.keys():
        raise ValueError("unexpected attempts cannot be silently ignored")
    split_rows = {sample_id(row): row for row in split_manifest.get("records", [])}
    validation_rows = {sid: row for sid, row in split_rows.items() if row.get("split") == "validation"}
    validation_hashes = {_content(row) for row in validation_rows.values()}
    split_audit = audit_split(split_manifest)
    blockers: Counter = Counter()
    labels_by_content: dict[str, set[bool]] = defaultdict(set)
    label_sources = set()
    for label in label_records:
        content = _content(label)
        if content not in validation_hashes or label.get("split", "validation") != "validation":
            raise ValueError("only frozen validation label rows may enter this evaluator")
        label_value = label.get("provided_video_label", label.get("has_violation"))
        if label.get("provided_label_origin", label.get("label_origin")) != "organizer_workbook" or not isinstance(label_value, bool):
            blockers["labels_not_independent_organizer_binary"] += 1
            continue
        if label.get("target_scope") != "video_binary_mil_only" or label.get("teacher_used_as_binary_ground_truth") is not False or any(key in label for key in ("facts", "events", "fact_targets", "event_targets")):
            blockers["provided_binary_label_contains_unsupported_teacher_or_event_truth"] += 1
            continue
        labels_by_content[content].add(label_value)
        label_sources.add(label.get("provided_label_source_sha256"))
    labels = {content: next(iter(values)) for content, values in labels_by_content.items() if len(values) == 1}
    conflicts = sum(len(values) > 1 for values in labels_by_content.values())
    if conflicts or validation_hashes - labels.keys():
        blockers["provided_validation_labels_missing_or_conflicting"] += conflicts + len(validation_hashes - labels.keys())
    pin = experiment_config.get("expected_assignments_sha256")
    source_pin = experiment_config.get("expected_provided_label_source_sha256")
    if not pin or weak_assignment_digest(split_manifest.get("records", [])) != pin or split_manifest.get("assignments_sha256") != pin or weak_gate_proof.get("expected_assignments_sha256") != pin:
        blockers["validation_assignment_pin_mismatch"] += 1
    if not source_pin or label_sources != {source_pin} or weak_gate_proof.get("provided_label_source_sha256") != source_pin:
        blockers["independent_label_source_pin_mismatch"] += 1
    manifest_labels = split_manifest.get("provided_label_manifests", {}).get("validation", {})
    if provided_label_digest(label_records) != provided_labels_sha256 or manifest_labels.get("sha256") != provided_labels_sha256 or weak_gate_proof.get("provided_label_manifest_sha256") != provided_labels_sha256 or experiment_config.get("expected_provided_label_manifests_sha256", {}).get("validation") != provided_labels_sha256:
        blockers["provided_validation_label_artifact_mismatch"] += 1
    if experiment_config.get("authorized") is not True or not experiment_config.get("authorization_reference") or experiment_config.get("allow_provisional_groups") is not True or experiment_config.get("validation_assignments_frozen") is not True or experiment_config.get("allowed_target") != "video_binary_mil":
        blockers["explicit_weak_experiment_authorization_or_freeze_missing"] += 1
    if weak_gate_proof.get("ok") is not True or not weak_gate_proof.get("exposure_registry_tip_sha256") or not weak_gate_proof.get("coverage_snapshot_sha256"):
        blockers["latest_group_exposure_or_accepted_job_coverage_gate_missing"] += 1
    if weak_gate_proof.get("label_split", "validation") != "validation":
        blockers["weak_gate_not_bound_to_validation_labels"] += 1
    if not split_audit["integrity_passed"]:
        blockers["duplicate_or_group_split_integrity_failed"] += 1
    if split_manifest.get("state") != "automated_provisional":
        blockers["wrong_track_split_state"] += 1
    if metadata.get("inference_kind") not in {"real_local", "synthetic"} or not metadata.get("model_revision") or not metadata.get("conditions_sha256"):
        blockers["model_execution_or_condition_identity_missing"] += 1
    if metadata.get("inference_kind") == "real_local" and metadata.get("execution_verified") is not True:
        blockers["real_local_execution_unverified"] += 1
    if not metadata.get("threshold_selection_sha256"):
        blockers["frozen_threshold_artifact_missing"] += 1
    if metadata.get("comparison_scope") not in {"weak_model_only", "full_service"}:
        blockers["model_vs_service_comparison_scope_missing"] += 1
    if metadata.get("expected_attempts_frozen_before_run") is not True:
        blockers["all_attempt_denominator_not_frozen"] += 1
    if not expected or not validation_hashes:
        blockers["empty_attempt_or_validation_population"] += 1
    rows_by_content: dict[str, list[dict]] = defaultdict(list)
    normalized_rows = []
    counters: Counter = Counter()
    per_attempt = _confusion()
    full_result_latencies, terminal_latencies = [], []
    for aid, plan in expected.items():
        sid = plan.get("sample_id")
        if sid not in validation_rows:
            raise ValueError("planned prediction is not from frozen validation")
        content = _content(validation_rows[sid])
        observed = actual.get(aid, {})
        if observed and observed.get("sample_id") != sid:
            raise ValueError("attempt and sample identity mismatch")
        verdict = observed.get("verdict")
        classification_complete = observed.get("status") == "done" and verdict in {"violation", "no_violation"} and (observed.get("prediction_validated") is True or observed.get("public_contract_validated") is True)
        public_complete = classification_complete and observed.get("public_contract_validated") is True
        elapsed_ms = number(observed.get("upload_start_to_full_result_ms"))
        e2e_known = public_complete and elapsed_ms is not None and elapsed_ms >= 0 and observed.get("timing_origin") == "client_upload_start_to_full_result_byte"
        fast = e2e_known and elapsed_ms < 10_000
        if e2e_known:
            full_result_latencies.append(elapsed_ms)
        terminal_ms = number(observed.get("terminal_observed_elapsed_ms"))
        if terminal_ms is not None and terminal_ms >= 0:
            terminal_latencies.append(terminal_ms)
        counters["classification_completed"] += classification_complete
        counters["public_result_completed"] += public_complete
        counters["strict_latency_passed"] += fast
        counters["public_result_timing_unknown"] += public_complete and not e2e_known
        counters["missing_predictions"] += not bool(observed)
        counters["nonterminal_predictions"] += bool(observed) and observed.get("status") not in {"done", "failed", "timeout"}
        normalized = {"classification_completed": classification_complete, "verdict": verdict, "public_result_completed": public_complete, "strict_latency_passed": fast}
        rows_by_content[content].append(normalized)
        normalized_rows.append({"attempt_id": aid, "sample_id": sid, **normalized})
        if content in labels:
            _counts(labels[content], normalized, per_attempt)
    if counters["nonterminal_predictions"]:
        blockers["run_contains_nonterminal_predictions"] += counters["nonterminal_predictions"]
    missing_planned = validation_hashes - rows_by_content.keys()
    if missing_planned:
        blockers["frozen_validation_contents_not_planned"] += len(missing_planned)
    unique = _confusion()
    repeat_disagreements = 0
    for content in sorted(validation_hashes):
        planned_rows = rows_by_content.get(content, [])
        if content in labels:
            _counts(labels[content], planned_rows[0] if planned_rows else None, unique)
        observed_verdicts = {row["verdict"] for row in planned_rows if row["classification_completed"]}
        repeat_disagreements += len(observed_verdicts) > 1
    positive, negative = sum(labels.values()), len(labels) - sum(labels.values())
    if not positive or not negative:
        blockers["provided_positive_or_negative_coverage_missing"] += 1
    real = metadata.get("inference_kind") == "real_local"
    operational_gate = not blockers
    eligible = operational_gate and real
    n = len(expected)
    all_binary_correct = eligible and len(labels) == len(validation_hashes) and per_attempt["tp"] + per_attempt["tn"] == n and unique["tp"] + unique["tn"] == len(validation_hashes)
    all_service_fast = eligible and n > 0 and counters["strict_latency_passed"] == n
    report = {
        "schema_version": SCHEMA,
        "track": "weak_supervision_independent_provided_video_binary",
        "status": "SYNTHETIC_ONLY" if metadata.get("inference_kind") == "synthetic" else "OPERATIONAL_VALIDATION_READY" if eligible else "BLOCKED_OR_UNMEASURED",
        "evaluation_gate_passed": operational_gate,
        "model_selection_eligible": eligible,
        "training_authorized_by_report": False,
        "human_gold_track_status": "UNMEASURED_NOT_REQUIRED_FOR_WEAK_VIDEO_BINARY_EVALUATION",
        "ground_truth_scopes": {"video_binary": "independent_organizer_PROVIDED_labels_not_human_reverified", "event_types": "UNKNOWN", "facts": "UNKNOWN", "event_timing": "UNKNOWN", "evidence_support": "UNKNOWN", "legal_applicability": "UNKNOWN", "teacher_support_agreement_is_accuracy": False},
        "split_assignments_sha256": pin,
        "provided_labels_sha256": provided_labels_sha256,
        "input_set_sha256": digest(sorted(validation_hashes)),
        "attempt_population_sha256": digest(sorted(_content(validation_rows[row["sample_id"]]) for row in expected.values())),
        "expected_attempts_sha256": digest(expected_attempts),
        "predictions_sha256": predictions_sha256 or digest(predictions),
        "provenance": {"model_id": metadata.get("model_id"), "model_revision": metadata.get("model_revision"), "head_identity_sha256": digest(metadata["head_identity"]) if isinstance(metadata.get("head_identity"), dict) else None, "conditions_sha256": metadata.get("conditions_sha256"), "comparison_scope": metadata.get("comparison_scope"), "threshold_selection_sha256": metadata.get("threshold_selection_sha256"), "provided_label_source_sha256": source_pin, "experiment_config_sha256": digest(experiment_config), "weak_gate_proof_sha256": digest(weak_gate_proof), "exposure_registry_tip_sha256": weak_gate_proof.get("exposure_registry_tip_sha256"), "exposure_manifest_sha256": weak_gate_proof.get("exposure_registry_file_sha256"), "coverage_snapshot_sha256": weak_gate_proof.get("coverage_snapshot_sha256"), "group_review_origin": split_manifest.get("group_review_origin"), "human_reviewed_groups": split_manifest.get("human_reviewed") is True, "grouping_independence": "automated_provisional_not_human_established", "inference_kind": metadata.get("inference_kind")},
        "labels": {"required_validation_unique_contents": len(validation_hashes), "provided_binary_labels": len(labels), "positive": positive, "negative": negative, "missing_or_conflicting": len(validation_hashes - labels.keys()), "event_labels_from_binary": False},
        "all_attempt_completion": {"required": n, "classification_predictions_completed": counters["classification_completed"], "classification_failed_or_unprocessed": n - counters["classification_completed"], "classification_completion_rate": ratio(counters["classification_completed"], n), "public_results_completed": counters["public_result_completed"], "public_results_failed_or_unprocessed": n - counters["public_result_completed"], "public_result_completion_rate": ratio(counters["public_result_completed"], n), "missing_predictions": counters["missing_predictions"], "nonterminal_predictions": counters["nonterminal_predictions"]},
        "whole_video_per_unique_content": {**_summary(unique), "policy": "first frozen planned attempt per unique content; repeats do not increase independent sample size", "repeat_verdict_disagreement_contents": repeat_disagreements},
        "whole_video_per_attempt": _summary(per_attempt),
        "end_to_end_10s": {"comparison": "strictly_less_than_10000_ms", "origin": "client_upload_start_to_full_result_byte", "required_attempts": n, "passed": counters["strict_latency_passed"], "failed_or_unmeasured": n - counters["strict_latency_passed"], "rate": ratio(counters["strict_latency_passed"], n), "completed_public_timing_unknown": counters["public_result_timing_unknown"], "upload_start_to_full_result_ms": distribution(full_result_latencies), "terminal_observed_elapsed_ms_diagnostic": distribution(terminal_latencies), "head_compute_time_substitutes_for_service_e2e": False},
        "target_observations": {"provided_binary_detection_target_met": all_binary_correct, "all_attempt_public_e2e_10s_target_met": all_service_fast, "human_event_grounding_100_percent_target_met": False, "universal_accuracy_claim_permitted": False},
        "gate_blocker_counts": dict(sorted(blockers.items())),
        "limitations": ["Operational weak supervision may proceed without manual annotations when the explicit weak gate passes; it does not satisfy the human-gold track.", "Provided binary labels are independent of teacher outputs but have not been human reverified by this evaluator.", "Automatic grouping and exposure audits prevent known leakage, not proof that all scenarios are independent.", "Teacher agreement, pseudo facts, event types, event timing, evidence and legal applicability are not independently evaluated accuracy here.", "Weak head classification completion is distinct from public-result completion and upload-start-to-last-byte latency.", "A finite validation result, even 100% against provided labels, is not universal accuracy or a pristine human-gold result."],
    }
    report["provided_evaluation_attestation_sha256"] = digest(report)
    return report


def validate_weak_attestation(report: dict, expected_bindings: dict[str, str] | None = None) -> dict:
    """Verify artifact integrity/bindings, not authorship or human ground truth."""
    body = {key: value for key, value in report.items() if key != "provided_evaluation_attestation_sha256"}
    errors = []
    if report.get("schema_version") != SCHEMA or report.get("provided_evaluation_attestation_sha256") != digest(body):
        errors.append("invalid_attestation_integrity")
    for key, value in (expected_bindings or {}).items():
        if key not in {"split_assignments_sha256", "provided_labels_sha256", "input_set_sha256", "predictions_sha256", "provided_evaluation_attestation_sha256"} or report.get(key) != value:
            errors.append("binding_mismatch")
    return {"ok": not errors, "errors": sorted(set(errors)), "operational_validation_eligible": not errors and report.get("model_selection_eligible") is True, "human_event_truth_established": False}


def select_weak_candidate(baseline: dict, candidate: dict) -> dict:
    """Select an operational weak model, never claim human-event readiness."""
    blockers = []
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if not validate_weak_attestation(report)["operational_validation_eligible"]:
            blockers.append(name + ":operational_evaluation_ineligible")
    for key in ("split_assignments_sha256", "provided_labels_sha256", "input_set_sha256", "attempt_population_sha256", "labels"):
        if baseline.get(key) != candidate.get(key):
            blockers.append("incomparable_" + key)
    for key in ("conditions_sha256", "comparison_scope", "provided_label_source_sha256", "experiment_config_sha256"):
        if baseline["provenance"].get(key) != candidate["provenance"].get(key):
            blockers.append("incomparable_" + key)
    if baseline["all_attempt_completion"]["required"] != candidate["all_attempt_completion"]["required"]:
        blockers.append("incomparable_attempt_count")
    diagnostics = {}
    service = candidate["provenance"].get("comparison_scope") == "full_service"
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        whole, attempts = report["whole_video_per_unique_content"], report["all_attempt_completion"]
        axes = {"unique_video_total_errors_including_failures": whole["fp"] + whole["fn"] + whole["unprocessed_positive"] + whole["unprocessed_negative"], "classification_unprocessed_attempts": attempts["classification_failed_or_unprocessed"], "repeat_verdict_disagreements": whole["repeat_verdict_disagreement_contents"]}
        if service:
            axes["public_result_unprocessed_attempts"] = attempts["public_results_failed_or_unprocessed"]
            axes["public_e2e_10s_failed_or_unmeasured"] = report["end_to_end_10s"]["failed_or_unmeasured"]
        diagnostics[name] = axes
    improved = [key for key in diagnostics["candidate"] if diagnostics["candidate"][key] < diagnostics["baseline"][key]]
    regressed = [key for key in diagnostics["candidate"] if diagnostics["candidate"][key] > diagnostics["baseline"][key]]
    if regressed:
        blockers.append("guarded_operational_regression")
    if not improved:
        blockers.append("no_observed_operational_error_or_slo_improvement")
    result = {"schema_version": "forte.weak_operational_selection.v1", "selected": "candidate" if not blockers else "baseline", "candidate_earned_provisional_selection": not blockers, "selection_scope": "provisional_weak_model_only" if not service else "provisional_weak_model_and_observed_service", "policy": "reduce total unique provided-label binary errors (including failures) without worsening classification completion/repeat consistency; full-service comparisons also guard completion and <10s", "blockers": sorted(set(blockers)), "improved_axes": improved, "regressed_axes": regressed, "guarded_diagnostics": diagnostics, "per_class_diagnostics": {name: {key: report["whole_video_per_unique_content"][key] for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")} for name, report in (("baseline", baseline), ("candidate", candidate))}, "candidate_evaluation_attestation_sha256": candidate.get("provided_evaluation_attestation_sha256"), "baseline_evaluation_attestation_sha256": baseline.get("provided_evaluation_attestation_sha256"), "human_event_ground_truth_established": False, "teacher_agreement_is_accuracy": False, "strict_human_gold_acceptance_achieved": False, "note": "Equal-count binary-error selection is a disclosed team policy, not official rubric scoring; per-class tradeoffs remain visible. Provisional progress may occur below 100%/10s and does not relabel unmet targets pass."}
    result["selection_attestation_sha256"] = digest(result)
    return result


def verify_provided_evaluation_for_manifest(report: dict, manifest: dict, selection_report: dict | None = None) -> dict:
    """A08 adapter for the provided-label portion of machine evidence only.

    This does not verify teacher/support or threshold-file contents. A08 must
    compose those independent checks; no permissive full verifier is created.
    Hashes establish artifact binding/integrity, not external authorship.
    """
    if hasattr(manifest, "model_dump"):
        manifest = manifest.model_dump(mode="json")
    bindings = {key: manifest.get(key) for key in ("split_assignments_sha256", "provided_labels_sha256", "input_set_sha256", "predictions_sha256", "provided_evaluation_attestation_sha256")}
    integrity = validate_weak_attestation(report, bindings)
    provenance = report.get("provenance", {})
    identity_ok = isinstance(manifest.get("identity"), dict) and provenance.get("head_identity_sha256") == digest(manifest["identity"])
    threshold_ok = bool(provenance.get("threshold_selection_sha256")) and provenance["threshold_selection_sha256"] == manifest.get("threshold_selection_sha256")
    exposure_ok = bool(provenance.get("exposure_manifest_sha256")) and provenance["exposure_manifest_sha256"] == manifest.get("exposure_manifest_sha256")
    selection_ok = False
    if selection_report is not None:
        body = {key: value for key, value in selection_report.items() if key != "selection_attestation_sha256"}
        selected_key = "candidate_evaluation_attestation_sha256" if selection_report.get("selected") == "candidate" else "baseline_evaluation_attestation_sha256"
        selection_ok = (selection_report.get("selection_attestation_sha256") == digest(body) == manifest.get("selection_attestation_sha256") and selection_report.get(selected_key) == report.get("provided_evaluation_attestation_sha256"))
    verified = integrity["operational_validation_eligible"] and identity_ok and threshold_ok and exposure_ok and selection_ok
    targets = report.get("target_observations", {})
    return {"provided_label_evaluation_verified": verified, "artifact_integrity_verified": integrity["ok"], "model_identity_binding_verified": identity_ok, "threshold_evaluation_binding_verified": threshold_ok, "exposure_artifact_binding_verified": exposure_ok, "selection_binding_verified": selection_ok, "provided_binary_target_met": targets.get("provided_binary_detection_target_met") if verified else None, "latency_target_met": targets.get("all_attempt_public_e2e_10s_target_met") if verified else None, "teacher_support_agreement_verified": False, "threshold_file_contents_verified": False, "independent_fact_event_accuracy_measured": False, "note": "A08 must additionally verify actual teacher/support and frozen threshold contents; those are not provided-label ground truth."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate", help="Run A02's latest exposure/group gate for validation labels")
    for name in ("experiment-config", "provided-labels", "output"):
        gate.add_argument("--" + name, required=True)
    run = commands.add_parser("evaluate")
    for name in ("provided-labels", "expected-attempts", "predictions", "splits", "experiment-config", "weak-gate-proof", "metadata", "output"):
        run.add_argument("--" + name, required=True)
    choose = commands.add_parser("select")
    for name in ("baseline", "candidate", "output"):
        choose.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "gate":
            from scripts.split_gate import validate_automated_experiment_inputs
            config = read_json(args.experiment_config)
            authorization = config.get("experiment_authorization", config)
            label_rows = read_records(args.provided_labels)
            report = validate_automated_experiment_inputs(read_json(config.get("automated_split", config.get("split_path"))), read_records(config.get("source_manifest", "data/manifests/recordings.jsonl")), read_json(config.get("scenario_groups", config.get("scenario_path"))), experiment_authorization=authorization, registry_path=Path(config.get("exposure_registry", config.get("registry_path"))), coverage_status_path=Path(config.get("exposure_inventory_status", config.get("coverage_status_path"))), provided_label_records=label_rows, expected_provided_label_source_sha256=config.get("expected_provided_label_source_sha256") or authorization["expected_provided_label_source_sha256"], label_split="validation", requested_content_hashes={_content(row) for row in label_rows}, source_inventory_path_override=Path(config["source_inventory_path_override"]) if config.get("source_inventory_path_override") else None)
            report["exposure_registry_file_sha256"] = hashlib.sha256(Path(config.get("exposure_registry", config.get("registry_path"))).read_bytes()).hexdigest()
            summary = {"ok": report.get("ok") is True, "label_split": "validation", "human_event_ground_truth_established": False}
            success = report.get("ok") is True
        elif args.command == "evaluate":
            config = read_json(args.experiment_config)
            label_rows = read_records(args.provided_labels)
            report = evaluate_weak_binary(label_rows, read_records(args.expected_attempts), read_records(args.predictions), split_manifest=read_json(args.splits), experiment_config=config.get("experiment_authorization", config), weak_gate_proof=read_json(args.weak_gate_proof), metadata=read_json(args.metadata), provided_labels_sha256=provided_label_digest(label_rows), predictions_sha256=hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest())
            summary = {"status": report["status"], "required_attempts": report["all_attempt_completion"]["required"], "classification_completed": report["all_attempt_completion"]["classification_predictions_completed"], "public_results_completed": report["all_attempt_completion"]["public_results_completed"], "human_event_truth": "UNKNOWN"}
            success = report["model_selection_eligible"]
        else:
            report = select_weak_candidate(read_json(args.baseline), read_json(args.candidate))
            summary = {"selected": report["selected"], "candidate_earned_provisional_selection": report["candidate_earned_provisional_selection"], "human_event_ground_truth_established": False}
            success = report["candidate_earned_provisional_selection"]
        write_json(args.output, report)
        print(json.dumps(summary, sort_keys=True))
        return 0 if success else 2
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"error": "weak_operational_evaluation_input_invalid", "private_values_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
