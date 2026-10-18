"""Conservative, reproducible baseline-versus-adapter gates."""

from __future__ import annotations


def _primary(report: dict) -> dict:
    return next(item for item in report["events"]["by_tolerance"] if item["tolerance_sec"] == report["events"]["primary_tolerance_sec"])


def select_candidate(baseline: dict, candidate: dict, *, minimum_reviewed: int = 6) -> dict:
    """A team policy, not a statistical test or an organizer scoring formula.

    Selection needs at least one observed error reduction without a regression
    in the guarded axes. Untuned baseline stays selected for missing evidence.
    """
    blockers = []
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        truth = report["ground_truth"]
        provenance = report["provenance"]
        audit = report.get("split_audit") or {}
        if not audit.get("training_gate_passed"):
            blockers.append(name + ":split_not_locked_or_leaky")
        if provenance.get("inference_kind") != "real_local":
            blockers.append(name + ":real_local_inference_unmeasured")
        if truth["human_reviewed_resolved"] < minimum_reviewed:
            blockers.append(name + ":insufficient_reviewed_cases")
        if truth["missing_truth"] or truth["unreviewed_or_unresolved"] or truth["invalid_events"] or truth["inconsistent_verdict_events"]:
            blockers.append(name + ":ground_truth_incomplete")
        if truth["positive"] == 0 or truth["negative"] == 0:
            blockers.append(name + ":positive_or_negative_coverage_missing")
        if report["protocol"]["split"] != "validation" or provenance.get("final_check_used_for_tuning") is not False:
            blockers.append(name + ":heldout_use_not_cleanly_declared")
        if report["completion"]["unexpected_predictions"]:
            blockers.append(name + ":unexpected_sample_predictions")
        if report["completion"]["timing_unknown"]:
            blockers.append(name + ":completion_timing_unmeasured")
        for key in ("conditions_sha256", "rules_sha256", "taxonomy_sha256", "model_revision"):
            if not provenance.get(key):
                blockers.append(name + ":missing_" + key)
    for key in ("input_set_sha256", "truth_records_sha256", "protocol"):
        if baseline[key] != candidate[key]:
            blockers.append("incomparable_" + key)
    for key in ("conditions_sha256", "rules_sha256", "taxonomy_sha256"):
        if baseline["provenance"].get(key) != candidate["provenance"].get(key):
            blockers.append("incomparable_" + key)
    if (baseline.get("split_audit") or {}).get("records_sha256") != (candidate.get("split_audit") or {}).get("records_sha256"):
        blockers.append("incomparable_split_lock")
    quality = {}
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        whole, events, grounding, completion = report["whole_video_within_deadline"], _primary(report), report["grounding_and_evidence"], report["completion"]
        quality[name] = {
            "video_false_positives": whole["fp"], "video_missed_positives_including_failures": whole["fn"] + whole["unprocessed_positive"],
            "event_false_positives": events["fp"], "event_false_negatives_including_failures": events["fn"],
            "unprocessed_or_late": completion["unprocessed_at_deadline_including_unknown_timing"],
            "wrong_rule_mappings": grounding.get("wrong_or_missing_rule_mappings", 0),
            "unverified_rule_ids": grounding.get("unverified_or_missing_rule_ids", 0),
            "unsupported_evidence": grounding.get("evidence_unsupported", 0),
        }
        if grounding.get("catalog_verification_unmeasured", 0) or grounding.get("evidence_unreviewed", 0) or grounding.get("unresolved_truth_rule_mappings", 0):
            blockers.append(name + ":grounding_or_evidence_review_incomplete")
    regressions = [key for key, value in quality["candidate"].items() if value > quality["baseline"][key]]
    improvements = [key for key, value in quality["candidate"].items() if value < quality["baseline"][key]]
    if regressions:
        blockers.append("guarded_quality_regression")
    if not improvements:
        blockers.append("no_observed_error_reduction")
    return {
        "schema_version": "forte.selection.v1", "selected": "candidate" if not blockers else "baseline",
        "candidate_earned_selection": not blockers, "blockers": sorted(set(blockers)),
        "guarded_diagnostics": quality, "improved_axes": improvements, "regressed_axes": regressions,
        "minimum_reviewed_cases_policy": minimum_reviewed,
        "qualification": "Conservative team release gate; small-sample improvement is not statistical significance or a guaranteed hidden-test gain.",
        "final_check_policy": "Run final_check once after selection. Any tuning on its errors changes its status to development data.",
    }
