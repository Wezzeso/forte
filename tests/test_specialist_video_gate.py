"""Synthetic whole-video promotion proofs; no supplied or recorded media is read."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from forte.fast.policy import analyzer_policy_sha256
from training.guards import TrainingGateError, file_sha256
from training.specialists.common import write_json
from training.specialists.video_gate import approve_video_types


TARGET = "cash_without_id"
OTHER = "door_left_open"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def reports(tmp_path: Path) -> dict:
    """One fixed inventory: 19 clean, 20 target positives, one other positive."""
    rows = []
    for index in range(40):
        type_id = None if index < 19 else TARGET if index < 39 else OTHER
        events = [] if type_id is None else [{"type_id": type_id, "t": 2.0}]
        rows.append({"name": f"synthetic-{index:02}", "processed": True,
                     "prediction": bool(events), "events": copy.deepcopy(events),
                     "label": {"violation": bool(events), "events": events}})
    inputs = [{"name": row["name"], "sha256": _sha(row["name"])} for row in rows]
    run = {"checkpoint_sha256": _sha("synthetic-checkpoint-not-a-model"),
           "dataset_lock_sha256": _sha("synthetic-dataset-lock"),
           "config_sha256": _sha("synthetic-config")}
    report = {"schema": "forte.fast-evaluation.v2", "config": {}, "labels_sha256": _sha("synthetic-labels"),
              "scoring": {"time_tolerance_s": 2.0},
              "integrity": {"complete": True, "missing": [], "unexpected": [],
                            "duplicates": [], "invalid_rows": []},
              "provenance": {"mode": "live_analyzer_inference", "inference_executed": True,
                             "data_hash_scope": "full_file_sha256",
                             "input_inventory_sha256": _sha("synthetic-inventory"),
                             "inputs": inputs}, "results": rows}
    baseline = copy.deepcopy(report)
    # The candidate recovers the final target-positive video the baseline misses.
    baseline["results"][38].update(prediction=False, events=[])
    candidate = copy.deepcopy(report)
    candidate["provenance"].update(
        evaluation_mode="candidate_only", provisional_types=[TARGET],
        analyzer_policy_sha256=analyzer_policy_sha256(candidate["config"]),
        specialist_identity={**run, "gate_sha256": _sha("synthetic-class-gate")})
    return {"candidate": candidate, "baseline": baseline, "run": run, "root": tmp_path}


def _approve(reports: dict, requested: list[str] | None = None) -> dict:
    candidate_path = reports["root"] / "candidate.json"
    baseline_path = reports["root"] / "baseline.json"
    write_json(candidate_path, reports["candidate"])
    write_json(baseline_path, reports["baseline"])
    return approve_video_types(candidate_path, baseline_path,
                               [TARGET] if requested is None else requested, reports["run"])


def test_scoring_errors_are_allowed_in_baseline_misses(reports):
    reports["baseline"]["results"][38]["error_reasons"] = ["missed_violation", "type_mismatch"]
    assert _approve(reports)["approved_types"] == [TARGET]


@pytest.mark.parametrize("timestamp", [-1, float("nan"), float("inf")])
def test_invalid_event_time_is_rejected(reports, timestamp):
    reports["candidate"]["results"][19]["events"][0]["t"] = timestamp
    # NaN/Infinity are rejected even before JSON persistence; use the in-memory
    # report guard directly to avoid producing invalid JSON fixture artifacts.
    from training.specialists.video_gate import _types
    with pytest.raises(TrainingGateError, match="time_invalid"):
        _types(reports["candidate"]["results"][19]["events"])


def test_untimed_labels_cannot_authorize_timecode_promotion(reports):
    for report in (reports["candidate"], reports["baseline"]):
        report["results"][19]["label"]["events"][0].pop("t")
    with pytest.raises(TrainingGateError, match="explicit_timestamps_required"):
        _approve(reports)


def test_duplicate_events_cannot_inflate_type_recall(reports):
    reports["candidate"]["results"][19]["events"].append({"type_id": TARGET, "t": 2.0})
    with pytest.raises(TrainingGateError, match="event_precision_or_recall_failed"):
        _approve(reports)


def test_wrong_timestamp_cannot_count_as_recovered_type(reports):
    reports["candidate"]["results"][38]["events"][0]["t"] = 10000.0
    with pytest.raises(TrainingGateError, match="event_precision_or_recall_failed"):
        _approve(reports)


def test_reported_tolerance_cannot_be_relaxed_after_evaluation(reports):
    reports["candidate"]["scoring"]["time_tolerance_s"] = 999
    with pytest.raises(TrainingGateError, match="time_protocol_required"):
        _approve(reports)


def test_policy_hash_is_recomputed_from_effective_candidate_configuration(reports):
    reports["candidate"]["config"]["id_confidence_threshold"] = .0001
    with pytest.raises(TrainingGateError, match="analyzer_policy_mismatch"):
        _approve(reports)


def test_repeated_event_miss_cannot_regress_baseline_event_matches(reports):
    for report in (reports["candidate"], reports["baseline"]):
        report["results"][19]["label"]["events"].append({"type_id": TARGET, "t": 6.0})
    reports["baseline"]["results"][19]["events"].append({"type_id": TARGET, "t": 6.0})
    with pytest.raises(TrainingGateError, match="baseline_event_regression"):
        _approve(reports)


def test_strict_type_gain_returns_hash_bound_forty_video_proof(reports):
    result = _approve(reports)

    assert result["approved_types"] == [TARGET]
    assert result["video_clean_count"] == 19
    assert result["video_type_metrics"] == {
        TARGET: {"tp": 20, "fp": 0, "fn": 0, "recall": 1.0,
                 "baseline_tp": 19, "positive_videos": 20}}
    assert result["video_validation_sha256"] == file_sha256(reports["root"] / "candidate.json")
    assert result["baseline_video_validation_sha256"] == file_sha256(reports["root"] / "baseline.json")
    assert result["video_input_inventory_sha256"] == reports["candidate"]["provenance"]["input_inventory_sha256"]
    assert result["video_labels_sha256"] == reports["candidate"]["labels_sha256"]


def test_recall_boundary_passes_when_gain_does_not_regress_a_correct_verdict(reports):
    reports["candidate"]["results"][38].update(prediction=False, events=[])
    reports["baseline"]["results"][37].update(prediction=False, events=[])

    metrics = _approve(reports)["video_type_metrics"][TARGET]

    assert metrics == {"tp": 19, "fp": 0, "fn": 1, "recall": .95,
                       "baseline_tp": 18, "positive_videos": 20}


@pytest.mark.parametrize("report_name", ["candidate", "baseline"])
def test_binary_only_positive_labels_cannot_unlock_a_type(reports, report_name):
    reports[report_name]["results"][19]["label"] = {"violation": True}

    with pytest.raises(TrainingGateError, match="specialist_video_type_invalid"):
        _approve(reports)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "dataset_lock_sha256", "config_sha256"])
def test_candidate_must_bind_the_exact_specialist_artifacts(reports, field):
    reports["candidate"]["provenance"]["specialist_identity"][field] = _sha("different-artifact")

    with pytest.raises(TrainingGateError, match="specialist_video_artifact_identity_mismatch"):
        _approve(reports)


def test_candidate_requires_a_well_formed_class_gate_hash(reports):
    reports["candidate"]["provenance"]["specialist_identity"]["gate_sha256"] = "unbound"

    with pytest.raises(TrainingGateError, match="specialist_video_artifact_identity_mismatch"):
        _approve(reports)


@pytest.mark.parametrize("field", ["input_inventory_sha256", "labels_sha256"])
def test_baseline_comparison_rejects_different_inventory_or_label_hash(reports, field):
    target = reports["candidate"]["provenance"] if field == "input_inventory_sha256" else reports["candidate"]
    target[field] = _sha("different-inputs")

    with pytest.raises(TrainingGateError, match="specialist_video_baseline_inputs_mismatch"):
        _approve(reports)


def test_equal_inventory_summary_cannot_hide_a_changed_video_hash(reports):
    reports["candidate"]["provenance"]["inputs"][0]["sha256"] = _sha("different-video")

    with pytest.raises(TrainingGateError, match="specialist_video_baseline_media_mismatch"):
        _approve(reports)


@pytest.mark.parametrize("change,code", [
    (lambda r: r.update(schema="forte.fast-evaluation.v1"), "report_schema"),
    (lambda r: r["integrity"].update(complete=False), "report_incomplete"),
    (lambda r: r["provenance"].update(mode="imported_predictions"), "live_inference_required"),
    (lambda r: r["provenance"].update(inference_executed=False), "live_inference_required"),
    (lambda r: r["provenance"].update(data_hash_scope="first_megabyte_sha256"), "full_media_hashes_required"),
    (lambda r: r["results"].pop(), "full_forty_required"),
    (lambda r: r["provenance"]["inputs"].pop(), "inventory_coverage"),
])
def test_only_complete_live_forty_video_reports_are_eligible(reports, change, code):
    change(reports["candidate"])

    with pytest.raises(TrainingGateError, match=f"specialist_video_{code}"):
        _approve(reports)


@pytest.mark.parametrize("change", [
    lambda p: p.update(evaluation_mode="baseline"),
    lambda p: p.update(provisional_types=[OTHER]),
    lambda p: p.pop("evaluation_mode"),
])
def test_provisional_inference_must_be_candidate_only_and_match_requested_types(reports, change):
    change(reports["candidate"]["provenance"])

    with pytest.raises(TrainingGateError, match="specialist_video_candidate_scope_mismatch"):
        _approve(reports)


@pytest.mark.parametrize("false_type", [TARGET, OTHER])
def test_any_clean_false_alarm_blocks_promotion(reports, false_type):
    reports["candidate"]["results"][0].update(prediction=True, events=[{"type_id": false_type}])

    with pytest.raises(TrainingGateError, match="specialist_video_clean_false_alarm"):
        _approve(reports)


@pytest.mark.parametrize("failure", [
    {"processed": False}, {"error": "synthetic_failure"},
    {"error_reasons": ["synthetic_failure"]}, {"prediction": None},
])
@pytest.mark.parametrize("report_name", ["candidate", "baseline"])
def test_processing_failure_in_either_report_blocks_promotion(reports, report_name, failure):
    reports[report_name]["results"][0].update(failure)

    with pytest.raises(TrainingGateError, match="specialist_video_inference_failure"):
        _approve(reports)


def test_gain_cannot_hide_a_new_miss_on_another_violation(reports):
    reports["candidate"]["results"][39].update(prediction=False, events=[])

    with pytest.raises(TrainingGateError, match="specialist_video_baseline_verdict_regression"):
        _approve(reports)


def test_requested_type_needs_a_strict_gain_over_baseline(reports):
    reports["baseline"]["results"][38].update(prediction=True, events=[{"type_id": TARGET, "t": 2.0}])

    with pytest.raises(TrainingGateError, match="specialist_video_type_recall_or_gain_failed"):
        _approve(reports)


def test_requested_type_false_alarm_on_a_different_positive_video_is_rejected(reports):
    reports["candidate"]["results"][39]["events"].append({"type_id": TARGET})

    with pytest.raises(TrainingGateError, match="specialist_video_type_recall_or_gain_failed"):
        _approve(reports)
