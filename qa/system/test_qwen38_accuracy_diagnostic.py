"""Synthetic metadata only: denial and exact-budget tests, no source/model I/O."""
from copy import deepcopy

import pytest

from benchmarks.performance.qwen38_accuracy_diagnostic import (
    DEADLINES, ORDER, bind_plan, freeze_protocol, validate_diagnostic_plan,
    validate_protocol, validate_retained_pilot, validate_reserved_role_policy)
from evaluation.io import digest
from evaluation.qwen38_comparison import ARMS
from evaluation.qwen38_quality_family import arm_plan


def inputs():
    worker = {"window_seconds": 8., "stride_seconds": 4., "sample_fps": 2., "max_frames_per_window": 16,
        "max_concurrent_jobs": 1, "max_concurrent_clips": 1, "max_inflight_per_job": 1,
        "compact_frame_ids": True, "concise_observations": True, "staged_screening": False,
        "max_refinement_passes": 8, "job_timeout_seconds": 285., "trusted_scene_context": "bank_cash_workstation"}
    recipe = {"worker_settings": worker, "provider_max_attempts": 2,
        "generation": {"max_new_tokens": 4096, "seed": 20261017}, "source_archive_sha256": "a"*64,
        "rules_sha256": "b"*64, "prompt_source_sha256": "c"*64, "context_tokens": 32768}
    records = [{"sample_id": f"generated-{i}", "content_sha256": str(i+1)*64, "split": "validation"} for i in range(6)]
    family = {"schema_version": "forte.qwen38-quality-family.v1", "records": records,
        "attempts": [{**row, "arm": arm, "repeat_index": 0, "attempt_id": f"original-{arm}-{i}"}
            for arm in ARMS for i, row in enumerate(records)],
        "base_models": {arm: {"model_id": "generated-model", "model_revision": "1"*40} for arm in ARMS},
        "recipe": recipe, "recipe_sha256": digest(recipe), "provided_labels_sha256": "d"*64}
    spec = {"arm": "qwen38_untuned", "expected_model_identity": {**family["base_models"]["qwen38_untuned"], "adapter_sha256": None},
        "provider": {"launch_config": "/generated/config.json"},
        "runtime_bindings": {"source_archive_sha256": "a"*64, "runtime_image_sha256": "e"*64, "base_manifest_sha256": "f"*64},
        "sources": [{**row, "owned_path": f"/generated/{i}.mov", "source_bytes": 100,
            "duration_sec": duration} for i, (row, duration) in enumerate(zip(records, [72.7,20.866667,23.8,22.6,17.3,78.733021]))]}
    actual = arm_plan(family, spec, role_config_sha256="9"*64)
    actual["role_policy"] = {"expected_assignments_sha256": "a"*64, "expected_source_manifest_sha256": "b"*64,
        "expected_automated_split_sha256": "c"*64, "expected_scenario_groups_sha256": "d"*64,
        "allowed_splits": ["validation"], "max_snapshot_age_sec": 300, "final_check_used": False}
    return family, actual


def retained():
    return {"retained_pilot_plan_path": "/generated/pilot.json", "retained_pilot_plan_sha256": "7"*64,
        "retained_pilot_source_terminals_path": "/generated/terminals.json", "retained_pilot_source_terminals_sha256": "8"*64,
        "retained_pilot_role_config_sha256": "9"*64}


def completed(pilot):
    return {**pilot["source_attempts"][4], "status": "done", "public_contract_validated": True,
        "server_receipt_verified": True, "model_identity": pilot["expected_model_identity"],
        "comparison_plan_sha256": pilot["comparison_plan_sha256"], "recipe_sha256": pilot["recipe_sha256"],
        "verdict": "arbitrary-diagnostic-value-never-used-for-gating", "technical_completion": {
            "schema_version": "forte.qwen38-accuracy-technical-completion.v1", "complete_source_coverage": True,
            "strict_publication": True, "native_source_complete": True, "returned_evidence_matches_native_source": True,
            "fresh_stage_witness": True, "server_receipt_sha256": "6"*64, "reference_or_verdict_agreement_used": False}}


def terminals(pilot, complete_row):
    return [complete_row if index == 4 else {**row, "status": "not_attempted"}
        for index, row in enumerate(pilot["source_attempts"])]


def test_protocol_preserves_original_and_pilot_has_exact_six_denominator():
    family, actual = inputs()
    original = digest({"family": family, "actual": actual})
    protocol = freeze_protocol(family, actual)
    pilot = bind_plan(protocol, actual)
    assert digest({"family": family, "actual": actual}) == original
    assert pilot["execution_order"] == ORDER and [case["original_case_ordinal"] for case in pilot["cases"]] == [5]
    assert len(pilot["attempts"]) == pilot["maximum_private_model_requests"] == 24
    assert pilot["maximum_execution_seconds"] == 960 and pilot["minimum_initial_generation_requests"] == 4
    assert len(pilot["source_attempts"]) == pilot["source_attempts_required"] == 6
    assert pilot["worker"]["job_timeout_seconds"] == 285
    assert pilot["cases"][0]["diagnostic_deadline_seconds"] == 960
    assert pilot["recipe_sha256"] != family["recipe_sha256"] and pilot["speed_benchmark_eligible"] is False
    assert not ({row["attempt_id"] for row in pilot["source_attempts"]} & {row["attempt_id"] for row in actual["source_attempts"]})


def test_remaining_reserves_only_184_and_retains_unchanged_success_regardless_verdict():
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    pilot = bind_plan(protocol, actual)
    remaining = bind_plan(protocol, actual, stage="remaining", retained=retained())
    assert [case["original_case_ordinal"] for case in remaining["cases"]] == [1,2,3,4,6]
    assert remaining["source_attempts"] == pilot["source_attempts"]
    assert len(remaining["attempts"]) == 184 and remaining["maximum_execution_seconds"] == 7260
    assert len(pilot["attempts"]) + len(remaining["attempts"]) == 208
    assert not ({row["attempt_id"] for row in pilot["attempts"]} & {row["attempt_id"] for row in remaining["attempts"]})
    row = completed(pilot)
    before = digest(row)
    assert validate_retained_pilot(remaining, pilot, terminals(pilot, row)) is row
    assert digest(row) == before
    row["verdict"] = {"different": "still-does-not-control-advance"}
    assert validate_retained_pilot(remaining, pilot, terminals(pilot, row)) is row


@pytest.mark.parametrize("mutation", [
    lambda p: p["execution_order"].reverse(),
    lambda p: p["deadlines_by_original_ordinal_seconds"].__setitem__(4, 961),
    lambda p: p.__setitem__("maximum_source_attempts_per_case", 2),
    lambda p: p.__setitem__("advance_uses_provided_verdict_agreement", True),
    lambda p: p["population"][0].__setitem__("split", "final_check"),
    lambda p: p["population"][4].__setitem__("duration_sec", 18.),
    lambda p: p["population"][0].__setitem__("coarse_windows", 17),
    lambda p: p.__setitem__("speed_benchmark_eligible", True),
])
def test_protocol_rejects_selection_geometry_role_retry_and_claim_changes(mutation):
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    mutation(protocol)
    with pytest.raises(ValueError):
        validate_protocol(protocol)


@pytest.mark.parametrize("mutation", [
    lambda p: p["attempts"].append(deepcopy(p["attempts"][0])),
    lambda p: p["attempts"][0].__setitem__("content_sha256", "0"*64),
    lambda p: p["source_attempts"].pop(),
    lambda p: p["cases"].append(deepcopy(p["protocol_cases"][0])),
    lambda p: p.__setitem__("maximum_execution_seconds", 961),
    lambda p: p["protocol_cases"][4].__setitem__("diagnostic_deadline_seconds", 285),
    lambda p: p["runtime_bindings"].__setitem__("runtime_image_sha256", None),
    lambda p: p.__setitem__("retained_pilot_plan_path", "/generated/retry.json"),
])
def test_plan_rejects_slot_budget_denominator_deadline_and_unbound_runtime(mutation):
    family, actual = inputs()
    pilot = bind_plan(freeze_protocol(family, actual), actual)
    mutation(pilot)
    with pytest.raises(ValueError):
        validate_diagnostic_plan(pilot)


@pytest.mark.parametrize("mutation", [
    lambda row: row.__setitem__("status", "timeout"),
    lambda row: row.__setitem__("public_contract_validated", False),
    lambda row: row.__setitem__("server_receipt_verified", False),
    lambda row: row.__setitem__("model_identity", None),
    lambda row: row["technical_completion"].__setitem__("complete_source_coverage", False),
    lambda row: row["technical_completion"].__setitem__("reference_or_verdict_agreement_used", True),
    lambda row: row["technical_completion"].__setitem__("server_receipt_sha256", None),
])
def test_remaining_never_advances_failed_incomplete_or_identity_unknown_pilot(mutation):
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    pilot = bind_plan(protocol, actual)
    remaining = bind_plan(protocol, actual, stage="remaining", retained=retained())
    row = completed(pilot)
    mutation(row)
    with pytest.raises(ValueError):
        validate_retained_pilot(remaining, pilot, terminals(pilot, row))


def test_remaining_requires_raw_pilot_files_and_same_actual_runtime():
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    with pytest.raises(ValueError, match="immutable_pilot"):
        bind_plan(protocol, actual, stage="remaining")
    pilot = bind_plan(protocol, actual)
    remaining = bind_plan(protocol, actual, stage="remaining", retained=retained())
    remaining["runtime_bindings"]["runtime_image_sha256"] = "0"*64
    with pytest.raises(ValueError, match="plan_mismatch"):
        validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))


def test_missing_denominator_or_an_unreleased_private_source_cannot_advance():
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    pilot = bind_plan(protocol, actual)
    remaining = bind_plan(protocol, actual, stage="remaining", retained=retained())
    with pytest.raises(ValueError, match="population_invalid"):
        validate_retained_pilot(remaining, pilot, [completed(pilot)])
    rows = terminals(pilot, completed(pilot))
    rows[0]["status"] = "failed"
    with pytest.raises(ValueError, match="unreleased_source"):
        validate_retained_pilot(remaining, pilot, rows)


def test_fresh_role_snapshot_is_separate_but_original_pilot_role_pin_remains():
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    pilot = bind_plan(protocol, actual)
    fresh = deepcopy(actual)
    fresh["role_config_sha256"] = "0"*64
    remaining = bind_plan(protocol, fresh, stage="remaining", retained=retained())
    assert remaining["role_config_sha256"] != remaining["retained_pilot_role_config_sha256"]
    validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))
    remaining["retained_pilot_role_config_sha256"] = "0"*64
    with pytest.raises(ValueError, match="plan_mismatch"):
        validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))


@pytest.mark.parametrize("field,value", [
    ("provider", {"launch_config": "/generated/different.json"}),
    ("effective_client_request_timeout_seconds", 60),
    ("comparison_runtime_profile", {"different": "runtime"}),
    ("source_equivalence_sha256", "0"*64),
])
def test_remaining_rejects_changed_provider_timeout_profile_or_equivalence(field, value):
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    pilot = bind_plan(protocol, actual)
    remaining = bind_plan(protocol, actual, stage="remaining", retained=retained())
    remaining[field] = value
    with pytest.raises(ValueError, match="plan_mismatch"):
        validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))


def test_contract_imports_in_stdlib_only_host_without_site_packages():
    import subprocess
    import sys
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / "benchmarks/performance/qwen38_accuracy_diagnostic.py"
    completed_process = subprocess.run([sys.executable, "-I", "-S", "-c",
        "import runpy,sys; module=runpy.run_path(sys.argv[1],run_name='metadata_only'); assert module['ORDER']==[5,1,2,3,4,6]",
        str(source)], capture_output=True, text=True, timeout=15)
    assert completed_process.returncode == 0, completed_process.stderr


def test_fixed_geometry_formula_matches_frozen_sampler():
    from forte.video.sampling import coverage_windows
    from benchmarks.performance.qwen38_accuracy_diagnostic import DURATIONS, WINDOWS
    for duration, expected in zip(DURATIONS, WINDOWS):
        windows = coverage_windows(duration, window_sec=8., stride_sec=4., sample_fps=2., max_frames_per_window=16)
        assert len(windows) == expected
        assert all(len(window.timestamps_sec) == 16 for window in windows)


@pytest.mark.parametrize("key,value", [
    ("expected_assignments_sha256", "e"*64), ("expected_source_manifest_sha256", "e"*64),
    ("expected_automated_split_sha256", "e"*64), ("expected_scenario_groups_sha256", "e"*64),
    ("max_snapshot_age_sec", 299),
])
def test_fresh_snapshot_cannot_change_original_immutable_role_policy(key, value):
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    pilot = bind_plan(protocol, actual)
    remaining = bind_plan(protocol, actual, stage="remaining", retained=retained())
    remaining["role_policy"][key] = value
    with pytest.raises(ValueError, match="plan_mismatch"):
        validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))


@pytest.mark.parametrize("key,value", [("allowed_splits", ["train", "validation"]),
    ("max_snapshot_age_sec", 301), ("final_check_used", True)])
def test_role_policy_rejects_wider_scope_or_stale_snapshot(key, value):
    family, actual = inputs()
    actual["role_policy"][key] = value
    with pytest.raises(ValueError, match="role_policy_invalid"):
        bind_plan(freeze_protocol(family, actual), actual)


def test_immutable_policy_must_match_genuine_central_role_preflight_bindings():
    family, actual = inputs()
    pilot = bind_plan(freeze_protocol(family, actual), actual)
    policy = pilot["role_policy"]
    gate = {"schema_version": "forte.qwen38-role-preflight.v1", "ok": True,
        "assignments_sha256": policy["expected_assignments_sha256"], "requested_contents": 1,
        "allowed_split_counts": {"validation": 1}, "input_checksums": {
            name: policy["expected_" + name + "_sha256"] for name in ("source_manifest", "automated_split", "scenario_groups")},
        "human_verified": False, "scenario_independence_verified": False, "compute_authorized": False,
        "media_opened": False, "model_calls_made": False}
    reservation = {"exposure_gate": gate}
    assert validate_reserved_role_policy(pilot, reservation) is True
    for mutate in (lambda row: row.__setitem__("assignments_sha256", "0"*64),
                   lambda row: row["input_checksums"].__setitem__("scenario_groups", "0"*64),
                   lambda row: row.__setitem__("allowed_split_counts", {"train": 1}),
                   lambda row: row.__setitem__("requested_contents", 6)):
        changed = deepcopy(reservation)
        mutate(changed["exposure_gate"])
        with pytest.raises(ValueError, match="reserved_role_policy_mismatch"):
            validate_reserved_role_policy(pilot, changed)
