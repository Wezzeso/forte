"""Generated-only tests: no provided media, GPU, weights or inference."""
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.performance import fresh_head_repeatability as driver
from benchmarks.performance.repeatability import (RepeatabilityError, canonical_semantics,
    digest, file_digest, freeze_plan, scalar_max_difference, summarize, validate_plan, write_new)


@pytest.fixture
def prepared(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    synthetic = private / "synthetic.bin"
    synthetic.write_bytes(b"generated-only-not-media")
    plan_path = private / "plan.json"
    plan = freeze_plan(plan_path,
        source={"split": "train", "content_sha256": file_digest(synthetic),
                "source_bytes": synthetic.stat().st_size, "asset": {"owned_path": str(synthetic)}},
        bindings={"synthetic": {"path": str(synthetic), "sha256": file_digest(synthetic)}},
        settings={"threshold": 0., "batch_size": 32, "concurrency": 1,
                  "cache_reuse": False, "process_timeout_sec": 15,
                  "model_lifetime": "persistent_three_calls"},
        runtime={"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
                 "allow_tf32": False, "seed": 20261017}, private_root=private / "attempts")
    return plan, plan_path, private, synthetic


def successful(plan, index):
    return {"attempt_id": plan["attempts"][index]["attempt_id"], "status": "done",
        "fresh_stages_verified": True, "cache_hits": {"decode": 0, "encoder": 0, "head": 0},
        "stage_calls": {"decode": 1, "encoder_batches": 3, "head": 1},
        "timings": {"source_start_to_binary_output_sec": 1.2, "whole_attempt_sec": 3.}}


def test_canonical_comparison_preserves_physical_and_uncertainty_changes():
    first = {"job_id": "one", "clip_id": "temporary-one", "timings": {"total": 1.},
        "result_path": "temporary", "violations": [{"rule_refs": ["rule"],
            "source_pts": 90001, "time_base_num": 1, "time_base_den": 90000,
            "source_origin_pts": 90000, "frame_ids": ["frame"], "evidence_sha256": "a" * 64,
            "uncertainty": "limited", "visibility": False}]}
    second = deepcopy(first)
    second.update(job_id="two", clip_id="temporary-two", timings={"total": 9.}, result_path="other")
    assert canonical_semantics(first) == canonical_semantics(second)
    for key, changed in (("source_pts", 90002), ("evidence_sha256", "b" * 64),
                         ("rule_refs", ["other"]), ("uncertainty", "unclear"), ("visibility", True)):
        candidate = deepcopy(second)
        candidate["violations"][0][key] = changed
        assert digest(canonical_semantics(candidate)) != digest(canonical_semantics(first))


def test_freeze_never_reallocates_denominator_and_detects_plan_changes(prepared):
    plan, path, _, _ = prepared
    before = path.read_bytes()
    assert len({row["attempt_id"] for row in validate_plan(plan)["attempts"]}) == 3
    with pytest.raises(FileExistsError):
        write_new(path, plan)
    assert path.read_bytes() == before
    plan["settings"]["batch_size"] = 16
    with pytest.raises(RepeatabilityError, match="frozen_plan_changed"):
        validate_plan(plan)


def test_held_out_source_and_feature_cache_reuse_are_rejected_before_plan_creation(prepared):
    plan, _, private, _ = prepared
    arguments = {key: deepcopy(plan[key]) for key in ("source", "bindings", "settings")}
    arguments.update(runtime=plan["expected_runtime"], private_root=private / "other")
    arguments["source"]["split"] = "final_check"
    with pytest.raises(RepeatabilityError, match="final_or_unknown"):
        freeze_plan(private / "bad.json", **arguments)
    arguments["source"]["split"] = "train"
    arguments["settings"]["cache_reuse"] = True
    with pytest.raises(RepeatabilityError, match="settings_invalid"):
        freeze_plan(private / "bad.json", **arguments)
    assert not (private / "bad.json").exists()


def test_code_bundle_pins_runtime_imports_and_lock_files_without_private_data():
    bindings = driver.code_pins()
    for name in ("src/forte/model/encoder_staging.py", "src/forte/decision/temporal_native.py",
                 "serving/fast_encoder.py", "training/temporal/checkpoint.py", "pyproject.toml"):
        assert "code:" + name in bindings
    assert not any("/private/" in key or "/runs/" in key for key in bindings)


def test_failures_missing_and_cache_hits_remain_in_denominator(prepared):
    plan, _, _, _ = prepared
    first = successful(plan, 0)
    second = successful(plan, 1)
    second["cache_hits"]["encoder"] = 1
    report = summarize(plan, [first, second], [])
    assert (report["expected_attempts"], report["successful_fresh_attempts"],
            report["failed_attempts"], report["missing_attempts"]) == (3, 1, 1, 1)
    assert report["status"] == "incomplete"
    assert report["jury_legal_completion_eligible"] is False
    assert report["primary_ten_second_evidence_eligible"] is False
    assert report["pair_comparison_unavailable"]["semantic_equal"] == 3


def test_zero_stage_calls_cannot_be_a_fresh_success(prepared):
    plan, _, _, _ = prepared
    rows = [successful(plan, index) for index in range(3)]
    rows[2]["stage_calls"]["decode"] = 0
    report = summarize(plan, rows, [])
    assert report["successful_fresh_attempts"] == 2
    assert report["failed_attempts"] == 1


def test_complete_observation_reports_bitwise_and_numeric_difference_separately(prepared):
    plan, _, _, _ = prepared
    pair = {"source_identity_equal": True, "decoded_rgb_equal": True,
        "feature_bits_equal": False, "output_bits_equal": False, "semantic_equal": True,
        "feature_max_abs_difference": .0001, "output_max_abs_difference": .0002}
    pairs = [{**pair, "first_attempt_id": first["attempt_id"], "second_attempt_id": second["attempt_id"]}
             for first, second in combinations(plan["attempts"], 2)]
    report = summarize(plan, [successful(plan, index) for index in range(3)], pairs)
    assert report["complete_three_attempt_evidence"] is True
    assert report["pair_differences"]["feature_bits_equal"] == 3
    assert report["pair_differences"]["semantic_equal"] == 0
    assert report["maximum_feature_abs_difference"] == .0001
    assert "binary_prediction" not in json.dumps(report)
    assert plan["source"]["content_sha256"] not in json.dumps(report)


def test_repeating_one_comparison_cannot_claim_three_independent_pairs(prepared):
    plan, _, _, _ = prepared
    pair = {"first_attempt_id": plan["attempts"][0]["attempt_id"],
            "second_attempt_id": plan["attempts"][1]["attempt_id"]}
    with pytest.raises(RepeatabilityError, match="duplicate_or_invalid_pair"):
        summarize(plan, [successful(plan, index) for index in range(3)], [pair] * 3)


def test_numeric_delta_rejects_shape_or_nan_and_keeps_small_difference():
    assert scalar_max_difference([[1., 2.]], [[1., 2.001]]) == pytest.approx(.001)
    with pytest.raises(RepeatabilityError, match="shape_mismatch"):
        scalar_max_difference([1], [1, 2])
    with pytest.raises(RepeatabilityError, match="nonfinite"):
        scalar_max_difference([float("nan")], [1.])


def test_execute_is_guarded_before_runtime_or_gpu_probe(prepared, monkeypatch):
    _, plan_path, private, _ = prepared
    monkeypatch.setattr(driver, "verify_bindings", lambda plan: pytest.fail("runtime probed before release"))
    with pytest.raises(RepeatabilityError, match="gpu_release_required"):
        driver.execute(SimpleNamespace(plan=plan_path, gpu_window_released=False))
    assert not (private / "attempts").exists()


def test_internal_attempt_needs_parent_authorization_before_model_import(prepared):
    plan, _, private, _ = prepared
    with pytest.raises(FileNotFoundError):
        driver.run_attempt(plan, plan["attempts"][0], {})
    assert not (private / "attempts").exists()


def test_persistent_process_failures_preserve_three_frozen_attempts(prepared, monkeypatch):
    plan, plan_path, private, synthetic = prepared
    receipt = private / "reservation.json"
    write_new(receipt, {"registered": True, "conditions_sha256": plan["conditions_sha256"],
        "content_sha256": plan["source"]["content_sha256"],
        "attempt_ids": [row["attempt_id"] for row in plan["attempts"]]})
    # Bypass only the initial orchestration preflight. Every attempt must catch
    # the changed pin BEFORE any nvidia-smi, torch, media or inference operation.
    actual_verify = driver.verify_bindings
    calls = []
    def initial_preflight_only(current):
        calls.append(True)
        if len(calls) > 1:
            actual_verify(current)
    monkeypatch.setattr(driver, "verify_bindings", initial_preflight_only)
    synthetic.write_bytes(b"generated-only-pin-changed")
    report = driver.execute(SimpleNamespace(plan=plan_path, receipt=receipt,
        receipt_sha256=file_digest(receipt), gpu_window_released=True, report=private / "aggregate.json"))
    assert (report["observed_attempts"], report["failed_attempts"], report["missing_attempts"]) == (3, 3, 0)
    rows = json.loads((private / "attempts/private-results.json").read_text())["records"]
    assert len(calls) == 4
    assert len({row["process_id"] for row in rows}) == 1
    assert all(row["error_code"] == "repeat_pinned_artifact_changed" for row in rows)
    assert not list((private / "attempts").rglob("*.safetensors"))
    assert [row["attempt_id"] for row in rows] == [row["attempt_id"] for row in plan["attempts"]]
    monkeypatch.setattr(driver, "verify_bindings", lambda plan: None)
    with pytest.raises(RepeatabilityError, match="already_exists"):
        driver.execute(SimpleNamespace(plan=plan_path, receipt=receipt,
            receipt_sha256=file_digest(receipt), gpu_window_released=True, report=private / "aggregate.json"))


def test_three_calls_share_one_persistent_runtime_and_report_cold_load_separately(prepared, monkeypatch):
    import os
    plan, plan_path, private, _ = prepared
    receipt = private / "reservation.json"
    write_new(receipt, {"registered": True, "conditions_sha256": plan["conditions_sha256"],
        "content_sha256": plan["source"]["content_sha256"],
        "attempt_ids": [row["attempt_id"] for row in plan["attempts"]]})
    runtime_ids, model_ids = [], []
    closed = []
    def fake_fresh_pass(current, attempt, shared):
        runtime_ids.append(id(shared))
        if not shared:
            shared.update(encoder=SimpleNamespace(close=lambda: closed.append(True)),
                          head=object(), encoder_load_calls=1, head_load_calls=1)
        model_ids.append((id(shared["encoder"]), id(shared["head"])))
        row = successful(current, attempt["ordinal"] - 1)
        row["process_id"] = os.getpid()
        row["timings"].update(encoder_load_sec=2. if attempt["ordinal"] == 1 else 0.,
                              head_load_sec=.1 if attempt["ordinal"] == 1 else 0.)
        return row
    monkeypatch.setattr(driver, "verify_bindings", lambda plan: None)
    monkeypatch.setattr(driver, "run_attempt", fake_fresh_pass)
    monkeypatch.setattr(driver, "compare_records", lambda plan, rows: [])
    report = driver.execute(SimpleNamespace(plan=plan_path, receipt=receipt,
        receipt_sha256=file_digest(receipt), gpu_window_released=True, report=private / "aggregate.json"))
    assert len(runtime_ids) == 3 and len(set(runtime_ids)) == len(set(model_ids)) == 1
    assert report["model_load_calls"] == {"encoder": 1, "head": 1}
    assert report["observed_process_count"] == 1 and closed == [True]
    assert report["stage_wall_seconds"]["encoder_load_sec"] == [2., 0., 0.]
