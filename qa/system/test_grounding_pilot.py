"""Generated contract/orchestration fixtures; no provided media, model or GPU."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.performance import grounding_measure as measure
from benchmarks.performance import grounding_plan as planner
from forte.shared.schemas import VideoAsset
from forte.video.batches import SourceFrameIdentity, source_frame_id
from serving.grounding_pilot import GroundingAttempt, MANIFEST_SHA256, REVISION
from training.grounding.contracts import (GroundingRawResult, PhysicalQuery, ProcessorClockProof,
    bind_intervals, parse_intervals, validate_source_ledger)


def asset_and_ledger(content="a" * 64):
    asset = VideoAsset(job_id="generated-" + content[:4], owned_path="/private/sources/" + content + ".mov",
        content_hash=content, duration_sec=3., width=224, height=224, fps=1., time_base="1/10", frame_count=3)
    frames = tuple(SourceFrameIdentity(source_frame_id(content, index * 10, 1, 10), content,
        asset.job_id, 0, index * 10, "1/10", 1, 10, 0, float(index),
        (index + 1) * 10 if index < 2 else None, float(index + 1) if index < 2 else None,
        None if index < 2 else 3., 224, 224, 0, index == 0) for index in range(3))
    return asset, frames


def good_result(asset, frames, *, query=None, tokens=(11, 12), text="[[0.2,1.8]]"):
    query = query or PhysicalQuery(query_id="CASH_COUNT_VISIBLE", text="generated physical counting query")
    clock = ProcessorClockProof(kind="verified_cfr_native_indices_over_source_fps",
        sampled_frame_indices=(0, 1, 2), fps_num=1, fps_den=1, processor_clock_sha256="c" * 64,
        processor_clock_verified=True, processor_timestamps_sec=(.5, 2.))
    source = validate_source_ledger(asset, frames)
    intervals = bind_intervals(parse_intervals(text, duration_sec=asset.duration_sec), asset=asset,
        source_index=frames, sampled_frame_ids=tuple(frame.frame_id for frame in frames), clock=clock)
    return GroundingRawResult(**source.model_dump(), query=query, model_id="MCG-NJU/TimeLens2-2B",
        model_revision=REVISION, processor_revision=REVISION, model_manifest_sha256=MANIFEST_SHA256,
        input_recipe_sha256="f" * 64, clock=clock, sampled_frame_ids=tuple(frame.frame_id for frame in frames),
        intervals=intervals, raw_response=text, generated_token_ids=tokens, metrics={})


def attempt(asset, frames, *, query=None, failed=False, generated=True, tokens=(11, 12), text="[[0.2,1.8]]"):
    raw = None if failed else good_result(asset, frames, query=query, tokens=tokens, text=text)
    return GroundingAttempt("failed" if failed else "done", "grounding_output_invalid_json" if failed else None,
        raw, frames, {"decode_sec": .1, "processor_sec": .2, "h2d_sec": .01, "generate_sec": .3,
            "prompt_tokens": 200., "generated_tokens": float(len(tokens))},
        {"decode_calls": 1, "processor_calls": int(generated), "generation_calls": int(generated),
            "source_result_cache_hits": 0, "model_load_calls": 1}, {"peak_allocated_bytes": 123},
        text if generated else "", tokens if generated else (), "stop" if generated else "not_started")


@pytest.fixture
def frozen(tmp_path):
    sources = []
    for index, content in enumerate(("a" * 64, "b" * 64)):
        asset, _ = asset_and_ledger(content)
        sources.append({"sample_id": "generated-" + str(index), "split": "train", "content_sha256": content,
            "source_bytes": 20, "asset": asset.model_dump(mode="json")})
    queries = [{"query_id": name, "text": "generated physical query " + str(index)} for index, name in enumerate(
        ("CASH_COUNT_VISIBLE", "PHYSICAL_QUERY_TWO", "PHYSICAL_QUERY_THREE", "PHYSICAL_QUERY_FOUR"))]
    profile = {"schema_version": "forte.grounding-pilot-profile.v1", "runner_config": {
        "snapshot": "/private/weights", "manifest": "/private/model.json", "manifest_sha256": MANIFEST_SHA256,
        "device": "cuda:0", "target_size": 224, "sample_fps": 2., "max_sampled_frames": 66,
        "max_new_tokens": 128, "max_native_frames": 10000, "max_source_duration_sec": 32.,
        "process_memory_fraction": .10, "min_free_bytes": 9 * 1024 ** 3, "clock_policy": "strict_cfr"}}
    plan = {"schema_version": planner.SCHEMA, "sources": sources, "queries": queries,
        "query_set_sha256": planner.canonical_sha256(queries), "profile": profile, "bindings": {},
        "clock_audit_summaries": [{"content_sha256": row["content_sha256"],
            "ledger_sha256": validate_source_ledger(*asset_and_ledger(row["content_sha256"])).source_ledger_sha256}
            for row in sources], "attempts": planner.freeze_attempts(sources, queries),
        "private_root": str(tmp_path / "private" / "runs"),
        "model_lifetime": "one_resident_model_for_all_ten_attempts", "fresh_source_decode_per_attempt": True,
        "source_result_cache_allowed": False, "legal_timing_eligible": False,
        "budgets": {"overall_sec": 360., "per_call_sec": 30., "minimum_free_gpu_bytes": 9 * 1024 ** 3,
            "per_process_memory_fraction": .10, "expected_attempts": 10}}
    plan["conditions_sha256"] = planner.canonical_sha256(plan)
    return planner.validate_plan(plan)


def repeat_records(plan, result):
    return [measure.attempt_record(row, result, .8) for row in plan["attempts"]
            if row["source_index"] == 0 and row["query_id"] == planner.REPEAT_QUERY]


def test_duration_only_selection_preserves_gapped_source_and_ignores_nontraining():
    rows, assignments = [], []
    for index, (duration, split) in enumerate(((1., "validation"), (7., "train"), (9., "train"),
        (11., "train"), (20., "train"), (30., "train"), (40., "train"))):
        content = str(index) * 64
        rows.append({"content_sha256": content, "sample_id": "s" + str(index), "split": split,
            "asset": {"duration_sec": duration}, "is_cfr": False, "label_must_not_be_consulted": object()})
        assignments.append({"content_hash": content, "split": split})
    selected, count = planner.select_sources(rows, {"records": assignments})
    assert count == 5 and [row["asset"]["duration_sec"] for row in selected] == [7., 11.]
    assert all(row["is_cfr"] is False for row in selected)
    assignments[1]["split"] = "final_check"
    with pytest.raises(planner.GroundingPlanError, match="split_mismatch"):
        planner.select_sources(rows, {"records": assignments})


def test_exact_ten_attempts_and_three_repeats_are_frozen(frozen):
    assert len({row["attempt_id"] for row in frozen["attempts"]}) == 10
    assert len([row for row in frozen["attempts"] if row["source_index"] == 0
        and row["query_id"] == planner.REPEAT_QUERY]) == 3
    frozen["attempts"][8]["source_index"] = 1
    with pytest.raises(planner.GroundingPlanError, match="frozen_plan_changed"):
        planner.validate_plan(frozen)


def test_profile_memory_or_clock_adaptation_fails_before_any_model(frozen):
    profile = deepcopy(frozen["profile"])
    profile["runner_config"]["process_memory_fraction"] = .5
    with pytest.raises(planner.GroundingPlanError, match="budget_invalid"):
        planner.validate_profile(profile)
    profile = deepcopy(frozen["profile"])
    profile["runner_config"]["clock_policy"] = "adaptive_fallback"
    with pytest.raises(planner.GroundingPlanError, match="budget_invalid"):
        planner.validate_profile(profile)


def test_no_generation_does_not_make_empty_tokens_repeatable(frozen):
    asset, frames = asset_and_ledger()
    rows = repeat_records(frozen, attempt(asset, frames, failed=True, generated=False))
    report, _ = measure.summarize(frozen, rows, cold_load=1., overall_elapsed=4., load_calls=1)
    assert (report["expected_attempts"], report["failed_attempts"], report["missing_attempts"]) == (10, 3, 7)
    assert report["repeat_comparable_pairs"]["raw_tokens_equal"] == 0
    assert report["repeat_unavailable_pairs"]["semantic_intervals_equal"] == 3


def test_failed_parse_raw_tokens_remain_comparable_without_success_claim(frozen):
    asset, frames = asset_and_ledger()
    result = attempt(asset, frames, failed=True, text="generated invalid private response")
    report, _ = measure.summarize(frozen, repeat_records(frozen, result), cold_load=1., overall_elapsed=4., load_calls=1)
    assert report["failed_attempts"] == 3 and report["done_attempts"] == 0
    assert report["repeat_comparable_pairs"]["raw_tokens_equal"] == 3
    assert report["repeat_comparable_pairs"]["semantic_intervals_equal"] == 0
    assert "generated invalid private response" not in json.dumps(report)


def test_bitwise_tokens_and_semantic_source_snaps_are_compared_separately(frozen):
    asset, frames = asset_and_ledger()
    rows = repeat_records(frozen, attempt(asset, frames))
    planned = [row for row in frozen["attempts"] if row["source_index"] == 0 and row["query_id"] == planner.REPEAT_QUERY]
    rows[2] = measure.attempt_record(planned[2], attempt(asset, frames, tokens=(11, 99)), .8)
    report, _ = measure.summarize(frozen, rows, cold_load=1., overall_elapsed=4., load_calls=1)
    assert report["repeat_differing_pairs"]["raw_tokens_equal"] == 2
    assert report["repeat_differing_pairs"]["semantic_intervals_equal"] == 0
    rows[2] = measure.attempt_record(planned[2], attempt(asset, frames, text="[[0.3,1.8]]"), .8)
    report, _ = measure.summarize(frozen, rows, cold_load=1., overall_elapsed=4., load_calls=1)
    assert report["repeat_differing_pairs"]["semantic_intervals_equal"] == 2


def test_empty_valid_interval_array_is_not_a_confirmed_absence(frozen):
    asset, frames = asset_and_ledger()
    rows = repeat_records(frozen, attempt(asset, frames, text="[]"))
    report, _ = measure.summarize(frozen, rows, cold_load=1., overall_elapsed=4., load_calls=1)
    assert report["no_grounding_proposal_attempts"] == 3
    assert report["empty_intervals_mean_confirmed_no_violation"] is False
    assert report["primary_jury_ten_second_evidence_eligible"] is False
    assert report["hundred_percent_accuracy_evidence_eligible"] is False


def test_unexpected_started_exception_keeps_unknown_stage_counts(frozen):
    row = measure.failed_record(frozen["attempts"][0], "grounding_runner_exception", started=True, elapsed=1.)
    report, _ = measure.summarize(frozen, [row], cold_load=1., overall_elapsed=2., load_calls=1)
    assert report["stage_call_count_unknown_attempts"]["generation_calls"] == 1
    assert report["actual_stage_call_counts"]["generation_calls"] == 0


def test_release_is_checked_before_runtime_or_source_access(frozen, tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    planner.write_new(plan, frozen)
    monkeypatch.setattr(measure, "validate_bindings", lambda value: pytest.fail("unreleased runtime accessed"))
    with pytest.raises(planner.GroundingPlanError, match="compute_release_required"):
        measure.execute(SimpleNamespace(plan=plan, compute_window_released=False))


def test_generated_orchestration_one_load_ten_fresh_calls_and_private_outputs(frozen, tmp_path, monkeypatch, capsys):
    import serving.grounding_pilot as serving
    plan = tmp_path / "private" / "plan.json"
    receipt = plan.with_name("reservation.json")
    planner.write_new(plan, frozen)
    planner.write_new(receipt, {"registered": True, "conditions_sha256": frozen["conditions_sha256"],
        "attempt_ids": [row["attempt_id"] for row in frozen["attempts"]],
        "source_content_sha256": [row["content_sha256"] for row in frozen["sources"]]})
    loads, calls, closes = [], [], []
    class FakeResident:
        def __init__(self, config):
            self.config = config
        def load(self):
            loads.append(True)
            return {"model_load_calls": 1, "model_revision": REVISION, "processor_revision": REVISION, "packages": {}}
        def run(self, asset, query, **kwargs):
            calls.append((asset.content_hash, query.query_id, kwargs["expected_source_ledger_sha256"]))
            _, frames = asset_and_ledger(asset.content_hash)
            assert kwargs["deadline"] > measure.time.monotonic()
            return attempt(asset, frames, query=query)
        def close(self):
            closes.append(True)
    monkeypatch.setattr(serving, "GroundingPilot", FakeResident)
    monkeypatch.setattr(measure, "validate_bindings", lambda value: None)
    def synthetic_watch(command, state_path, allowed_attempts):
        # Only replace process creation: exercise the actual internal CLI,
        # parent authorization, resident runner, immutable artifacts and parent
        # reconciliation using the explicit synthetic model above.
        assert command[1:4] == ["-m", "benchmarks.performance.grounding_measure", "_session"]
        assert allowed_attempts == {row["attempt_id"] for row in frozen["attempts"]}
        with monkeypatch.context() as child:
            child.setattr(measure.os, "getppid", measure.os.getpid)
            assert measure.main(command[3:]) == 0
        return {"termination_code": None, "child_exit_code": 0,
            "process_termination_confirmed": True, "active_attempt_id": None,
            "last_state": measure.read_json(state_path), "observed_wall_sec": .1,
            "overall_limit_sec": 360., "per_call_limit_sec": 30., "poll_interval_sec": .05}
    monkeypatch.setattr(measure, "watch_process", synthetic_watch)
    report_path = tmp_path / "aggregate.json"
    report = measure.execute(SimpleNamespace(plan=plan, receipt=receipt, receipt_sha256=planner.file_sha256(receipt),
        compute_window_released=True, report=report_path))
    assert len(loads) == len(closes) == 1 and len(calls) == 10
    assert report["done_attempts"] == 10 and report["actual_stage_call_counts"]["decode_calls"] == 10
    assert report["actual_stage_call_counts"]["generation_calls"] == 10
    assert report["repeat_comparable_pairs"]["semantic_intervals_equal"] == 3
    assert report["model_load_calls"] == 1 and report["legal_timing_eligible"] is False
    public = report_path.read_text()
    assert all(text not in public for text in ("generated physical", "[[0.2,1.8]]", "a" * 64,
        str(tmp_path), frozen["attempts"][0]["attempt_id"]))
    assert len(list(Path(frozen["private_root"]).glob("ground_*.json"))) == 10
    assert json.loads(capsys.readouterr().out) == {"status": "resident_session_complete", "recorded_attempts": 10}
    with pytest.raises(FileExistsError):
        with monkeypatch.context() as child:
            child.setattr(measure.os, "getppid", measure.os.getpid)
            measure.resident_session(SimpleNamespace(plan=plan,
                authorization=Path(frozen["private_root"]) / "execution-authorization.private.json"))
    assert len(loads) == 1  # One release cannot be replayed.


@pytest.mark.parametrize("invalid", ["wrong_parent", "changed_plan", "unreleased", "expired", "oversized_deadline", "changed_receipt", "wrong_authorization_path"])
def test_internal_resident_rejects_invalid_release_before_model_or_binding_access(frozen, tmp_path, monkeypatch, invalid):
    plan_path = tmp_path / "private" / "plan.json"
    receipt_path = plan_path.with_name("reservation.json")
    root = Path(frozen["private_root"])
    planner.write_new(plan_path, frozen)
    planner.write_new(receipt_path, {"registered": True, "conditions_sha256": frozen["conditions_sha256"],
        "attempt_ids": [row["attempt_id"] for row in frozen["attempts"]],
        "source_content_sha256": [row["content_sha256"] for row in frozen["sources"]]})
    now = measure.time.monotonic()
    authorization = {"schema_version": "forte.grounding-session-authorization.v1",
        "conditions_sha256": frozen["conditions_sha256"], "compute_window_released": True,
        "parent_process_id": measure.os.getppid(), "plan_sha256": planner.file_sha256(plan_path),
        "receipt_path": str(receipt_path.resolve()), "receipt_sha256": planner.file_sha256(receipt_path),
        "attempt_ids": [row["attempt_id"] for row in frozen["attempts"]],
        "overall_started_monotonic": now, "overall_deadline_monotonic": now + 360.}
    expected = "grounding_parent_session_authorization_required"
    if invalid == "wrong_parent": authorization["parent_process_id"] = -1
    if invalid == "changed_plan": authorization["plan_sha256"] = "0" * 64
    if invalid == "unreleased": authorization["compute_window_released"] = False
    if invalid == "expired":
        authorization["overall_started_monotonic"] = now - 20.
        authorization["overall_deadline_monotonic"] = now - 1.
        expected = "grounding_session_authorization_expired_or_invalid"
    if invalid == "oversized_deadline":
        authorization["overall_deadline_monotonic"] = now + 361.
        expected = "grounding_session_authorization_expired_or_invalid"
    if invalid == "changed_receipt":
        authorization["receipt_sha256"] = "0" * 64
        expected = "grounding_session_exposure_receipt_mismatch"
    name = "wrong.private.json" if invalid == "wrong_authorization_path" else "execution-authorization.private.json"
    if invalid == "wrong_authorization_path": expected = "grounding_session_authorization_path_invalid"
    authorization_path = root / name
    planner.write_new(authorization_path, authorization)
    monkeypatch.setattr(measure, "validate_bindings", lambda value: pytest.fail("bindings touched before release check"))
    monkeypatch.setattr(measure, "run_resident_session", lambda *args, **kwargs: pytest.fail("model runner touched before release check"))
    with pytest.raises(planner.GroundingPlanError, match=expected):
        measure.resident_session(SimpleNamespace(plan=plan_path, authorization=authorization_path))
    assert not (root / "resident-session.started.private.json").exists()


@pytest.mark.parametrize("kind", ["overall", "call", "child_exit"])
def test_real_cpu_child_watchdog_terminates_timeouts_and_reports_exit(tmp_path, kind):
    import sys
    state = tmp_path / "state.json"
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    if kind != "child_exit":
        command = [sys.executable, "-c", "import json,sys,time; from pathlib import Path; "
            "Path(sys.argv[1]).write_text(json.dumps({'phase':'running','attempt_id':'synthetic-only',"
            "'started_monotonic':time.monotonic()})); time.sleep(30)", str(state)]
    report = measure.watch_process(command, state, {"synthetic-only"},
        overall_sec=.15 if kind == "overall" else 5., per_call_sec=.15 if kind == "call" else 10., poll_sec=.01)
    assert report["process_termination_confirmed"] is True
    assert report["observed_wall_sec"] < 8.
    if kind == "child_exit":
        assert report["child_exit_code"] == 7 and report["termination_code"] is None
    else:
        assert report["termination_code"] == "grounding_supervisor_" + kind + "_timeout"
        assert report["child_exit_code"] != 0


def test_actual_internal_subprocess_cli_recognizes_session_and_fails_closed(tmp_path):
    import subprocess
    import sys
    secret_sentinel = tmp_path / "private" / "SYNTHETIC_PATH_MUST_NOT_BE_LOGGED.json"
    completed = subprocess.run([sys.executable, "-m", "benchmarks.performance.grounding_measure", "_session",
        "--plan", str(secret_sentinel), "--authorization", str(secret_sentinel)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {"status": "failed", "error_code": "grounding_session_private_file_invalid"}
    assert completed.stderr == "" and "SYNTHETIC_PATH" not in completed.stdout


def test_cpu_audit_binding_uses_full_real_identity_contract_without_decoding(frozen, tmp_path):
    selection = tmp_path / "private" / "selection.json"
    planner.write_new(selection, {"schema_version": "forte.grounding-pilot-selection.v1",
        "input_manifest_sha256": "d" * 64, "records": frozen["sources"]})
    paths = []
    for index, row in enumerate(frozen["sources"]):
        asset, frames = asset_and_ledger(row["content_sha256"])
        source = validate_source_ledger(asset, frames)
        path = selection.with_name("audit-" + str(index) + ".json")
        planner.write_new(path, {"schema_version": "forte.grounding-source-clock-audit.v1",
            **source.model_dump(mode="json"), "source_ledger": [asdict(frame) for frame in frames],
            "clock_policy": "native_pts_ticks", "model_inference_performed": False})
        paths.append(path)
    output = selection.with_name("bound.json")
    report = planner.bind_source_audits(SimpleNamespace(selection=selection, source_audits=paths,
        clock_policy="native_pts_ticks", output=output))
    assert report["audited_sources"] == report["cfr_sources"] == 2
    assert report["gpu_or_model_calls_made"] == 0
    bound = json.loads(output.read_text())
    assert all(row["complete_native_ledger"] for row in bound["records"])
