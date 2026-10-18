"""Generated-only acceptance checks: no private media, model, GPU or network."""
from copy import deepcopy
import base64
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json

import httpx
import pytest

from benchmarks.performance import qwen38_acceptance as driver
from benchmarks.performance.repeatability import RepeatabilityError, digest, file_digest


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr(driver, "PRIVATE", root)
    source = root / "generated.mov"
    source.write_bytes(b"generated transport fixture; deliberately not real video")
    spec = {"schema_version": "forte.qwen38-acceptance-spec.v1", "arm": "qwen38_tuned",
        "cases": [{"case_id": "case1", "sample_id": "sample1", "split": "train",
                   "owned_path": str(source), "source_bytes": source.stat().st_size,
                   "content_sha256": file_digest(source)}],
        "expected_model_identity": {"model_id": "generated/Qwen3.8-27B", "model_revision": "r1",
                                    "adapter_sha256": "a" * 64},
        "recipe_sha256": "b" * 64, "comparison_plan_sha256": "c" * 64,
        "settings": {"base_url": "https://generated.invalid", "whole_result_cache": False,
                     "concurrency": 1, "normalization": ["job_id"], "deadline_seconds": 1.,
                     "request_timeout_seconds": 1., "poll_seconds": .001}}
    path = root / "plan.json"
    driver.prepare(spec, path)
    return driver.read(path), path, root, source


def authorization(plan, path):
    release = {"schema_version": "forte.qwen38-root-release.v1", "released_by_root": True,
        "uploads_released": True, "phase": "acceptance", "attempt_plan_sha256": file_digest(path),
        "maximum_attempts": 3, "maximum_execution_seconds": 60,
        "exclusive_idle_gpu_handoff_verified": True, "idle_handoff_receipt_sha256": "d" * 64,
        "expected_model_identity": plan["expected_model_identity"],
        "not_before_utc": "2026-10-17T00:00:00+00:00", "expires_at_utc": "2026-10-17T23:59:59+00:00",
        **{key: "e" * 64 for key in ("config_sha256", "source_archive_sha256", "runtime_image_sha256", "base_manifest_sha256")}}
    receipt = {"schema_version": "forte.qwen38-reservation.v1", "status": "reserved", "phase": "acceptance",
        "attempt_plan_sha256": file_digest(path), "released_by_root": True, "release_sha256": "f" * 64,
        "requested_attempts": 3, "registry_tip_sha256": "0" * 64,
        "attempts": [{key: row[key] for key in ("attempt_id", "content_sha256", "split")} for row in plan["attempts"]],
        **{key: release[key] for key in ("config_sha256", "expires_at_utc", "source_archive_sha256", "runtime_image_sha256", "base_manifest_sha256")}}
    return release, receipt


def successful(plan, index):
    return {**plan["attempts"][index], "status": "done", "post_attempted": True,
        "job_id": f"generated-job-{index}", "public_contract_validated": True,
        "server_receipt_verified": True, "upload_start_to_full_result_ms": 9_999.,
        **{key: "a" * 64 for key in ("normalized_result_sha256", "native_source_sha256", "decision_sha256", "native_evidence_sha256")}}


def test_prepare_freezes_three_slots_without_execution(prepared):
    plan, path, root, source = prepared
    assert len(plan["attempts"]) == 3
    assert [row["repeat_index"] for row in plan["attempts"]] == [0, 1, 2]
    assert driver.validate_plan(plan) == plan
    with pytest.raises(FileExistsError):
        driver.prepare({**plan, "schema_version": "forte.qwen38-acceptance-spec.v1"}, path)


@pytest.mark.parametrize("change", ["few_repeats", "final", "normalization", "cache", "wrong_hash"])
def test_prepare_rejects_scope_changes(prepared, change):
    plan, path, root, source = prepared
    changed = deepcopy(plan)
    if change == "few_repeats":
        changed["repeats"] = 2
    elif change == "final":
        changed["cases"][0]["split"] = "final_check"
    elif change == "normalization":
        changed["settings"]["normalization"].append("event_id")
    elif change == "cache":
        changed["settings"]["whole_result_cache"] = True
    else:
        changed["attempts"][0]["content_sha256"] = "0" * 64
    changed["conditions_sha256"] = digest({key: value for key, value in changed.items() if key != "conditions_sha256"})
    with pytest.raises(RepeatabilityError):
        driver.validate_plan(changed)


def test_only_job_id_normalized():
    left = {"job_id": "a", "event_id": "e", "timings": {"model": 1}, "facts": ["first", "second"]}
    right = {**left, "job_id": "b"}
    assert driver.normalized(left) == driver.normalized(right)
    for field, value in (("event_id", "other"), ("timings", {"model": 2}), ("facts", ["second", "first"])):
        assert driver.normalized(left) != driver.normalized({**right, field: value})


@pytest.mark.parametrize("change", [None, "expired", "attempt", "runtime", "release_hash", "not_root"])
def test_exact_release_and_reservation(prepared, change):
    plan, path, root, source = prepared
    release, receipt = authorization(plan, path)
    if change == "expired":
        release["expires_at_utc"] = "2026-10-16T23:59:59+00:00"
    elif change == "attempt":
        receipt["attempts"].pop()
    elif change == "runtime":
        receipt["runtime_image_sha256"] = "0" * 64
    elif change == "release_hash":
        receipt["release_sha256"] = "0" * 64
    elif change == "not_root":
        release["released_by_root"] = False
    kwargs = {"plan_sha256": file_digest(path), "release_sha256": "f" * 64,
              "now": datetime(2026, 10, 17, 12, tzinfo=timezone.utc)}
    if change:
        with pytest.raises(ValueError):
            driver.validate_authorization(plan, release, receipt, **kwargs)
    else:
        driver.validate_authorization(plan, release, receipt, **kwargs)


async def test_fresh_posts_and_complete_response_bytes_measured(prepared):
    plan, path, root, source = prepared
    calls = []
    async def handler(request):
        calls.append(request.method)
        if request.method == "POST":
            assert await request.aread()  # Consume the actual fresh UploadStream.
            return httpx.Response(202, json={"job_id": "new-job"})
        return httpx.Response(200, json={"job_id": "new-job", "status": "done",
                                       "verdict": "no_violation", "violations": []})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await driver.run_attempt(client, source, plan["attempts"][0], plan["settings"], "generated-key")
    assert calls == ["POST", "GET"]
    assert row["status"] == "done" and row["public_contract_validated"]
    assert row["upload_start_to_full_result_ms"] > 0
    assert row["upload_start_to_validated_result_ms"] >= row["upload_start_to_full_result_ms"]
    assert row["upload_iterator"]["file_bytes_yielded"] == source.stat().st_size
    assert row["complete_result"]["violations"] == []
    assert row["evidence_sha256"] == []


@pytest.mark.parametrize("mode", ["failed", "wrong_job", "invalid_evidence", "timeout", "post_error"])
async def test_failed_uploads_and_full_contract_failures_recorded(prepared, mode):
    plan, path, root, source = prepared
    async def handler(request):
        if mode == "timeout":
            raise httpx.ReadTimeout("generated secret/path must not be exported")
        if request.method == "POST":
            return httpx.Response(503 if mode == "post_error" else 202, json={"job_id": "new-job"})
        value = {"job_id": "other" if mode == "wrong_job" else "new-job", "status": "done",
                 "verdict": "no_violation", "violations": []}
        if mode == "failed":
            value["status"] = "failed"
        if mode == "invalid_evidence":
            value.update(verdict="violation", violations=[{"type": "cash", "description": "generated",
                "timecode_sec": 0., "evidence_frames": [{"timestamp_sec": 0., "image_base64": "garbage"}]}])
        return httpx.Response(200, json=value)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await driver.run_attempt(client, source, plan["attempts"][0], plan["settings"], "generated-key")
    assert row["status"] == ("timeout" if mode == "timeout" else "failed")
    assert row["post_attempted"] and row["upload_start_to_full_result_ms"] is None
    assert not row["public_contract_validated"]
    assert "generated secret/path" not in json.dumps(row)


def test_failure_inclusive_denominator_and_strict_less_than(prepared):
    plan = prepared[0]
    rows = [successful(plan, index) for index in range(3)]
    summary = driver.summarize(plan, rows)
    assert summary["less_than_ten_seconds"] == summary["fresh_repeatability"] == "demonstrated"
    rows[0]["upload_start_to_full_result_ms"] = 10_000.
    assert driver.summarize(plan, rows)["less_than_ten_seconds"] == "failed"
    rows[0] = {**plan["attempts"][0], "status": "timeout", "post_attempted": True}
    summary = driver.summarize(plan, rows)
    assert summary["expected_attempts"] == 3 and summary["completed"] == 2
    assert summary["less_than_ten_seconds"] == summary["fresh_repeatability"] == "failed"
    assert driver.summarize(plan, rows[1:])["less_than_ten_seconds"] == "unverified"


def test_model_or_native_receipts_required_and_all_pairs_compared(prepared):
    plan = prepared[0]
    rows = [successful(plan, index) for index in range(3)]
    rows[1]["server_receipt_verified"] = False
    assert driver.summarize(plan, rows)["fresh_repeatability"] == "unverified"
    rows[1]["server_receipt_verified"] = True
    rows[1]["decision_sha256"] = "b" * 64
    summary = driver.summarize(plan, rows)
    assert summary["fresh_repeatability"] == "failed"
    assert summary["repeatability_cases"][0] == {"complete": True, "verified_attempts": 3,
                                                "compared_pairs": 3, "equal_pairs": 1}


def test_cached_job_identity_cannot_be_fresh_repeatability(prepared):
    plan = prepared[0]
    rows = [successful(plan, index) for index in range(3)]
    rows[1]["job_id"] = rows[0]["job_id"]
    summary = driver.summarize(plan, rows)
    assert summary["fresh_repeatability"] == "failed"
    assert not summary["fresh_job_identities_distinct"]


async def test_actual_jpeg_bytes_and_evidence_transfer_are_captured(prepared):
    from PIL import Image
    plan, path, root, source = prepared
    output = BytesIO()
    Image.new("RGB", (9, 7), "green").save(output, format="JPEG")
    jpeg = output.getvalue()
    result = {"job_id": "positive-job", "status": "done", "verdict": "violation",
        "violations": [{"type": "generated", "timecode_sec": .03336666666666667,
            "description": "Generated fixture only", "evidence_frames": [{
                "timestamp_sec": .03336666666666667, "image_base64": base64.b64encode(jpeg).decode()}]}]}
    async def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "positive-job"})
        return httpx.Response(200, json=result)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await driver.run_attempt(client, source, plan["attempts"][0], plan["settings"], "generated-key")
    assert row["status"] == "done" and row["public_contract_validated"]
    assert row["polls"][0]["response_bytes"] > len(jpeg)
    assert row["evidence_sha256"] == [{"event_index": 0, "frame_index": 0,
        "timestamp_sec": .03336666666666667, "jpeg_sha256": driver.hashlib.sha256(jpeg).hexdigest()}]
    assert row["upload_start_to_usable_result_ms"] is None and row["ui_timing_origin"] is None


def test_offline_report_retains_missing_terminal_slots(prepared):
    plan, path, root, source = prepared
    run = root / "run"
    run.mkdir()
    driver.write_new(run / "plan.json", plan)
    directory = run / plan["attempts"][0]["attempt_id"]
    directory.mkdir()
    driver.write_new(directory / "submission-intent.json", plan["attempts"][0])
    summary = driver.report(run, root / "terminals.json", root / "summary.json")
    private = driver.read(root / "terminals.json")
    assert summary["expected_attempts"] == summary["missing_or_not_attempted"] == 3
    assert len(private["records"]) == 3
    assert private["records"][0]["post_attempted"] is True
    assert summary["less_than_ten_seconds"] == summary["fresh_repeatability"] == "unverified"
    assert "content_sha256" not in json.dumps(summary)


async def test_executor_rejects_invalid_release_before_client_or_claim(prepared, monkeypatch):
    plan, path, root, source = prepared
    driver.write_new(root / "bad-release.json", {})
    driver.write_new(root / "bad-reservation.json", {})
    def forbidden_client(**kwargs):
        raise AssertionError("HTTP client must never be constructed")
    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)
    with pytest.raises(RepeatabilityError):
        await driver.execute(path, root / "bad-release.json", root / "bad-reservation.json", root / "run", "GENERATED_KEY")
    assert not path.with_suffix(".execution-claim.json").exists()
    assert not (root / "run").exists()


async def test_one_use_plan_cannot_replay_same_slots_in_another_output(prepared, monkeypatch):
    plan, path, root, source = prepared
    release = {"maximum_execution_seconds": 30,
               "expires_at_utc": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()}
    driver.write_new(root / "release.json", release)
    driver.write_new(root / "reservation.json", {})
    monkeypatch.setattr(driver, "validate_authorization", lambda *args, **kwargs: None)
    monkeypatch.setenv("GENERATED_KEY", "generated-only")
    posts = []
    async def handler(request):
        if request.method == "POST":
            await request.aread()
            posts.append(request.method)
            return httpx.Response(202, json={"job_id": f"new-job-{len(posts)}"})
        return httpx.Response(200, json={"job_id": request.url.path.rsplit("/", 1)[1], "status": "done",
                                       "verdict": "no_violation", "violations": []})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs:
        real_client(transport=httpx.MockTransport(handler), **kwargs))
    await driver.execute(path, root / "release.json", root / "reservation.json", root / "run", "GENERATED_KEY")
    assert len(posts) == 3
    assert path.with_suffix(".execution-claim.json").is_file()
    for attempt in plan["attempts"]:
        row = driver.read(root / "run" / attempt["attempt_id"] / "attempt.json")
        assert row["expected_model_identity"] == plan["expected_model_identity"]
    with pytest.raises(FileExistsError):
        await driver.execute(path, root / "release.json", root / "reservation.json", root / "other-run", "GENERATED_KEY")
    assert len(posts) == 3 and not (root / "other-run").exists()


def test_native_receipt_rational_clock_and_model_binding(prepared):
    plan = prepared[0]
    row = successful(plan, 0)
    row["evidence_sha256"] = []
    receipt = {"schema_version": "forte.qwen38-acceptance-server-receipt.v1",
        "attempt_id": row["attempt_id"], "job_id": row["job_id"], "content_sha256": row["content_sha256"],
        "conditions_sha256": plan["conditions_sha256"], "normalized_result_sha256": row["normalized_result_sha256"],
        "recipe_sha256": plan["recipe_sha256"],
        "actual_model_identity": plan["expected_model_identity"],
        "cache_hits": {"decode": 0, "model": 0, "evidence": 0, "whole_result": 0},
        "stage_calls": {"decode": 1, "model": 2, "evidence": 1},
        "native_source": {"content_sha256": row["content_sha256"], "source_complete": True,
            "tail_included": True, "origin_pts": 90_000, "time_base_num": 1, "time_base_den": 90_000,
            "decoded_source_frame_count": 2,
            "frames": [{"frame_id": "f0", "source_pts": 90_000, "timestamp_sec": 0.},
                       {"frame_id": "f1", "source_pts": 93_003, "timestamp_sec": 3003 / 90_000}]},
        "evidence_frames": [], "decision_projection": {"facts": [{"state": "unclear"}], "limitations": []}}
    assert driver.bind_server_receipt(plan, row, receipt)["server_receipt_verified"]
    changed = deepcopy(receipt)
    changed["native_source"]["frames"][1]["source_pts"] += 1
    with pytest.raises(RepeatabilityError, match="native_time"):
        driver.bind_server_receipt(plan, row, changed)
    changed = deepcopy(receipt)
    changed["actual_model_identity"] = {"model_id": "other"}
    with pytest.raises(RepeatabilityError, match="identity"):
        driver.bind_server_receipt(plan, row, changed)
    changed = deepcopy(receipt)
    changed["cache_hits"]["model"] = 1
    with pytest.raises(RepeatabilityError, match="fresh_stages"):
        driver.bind_server_receipt(plan, row, changed)
