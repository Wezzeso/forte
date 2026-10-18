from __future__ import annotations

import asyncio
import json

import pytest

from forte.observability import (AggregateMetrics, StageTimer, TimingMiddleware,
                                 discover_vllm_metric_names, mark_durable_acceptance)
from forte.observability.benchmark import statistics_ms, summarize


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance_ms(self, milliseconds):
        self.now += milliseconds * 1_000_000


def test_monotonic_stage_boundaries_and_failure_duration():
    clock = Clock()
    timer = StageTimer(clock=clock)
    timer.mark("T1")
    clock.advance_ms(12)
    timer.mark("T2")
    timer.mark("T3")
    with pytest.raises(RuntimeError):
        with timer.stage("inference"):
            clock.advance_ms(7)
            raise RuntimeError("synthetic failure")
    timer.mark("T4")
    assert timer.snapshot()["stages_ms"] == {"inference": 7}
    assert timer.elapsed_ms("T1", "T2") == 12
    assert timer.elapsed_ms("T0", "T5") is None
    with pytest.raises(ValueError):
        timer.mark("T2")
    with pytest.raises(ValueError):
        with timer.stage("private caption"):
            pass


def test_asgi_timing_distinguishes_last_body_durable_and_transfer():
    async def scenario():
        clock = Clock()
        metrics = AggregateMetrics()
        messages = []
        body = iter([
            {"type": "http.request", "body": b"first", "more_body": True},
            {"type": "http.request", "body": b"last", "more_body": False},
        ])

        async def receive():
            clock.advance_ms(4)
            return next(body)

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                clock.advance_ms(3)

        async def app(scope, receive, send):
            await receive()
            await receive()
            clock.advance_ms(6)
            mark_durable_acceptance(scope)
            await send({"type": "http.response.start", "status": 202, "headers": []})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        scope = {"type": "http", "path": "/api/v1/analyze"}
        await TimingMiddleware(app, metrics, clock)(scope, receive, send)
        headers = dict(messages[0]["headers"])
        assert headers[b"server-timing"] == b"acceptance;dur=6.000"
        assert headers[b"x-forte-timing-origin"] == b"asgi-local"
        assert scope["forte.timing"].elapsed_ms("request_start", "response_end") == 17
        assert "T0" not in scope["forte.timing"].boundaries_ns
        assert "T5" not in scope["forte.timing"].boundaries_ns
        assert metrics.snapshot()["requests"][0]["count"] == 1

    asyncio.run(scenario())


def test_auth_rejection_never_consumes_body_or_exports_identifier():
    async def scenario():
        metrics = AggregateMetrics()

        async def receive():
            raise AssertionError("auth rejection must not consume body")

        async def send(message):
            pass

        async def reject(scope, receive, send):
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        scope = {"type": "http", "path": "/api/v1/jobs/private-synthetic-id"}
        await TimingMiddleware(reject, metrics)(scope, receive, send)
        assert "T1" not in scope["forte.timing"].boundaries_ns
        assert "private-synthetic-id" not in json.dumps(metrics.snapshot())
        assert metrics.snapshot()["requests"] == [{"route": "job", "status_class": 4, "count": 1}]

    asyncio.run(scenario())


def test_metric_discovery_discards_labels_values_and_does_not_assume_names():
    payload = '# HELP vllm:old_metric help\nvllm:request_success_total{model_name="private-model"} 2\n'
    payload += 'vllm:time_to_first_token_seconds_bucket{le="0.1"} 0\nother_metric{private="text"} 3\n'
    names = discover_vllm_metric_names(payload)
    assert names == ["vllm:request_success_total", "vllm:time_to_first_token_seconds_bucket"]
    assert "private" not in json.dumps(names)
    assert discover_vllm_metric_names("not a metric") == []


def test_small_sample_tail_unmeasured_and_failures_not_discarded():
    assert statistics_ms([1, 2, 3, 4, 5])["p95_ms"] is None
    assert statistics_ms(list(range(1, 21)))["p95_ms"] == 19
    rows = [{"outcome": "completed", "poll_count": 1,
             "client_upload_start_to_full_result_ms": 1200},
            {"outcome": "failed", "poll_count": 0, "error_code": "client_deadline"}]
    report = summarize(rows, cache_mode="hit", model_warmth="warm")
    assert report["failed"] == 1
    assert report["timeouts"] == 1
    assert not report["principal_benchmark_conditions_declared"]
    targets = {entry["target"]: entry["status"] for entry in report["targets"]}
    assert targets["post_body_acceptance_p95"] == "unmeasured"
    assert targets["conservative_organizer_cutoff"] == "unmeasured"


def test_benchmark_upload_stream_and_completed_polling_preserve_privacy(tmp_path):
    import httpx
    from forte.observability.benchmark import run_sample

    source = tmp_path / "private-filename.mov"
    source.write_bytes(b"synthetic media transport fixture")

    async def scenario():
        calls = []

        async def handler(request):
            calls.append(request.method)
            if request.method == "POST":
                content = await request.aread()
                assert b'filename="sample.mov"' in content
                assert b"private-filename" not in content
                return httpx.Response(202, json={"job_id": "opaque-test-job"}, headers={
                    "Server-Timing": "acceptance;dur=4.25", "x-forte-timing-origin": "asgi-local"})
            return httpx.Response(200, json={"job_id": "opaque-test-job", "status": "done",
                                            "verdict": "no_violation", "violations": []}, headers={
                                                "Server-Timing": "lookup;dur=0.4", "x-forte-timing-origin": "asgi-local"})

        exports = {}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            row = await run_sample(client, source, "s001", "http://synthetic", "synthetic-key", 1, .01, 2,
                                   terminal_exports=exports, planned_attempt_id="attempt_" + "a" * 32)
        assert row["outcome"] == "completed"
        assert row["post"]["asgi_post_body_acceptance_ms"] == 4.25
        assert row["http_requests_attempted"] == 4
        assert len(row["completed_get"]) == 3
        assert exports["s001"]["attempt_id"] == "attempt_" + "a" * 32
        assert exports["s001"]["timing_origin"] == "client_upload_start_to_full_result_byte"
        assert exports["s001"]["upload_start_to_full_result_ms"] == row["client_upload_start_to_full_result_ms"]
        exported = json.dumps(row)
        for forbidden in ("private-filename", "opaque-test-job", "no_violation", "synthetic-key", "violations"):
            assert forbidden not in exported

    asyncio.run(scenario())


def test_benchmark_absolute_deadline_counts_timed_out_request(tmp_path):
    import httpx
    from forte.observability.benchmark import run_sample

    source = tmp_path / "synthetic.mov"
    source.write_bytes(b"synthetic")

    async def scenario():
        async def handler(request):
            await asyncio.sleep(.1)
            return httpx.Response(202, json={"job_id": "late"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            row = await run_sample(client, source, "s001", "http://synthetic", "synthetic-key", .01, .01)
        assert row["error_code"] == "client_deadline"
        assert row["http_requests_attempted"] == 1
        assert row["outcome"] == "failed"

    asyncio.run(scenario())


def test_failed_upload_retains_partial_byte_boundaries_without_exception_messages(tmp_path):
    import httpx
    from forte.observability.benchmark import run_sample

    source = tmp_path / "private-source-name.mov"
    source.write_bytes(b"0" * 500_000)

    async def scenario():
        class PartialTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                chunks = request.stream.__aiter__()
                await anext(chunks)  # multipart header
                await anext(chunks)  # first file chunk only
                try:
                    raise ConnectionResetError(104, "private-key-and-caption")
                except ConnectionResetError as cause:
                    raise httpx.ReadError("private-source-name.mov") from cause

        exports = {}
        async with httpx.AsyncClient(transport=PartialTransport()) as client:
            row = await run_sample(client, source, "s001", "http://synthetic", "secret-key", 1, .01,
                                   terminal_exports=exports, planned_attempt_id="attempt_" + "b" * 32)
        progress = row["upload_iterator"]
        assert progress["file_bytes_yielded"] == 256 * 1024
        assert progress["multipart_bytes_yielded"] > progress["file_bytes_yielded"]
        assert progress["first_multipart_chunk_ms"] <= progress["first_file_chunk_ms"]
        assert progress["final_multipart_chunk_ms"] is None
        assert progress["multipart_footer_yielded"] is False
        assert row["transport_diagnostic"]["exception_chain"] == [
            {"type": "ReadError"}, {"type": "ConnectionResetError", "errno": 104}]
        assert exports["s001"]["attempt_id"] == "attempt_" + "b" * 32
        assert exports["s001"]["upload_start_to_full_result_ms"] is None
        assert exports["s001"]["terminal_observed_elapsed_ms"] >= 0
        assert exports["s001"]["durable_acknowledgement_received"] is False
        exported = json.dumps(row)
        for private in ("private-key-and-caption", "private-source-name", "secret-key"):
            assert private not in exported

    asyncio.run(scenario())


def test_reused_content_completes_but_cannot_claim_primary_slo(tmp_path, monkeypatch):
    import httpx
    from forte.observability import benchmark

    source = tmp_path / "fixture.mov"
    source.write_bytes(b"generated fixture")
    output = tmp_path / "paired.json"
    real_client = httpx.AsyncClient

    async def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "opaque-paired-job"})
        return httpx.Response(200, json={"job_id": "opaque-paired-job", "status": "done",
                                        "verdict": "no_violation", "violations": []})

    def client_factory(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    monkeypatch.setenv("FORTE_API_KEY", "generated-test-key")
    assert benchmark.main(["--base-url", "http://synthetic", "--files", str(source), "--output", str(output),
                           "--reused-content-declared", "--model-warmth", "warm", "--cache-mode", "disabled",
                           "--repeat-completed-get", "0"]) == 0
    report = json.loads(output.read_text())
    assert report["summary"]["completed"] == 1
    assert report["evidence_scope"] == "paired_reused_content_exploratory"
    assert report["summary"]["primary_slo_evidence_eligible"] is False
    assert all(row["status"] == "ineligible_reused_content" for row in report["summary"]["targets"])
    assert all("exploratory_observed_status" in row for row in report["summary"]["targets"])
    assert "no_violation" not in output.read_text()


def test_paired_prepare_only_never_launches_and_preserves_original_artifact(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    from pathlib import Path
    import sys

    runner_path = Path(__file__).resolve().parents[1] / "benchmarks/performance/run_selected.py"
    spec = importlib.util.spec_from_file_location("forte_selected_runner_test", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    items = []
    for index in range(6):
        source = tmp_path / f"generated{index}.mov"
        source.write_bytes(bytes([index + 1]))
        items.append({"sample_id": f"fixture{index}", "path": str(source), "content_hash": hashlib.sha256(source.read_bytes()).hexdigest(), "proposed_split": "train"})
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_scope": "proposed_train_benchmark_only", "baseline": items[0], "burst": items[1:]}))
    original = tmp_path / "benchmarks/performance/gpu-observed-summary.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"original":"keep"}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "benchmark_main", lambda *args: pytest.fail("prepare-only must not launch requests"))
    monkeypatch.setattr(sys, "argv", [str(runner_path), "--phase", "paired", "--selection", str(selection),
                                     "--trial-id", "generated-test", "--repeat-content", "--prepare-only"])
    assert module.main() == 0
    plan = json.loads((tmp_path / "benchmarks/quality/trials/generated-test/paired.plan.json").read_text())
    assert plan["input_count"] == 6 and plan["mode"] == "single"
    assert plan["api_requests_made"] == plan["model_requests_made"] == 0
    assert plan["paired_trial"]["content_reused_declared"] is True
    assert plan["paired_trial"]["primary_slo_evidence_eligible"] is False
    assert original.read_text() == '{"original":"keep"}'
    expected_path = tmp_path / "benchmarks/private/trials/generated-test/expected-paired-attempts.json"
    frozen_bytes = expected_path.read_bytes()
    assert module.ensure_attempt_plan(expected_path, items, trial_id="generated-test", phase="paired", mode="single") == hashlib.sha256(frozen_bytes).hexdigest()
    with pytest.raises(ValueError):
        module.ensure_attempt_plan(expected_path, items[:-1], trial_id="generated-test", phase="paired", mode="single")
    assert expected_path.read_bytes() == frozen_bytes
    # Missing exposure-manifest registration must stop before benchmark_main.
    output = tmp_path / "new-paired-report.json"
    monkeypatch.setattr(sys, "argv", [str(runner_path), "--phase", "paired", "--selection", str(selection),
                                     "--trial-id", "generated-fail-closed", "--repeat-content",
                                     "--base-url", "http://synthetic", "--output", str(output)])
    assert module.main() == 1
    assert not output.exists()
    assert not (tmp_path / "benchmarks/private/model-exposure.jsonl").exists()


def test_deployed_profile_capture_discards_secrets_and_unknown_container_strings():
    from pathlib import Path
    import runpy

    sanitize = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/performance/capture_remote_profile.py"))["sanitized_profile"]
    rows = sanitize([{"Name": "/forte-worker-1", "Image": "sha256:" + "a" * 64,
                      "Config": {"Env": ["FORTE_STAGED_SCREENING=true", "FORTE_MAX_INFLIGHT_PER_JOB=2",
                                           "FORTE_API_KEY=private-secret", "FORTE_DATABASE_URL=private-database",
                                           "FORTE_CONCISE_OBSERVATIONS=private-invalid-flag"]},
                      "State": {"Running": True, "Health": {"Status": "healthy"}}},
                     {"Name": "/private-container-name", "Image": "private-image-name", "Config": {"Env": []},
                      "State": {"Health": {"Status": "private-health-text"}}}])
    assert rows[0]["profile_flags"]["FORTE_STAGED_SCREENING"] is True
    assert rows[0]["profile_flags"]["FORTE_MAX_INFLIGHT_PER_JOB"] == 2
    assert rows[0]["profile_flags"]["FORTE_CONCISE_OBSERVATIONS"] is None
    assert rows[1]["container_role"] == "unrecognized" and rows[1]["image_id"] is None
    assert "private-" not in json.dumps(rows)


def test_stream_failure_after_headers_is_counted_even_with_success_status():
    async def scenario():
        metrics = AggregateMetrics()

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            pass

        async def broken(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            raise RuntimeError("synthetic stream failure")

        with pytest.raises(RuntimeError):
            await TimingMiddleware(broken, metrics)({"type": "http", "path": "/api/v1/jobs/synthetic"}, receive, send)
        assert metrics.snapshot()["requests"][0]["status_class"] == 2
        assert metrics.snapshot()["failures"] == [{"route": "job", "count": 1}]

    asyncio.run(scenario())


def test_stage_merge_exports_only_fixed_numeric_fields(tmp_path):
    from pathlib import Path
    import runpy

    merge = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/performance/merge_stages.py"))["merge"]
    http_path = tmp_path / "http.json"
    map_path = tmp_path / "private-map.json"
    log_path = tmp_path / "private-worker.jsonl"
    http_path.write_text(json.dumps({"schema_version": "1.0", "summary": {"attempted": 1, "completed": 1},
                                     "samples": [{"sample_id": "s001", "outcome": "completed"}]}))
    map_path.write_text(json.dumps({"sample_jobs": {"s001": "private-synthetic-job"}}))
    log_path.write_text(json.dumps({"job_id": "private-synthetic-job", "attempt": 2, "status": "done",
                                    "caption": "never export this synthetic caption",
                                    "timings": {"clock": "process_monotonic", "stages_ms": {"analysis": 140},
                                                "analysis_to_publication_ms": 141},
                                    "pipeline_timings": {"clock": "process_monotonic",
                                                         "stages_ms": {"coarse_model": 123, "private_caption": 99}}}))
    report = merge(http_path, map_path, log_path)
    assert report["stage_duration_ms"]["coarse_model"]["max_ms"] == 123
    assert report["missing_terminal_worker_records"] == 0
    assert report["completed_worker_analysis_to_publication"]["p95_ms"] is None
    for forbidden in ("private-synthetic-job", "never export", "private_caption"):
        assert forbidden not in json.dumps(report)


def test_model_metric_aggregation_preserves_only_actual_numeric_observations():
    from forte.observability import aggregate_inference_metrics

    report = aggregate_inference_metrics([
        {"elapsed_sec": 1.5, "http_elapsed_sec": 1.2, "preprocessing_sec": .2,
         "input_dimensions": [(464, 768), (464, 768)], "input_frames": 2,
         "prompt_tokens": 12, "generated_tokens": 5, "visual_tokens": None,
         "caption": "private synthetic caption", "serving_attestation": {"model": "private-path"},
         "server_metrics": {"time_to_first_token_ms": 60, "generation_time_ms": 900,
                            "queue_time_ms": None, "mean_itl_ms": True, "tokens_per_second": 8,
                            "private_field": 999}},
    ])
    assert report["request_durations"]["elapsed_ms"]["sum"] == 1500
    assert report["request_durations"]["server_time_to_first_token_ms"]["sum"] == 60
    assert report["request_durations"]["server_queue_time_ms"]["count"] == 0
    assert report["request_durations"]["server_mean_itl_ms"]["count"] == 0
    assert report["numeric_counts"]["visual_tokens"]["count"] == 0
    assert report["encoded_input_frame_resolutions"][0]["frame_count"] == 2
    for value in ("private synthetic", "private-path", "private_field"):
        assert value not in json.dumps(report)


def test_model_metric_sanitizer_recreates_units_and_strips_malicious_notes():
    from forte.observability import sanitize_model_metrics

    report = sanitize_model_metrics({
        "request_record_count": 1, "scope": "private-synthetic-note", "unknown": "private-synthetic-note",
        "timing_semantics": {"caption": "private-synthetic-note"},
        "request_durations": {"elapsed_ms": {"unit": "private-synthetic-unit", "count": 1,
                                                "sum": 10, "min": True, "max": float("inf"),
                                                "p50": "private-synthetic-note", "p95": None,
                                                "caption": "private-synthetic-note"},
                              "private-synthetic-note": {"count": 1, "sum": 99}},
        "encoded_input_frame_resolutions": [{"width": 464, "height": 768, "frame_count": 2,
                                                "caption": "private-synthetic-note"}],
    })
    assert report["request_durations"]["elapsed_ms"]["unit"] == "ms"
    assert report["request_durations"]["elapsed_ms"]["min"] is None
    assert report["request_durations"]["elapsed_ms"]["max"] is None
    assert report["request_durations"]["elapsed_ms"]["p50"] is None
    assert "private-synthetic" not in json.dumps(report)
    assert sanitize_model_metrics({"request_record_count": True}) is None


def test_metric_overflow_cannot_break_strict_json_logging():
    from forte.observability import aggregate_inference_metrics

    report = aggregate_inference_metrics([{"server_metrics": {"generation_time_ms": 1e308}}] * 2)
    assert report["request_durations"]["server_generation_time_ms"]["sum"] is None
    json.dumps(report, allow_nan=False)


def test_completed_response_for_different_job_is_counted_as_failure(tmp_path):
    import httpx
    from forte.observability.benchmark import run_sample

    source = tmp_path / "synthetic.mov"
    source.write_bytes(b"synthetic")

    async def scenario():
        async def handler(request):
            if request.method == "POST":
                await request.aread()
                return httpx.Response(202, json={"job_id": "submitted-synthetic"})
            return httpx.Response(200, json={"job_id": "different-synthetic", "status": "done",
                                            "verdict": "no_violation", "violations": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            row = await run_sample(client, source, "s001", "http://synthetic", "synthetic-key", 1, .01)
        assert row["error_code"] == "invalid_completed_contract"
        assert row["outcome"] == "failed"
        assert row["http_requests_attempted"] == 2

    asyncio.run(scenario())


def test_metric_totals_strip_labels_and_reject_unknown_or_nonfinite_values():
    from forte.observability.metrics import parse_vllm_metric_totals

    report = parse_vllm_metric_totals(
        'vllm:request_decode_time_seconds_sum{model_name="private-model"} 1.5\n'
        'vllm:request_success_total{finished_reason="private-reason"} 2\n'
        'vllm:request_success_total{finished_reason="other-private-reason"} 3\n'
        'vllm:request_queue_time_seconds_sum{private="text"} NaN\n'
        'unapproved_metric{private="text"} 99\n'
    )
    assert report["vllm:request_success_total"] == {"series_count": 2, "sum_value": 5}
    assert report["vllm:request_decode_time_seconds_sum"]["sum_value"] == 1.5
    assert "vllm:request_queue_time_seconds_sum" not in report
    assert "private" not in json.dumps(report)


def test_metric_delta_requires_quiet_exclusive_window_and_detects_reset():
    from pathlib import Path
    import runpy

    delta = runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/performance/metric_delta.py"))["delta"]

    def snapshot(total, count, process=10):
        return {"status": "measured", "numeric_metric_totals": {
            "process_start_time_seconds": {"sum_value": process},
            "vllm:num_requests_running": {"sum_value": 0}, "vllm:num_requests_waiting": {"sum_value": 0},
            "vllm:request_decode_time_seconds_sum": {"sum_value": total},
            "vllm:request_decode_time_seconds_count": {"sum_value": count}}}

    report = delta(snapshot(5, 2), snapshot(9, 4), exclusive=True)
    assert report["attributed_to_selected_benchmark"]
    assert report["timing_families"]["request_decode_time_seconds"]["server_reported_mean_ms"] == 2000
    assert report["timing_families"]["request_decode_time_seconds"]["p95_ms"] is None
    assert not delta(snapshot(5, 2), snapshot(9, 4), exclusive=False)["attributed_to_selected_benchmark"]
    assert delta(snapshot(5, 2), snapshot(1, 1, process=20), exclusive=True)["status"] == "failed"
