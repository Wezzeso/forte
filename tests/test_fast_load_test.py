"""Offline public-API load checks; all media and responses are synthetic."""

import asyncio
import base64
from io import BytesIO
import json

import httpx
from PIL import Image
import pytest

from tools import fast_load_test as load


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic.mov"
    path.write_bytes(b"generated transport fixture; not real video")
    return path


def execute(video, handler, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await load.run_load("https://synthetic.example", "synthetic-key", [video],
                                       client=client, poll_seconds=0.001, **kwargs)
    return asyncio.run(run())


def completed(job_id):
    return {"job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []}


def test_default_five_parallel_uploads_and_public_contract(video):
    active, peak, next_id = 0, 0, 0
    polled = set()

    async def handler(request):
        nonlocal active, peak, next_id
        assert request.headers["X-API-Key"] == "synthetic-key"
        if request.method == "POST":
            assert request.url.path == "/api/v1/analyze"
            assert b'name="file"' in request.content
            assert b"video/quicktime" in request.content
            active += 1
            peak = max(peak, active)
            next_id += 1
            job_id = str(next_id)
            await asyncio.sleep(.005)
            active -= 1
            return httpx.Response(202, json={"job_id": job_id})
        job_id = request.url.path.rsplit("/", 1)[1]
        assert request.url.path == "/api/v1/jobs/" + job_id
        if job_id not in polled:
            polled.add(job_id)
            return httpx.Response(200, json={"job_id": job_id, "status": "processing"})
        return httpx.Response(200, json=completed(job_id))

    report = execute(video, handler)
    assert report["pass"] and report["passed"] == report["requests"] == 5
    assert report["parallel"] == peak == 5
    assert all(result["polls"] == 2 for result in report["results"])
    assert "synthetic-key" not in json.dumps(report)
    assert str(video) not in json.dumps(report)


def test_six_parallel_is_configurable(video):
    counter = 0

    def handler(request):
        nonlocal counter
        if request.method == "POST":
            counter += 1
            return httpx.Response(202, json={"job_id": str(counter)})
        return httpx.Response(200, json=completed(request.url.path.rsplit("/", 1)[1]))

    report = execute(video, handler, parallel=6)
    assert report["pass"] and report["requests"] == report["parallel"] == 6


@pytest.mark.parametrize("phase,status,error", [
    ("upload", 401, "upload_http_failure"),
    ("upload", 302, "upload_http_failure"),
    ("poll", 503, "poll_http_failure"),
])
def test_http_failures_never_pass(video, phase, status, error):
    def handler(request):
        if request.method == "POST" and phase != "upload":
            return httpx.Response(202, json={"job_id": "synthetic"})
        return httpx.Response(status, json={"private": "response-must-not-be-saved"})

    report = execute(video, handler, parallel=1)
    assert not report["pass"]
    assert report["results"][0]["error_code"] == error
    assert report["results"][0]["http_status"] == status
    assert "response-must-not-be-saved" not in json.dumps(report)


@pytest.mark.parametrize("payload,error", [
    ({"job_id": "synthetic", "status": "failed"}, "analysis_failed"),
    ({"job_id": "other", "status": "done", "verdict": "no_violation", "violations": []}, "job_id_mismatch"),
    ({"job_id": "synthetic", "status": "queued"}, "invalid_job_payload"),
    ({"job_id": "synthetic", "status": "done"}, "invalid_job_payload"),
    ({"job_id": "synthetic", "status": "done", "verdict": "violation", "violations": []}, "invalid_job_payload"),
    ({"job_id": "synthetic", "status": "processing", "private": "do-not-save"}, "invalid_job_payload"),
])
def test_failed_malformed_and_wrong_job_terminal_results_fail(video, payload, error):
    def handler(request):
        return httpx.Response(202, json={"job_id": "synthetic"}) if request.method == "POST" else httpx.Response(200, json=payload)

    report = execute(video, handler, parallel=1)
    assert not report["pass"] and report["failed"] == 1
    assert report["results"][0]["error_code"] == error
    assert "do-not-save" not in json.dumps(report)


@pytest.mark.parametrize("payload", [{}, {"job_id": 123}, {"job_id": "synthetic", "extra": "private"}])
def test_malformed_acceptance_fails(video, payload):
    report = execute(video, lambda request: httpx.Response(202, json=payload), parallel=1)
    assert report["results"][0]["error_code"] == "invalid_upload_payload"


def test_invalid_json_is_not_echoed(video):
    report = execute(video, lambda request: httpx.Response(202, text="private-body"), parallel=1)
    assert report["results"][0]["error_code"] == "invalid_upload_payload"
    assert "private-body" not in json.dumps(report)


def test_upload_time_counts_against_deadline(video):
    now = [0.0]

    def handler(request):
        now[0] += 241
        return httpx.Response(202, json={"job_id": "synthetic"})

    report = execute(video, handler, parallel=1, clock=lambda: now[0])
    result = report["results"][0]
    assert not report["pass"] and result["error_code"] == "deadline_exceeded"
    assert result["elapsed_seconds"] == 241 and result["phase"] == "upload"


def test_done_observed_after_deadline_fails(video):
    now = [0.0]

    def handler(request):
        if request.method == "POST":
            now[0] += 100
            return httpx.Response(202, json={"job_id": "synthetic"})
        now[0] += 141
        return httpx.Response(200, json=completed("synthetic"))

    report = execute(video, handler, parallel=1, clock=lambda: now[0])
    assert report["results"][0]["error_code"] == "deadline_exceeded"
    assert report["results"][0]["elapsed_seconds"] == 241


def test_client_queue_time_counts_against_deadline(video):
    now = [0.0]

    async def handler(request):
        # Yield so the second job reaches the client semaphore, then consume
        # the first job's whole budget without a slow real-world test.
        await asyncio.sleep(.001)
        now[0] += 241
        return httpx.Response(202, json={"job_id": "synthetic"})

    report = execute(video, handler, parallel=1, requests=2, clock=lambda: now[0])
    assert not report["pass"] and report["failed"] == 2
    assert {result["phase"] for result in report["results"]} == {"upload", "client_queue"}
    assert all(result["error_code"] == "deadline_exceeded" for result in report["results"])


def test_slow_requests_are_cancelled_at_wall_clock_deadline(video):
    async def handler(request):
        if request.method == "POST":
            await asyncio.sleep(.05)
            return httpx.Response(202, json={"job_id": "synthetic"})
        return httpx.Response(200, json={"job_id": "synthetic", "status": "processing"})

    report = execute(video, handler, parallel=1, requests=2, deadline_seconds=.01)
    assert not report["pass"] and report["failed"] == 2
    assert all(result["error_code"] == "deadline_exceeded" for result in report["results"])


def test_transport_failure_does_not_leak_exception(video):
    def handler(request):
        raise httpx.ConnectError("private-token-and-host", request=request)

    report = execute(video, handler, parallel=1)
    assert report["results"][0]["error_code"] == "transport_failure"
    assert "private-token-and-host" not in json.dumps(report)


def test_request_timeout_is_distinct_from_elapsed_budget(video):
    def handler(request):
        raise httpx.ReadTimeout("private-request-context", request=request)

    report = execute(video, handler, parallel=1)
    assert report["results"][0]["error_code"] == "request_timeout"
    assert "private-request-context" not in json.dumps(report)


def test_validated_violation_evidence_is_not_persisted(video):
    encoded = BytesIO()
    Image.new("RGB", (8, 8), "white").save(encoded, format="JPEG")
    frame = base64.b64encode(encoded.getvalue()).decode("ascii")
    payload = {"job_id": "synthetic", "status": "done", "verdict": "violation", "violations": [
        {"type": "synthetic-type", "timecode_sec": 1.0, "description": "private-description",
         "evidence_frames": [{"timestamp_sec": 1.0, "image_base64": frame}]},
    ]}

    def handler(request):
        return httpx.Response(202, json={"job_id": "synthetic"}) if request.method == "POST" else httpx.Response(200, json=payload)

    report = execute(video, handler, parallel=1)
    assert report["pass"] and report["results"][0]["completed_schema_valid"]
    assert all(value not in json.dumps(report) for value in [frame, "private-description", "synthetic-type"])


def test_missing_file_is_reported_without_path(tmp_path):
    report = execute(tmp_path / "private-missing.mov", lambda request: pytest.fail("no upload expected"), parallel=1)
    assert report["results"][0]["error_code"] == "input_unavailable"
    assert "private-missing" not in json.dumps(report)


def test_cli_writes_report_and_returns_failure(video, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SYNTHETIC_KEY", "secret-never-printed")

    async def fake_run(*args, **kwargs):
        assert args[1] == "secret-never-printed"
        assert kwargs["parallel"] == 6
        return {"pass": False, "requests": 6, "passed": 5, "failed": 1, "max_request_seconds": 241}

    monkeypatch.setattr(load, "run_load", fake_run)
    report_path = tmp_path / "load.json"
    code = load.main(["--base-url", "https://synthetic.example", "--video", str(video),
                      "--parallel", "6", "--api-key-env", "SYNTHETIC_KEY", "--report", str(report_path)])
    assert code == 1
    assert json.loads(report_path.read_text())["failed"] == 1
    assert "secret-never-printed" not in capsys.readouterr().out


@pytest.mark.parametrize("url", ["https://user:secret@example.test", "https://example.test?secret=1", "https://example.test/api"])
def test_reject_credentialled_or_non_origin_url(video, url):
    with pytest.raises(ValueError):
        asyncio.run(load.run_load(url, "key", [video]))
