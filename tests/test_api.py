"""Synthetic-only API reliability checks; these never inspect private media."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from forte.api.config import Settings
from forte.api.main import _worker_alive, create_app
import forte.api.metadata as ui_metadata
from forte.api.reviews import MAX_REVIEW_BODY_BYTES
from forte.jobs import JobStore
from forte.shared.schemas import CONTRACT_VERSION


@pytest.fixture
def api(tmp_path):
    config = Settings(api_keys=("synthetic-test-key",), data_dir=tmp_path / "runtime")
    store = JobStore("sqlite:///:memory:", config.result_root)
    app = create_app(config, store, readiness=lambda: {"database": True, "storage": True, "model": False, "worker": False})
    with TestClient(app) as client:
        yield client, store, config, app
    store.engine.dispose()


AUTH = {"X-API-Key": "synthetic-test-key"}


def upload(client, contents=b"synthetic MOV input; intentionally undecoded", name="fixture.mov"):
    return client.post("/api/v1/analyze", headers=AUTH, files={"file": (name, contents, "video/quicktime")})


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_auth_rejects_before_reading_request_body(api, headers):
    _, _, _, app = api
    messages = []
    reads = []

    async def run():
        async def receive():
            reads.append(True)
            raise AssertionError("Unauthorized bodies must never be consumed")

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/api/v1/analyze",
            "raw_path": b"/api/v1/analyze", "query_string": b"",
            "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
            "server": ("test", 80), "client": ("test", 123),
        }
        await app(scope, receive, send)

    asyncio.run(run())
    assert messages[0]["status"] == 401
    assert not reads


def test_duplicate_auth_headers_are_rejected(api):
    client, *_ = api
    response = client.get("/api/v1/jobs/unknown", headers=[("X-API-Key", "synthetic-test-key"), ("X-API-Key", "wrong")])
    assert response.status_code == 401


def test_upload_is_owned_durable_then_recoverable(api):
    client, store, config, _ = api
    content = b"synthetic-video" * 16000
    response = upload(client, content, "../../unsafe.MOV")
    assert response.status_code == 202
    assert set(response.json()) == {"job_id"}
    job_id = response.json()["job_id"]
    record = store.get_job(job_id)
    assert Path(record.input_path).parent == config.upload_root
    assert Path(record.input_path).read_bytes() == content
    assert record.input_sha256 == hashlib.sha256(content).hexdigest()
    assert record.input_size_bytes == len(content)
    assert record.original_filename == "unsafe.MOV"
    assert not list(config.upload_root.glob("*.part"))
    assert "acceptance;dur=" in response.headers["server-timing"]
    assert response.headers["x-forte-timing-origin"] == "asgi-local"
    assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json() == {"job_id": job_id, "status": "processing"}


def test_completed_result_is_immutable_file_without_reencoding(api, monkeypatch):
    client, store, _, _ = api
    job_id = upload(client).json()["job_id"]
    claimed = store.claim("test-worker")
    result = {"job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []}
    record = store.publish(job_id, "test-worker", claimed.attempt, result)
    exact = Path(record.result_path).read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Polling must serve the immutable published file")

    monkeypatch.setattr(store, "get_result", forbidden)
    monkeypatch.setattr(store, "get_result_bytes", forbidden)
    first = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
    second = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
    assert first.content == second.content == exact
    assert first.json() == result
    assert first.headers["cache-control"] == "no-store"
    assert "lookup;dur=" in first.headers["server-timing"]


def test_positive_contract_fixture_is_published_unchanged(api):
    client, store, *_ = api
    job_id = upload(client).json()["job_id"]
    fixture_path = Path(__file__).resolve().parents[1] / "contracts" / "fixtures" / "positive.json"
    # A01 fixture contains only generated synthetic JPEG pixels.
    result = json.loads(fixture_path.read_text(encoding="utf-8"))
    result["job_id"] = job_id
    claimed = store.claim("test-worker")
    store.publish(job_id, "test-worker", claimed.attempt, result)
    response = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == result


def test_failed_poll_exposes_no_private_diagnostics(api):
    client, store, *_ = api
    job_id = upload(client).json()["job_id"]
    record = store.claim("test-worker")
    store.fail(job_id, "test-worker", record.attempt, "secret path and private diagnostics")
    response = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH)
    assert response.json() == {"job_id": job_id, "status": "failed"}
    metadata = client.get(f"/api/ui/jobs/{job_id}", headers=AUTH).json()
    assert metadata["stage"] == "failed"
    assert metadata["error_code"] == "analysis_failed"
    assert "secret" not in json.dumps(metadata)
    assert "input_path" not in metadata


@pytest.mark.parametrize("name,contents,status", [("wrong.mp4", b"abc", 415), ("empty.mov", b"", 400)])
def test_invalid_uploads_are_cleaned(api, name, contents, status):
    client, _, config, _ = api
    assert upload(client, contents, name).status_code == status
    assert list(config.upload_root.iterdir()) == []


def test_missing_duplicate_or_truncated_multipart_fails(api):
    client, _, config, _ = api
    assert client.post("/api/v1/analyze", headers=AUTH, files={"other": ("fixture.mov", b"abc")}).status_code == 400
    assert client.post("/api/v1/analyze", headers=AUTH, files=[("file", ("one.mov", b"abc")), ("file", ("two.mov", b"def"))]).status_code == 400
    body = b'--cut\r\nContent-Disposition: form-data; name="file"; filename="a.mov"\r\n\r\nunfinished'
    response = client.post("/api/v1/analyze", headers={**AUTH, "Content-Type": "multipart/form-data; boundary=cut"}, content=body)
    assert response.status_code == 400
    assert list(config.upload_root.iterdir()) == []


def test_streamed_upload_enforces_file_limit_without_content_length(tmp_path):
    config = Settings(api_keys=("synthetic-test-key",), data_dir=tmp_path, max_upload_bytes=10)
    store = JobStore("sqlite:///:memory:", config.result_root)
    with TestClient(create_app(config, store)) as client:
        body = b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.mov"\r\n\r\n' + b"a" * 11 + b"\r\n--x--\r\n"
        response = client.post("/api/v1/analyze", headers={**AUTH, "Content-Type": "multipart/form-data; boundary=x"}, content=iter([body[:40], body[40:]]))
        assert response.status_code == 413
        assert not list(config.upload_root.iterdir())
    store.engine.dispose()


def test_oversized_multipart_headers_are_bounded(api):
    client, _, config, _ = api
    body = b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.mov"\r\nX-Large: ' + b"a" * 17000 + b"\r\n\r\nabc\r\n--x--\r\n"
    response = client.post("/api/v1/analyze", headers={**AUTH, "Content-Type": "multipart/form-data; boundary=x"}, content=body)
    assert response.status_code == 400
    assert list(config.upload_root.iterdir()) == []


def test_failed_job_insert_never_acknowledges_or_leaves_owned_file(api, monkeypatch):
    client, store, config, _ = api

    def failed(*args, **kwargs):
        raise RuntimeError("synthetic database unavailable")

    monkeypatch.setattr(store, "create_job", failed)
    assert upload(client).status_code == 503
    assert list(config.upload_root.iterdir()) == []


def test_committed_but_uncertain_insert_keeps_input(api, monkeypatch):
    client, store, config, _ = api
    original = store.create_job

    def committed_then_timeout(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError("synthetic lost commit acknowledgement")

    monkeypatch.setattr(store, "create_job", committed_then_timeout)
    assert upload(client).status_code == 503
    assert len(list(config.upload_root.glob("*.mov"))) == 1
    assert store.claim("worker") is not None


def test_source_media_requires_auth_and_supports_ranges(api):
    client, _, _, _ = api
    job_id = upload(client, b"synthetic0123456789").json()["job_id"]
    endpoint = f"/api/ui/jobs/{job_id}/video"
    assert client.get(endpoint).status_code == 401
    response = client.get(endpoint, headers={**AUTH, "Range": "bytes=0-8"})
    assert response.status_code == 206
    assert response.content == b"synthetic"
    assert response.headers["content-type"] == "video/quicktime"


def test_unknown_jobs_and_missing_result_fail_honestly(api, monkeypatch):
    client, store, config, _ = api
    assert client.get("/api/v1/jobs/missing", headers=AUTH).status_code == 404
    job_id = upload(client).json()["job_id"]
    record = store.get_job(job_id)
    monkeypatch.setattr(store, "get_job", lambda _: replace(record, status="done", result_path=str(config.result_root / "missing.json")))
    assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).status_code == 503


def test_health_is_alive_but_readiness_requires_model_and_worker(api):
    client, *_ = api
    assert client.get("/healthz").json() == {"status": "alive"}
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["components"]["model"] is False


def test_worker_readiness_requires_recent_valid_heartbeat(tmp_path):
    config = Settings(data_dir=tmp_path)
    assert not _worker_alive(config)
    config.worker_heartbeat_file.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat()}))
    assert _worker_alive(config)
    config.worker_heartbeat_file.write_text(json.dumps({"updated_at": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()}))
    assert not _worker_alive(config)


def test_settings_keys_file_and_no_secret_repr(tmp_path, monkeypatch):
    keyfile = tmp_path / "test.keys"
    keyfile.write_text("second-key\nthird-key\n")
    monkeypatch.setenv("FORTE_API_KEYS", "first-key")
    monkeypatch.setenv("FORTE_API_KEYS_FILE", str(keyfile))
    monkeypatch.setenv("FORTE_DATA_DIR", str(tmp_path))
    settings = Settings.from_env()
    assert settings.api_keys == ("first-key", "second-key", "third-key")
    assert "first-key" not in repr(settings)
    assert "second-key" not in repr(settings)


def test_default_store_uses_configured_data_directory(tmp_path):
    config = Settings(api_keys=("synthetic-test-key",), data_dir=tmp_path)
    with TestClient(create_app(config)) as client:
        assert upload(client).status_code == 202
    assert (tmp_path / "jobs.db").is_file()


def test_database_outage_keeps_health_alive_without_acknowledging_upload(tmp_path):
    config = Settings(api_keys=("synthetic-test-key",), data_dir=tmp_path, database_url="unsupported://synthetic")
    with TestClient(create_app(config)) as client:
        assert client.get("/healthz").status_code == 200
        response = upload(client)
        assert response.status_code == 503
        assert "unsupported" not in response.text
        assert list(config.upload_root.iterdir()) == []


_SYNTHETIC_ATTESTATION = {
    "ready": True,
    "model_files_verified": True,
    "model_revision": "a" * 40,
    "vllm_version": "9.9.9",
}


@pytest.mark.parametrize("change,health_status,attestation_status,expected", [
    ({}, 200, 200, True),
    ({"ready": False}, 200, 200, False),
    ({"ready": "true"}, 200, 200, False),
    ({"model_files_verified": False}, 200, 200, False),
    ({"model_files_verified": "true"}, 200, 200, False),
    ({"model_revision": "b" * 40}, 200, 200, False),
    ({"vllm_version": "9.9.8"}, 200, 200, False),
    ({"model_revision": None}, 200, 200, False),
    ({}, 503, 200, False),
    ({}, 200, 503, False),
])
def test_readiness_requires_verified_matching_model_attestation(
    tmp_path, monkeypatch, change, health_status, attestation_status, expected,
):
    manifest = tmp_path / "manifest.json"
    # Deliberately different synthetic pins prove the API reads its manifest.
    manifest.write_text(json.dumps({"revision": "a" * 40, "serving": {"version": "9.9.9"}}))
    config = Settings(api_keys=("synthetic-test-key",), data_dir=tmp_path / "runtime", model_manifest_path=manifest)
    config.data_dir.mkdir()
    config.worker_heartbeat_file.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat()}))
    store = JobStore("sqlite:///:memory:", config.result_root)
    original_async_client = httpx.AsyncClient
    requested_paths = []

    def model_route(request):
        requested_paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(health_status, json={"status": "alive"})
        if request.url.path == "/readyz":
            return httpx.Response(attestation_status, json={**_SYNTHETIC_ATTESTATION, **change})
        raise AssertionError("Readiness must never invoke inference")

    def synthetic_client(**kwargs):
        return original_async_client(transport=httpx.MockTransport(model_route), **kwargs)

    monkeypatch.setattr("forte.api.main.httpx.AsyncClient", synthetic_client)
    with TestClient(create_app(config, store)) as client:
        response = client.get("/readyz")
    assert response.status_code == (200 if expected else 503)
    assert response.json()["components"]["model"] is expected
    assert sorted(requested_paths) == ["/health", "/readyz"]
    assert "model_revision" not in response.text
    store.engine.dispose()


def test_missing_model_manifest_fails_readiness_without_model_request(tmp_path, monkeypatch):
    config = Settings(api_keys=("synthetic-test-key",), data_dir=tmp_path, model_manifest_path=tmp_path / "missing.json")
    store = JobStore("sqlite:///:memory:", config.result_root)

    def forbidden(**_kwargs):
        raise AssertionError("Expected pins must exist before querying a model service")

    monkeypatch.setattr("forte.api.main.httpx.AsyncClient", forbidden)
    with TestClient(create_app(config, store)) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["components"]["model"] is False
    store.engine.dispose()


def _publish_synthetic_ui_audit(client, store, config, rule_ids=None):
    from rules.catalog import load_catalog

    catalog = load_catalog(config.rules_path)
    selected = ["RK_NBK231_P22_RECEIPT_DEPARTURE"] if rule_ids is None else rule_ids
    job_id = upload(client).json()["job_id"]
    claimed = store.claim("synthetic-worker")
    fixture_path = Path(__file__).resolve().parents[1] / "contracts" / "fixtures" / "positive.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    result = {"job_id": job_id, "status": "done", "verdict": "violation" if selected else "no_violation", "violations": []}
    events = []
    for index, rule_id in enumerate(selected):
        rule = catalog.get_rule(rule_id)
        binding = catalog.bindings[rule_id]
        facts = [{"fact_id": key, "state": state, "supporting_frame_ids": ["synthetic-frame"],
                  "notes": "PRIVATE_MODEL_NOTE_MUST_NOT_LEAVE_AUDIT"}
                 for key, state in binding["required_fact_states"].items()]
        event = {
            "event_id": f"synthetic-event-{index}", "event_type_id": binding["event_type_id"], "rule_id": rule_id,
            "start_sec": 1.0, "end_sec": 2.0, "anchor_frame_id": "synthetic-frame",
            "evidence_frame_ids": ["synthetic-frame"], "observed_facts": facts,
            "description": "PRIVATE_MODEL_DESCRIPTION_MUST_NOT_LEAVE_AUDIT",
            "limitations": ["unclear_visual_context", "PRIVATE_VISIBILITY_NOTE_MUST_NOT_LEAVE_AUDIT"],
        }
        events.append(event)
        violation = deepcopy(fixture["violations"][0])
        violation["type"] = catalog.event_type_label(binding["event_type_id"])
        violation["description"] = catalog.render_description(rule, [
            catalog.fact_description(key, state) for key, state in binding["required_fact_states"].items()
        ])
        result["violations"].append(violation)
    store.publish(job_id, "synthetic-worker", claimed.attempt, result)
    audit = {
        "schema_version": "forte.local-proposal.v1", "phase": "done", "publication_ready": True,
        "analysis_version": CONTRACT_VERSION, "catalog_version": catalog.version,
        "job_id": job_id, "attempt": claimed.attempt,
        "coverage": {"complete": True, "policy": "overlapping_sparse_windows", "private_detail": "PRIVATE_COVERAGE_DETAIL"},
        "evidence": {"event_count": len(events)}, "events": events, "verified_events": deepcopy(events),
        "limitations": ["unresolved_coarse_candidate", "PRIVATE_LIMIT_MUST_NOT_LEAVE_AUDIT"],
        "frames": [{"owned_path": "C:/PRIVATE_INTERNAL_FRAME_PATH"}],
        "raw_predictions": "PRIVATE_RAW_PREDICTION",
    }
    path = config.data_dir / "work" / job_id / f"attempt-{claimed.attempt}" / "analysis.private.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(audit), encoding="utf-8")
    return job_id, path, audit, catalog, result


def test_ui_metadata_emits_reviewed_sources_and_limits_without_raw_notes(api):
    client, store, config, _ = api
    job_id, _, _, catalog, result = _publish_synthetic_ui_audit(client, store, config)
    jury_before = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).content
    response = client.get(f"/api/ui/jobs/{job_id}", headers=AUTH)
    metadata = response.json()
    assert response.status_code == 200
    event = metadata["events"][0]
    rule = catalog.get_rule("RK_NBK231_P22_RECEIPT_DEPARTURE")
    assert event["rule"] == {"title": rule.title, "clause": rule.clause, "url": rule.source_url}
    assert event["observed_facts"] == [catalog.fact_description(key) for key in catalog.bindings[rule.rule_id]["description_fact_ids"]]
    assert ui_metadata._LIMIT_TEXT["unclear_visual_context"] in event["limitations"]
    assert ui_metadata._LIMIT_TEXT["unresolved_coarse_candidate"] in metadata["limitations"]
    assert any("временным шагом" in text for text in metadata["limitations"])
    assert "PRIVATE_" not in response.text
    assert "synthetic-frame" not in response.text
    assert "observed_facts" in response.text
    assert metadata["result"] is None
    assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).content == jury_before
    assert json.loads(jury_before) == result
    assert client.get(f"/api/ui/jobs/{job_id}").status_code == 401


def test_ui_metadata_preserves_multiple_event_order_and_absence_phrases(api):
    client, store, config, _ = api
    selected = ["RK_NBK231_P19_2_RECEIPT_COUNT", "RK_NBK231_P20_PREVIOUS_CASH_POCKET"]
    job_id, _, _, catalog, _ = _publish_synthetic_ui_audit(client, store, config, selected)
    metadata = client.get(f"/api/ui/jobs/{job_id}", headers=AUTH).json()
    assert [event["rule"]["clause"] for event in metadata["events"]] == [catalog.get_rule(key).clause for key in selected]
    assert catalog.fact_description("CASH_COUNT_VISIBLE", "absent") in metadata["events"][0]["observed_facts"]
    assert catalog.fact_description("CASH_COUNT_VISIBLE", "present") not in metadata["events"][0]["observed_facts"]


def test_negative_ui_metadata_preserves_visual_scope_without_inventing_events(api):
    client, store, config, _ = api
    job_id, path, audit, catalog, _ = _publish_synthetic_ui_audit(client, store, config, [])
    audit["limitations"] = ["unclear_visual_context_irrelevant_to_refuted_rules"]
    path.write_text(json.dumps(audit), encoding="utf-8")
    metadata = client.get(f"/api/ui/jobs/{job_id}", headers=AUTH).json()
    assert metadata["events"] == []
    assert catalog.payload["approved_limitations_ru"][0] in metadata["limitations"]
    assert catalog.payload["approved_limitations_ru"][2] not in metadata["limitations"]
    assert ui_metadata._LIMIT_TEXT["unclear_visual_context_irrelevant_to_refuted_rules"] in metadata["limitations"]
    assert any("временным шагом" in text for text in metadata["limitations"])
    assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json()["verdict"] == "no_violation"


@pytest.mark.parametrize("invalid_field", [
    "job_id", "attempt", "schema_version", "analysis_version", "catalog_version", "phase",
    "publication_ready", "coverage", "event_count", "event_schema", "rule_id", "event_type",
    "fact_state", "duplicate_event", "verified_events",
])
def test_invalid_or_stale_private_audit_does_not_break_ui_metadata(api, invalid_field):
    client, store, config, _ = api
    job_id, path, audit, _, _ = _publish_synthetic_ui_audit(client, store, config)
    if invalid_field in {"job_id", "schema_version", "analysis_version", "catalog_version", "phase"}:
        audit[invalid_field] = "invalid-synthetic-value"
    elif invalid_field == "attempt":
        audit["attempt"] += 1
    elif invalid_field == "publication_ready":
        audit["publication_ready"] = "true"  # Strict booleans only.
    elif invalid_field == "coverage":
        audit["coverage"]["complete"] = False
    elif invalid_field == "event_count":
        audit["evidence"]["event_count"] += 1
    elif invalid_field == "event_schema":
        audit["events"][0]["anchor_frame_id"] = "not-in-evidence"
    elif invalid_field == "rule_id":
        audit["events"][0]["rule_id"] = "unreviewed-rule"
    elif invalid_field == "event_type":
        audit["events"][0]["event_type_id"] = "unreviewed-event-type"
    elif invalid_field == "fact_state":
        audit["events"][0]["observed_facts"][0]["state"] = "unclear"
    elif invalid_field == "duplicate_event":
        audit["events"].append(deepcopy(audit["events"][0]))
        audit["evidence"]["event_count"] = 2
    elif invalid_field == "verified_events":
        audit["verified_events"] = []
    if invalid_field not in {"verified_events"}:
        audit["verified_events"] = deepcopy(audit["events"])
    path.write_text(json.dumps(audit), encoding="utf-8")
    response = client.get(f"/api/ui/jobs/{job_id}", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["status"] == "done"
    assert response.json()["events"] == []
    assert response.json()["limitations"] == []
    assert "PRIVATE_" not in response.text
    assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json()["verdict"] == "violation"


def test_missing_corrupt_or_oversized_audit_falls_back_without_diagnostics(api, monkeypatch):
    client, store, config, _ = api
    job_id, path, audit, _, _ = _publish_synthetic_ui_audit(client, store, config)
    endpoint = f"/api/ui/jobs/{job_id}"
    path.unlink()
    assert client.get(endpoint, headers=AUTH).json()["events"] == []
    path.write_text("malformed private JSON")
    assert client.get(endpoint, headers=AUTH).json()["events"] == []
    path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(ui_metadata, "MAX_PRIVATE_AUDIT_BYTES", 128)
    response = client.get(endpoint, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["events"] == []
    assert "PRIVATE_" not in response.text


def test_processing_jobs_never_read_private_audit(api, monkeypatch):
    client, *_ = api
    job_id = upload(client).json()["job_id"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Only completed jobs may read the final private audit")

    monkeypatch.setattr(ui_metadata, "_read_audit", forbidden)
    response = client.get(f"/api/ui/jobs/{job_id}", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["events"] == []


def test_private_audit_cannot_resolve_outside_owned_root(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"private":"must not be read"}')
    assert ui_metadata._read_audit(outside, root) is None


@pytest.fixture
def cors_api(tmp_path):
    config = Settings(
        api_keys=("synthetic-test-key",), data_dir=tmp_path,
        cors_allowed_origins=("https://hackathon.wezeso.me",),
        public_api_base_url="https://forte-review-wd88a7vvt.gobrev.dev",
    )
    store = JobStore("sqlite:///:memory:", config.result_root)
    app = create_app(config, store, readiness=lambda: True)
    with TestClient(app) as client:
        yield client, store, config, app
    store.engine.dispose()


_UI_ORIGIN = "https://hackathon.wezeso.me"


def test_approved_cors_preflight_needs_no_key_and_accepts_only_scoped_methods_headers(cors_api):
    client, *_ = cors_api
    response = client.options("/api/v1/analyze", headers={
        "Origin": _UI_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key, Content-Type",
    })
    assert response.status_code == 204
    assert response.headers["access-control-allow-origin"] == _UI_ORIGIN
    assert set(response.headers["access-control-allow-methods"].replace(" ", "").split(",")) == {"GET", "POST"}
    assert set(response.headers["access-control-allow-headers"].lower().replace(" ", "").split(",")) == {"x-api-key", "content-type", "range"}
    assert "access-control-allow-credentials" not in response.headers
    assert "Origin" in response.headers["vary"]
    video_preflight = client.options("/api/ui/jobs/synthetic/video", headers={
        "Origin": _UI_ORIGIN, "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Range, X-API-Key",
    })
    assert video_preflight.status_code == 204


@pytest.mark.parametrize("change", [
    {"Origin": "https://unapproved.example"},
    {"Origin": "null"},
    {"Access-Control-Request-Method": "DELETE"},
    {"Access-Control-Request-Headers": "X-API-Key, Authorization"},
    {"Access-Control-Request-Headers": "X-API-Key, Cookie"},
])
def test_forbidden_cors_preflights_are_rejected(cors_api, change):
    client, *_ = cors_api
    headers = {"Origin": _UI_ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-API-Key"}
    headers.update(change)
    response = client.options("/api/v1/analyze", headers=headers)
    assert response.status_code == 403
    assert "access-control-allow-methods" not in response.headers
    if change.get("Origin"):
        assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("origin,key,expected_status", [
    (_UI_ORIGIN, None, 401),
    (_UI_ORIGIN, "wrong", 401),
    ("https://unapproved.example", "synthetic-test-key", 403),
])
def test_cors_actual_uploads_authenticate_or_reject_before_reading_body(cors_api, origin, key, expected_status):
    _, _, _, app = cors_api
    messages = []
    reads = []

    async def run():
        async def receive():
            reads.append(True)
            raise AssertionError("Rejected CORS/auth requests must never read upload bodies")

        async def send(message):
            messages.append(message)

        headers = [(b"origin", origin.encode())]
        if key is not None:
            headers.append((b"x-api-key", key.encode()))
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "https", "path": "/api/v1/analyze",
            "raw_path": b"/api/v1/analyze", "query_string": b"", "headers": headers,
            "server": ("api.example", 443), "client": ("test", 123),
        }
        await app(scope, receive, send)

    asyncio.run(run())
    assert messages[0]["status"] == expected_status
    assert not reads
    response_headers = dict(messages[0]["headers"])
    if origin == _UI_ORIGIN:
        assert response_headers[b"access-control-allow-origin"] == _UI_ORIGIN.encode()
    else:
        assert b"access-control-allow-origin" not in response_headers


def test_authorized_cors_upload_poll_video_and_readiness_preserve_contract(cors_api):
    client, _, _, _ = cors_api
    headers = {**AUTH, "Origin": _UI_ORIGIN}
    response = client.post("/api/v1/analyze", headers=headers, files={"file": ("synthetic.mov", b"synthetic transport input")})
    assert response.status_code == 202
    assert set(response.json()) == {"job_id"}
    assert response.headers["access-control-allow-origin"] == _UI_ORIGIN
    job_id = response.json()["job_id"]
    poll = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    assert poll.json() == {"job_id": job_id, "status": "processing"}
    assert poll.headers["access-control-allow-origin"] == _UI_ORIGIN
    video = client.get(f"/api/ui/jobs/{job_id}/video", headers={**headers, "Range": "bytes=0-8"})
    assert video.status_code == 206
    assert video.content == b"synthetic"
    assert video.headers["access-control-allow-origin"] == _UI_ORIGIN
    assert "Content-Range" in video.headers["access-control-expose-headers"]
    ready = client.get("/readyz", headers={"Origin": _UI_ORIGIN})
    assert ready.status_code == 200
    assert ready.headers["access-control-allow-origin"] == _UI_ORIGIN


def test_runtime_bootstrap_exposes_only_operator_selected_public_origin(cors_api):
    client, *_ = cors_api
    response = client.get("/runtime-config.json")
    assert response.status_code == 200
    assert response.json() == {"apiBaseUrl": "https://forte-review-wd88a7vvt.gobrev.dev"}
    assert response.headers["cache-control"] == "no-store"
    assert "synthetic-test-key" not in response.text


def test_cors_is_disabled_by_default_and_same_origin_runtime_is_empty(api):
    client, *_ = api
    response = client.options("/api/v1/analyze", headers={"Origin": _UI_ORIGIN, "Access-Control-Request-Method": "POST"})
    assert response.status_code == 401
    assert "access-control-allow-origin" not in response.headers
    assert client.get("/runtime-config.json").json() == {"apiBaseUrl": ""}


@pytest.mark.parametrize("origin", ["*", "null", "http://hackathon.wezeso.me", "https://user:password@example.com", "https://example.com/path", "https://example.com?token=value", "https://example.com#fragment"])
def test_cors_configuration_rejects_wildcards_non_https_and_non_origin_urls(origin):
    with pytest.raises(ValueError):
        Settings(cors_allowed_origins=(origin,))
    with pytest.raises(ValueError):
        Settings(public_api_base_url=origin)


def _review_payload(snapshot, *, index=0, decision="confirm", revision=0, name="Synthetic reviewer", comment="Synthetic review"):
    return {
        "result_sha256": snapshot["result_sha256"],
        "evidence_sha256": snapshot["violations"][index]["evidence_sha256"],
        "decision": decision, "claimed_reviewer_name": name, "comment": comment,
        "expected_revision": revision,
    }


def test_human_review_confirm_reject_history_and_immutable_model_result(api):
    client, store, config, _ = api
    job_id, _, _, _, _ = _publish_synthetic_ui_audit(client, store, config)
    endpoint = f"/api/ui/jobs/{job_id}/reviews"
    before = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).content
    original_job = store.get_job(job_id)
    response = client.get(endpoint, headers=AUTH)
    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["result_sha256"] == hashlib.sha256(before).hexdigest()
    assert snapshot["source_sha256"] == original_job.input_sha256
    assert snapshot["reviewer_identity_verified"] is False
    initial = snapshot["violations"][0]["review"]
    assert initial["state"] == "pending" and initial["revision"] == 0
    assert initial["claimed_reviewer_name"] is None and initial["reviewed_at"] is None
    expected_evidence = hashlib.sha256(json.dumps(json.loads(before)["violations"][0]["evidence_frames"], ensure_ascii=False,
                                                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert snapshot["violations"][0]["evidence_sha256"] == expected_evidence
    confirmed = client.post(endpoint + "/0", headers=AUTH, json=_review_payload(snapshot)).json()
    assert confirmed["state"] == "confirmed" and confirmed["revision"] == 1
    assert confirmed["claimed_reviewer_name"] == "Synthetic reviewer"
    assert confirmed["reviewer_identity_verified"] is False
    assert datetime.fromisoformat(confirmed["reviewed_at"]).utcoffset() == timedelta(0)
    rejected = client.post(endpoint + "/0", headers=AUTH, json=_review_payload(snapshot, decision="reject", revision=1, comment="Correction"))
    assert rejected.status_code == 200
    assert rejected.json()["state"] == "rejected" and rejected.json()["revision"] == 2
    assert client.get(endpoint, headers=AUTH).json()["violations"][0]["review"] == rejected.json()
    page = client.get(endpoint + "/0/history", headers=AUTH, params={"result_sha256": snapshot["result_sha256"], "limit": 1}).json()
    assert page["has_more"] is True and page["next_after_revision"] == 1
    assert [item["state"] for item in page["entries"]] == ["confirmed"]
    next_page = client.get(endpoint + "/0/history", headers=AUTH, params={"result_sha256": snapshot["result_sha256"], "after_revision": 1}).json()
    assert next_page["has_more"] is False and next_page["next_after_revision"] == 2
    assert [item["state"] for item in next_page["entries"]] == ["rejected"]
    assert client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).content == before
    assert store.get_job(job_id) == original_job


def test_concurrent_reviewers_get_one_revision_and_an_explicit_conflict(api):
    client, store, config, _ = api
    job_id, *_ = _publish_synthetic_ui_audit(client, store, config)
    endpoint = f"/api/ui/jobs/{job_id}/reviews"
    snapshot = client.get(endpoint, headers=AUTH).json()
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda name: client.post(endpoint + "/0", headers=AUTH,
                                                           json=_review_payload(snapshot, name=name)), ["Reviewer A", "Reviewer B"]))
    assert sorted(response.status_code for response in responses) == [200, 409]
    conflict = next(response for response in responses if response.status_code == 409).json()
    assert conflict == {"detail": "review_conflict", "current_revision": 1, "current_result_sha256": snapshot["result_sha256"]}
    history = client.get(endpoint + "/0/history", headers=AUTH, params={"result_sha256": snapshot["result_sha256"]}).json()
    assert len(history["entries"]) == 1


@pytest.mark.parametrize("field", ["result_sha256", "evidence_sha256", "violation_index"])
def test_review_source_or_evidence_mismatch_cannot_create_an_assessment(api, field):
    client, store, config, _ = api
    job_id, *_ = _publish_synthetic_ui_audit(client, store, config)
    endpoint = f"/api/ui/jobs/{job_id}/reviews"
    snapshot = client.get(endpoint, headers=AUTH).json()
    payload = _review_payload(snapshot)
    index = 0
    if field == "violation_index":
        index = 1
    else:
        payload[field] = "f" * 64
    response = client.post(endpoint + f"/{index}", headers=AUTH, json=payload)
    assert response.status_code == 409
    assert client.get(endpoint, headers=AUTH).json()["violations"][0]["review"]["revision"] == 0


@pytest.mark.parametrize("change", [
    {"claimed_reviewer_name": " "}, {"claimed_reviewer_name": "a" * 121},
    {"claimed_reviewer_name": "invalid\x7fname"}, {"comment": "a" * 4001},
    {"decision": "pending"}, {"expected_revision": True}, {"expected_revision": -1},
    {"unexpected_private_field": "do not echo"},
])
def test_review_payload_validation_is_bounded_and_does_not_echo_input(api, change):
    client, store, config, _ = api
    job_id, *_ = _publish_synthetic_ui_audit(client, store, config)
    endpoint = f"/api/ui/jobs/{job_id}/reviews"
    snapshot = client.get(endpoint, headers=AUTH).json()
    payload = {**_review_payload(snapshot), **change}
    response = client.post(endpoint + "/0", headers=AUTH, json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid review payload"}
    assert client.get(endpoint, headers=AUTH).json()["violations"][0]["review"]["revision"] == 0


def test_review_body_limit_applies_to_streamed_requests_and_unicode_comments_fit(api):
    client, store, config, _ = api
    job_id, *_ = _publish_synthetic_ui_audit(client, store, config)
    endpoint = f"/api/ui/jobs/{job_id}/reviews"
    snapshot = client.get(endpoint, headers=AUTH).json()
    oversized = iter([b" " * 16384, b" " * 16384, b" "])
    response = client.post(endpoint + "/0", headers={**AUTH, "Content-Type": "application/json"}, content=oversized)
    assert response.status_code == 413
    good = client.post(endpoint + "/0", headers=AUTH, json=_review_payload(snapshot, comment="🙂" * 4000))
    assert good.status_code == 200
    assert len(good.json()["comment"]) == 4000


@pytest.mark.parametrize("authorized,length,expected", [(False, MAX_REVIEW_BODY_BYTES + 1, 401), (True, MAX_REVIEW_BODY_BYTES + 1, 413)])
def test_review_auth_and_declared_size_are_checked_before_body_receive(api, authorized, length, expected):
    client, store, config, app = api
    job_id, *_ = _publish_synthetic_ui_audit(client, store, config)
    messages, reads = [], []

    async def run():
        async def receive():
            reads.append(True)
            raise AssertionError("A rejected review must not consume its body")

        async def send(message):
            messages.append(message)

        path = f"/api/ui/jobs/{job_id}/reviews/0"
        headers = [(b"content-type", b"application/json"), (b"content-length", str(length).encode())]
        if authorized:
            headers.append((b"x-api-key", b"synthetic-test-key"))
        await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                   "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "headers": headers,
                   "server": ("test", 80), "client": ("test", 123)}, receive, send)

    asyncio.run(run())
    assert messages[0]["status"] == expected
    assert reads == []


def test_review_does_not_allow_processing_or_negative_jobs_and_requires_read_auth(api):
    client, store, config, _ = api
    negative, *_ = _publish_synthetic_ui_audit(client, store, config, [])
    assert client.get(f"/api/ui/jobs/{negative}/reviews", headers=AUTH).status_code == 409
    processing = upload(client).json()["job_id"]
    assert client.get(f"/api/ui/jobs/{processing}/reviews", headers=AUTH).status_code == 409
    assert client.get(f"/api/ui/jobs/{negative}/reviews").status_code == 401
    assert client.get("/api/ui/jobs/unknown/reviews", headers=AUTH).status_code == 404


def test_review_rejects_duplicate_json_members_and_history_query_abuse(api):
    client, store, config, _ = api
    job_id, *_ = _publish_synthetic_ui_audit(client, store, config)
    endpoint = f"/api/ui/jobs/{job_id}/reviews"
    snapshot = client.get(endpoint, headers=AUTH).json()
    body = json.dumps(_review_payload(snapshot))[:-1] + ',"expected_revision":1}'
    assert client.post(endpoint + "/0", headers={**AUTH, "Content-Type": "application/json"}, content=body).status_code == 422
    for query in ({"limit": 51}, {"after_revision": -1}):
        response = client.get(endpoint + "/0/history", headers=AUTH, params={"result_sha256": snapshot["result_sha256"], **query})
        assert response.status_code == 422
    assert client.get(endpoint, headers=AUTH).json()["violations"][0]["review"]["revision"] == 0


def test_review_source_integrity_failure_is_not_recorded(api):
    client, store, config, _ = api
    job_id, *_ = _publish_synthetic_ui_audit(client, store, config)
    record = store.get_job(job_id)
    result_path = Path(record.result_path)
    original = result_path.read_bytes()
    result_path.write_bytes(original + b" ")
    response = client.get(f"/api/ui/jobs/{job_id}/reviews", headers=AUTH)
    assert response.status_code == 409
    assert response.json() == {"detail": "review_not_allowed"}
    result_path.write_bytes(original)
    assert client.get(f"/api/ui/jobs/{job_id}/reviews", headers=AUTH).json()["violations"][0]["review"]["revision"] == 0


def test_human_review_routes_use_the_existing_explicit_cors_policy(cors_api):
    client, store, config, _ = cors_api
    job_id, *_ = _publish_synthetic_ui_audit(client, store, config)
    endpoint = f"/api/ui/jobs/{job_id}/reviews"
    headers = {**AUTH, "Origin": _UI_ORIGIN}
    snapshot = client.get(endpoint, headers=headers).json()
    preflight = client.options(endpoint + "/0", headers={"Origin": _UI_ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "Content-Type, X-API-Key"})
    assert preflight.status_code == 204
    assert client.post(endpoint + "/0", headers=headers, json=_review_payload(snapshot)).status_code == 200
    stale = client.post(endpoint + "/0", headers=headers, json=_review_payload(snapshot))
    assert stale.status_code == 409 and stale.json()["current_revision"] == 1
    assert stale.headers["access-control-allow-origin"] == _UI_ORIGIN
