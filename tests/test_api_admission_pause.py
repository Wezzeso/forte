"""Maintenance closes new uploads before body reads and preserves existing data."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from forte.api.config import Settings
from forte.api.main import create_app
from forte.jobs import JobStore


@pytest.mark.parametrize("headers,status", [([], 401), ([(b"x-api-key", b"wrong")], 401),
    ([(b"x-api-key", b"synthetic-key")], 503)])
def test_paused_upload_authenticates_before_rejecting_without_reading_body(tmp_path, headers, status):
    config = Settings(api_keys=("synthetic-key",), data_dir=tmp_path, admission_paused=True)
    store = JobStore("sqlite:///:memory:", config.result_root)
    app = create_app(config, store, readiness=lambda: True)
    sent, reads = [], []
    async def run():
        async def receive():
            reads.append(True)
            raise AssertionError("Paused upload body must never be read")
        async def send(message):
            sent.append(message)
        await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/api/v1/analyze", "raw_path": b"/api/v1/analyze",
            "query_string": b"", "server": ("test", 80), "client": ("test", 123),
            "headers": [*headers, (b"content-length", b"999999999999")]}, receive, send)
    asyncio.run(run())
    assert sent[0]["status"] == status
    assert reads == []
    assert not config.upload_root.exists()
    if status == 503:
        assert dict(sent[0]["headers"])[b"retry-after"] == b"60"
    store.engine.dispose()


def test_paused_api_preserves_completed_results_reviews_config_and_health(tmp_path, monkeypatch):
    config = Settings(api_keys=("synthetic-key",), data_dir=tmp_path)
    store = JobStore("sqlite:///:memory:", config.result_root)
    auth = {"X-API-Key": "synthetic-key"}
    with TestClient(create_app(config, store, readiness=lambda: True)) as client:
        response = client.post("/api/v1/analyze", headers=auth,
            files={"file": ("generated.mov", b"synthetic undeccoded bytes", "video/quicktime")})
        job_id = response.json()["job_id"]
        claimed = store.claim("synthetic-worker")
        result = json.loads((Path(__file__).resolve().parents[1] / "contracts/fixtures/positive.json").read_text(encoding="utf-8"))
        result["job_id"] = job_id
        store.publish(job_id, "synthetic-worker", claimed.attempt, result)
        original = client.get(f"/api/v1/jobs/{job_id}", headers=auth).content
    def forbidden():
        pytest.fail("Paused readiness must not query model/dependency readiness")
    monkeypatch.setattr("forte.api.main._model_alive", forbidden)
    with TestClient(create_app(replace(config, admission_paused=True), store, readiness=forbidden)) as client:
        assert client.get(f"/api/v1/jobs/{job_id}", headers=auth).content == original
        assert client.get(f"/api/ui/jobs/{job_id}", headers=auth).status_code == 200
        assert client.get(f"/api/ui/jobs/{job_id}/reviews", headers=auth).status_code == 200
        assert client.get("/api/ui/config", headers=auth).status_code == 200
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json() == {"status": "not_ready", "components": {"admission": False}}
        assert ready.headers["cache-control"] == "no-store"
        paused = client.post("/api/v1/analyze", headers=auth,
            files={"file": ("second.mov", b"must not persist", "video/quicktime")})
        assert paused.status_code == 503 and paused.headers["retry-after"] == "60"
        assert len(list(config.upload_root.glob("*.mov"))) == 1
    store.engine.dispose()


def test_admission_pause_defaults_false_and_environment_is_strict(tmp_path, monkeypatch):
    monkeypatch.setenv("FORTE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FORTE_ADMISSION_PAUSED", raising=False)
    assert Settings.from_env().admission_paused is False
    monkeypatch.setenv("FORTE_ADMISSION_PAUSED", "true")
    assert Settings.from_env().admission_paused is True
    monkeypatch.setenv("FORTE_ADMISSION_PAUSED", "FALSE")
    assert Settings.from_env().admission_paused is False
    for value in ("1", "yes", "", "pause"):
        monkeypatch.setenv("FORTE_ADMISSION_PAUSED", value)
        with pytest.raises(ValueError, match="must be true or false"):
            Settings.from_env()
    with pytest.raises(ValueError, match="must be a boolean"):
        Settings(admission_paused=1)
