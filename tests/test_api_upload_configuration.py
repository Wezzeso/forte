"""UI constraints describe the actual authenticated upload configuration."""

from fastapi.testclient import TestClient
import pytest

from forte.api.config import Settings
from forte.api.main import create_app
from forte.jobs import JobStore


@pytest.fixture
def upload_configuration_api(tmp_path):
    config = Settings(api_keys=("synthetic-ui-key",), data_dir=tmp_path, max_upload_bytes=1_234_567,
        cors_allowed_origins=("https://hackathon.wezeso.me",))
    store = JobStore("sqlite:///:memory:", config.result_root)
    with TestClient(create_app(config, store, readiness=lambda: True)) as client:
        yield client
    store.engine.dispose()


def test_ui_upload_constraints_report_actual_server_bytes_and_mov(upload_configuration_api):
    response = upload_configuration_api.get("/api/ui/config", headers={"X-API-Key": "synthetic-ui-key"})
    assert response.status_code == 200
    assert response.json() == {"max_upload_bytes": 1_234_567, "accepted_suffixes": [".mov"]}
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_ui_upload_constraints_require_api_key(upload_configuration_api, headers):
    response = upload_configuration_api.get("/api/ui/config", headers=headers)
    assert response.status_code == 401
    assert "max_upload_bytes" not in response.json()


def test_ui_upload_constraints_keep_exact_cors_boundary(upload_configuration_api):
    allowed = upload_configuration_api.get("/api/ui/config", headers={
        "X-API-Key": "synthetic-ui-key", "Origin": "https://hackathon.wezeso.me"})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://hackathon.wezeso.me"
    denied = upload_configuration_api.get("/api/ui/config", headers={
        "X-API-Key": "synthetic-ui-key", "Origin": "https://hackathon.wezeso.me.attacker.example"})
    assert denied.status_code == 403
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_upload_byte_cap_must_be_positive_integer(limit):
    with pytest.raises(ValueError, match="Upload limits"):
        Settings(max_upload_bytes=limit)
