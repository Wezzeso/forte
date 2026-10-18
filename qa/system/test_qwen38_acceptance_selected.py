"""Generated Docker metadata only; no Docker, service, upload or model calls."""
import json
from types import SimpleNamespace

from deploy import qwen38_acceptance_selected as selected


def test_metadata_helper_uses_worker_image_without_secrets_network_or_gpu(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ops"
    root.mkdir()
    arguments = root / "actual-arguments.json"
    arguments.write_text(json.dumps(["--arm", "qwen38_tuned"]), encoding="utf-8")
    worker = {"Image": "sha256:" + "b" * 64, "Config": {"Env": [
        "FORTE_MAX_CONCURRENT_JOBS=1", "FORTE_STAGED_SCREENING=false",
        "FORTE_API_KEY=generated-secret", "FORTE_MODEL_API_KEY=generated-secret",
        "FORTE_DATABASE_URL=generated-secret", "CUDA_VISIBLE_DEVICES=0"]}, "Mounts": [
        {"Type": "bind", "Source": "/actual/configs", "Destination": "/run/configs"},
        {"Type": "bind", "Source": "/actual/secret", "Destination": "/run/secrets/api_keys"},
        {"Type": "bind", "Source": "/actual/weights", "Destination": "/models"}]}
    inspected = []
    def inspect(name):
        inspected.append(name)
        return worker
    monkeypatch.setattr(selected.base, "inspect", inspect)
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "actual_context_rebound_not_authorized",
            "model_calls_made": 0, "generated_private_field": "generated-secret"}).encode(), stderr=b"generated-secret")
    monkeypatch.setattr(selected.subprocess, "run", launch)
    assert selected.main(["prepare", "--ops-root", str(root), "--worker-container", "selected-worker",
                          "--arguments-file", str(arguments)]) == 0
    assert inspected == ["selected-worker"]
    command = commands[0]
    assert command[command.index("--network") + 1] == "none"
    assert "--gpus" not in command
    assert worker["Image"] in command
    assert command[command.index("--cpu-image-sha256") + 1] == "b" * 64
    assert "FORTE_MAX_CONCURRENT_JOBS=1" in command and "FORTE_STAGED_SCREENING=false" in command
    joined = " ".join(command)
    assert "generated-secret" not in joined and "/actual/weights" not in joined and "/actual/secret" not in joined
    assert "/actual/configs:/run/configs:ro" in command
    assert "generated-secret" not in capsys.readouterr().out


def test_execution_uses_selected_worker_for_image_and_key_mount_provider(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(selected.base, "run", lambda args: captured.append(args) or 0)
    assert selected.main(["execute", "--ops-root", str(tmp_path), "--worker-container", "actual-new-worker",
        "--bound-root", "benchmarks/private/new/bound", "--run-root", "benchmarks/private/new/run",
        "--release", "benchmarks/private/new/release.json", "--reservation", "benchmarks/private/new/reservation.json"]) == 0
    assert captured[0].api_container == captured[0].worker_container == "actual-new-worker"
