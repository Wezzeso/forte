"""Fail-closed snapshot/host checks; no real GPU, network or Docker mutations."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import fast_host_preflight as preflight


REVISION = "a" * 40  # Synthetic fixture identity, never a suggested production pin.
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    files = {
        "config.json": {"model_type": "qwen3_vl_moe"},
        "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
        "tokenizer.json": {"version": "1.0"},
        "preprocessor_config.json": {"image_processor_type": "Qwen3VLImageProcessor"},
        "model.safetensors.index.json": {"weight_map": {"layer.0": "model-00001.safetensors"}},
        preflight.RECEIPT_NAME: {
            "schema_version": "forte.fast-model-snapshot.v1",
            "repo_id": preflight.DEFAULT_REPO, "revision": REVISION,
        },
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    (root / "model-00001.safetensors").write_bytes(b"synthetic weight bytes")
    return root


@pytest.mark.parametrize("revision", ["", "main", "latest", "a" * 39, "g" * 40])
def test_snapshot_requires_immutable_model_pin(snapshot, revision):
    with pytest.raises(preflight.PreflightError, match="immutable_model_revision_required"):
        preflight.check_snapshot(snapshot, revision)


def test_complete_pinned_snapshot_is_checked(snapshot):
    result = preflight.check_snapshot(snapshot, REVISION)
    assert result["revision"] == REVISION
    assert result["weight_shards"] == 1
    assert result["weight_bytes"] == len(b"synthetic weight bytes")
    assert result["files_checked"] == 7


def test_snapshot_cannot_silently_reuse_other_revision(snapshot):
    with pytest.raises(preflight.PreflightError, match="snapshot_revision_mismatch"):
        preflight.check_snapshot(snapshot, "b" * 40)


def test_missing_weight_shard_fails_before_host_checks(snapshot):
    (snapshot / "model-00001.safetensors").unlink()
    with pytest.raises(preflight.PreflightError, match="snapshot_file_missing_or_unreadable"):
        preflight.check_snapshot(snapshot, REVISION)


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/tmp/outside.safetensors", "C:\\weights.safetensors"])
def test_weight_index_cannot_read_outside_snapshot(snapshot, filename):
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": filename}}))
    with pytest.raises(preflight.PreflightError, match="snapshot_unsafe_file_path"):
        preflight.check_snapshot(snapshot, REVISION)


def test_model_file_must_be_readable(snapshot, monkeypatch):
    original = Path.open

    def open_without_weight_permission(path, *args, **kwargs):
        if path.name.endswith(".safetensors"):
            raise PermissionError("private diagnostic must not escape")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_without_weight_permission)
    with pytest.raises(preflight.PreflightError, match="^snapshot_file_missing_or_unreadable$"):
        preflight.check_snapshot(snapshot, REVISION)


def test_remote_code_model_is_rejected(snapshot):
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_vl_moe", "auto_map": {"AutoModel": "remote.Code"}}))
    with pytest.raises(preflight.PreflightError, match="snapshot_remote_code_not_allowed"):
        preflight.check_snapshot(snapshot, REVISION)


def test_host_checks_actual_isolated_container_and_not_just_nvidia_smi(snapshot, monkeypatch):
    calls = []

    def run(command, code, timeout=30):
        calls.append(command)
        if command[0] == "nvidia-smi":
            return "NVIDIA A100 80GB PCIe, 81920, 80000"
        if command[1:3] == ["compose", "version"]:
            return "5.3.1"
        if command[1:3] == ["image", "inspect"]:
            return "sha256:" + "b" * 64
        if command[1] == "run":
            return json.dumps({"cuda": True, "bf16": True, "model_readable": True})
        return "29.6.2"

    monkeypatch.setattr(preflight, "_run", run)
    result = preflight.check_host(snapshot, "configured-runtime:tag", "0")
    assert result["cuda"] is True
    assert result["container_image_id"] == "sha256:" + "b" * 64
    command = calls[-1]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--gpus") + 1] == "device=0"
    assert "--pull=never" in command and "--read-only" in command
    assert f"type=bind,src={snapshot.resolve()},dst=/models/vlm,readonly" in command
    assert "configured-runtime:tag" in command
    assert "torch.ones(1, device='cuda'" in command[-1]


def test_busy_or_small_gpu_fails_before_launching_container(snapshot, monkeypatch):
    calls = []

    def run(command, code, timeout=30):
        calls.append(command)
        return "NVIDIA A100 80GB PCIe, 81920, 7000"

    monkeypatch.setattr(preflight, "_run", run)
    with pytest.raises(preflight.PreflightError, match="host_gpu_memory_insufficient"):
        preflight.check_host(snapshot, "image", "0")
    assert len(calls) == 1


def test_multiple_gpu_selector_is_rejected(snapshot):
    with pytest.raises(preflight.PreflightError, match="single_gpu_id_required"):
        preflight.check_host(snapshot, "image", "0,1")


def test_failed_container_diagnostics_do_not_leak(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "secret stdout", "secret stderr"))
    with pytest.raises(preflight.PreflightError, match="^safe_static_code$"):
        preflight._run(["docker", "info"], "safe_static_code")


def test_snapshot_only_success_does_not_claim_host_is_ready(snapshot, capsys, monkeypatch):
    monkeypatch.setattr(preflight, "check_host", lambda *a, **k: pytest.fail("host should not run"))
    assert preflight.main(["--snapshot-only", "--model-dir", str(snapshot), "--model-revision", REVISION]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is False and result["snapshot_valid"] is True
    assert result["host"] is None


def test_fast_compose_enables_prefix_cache_and_single_worker():
    source = (ROOT / "deploy/compose.fast.yaml").read_text()
    assert '"--enable-prefix-caching"' in source
    assert '"--max-num-seqs", "64"' in source
    assert 'FORTE_MAX_CONCURRENT_JOBS: "4"' in source
    assert "container_name: forte-fast-worker" in source
    assert "replicas: 1" in source
    assert "read_only: true" in source
    assert "networks: [private]" in source
    assert "ipc: host" not in source


@pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="Linux bootstrap execution")
@pytest.mark.parametrize("ready_exit,expected", [(0, 0), (1, 1)])
def test_bootstrap_readiness_controls_exit_and_preserves_bind(tmp_path, ready_exit, expected):
    repo = tmp_path / "repo"
    (repo / "deploy").mkdir(parents=True)
    (repo / "tools").mkdir()
    shutil.copy(ROOT / "deploy/bootstrap_fast_host.sh", repo / "deploy")
    (repo / "deploy/.env.example").write_text(f"FORTE_FAST_MODEL_REVISION={REVISION}\nFORTE_BIND_HOST=127.0.0.1\n")
    (repo / "deploy/prepare_secrets.py").write_text("print('secrets preserved')\n")
    (repo / "tools/fast_host_preflight.py").write_text("print('synthetic preflight')\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f'#!/bin/bash\nif [[ "$*" == *"exec -T api"* ]]; then exit {ready_exit}; fi\nexit 0\n')
    gpu = bin_dir / "nvidia-smi"
    gpu.write_text("#!/bin/bash\necho 'Synthetic GPU'\n")
    for script in (docker, gpu):
        script.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("FORTE_")}
    env.update(PATH=str(bin_dir) + os.pathsep + env["PATH"], FORTE_FAST_MODEL_DIR=str(model),
               FORTE_FAST_READY_TIMEOUT_SECONDS="1")
    result = subprocess.run(["bash", str(repo / "deploy/bootstrap_fast_host.sh")], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == expected, result.stdout + result.stderr
    assert "FORTE_BIND_HOST=127.0.0.1" in (repo / "deploy/.env").read_text()
    if expected:
        assert "deployment is not ready" in result.stderr
        assert "6/6 Ready" not in result.stdout
