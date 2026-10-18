"""Mock Docker/startup-only audit; no daemon, process, GPU or model is touched."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy import qwen38_startup_ops as ops
from serving.qwen38_untuned import expected_untuned_identity
from test_qwen38_untuned import config_fixture


def test_expected_startup_identity_equals_frozen_untuned_contract():
    config = config_fixture()
    assert ops.expected_identity(config) == expected_untuned_identity(config)


def trained_config_fixture():
    config = json.loads(Path("configs/serving/qwen38/candidate.runtime.example.json").read_text())
    config.update(adapter_sha256="a" * 64, adapter_config_sha256="b" * 64,
                  runtime_image_sha256="c" * 64,
                  runtime_profile_sha256="d" * 64)
    config["training_run"]["sha256"] = "e" * 64
    config["source_archive"]["sha256"] = "f" * 64
    config["package_versions"] = {name: "synthetic-version" for name in config["package_versions"]}
    config["package_versions"]["xgrammar"] = config["structured_decoding"]["version"]
    return config


def test_trained_startup_uses_exact_fifteen_field_serving_identity(tmp_path):
    from serving.qwen38_identity import load_expected_runtime_identity
    config = trained_config_fixture()
    path = tmp_path / "generated-config.json"
    path.write_text(json.dumps(config))
    expected = load_expected_runtime_identity(path)
    assert len(expected) == 15 and ops.expected_identity(config) == expected
    assert expected["adapter_sha256"] == config["adapter_sha256"]
    assert expected["training_run_sha256"] == config["training_run"]["sha256"]


@pytest.mark.parametrize("line", ["PRIVATE_SENTINEL/path/video.jpg", "{invalid", '["PRIVATE_SENTINEL"]',
    '{"event":"unrecognized","description":"PRIVATE_SENTINEL"}'])
def test_runtime_text_and_unknown_events_are_not_logged(line):
    assert ops.safe_line(line) is None


def test_allowed_events_and_exception_classes_drop_text_paths_and_descriptions():
    assert ops.safe_line('CandidateError: PRIVATE_SENTINEL /private/frame.jpg') == {
        "event": "runtime_exception_class", "error_type": "CandidateError"}
    result = ops.safe_line(json.dumps({"event": "untuned_start_failed", "reason_code": "static_failure",
        "error_type": "CandidateError", "description": "PRIVATE_SENTINEL", "path": "/private/frame.jpg"}))
    assert result == {"event": "untuned_start_failed", "reason_code": "static_failure", "error_type": "CandidateError"}


def test_private_outputs_are_exclusive(tmp_path):
    target = tmp_path / "receipt.json"
    ops.private_write(target, {"status": "first"})
    with pytest.raises(FileExistsError):
        ops.private_write(target, {"status": "overwrite"})
    assert json.loads(target.read_text()) == {"status": "first"}


@pytest.mark.parametrize("ports", [["-p", "8000:8000"], ["--publish", "8000:8000"],
    ["--publish=8000:8000"], ["-p8000:8000"], ["--publish-all"]])
def test_preflight_rejects_all_public_port_spellings(tmp_path, monkeypatch, ports):
    config = tmp_path / "config.json"
    config.write_text("{}")
    plan = {"remote_config_path": str(config), "config_file_sha256": ops.digest(config),
        "config_sha256": ops.canonical({}), "image_sha256": "a" * 64,
        "network_preflight": {"network_name": "generated_private", "network_id": "generated-net", "alias": "model-3"},
        "gpu_container_create_argv": ["docker", "create", *ports, "--network", "generated_private", "--network-alias", "model-3"]}
    def command(args):
        if args[1:3] == ["image", "inspect"]:
            return json.dumps([{"Id": "sha256:" + "a" * 64}])
        return json.dumps([{"Internal": True, "Id": "generated-net", "Name": "generated_private", "Containers": {}}])
    monkeypatch.setattr(ops, "command", command)
    with pytest.raises(ops.OpsError, match="network_or_ports_invalid"):
        ops.preflight(plan)


def startup_fixture(tmp_path, monkeypatch, *, bad_identity=False, trained=False):
    config = trained_config_fixture() if trained else config_fixture()
    ready = {**ops.expected_identity(config), "ready": True, "schema_smoke": {"metrics": {"attempts": 2}},
        "warmup": [{"generated": True}] * 4,
        "structured_decoding": {"initialization_seconds": .2, "public_schema_warmup": {"fact_count": 19}}}
    if bad_identity:
        ready["model_revision"] = "0" * 40
    plan = {"private_cache_directory": {"path": "synthetic-cache"},
        "gpu_container_create_argv": ["docker", "create", "--name", "synthetic-only"],
        "network_preflight": {"alias": "model-3"}, "source_archive_sha256": "b" * 64,
        "image_sha256": "e" * 64, "config_file_sha256": "f" * 64,
        "config_sha256": config["config_sha256"] if "config_sha256" in config else ops.canonical(config),
        "runtime_profile_sha256": config["runtime_profile_sha256"],
        "readiness_probe_argv": ["MOCK-READINESS-ONLY"], "failure_stop_argv": ["MOCK-STOP-ONLY"],
        "failure_kill_only_if_still_running_argv": ["MOCK-KILL-ONLY"]}
    controls, stopped = [], {"value": False}
    class Cache:
        def __init__(self, relative=""): self.relative = relative
        def __truediv__(self, relative): return Cache(relative)
        def exists(self): return False
        def mkdir(self, **kwargs): controls.append(["MOCK-CACHE-MKDIR", self.relative, kwargs])
        def stat(self): return SimpleNamespace(st_uid=1000)
    original_path = Path
    monkeypatch.setattr(ops, "Path", lambda value: Cache() if value == "synthetic-cache" else original_path(value))
    monkeypatch.setattr(ops, "preflight", lambda plan, activation: (config, {"Name": "generated_private", "Id": "net"}))
    def command(args):
        controls.append(args)
        return json.dumps(ready) if args[:2] == ["docker", "exec"] else ""
    monkeypatch.setattr(ops, "command", command)
    monkeypatch.setattr(ops, "inspect_container", lambda name: {"State": {"Running": not stopped["value"]}})
    def run(args, **kwargs):
        controls.append(args)
        if args == plan["failure_stop_argv"]:
            stopped["value"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(ops.subprocess, "run", run)
    class Attach:
        stdout = StringIO('CandidateError: PRIVATE_SENTINEL\n{"event":"untuned_diagnostic_ready","path":"PRIVATE_SENTINEL"}\n')
        def poll(self): return None
        def terminate(self): controls.append(["MOCK-ATTACH-TERMINATED"])
        def wait(self, timeout): return 0
        def kill(self): raise AssertionError("attach kill was unnecessary")
    monkeypatch.setattr(ops.subprocess, "Popen", lambda *args, **kwargs: Attach())
    return plan, controls


def test_mock_startup_records_exact_identity_six_generated_calls_and_no_private_text(tmp_path, monkeypatch):
    plan, controls = startup_fixture(tmp_path, monkeypatch)
    result = ops.supervise(plan, tmp_path, "synthetic-release", wall_seconds=60)
    assert result["status"] == "passed" and result["generated_startup_posts"] == 6
    assert result["private_quality_calls"] == 0 and result["automatic_retry"] is False
    assert result["actual_ready_identity"] == result["expected_runtime_identity"]
    assert ["docker", "start", "synthetic-only"] in controls
    assert ["MOCK-ATTACH-TERMINATED"] in controls
    assert plan["failure_stop_argv"] not in controls
    assert "PRIVATE_SENTINEL" not in (tmp_path / "startup.safe-events.jsonl").read_text()


def test_mock_startup_identity_failure_stops_container_without_retry(tmp_path, monkeypatch):
    plan, controls = startup_fixture(tmp_path, monkeypatch, bad_identity=True)
    with pytest.raises(ops.OpsError, match="ready_identity_mismatch"):
        ops.supervise(plan, tmp_path, "synthetic-release", wall_seconds=60)
    receipt = json.loads((tmp_path / "startup.failed.json").read_text())
    assert receipt["status"] == "failed" and receipt["container_stopped_verified"] is True
    assert receipt["private_quality_calls"] == 0
    assert controls.count(plan["failure_stop_argv"]) == 1
    assert controls.count(["docker", "start", "synthetic-only"]) == 1
    assert plan["failure_kill_only_if_still_running_argv"] not in controls


@pytest.mark.parametrize("trained", [False, True])
def test_backend_selects_own_state_file_and_prepares_cache_before_start(tmp_path, monkeypatch, trained):
    plan, controls = startup_fixture(tmp_path, monkeypatch, trained=trained)
    result = ops.supervise(plan, tmp_path, "synthetic-release", wall_seconds=60)
    expected = "/run/forte/qwen38-ready.json" if trained else "/run/forte/qwen38-untuned-ready.json"
    assert result["status"] == "passed" and result["actual_state_path"] == expected
    reads = [args for args in controls if args[:2] == ["docker", "exec"]]
    assert len(reads) == 1 and reads[0][-1] == expected
    expected_paths = {"", "xdg/torch/kernels", "hf", "triton", "cuda", "torchinductor"}
    creates = [args for args in controls if args[0] == "MOCK-CACHE-MKDIR"]
    assert {args[1] for args in creates} == expected_paths
    assert all(args[2]["mode"] == 0o700 for args in creates)
    assert max(controls.index(args) for args in creates) < controls.index(["docker", "start", "synthetic-only"])


def test_trained_identity_mismatch_cannot_become_ready_or_retry(tmp_path, monkeypatch):
    plan, controls = startup_fixture(tmp_path, monkeypatch, trained=True, bad_identity=True)
    with pytest.raises(ops.OpsError, match="ready_identity_mismatch"):
        ops.supervise(plan, tmp_path, "synthetic-release", wall_seconds=60)
    assert controls.count(plan["failure_stop_argv"]) == 1
    assert not (tmp_path / "startup.ready.json").exists()
