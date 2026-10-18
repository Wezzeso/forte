"""Host containment contracts; no Docker, GPU, real dataset or network is used."""
from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import pytest

from tools import run_public_training_host as host


@pytest.fixture(autouse=True)
def portable_identity(monkeypatch):
    if not hasattr(os, "getuid"):
        monkeypatch.setattr(host.os, "getuid", lambda: 1000, raising=False)
        monkeypatch.setattr(host.os, "getgid", lambda: 1000, raising=False)


def closure(tmp_path):
    source = tmp_path / "source"
    package = source / "training"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "fit.py").write_text("value = 1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {
        p.relative_to(source).as_posix(): host.sha256(p) for p in source.rglob("*") if p.is_file()}}))
    return source, manifest


def test_source_closure_is_pinned_including_empty_init(tmp_path):
    source, manifest = closure(tmp_path)
    assert host.verify_sources(source, manifest, host.sha256(manifest))["file_count"] == 2
    (source / "training" / "fit.py").write_text("value = 2\n")
    with pytest.raises(host.GuardError, match="source_file_hash_mismatch"):
        host.verify_sources(source, manifest, host.sha256(manifest))


def test_unmanifested_importable_file_rejected(tmp_path):
    source, manifest = closure(tmp_path)
    (source / "training" / "untracked.py").write_text("pass\n")
    with pytest.raises(host.GuardError, match="source_closure_mismatch"):
        host.verify_sources(source, manifest, host.sha256(manifest))


def test_manifest_must_stay_outside_closure(tmp_path):
    source, _ = closure(tmp_path)
    manifest = source / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(host.GuardError, match="source_manifest_must_be_external"):
        host.verify_sources(source, manifest, host.sha256(manifest))


@pytest.mark.parametrize("name", ["private/data.json", ".env", "training/weights.pth"])
def test_private_and_binary_source_payload_rejected(tmp_path, name):
    source, manifest = closure(tmp_path)
    extra = source / name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"synthetic")
    data = json.loads(manifest.read_text())
    data["files"][name] = host.sha256(extra)
    manifest.write_text(json.dumps(data))
    with pytest.raises(host.GuardError, match="source_relative_path_invalid"):
        host.verify_sources(source, manifest, host.sha256(manifest))


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink permissions vary")
def test_source_symlink_rejected(tmp_path):
    source, manifest = closure(tmp_path)
    (source / "training" / "alias.py").symlink_to(source / "training" / "fit.py")
    with pytest.raises(host.GuardError, match="source_symlink_forbidden"):
        host.verify_sources(source, manifest, host.sha256(manifest))


def test_hash_pin_rejects_replaced_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"limit": 1}')
    pinned = host.sha256(path)
    path.write_text('{"limit": 2}')
    with pytest.raises(host.GuardError, match="input_hash_mismatch"):
        host.pinned_json(path, pinned)


@pytest.mark.parametrize("free, own, accepted", [(1536, 5632, True), (1535, 1000, False),
                                                (7000, 5633, False)])
def test_memory_guards_include_unattributed_global_usage(free, own, accepted):
    if accepted:
        host.check_sample({"free_mib": free}, own)
    else:
        with pytest.raises(host.GuardError):
            host.check_sample({"free_mib": free}, own)


def container_fixture(tmp_path, phase="rf-fit"):
    args = Namespace(phase=phase, name="forte-public-contract", image="sha256:" + "a" * 64)
    source = tmp_path / "source"
    config = tmp_path / "config.json"
    data = tmp_path / "data"
    checked = {"source": source, "config_path": config, "output": tmp_path / "run" / "output",
               "registry": tmp_path / "registry", "readonly": [source, config, data]}
    return args, checked


def test_cpu_preflight_has_no_gpu_and_no_writable_host_mounts(tmp_path):
    args, checked = container_fixture(tmp_path)
    command = host.container_args(args, checked, preflight=True)
    assert "--gpus" not in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "PYTHONDONTWRITEBYTECODE=1" in command
    assert "PYTORCH_ALLOC_CONF=backend:native" in command
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    assert mounts and all(item.endswith(",readonly") for item in mounts)
    bootstrap = command[-1]
    assert "--preflight" in bootstrap and "torch.cuda.is_initialized()" in bootstrap


def test_gpu_container_is_offline_readonly_and_only_two_writable_mounts(tmp_path):
    args, checked = container_fixture(tmp_path)
    command = host.container_args(args, checked, preflight=False)
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--gpus") + 1] == "device=0"
    assert "--read-only" in command and "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert command[-1] == "--defer-test"
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    writable = [item for item in mounts if not item.endswith(",readonly")]
    assert set(writable) == {f"type=bind,src={p},dst={p}"
                             for p in (checked["output"].parent, checked["registry"])}


@pytest.mark.parametrize("phase, last", [("crop-smoke", "--memory-smoke"),
                                         ("crop-fit", None)])
def test_crop_phases_never_invoke_sealed_evaluation(tmp_path, phase, last):
    args, checked = container_fixture(tmp_path, phase)
    command = host.container_args(args, checked, preflight=False)
    assert "training.public_crop_lora" in command
    assert not any("final-eval" in item or "cohort" in item for item in command)
    if last:
        assert command[-1] == last


@pytest.mark.parametrize('preflight', [True, False])
@pytest.mark.parametrize('phase', ['crop-smoke', 'crop-fit'])
def test_crop_cache_exec_is_bounded_separate_and_only_containment_change(tmp_path, phase, preflight):
    args, checked = container_fixture(tmp_path, phase)
    command = host.container_args(args, checked, preflight=preflight)
    tmpfs = [command[i + 1] for i, value in enumerate(command) if value == '--tmpfs']
    cache = '/triton-cache:rw,nosuid,nodev,exec,size=536870912,mode=1777'
    assert tmpfs == ['/tmp:rw,size=2147483648,mode=1777', cache]
    assert command.count('TRITON_CACHE_DIR=/triton-cache') == 1
    assert command[command.index('--cpus') + 1] == '4'
    assert command[command.index('--memory') + 1] == '24g'
    assert command[command.index('--shm-size') + 1] == '1g'
    assert command[command.index('--cap-drop') + 1] == 'ALL'
    assert command[command.index('--security-opt') + 1] == 'no-new-privileges'
    prefix = command[:command.index('--entrypoint')]
    for flag, value in [('--tmpfs', cache), ('--env', 'TRITON_CACHE_DIR=/triton-cache')]:
        index = prefix.index(value)
        assert prefix[index - 1] == flag
        del prefix[index - 1:index + 1]
    args.phase = 'rf-fit'
    rf = host.container_args(args, checked, preflight=preflight)
    assert prefix == rf[:rf.index('--entrypoint')]
    assert '/triton-cache' not in ' '.join(rf) and 'TRITON_CACHE_DIR' not in ' '.join(rf)


def test_progress_emits_only_finite_small_allowlist(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"state": "training", "optimizer_steps": 4,
        "credentials": "NEVER-EMIT", "description": "NEVER-EMIT", "recent_loss": float("nan")}))
    (tmp_path / "progress.json").write_text(json.dumps({"epoch": 2, "physical_batches": 32,
        "recent_losses": [4.0, 2.0], "metrics": {"secret": "NEVER-EMIT"}}))
    assert host.compact_progress(tmp_path, 10) == {"state": "training", "optimizer_steps": 4,
        "epoch": 2, "physical_batches": 32, "recent_loss": 2.0}


def test_progress_rejects_arbitrary_error_strings_and_partial_json(tmp_path):
    (tmp_path / "run.json").write_text('{"state": "error body secret@example.org", "optimizer_steps": -1}')
    (tmp_path / "progress.json").write_text('{"partial":')
    assert host.compact_progress(tmp_path, 1) == {}


def test_cleanup_will_not_kill_name_reused_for_different_id(monkeypatch):
    monkeypatch.setattr(host, "inspect_container", lambda _: {"Id": "b" * 64, "State": {"Running": True}})
    commands = []
    monkeypatch.setattr(host, "command", lambda args, **kwargs: commands.append(args))
    with pytest.raises(host.GuardError, match="owned_container_identity_changed"):
        host.stop_owned("forte-public-contract", "a" * 64)
    assert not commands


def test_cleanup_kills_only_exact_created_id_and_retains_container(monkeypatch):
    container_id = "a" * 64
    states = iter([{"Id": container_id, "State": {"Running": True}},
                   {"Id": container_id, "State": {"Running": False}}])
    monkeypatch.setattr(host, "inspect_container", lambda _: next(states))
    commands = []
    monkeypatch.setattr(host, "command", lambda args, **kwargs: commands.append(args))
    host.stop_owned("forte-public-contract", container_id)
    assert commands == [["docker", "kill", container_id]]


@pytest.mark.parametrize("wall, deadline, code", [(0, "2099-01-01T00:00:00Z", "hard_wall"),
    (10, "2000-01-01T00:00:00Z", "absolute_deadline")])
def test_expired_budget_never_starts_container(monkeypatch, wall, deadline, code):
    commands = []
    monkeypatch.setattr(host, "command", lambda args, **kwargs: commands.append(args))
    with pytest.raises(host.GuardError, match=code):
        host.wait_container("forte-public-contract", "a" * 64, None,
                            deadline=host.utc(deadline), wall_started=host.time.monotonic(),
                            wall_seconds=wall)
    assert commands == []


@pytest.mark.skipif(os.name == "nt", reason="Validating canonical Linux host paths")
def test_validated_inputs_and_writable_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "WRITABLE_ROOT", tmp_path / "authorized")
    root = host.WRITABLE_ROOT
    root.mkdir()
    source, manifest = closure(tmp_path)
    dataset = tmp_path / "data"
    dataset.mkdir()
    run = root / "runs" / "one"
    run.mkdir(parents=True)
    registry = root / "registry"
    registry.mkdir()
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"synthetic-not-real-weights")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"stop_at_utc": "2026-10-18T05:45:00Z", "gpu_memory_budget_mib": 5120,
        "defer_sealed_test": True, "model_class": "RFDETRSmall", "max_fit_seconds": 7200,
        "dataset_dir": str(dataset), "output_dir": str(run / "output"),
        "test_exposure_registry_dir": str(registry), "pretrain_weights": str(weights),
        "expected_pretrain_weights_sha256": host.sha256(weights)}))
    args = Namespace(image="sha256:" + "a" * 64, name="forte-public-contract", phase="rf-fit",
        max_wall_seconds=7800, stop_at_utc="2026-10-18T05:45:00Z", config=config,
        expected_config_sha256=host.sha256(config), source_dir=source, source_manifest=manifest,
        expected_source_manifest_sha256=host.sha256(manifest), report=tmp_path / "receipt.json")
    now = datetime(2026, 10, 18, 1, tzinfo=timezone.utc)
    assert host.validate_inputs(args, now=now)["output"] == run / "output"
    data = json.loads(config.read_text())
    outside = tmp_path / "outside"
    outside.mkdir()
    data["output_dir"] = str(outside / "output")
    config.write_text(json.dumps(data))
    args.expected_config_sha256 = host.sha256(config)
    with pytest.raises(host.GuardError, match="writable_mount_outside_authorized_root"):
        host.validate_inputs(args, now=now)


def test_utc_and_authorization_deadline_are_fixed():
    assert host.HARD_STOP == datetime(2026, 10, 18, 5, 45, tzinfo=timezone.utc)
    with pytest.raises(host.GuardError, match="deadline_must_be_utc"):
        host.utc("2026-10-18T10:45:00+05:00")


@pytest.mark.skipif(os.name == "nt", reason="Linux flock and SIGALRM supervision")
@pytest.mark.parametrize("failure_phase", ["cpu", "gpu"])
def test_supervisor_failure_stops_only_owned_containers_and_checks_production(tmp_path, monkeypatch, failure_phase):
    args, checked = container_fixture(tmp_path)
    args.max_wall_seconds = 10
    args.stop_at_utc = "2026-10-18T05:45:00Z"
    args.expected_config_sha256 = "c" * 64
    args.expected_source_manifest_sha256 = "d" * 64
    checked.update(config={}, manifest=tmp_path / "manifest.json", report=tmp_path / "receipt.json",
                   source_receipt={"manifest_sha256": "d" * 64}, pins={},
                   stop=datetime.now(timezone.utc) + timedelta(seconds=30))
    monkeypatch.setattr(host, "validate_inputs", lambda _: checked)
    monkeypatch.setattr(host, "verify_sources", lambda *a: {})
    monkeypatch.setattr(host, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(host, "inspect_container", lambda _: None)
    binding = {"all_five": "immutable-test-binding"}
    monkeypatch.setattr(host, "production_bindings", lambda: dict(binding))
    monkeypatch.setattr(host, "readiness", lambda: {"status": "ready"})
    monkeypatch.setattr(host, "queue_state", lambda: {"active": 0, "queued": 0})
    monkeypatch.setattr(host, "gpu_state", lambda: ({"free_mib": 7000}, {}))
    owned = []
    monkeypatch.setattr(host, "stop_owned", lambda name, cid: owned.append((name, cid)))
    creates = iter(["a" * 64, "b" * 64])

    def command(command_args, **kwargs):
        if command_args[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"Id": args.image}])
        assert command_args[:2] == ["docker", "create"]
        return next(creates)

    monkeypatch.setattr(host, "command", command)

    def wait(name, *unused, **kwargs):
        if name.endswith("-preflight"):
            return {"State": {"ExitCode": 1 if failure_phase == "cpu" else 0}}
        raise host.GuardError("global_free_memory_guard")

    monkeypatch.setattr(host, "wait_container", wait)
    assert host.execute(args) == 1
    report = json.loads(checked["report"].read_text())
    assert report["production_unchanged"] is True
    assert report["queue_after"] == {"active": 0, "queued": 0}
    assert report["owned_containers_retained_stopped"] is True
    if failure_phase == "cpu":
        assert owned == [(args.name + "-preflight", "a" * 64)]
        assert report["failure"] == "cpu_preflight_failed"
    else:
        assert owned == [(args.name, "b" * 64), (args.name + "-preflight", "a" * 64)]
        assert report["failure"] == "global_free_memory_guard"

