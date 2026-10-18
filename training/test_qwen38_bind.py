"""Generated local-only binder checks; no GPU, remote account or private data."""
import hashlib
import io
import json
import tarfile

import pytest

import training.qwen38_bind as binder
from training.guards import TrainingGateError, file_sha256


def test_tiny_proof_sources_join_exact_archive(tmp_path):
    archive_path = tmp_path / "generated.tar.gz"
    pins = {}
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in ("check_gradient.py", "architectures.py", "data.py", "train.py"):
            data = ("# generated " + name).encode()
            member = tarfile.TarInfo("training/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
            pins[name] = hashlib.sha256(data).hexdigest()
    binder.validate_tiny_source_archive({"source_sha256": pins}, archive_path)
    with pytest.raises(TrainingGateError, match="source_archive_pin_mismatch"):
        binder.validate_tiny_source_archive({"source_sha256": {**pins, "data.py": "0" * 64}}, archive_path)


def test_pending_proof_binds_new_recipe_and_preserves_inputs(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    source = private / "generated-source.tar.gz"
    source.write_bytes(b"generated-public-source")
    model, runtime = private / "base.json", private / "runtime.json"
    model.write_text("{}")
    runtime.write_text("{}")
    local = {"model_manifest": str(model), "runtime_lock": str(runtime),
             "expected_runtime_lock_sha256": file_sha256(runtime), "max_steps": 2,
             "code_archive_sha256": None, "expected_tiny_smoke_receipt_sha256": None}
    remote = {**local, "model_manifest": "/private/generated/base.json", "runtime_lock": "/private/generated/runtime.json"}
    monkeypatch.setattr(binder, "load_inputs", lambda _: ([{"generated": True}], {},
        {"coverage": {"target_rows_sha256": "a" * 64, "eligible_windows": 1}}))
    monkeypatch.setattr(binder, "collect_release_bindings", lambda *_args:
        {"model_manifest_sha256": file_sha256(model), "runtime_lock_sha256": file_sha256(runtime),
         "target_rows_sha256": "a" * 64})
    output_local, output_remote = private / "bound.local.json", private / "bound.remote.json"
    result = binder.bind_recipes(local, remote, source_archive=source, tiny_proof=None,
                                output_local=output_local, output_runtime=output_remote)
    assert result["source_archive_sha256"] == file_sha256(source)
    assert result["reservation_created"] is False and result["tiny_gpu_proof_ready"] is False
    assert local["code_archive_sha256"] is None and remote["code_archive_sha256"] is None
    assert json.loads(output_remote.read_text())["model_manifest"] == remote["model_manifest"]
    with pytest.raises(TrainingGateError, match="outputs_already_exist"):
        binder.bind_recipes(local, remote, source_archive=source, tiny_proof=None,
                            output_local=output_local, output_runtime=output_remote)
    with pytest.raises(TrainingGateError, match="reservation_authority_missing"):
        binder.bind_recipes(local, remote, source_archive=source, tiny_proof=None,
            output_local=private / "unauthorized.local.json", output_runtime=private / "unauthorized.remote.json", reserve=True)
    assert not (private / "unauthorized.local.json").exists()


def test_phase3_nf4_proof_bindings_join_exact_source_helper(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    model, runtime = private / "base.json", private / "runtime.json"
    model.write_text("{}")
    runtime.write_text("{}")
    helper = b"# generated proof helper"
    source = private / "generated-phase3.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        member = tarfile.TarInfo("training/qwen38_proof.py")
        member.size = len(helper)
        archive.addfile(member, io.BytesIO(helper))
    helper_sha = hashlib.sha256(helper).hexdigest()
    config = {"model_manifest": str(model), "runtime_lock": str(runtime), "quantization": "nf4",
              "expected_runtime_lock_sha256": file_sha256(runtime), "expected_tiny_smoke_receipt_sha256": None}
    monkeypatch.setattr(binder, "load_inputs", lambda _: ([], {}, {"coverage": {"target_rows_sha256": "a" * 64}}))
    bindings = {"model_manifest_sha256": file_sha256(model), "runtime_lock_sha256": file_sha256(runtime),
                "target_rows_sha256": "a" * 64, "nf4_bias_initialization_proof_sha256": "b" * 64,
                "nf4_proof_helper_source_sha256": helper_sha}
    monkeypatch.setattr(binder, "collect_release_bindings", lambda *_args: dict(bindings))
    result = binder.bind_recipes(config, config, source_archive=source, tiny_proof=None,
        output_local=private / "phase3.local.json", output_runtime=private / "phase3.remote.json")
    assert result["input_bindings"]["nf4_bias_initialization_proof_sha256"] == "b" * 64
    assert result["input_bindings"]["nf4_proof_helper_source_sha256"] == helper_sha
    bindings["nf4_proof_helper_source_sha256"] = "0" * 64
    with pytest.raises(TrainingGateError, match="nf4_helper_archive_pin_mismatch"):
        binder.bind_recipes(config, config, source_archive=source, tiny_proof=None,
            output_local=private / "wrong.local.json", output_runtime=private / "wrong.remote.json")
