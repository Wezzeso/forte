"""Fail-closed candidate package/ABI pin checks; no CUDA or model imports."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from deploy import qwen38_runtime_check as subject


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    lock = json.loads(Path("deploy/qwen38-training-runtime-lock.json").read_text())
    files = {}
    for index, name in enumerate(lock["bitsandbytes"]["installed_file_sha256"]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ("synthetic fixture %d" % index).encode()
        path.write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    lock["bitsandbytes"]["installed_file_sha256"] = files
    versions = {**lock["core_packages"], **lock["addon_packages"]}

    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(subject.metadata, "version", version)
    monkeypatch.setattr(subject.metadata, "distribution", lambda _: SimpleNamespace(
        locate_file=lambda name: tmp_path / name))
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    return lock, versions, tmp_path


def check(lock):
    return subject.verify_metadata(lock, platform_name="linux", python_version=(3, 12), machine="x86_64")


def test_metadata_verifies_pins_without_importing_gpu_packages(runtime):
    lock, _, _ = runtime
    before = set(sys.modules)
    result = check(lock)
    assert result["status"] == "passed"
    assert result["gpu_smoke_performed"] is False
    assert result["cuda_initialized"] is False
    assert result["hybrid_kernel_execution_verified"] is False
    assert result["full_qwen38_model_verified"] is False
    assert set(result["optional_kernel_packages"].values()) == {None}
    newly_loaded = set(sys.modules) - before
    assert not any(x == "torch" or x.startswith(("torch.", "bitsandbytes", "transformers"))
                   for x in newly_loaded)


@pytest.mark.parametrize("package", ["torch", "transformers", "peft", "bitsandbytes"])
def test_runtime_package_drift_rejected(runtime, package):
    lock, versions, _ = runtime
    versions[package] = "999.0"
    with pytest.raises(subject.RuntimeCheckError, match="package_version_mismatch"):
        check(lock)


def test_missing_quantization_package_rejected(runtime):
    lock, versions, _ = runtime
    del versions["bitsandbytes"]
    with pytest.raises(subject.RuntimeCheckError, match="package_missing"):
        check(lock)


@pytest.mark.parametrize("values", [
    {"platform_name": "win32", "python_version": (3, 12), "machine": "AMD64"},
    {"platform_name": "linux", "python_version": (3, 14), "machine": "x86_64"},
    {"platform_name": "linux", "python_version": (3, 12), "machine": "aarch64"},
])
def test_wrong_abi_platform_rejected(runtime, values):
    lock, _, _ = runtime
    with pytest.raises(subject.RuntimeCheckError, match="platform_mismatch"):
        subject.verify_metadata(lock, **values)


def test_changed_cuda_library_rejected(runtime):
    lock, _, root = runtime
    (root / "bitsandbytes/libbitsandbytes_cuda128.so").write_bytes(b"altered")
    with pytest.raises(subject.RuntimeCheckError, match="file_hash_mismatch"):
        check(lock)


@pytest.mark.parametrize("override", ["130", ""])
def test_unplanned_cuda_override_rejected(runtime, monkeypatch, override):
    lock, _, _ = runtime
    monkeypatch.setenv("BNB_CUDA_VERSION", override)
    with pytest.raises(subject.RuntimeCheckError, match="unplanned_bnb_cuda_override"):
        check(lock)


def test_missing_hashes_and_unsafe_paths_rejected(runtime):
    lock, _, _ = runtime
    missing = copy.deepcopy(lock)
    missing["bitsandbytes"]["installed_file_sha256"] = {}
    with pytest.raises(subject.RuntimeCheckError, match="file_pins_missing"):
        check(missing)
    unsafe = copy.deepcopy(lock)
    unsafe["bitsandbytes"]["installed_file_sha256"] = {"bitsandbytes/../secret": "0" * 64}
    with pytest.raises(subject.RuntimeCheckError, match="path_invalid"):
        check(unsafe)
