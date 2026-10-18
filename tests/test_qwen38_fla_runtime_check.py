"""CPU-only guards for the optional FLA image; never import Torch/FLA."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy import qwen38_fla_runtime_check as subject


@pytest.fixture
def checked_runtime(tmp_path, monkeypatch):
    name = "fla/__init__.py"
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"public synthetic FLA file")
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    lock = {"schema_version": "forte.qwen38.fla-runtime.v1",
            "preserved_packages": {"torch": "2.13.0+cu129", "transformers": "5.17.0"},
            "addon_packages": {"fla-core": "0.5.2"},
            "addon_licenses": {"fla-core": {"license": "MIT"}},
            "installed_file_sha256": {"fla-core": {name: sha}}}
    versions = {**lock["preserved_packages"], **lock["addon_packages"]}
    monkeypatch.setattr(subject.sys, "platform", "linux")
    monkeypatch.setattr(subject.sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(subject.metadata, "version", lambda value: versions[value])
    monkeypatch.setattr(subject.metadata, "distribution", lambda value: SimpleNamespace(
        locate_file=lambda path: tmp_path / path))
    monkeypatch.setattr(subject.metadata, "distributions", lambda: [SimpleNamespace(metadata={"Name": x}) for x in versions])
    return lock, versions, target


def test_metadata_scope_is_not_gpu_execution(checked_runtime):
    lock, _, _ = checked_runtime
    result = subject.metadata_check(lock)
    assert result["status"] == "passed"
    assert result["gpu_kernel_calls"] == 0
    assert result["model_instances_created"] == 0
    assert result["cpu_imports_performed"] is False
    assert result["actual_gpu_kernel_correctness_verified"] is False


def test_core_version_drift_blocks(checked_runtime):
    lock, versions, _ = checked_runtime
    versions["torch"] = "2.14.0"
    with pytest.raises(ValueError, match="fla_package_version_mismatch"):
        subject.metadata_check(lock)


def test_source_file_drift_blocks(checked_runtime):
    lock, _, target = checked_runtime
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fla_installed_file_mismatch"):
        subject.metadata_check(lock)


def test_unplanned_causal_conv_blocks(checked_runtime, monkeypatch):
    lock, versions, _ = checked_runtime
    monkeypatch.setattr(subject.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": x}) for x in [*versions, "causal_conv1d"]])
    with pytest.raises(ValueError, match="fla_unplanned_causal_conv1d_installed"):
        subject.metadata_check(lock)


def test_closure_reads_actual_selected_function():
    def selected():
        pass

    def wrapper(implementation):
        def wrapped(*args, **kwargs):
            return implementation(*args, **kwargs)
        return wrapped

    assert subject.closure_implementation(wrapper(selected)) is selected
