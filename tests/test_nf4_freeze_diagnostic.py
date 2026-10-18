"""Synthetic CPU fingerprint contracts; actual NF4 behavior has a separate GPU receipt."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.nf4_freeze_diagnostic import compare, raw_tensor_hash, snapshot


def frozen_model():
    model = torch.nn.Linear(4, 2, bias=True).to(torch.bfloat16)
    model.requires_grad_(False)
    return model


def test_stable_frozen_storage_and_legacy_hash_agree():
    model = frozen_model()
    before = snapshot(model)
    after = snapshot(model, reference=before)
    assert compare(before, after)["changed_parameter_count"] == 0
    assert before["legacy_hash_mutated_parameters"] == []
    assert all(item["raw_sha256"] == item["legacy_raw_sha256"] for item in before["parameters"].values())


def test_exact_dtype_expansion_changes_raw_storage_and_preserves_logical_values():
    model = frozen_model()
    before = snapshot(model)
    model.bias.data = model.bias.data.float()
    after = snapshot(model, reference=before)
    difference = compare(before, after)
    assert difference["changed_parameter_count"] == 1
    assert difference["logical_changed_parameter_count"] == 0
    assert difference["changes"][0]["name"] == "bias"
    assert "dtype" in difference["changes"][0]["changed_fields"]


def test_actual_frozen_value_change_is_detected():
    model = frozen_model()
    before = snapshot(model)
    model.bias.data.add_(1)
    after = snapshot(model, reference=before)
    difference = compare(before, after)
    assert difference["logical_changed_parameter_count"] == 1


def test_lossy_dtype_cast_is_rejected_by_logical_fingerprint():
    model = torch.nn.Linear(4, 2, bias=True)
    model.requires_grad_(False)
    model.bias.data.fill_(1.0001)
    before = snapshot(model)
    model.bias.data = model.bias.data.to(torch.bfloat16)
    after = snapshot(model, reference=before)
    assert compare(before, after)["logical_changed_parameter_count"] == 1


def test_trainable_parameters_are_not_silently_in_frozen_scope():
    model = frozen_model()
    model.bias.requires_grad_(True)
    assert set(snapshot(model)["parameters"]) == {"weight"}


def test_fingerprint_handles_scalar_and_noncontiguous_views():
    assert len(raw_tensor_hash(torch.tensor(3.0))) == 64
    values = torch.arange(6).reshape(2, 3).transpose(0, 1)
    assert raw_tensor_hash(values) == raw_tensor_hash(values.contiguous())
