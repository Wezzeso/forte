"""A selector of twelve names is accepted only with all 496 actual targets."""

from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from serving.qwen38_adapter_contract import (AdapterContractError, canonical, expected_targets,
    validate_inventory, verify_adapter_contract)
from serving.qwen38_source_lineage import SourceLineageError, verify_source_lineage
from serving.qwen38_identity import file_sha256
from serving.tests.qwen38_adapter_fixtures import (adapter_config_fixture, archive_fixture,
    inventory_fixture, tensor_header_fixture, write_adapter_fixture)


def verify_fixture(tmp_path, *, inventory=None, config=None, header=None, size=None):
    path = tmp_path / "generated.safetensors"
    write_adapter_fixture(path, header, size)
    return verify_adapter_contract(inventory or inventory_fixture(), config or adapter_config_fixture(), path)


def test_leaf_serialization_proves_exact_hybrid_tensor_topology_without_importing_model(tmp_path):
    with patch.dict("sys.modules", {"torch": None, "peft": None, "safetensors": None}):
        proof = verify_fixture(tmp_path)
    assert proof["full_target_count"] == 496 and proof["serialized_tensor_count"] == 992
    assert proof["serialized_leaf_selector_count"] == 12 and proof["rank"] == 8
    assert proof["trainable_parameters"] == 58_363_904
    assert proof["tensor_values_inspected"] is False and proof["model_loaded"] is False


@pytest.mark.parametrize("mutation", ["missing_target", "duplicate_target", "wrong_layer_family", "vision_target",
    "missing_layer", "wrong_layer_schedule", "wrong_dimensions", "fake_digest", "fake_groups"])
def test_inventory_requires_real_per_layer_targets_not_claimed_counts(mutation):
    inventory = inventory_fixture()
    if mutation == "missing_target": inventory["target_modules"].pop()
    elif mutation == "duplicate_target": inventory["target_modules"][-1] = inventory["target_modules"][0]
    elif mutation == "wrong_layer_family": inventory["target_modules"][0] = "model.language_model.layers.0.self_attn.q_proj"
    elif mutation == "vision_target": inventory["target_modules"][0] = "model.visual.layers.0.self_attn.q_proj"
    elif mutation == "missing_layer": inventory["layers"].pop()
    elif mutation == "wrong_layer_schedule": inventory["layers"][0]["layer_type"] = "full_attention"
    elif mutation == "wrong_dimensions": inventory["targets"][0]["in_features"] = 1024
    elif mutation == "fake_groups": inventory["groups"]["mlp"] = 1
    else: inventory["inventory_sha256"] = "f"*64
    if mutation != "fake_digest":
        inventory["inventory_sha256"] = canonical({key:value for key,value in inventory.items() if key != "inventory_sha256"})
    with pytest.raises(AdapterContractError): validate_inventory(inventory)


@pytest.mark.parametrize("mutation", ["missing_pair", "vision_tensor", "wrong_rank", "wrong_shape", "overlap",
    "trailing_bytes", "boolean_shape", "wrong_dtype", "out_of_bounds"])
def test_every_serialized_tensor_key_shape_and_storage_range_is_required(tmp_path, mutation):
    header, size = tensor_header_fixture()
    key = next(iter(header))
    if mutation == "missing_pair": del header[key]
    elif mutation == "vision_tensor": header[key.replace("language_model", "visual")] = header.pop(key)
    elif mutation == "wrong_rank": header[key]["shape"][0] = 4
    elif mutation == "wrong_shape": header[key]["shape"][1] += 1
    elif mutation == "boolean_shape": header[key]["shape"][0] = True
    elif mutation == "wrong_dtype": header[key]["dtype"] = "I8"
    elif mutation == "out_of_bounds": header[key]["data_offsets"][1] = size + 1
    elif mutation == "overlap":
        second = list(header)[1]
        extent = header[second]["data_offsets"][1] - header[second]["data_offsets"][0]
        header[second]["data_offsets"] = [0, extent]
    else: size += 2
    with pytest.raises(AdapterContractError): verify_fixture(tmp_path, header=header, size=size)


@pytest.mark.parametrize("mutation", [{"target_modules": ["q_proj", "v_proj"]},
    {"target_modules": ["visual.q_proj"]}, {"r": 16}, {"rank_pattern": {"q_proj": 4}},
    {"modules_to_save": ["lm_head"]}, {"bias": "all"}, {"use_dora": True}, {"use_rslora": True},
    {"target_parameters": ["visual.weight"]}, {"layer_replication": [[0, 1]]},
    {"layers_to_transform": [0]}, {"layers_to_transform": 0}, {"target_parameters": 0},
    {"exclude_modules": ["q_proj"]}, {"layers_pattern": "layers"},
    {"trainable_token_indices": [1]}, {"use_qalora": True}, {"lora_bias": True}])
def test_suffix_names_cannot_weaken_full_adapter_scope(tmp_path, mutation):
    with pytest.raises(AdapterContractError):
        verify_fixture(tmp_path, config={**adapter_config_fixture(), **mutation})


def test_explicit_new_serving_archive_keeps_original_training_digest_and_bytes(tmp_path):
    training, runtime = tmp_path/"training.tar", tmp_path/"runtime.tar"
    archive_fixture(training, {"serving/qwen38_identity.py": b"original"})
    archive_fixture(runtime, {"serving/qwen38_identity.py": b"reviewed validator correction",
        "serving/qwen38_adapter_contract.py": b"new strict helper"})
    training_sha, runtime_sha = file_sha256(training), file_sha256(runtime)
    proof = verify_source_lineage(training_archive=training, training_sha256=training_sha,
        recorded_training_sha256=training_sha, runtime_archive=runtime, runtime_sha256=runtime_sha)
    assert proof["verified"] and not proof["same_archive"]
    assert proof["training_source_archive_sha256"] == training_sha
    assert proof["runtime_source_archive_sha256"] == runtime_sha and file_sha256(training) == training_sha


@pytest.mark.parametrize("changed", ["training/data.py", "training/architectures.py", "src/forte/pipeline.py",
    "rules/catalog.json", "annotations/taxonomy.v1.json", "serving/qwen38_service.py"])
def test_lineage_rejects_unaudited_training_prompt_rule_or_generation_changes(tmp_path, changed):
    training, runtime = tmp_path/"training.tar", tmp_path/"runtime.tar"
    archive_fixture(training)
    archive_fixture(runtime, {changed: b"unreviewed change"})
    with pytest.raises(SourceLineageError):
        verify_source_lineage(training_archive=training, training_sha256=file_sha256(training),
            recorded_training_sha256=file_sha256(training), runtime_archive=runtime, runtime_sha256=file_sha256(runtime))


def test_lineage_rejects_run_pin_relabel_and_wrong_archive_bytes(tmp_path):
    path = tmp_path/"generated.tar"
    archive_fixture(path)
    with pytest.raises(SourceLineageError, match="training_run_source_mismatch"):
        verify_source_lineage(training_archive=path, training_sha256=file_sha256(path),
            recorded_training_sha256="0"*64, runtime_archive=path, runtime_sha256=file_sha256(path))
    with pytest.raises(SourceLineageError, match="archive_bytes_mismatch"):
        verify_source_lineage(training_archive=path, training_sha256="0"*64,
            recorded_training_sha256="0"*64, runtime_archive=path, runtime_sha256=file_sha256(path))


def test_safe_tensor_duplicate_keys_and_broken_header_are_static_errors(tmp_path):
    path = tmp_path/"invalid.safetensors"
    for raw in (b'{"same":{},"same":{}}', b'[]', b'{"x":NaN}'):
        path.write_bytes(len(raw).to_bytes(8,"little") + raw)
        with pytest.raises(AdapterContractError):
            verify_adapter_contract(inventory_fixture(), adapter_config_fixture(), path)
