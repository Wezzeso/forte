"""Synthetic byte fixtures only: no public/private media or model execution."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from training.guards import TrainingGateError
from training.specialists.common import canonical_sha256, file_sha256, load_json, write_json
from training.specialists.public_cohort import (
    AUTHORIZATION_SCHEMA, claim_cohort_member, register_cohort, sealed_inventory,
    verify_cohort,
)
from training.specialists.public_train import claim_sealed_test


def pin(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


@pytest.fixture
def cohort_inputs(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "sealed_test").mkdir(parents=True)
    (dataset / "sealed_test/sample.png").write_bytes(b"synthetic bytes, never decoded")
    (dataset / "sealed_test/_annotations.coco.json").write_bytes(b"synthetic annotation bytes")
    files = {str(p.relative_to(dataset)).replace("\\", "/"): file_sha256(p)
             for p in (dataset / "sealed_test").iterdir()}
    lock = {"schema_version": "forte.specialists.public-dataset-lock.v1", "state": "frozen",
            "origin": "public", "files": files}
    lock["dataset_sha256"] = canonical_sha256(lock)
    write_json(dataset / "dataset-lock.json", lock)
    lock_sha = file_sha256(dataset / "dataset-lock.json")
    inventory = canonical_sha256(files)
    registry = tmp_path / "global-registry"
    auth = {"schema_version": AUTHORIZATION_SCHEMA, "authorized_by": "root",
            "authorization_reference": "synthetic fixture predeclaration", "registry_dir": str(registry),
            "dataset_lock_sha256": lock_sha, "sealed_inventory_sha256": inventory,
            "member_ids": ["crop_vlm", "detector"]}
    write_json(tmp_path / "authorization.json", auth)
    members = {}
    for name in ("detector", "crop_vlm"):
        output = tmp_path / name
        output.mkdir()
        config = {"dataset_dir": str(dataset), "expected_dataset_lock_sha256": lock_sha,
                  "test_exposure_registry_dir": str(registry), "output_dir": str(output)}
        write_json(output / "config.json", config)
        (output / "checkpoint.bin").write_bytes((name + " frozen checkpoint").encode())
        (output / "evaluation.py").write_bytes(b"# pinned synthetic source")
        (output / "dependency.lock").write_bytes(b"synthetic==1")
        run = {"state": "trained_unselected_pending_test", "optimizer_steps": 3,
               "dataset_lock_sha256": lock_sha, "config_sha256": file_sha256(output / "config.json"),
               "test_evaluation_count": 0}
        checkpoint_sha = file_sha256(output / "checkpoint.bin")
        if name == "detector":
            run["schema_version"] = "forte.specialists.public-run.v1"
            run["checkpoint_sha256"] = checkpoint_sha
        else:
            run["schema_version"] = "forte.public-crop-lora.run.v1"
            run["adapter_files"] = {"adapter_model.safetensors": checkpoint_sha}
        write_json(output / "evaluation-protocol.json", {
            "schema_version": "synthetic.protocol.v1", "member_id": name,
            "dataset_lock_sha256": lock_sha, "sealed_inventory_sha256": inventory,
            "selection_scope": "public_train_dev_only", "test_reuse_for_tuning_allowed": False,
            "outcomes": ["selected_model"], "resolution_stress": [0, 64, 32, 16],
            "runner_sha256" if name == "detector" else "source_sha256": file_sha256(output / "evaluation.py")})
        run["evaluation_protocol_sha256"] = file_sha256(output / "evaluation-protocol.json")
        write_json(output / "run.json", run)
        members[name] = {"run": pin(output / "run.json"), "checkpoint": pin(output / "checkpoint.bin"),
                         "config": pin(output / "config.json"), "protocol": pin(output / "evaluation-protocol.json"),
                         "artifacts": {"evaluation_source": pin(output / "evaluation.py"),
                                       "dependencies": pin(output / "dependency.lock")}}
    return {"registry_dir": registry, "dataset_lock_path": dataset / "dataset-lock.json",
            "expected_dataset_lock_sha256": lock_sha, "members": members,
            "authorization_path": tmp_path / "authorization.json",
            "expected_authorization_sha256": file_sha256(tmp_path / "authorization.json")}


def claim_args(inputs, receipt, member="detector"):
    selected = inputs["members"][member]
    return {"config": load_json(Path(selected["config"]["path"])),
            "lock": load_json(inputs["dataset_lock_path"]),
            "checkpoint_sha256": selected["checkpoint"]["sha256"],
            "cohort_receipt": receipt, "expected_cohort_sha256": file_sha256(receipt),
            "member_id": member, "run_path": Path(selected["run"]["path"]),
            "config_path": Path(selected["config"]["path"]),
            "protocol_path": Path(selected["protocol"]["path"])}


def test_joint_registration_and_both_single_attempts(cohort_inputs):
    receipt = register_cohort(**cohort_inputs)
    original = receipt.read_bytes()
    value = verify_cohort(receipt, file_sha256(receipt))
    inventory = canonical_sha256(sealed_inventory(load_json(cohort_inputs["dataset_lock_path"])))
    assert receipt.name == f"public-sealed-{inventory}.json"
    assert value["member_ids"] == ["crop_vlm", "detector"]
    assert "not_independent_replications" in value["interpretation"]
    for member in ("detector", "crop_vlm"):
        attempt = claim_cohort_member(**claim_args(cohort_inputs, receipt, member))
        assert attempt.parent == receipt.parent
        assert load_json(attempt)["failure_consumes_attempt"] is True
        with pytest.raises(TrainingGateError, match="already_attempted"):
            claim_cohort_member(**claim_args(cohort_inputs, receipt, member))
    assert receipt.read_bytes() == original


def test_cohort_blocks_legacy_noncohort_claim(cohort_inputs):
    receipt = register_cohort(**cohort_inputs)
    args = claim_args(cohort_inputs, receipt)
    args["config"]["output_dir"] += "-different-task"
    with pytest.raises(TrainingGateError, match="sealed_test_already_exposed"):
        claim_sealed_test(args["config"], args["lock"], args["checkpoint_sha256"])


def test_legacy_exposure_blocks_cohort_even_if_outdir_or_checkpoint_changes(cohort_inputs):
    member = cohort_inputs["members"]["detector"]
    config = load_json(Path(member["config"]["path"]))
    lock = load_json(cohort_inputs["dataset_lock_path"])
    claim_sealed_test(config, lock, "a" * 64)
    with pytest.raises(TrainingGateError, match="already_reserved_or_exposed"):
        register_cohort(**cohort_inputs)


@pytest.mark.parametrize("bad_members", [{}, {"detector": {}}, {"detector": {}, "crop_vlm": {}, "retry": {}}])
def test_every_member_must_be_declared_upfront(cohort_inputs, bad_members):
    cohort_inputs["members"] = bad_members
    with pytest.raises(TrainingGateError, match="exact_members_required"):
        register_cohort(**cohort_inputs)


@pytest.mark.parametrize("field", ["run", "checkpoint", "config", "protocol", "evaluation_source", "dependencies"])
def test_other_member_artifact_changes_block_claim(cohort_inputs, field):
    receipt = register_cohort(**cohort_inputs)
    member = cohort_inputs["members"]["crop_vlm"]
    selected = member[field] if field in member else member["artifacts"][field]
    Path(selected["path"]).write_bytes(b"changed after both selected")
    with pytest.raises(TrainingGateError, match="artifact_hash_mismatch"):
        claim_cohort_member(**claim_args(cohort_inputs, receipt, "detector"))


def test_after_one_attempt_other_frozen_model_still_required(cohort_inputs):
    receipt = register_cohort(**cohort_inputs)
    claim_cohort_member(**claim_args(cohort_inputs, receipt, "detector"))
    run = Path(cohort_inputs["members"]["detector"]["run"]["path"])
    run.write_bytes(b"mutated after first evaluation")
    with pytest.raises(TrainingGateError, match="artifact_hash_mismatch"):
        claim_cohort_member(**claim_args(cohort_inputs, receipt, "crop_vlm"))


def test_test_payload_mutation_is_rejected(cohort_inputs):
    receipt = register_cohort(**cohort_inputs)
    (cohort_inputs["dataset_lock_path"].parent / "sealed_test/sample.png").write_bytes(b"new bytes")
    with pytest.raises(TrainingGateError, match="test_file_hash_mismatch"):
        claim_cohort_member(**claim_args(cohort_inputs, receipt))


def test_claim_cannot_substitute_subset_lock(cohort_inputs):
    receipt = register_cohort(**cohort_inputs)
    args = claim_args(cohort_inputs, receipt)
    args["lock"]["files"].pop("sealed_test/sample.png")
    with pytest.raises(TrainingGateError, match="claim_config_or_dataset_mismatch"):
        claim_cohort_member(**args)


@pytest.mark.parametrize("kind", ["base", "attempt"])
def test_partial_or_malformed_marker_fails_closed(cohort_inputs, kind):
    if kind == "base":
        registry = cohort_inputs["registry_dir"]
        registry.mkdir()
        identity = canonical_sha256(sealed_inventory(load_json(cohort_inputs["dataset_lock_path"])))
        (registry / f"public-sealed-{identity}.json").write_bytes(b"")
        with pytest.raises(TrainingGateError, match="already_reserved_or_exposed"):
            register_cohort(**cohort_inputs)
    else:
        receipt = register_cohort(**cohort_inputs)
        receipt.with_name(receipt.stem + ".member-detector.attempt.json").write_bytes(b"{partial")
        with pytest.raises(TrainingGateError, match="already_attempted"):
            claim_cohort_member(**claim_args(cohort_inputs, receipt))


def test_failure_after_claim_cannot_retry(cohort_inputs):
    receipt = register_cohort(**cohort_inputs)
    attempt = claim_cohort_member(**claim_args(cohort_inputs, receipt))
    # Caller failed before producing metrics. Immutable marker remains spent.
    assert load_json(attempt)["state"] == "attempted"
    with pytest.raises(TrainingGateError, match="already_attempted"):
        claim_cohort_member(**claim_args(cohort_inputs, receipt))


def test_copied_receipt_or_different_registry_cannot_claim(cohort_inputs, tmp_path):
    receipt = register_cohort(**cohort_inputs)
    other = tmp_path / "new-registry"
    other.mkdir()
    copied = other / receipt.name
    copied.write_bytes(receipt.read_bytes())
    args = claim_args(cohort_inputs, copied)
    with pytest.raises(TrainingGateError, match="receipt_registry_mismatch"):
        claim_cohort_member(**args)
    changed = dict(cohort_inputs, registry_dir=other)
    with pytest.raises(TrainingGateError, match="authorization_scope_mismatch"):
        register_cohort(**changed)


def test_frozen_config_cannot_change_outdir_or_registry(cohort_inputs):
    receipt = register_cohort(**cohort_inputs)
    args = claim_args(cohort_inputs, receipt)
    args["config"]["output_dir"] += "-retry"
    with pytest.raises(TrainingGateError, match="claim_config_or_dataset_mismatch"):
        claim_cohort_member(**args)


@pytest.mark.parametrize("member", ["retry", "../detector", "detector-2", "Detector"])
def test_task_namespace_cannot_allocate_new_member(cohort_inputs, member):
    receipt = register_cohort(**cohort_inputs)
    args = claim_args(cohort_inputs, receipt)
    args["member_id"] = member
    with pytest.raises(TrainingGateError, match="member_id_invalid"):
        claim_cohort_member(**args)


@pytest.mark.parametrize("field,value,error", [
    ("state", "training", "selected_run_mismatch"),
    ("optimizer_steps", 0, "selected_run_mismatch"),
    ("test_evaluation_count", 1, "already_exposed_or_smoke"),
    ("sealed_test_used_for_training_or_selection", True, "already_exposed_or_smoke"),
    ("memory_smoke_only", True, "already_exposed_or_smoke"),
])
def test_unfinished_or_exposed_models_cannot_register(cohort_inputs, field, value, error):
    member = cohort_inputs["members"]["crop_vlm"]
    path = Path(member["run"]["path"])
    run = load_json(path)
    run[field] = value
    write_json(path, run)
    member["run"] = pin(path)
    with pytest.raises(TrainingGateError, match=error):
        register_cohort(**cohort_inputs)


def test_protocol_cannot_describe_another_inventory(cohort_inputs):
    member = cohort_inputs["members"]["detector"]
    path = Path(member["protocol"]["path"])
    protocol = load_json(path)
    protocol["sealed_inventory_sha256"] = "a" * 64
    write_json(path, protocol)
    member["protocol"] = pin(path)
    run_path = Path(member["run"]["path"])
    run = load_json(run_path)
    run["evaluation_protocol_sha256"] = member["protocol"]["sha256"]
    write_json(run_path, run)
    member["run"] = pin(run_path)
    with pytest.raises(TrainingGateError, match="protocol_scope_mismatch"):
        register_cohort(**cohort_inputs)


def test_receipt_hash_and_implementation_are_pinned(cohort_inputs):
    receipt = register_cohort(**cohort_inputs)
    expected = file_sha256(receipt)
    record = load_json(receipt)
    record["protocol_implementation_sha256"] = "a" * 64
    write_json(receipt, record)
    with pytest.raises(TrainingGateError, match="input_hash_mismatch"):
        verify_cohort(receipt, expected)
    with pytest.raises(TrainingGateError, match="schema_or_source_mismatch"):
        verify_cohort(receipt, file_sha256(receipt))


def test_declared_config_and_run_pins_are_cross_bound(cohort_inputs):
    member = cohort_inputs["members"]["detector"]
    path = Path(member["config"]["path"])
    config = load_json(path)
    config["new_threshold"] = 0.9
    write_json(path, config)
    member["config"] = pin(path)
    with pytest.raises(TrainingGateError, match="selected_run_mismatch"):
        register_cohort(**cohort_inputs)


def test_changed_protocol_cannot_replace_run_frozen_protocol(cohort_inputs):
    member = cohort_inputs["members"]["detector"]
    path = Path(member["protocol"]["path"])
    protocol = load_json(path)
    protocol["resolution_stress"] = [0]
    write_json(path, protocol)
    member["protocol"] = pin(path)
    with pytest.raises(TrainingGateError, match="selected_run_mismatch"):
        register_cohort(**cohort_inputs)


@pytest.mark.parametrize("member_id", ["detector", "crop_vlm"])
def test_changed_evaluator_cannot_replace_protocol_frozen_source(cohort_inputs, member_id):
    member = cohort_inputs["members"][member_id]
    path = Path(member["artifacts"]["evaluation_source"]["path"])
    path.write_bytes(b"# different evaluation implementation")
    member["artifacts"]["evaluation_source"] = pin(path)
    with pytest.raises(TrainingGateError, match="protocol_source_mismatch"):
        register_cohort(**cohort_inputs)


_PROCESS_CODE = """
import json, sys
from training.specialists.public_cohort import register_cohort, claim_cohort_member
from training.guards import TrainingGateError
args = json.load(open(sys.argv[2], encoding='utf-8'))
try:
    (register_cohort if sys.argv[1] == 'register' else claim_cohort_member)(**args)
except TrainingGateError as exc:
    print(str(exc))
    raise SystemExit(2)
"""


@pytest.mark.parametrize("operation", ["register", "claim"])
def test_multi_process_race_has_exactly_one_winner(cohort_inputs, tmp_path, operation):
    args = cohort_inputs if operation == "register" else claim_args(cohort_inputs, register_cohort(**cohort_inputs))
    arguments = tmp_path / "process-arguments.json"
    arguments.write_text(json.dumps(args, default=str), encoding="utf-8")
    processes = [subprocess.Popen([sys.executable, "-c", _PROCESS_CODE, operation, str(arguments)],
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    results = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            results.append((process.returncode, stdout, stderr))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert sorted(code for code, _, _ in results) == [0, 2, 2, 2], results
    assert all(not stderr for _, _, stderr in results), results
    assert all("already_" in stdout for code, stdout, _ in results if code == 2), results
