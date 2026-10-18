"""Actual historical raw chain plus explicitly synthetic adversarial edits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.performance import qwen38_native_gpu_chain as chain

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def genuine():
    base = ROOT / "training/private/qwen38-vllm-bf16-generated-pilot-a08-v2"
    release = ROOT / "tmp/root-native-v2-release/trained-terminal-native-v2-release-20261017T1824Z"
    return chain.prepare_gpu_gate_chain({"terminal_dir": base / "terminal",
        "launch_plan": base / "instrumentation.v2.launch.plan.json",
        "source_archive": base / "instrumentation.v2.public-only.tar.gz",
        "root_release": Path(str(release) + ".json"), "root_launch": Path(str(release) + ".launch.json")})


def encode(document):
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return {"raw_utf8": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}


def synthetic_rebind_proofs(receipt, changes, monkeypatch):
    """Patch only unit-test trust anchors to exercise deep semantic denials."""
    terminal = json.loads(receipt["artifacts"]["terminal"]["raw_utf8"])
    for name, document in changes.items():
        descriptor = encode(document)
        receipt["terminal_files"][name] = descriptor
        terminal["files"][name]["sha256"] = descriptor["sha256"]
        terminal["files"][name]["size_bytes"] = len(descriptor["raw_utf8"].encode())
    receipt["artifacts"]["terminal"] = encode(terminal)
    monkeypatch.setitem(chain.PINS, "terminal", receipt["artifacts"]["terminal"]["sha256"])


def test_actual_raw_54_proof_chain_and_source_closure_are_historical_only(genuine):
    normalized = chain.validate_gpu_gate_chain(genuine)
    assert normalized["status"] == "passed"
    assert normalized["scope"] == "historical_offline_sequence_one"
    assert normalized["generated_calls"] == 6 and len(genuine["terminal_files"]) == 54
    assert normalized["base_files_verified"] == 32
    assert normalized["logical_targets_verified"] == 496
    assert normalized["original_tensors_verified"] == 992
    assert normalized["physical_wrappers_verified"] == 304
    assert normalized["base_quantization"] is None and normalized["max_num_seqs"] == 1
    assert normalized["does_not_prove_current_HTTP_engine"] is True
    assert normalized["does_not_prove_current_profile"] is True
    assert normalized["does_not_prove_parallel4"] is True
    assert normalized["execution_or_retry_authorized"] is False


@pytest.mark.parametrize("change", ["aggregate_only", "raw_drift", "raw_sha_echo", "archive_drift",
                                    "missing_call", "missing_root", "current_authority"])
def test_aggregate_echoes_unbound_bytes_and_missing_raw_proofs_reject(genuine, change):
    value = deepcopy(genuine)
    if change == "aggregate_only":
        value["terminal_files"] = {"status": "passed", "generated_calls": 6}
    elif change == "raw_drift":
        value["terminal_files"]["preengine.bytes.json"]["raw_utf8"] += " "
    elif change == "raw_sha_echo":
        descriptor = value["terminal_files"]["preengine.bytes.json"]
        descriptor["raw_utf8"] += " "
        descriptor["sha256"] = hashlib.sha256(descriptor["raw_utf8"].encode()).hexdigest()
    elif change == "archive_drift":
        value["source_archive"]["raw_base64"] = "AAAA"
    elif change == "missing_call":
        value["terminal_files"].pop("call-5.receipt.json")
    elif change == "missing_root":
        value["artifacts"].pop("root_launch")
    else:
        value["certifies_current_HTTP_engine"] = True
    with pytest.raises(chain.GPUChainError):
        chain.validate_gpu_gate_chain(value)


@pytest.mark.parametrize("name,path,value", [
    ("adapter.loaded-proof.json", (0, "original_tensors_verified"), 990),
    ("adapter.loaded-proof.json", (0, "logical_targets_verified"), 496.0),
    ("adapter.loaded-proof.json", (0, "loaded_cpu_and_active_gpu_exact_equality"), False),
    ("adapter.loaded-proof.json", (0, "no_partial_adapter_allowed"), False),
    ("adapter.loaded-proof.json", (0, "b_scale_folded"), 1),
    ("adapter.loaded-proof.json", (0, "execution_dtype"), "F32"),
    ("adapter.loaded-proof.json", (0, "wrapper_classes", "MergedQKVParallelLinearWithLoRA"), 15),
    ("preengine.bytes.json", ("base_files_verified",), 31),
    ("preengine.bytes.json", ("adapter_tree_sha256",), "a" * 64),
    ("preengine.bytes.json", ("model_calls",), True),
    ("runtime.after.json", (0, "quantization"), "nf4"),
    ("runtime.after.json", (0, "gpu_compute_capability"), [9, 0]),
    ("runtime.after.json", (0, "linear_attention_layers"), 47),
    ("runtime.after.json", (0, "active_adapter_ids"), []),
    ("native-grammar.audit.json", ("tests", "duplicate_rejected"), False),
    ("native-grammar.audit.json", ("backend",), "fallback"),
    ("native-grammar.audit.json", ("stop_tokens", "248046"), False),
    ("execution.attempt.json", ("source_manifest_sha256",), "b" * 64),
    ("execution.attempt.json", ("worker_extension_sha256",), "c" * 64),
    ("phase-001.json", ("generation_calls_entered",), 1),
    ("phase-034.json", ("phase",), "still_running"),
    ("phase-034.json", ("completions_validated",), 5),
    ("call-0.attempt.json", ("retry",), True),
    ("call-0.attempt.json", ("request_sha256",), "a" * 64),
    ("call-1.attempt.json", ("repeat",), 1),
    ("call-1.attempt.json", ("seed",), 17),
])
def test_deep_gate_semantics_fail_even_with_synthetic_self_consistent_hashes(genuine, monkeypatch, name, path, value):
    receipt = deepcopy(genuine)
    document = json.loads(receipt["terminal_files"][name]["raw_utf8"])
    target = document
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    synthetic_rebind_proofs(receipt, {name: document}, monkeypatch)
    with pytest.raises(chain.GPUChainError):
        chain.validate_gpu_gate_chain(receipt)


@pytest.mark.parametrize("field,value", [("strict_original_schema_and_semantics", False),
    ("actual_lora_int_id", None), ("finish_reason", "length"), ("elapsed_seconds", 241),
    ("num_cached_tokens", 1), ("completion_tokens", True)])
def test_call_receipt_and_pilot_echo_cannot_weaken_actual_completion_gates(genuine, monkeypatch, field, value):
    receipt = deepcopy(genuine)
    call = json.loads(receipt["terminal_files"]["call-1.receipt.json"]["raw_utf8"])
    pilot = json.loads(receipt["terminal_files"]["pilot.receipt.json"]["raw_utf8"])
    call[field] = value
    pilot["calls"][1] = call
    synthetic_rebind_proofs(receipt, {"call-1.receipt.json": call, "pilot.receipt.json": pilot}, monkeypatch)
    with pytest.raises(chain.GPUChainError):
        chain.validate_gpu_gate_chain(receipt)


def test_flat_contract_import_needs_no_site_packages_or_runtime():
    import subprocess
    import sys
    source = ROOT / "benchmarks/performance/qwen38_native_gpu_chain.py"
    script = "import runpy,sys;runpy.run_path(sys.argv[1]);assert not any(k.split('.')[0] in {'torch','vllm','PIL','httpx','forte','serving'} for k in sys.modules)"
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(source)], capture_output=True)
    assert result.returncode == 0
