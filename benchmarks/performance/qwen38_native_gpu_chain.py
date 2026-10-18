"""Pure validation of the historical, offline sequence-one native V2 proof.

This module imports only the standard library. It never loads weights, media,
models or runtime packages and never grants current HTTP/engine authority.
"""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile

SCHEMA = "forte.qwen38.native-historical-gpu-chain.v1"
PINS = {
    "terminal": "5ea39bd22a19d409a8103fe4a0f99c78017a966ded07d028786629b681faaa63",
    "launch_plan": "1127eba7a12bcb679953cfd713bbb32d8a04ec308a253e1289f983716b4112f0",
    "source_archive": "9de54a58de74c850cc652bb5412f0a01dcaa93c4475fd89d39ea0fc49080078c",
    "root_release": "5e83146b1921d3a9ca1e4961a6c696bfe9968f2f4991c4cae99af58fcc6bf5a6",
    "root_launch": "c1579875823c4db29f0fa8a2e1b55d7209693a3aa46e09c6420bd2f39137a0b3",
}
SOURCE_MANIFEST = "7cd2020b54de0128fd803205ca4cb5ac3ab4b8ca7d339381fe6e10db3809de21"
BASE = "9be85d7b1a57be0f59e707b9a8872b964c506a515f7670958d2d5d60de197225"
IMAGE = "baaaafe69f963b32f2fa698166fe9abaf85afb68b91974f59527c6bfd7c95f02"
ADAPTER = "51780bf7369885f4157c6b08966f425a2196f0b4a0480f07dc9255b6fb6760d5"
WEIGHTS = "dd9ba2303067d843b362e2881ffc456fc05c0b35231108a3b6270fcec4818ed2"
WORKER = "16f7d9b948177cd4488b20a439e469deb591f3e2ba4fe6d6d749b02ab6391328"
MAPPING = "6e9f8c821aab28be6476e46baeeba44e2ca10005b73587fb3933637a4118f359"
DERIVED_SCHEMA = "2bbeee8123a306837c0c713933d905709f8b544670702662e45843ac551b357e"
ROOT_RELEASE_ID = "root-native-bf16-v2-generated-20261017T1824Z"
CONTAINER = "40cca8906b0dbe0c074e87d955aa9eb095f5f397acde0511c73098a4954a95e5"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
RESULTS = "/data/forte-ml/private/qwen38-vllm-bf16-generated-pilot-a08-v2-results-root/" + ROOT_RELEASE_ID
OMITTED_INPUT_FILES = {"request.base.json", "request.fit.json", "request.metadata.json",
                       "schema.original.json", "schema.canonical19.json"}


class GPUChainError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise GPUChainError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf8")


def equal(actual, expected):
    return canonical(actual) == canonical(expected)


def _pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "native_gpu_chain_duplicate_json_key")
        result[key] = value
    return result


def parse(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(GPUChainError("native_gpu_chain_nonfinite_json")))
        canonical(value)  # Also reject overflowed JSON floats.
        return value
    except GPUChainError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise GPUChainError("native_gpu_chain_invalid_json") from None


def _raw(row, pin=None):
    require(isinstance(row, dict) and set(row) == {"raw_utf8", "sha256"}
            and isinstance(row["raw_utf8"], str), "native_gpu_chain_raw_document_required")
    try:
        raw = row["raw_utf8"].encode("utf8")
    except UnicodeError:
        raise GPUChainError("native_gpu_chain_raw_utf8_invalid") from None
    require(len(raw) <= 2 * 1024 * 1024 and digest(raw) == row["sha256"]
            and (pin is None or row["sha256"] == pin), "native_gpu_chain_raw_sha_mismatch")
    return parse(raw), raw


def _archive(row):
    require(isinstance(row, dict) and set(row) == {"raw_base64", "sha256", "size_bytes"}
            and row["sha256"] == PINS["source_archive"]
            and type(row["size_bytes"]) is int and 0 < row["size_bytes"] <= 4 * 1024 * 1024,
            "native_gpu_chain_public_archive_required")
    try:
        raw = base64.b64decode(row["raw_base64"], validate=True)
    except (ValueError, TypeError):
        raise GPUChainError("native_gpu_chain_archive_encoding_invalid") from None
    require(len(raw) == row["size_bytes"] and digest(raw) == row["sha256"],
            "native_gpu_chain_archive_sha_mismatch")
    files, total = {}, 0
    with tarfile.open(fileobj=BytesIO(raw), mode="r:*") as archive:
        for member in archive:
            name = member.name
            path = PurePosixPath(name)
            require(member.isfile() and not member.issym() and not member.islnk()
                    and name not in files and not path.is_absolute() and path.as_posix() == name
                    and ".." not in path.parts and "\\" not in name and ":" not in name
                    and 0 <= member.size <= 2 * 1024 * 1024, "native_gpu_chain_archive_member_invalid")
            total += member.size
            require(total <= 16 * 1024 * 1024 and len(files) < 100,
                    "native_gpu_chain_archive_limit")
            stream = archive.extractfile(member)
            require(stream is not None, "native_gpu_chain_archive_unreadable")
            files[name] = stream.read()
    require(len(files) == 75 and "pilot-source.sha256.json" in files,
            "native_gpu_chain_exact_source_inventory_required")
    manifest = files.pop("pilot-source.sha256.json")
    require(digest(manifest) == SOURCE_MANIFEST, "native_gpu_chain_source_manifest_changed")
    source = parse(manifest)
    require(isinstance(source, dict) and len(source) == 74
            and {name: digest(value) for name, value in files.items()} == source,
            "native_gpu_chain_source_closure_mismatch")
    require(digest(files["base-public.manifest.json"]) == BASE
            and digest(files["serving/qwen38_vllm_worker.py"]) == WORKER
            and digest(files["all-496-logical-targets.expected.json"]) == MAPPING,
            "native_gpu_chain_base_worker_mapping_pins_changed")
    base = parse(files["base-public.manifest.json"])
    require(base.get("model_id") == "Qwen/Qwen3.8-27B" and base.get("revision") == REVISION
            and isinstance(base.get("files"), list) and len(base["files"]) == 32,
            "native_gpu_chain_32_base_files_required")
    return files, source


def _number(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum


def _one(document, code):
    require(isinstance(document, list) and len(document) == 1 and isinstance(document[0], dict), code)
    return document[0]


def validate_gpu_gate_chain(receipt):
    require(isinstance(receipt, dict) and set(receipt) == {"schema_version", "artifacts", "source_archive", "terminal_files"}
            and receipt["schema_version"] == SCHEMA, "native_gpu_chain_receipt_schema_invalid")
    artifacts = receipt["artifacts"]
    require(isinstance(artifacts, dict) and set(artifacts) == set(PINS) - {"source_archive"},
            "native_gpu_chain_root_artifacts_missing")
    documents = {key: _raw(value, PINS[key])[0] for key, value in artifacts.items()}
    terminal, plan, release, launch = (documents[key] for key in ("terminal", "launch_plan", "root_release", "root_launch"))
    files, source = _archive(receipt["source_archive"])
    require(plan["source_hashes"] == source, "native_gpu_chain_launch_source_closure_mismatch")
    bounds = plan["bounds"]
    require(all(equal(bounds.get(k), v) for k, v in {"arms": 2, "calls_per_arm": 3,
        "max_autoregressive_requests": 6, "automatic_retries": 0,
        "startup_and_full_base_hash_deadline_seconds": 1200, "per_request_hard_deadline_seconds": 240,
        "no_generation_before_adapter_and_native_grammar_gates": True}.items()),
        "native_gpu_chain_launch_bounds_changed")
    require(release.get("schema_version") == "forte.qwen38.root-trained-terminal-native-v2-release.v1"
            and release.get("root_release_id") == launch.get("root_release_id") == ROOT_RELEASE_ID
            and release.get("released_by_root") is True and release.get("root_exclusive_gpu_owner") is True
            and release.get("native_plan_sha256") == PINS["launch_plan"]
            and release.get("native_source_archive_sha256") == PINS["source_archive"]
            and launch.get("release_sha256") == PINS["root_release"]
            and launch.get("container_id") == terminal.get("container_id") == CONTAINER
            and launch.get("host_pid") == 621414 and launch.get("results_root") == RESULTS,
            "native_gpu_chain_root_controller_join_failed")
    for document in (release, launch):
        require(equal(document.get("private_model_posts_authorized"), 0)
                and equal(document.get("maximum_autoregressive_requests"), 6),
                "native_gpu_chain_generated_only_scope_changed")
    require(release.get("runtime_image_sha256") == "sha256:" + IMAGE
            and equal(release.get("calls_per_arm"), 3) and equal(release.get("automatic_retries"), 0)
            and launch.get("automatic_retry") is False, "native_gpu_chain_root_scope_changed")
    state = terminal["state"]
    require(state.get("Running") is False and equal(state.get("Pid"), 0)
            and equal(state.get("ExitCode"), 0) and state.get("OOMKilled") is False
            and terminal.get("controller_failed_receipt_file_present") is False
            and terminal.get("final_pilot_receipt_file_present") is True
            and equal(terminal.get("call_attempt_file_count"), 6) and equal(terminal.get("call_receipt_file_count"), 6)
            and equal(terminal.get("gpu_mutations_by_collector"), 0) and equal(terminal.get("model_calls_by_collector"), 0),
            "native_gpu_chain_controller_not_successfully_terminal")
    inventory = terminal["files"]
    raw_files = receipt["terminal_files"]
    require(isinstance(inventory, dict) and len(inventory) == 59
            and isinstance(raw_files, dict) and len(raw_files) == 54
            and set(raw_files) == set(inventory) - OMITTED_INPUT_FILES,
            "native_gpu_chain_54_raw_proofs_required")
    proofs = {}
    for name, descriptor in raw_files.items():
        document, raw = _raw(descriptor)
        pin = inventory[name]
        require(pin["path"] == RESULTS + "/" + name and descriptor["sha256"] == pin["sha256"]
                and type(pin["size_bytes"]) is int and len(raw) == pin["size_bytes"],
                "native_gpu_chain_terminal_raw_proof_join_failed")
        proofs[name] = document
    execution = proofs["execution.attempt.json"]
    require(execution.get("root_release_id") == ROOT_RELEASE_ID
            and execution.get("runtime_image_sha256") == IMAGE
            and execution.get("source_manifest_sha256") == SOURCE_MANIFEST
            and execution.get("worker_extension_sha256") == WORKER
            and execution.get("worker_extension_class") == "serving.qwen38_vllm_worker.NativeAuditWorkerExtension"
            and equal(execution.get("max_generated_requests"), 6) and execution.get("equivalent_to_hf_nf4") is False,
            "native_gpu_chain_execution_identity_changed")
    pre = proofs["preengine.bytes.json"]
    require(pre.get("status") == "passed" and equal(pre.get("base_files_verified"), 32)
            and pre.get("adapter_tree_sha256") == ADAPTER and equal(pre.get("adapter_files_verified"), 3)
            and pre.get("worker_extension_sha256") == WORKER and equal(pre.get("model_calls"), 0),
            "native_gpu_chain_full_byte_preflight_missing")
    adapter = _one(proofs["adapter.loaded-proof.json"], "native_gpu_chain_one_adapter_worker_required")
    required = {"status": "passed", "logical_targets_verified": 496, "original_tensors_verified": 992,
        "physical_wrappers_verified": 304, "physical_output_slices_verified": 592, "adapter_id": 1,
        "active_slot": 0, "checkpoint_dtype": "F32", "execution_dtype": "BF16", "b_scale_folded": 2,
        "loaded_cpu_and_active_gpu_exact_equality": True, "no_partial_adapter_allowed": True,
        "no_model_forward_by_probe": True, "arithmetic_equivalent_to_hf_nf4": False,
        "wrapper_classes": {"MergedColumnParallelLinearWithLoRA": 160,
            "RowParallelLinearWithLoRA": 128, "MergedQKVParallelLinearWithLoRA": 16}}
    require(all(equal(adapter.get(k), v) for k, v in required.items()),
            "native_gpu_chain_all992_cpu_and_gpu_proof_missing")
    for name in ("runtime.before.json", "runtime.after.json"):
        runtime = _one(proofs[name], "native_gpu_chain_one_runtime_worker_required")
        required = {"model_class": "Qwen3_5ForConditionalGeneration", "base_dtype": "BF16", "quantization": None,
            "gdn_layers": 48, "gpu_compute_capability": [8, 0], "cuda_version": "12.9",
            "language_layer_count": 64, "full_attention_layers": 16, "linear_attention_layers": 48,
            "active_adapter_ids": [1], "kernel_execution_profiled": False}
        require(all(equal(runtime.get(k), v) for k, v in required.items()),
                "native_gpu_chain_loaded_runtime_changed")
    grammar = proofs["native-grammar.audit.json"]
    expected_tests = {name: True for name in ("all_unclear_allowed", "all_present_allowed", "all_absent_allowed",
        "duplicate_rejected", "missing_rejected", "unknown_rejected", "extra_rejected", "noncanonical_rejected",
        "prose_rejected", "duplicate_json_key_rejected", "fresh_matcher")}
    require(grammar.get("backend") == "vllm.v1.structured_output.backend_xgrammar.XgrammarBackend"
            and grammar.get("schema_sha256") == DERIVED_SCHEMA and equal(grammar.get("vocab_size"), 248320)
            and equal(grammar.get("model_calls"), 0) and equal(grammar.get("tests"), expected_tests)
            and equal(grammar.get("stop_tokens"), {"248046": True, "248044": True})
            and equal(grammar.get("native_json_separators"), [", ", ": "]),
            "native_gpu_chain_native_grammar_gate_missing")
    engine = proofs["engine.initialization.json"]
    require(equal(engine.get("autoregressive_requests"), 0)
            and engine.get("startup_profiling_forwards_not_counted_as_requests") is True
            and _number(engine.get("elapsed_seconds"), 1200), "native_gpu_chain_engine_startup_invalid")
    phases = [proofs[f"phase-{index:03d}.json"] for index in range(1, 35)]
    previous = (0, 0, 0)
    for index, phase in enumerate(phases, 1):
        counts = tuple(phase.get(k) for k in ("generation_calls_entered", "generation_calls_returned", "completions_validated"))
        require(equal(phase.get("phase_index"), index) and phase.get("root_release_id") == ROOT_RELEASE_ID
                and phase.get("output_owned") is True and all(type(n) is int and 0 <= n <= 6 for n in counts)
                and counts[0] >= counts[1] >= counts[2] and all(a <= b for a, b in zip(previous, counts)),
                "native_gpu_chain_durable_phase_counts_invalid")
        if counts[0]:
            require(phase.get("engine_initialized") is True and phase.get("adapter_gate_passed") is True,
                    "native_gpu_chain_generation_preceded_adapter_gate")
        previous = counts
    require(phases[-1]["phase"] == "completed" and previous == (6, 6, 6),
            "native_gpu_chain_controller_incomplete")
    pilot = proofs["pilot.receipt.json"]
    require(pilot.get("status") == "passed" and equal(pilot.get("generated_requests"), 6)
            and pilot.get("root_release_id") == ROOT_RELEASE_ID and pilot.get("private_quality_measured") is False
            and pilot.get("prefill_decode_kernel_time_measured") is False
            and pilot.get("arithmetic_equivalent_to_hf_nf4") is False
            and isinstance(pilot.get("calls"), list) and len(pilot["calls"]) == 6,
            "native_gpu_chain_six_genuine_generated_calls_required")
    for index, call in enumerate(pilot["calls"]):
        arm, repeat = ("base" if index % 2 == 0 else "fit"), index // 2
        attempt, returned = proofs[f"call-{index}.attempt.json"], proofs[f"call-{index}.receipt.json"]
        require(equal(call, returned) and returned.get("arm") == attempt.get("arm") == arm
                and equal(returned.get("repeat"), repeat) and equal(attempt.get("repeat"), repeat)
                and attempt.get("request_sha256") == inventory[f"request.{arm}.json"]["sha256"]
                and equal(attempt.get("max_tokens"), 4096) and equal(attempt.get("seed"), 20261017)
                and attempt.get("retry") is False and returned.get("finish_reason") == "stop"
                and equal(returned.get("actual_lora_int_id"), None if arm == "base" else 1)
                and returned.get("strict_original_schema_and_semantics") is True
                and _number(returned.get("elapsed_seconds"), 240)
                and type(returned.get("prompt_tokens")) is int and returned["prompt_tokens"] > 0
                and type(returned.get("completion_tokens")) is int and 0 < returned["completion_tokens"] <= 4096
                and isinstance(returned.get("output_sha256"), str)
                and re.fullmatch(r"[a-f0-9]{64}", returned["output_sha256"])
                and equal(returned.get("num_cached_tokens"), 0),
                "native_gpu_chain_call_attempt_receipt_join_failed")
    return {"status": "passed", "scope": "historical_offline_sequence_one", "generated_calls": 6,
        "model_id": "Qwen/Qwen3.8-27B", "model_revision": REVISION, "model_manifest_sha256": BASE,
        "runtime_image_sha256": IMAGE, "adapter_sha256": ADAPTER, "adapter_weights_sha256": WEIGHTS,
        "historical_source_archive_sha256": PINS["source_archive"], "historical_source_manifest_sha256": SOURCE_MANIFEST,
        "historical_container_id": CONTAINER, "root_release_sha256": PINS["root_release"],
        "terminal_safe_receipt_sha256": PINS["terminal"], "logical_targets_verified": 496,
        "original_tensors_verified": 992, "physical_wrappers_verified": 304, "base_files_verified": 32,
        "base_dtype": "BF16", "base_quantization": None, "max_num_seqs": 1,
        "does_not_prove_current_HTTP_engine": True, "does_not_prove_current_profile": True,
        "does_not_prove_parallel4": True, "actual_HTTP_or_current_engine_calls_by_validator": 0,
        "private_quality_measured": False, "execution_or_retry_authorized": False,
        "chain_receipt_sha256": digest(canonical(receipt))}


def prepare_gpu_gate_chain(paths):
    require(isinstance(paths, dict) and set(paths) == {"terminal_dir", "launch_plan", "source_archive", "root_release", "root_launch"},
            "native_gpu_chain_exact_input_paths_required")
    def descriptor(path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 * 1024 * 1024,
                "native_gpu_chain_raw_input_invalid")
        raw = path.read_bytes()
        return {"raw_utf8": raw.decode("utf8"), "sha256": digest(raw)}
    terminal_dir = Path(paths["terminal_dir"])
    terminal = descriptor(terminal_dir / "terminal.safe.receipt.json")
    data, _ = _raw(terminal, PINS["terminal"])
    archive = Path(paths["source_archive"])
    require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size <= 4 * 1024 * 1024,
            "native_gpu_chain_archive_input_invalid")
    raw_archive = archive.read_bytes()
    receipt = {"schema_version": SCHEMA, "artifacts": {"terminal": terminal,
        **{name: descriptor(paths[name]) for name in ("launch_plan", "root_release", "root_launch")}},
        "source_archive": {"raw_base64": base64.b64encode(raw_archive).decode("ascii"),
            "sha256": digest(raw_archive), "size_bytes": len(raw_archive)},
        "terminal_files": {name: descriptor(terminal_dir / name) for name in data["files"] if name not in OMITTED_INPUT_FILES}}
    validate_gpu_gate_chain(receipt)
    return receipt
