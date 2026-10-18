"""Record offline artifact identity for a Gemma experiment; never predicts.

Run this in the same source checkout used by an experiment. It reads code,
configuration, Hugging Face download metadata, and explicitly selected model
artifacts only. It never imports the analyzer, reads videos/labels/reports, uses
the network, inspects hosts, or copies model text. Hashes are audit identifiers,
never model inputs. A manifest observes files at collection time; it is not proof
that an earlier run used them.

Base tensors are expensive to reread. Supply --hash-base-tensors to verify them,
or --base-tensor-hashes with an earlier structured filename:SHA256 inventory.
Imported hashes are explicitly marked recorded_not_rechecked. Adapter bytes are
always hashed when --adapter-dir is supplied. All omitted identities/counts are
explicitly unknown. --base-only explicitly declares no adapter.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SCHEMA = "forte.gemma4-run-manifest.v1"
BASE_REPO = "google/gemma-4-31B-it"
BASE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
REV_RE = re.compile(r"[0-9a-f]{40}\Z")
BASE_TENSOR_RE = re.compile(r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors\Z")
IMAGE_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
SMALL_MODEL_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "tokenizer_config.json", "chat_template.jinja",
    "model.safetensors.index.json",
)
DEPENDENCY_ROLES = ("vllm", "evaluator", "trainer")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Fail if a selected artifact changes while it is being hashed."""
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("artifact_changed_during_hash")
    return h.hexdigest()


def load_object(path: Path, limit: int = 4 * 1024 * 1024) -> dict:
    if path.stat().st_size > limit:
        raise ValueError("oversize_json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("object_required")
    return value


def hash_inventory(files: dict[str, str]) -> dict:
    return {"files_sha256": dict(sorted(files.items())),
            "root_sha256": canonical_hash(files) if files else None}


def base_identity(directory: Path, hashes: Path | None, verify: bool) -> dict:
    if not directory.is_dir():
        raise FileNotFoundError("base_directory_required")
    counts: Counter[str] = Counter()
    metadata = sorted((directory / ".cache/huggingface/download").rglob("*.metadata"))
    if len(metadata) > 20000:
        raise ValueError("too_many_metadata_files")
    for path in metadata:
        with path.open(encoding="utf-8") as stream:
            revision = stream.readline(128).strip()
        if REV_RE.fullmatch(revision):
            counts[revision] += 1
    config = directory / "config.json"
    if config.is_file():
        revision = load_object(config).get("_commit_hash")
        if isinstance(revision, str) and REV_RE.fullmatch(revision):
            counts[revision] += 1
    observed = next(iter(counts)) if len(counts) == 1 else None
    small = {name: file_hash(directory / name) for name in SMALL_MODEL_FILES
             if (directory / name).is_file()}
    tensor_paths = sorted(p for p in directory.iterdir()
                          if p.is_file() and BASE_TENSOR_RE.fullmatch(p.name))
    tensor_hashes: dict[str, str] = {}
    imported_sha256 = None
    status = "unknown_not_hashed"
    if verify:
        if not tensor_paths:
            raise ValueError("base_tensor_files_missing")
        tensor_hashes = {p.name: file_hash(p) for p in tensor_paths}
        status = "content_verified_at_collection"
    elif hashes is not None:
        inventory = load_object(hashes)
        if (not inventory or not all(isinstance(k, str) and BASE_TENSOR_RE.fullmatch(k)
                and isinstance(v, str) and SHA_RE.fullmatch(v) for k, v in inventory.items())):
            raise ValueError("invalid_base_tensor_inventory")
        if set(inventory) != {p.name for p in tensor_paths}:
            raise ValueError("base_tensor_inventory_mismatch")
        tensor_hashes = inventory
        imported_sha256 = file_hash(hashes)
        status = "recorded_not_rechecked"
    return {
        "repo": BASE_REPO, "expected_revision": BASE_REVISION,
        "observed_revision": observed,
        "revision_status": "single_observed" if observed else "mixed" if counts else "unknown",
        "local_metadata_revision_counts": dict(sorted(counts.items())),
        "expected_revision_matches": observed == BASE_REVISION if observed else None,
        "download_revision_pin_verified": False,
        "small_files": hash_inventory(small),
        "tensor_files": {**hash_inventory(tensor_hashes), "status": status,
                         "file_count_on_disk": len(tensor_paths),
                         "imported_inventory_sha256": imported_sha256},
    }


def adapter_identity(directory: Path | None, base_only: bool, receipt: Path | None,
                     base: dict) -> dict:
    if base_only:
        return {"status": "explicitly_none", "root_sha256": None}
    if directory is None:
        return {"status": "unknown_not_supplied", "root_sha256": None}
    expected = ("adapter_config.json", "adapter_model.safetensors")
    if not directory.is_dir() or not all((directory / name).is_file() for name in expected):
        raise ValueError("adapter_artifacts_missing")
    # Export contract is unsharded. Never silently omit extra adapter tensors.
    tensors = {p.name for p in directory.glob("adapter_model*") if p.is_file()}
    if tensors != {"adapter_model.safetensors"}:
        raise ValueError("unsupported_adapter_tensor_inventory")
    files = {name: file_hash(directory / name) for name in expected}
    out = {"status": "content_verified_at_collection", **hash_inventory(files),
           "tensor_bytes": (directory / "adapter_model.safetensors").stat().st_size,
           "receipt": {"status": "unknown_not_supplied", "sha256": None}}
    if receipt is None:
        return out
    value = load_object(receipt)
    adapter = value.get("adapter")
    if value.get("schema") != "forte.gemma4-adapter-export.v1" or not isinstance(adapter, dict):
        raise ValueError("invalid_adapter_receipt")
    if (adapter.get("config_sha256") != files["adapter_config.json"]
            or adapter.get("weights_sha256") != files["adapter_model.safetensors"]
            or value.get("adapter_sha256") != out["root_sha256"]):
        raise ValueError("adapter_receipt_content_mismatch")
    receipt_base = value.get("base")
    if (not isinstance(receipt_base, dict) or receipt_base.get("repo") != BASE_REPO
            or receipt_base.get("revision") != BASE_REVISION
            or receipt_base.get("config_sha256") != base["small_files"]["files_sha256"].get("config.json")):
        raise ValueError("adapter_receipt_base_mismatch")
    projected = {"status": "current_artifact_hashes_match", "sha256": file_hash(receipt)}
    observed_weights = base["tensor_files"]["files_sha256"]
    projected["base_tensor_inventory_matches"] = (
        receipt_base.get("weights") == observed_weights if observed_weights else None)
    if projected["base_tensor_inventory_matches"] is False:
        raise ValueError("adapter_receipt_base_tensors_mismatch")
    for key in ("tensor_keys_sha256", "tensors_sha256"):
        v = adapter.get(key)
        if not isinstance(v, str) or not SHA_RE.fullmatch(v):
            raise ValueError("invalid_adapter_receipt_hash")
        projected[key] = v
    for key in ("parameter_count", "tensor_count", "rank"):
        v = adapter.get(key)
        if type(v) is not int or v < 0:
            raise ValueError("invalid_adapter_receipt_count")
        projected[key] = v
    out["receipt"] = projected
    return out


def source_identity(root: Path) -> dict:
    paths = set(root.glob("src/forte/fast/*.py")) | set(root.glob("tools/*.py"))
    if not (root / "src/forte/fast/prompts.py").is_file():
        raise ValueError("prompt_source_missing")
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths) if p.is_file()}
    dependencies = {name: file_hash(root / name) for name in ("pyproject.toml", "uv.lock")
                    if (root / name).is_file()}
    return {**hash_inventory(files),
            "prompts_sha256": files["src/forte/fast/prompts.py"],
            "dependency_definitions": hash_inventory(dependencies)}


def count_arg(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("nonnegative_integer_required")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    base = parser.add_mutually_exclusive_group()
    base.add_argument("--hash-base-tensors", action="store_true")
    base.add_argument("--base-tensor-hashes", type=Path)
    adapter = parser.add_mutually_exclusive_group()
    adapter.add_argument("--adapter-dir", type=Path)
    adapter.add_argument("--base-only", action="store_true")
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--serving-payload", type=Path)
    parser.add_argument("--training-script", type=Path)
    parser.add_argument("--dependency-image", action="append", default=[], metavar="ROLE=SHA256_ID")
    parser.add_argument("--chosen-control-count", type=count_arg)
    parser.add_argument("--label-tuned-parameter-count", type=count_arg)
    parser.add_argument("--training-hyperparameter-count", type=count_arg)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.adapter_receipt is not None and args.adapter_dir is None:
        raise ValueError("adapter_receipt_requires_adapter")
    images = {role: {"status": "unknown_not_supplied", "image_id": None}
              for role in DEPENDENCY_ROLES}
    seen = set()
    for raw in args.dependency_image:
        role, equal, digest = raw.partition("=")
        if not equal or role not in images or role in seen or not IMAGE_RE.fullmatch(digest):
            raise ValueError("invalid_dependency_image")
        seen.add(role)
        images[role] = {"status": "caller_recorded_not_inspected", "image_id": digest}
    # Never overwrite any input or anything inside the selected model trees.
    output = args.out.expanduser().resolve()
    if output.exists():
        raise ValueError("manifest_output_already_exists")
    inputs = [args.config, args.base_tensor_hashes, args.adapter_receipt,
              args.serving_payload, args.training_script]
    directories = [args.base_model_dir, args.adapter_dir, args.source_root / "src",
                   args.source_root / "tools"]
    if (any(p is not None and output == p.expanduser().resolve() for p in inputs)
            or any(p is not None and output.is_relative_to(p.expanduser().resolve()) for p in directories)):
        raise ValueError("output_overlaps_input")
    record = {
        "schema": SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "artifact_observation_at_collection_time_not_historical_run_attestation",
        "manifest_identifiers_are_prediction_inputs": False,
        "run_prediction_inputs_verified": False,
        "heldout_accuracy_claim": False, "cross_validation_claim": False,
        "base": base_identity(args.base_model_dir.expanduser(), args.base_tensor_hashes,
                              args.hash_base_tensors),
        "source": source_identity(args.source_root), "config_sha256": file_hash(args.config),
        "serving_payload_sha256": file_hash(args.serving_payload) if args.serving_payload else None,
        "training_script_sha256": file_hash(args.training_script) if args.training_script else None,
        "dependency_images": images,
        "parameter_accounting": {
            "chosen_control_count": args.chosen_control_count,
            "label_tuned_parameter_count": args.label_tuned_parameter_count,
            "training_hyperparameter_count": args.training_hyperparameter_count,
            "count_status": "unknown_for_null_fields_caller_declared_for_numbers",
            "learned_adapter_parameter_count": 0 if args.base_only else None,
        },
    }
    record["adapter"] = adapter_identity(args.adapter_dir, args.base_only,
                                         args.adapter_receipt, record["base"])
    record["parameter_accounting"]["learned_adapter_parameter_count"] = (
        0 if args.base_only else record["adapter"].get("receipt", {}).get("parameter_count"))
    record["manifest_content_sha256"] = canonical_hash(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".gemma4-manifest-", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, output)
    print(json.dumps({"status": "written", "manifest_file_sha256": file_hash(output),
                      "manifest_content_sha256": record["manifest_content_sha256"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        allowed = {"FileNotFoundError", "PermissionError", "ValueError", "TypeError",
                   "OSError", "UnicodeDecodeError", "JSONDecodeError"}
        name = type(exc).__name__
        print(json.dumps({"status": "failed", "error_class": name if name in allowed else "OtherError"}))
        raise SystemExit(1)
