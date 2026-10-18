"""Read-only checks for a pinned fast-analyzer snapshot and its Linux GPU host.

No image pulls, package installs, model downloads or model generation occur.
The GPU check launches one temporary, network-isolated container and performs
a tiny CUDA allocation, then removes the container. Failures use static codes
so Docker diagnostics cannot accidentally expose host credentials.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
from typing import Any


RECEIPT_NAME = ".forte-model-receipt.json"
DEFAULT_REPO = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_IMAGE = "forte-model:vllm-0.31.0"


class PreflightError(RuntimeError):
    """A static, safe-to-publish check failure code."""


def _json_file(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise PreflightError(code) from None
    if not isinstance(value, dict):
        raise PreflightError(code)
    return value


def _model_file(root: Path, name: str) -> Path:
    # Apply both path grammars so a manifest checked on Windows is also safe
    # when copied to Linux. External HF-cache symlinks cannot be bind-mounted.
    if (not isinstance(name, str) or not name or "\\" in name
            or PureWindowsPath(name).drive or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts):
        raise PreflightError("snapshot_unsafe_file_path")
    path = root / name
    try:
        path.resolve(strict=True).relative_to(root)
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError
        with path.open("rb") as stream:
            stream.read(1)
    except (OSError, ValueError):
        raise PreflightError("snapshot_file_missing_or_unreadable") from None
    return path


def check_snapshot(model_dir: Path, revision: str, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise PreflightError("immutable_model_revision_required")
    root = model_dir.resolve()
    if not root.is_dir():
        raise PreflightError("snapshot_directory_missing")
    revision = revision.lower()
    receipt = _json_file(_model_file(root, RECEIPT_NAME), "snapshot_receipt_invalid")
    if (receipt.get("schema_version") != "forte.fast-model-snapshot.v1"
            or receipt.get("repo_id") != repo or receipt.get("revision") != revision):
        raise PreflightError("snapshot_revision_mismatch")
    config = _json_file(_model_file(root, "config.json"), "snapshot_config_invalid")
    if config.get("model_type") not in {"qwen3_vl", "qwen3_vl_moe"}:
        raise PreflightError("snapshot_model_type_unsupported")
    if config.get("auto_map"):
        raise PreflightError("snapshot_remote_code_not_allowed")
    required = {"config.json", "tokenizer_config.json", "preprocessor_config.json", RECEIPT_NAME}
    if (root / "tokenizer.json").is_file():
        required.add("tokenizer.json")
    else:
        required.update({"vocab.json", "merges.txt"})
    index_name = "model.safetensors.index.json"
    index = _json_file(_model_file(root, index_name), "snapshot_weight_index_invalid")
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights or any(not isinstance(v, str) for v in weights.values()):
        raise PreflightError("snapshot_weight_index_invalid")
    shards = set(weights.values())
    if any(not name.endswith(".safetensors") for name in shards):
        raise PreflightError("snapshot_weight_format_unsupported")
    for name in required:
        _model_file(root, name)
    weight_bytes = sum(_model_file(root, name).stat().st_size for name in shards)
    return {"repo_id": repo, "revision": revision, "weight_shards": len(shards),
            "weight_bytes": weight_bytes, "files_checked": len(required | shards | {index_name})}


def _run(command: list[str], code: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise PreflightError(code) from None
    if result.returncode:
        raise PreflightError(code)
    return result.stdout.strip()


def check_host(model_dir: Path, image: str, gpu: str, *, min_total_mib: int = 75000,
               min_free_mib: int = 70000) -> dict[str, Any]:
    if not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", gpu):
        raise PreflightError("single_gpu_id_required")
    if min_total_mib <= 0 or min_free_mib < 0:
        raise PreflightError("gpu_memory_requirement_invalid")
    raw = _run(["nvidia-smi", "-i", gpu, "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"], "host_gpu_unavailable")
    try:
        rows = raw.splitlines()
        if len(rows) != 1:
            raise ValueError
        name, total, free = [field.strip() for field in rows[0].split(",")]
        total_mib, free_mib = int(total), int(free)
    except ValueError:
        raise PreflightError("host_gpu_query_invalid") from None
    if total_mib < min_total_mib or free_mib < min_free_mib:
        raise PreflightError("host_gpu_memory_insufficient")
    _run(["docker", "info", "--format", "{{.ServerVersion}}"], "docker_daemon_unavailable")
    compose = _run(["docker", "compose", "version", "--short"], "docker_compose_unavailable")
    image_id = _run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                    "model_container_image_missing")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise PreflightError("model_container_image_identity_invalid")
    # Test the actual image, GPU runtime, CUDA/bf16 support, and read-only bind
    # from inside the container. No model is loaded and no network is attached.
    probe = """import json, pathlib, torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
x = torch.ones(1, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize()
root = pathlib.Path('/models/vlm')
names = ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', '.forte-model-receipt.json']
names += ['tokenizer.json'] if (root / 'tokenizer.json').is_file() else ['vocab.json', 'merges.txt']
for name in names:
    with (root / name).open('rb') as stream: assert stream.read(1)
index = json.loads((root / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    with (root / name).open('rb') as stream: assert stream.read(1)
print(json.dumps({'cuda': True, 'bf16': True, 'model_readable': True}))
"""
    command = ["docker", "run", "--rm", "--pull=never", "--network", "none",
               "--gpus", f"device={gpu}", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges:true", "--tmpfs", "/tmp:rw,nosuid,size=512m",
               "--mount", f"type=bind,src={model_dir.resolve()},dst=/models/vlm,readonly",
               "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
               "--env", "HF_HUB_DISABLE_TELEMETRY=1", "--env", "XDG_CACHE_HOME=/tmp/cache",
               "--entrypoint", "python3", image, "-c", probe]
    result = _run(command, "container_cuda_or_model_permission_failed", timeout=120)
    try:
        proof = json.loads(result.splitlines()[-1])
    except (ValueError, IndexError):
        raise PreflightError("container_probe_invalid") from None
    if proof != {"cuda": True, "bf16": True, "model_readable": True}:
        raise PreflightError("container_probe_invalid")
    return {"gpu_name": name, "gpu_total_mib": total_mib, "gpu_free_mib": free_mib,
            "container_image_id": image_id, "compose_version": compose, **proof}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("FORTE_FAST_MODEL_DIR"))
    parser.add_argument("--model-revision", default=os.environ.get("FORTE_FAST_MODEL_REVISION", ""))
    parser.add_argument("--model-repo", default=os.environ.get("FORTE_FAST_MODEL_REPO", DEFAULT_REPO))
    parser.add_argument("--image", default=os.environ.get("FORTE_MODEL_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--gpu", default=os.environ.get("FORTE_INFERENCE_GPU_ID", "0"))
    parser.add_argument("--min-total-vram-mib", type=int, default=75000)
    parser.add_argument("--min-free-vram-mib", type=int, default=70000)
    parser.add_argument("--snapshot-only", action="store_true", help="skip host/container checks; never means deploy-ready")
    args = parser.parse_args(argv)
    try:
        if args.model_dir is None:
            raise PreflightError("snapshot_directory_required")
        snapshot = check_snapshot(args.model_dir, args.model_revision, args.model_repo)
        host = None if args.snapshot_only else check_host(
            args.model_dir, args.image, args.gpu, min_total_mib=args.min_total_vram_mib,
            min_free_mib=args.min_free_vram_mib)
    except PreflightError as error:
        print(json.dumps({"ready": False, "error_code": str(error)}))
        return 1
    print(json.dumps({"ready": host is not None, "snapshot_valid": True, "snapshot": snapshot, "host": host}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
