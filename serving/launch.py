"""Supervise pinned vLLM, verify local weights and gate readiness on warmup."""

from __future__ import annotations

import argparse
import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time

from serving.readiness import prewarm, publish_ready, request_local


def build_command(config: dict) -> list[str]:
    model_path = os.environ.get("FORTE_MODEL_PATH", config["model_path"])
    if not (Path(model_path).is_absolute() or PurePosixPath(model_path).is_absolute()):
        raise ValueError("model path must be absolute and local")
    return [
        "vllm", "serve", model_path,
        "--served-model-name", config["served_model_name"],
        "--host", config["host"], "--port", str(config["port"]),
        "--revision", config["model_revision"],
        "--tokenizer-revision", config["model_revision"],
        "--dtype", config["dtype"],
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
        "--max-model-len", str(config["max_model_len"]),
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--limit-mm-per-prompt", json.dumps(config["limit_mm_per_prompt"]),
        "--mm-processor-kwargs", json.dumps(config["mm_processor_kwargs"]),
        "--mm-processor-cache-gb", str(config["mm_processor_cache_gb"]),
        "--no-skip-mm-profiling", "--no-enable-prefix-caching",
        "--generation-config", "vllm",
        "--override-generation-config", json.dumps({"max_new_tokens": config["generation_max_tokens"]}),
        "--no-enable-log-requests", "--no-enable-log-outputs", "--no-enable-log-deltas",
        "--disable-uvicorn-access-log", "--disable-fastapi-docs",
        "--no-trust-request-chat-template", "--no-trust-request-mm-kwargs",
        "--middleware", "serving.security.DataOnlyMiddleware",
        "--structured-outputs-config", json.dumps(config["structured_outputs_config"]),
        "--enable-per-request-metrics",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned private local model launcher")
    parser.add_argument("--config", type=Path, default=Path("configs/serving/baseline.json"))
    parser.add_argument("--state", type=Path, default=Path("/run/forte/ready.json"))
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    command = build_command(config)
    if args.print_command:
        # No API keys or actual private inputs appear in command arguments.
        print(json.dumps(command))
        return 0
    args.state.unlink(missing_ok=True)
    if version("vllm").split("+", 1)[0] != config["vllm_version"]:
        raise RuntimeError("installed vLLM does not match the pinned config")
    model_path = os.environ.get("FORTE_MODEL_PATH", config["model_path"])
    if not Path(model_path).is_dir():
        raise RuntimeError("verified local model snapshot is unavailable")
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                "VLLM_NO_USAGE_STATS": "1", "VLLM_MEDIA_URL_ALLOW_REDIRECTS": "0",
                "VLLM_LOGGING_LEVEL": "WARNING",
                "FORTE_SERVING_CONFIG": str(args.config.resolve()),
                "FORTE_SERVING_STATE": str(args.state.resolve())})
    # A07 verifier checks every declared upstream SHA256 and required file.
    print(json.dumps({"event": "model_snapshot_verification_started"}), flush=True)
    verification_started = time.perf_counter()
    verification_result = subprocess.run(
        [sys.executable, "-m", "forte.model", "verify", "--snapshot", model_path,
         "--manifest", "/app/models/manifest.json"],
        env=env, check=True, capture_output=True, text=True,
    )
    verification = json.loads(verification_result.stdout)
    if (verification.get("verified") is not True
            or verification.get("revision") != config["model_revision"]
            or verification.get("processor_revision") != config["model_revision"]
            or verification.get("model_id") != config["model_id"]):
        raise RuntimeError("local verified snapshot differs from the selected baseline")
    print(json.dumps({"event": "model_snapshot_verified", "files": verification["files_verified"],
                      "bytes": verification["bytes_verified"],
                      "elapsed_seconds": time.perf_counter() - verification_started}), flush=True)
    server = subprocess.Popen(command, env=env)

    def stop(*_):
        args.state.unlink(missing_ok=True)
        if server.poll() is None:
            server.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deadline = time.monotonic() + config["startup_timeout_seconds"]
        while True:
            if server.poll() is not None:
                return server.returncode or 1
            if time.monotonic() >= deadline:
                raise TimeoutError("local model startup deadline exceeded")
            try:
                request_local(config["port"], "/health")
                break
            except (OSError, ValueError):
                time.sleep(1)
        print(json.dumps({"event": "model_engine_healthy", "prewarm_completed": False}), flush=True)
        warmup = prewarm(config)
        print(json.dumps({"event": "model_shapes_warmed", "synthetic_requests": len(warmup)}), flush=True)
        # Exercise the actual adapter and A08 output schema/semantic validator
        # on generated media before opening the warm readiness gate. This
        # bootstrap bypass is local to synthetic smoke; production still checks
        # warm readiness for every provider connection.
        from forte.model.smoke import synthetic_smoke
        schema_smoke = asyncio.run(synthetic_smoke(
            base_url=f"http://127.0.0.1:{config['port']}/v1",
            media_root="/tmp/forte-model-smoke",
            timeout_sec=min(config["request_timeout_seconds"], 120),
            api_key=os.environ.get("VLLM_API_KEY"),
            require_readiness=False,
        ))
        if server.poll() is not None:
            raise RuntimeError("model process stopped during prewarm")
        publish_ready(config, args.state, server.pid, warmup,
                      verification=verification, schema_smoke=schema_smoke)
        print(json.dumps({"event": "model_ready", "revision": config["model_revision"],
                          "synthetic_shapes_warmed": len(warmup), "gpu_capacity_measured": False}), flush=True)
        return server.wait()
    finally:
        stop()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # A public error type suffices; exception messages may contain inputs.
        code = getattr(error, "code", None)
        if type(error).__module__ == "forte.model.adapter":
            # ModelError messages are static adapter-owned codes by contract.
            code = str(error)
        if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]{1,100}", code):
            code = None
        print(json.dumps({"event": "model_start_failed", "error_type": type(error).__name__,
                          "error_code": code}), flush=True)
        raise SystemExit(1)
