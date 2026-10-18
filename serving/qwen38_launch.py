"""Launch the separately pinned private Qwen3.8 base-plus-adapter candidate.

--verify-only performs hash/receipt checks and never initializes CUDA. Actual
readiness requires loading the adapter, comparing its tensors and completing
generated-image warmup plus the application's real decision-schema smoke.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import threading
import time

from serving.qwen38_identity import (CandidateError, canonical_sha256, load_expected_runtime_identity,
                                      load_json, require, verify_candidate)


def private_environment() -> None:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                       "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
                       "VLLM_NO_USAGE_STATS": "1"})
    # Environment proxies must not introduce a media/control egress route.
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(name, None)


def verify_private_namespace(route_path: Path = Path("/proc/net/route")) -> None:
    """Fail outside the Linux isolated runtime or with an active default route.

    This namespace check supplements the compose internal network. Deployment
    still measures no-egress behavior independently; it is not a firewall audit.
    """
    require(route_path.is_file(), "candidate_linux_network_namespace_required")
    for line in route_path.read_text(encoding="ascii").splitlines()[1:]:
        row = line.split()
        require(len(row) >= 8, "candidate_route_table_invalid")
        require(not (row[1] == "00000000" and int(row[3], 16) & 1),
                "candidate_default_network_route_forbidden")
    ipv6 = route_path.with_name("ipv6_route")
    if ipv6.is_file():
        for line in ipv6.read_text(encoding="ascii").splitlines():
            row = line.split()
            if len(row) >= 10:
                require(not (row[0] == "0" * 32 and row[1] == "00"
                             and int(row[8], 16) & 1 and not int(row[8], 16) & 0x200),
                        "candidate_default_ipv6_route_forbidden")


def verify_runtime_packages(config: dict) -> None:
    for name, expected in config["package_versions"].items():
        require(version(name) == expected, "candidate_installed_package_mismatch")
    # This is a deployment-supplied immutable image identity assertion. The
    # release receipt must independently inspect Docker's actual image digest.
    require(os.environ.get("FORTE_RUNTIME_IMAGE_SHA256") == config["runtime_image_sha256"],
            "candidate_runtime_image_assertion_mismatch")


def write_state(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(config: dict, identity: dict, config_path: Path, state_path: Path) -> int:
    import uvicorn
    from serving.qwen38_service import CandidateRuntime, create_app
    from serving.readiness import prewarm, request_local
    from forte.model.smoke import synthetic_smoke

    private_environment()
    verify_private_namespace()
    verify_runtime_packages(config)
    runtime = CandidateRuntime(config, identity)
    print(json.dumps({"event": "candidate_load_started", "backend": config["backend"]}), flush=True)
    runtime.load()
    print(json.dumps({"event": "candidate_adapter_loaded_and_tensor_verified",
                      "adapter_tensor_count": runtime.adapter_tensor_count}), flush=True)
    server = uvicorn.Server(uvicorn.Config(create_app(runtime), host=config["host"], port=config["port"],
                                           access_log=False, log_level="critical"))
    thread = threading.Thread(target=server.run, name="forte-candidate-http", daemon=True)
    thread.start()

    def stop(*_):
        runtime.ready = False
        state_path.unlink(missing_ok=True)
        server.should_exit = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            require(thread.is_alive() and time.monotonic() < deadline, "candidate_http_start_failed")
            time.sleep(.1)
        request_local(config["port"], "/health")
        warm_config = {**config, "request_timeout_seconds": config["limits"]["request_timeout_seconds"],
                       "prewarm_concurrency": 1}
        warmup = prewarm(warm_config)
        smoke = asyncio.run(synthetic_smoke(
            base_url=f"http://127.0.0.1:{config['port']}/v1", media_root="/tmp/forte-qwen38-smoke",
            timeout_sec=config["limits"]["request_timeout_seconds"], api_key=os.environ.get("VLLM_API_KEY"),
            require_readiness=False, model=config["served_model_name"], expected_revision=config["model_revision"],
            seed=20261017, chat_template_kwargs={"enable_thinking": False, "preserve_thinking": False},
            expected_runtime_identity=load_expected_runtime_identity(config_path),
            max_tokens=config["generation"]["max_new_tokens"]))
        require(smoke.get("synthetic_only") is True and smoke.get("real_local_inference") is True
                and smoke.get("schema_and_semantic_validation_passed") is True,
                "candidate_actual_schema_smoke_failed")
        require(thread.is_alive(), "candidate_http_stopped_during_warmup")
        runtime.ready = True
        record = {**runtime.readiness(), "server_pid": os.getpid(), "ready_at": datetime.now(timezone.utc).isoformat(),
                  "package_versions": config["package_versions"], "warmup": warmup, "schema_smoke": smoke,
                  "namespace_default_route_absent": True, "external_no_egress_probe_completed": False,
                  "image_digest_independently_inspected": False}
        write_state(state_path, record)
        print(json.dumps({"event": "candidate_ready", "adapter_sha256": identity["adapter_sha256"],
                          "synthetic_requests": len(warmup) + 1, "quality_or_latency_measured": False}), flush=True)
        while thread.is_alive():
            thread.join(timeout=1)
        return 0 if server.should_exit else 1
    finally:
        stop()
        thread.join(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("/run/forte/qwen38-ready.json"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check-ready", action="store_true", help="Check local actual warm readiness against every pin")
    parser.add_argument("--expectations-only", action="store_true",
                        help="Print expected public pins only; no artifact verification or CUDA")
    args = parser.parse_args()
    if args.check_ready:
        from serving.readiness import request_local
        expected = load_expected_runtime_identity(args.config)
        config = load_json(args.config)
        actual = request_local(config["port"], "/readyz")
        require(all(actual.get(key) == value for key, value in expected.items())
                and actual.get("ready") is True and actual.get("model_loaded") is True
                and actual.get("prewarm_completed") is True
                and type(actual.get("adapter_loaded_tensor_count")) is int
                and actual["adapter_loaded_tensor_count"] > 0, "candidate_not_ready_or_identity_mismatch")
        return 0
    if args.expectations_only:
        print(json.dumps({"expected_runtime_identity": load_expected_runtime_identity(args.config),
                          "artifacts_verified": False, "model_loaded": False}))
        return 0
    config = load_json(args.config)
    if not args.verify_only:
        args.state.unlink(missing_ok=True)
    private_environment()
    print(json.dumps({"event": "candidate_snapshot_verification_started"}), flush=True)
    identity = verify_candidate(config)
    if args.verify_only:
        print(json.dumps(identity, allow_nan=False))
        return 0
    return run(config, identity, args.config.resolve(), args.state)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"event": "candidate_start_failed", "reason_code": str(error)
                          if isinstance(error, CandidateError) else "candidate_runtime_error",
                          "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)

