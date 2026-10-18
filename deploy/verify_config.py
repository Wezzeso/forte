"""Validate Compose boundaries without printing resolved configuration or secrets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.gpu and not os.environ.get("FORTE_MODEL_SNAPSHOT"):
        print(json.dumps({"compose_valid": False,
                          "error": "Set FORTE_MODEL_SNAPSHOT to the verified absolute local snapshot path before validating the GPU profile"}))
        return 1
    root = Path(__file__).resolve().parents[1]
    command = ["docker", "compose", "--env-file", str(root / "deploy/.env.example"),
               "-f", str(root / "deploy/compose.yaml")]
    if args.gpu:
        command.extend(["-f", str(root / "deploy/compose.gpu.yaml"), "--profile", "gpu"])
    command.extend(["config", "--format", "json"])
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if result.returncode:
        print(json.dumps({"compose_valid": False, "error": "Compose validation failed; inspect configuration paths locally"}))
        return 1
    config = json.loads(result.stdout)
    services = config["services"]
    errors: list[str] = []
    for name in ("db", "api", "worker", "migrate", "model"):
        service = services.get(name)
        if service is None:
            continue
        if service.get("ports"):
            errors.append(f"{name} publishes a host port")
        if set(service.get("networks", {})) != {"private"}:
            errors.append(f"{name} is not confined to the private network")
        if name != "model" and service.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices"):
            errors.append(f"{name} has an unexpected GPU reservation")
        for mount in service.get("volumes", []):
            if name != "model" and mount.get("type") == "bind":
                errors.append(f"{name} has a host bind mount")
    if not config["networks"]["private"].get("internal"):
        errors.append("private network permits external routing")
    required_volumes = {"forte_data", "postgres_data", "caddy_data", "caddy_config"}
    if not required_volumes.issubset(config.get("volumes", {})):
        errors.append("persistent volumes missing")
    if args.gpu:
        weights = [mount for mount in services["model"]["volumes"] if mount.get("target") == "/models/Qwen3-VL-8B-Instruct"]
        if len(weights) != 1 or not weights[0].get("read_only"):
            errors.append("model snapshot is not a single read-only mount")
    print(json.dumps({"compose_valid": True, "service_count": len(services),
                      "private_boundary_valid": not errors, "errors": errors}))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
