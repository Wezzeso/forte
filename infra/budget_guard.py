"""Stop the one authorized Brev instance at its approved runtime cutoff.

Runs from the existing WSL profile. It never copies credentials, deletes data,
creates instances, or operates on another organization/instance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

INSTANCE_NAME = "forte-a100-smoke-20261017"
INSTANCE_ID = "wd88a7vvt"
ORGANIZATION = "weezzeeso-cfe46c-tu9j"
CUTOFF = datetime.fromisoformat("2026-10-18T06:01:25+00:00")


def run_cli(*arguments: str, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(Path.home() / ".local/bin/brev"), "--org", ORGANIZATION, *arguments],
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def instance_status() -> str:
    result = run_cli("ls", "--json")
    if result.returncode:
        raise RuntimeError("provider_inventory_unavailable")
    payload = json.loads(result.stdout)
    found = [item for item in payload.get("workspaces", []) if item.get("id") == INSTANCE_ID]
    if len(found) != 1 or found[0].get("name") != INSTANCE_NAME:
        raise RuntimeError("authorized_instance_identity_mismatch")
    return str(found[0]["status"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    try:
        status = instance_status()
        if not args.execute or now < CUTOFF:
            print(json.dumps({"guard": "ready", "instance_id": INSTANCE_ID, "status": status,
                              "cutoff_utc": CUTOFF.isoformat(), "due": now >= CUTOFF,
                              "remaining_seconds": max(0, int((CUTOFF - now).total_seconds()))}))
            return 0
        if status in {"STOPPED", "DELETED", "STOPPING"}:
            print(json.dumps({"guard": "already_stopped_or_stopping", "instance_id": INSTANCE_ID, "status": status}))
            return 0
        result = run_cli("stop", INSTANCE_NAME, timeout=180)
        if result.returncode:
            raise RuntimeError("provider_stop_failed")
        for _ in range(12):
            status = instance_status()
            if status in {"STOPPED", "STOPPING"}:
                print(json.dumps({"guard": "stop_verified", "instance_id": INSTANCE_ID, "status": status,
                                  "storage_still_bills": True, "cleanup_owner": "project operator / root integration"}))
                return 0
            time.sleep(5)
        raise RuntimeError("provider_stop_not_verified")
    except Exception as error:
        # Never include raw provider output or exception text in logs.
        print(json.dumps({"guard": "failed", "instance_id": INSTANCE_ID, "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
