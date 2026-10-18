"""Discover only actual vLLM metric names through the team's private container."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import inspect

from forte.observability.metrics import VLLM_AGGREGATE_METRICS, parse_vllm_metric_totals


SOURCE = '''
import json, math, re
import httpx
from forte.observability.metrics import discover_vllm_metric_names
PARSER_SOURCE
try:
    response = httpx.get("http://model:8000/metrics", timeout=10, trust_env=False)
    print(json.dumps({"status": "measured" if response.status_code == 200 else "failed",
                      "http_status": response.status_code,
                      "metric_names": discover_vllm_metric_names(response.text) if response.status_code == 200 else [],
                      "numeric_metric_totals": parse_vllm_metric_totals(response.text) if INCLUDE_TOTALS and response.status_code == 200 else {}}))
except Exception as error:
    print(json.dumps({"status": "failed", "error_type": type(error).__name__, "metric_names": []}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-alias", default="forte-a100-smoke-20261017")
    parser.add_argument("--worker-container", default="forte-worker-1")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-timing-aggregates", action="store_true")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) for value in (args.ssh_alias, args.worker_container)):
        parser.error("invalid team SSH/container reference")
    try:
        command = f"docker exec -i {args.worker_container} python -"
        parser_source = "VLLM_AGGREGATE_METRICS = " + repr(set(VLLM_AGGREGATE_METRICS)) + "\n" + inspect.getsource(parse_vllm_metric_totals)
        source = SOURCE.replace("PARSER_SOURCE", parser_source).replace("INCLUDE_TOTALS", repr(args.include_timing_aggregates))
        result = subprocess.run(["wsl", "-d", args.wsl_distro, "--", "ssh", "-T", "-o", "BatchMode=yes",
                                 "-o", "ConnectTimeout=12", args.ssh_alias, command], input=source,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0:
            report = {"status": "failed", "error_code": "team_ssh_or_container_command_failed", "metric_names": []}
        else:
            report = json.loads(result.stdout)
        report.update({"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "source": "actual private model:8000/metrics from team worker container",
                       "labels_exported": False, "numeric_aggregates_exported": args.include_timing_aggregates, "model_readiness_inferred": False,
                       "inference_latency_measured": False})
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__, "metric_names": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "metric_name_count": len(report["metric_names"])}))
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
