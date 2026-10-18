"""Run the local CPU checks and emit aggregate diagnostics only.

This uses the existing interpreter and installed dependencies. It never starts
GPU inference, downloads weights, changes deployment, or prints private results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SUITES = ("tests", "qa/integration", "qa/system", "annotations", "rules", "scripts", "serving/tests", "training")


def code_hashes() -> dict[str, str]:
    hashes = {}
    for folder in ("src", "tests", "qa", "annotations", "rules", "scripts", "serving", "training", "evaluation", "benchmarks/performance", "deploy"):
        for source in (ROOT / folder).rglob("*.py"):
            if not any(part in {"private", "runs", "__pycache__", ".venv"} for part in source.relative_to(ROOT).parts):
                hashes[source.relative_to(ROOT).as_posix()] = sha256(source.read_bytes()).hexdigest()
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="qwen38_cpu_final")
    args = parser.parse_args()
    if not args.name.startswith("qwen38_") or Path(args.name).name != args.name:
        parser.error("name must be a qwen38_ filename stem")
    destination = ROOT / "qa/integration"
    log = destination / f"{args.name}.log"
    junit = destination / f"{args.name}.xml"
    output = destination / f"{args.name}.json"
    command = [sys.executable, "-m", "pytest", *SUITES,
               "--ignore=qa/integration/private", "--import-mode=importlib", "-m", "not gpu", "-q",
               f"--junitxml={junit}"]
    environment = os.environ.copy()
    # Live PostgreSQL tests need separate explicit evidence; local CPU runs use
    # each fixture's isolated SQLite fallback rather than an inherited endpoint.
    environment.pop("FORTE_TEST_POSTGRES_URL", None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    # Legacy fixtures import sibling helpers by basename. Keep these two
    # helper roots explicit while importlib mode disambiguates test modules
    # such as qa/system and serving/tests test_grounding_pilot.py.
    environment["PYTHONPATH"] = os.pathsep.join(str(ROOT / name) for name in ("src", "tests", "scripts"))
    started_at = datetime.now(timezone.utc).isoformat()
    before = code_hashes()
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    after = code_hashes()
    summary: dict = {
        "schema_version": "forte.qwen38.cpu-qa.v1",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "interpreter": sys.executable,
        "suites": SUITES,
        "excluded_marker": "gpu",
        "environment": {"offline_model_downloads": True, "cuda_visible_devices": "", "live_postgresql_opt_in": False},
        "exit_code": completed.returncode,
        "code_snapshot_before_sha256": sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        "code_snapshot_after_sha256": sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
        "code_changed_during_check": sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name)),
        "counts": {}, "failures": [],
        "log": log.relative_to(ROOT).as_posix(),
        "junit": junit.relative_to(ROOT).as_posix(),
        "limitations": [
            "Synthetic/component software verification; no private model inference or GPU execution.",
            "Does not establish real-video accuracy, semantic onset precision, model training, deployment, or upload-to-ready latency.",
            "Live PostgreSQL behavior requires its own opt-in integration receipt.",
        ],
    }
    if junit.is_file():
        tree = ET.parse(junit)
        suites = list(tree.getroot().iter("testsuite"))
        summary["counts"] = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                             for key in ("tests", "failures", "errors", "skipped")}
        for case in tree.getroot().iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                summary["failures"].append({"class": case.attrib.get("classname"), "name": case.attrib.get("name")})
    log_text = log.read_text(encoding="utf-8")
    summary_lines = re.findall(r"(?m)^.*\bpassed\b.*\bin [0-9.]+s.*$", log_text)
    if summary_lines:
        summary["pytest_summary"] = summary_lines[-1]
        for label, pattern in (("passed_cases", r"(?<!subtests )(\d+) passed"),
                               ("passed_subtests", r"(\d+) subtests passed")):
            match = re.search(pattern, summary_lines[-1])
            summary["counts"][label] = int(match.group(1)) if match else 0
    summary["junit_test_count_includes_subtests"] = True
    summary["source_hashes"] = {
        name: sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("pyproject.toml", "src/forte/pipeline.py", "src/forte/config.py", "src/forte/worker.py")
        if (ROOT / name).is_file()
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": output.relative_to(ROOT).as_posix(), "exit_code": completed.returncode,
                      "counts": summary["counts"], "failures": summary["failures"]}, ensure_ascii=False))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
