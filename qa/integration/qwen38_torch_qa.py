"""Actual Torch CPU checks using the existing isolated training interpreter.

Only generated tensor/media fixtures and the staged public processor are used.
The existing CPU test runner is appended as a fallback import path; no package
is installed, upgraded, or copied into either virtual environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
TESTS = (
    "training/test_data_masks.py", "tests/test_silver_balance.py",
    "tests/test_qwen38_balanced_runner.py", "tests/test_qwen38_proof.py",
    "tests/test_nf4_freeze_diagnostic.py",
)
SOURCES = (
    "training/data.py", "training/silver_balance.py", "training/qwen38_silver.py",
    "training/qwen38_proof.py", "training/nf4_freeze_diagnostic.py",
    "training/nf4_initialization_proof.py", *TESTS,
)


def source_hashes(names):
    return {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="qwen38_torch_cpu_final")
    parser.add_argument("--tests", nargs="+", default=list(TESTS))
    args = parser.parse_args()
    if not args.name.startswith("qwen38_") or Path(args.name).name != args.name:
        parser.error("name must be a qwen38_ filename stem")
    if any(not (ROOT / name).resolve().is_relative_to(ROOT) or not (ROOT / name).is_file() for name in args.tests):
        parser.error("test paths must be existing workspace files")
    destination = ROOT / "qa/integration"
    stem = args.name
    log, junit, report = [destination / (stem + suffix) for suffix in (".log", ".xml", ".json")]
    interpreter = ROOT / "training/.venv/Scripts/python.exe"
    fallback = ROOT / ".venv/Lib/site-packages"
    # Keep the training environment's Torch/Transformers first in resolution.
    bootstrap = (
        "import json,sys,importlib.metadata; "
        "sys.path.append(sys.argv.pop(1)); "
        "print(json.dumps({'runtime_packages':{name:importlib.metadata.version(name) "
        "for name in ['torch','transformers','peft','pytest','safetensors']}}),flush=True); "
        "import pytest; raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       WANDB_DISABLED="true", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    environment.pop("FORTE_TEST_POSTGRES_URL", None)
    sources = tuple(sorted(set(SOURCES) | set(args.tests)))
    before = source_hashes(sources)
    start = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([str(interpreter), "-c", bootstrap, str(fallback), *args.tests,
            "--import-mode=importlib", "-q", f"--junitxml={junit}"], cwd=ROOT,
            env=environment, stdout=stream, stderr=subprocess.STDOUT)
    after = source_hashes(sources)
    counts, failures = {}, []
    if junit.is_file():
        tree = ET.parse(junit)
        counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in tree.getroot().iter("testsuite"))
                  for key in ("tests", "errors", "failures", "skipped")}
        failures = [{"class": case.attrib.get("classname"), "name": case.attrib.get("name")}
                    for case in tree.getroot().iter("testcase")
                    if case.find("failure") is not None or case.find("error") is not None]
    lines = log.read_text(encoding="utf-8").splitlines()
    runtime = next((json.loads(line) for line in lines if line.startswith('{"runtime_packages":')), {})
    summary = {"schema_version": "forte.qwen38.actual-torch-cpu-qa.v1",
        "started_at_utc": start, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.monotonic() - began, 3), "interpreter": str(interpreter),
        "pytest_fallback_import_path": str(fallback), "tests": args.tests,
        **runtime, "exit_code": completed.returncode, "counts": counts, "failures": failures,
        "pytest_summary": next((line for line in reversed(lines) if "passed" in line and " in " in line), None),
        "source_hashes_before": before, "source_hashes_after": after,
        "source_changed_during_check": [name for name in sources if before[name] != after[name]],
        "limitations": ["CPU tensor/processor and synthetic model fixtures only; no CUDA, NF4 kernel, private video or real 27B load.",
                        "The NF4 initialization unit check uses a fake Linear4bit boundary with real CPU tensors; actual GPU proof is separate."]}
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": report.relative_to(ROOT).as_posix(), "counts": counts, "failures": failures,
                      "exit_code": completed.returncode}))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
