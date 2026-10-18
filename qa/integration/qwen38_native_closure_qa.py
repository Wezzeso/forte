"""Focused generated checks for the final grammar/equivalence source closure."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "tmp/qwen38-ml-support-grammar-final-20261017T151420Z.tar.gz"
TESTS = (
    "serving/tests/test_qwen38_grammar.py", "serving/tests/test_qwen38_candidate.py",
    "serving/tests/test_qwen38_runtime_binding.py", "tests/test_api_candidate_readiness.py",
    "tests/test_candidate_pipeline_config.py", "tests/test_qwen38_quality_equivalence.py",
    "tests/test_qwen38_quality_family.py", "tests/test_qwen38_untuned.py",
    "tests/test_qwen38_operational_evaluation.py", "tests/test_release_packaging.py",
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="qwen38_native_closure_final")
    parser.add_argument("--tests", nargs="+", default=list(TESTS))
    args = parser.parse_args()
    if not args.name.startswith("qwen38_") or Path(args.name).name != args.name:
        parser.error("name must be a qwen38_ filename stem")
    if any(not (ROOT / name).resolve().is_relative_to(ROOT) or not (ROOT / name).is_file() for name in args.tests):
        parser.error("test paths must be existing workspace files")
    destination = ROOT / "qa/integration"
    stem = args.name
    log, junit, output = [destination / (stem + suffix) for suffix in (".log", ".xml", ".json")]
    manifest_path = Path(str(ARCHIVE) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    archive_digest = digest(ARCHIVE)
    archived = {}
    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                raise ValueError("QA source archive contains a non-regular member")
            archived[member.name] = sha256(bundle.extractfile(member).read()).hexdigest()
    if archived != files or archive_digest != "91263a59cb2e73af48b46166b427d507ce7c3fa05106a46402acec950c20048d":
        raise ValueError("QA source archive or manifest binding changed")
    names = tuple(sorted(set(files) | set(args.tests)))
    before = {name: digest(ROOT / name) for name in names}
    differing = [name for name, expected in files.items() if before[name] != expected]
    if differing:
        raise ValueError("Current public source differs from frozen archive: " + ", ".join(differing))
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(str(ROOT / name) for name in ("src", "tests", "scripts")))
    environment.pop("FORTE_TEST_POSTGRES_URL", None)
    started_at = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([sys.executable, "-m", "pytest", *args.tests,
            "--import-mode=importlib", "-m", "not gpu", "-q", f"--junitxml={junit}"],
            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    after = {name: digest(ROOT / name) for name in names}
    tree = ET.parse(junit)
    counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in tree.getroot().iter("testsuite"))
              for key in ("tests", "errors", "failures", "skipped")}
    failures = [{"class": case.attrib.get("classname"), "name": case.attrib.get("name")}
                for case in tree.getroot().iter("testcase")
                if case.find("failure") is not None or case.find("error") is not None]
    lines = log.read_text(encoding="utf-8").splitlines()
    summary = next((line for line in reversed(lines) if re.search(r"\bpassed\b.*\bin [0-9.]+s", line)), None)
    report = {"schema_version": "forte.qwen38.native-source-closure-qa.v1",
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.monotonic() - began, 3), "exit_code": completed.returncode,
        "interpreter": sys.executable, "tests": args.tests, "counts": counts,
        "failures": failures, "pytest_summary": summary,
        "archive": ARCHIVE.relative_to(ROOT).as_posix(), "archive_sha256": archive_digest,
        "manifest_sha256": digest(manifest_path), "archive_member_count": len(files),
        "all_archive_members_verified": True, "all_current_archive_sources_matched_before_checks": True,
        "source_hashes_before": before, "source_hashes_after": after,
        "source_changed_during_check": [name for name in names if before[name] != after[name]],
        "limitations": ["Focused CPU/generated grammar, provider, binding, equivalence and packaging tests only.",
            "No real model, tokenizer compilation on GPU, private inference, CUDA kernel or serving activation proof.",
            "Counts overlap earlier broad checks and are not additive."]}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": output.relative_to(ROOT).as_posix(), "exit_code": completed.returncode,
        "pytest_summary": summary, "failures": failures,
        "source_changed_during_check": report["source_changed_during_check"]}))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
