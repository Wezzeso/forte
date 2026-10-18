"""Run a preselected proposed-train phase without exposing source paths or keys."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets

from forte.observability.benchmark import main as benchmark_main


def ensure_attempt_plan(path, items, *, trial_id, phase, mode):
    expected = [(item["sample_id"], f"s{index:03d}") for index, item in enumerate(items, 1)]
    if path.is_file():
        plan = json.loads(path.read_text(encoding="utf-8"))
        records = plan.get("records", [])
        if (plan.get("schema_version") != "forte.benchmark-attempt-plan.v1" or plan.get("trial_id") != trial_id
                or plan.get("phase") != phase or plan.get("mode") != mode
                or [(item.get("sample_id"), item.get("wire_sample_id")) for item in records] != expected
                or any(not re.fullmatch(r"attempt_[0-9a-f]{32}", item.get("attempt_id", "")) for item in records)
                or len({item["attempt_id"] for item in records}) != len(records)):
            raise ValueError("frozen attempt plan mismatch")
    else:
        plan = {"schema_version": "forte.benchmark-attempt-plan.v1", "trial_id": trial_id, "phase": phase, "mode": mode,
                "records": [{"sample_id": sample_id, "wire_sample_id": wire_id, "attempt_id": "attempt_" + secrets.token_hex(16)}
                            for sample_id, wire_id in expected]}
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as target:
                target.write(json.dumps(plan, indent=2))
        except FileExistsError:
            return ensure_attempt_plan(path, items, trial_id=trial_id, phase=phase, mode=mode)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "burst", "paired"), required=True)
    parser.add_argument("--selection", type=Path, default=Path("benchmarks/private/a18-inputs.json"))
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--trial-id", help="new isolated reused-content trial; existing artifacts are never overwritten")
    parser.add_argument("--repeat-content", action="store_true")
    parser.add_argument("--profile-label", help="technical profile label for the paired trial")
    parser.add_argument("--prepare-only", action="store_true", help="write an aggregate trial plan without API or model requests")
    parser.add_argument("--exposure-registry", type=Path, default=Path("benchmarks/private/model-exposure.jsonl"))
    parser.add_argument("--exposure-manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    args, forwarded = parser.parse_known_args()
    if any(value.split("=", 1)[0] in {"--files", "--mode", "--private-job-map", "--private-terminal-export", "--private-attempt-plan"} for value in forwarded):
        parser.error("phase owns file, mode and private correlation arguments")
    if args.trial_id:
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", args.trial_id):
            parser.error("trial ID must be a short lowercase technical label")
        if not args.repeat_content:
            parser.error("a paired trial must explicitly declare repeat content")
        if args.profile_label and not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", args.profile_label):
            parser.error("profile label must be a short lowercase technical label")
    elif args.repeat_content or args.profile_label or args.prepare_only or args.phase == "paired":
        parser.error("paired/repeat preparation requires an isolated trial ID")
    if args.repeat_content and any(value.split("=", 1)[0] == "--novel-content-declared" for value in forwarded):
        parser.error("reused trial content cannot be declared novel")
    original_reference = Path("benchmarks/performance") / ("gpu-observed-summary.json" if args.phase == "paired" else f"gpu-{args.phase}.json")
    trial_root = Path("benchmarks/private/trials") / args.trial_id if args.trial_id else Path("benchmarks/private")
    attempt_plan_path = trial_root / f"expected-{args.phase}-attempts.json"
    report_output = None
    for index, value in enumerate(forwarded):
        if value == "--output" and index + 1 < len(forwarded):
            report_output = Path(forwarded[index + 1])
        elif value.startswith("--output="):
            report_output = Path(value.split("=", 1)[1])
    if report_output is not None and report_output.exists():
        parser.error("benchmark report already exists; choose a new isolated output")
    try:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if selection.get("selection_scope") != "proposed_train_benchmark_only":
            raise ValueError("unexpected selection scope")
        items = [selection["baseline"]] if args.phase == "baseline" else [selection["baseline"], *selection["burst"]] if args.phase == "paired" else selection["burst"]
        if len(items) != {"baseline": 1, "burst": 5, "paired": 6}[args.phase]:
            raise ValueError("unexpected phase size")
        for item in items:
            if item.get("proposed_split") != "train":
                raise ValueError("non-train benchmark input")
            with Path(item["path"]).open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != item["content_hash"]:
                    raise ValueError("selected source changed")
        trial_metadata = {"trial_id": args.trial_id, "profile_label": args.profile_label or args.trial_id,
                          "content_reused_declared": bool(args.repeat_content), "primary_slo_evidence_eligible": False,
                          "comparison_scope": "exploratory paired proposed-train contents; original burst and sequential trial have different load conditions" if args.phase == "paired" else "exploratory paired proposed-train contents; preserve failure denominator",
                          "reference_artifact": original_reference.as_posix(),
                          "reference_artifact_sha256": hashlib.sha256(original_reference.read_bytes()).hexdigest() if original_reference.is_file() else None}
        attempt_plan_digest = ensure_attempt_plan(attempt_plan_path, items, trial_id=args.trial_id or "original-six",
                                                  phase=args.phase, mode="burst" if args.phase == "burst" else "single")
        if args.prepare_only:
            plan = {"schema_version": "1.0", "status": "prepared_not_launched", "phase": args.phase,
                    "input_count": len(items), "mode": "burst" if args.phase == "burst" else "single",
                    "api_requests_made": 0, "model_requests_made": 0, "paired_trial": trial_metadata,
                    "expected_attempt_plan_sha256": attempt_plan_digest}
            plan_path = Path("benchmarks/quality/trials") / args.trial_id / (args.phase + ".plan.json")
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            if plan_path.exists():
                previous_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if any(previous_plan.get(key) != plan[key] for key in ("phase", "input_count", "mode", "paired_trial")):
                    raise ValueError("existing aggregate plan mismatch")
            else:
                plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            binding_path = plan_path.with_name(args.phase + ".attempt-plan-binding.json")
            binding = {"schema_version": "1.0", "status": "frozen_before_requests", "expected_attempt_count": len(items),
                       "expected_attempt_plan_sha256": attempt_plan_digest, "api_requests_made": 0, "model_requests_made": 0}
            if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
                raise ValueError("existing attempt-plan binding mismatch")
            if not binding_path.exists():
                binding_path.write_text(json.dumps(binding, indent=2), encoding="utf-8")
            print(json.dumps({"status": "prepared_not_launched", "input_count": len(items), "primary_slo_evidence_eligible": False}))
            return 0
        if args.api_key_file:
            key = args.api_key_file.read_text(encoding="utf-8").strip().splitlines()[0].split(",")[0].strip()
            if not key:
                raise ValueError("empty benchmark key file")
            os.environ["FORTE_BENCHMARK_API_KEY"] = key
            forwarded.extend(["--api-key-env", "FORTE_BENCHMARK_API_KEY"])
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "scope": "selection_preflight"}))
        return 1
    terminal_path = trial_root / f"a18-{args.phase}-terminals.json"
    job_map = trial_root / f"a18-{args.phase}-jobs.json"
    if terminal_path.exists() or job_map.exists():
        parser.error("private trial artifacts already exist; choose a new isolated trial ID")
    try:
        helper_path = Path(__file__).resolve().parents[2] / "scripts/exposure_registry.py"
        spec = importlib.util.spec_from_file_location("forte_benchmark_exposure_registry", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        exposure_registration = helper.register_selection(
            args.selection, args.exposure_registry, args.exposure_manifest,
            trial_id=args.trial_id or "original-six", phase=args.phase, purpose="benchmark_attempt")
        if exposure_registration.get("ok") is not True:
            raise ValueError("exposure registration unsuccessful")
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_code": "exposure_registration_failed",
                          "error_type": type(exc).__name__, "api_requests_made": 0}))
        return 1
    if args.repeat_content and "--reused-content-declared" not in forwarded:
        forwarded.append("--reused-content-declared")
    result = benchmark_main(["--files", *[item["path"] for item in items],
                           "--mode", "burst" if args.phase == "burst" else "single",
                           "--private-job-map", str(job_map),
                           "--private-terminal-export", str(terminal_path), "--private-attempt-plan", str(attempt_plan_path), *forwarded])
    if args.trial_id and report_output is not None and report_output.is_file():
        report = json.loads(report_output.read_text(encoding="utf-8"))
        report["paired_trial"] = trial_metadata
        report["exposure_registration"] = exposure_registration
        report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if terminal_path.is_file():
        try:
            phase_records = json.loads(terminal_path.read_text(encoding="utf-8"))["records"]
            renamed = []
            for record in phase_records:
                index = int(record["sample_id"][1:]) - 1
                renamed.append({**record, "sample_id": items[index]["sample_id"]})
            canonical = trial_root / ("terminal-verdicts.json" if args.trial_id else "a18-terminal-verdicts.json")
            canonical.parent.mkdir(parents=True, exist_ok=True)
            previous = json.loads(canonical.read_text(encoding="utf-8"))["records"] if canonical.is_file() else []
            replaced = {item["sample_id"] for item in renamed}
            records = [item for item in previous if item["sample_id"] not in replaced] + renamed
            canonical.write_text(json.dumps({"schema_version": "1.0", "records": records}, indent=2), encoding="utf-8")
        except Exception as exc:
            print(json.dumps({"private_export_error_type": type(exc).__name__}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
