"""Evaluate live inference, or gate a recorded per-video report without a GPU.

Original CLI: --videos DIR --labels labels.json --base-url URL [--model NAME]
Offline gate: --labels labels.json --report results.json --gate --expect-count 40
              --expect-clean 19 --baseline previous.json
Labels are scoring inputs only and never reach the analyzer. Replaying a report
does not execute the current analyzer or establish cross-validation accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_validation import file_hash, grouped_folds, json_hash, regression_gate, score_results, validate_labels  # noqa: E402


class AuditedVideoClient:
    """Record per-video model errors even if a rule catches the exception."""

    def __init__(self, client):
        self.client = client
        self.failures: list[str] = []
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def chat_json(self, *args, **kwargs):
        self.calls += 1
        try:
            return await self.client.chat_json(*args, **kwargs)
        except Exception as exc:
            self.failures.append(type(exc).__name__)
            raise


async def run_one(name: str, path: Path, client, cfg: dict, *, runtime=None,
                  runtime_lock: asyncio.Lock | None = None, candidate_types: tuple[str, ...] = (),
                  priority: int | None = None) -> dict:
    from forte.fast.analyzer import analyze
    from forte.fast.service import decode_for_analysis

    started = time.monotonic()
    if runtime is not None:
        if runtime_lock is None:
            raise ValueError("shared specialist runtime requires a per-video perception lock")
        async with runtime_lock:
            video, specialist_analysis = await asyncio.to_thread(decode_for_analysis, str(path), cfg, runtime)
    else:
        video, specialist_analysis = await asyncio.to_thread(decode_for_analysis, str(path), cfg)
    decode_s = time.monotonic() - started
    try:
        if candidate_types:
            if specialist_analysis is None:
                raise ValueError("candidate types require a class-gated specialist runtime")
            # This override exists only in this evaluator. The service still
            # requires a separately promoted type gate before reporting them.
            specialist_analysis = replace(specialist_analysis, approved_types=tuple(sorted(
                set(specialist_analysis.approved_types) | set(candidate_types))),
                provenance={**specialist_analysis.provenance, "evaluation_only": True})
        audited = AuditedVideoClient(client)
        kwargs = {"priority": priority} if priority is not None and cfg.get("job_priority") else {}
        result = await analyze(audited, video, cfg, specialist_analysis=specialist_analysis, **kwargs)
        row = {
            "name": name, "duration": video.duration, "decode_s": round(decode_s, 2),
            "timings": {k: round(v, 2) for k, v in result.timings.items()},
            "events": [asdict(e) for e in result.events],
            "candidates": [asdict(c) for c in result.candidates],
            "verifications": result.verifications, "states": result.states, "layout": result.layout,
            "decisions": getattr(result, "audit", {}), "wall_s": round(time.monotonic() - started, 2),
            "calls": audited.calls,
        }
        if specialist_analysis is not None:
            row["specialist_provenance"] = specialist_analysis.provenance
        if audited.failures:
            row["error"] = "model_call_failed:" + ",".join(audited.failures)
        return row
    finally:
        for frame in video.frames:
            frame.image.close()


def argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--config")
    ap.add_argument("--only", help="Comma-separated exact names; subset diagnostic only")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--report", help="Re-score existing per-video JSON; no model execution")
    ap.add_argument("--baseline", help="Previous complete per-video report, scored with the same labels")
    ap.add_argument("--gate", action="store_true", help="Fail on incomplete results, errors or any false alarm")
    ap.add_argument("--expect-count", type=int)
    ap.add_argument("--expect-clean", type=int)
    ap.add_argument("--min-accuracy", type=float)
    ap.add_argument("--max-wall-s", type=float)
    ap.add_argument("--time-tolerance", type=float, default=2.0)
    ap.add_argument("--require-type-time", action="store_true", help="Also gate explicitly labeled type and timing")
    ap.add_argument("--manifest", help="Explicit source/scenario manifest; recorded as provenance only")
    ap.add_argument("--candidate-types", help="Staging only: comma-separated specialist types to evaluate before promotion")
    ap.add_argument("--hash-videos", action=argparse.BooleanOptionalAction, default=True,
                    help="Stream SHA256 of inputs for exact data provenance (default enabled)")
    return ap


async def main(argv: list[str] | None = None) -> int:
    ap = argument_parser()
    args = ap.parse_args(argv)
    if args.par < 1 or args.concurrency < 1:
        ap.error("--par and --concurrency must be positive")
    labels_path = Path(args.labels)
    labels = validate_labels(json.loads(labels_path.read_text(encoding="utf-8")))
    names = sorted(labels)
    if args.only:
        requested = args.only.split(",")
        if not all(requested) or len(set(requested)) != len(requested) or set(requested) - set(labels):
            ap.error("--only must contain unique, non-empty exact label names")
        names = sorted(requested)
    if args.gate and set(names) != set(labels):
        ap.error("--gate requires every labeled video; --only is diagnostic")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(cfg, dict):
        ap.error("--config must contain an object")
    candidate_types = tuple(args.candidate_types.split(",")) if args.candidate_types else ()
    if candidate_types:
        if args.report:
            ap.error("--candidate-types requires live inference; replay cannot enable a detector")
        from forte.fast.temporal_events import SUPPORTED_TYPES
        if len(set(candidate_types)) != len(candidate_types) or not set(candidate_types) <= SUPPORTED_TYPES:
            ap.error("--candidate-types must contain unique supported specialist event types")
    provenance = {
        "mode": "cached_report" if args.report else "live_analyzer_inference",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_file_sha256": file_hash(labels_path), "subset": len(names) != len(labels),
        "expected_names": names, "heldout_accuracy_claim": False, "cross_validation_claim": False,
        "claim_scope": "development_run_only; cached replay and split plans are not cross-validation",
        "evaluation_mode": "candidate_only" if candidate_types else "service_configuration",
        "provisional_types": sorted(candidate_types),
        "scorer_code_sha256": file_hash(Path(__file__).with_name("fast_validation.py")),
        "runner_code_sha256": file_hash(Path(__file__)),
    }
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        count = len({c.get("group_id") for c in manifest.get("clips", []) if c.get("role", "development") == "development"})
        grouped_folds(manifest, n_splits=min(5, count))
        if set(names) - {c["name"] for c in manifest["clips"]}:
            ap.error("manifest does not cover every evaluated name")
        provenance["manifest_sha256"] = json_hash(manifest)
        provenance["grouping_provenance"] = manifest["provenance"]
    counters = {}
    if args.report:
        source_path = Path(args.report)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(source, dict) or "results" not in source:
            ap.error("--report needs per-video results; aggregate summaries cannot be gated")
        if args.config and cfg != source.get("config", {}):
            ap.error("--config differs from cached report; replay cannot test a changed config")
        cfg = source.get("config", {})
        if not isinstance(cfg, dict):
            ap.error("cached report config must contain an object")
        raw = source["results"]
        source_provenance = source.get("provenance", {})
        if not isinstance(source_provenance, dict):
            ap.error("cached report provenance must contain an object")
        provenance.update(source_report_sha256=file_hash(source_path),
                          source_provenance=source_provenance, inference_executed=False,
                          evaluation_mode=source_provenance.get("evaluation_mode", "unknown_recorded_run"),
                          provisional_types=source_provenance.get("provisional_types", []))
    else:
        if not args.videos or not args.base_url:
            ap.error("inference requires --videos and --base-url, or use --report for offline scoring")
        from forte.fast.analyzer import DEFAULT_CONFIG
        from forte.fast.specialists import SpecialistRuntime
        from forte.fast.vlm import VLMClient

        cfg = {**DEFAULT_CONFIG, **cfg}
        specialist_cfg = cfg.get("specialists", {"enabled": False})
        if not isinstance(specialist_cfg, dict):
            ap.error("specialists config must contain an object")
        runtime = await asyncio.to_thread(SpecialistRuntime.from_config, specialist_cfg)
        if candidate_types and runtime is None:
            ap.error("--candidate-types requires enabled specialists and a passed object-class gate")
        if runtime is not None:
            provenance["specialist_identity"] = {
                **{key: runtime.gate[key] for key in ("checkpoint_sha256", "dataset_lock_sha256", "config_sha256")},
                "gate_sha256": runtime.provenance["gate_sha256"],
            }
        runtime_lock = asyncio.Lock()
        client = VLMClient(args.base_url, args.model, concurrency=args.concurrency)
        semaphore = asyncio.Semaphore(args.par)
        inputs = []

        async def guarded(name: str):
            started = time.monotonic()
            path = Path(args.videos) / f"{name}.mov"
            async with semaphore:
                try:
                    entry = {"name": name, "file_name": path.name, "bytes": path.stat().st_size}
                    if args.hash_videos:
                        entry["sha256"] = await asyncio.to_thread(file_hash, path)
                    inputs.append(entry)
                    result = await run_one(name, path, client, cfg, runtime=runtime,
                                           runtime_lock=runtime_lock, candidate_types=candidate_types,
                                           priority=names.index(name))
                    result["analysis_wall_s"] = result["wall_s"]
                except Exception as exc:
                    result = {"name": name, "error": repr(exc)}
                result["wall_s"] = round(time.monotonic() - started, 3)
                return result

        started = time.monotonic()
        try:
            raw = await asyncio.gather(*(guarded(n) for n in names))
            counters = {"total_s": round(time.monotonic() - started, 3),
                        "calls": client.calls, "prompt_tokens": client.prompt_tokens,
                        "completion_tokens": client.completion_tokens}
        finally:
            await client.aclose()
        provenance.update(inference_executed=True, model=args.model,
                          input_inventory_sha256=json_hash(sorted(inputs, key=lambda i: i["name"])),
                          data_hash_scope="full_file_sha256" if args.hash_videos else "names_and_file_sizes_only",
                          inputs=sorted(inputs, key=lambda i: i["name"]))
    provenance["config_sha256"] = json_hash(cfg)
    if not args.report:
        from forte.fast.policy import analyzer_policy_sha256

        provenance["analyzer_policy_sha256"] = analyzer_policy_sha256(cfg)
    report = score_results(labels, raw, expected_names=names, time_tolerance=args.time_tolerance)
    report.update(schema="forte.fast-evaluation.v2", config=cfg, provenance=provenance)
    report["summary"].update(counters)
    baseline = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        prior = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not isinstance(prior, dict) or "results" not in prior:
            ap.error("--baseline requires a complete per-video report")
        baseline = score_results(labels, prior.get("results"), expected_names=names,
                                 time_tolerance=args.time_tolerance)
        if "labels_sha256" in prior and prior["labels_sha256"] != baseline["labels_sha256"]:
            ap.error("baseline label identity differs; use the original labels for regression comparison")
        current_inventory = provenance.get("input_inventory_sha256")
        previous_inventory = prior.get("provenance", {}).get("input_inventory_sha256")
        if current_inventory and previous_inventory and current_inventory != previous_inventory:
            ap.error("baseline media inventory differs from the current inputs")
        provenance["baseline_sha256"] = file_hash(baseline_path)
    if args.gate:
        report["gate"] = regression_gate(report, expected_count=args.expect_count, expected_clean=args.expect_clean,
                                          min_accuracy=args.min_accuracy, max_wall_s=args.max_wall_s,
                                          baseline=baseline, require_type_time=args.require_type_time)
    for row in report["results"]:
        verdict = "failed" if not row["processed"] else "violation" if row["prediction"] else "clean"
        print(f"{row['name']}: {verdict}; {'OK' if row['correct'] else 'BAD'}; "
              f"reasons={','.join(row['error_reasons']) or '-'}", flush=True)
    print(json.dumps({"mode": provenance["mode"], **report["summary"], "gate": report.get("gate")}, ensure_ascii=False))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 1 if args.gate and not report["gate"]["passed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except ValueError as exc:
        print(f"evaluation input error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
