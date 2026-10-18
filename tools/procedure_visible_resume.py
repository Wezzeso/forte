"""Resume technical failures in a visibility diagnostic, without changing perception.

The original procedure_visible_probe.py, its schemas, prompts, sampling, and
per-window implementation are imported unchanged. Completed windows are copied
byte-semantically with canonical hashes. Only failed technical windows are
retried; labels, verdicts, correctness and type outcomes never select work.
Cross-host reuse is an explicitly mixed-attempt diagnostic, never a fresh full
pipeline result, independent repetition, or type-qualification measurement.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 100 * 1024**2:
        raise ValueError("invalid_resume_input")
    return json.loads(path.read_text(encoding="utf-8"))


def original_tool(path):
    spec = importlib.util.spec_from_file_location("unchanged_procedure_visible_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failure_kind(row, prior):
    """Do not claim cause information that the original tool discarded."""
    if row.get("status") != "failed":
        return None
    if row.get("error_class") == "TimeoutError" and not row.get("windows"):
        return "explicit_timeout"
    windows = row.get("windows", [])
    near_deadline = (prior.get("summary", {}).get("wall_s", 0) >= 539)
    failed = [w for w in windows if w.get("status") == "failed"]
    if near_deadline and failed and all(w.get("error_class") == "RuntimeError" for w in failed):
        return "batch_failure_at_global_deadline;cause_not_independently_proven"
    raise ValueError("non_deadline_technical_failure_not_authorized_for_resume")


def reused_window_hashes(prior):
    return [{"name": row["name"], "start": window["start"], "end": window["end"], "sha256": digest(window)}
            for row in prior["results"] for window in row["windows"] if window["status"] == "done"]


async def run(args):
    sys.path.insert(0, "/forte/src")
    from forte.fast.video import decode
    T = original_tool(Path(args.tool))
    started = time.monotonic()
    source = load(Path(args.source))
    previous = load(Path(args.prior))
    prior = previous["data"]
    if previous.get("sanitized_input_sha256") != hashlib.sha256(Path(args.source).read_bytes()).hexdigest():
        raise ValueError("resume_source_binding_changed")
    if previous.get("tool_sha256") != hashlib.sha256(Path(args.tool).read_bytes()).hexdigest():
        raise ValueError("original_tool_changed")
    sources = {r["name"]: r for r in T.cached_rows(source)}
    rows = {r["name"]: copy.deepcopy(r) for r in prior["results"]}
    if len(rows) != 40 or set(rows) != set(sources):
        raise ValueError("resume_inventory_differs")
    causes = {name: failure_kind(row, prior) for name, row in rows.items() if row["status"] == "failed"}
    unchanged_hashes = reused_window_hashes(prior)
    selected, completed = [], []
    client = T.LocalClient(args.base_url, args.model, args.concurrency, started + args.max_wall_s)
    semaphore = asyncio.Semaphore(2)

    def checkpoint():
        value = {"schema": "forte.procedure-visible-resume-checkpoint.v1", "diagnostic_only": True,
                 "type_enablement_qualified": False, "prior_structured_input_sha256": hashlib.sha256(Path(args.prior).read_bytes()).hexdigest(),
                 "reused_windows": unchanged_hashes, "completed_retry_windows": list(completed),
                 "results": [rows[name] for name in sorted(rows)]}
        path = Path(args.out).with_name("completed-window-checkpoint.json")
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)

    async def one(name):
        async with semaphore:
            source_row, old = sources[name], rows[name]
            video = None
            try:
                if time.monotonic() >= client.deadline:
                    raise TimeoutError("resume_deadline")
                video = await asyncio.to_thread(decode, str(Path(args.videos) / (name + ".mov")), T.FPS,
                    max_frames=1000, max_memory_bytes=2 * 1024**3, deadline=client.deadline)
                schedule = T.transaction_windows(source_row["states"], video.duration)
                old_by_interval = {(w["start"], w["end"]): w for w in old["windows"]}
                if not set(old_by_interval) <= {(w["start"], w["end"]) for w in schedule}:
                    raise ValueError("prior_window_schedule_changed")
                output = []
                for interval in schedule:
                    key = (interval["start"], interval["end"])
                    existing = old_by_interval.get(key)
                    if existing is not None and existing["status"] == "done":
                        output.append(existing)
                        continue
                    selected.append({"name": name, **interval, "selection_cause": causes[name]})
                    try:
                        result = await T.inspect_window(client, video, source_row["layout"], interval, 8)
                    except Exception as exc:
                        result = {**interval, "status": "failed", "error_class": T.error_code(exc)}
                    output.append(result)
                    if result["status"] == "done":
                        completed.append({"name": name, **interval, "sha256": digest(result)})
                    old.update(status="failed", windows=output + [w for w in old["windows"] if (w["start"], w["end"]) not in {(x["start"], x["end"]) for x in output}])
                    checkpoint()
                if schedule:
                    old.pop("error_class", None)
                    old.update(windows=output, proposed_transactions=len(schedule),
                               status="done" if all(w["status"] == "done" for w in output) else "failed")
                else:
                    old.pop("error_class", None)
                    old.update(status="abstained", reason="no_state_transaction_proposal", windows=[], proposed_transactions=0)
                old["located_regions"] = {key: key in source_row["layout"] for key in T.REGIONS}
            except Exception as exc:
                old.update(status="failed", error_class=T.error_code(exc))
            finally:
                if video is not None:
                    for frame in video.frames:
                        frame.image.close()
                checkpoint()

    try:
        await asyncio.gather(*(one(name) for name in sorted(causes)))
    finally:
        await client.http.aclose()
    current = [rows[name] for name in sorted(rows)]
    current_windows = {(r["name"], w["start"], w["end"]): w for r in current for w in r["windows"]}
    if any(digest(current_windows[(r["name"], r["start"], r["end"])]) != r["sha256"] for r in unchanged_hashes):
        raise ValueError("completed_prior_window_changed")
    return {"schema": "forte.procedure-visible-probe.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model, "tool_sha256": previous["tool_sha256"],
            "input_sha256": previous["sanitized_input_sha256"], "type_enablement_qualified": False,
            "continuous_visibility_proven": False, "diagnostic_only": True,
            "summary": {"videos": 40, "done": sum(r["status"] == "done" for r in current),
                        "failed": sum(r["status"] == "failed" for r in current),
                        "abstained": sum(r["status"] == "abstained" for r in current),
                        "calls": client.calls, "attempted_calls": client.attempted_calls,
                        "prompt_tokens": client.prompt_tokens, "completion_tokens": client.completion_tokens,
                        "wall_s": round(time.monotonic() - started, 3)},
            "results": current,
            "resume_provenance": {"schema": "forte.procedure-visible-resume.v1",
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prior_structured_input_sha256": hashlib.sha256(Path(args.prior).read_bytes()).hexdigest(),
                "selection_policy": "technical_failure_status_and_deadline_only;no_labels_or_correctness",
                "causes": [{"name": n, "cause": causes[n]} for n in sorted(causes)],
                "prior_completed_window_hashes": unchanged_hashes,
                "selected_windows": selected, "completed_retry_windows": completed,
                "prior_failures_preserved_in_prior_receipt": True,
                "counter_semantics": "summary_video_statuses_combined;calls_tokens_wall_latest_attempt_only",
                "cross_host_or_partial_retry_is_not_fresh_qualification": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tool", "source", "prior", "videos", "model", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-wall-s", type=float, default=300)
    args = parser.parse_args()
    if args.model != "q30test" or args.concurrency not in (16, 32) or not 30 <= args.max_wall_s <= 300:
        raise ValueError("invalid_resume_controls")
    report = asyncio.run(run(args))
    path = Path(args.out)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps({"status": "done", "summary": report["summary"], "type_enablement_qualified": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        safe = type(exc).__name__ if type(exc).__name__ in {"ValueError", "TypeError", "KeyError", "RuntimeError", "TimeoutError", "OSError"} else "OtherError"
        print(json.dumps({"status": "failed", "failure_code": "procedure_resume_failed", "error_class": safe}))
        raise SystemExit(1)
