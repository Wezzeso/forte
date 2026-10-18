"""Closed-answer, local-only PERSONAL_ITEMS proposal diagnostic; never enables a rule."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


KINDS = ("drink_container", "coat_or_handbag", "fixed_equipment", "cash_or_documents", "other", "unclear", "none")
BOOLS = ("recognizable", "on_work_surface", "consistent_across_frames", "fixed_equipment", "touched_or_lifted")
SCHEMA = {"type": "object", "properties": {"object_kind": {"type": "string", "enum": list(KINDS)},
          **{key: {"type": "boolean"} for key in BOOLS}},
          "required": ["object_kind", *BOOLS], "additionalProperties": False}
POLICY = {"full_frames": 6, "desk_crops": 3, "phrasings": 2, "max_tokens": 128,
          "max_requests": 64, "parallel_videos": 2, "request_concurrency": 4, "max_wall_s": 150,
          "preset_controls": 8, "label_tuned_parameters": 0, "decode_fps": 2,
          "proposal_window": "inherited_reference_verify_window", "native_unresized_crops": True,
          "transport": "existing_jpeg_data_url_default_quality_88", "decision_thresholds_fitted": 0}
ERRORS = {"ValueError", "RuntimeError", "TimeoutError", "ValidationError", "CancelledError",
          "OSError", "FileNotFoundError", "PermissionError", "MemoryError", "OtherError"}
NAME = re.compile(r"X-[1-9][0-9]{0,2}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
PROMPTS = (
    "Identify the proposed personal object on the cashier's desk using the ordered full frames and native desk crops. "
    "Distinguish a real drinking container, coat or handbag from fixed desk equipment (counter, detector, computer, "
    "printer, cables or fixture). Use cash_or_documents for banknotes, cards or paperwork; never call those personal "
    "belongings. recognizable means its physical object category is actually visible, not inferred from a bright patch. "
    "on_work_surface means the item rests on the desk, not behind glass, on a person or in another area. "
    "consistent_across_frames means the same recognizable item persists in multiple frames. fixed_equipment means "
    "bank equipment or a desk fixture. touched_or_lifted means a hand visibly handles that same item. If uncertain use "
    "object_kind=unclear and recognizable=false. Answer only the closed JSON fields, no description.",
    "Inspect the same object through this time-ordered window. Is there a physically recognizable cup/bottle/mug, "
    "or a coat/handbag, sitting on the cashier's work surface? A static white/bright shape alone proves no such object. "
    "First distinguish cash-room machinery, screen/keyboard, printer, wiring and furniture from movable personal "
    "items. Report object_kind from the enum. Mark recognizable only when the object's category is clear; "
    "on_work_surface only for the desk itself; consistent_across_frames only when multiple frames show the same item; "
    "fixed_equipment for installed equipment or fixtures; touched_or_lifted only when visible hands move or handle "
    "that item. Banknotes/cards/paperwork belong to cash_or_documents. Uncertain category is unclear, never a guess. "
    "Return only the requested JSON fields without prose."
)


def need(value, code):
    if not value:
        raise ValueError(code)


def number(value, high=86400):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= high


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def error_class(error):
    name = type(error).__name__
    return name if name in ERRORS else "OtherError"


def read(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 100 * 1024**2, "input_file_invalid")
    return json.loads(path.read_text(encoding="utf-8"))


def cached_pair(first, second, verify_window):
    """Project all forty rows from each run; never inspect labels or correctness."""
    need(number(verify_window, 30) and verify_window > 0, "proposal_window_invalid")
    pair = []
    names = None
    for run, raw in zip(("a", "b"), (first, second)):
        rows = raw.get("results", raw.get("videos")) if isinstance(raw, dict) else None
        need(isinstance(rows, list) and len(rows) == 40, "complete_40_video_pair_required")
        seen = set()
        for original in rows:
            need(isinstance(original, dict) and isinstance(original.get("name"), str)
                 and NAME.fullmatch(original["name"]) and original["name"] not in seen, "video_inventory_invalid")
            seen.add(original["name"])
            need(original.get("processed") is True and not original.get("error")
                 and original.get("status") not in ("failed", "processing"), "source_video_failed")
            candidates = original.get("candidates")
            need(isinstance(candidates, list) and len(candidates) <= 1000, "source_candidates_invalid")
            proposals = []
            for candidate in candidates:
                need(isinstance(candidate, dict), "source_candidate_invalid")
                if candidate.get("type_id") != "PERSONAL_ITEMS":
                    continue
                need(number(candidate.get("t")), "source_candidate_time_invalid")
                proposals.append({"t": candidate["t"], "window": verify_window})
            layout = original.get("layout") or {}
            box = layout.get("desk") if isinstance(layout, dict) else None
            valid = isinstance(box, list) and len(box) == 4 and all(number(v, 1000) for v in box)
            valid = valid and box[0] < box[2] and box[1] < box[3]
            pair.append({"run": run, "name": original["name"], "proposals": proposals,
                         "desk": list(box) if valid else None})
        need(names is None or names == seen, "pair_inventory_differs")
        names = seen
    count = sum(len(row["proposals"]) for row in pair)
    need(2 * count <= POLICY["max_requests"], "request_budget_exceeded")
    return {"schema": "forte.personal-items-input.v1", "verify_window": verify_window,
            "videos": 40, "source_runs": 2, "proposals": count, "results": pair}


def project_answer(value):
    need(isinstance(value, dict) and set(value) == {"object_kind", *BOOLS}, "answer_fields_invalid")
    need(isinstance(value.get("object_kind"), str) and value["object_kind"] in KINDS, "answer_kind_invalid")
    need(all(type(value.get(key)) is bool for key in BOOLS), "answer_boolean_invalid")
    return {"object_kind": value["object_kind"], **{key: value[key] for key in BOOLS}}


def supports(answers):
    """Predeclared object test; touched/lifted is diagnostic only, never required."""
    need(isinstance(answers, list) and len(answers) == 2, "paired_answers_required")
    answers = [project_answer(answer) for answer in answers]
    return answers[0]["object_kind"] == answers[1]["object_kind"] and all(
        answer["object_kind"] in {"drink_container", "coat_or_handbag"}
        and answer["recognizable"] and answer["on_work_surface"]
        and answer["consistent_across_frames"] and not answer["fixed_equipment"] for answer in answers)


def score_after_inference(report, labels):
    """Scorer only; no object-type truth is claimed from binary video labels."""
    labels = labels.get("labels", labels) if isinstance(labels, dict) else None
    names = {row["name"] for row in report["results"]}
    need(isinstance(labels, dict) and set(labels) == names and all(
        isinstance(value, dict) and type(value.get("violation")) is bool for value in labels.values()), "scoring_labels_invalid")
    out = {}
    for run in ("a", "b"):
        rows = [row for row in report["results"] if row["run"] == run]
        selected = [row for row in rows if row["proposals"]]
        done = [row for row in selected if all(item["status"] == "done" for item in row["proposals"])]
        out[run] = {"videos": 40, "negative_videos": sum(not labels[row["name"]]["violation"] for row in rows),
            "videos_with_cached_personal_proposal": len(selected), "proposed_videos_completed": len(done),
            "failed_proposed_videos": len(selected) - len(done),
            "supported_on_negative_videos": sum(not labels[row["name"]]["violation"] and any(
                item.get("supported") is True for item in row["proposals"]) for row in done),
            "supported_on_positive_videos": sum(labels[row["name"]]["violation"] and any(
                item.get("supported") is True for item in row["proposals"]) for row in done)}
    return {"scope": "cached_personal_item_proposals_only; binary_labels_have_no_type_ground_truth",
            "labels_opened_only_after_all_queries": True, "pipeline_accuracy_claim": False,
            "type_enablement_qualified": False, "per_run": out}


def select_frames(frames, count):
    return frames if len(frames) <= count else [frames[round(i * (len(frames) - 1) / (count - 1))] for i in range(count)]


async def infer(data, videos, model):
    # Third-party imports and media decoding happen only on the user's GPU host.
    import httpx
    from forte.fast.video import decode
    from forte.fast.vlm import VLMClient, image_part, text_part

    started = time.monotonic()
    deadline = started + POLICY["max_wall_s"]
    by_name = {}
    for row in data["results"]:
        by_name.setdefault(row["name"], []).append(row)
    need({path.stem for path in videos.glob("*.mov")} == set(by_name), "media_inventory_differs")
    client = VLMClient("http://127.0.0.1:8000/v1", model, concurrency=POLICY["request_concurrency"], timeout=35)
    await client._client.aclose()
    client._client = httpx.AsyncClient(base_url="http://127.0.0.1:8000/v1", timeout=35,
                                      trust_env=False, follow_redirects=False)
    semaphore = asyncio.Semaphore(POLICY["parallel_videos"])
    attempted = 0
    results = []

    async def one(name, sources):
        nonlocal attempted
        rows = [{"run": row["run"], "name": name, "proposals": []} for row in sources]
        for target, source in zip(rows, sources):
            for proposal in source["proposals"]:
                target["proposals"].append({"t": proposal["t"], "status": "failed", "error_class": "TimeoutError"})
        results.extend(rows)
        if not any(source["proposals"] for source in sources):
            return
        video = None
        async with semaphore:
            try:
                need(time.monotonic() < deadline, "diagnostic_budget_exhausted")
                video = await asyncio.to_thread(decode, str(videos / (name + ".mov")), 2,
                                               deadline=deadline, max_frames=600)
                for row, source in zip(rows, sources):
                    for target, proposal in zip(row["proposals"], source["proposals"]):
                        if source["desk"] is None:
                            target.clear()
                            target.update(t=proposal["t"], status="abstained", reason="desk_not_located")
                            continue
                        try:
                            need(0 <= proposal["t"] <= video.duration, "proposal_outside_video")
                            lo, hi = max(0, proposal["t"] - proposal["window"]), min(video.duration, proposal["t"] + proposal["window"])
                            frames = select_frames(video.between(lo, hi), POLICY["full_frames"])
                            need(len(frames) >= 2, "proposal_window_unavailable")
                            content = []
                            for frame in frames:
                                content += [text_part(f"Original full frame, t={frame.t:.3f}s"), image_part(frame.image)]
                            for frame in select_frames(frames, POLICY["desk_crops"]):
                                box = source["desk"]
                                w, h = frame.image.size
                                bounds = (int(box[0] * w / 1000), int(box[1] * h / 1000),
                                          min(w, math.ceil(box[2] * w / 1000)), min(h, math.ceil(box[3] * h / 1000)))
                                need(bounds[0] < bounds[2] and bounds[1] < bounds[3], "native_desk_crop_empty")
                                region = frame.image.crop(bounds)
                                try:
                                    content += [text_part(f"Native desk crop of same original frame, t={frame.t:.3f}s; no enlargement"), image_part(region)]
                                finally:
                                    region.close()

                            async def ask(prompt):
                                nonlocal attempted
                                attempted += 1
                                return project_answer(await client.chat_json([*content, text_part(prompt)],
                                    system="Inspect visible objects only. Never identify people. Return only closed JSON.",
                                    json_schema=SCHEMA, max_tokens=POLICY["max_tokens"], strict_response=True, retries=0))

                            answers = await asyncio.gather(*(ask(prompt) for prompt in PROMPTS), return_exceptions=True)
                            for answer in answers:
                                if isinstance(answer, BaseException):
                                    raise answer
                            target.clear()
                            target.update(t=proposal["t"], start=lo, end=hi, status="done", answers=answers,
                                          supported=supports(answers), full_frames=len(frames),
                                          desk_crops=min(len(frames), POLICY["desk_crops"]))
                        except asyncio.CancelledError:
                            raise
                        except Exception as error:
                            target.clear()
                            target.update(t=proposal["t"], status="failed", error_class=error_class(error))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                for row in rows:
                    for target in row["proposals"]:
                        at = target["t"]
                        target.clear()
                        target.update(t=at, status="failed", error_class=error_class(error))
            finally:
                if video is not None:
                    for frame in video.frames:
                        frame.image.close()

    tasks = [asyncio.create_task(one(name, sources)) for name, sources in sorted(by_name.items())]
    timed_out = False
    try:
        async with asyncio.timeout(POLICY["max_wall_s"]):
            await asyncio.gather(*tasks)
    except TimeoutError:
        timed_out = True
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.aclose()
    return {"schema": "forte.personal-items-probe.v1", "model": model,
            "diagnostic_only": True, "pipeline_change_qualified": False, "type_enablement_qualified": False,
            "labels_used_for_processing": False, "policy": dict(POLICY),
            "results": sorted(results, key=lambda row: (row["run"], row["name"])),
            "summary": {"videos": 40, "source_runs": 2, "proposals": data["proposals"],
                "calls": client.calls, "attempted_calls": attempted, "prompt_tokens": client.prompt_tokens,
                "completion_tokens": client.completion_tokens, "timed_out": timed_out,
                "wall_s": round(time.monotonic() - started, 3)}}


def project_report(raw, expected):
    """Strict export boundary, also called by the host wrapper before collection."""
    need(isinstance(raw, dict) and raw.get("schema") == "forte.personal-items-probe.v1"
         and raw.get("model") == "q30test" and raw.get("policy") == POLICY, "report_header_invalid")
    source = {(row["run"], row["name"]): row for row in expected["results"]}
    rows, seen = [], set()
    for row in raw.get("results", []):
        key = (row.get("run"), row.get("name"))
        need(key in source and key not in seen, "report_inventory_invalid")
        seen.add(key)
        proposals = row.get("proposals")
        need(isinstance(proposals, list) and len(proposals) == len(source[key]["proposals"]), "report_proposal_count_invalid")
        clean = []
        for item, before in zip(proposals, source[key]["proposals"]):
            need(item.get("t") == before["t"] and item.get("status") in {"done", "failed", "abstained"}, "report_proposal_invalid")
            result = {"t": before["t"], "status": item["status"]}
            if item["status"] == "done":
                need(number(item.get("start")) and number(item.get("end")) and item["start"] <= item["t"] <= item["end"], "report_interval_invalid")
                answers = [project_answer(answer) for answer in item.get("answers", [])]
                result.update(start=item["start"], end=item["end"], answers=answers, supported=supports(answers))
                need(item.get("supported") is result["supported"], "report_decision_changed")
                for field, high in (("full_frames", 6), ("desk_crops", 3)):
                    need(type(item.get(field)) is int and 1 <= item[field] <= high, "report_frame_count_invalid")
                    result[field] = item[field]
            elif item["status"] == "failed":
                need(item.get("error_class") in ERRORS, "report_error_invalid")
                result["error_class"] = item["error_class"]
            else:
                need(item.get("reason") == "desk_not_located", "report_abstention_invalid")
                result["reason"] = "desk_not_located"
            clean.append(result)
        rows.append({"run": key[0], "name": key[1], "proposals": clean})
    need(seen == set(source), "report_inventory_incomplete")
    summary = raw.get("summary")
    need(isinstance(summary, dict), "report_summary_invalid")
    numeric = {}
    for field in ("calls", "attempted_calls", "prompt_tokens", "completion_tokens"):
        need(type(summary.get(field)) is int and 0 <= summary[field] <= 10**9, "report_metric_invalid")
        numeric[field] = summary[field]
    need(numeric["attempted_calls"] <= POLICY["max_requests"] and type(summary.get("timed_out")) is bool
         and number(summary.get("wall_s"), 180), "report_budget_invalid")
    numeric.update(videos=40, source_runs=2, proposals=expected["proposals"],
                   timed_out=summary["timed_out"], wall_s=summary["wall_s"])
    return {"schema": "forte.personal-items-probe.v1", "model": "q30test", "diagnostic_only": True,
            "pipeline_change_qualified": False, "type_enablement_qualified": False,
            "labels_used_for_processing": False, "policy": dict(POLICY), "summary": numeric, "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--model", choices=["q30test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    need(HEX.fullmatch(args.input_sha256) and sha(args.input) == args.input_sha256, "input_hash_mismatch")
    data = read(args.input)
    need(data.get("schema") == "forte.personal-items-input.v1" and len(data.get("results", [])) == 80,
         "sanitized_pair_required")
    result = project_report(asyncio.run(infer(data, args.videos, args.model)), data)
    result.update(tool_sha256=sha(__file__), input_sha256=args.input_sha256,
                  generated_at_utc=datetime.now(timezone.utc).isoformat())
    need(not args.out.exists(), "existing_output_refused")
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(args.out)
    return {"status": "done", "summary": result["summary"], "pipeline_change_qualified": False}


if __name__ == "__main__":
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            receipt = main()
        print(json.dumps(receipt, allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_class": error_class(error)}))
        raise SystemExit(1)
