"""Fixed public desk-field transfer check of local Qwen30 BF16 base and adapter.

Only published/synthetically known cash and ID boolean values are scored.
The other four required DESK fields remain unscored; note-only ID is unscored.
This is reused public validation, never a Forte video score or fresh lockbox.
Run on a named rented Linux host only. No model loading, image/prose output,
checkpoint selection, production changes, or cloud-model requests occur here.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import ipaddress
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VALIDATION_SHA256 = "dffa4783fc13973b8ad761bf07a38ad9e6667f1f5ddc6bd873da8e3a333e7099"
TRAIN_SHA256 = "1c69c8d6c06fb8ce46b95679188fbd9ba0378e149dbe474776d8f9b5299941e9"
FIELDS = ("cash", "cash_bag", "coins_or_valuables", "drink", "coat_or_handbag", "id_or_document")
SCORED_FIELDS = ("cash", "id_or_document")
CONDITIONS = ("note_only", "id_only", "both", "blank")
OUTPUT_SCHEMA = {"type": "object", "properties": {name: {"type": "boolean"} for name in FIELDS},
                 "required": list(FIELDS), "additionalProperties": False}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def unique_pairs(pairs):
    answer = {}
    for key, value in pairs:
        if key in answer:
            raise ValueError("duplicate_json_key")
        answer[key] = value
    return answer


def reject_constant(_value):
    raise ValueError("nonfinite_json")


def strict_desk(text: str) -> dict:
    value = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    if (not isinstance(value, dict) or set(value) != set(FIELDS)
            or any(type(item) is not bool for item in value.values())):
        raise ValueError("strict_six_boolean_desk_schema_required")
    return value


def local_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("literal_loopback_vllm_v1_required")
    host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        raise ValueError("literal_loopback_vllm_v1_required") from None
    if not address.is_loopback:
        raise ValueError("nonlocal_model_endpoint_forbidden")
    host = f"[{address}]" if address.version == 6 else str(address)
    return urlunsplit(("http", f"{host}:{parsed.port or 8000}", "/v1", "", ""))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def request_body(model: str, image: str, system: str, prompt: str) -> dict:
    # No target, condition, filename, source class or grouping identifier enters the model.
    return {"model": model, "messages": [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                       {"type": "text", "text": prompt}]}],
        "temperature": 0, "seed": 0, "max_tokens": 128, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "desk", "strict": True, "schema": OUTPUT_SCHEMA}}}


def failed_row(row: dict, failure="not_processed") -> dict:
    fields = row["supervision_fields"]
    return {"example_id": row["example_id"], "source_group": row["source_group"],
            "condition": row["condition"], "supervision_fields": fields,
            "expected_known": {name: row["target"][name] for name in fields},
            "predicted": None, "status": "failed", "failure": failure,
            "calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "wall_s": 0.0}


def known_correct(row: dict) -> bool:
    return row["status"] == "done" and all(
        row["predicted"][name] == truth for name, truth in row["expected_known"].items())


def metrics(rows: list[dict]) -> dict:
    per_field = {}
    for name in SCORED_FIELDS:
        labeled = [row for row in rows if name in row["expected_known"]]
        counts = Counter()
        for row in labeled:
            if row["status"] != "done":
                counts["failed"] += 1
            else:
                truth, prediction = row["expected_known"][name], row["predicted"][name]
                counts["tp" if truth and prediction else "fn" if truth else "fp" if prediction else "tn"] += 1
        correct = counts["tp"] + counts["tn"]
        positives = sum(row["expected_known"][name] for row in labeled)
        per_field[name] = {"known_values": len(labeled), "positive": positives,
            "negative": len(labeled) - positives, "correct": correct,
            "accuracy": correct / len(labeled) if labeled else None,
            **{key: counts[key] for key in ("tp", "tn", "fp", "fn", "failed")}}
    known = sum(value["known_values"] for value in per_field.values())
    correct = sum(value["correct"] for value in per_field.values())
    walls = sorted(row["wall_s"] for row in rows)
    return {"examples": len(rows), "processed": sum(row["status"] == "done" for row in rows),
        "failed": sum(row["status"] != "done" for row in rows),
        "strict_schema_rate": sum(row["status"] == "done" for row in rows) / len(rows) if rows else None,
        "known_values": known, "correct_known_values": correct,
        "known_value_accuracy": correct / known if known else None,
        "all_known_fields_correct_examples": sum(known_correct(row) for row in rows),
        "all_known_fields_accuracy": sum(known_correct(row) for row in rows) / len(rows) if rows else None,
        "per_field": per_field, "unscored_fields": [name for name in FIELDS if name not in SCORED_FIELDS],
        "calls": sum(row["calls"] for row in rows), "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "failure_codes": dict(Counter(row["failure"] for row in rows if row["failure"])),
        "latency_s_including_client_queue": {"p50": statistics.median(walls) if walls else None,
            "p95": walls[max(0, math.ceil(.95 * len(walls)) - 1)] if walls else None,
            "max": max(walls) if walls else None}}


def paired_metrics(base: list[dict], adapter: list[dict]) -> dict:
    if [row["example_id"] for row in base] != [row["example_id"] for row in adapter]:
        raise ValueError("paired_inventories_differ")
    examples, values = Counter(), Counter()
    for old, new in zip(base, adapter):
        examples[(known_correct(old), known_correct(new))] += 1
        for name, truth in old["expected_known"].items():
            old_ok = old["status"] == "done" and old["predicted"][name] == truth
            new_ok = new["status"] == "done" and new["predicted"][name] == truth
            values[(old_ok, new_ok)] += 1
    def project(counts):
        total = sum(counts.values())
        return {"n": total, "both_correct": counts[(True, True)],
            "both_wrong_or_failed": counts[(False, False)], "adapter_only_correct": counts[(False, True)],
            "base_only_correct": counts[(True, False)],
            "accuracy_delta": (counts[(False, True)] - counts[(True, False)]) / total if total else None}
    return {"examples_all_known_fields": project(examples), "known_field_values": project(values)}


async def infer(client, semaphore, model, image, system, prompt):
    answer = {"status": "failed", "failure": None, "predicted": None,
              "calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "wall_s": 0.0}
    started = time.monotonic()
    try:
        async with semaphore:
            answer["calls"] = 1
            response = await client.post("/v1/chat/completions", json=request_body(model, image, system, prompt))
        if response.status_code != 200:
            answer["failure"] = "http_" + str(response.status_code)
            return answer
        data = response.json()
        if data.get("model") != model:
            answer["failure"] = "served_alias_mismatch"
            return answer
        usage = data.get("usage") or {}
        answer["prompt_tokens"] = max(0, int(usage.get("prompt_tokens") or 0))
        answer["generated_tokens"] = max(0, int(usage.get("completion_tokens") or 0))
        choices = data["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            answer["failure"] = "incomplete_completion"
            return answer
        message = choices[0]["message"]
        if any(message.get(name) for name in ("reasoning", "reasoning_content", "refusal", "tool_calls", "function_call")):
            answer["failure"] = "unexpected_reasoning_or_nonanswer"
            return answer
        answer["predicted"] = strict_desk(message["content"])
        answer["status"] = "done"
    except asyncio.CancelledError:
        answer["failure"] = "evaluation_deadline"
    except (ValueError, KeyError, TypeError, IndexError):
        answer["failure"] = "invalid_closed_json_or_response"
    except Exception as exc:
        answer["failure"] = "technical:" + type(exc).__name__
    finally:
        answer["wall_s"] = round(time.monotonic() - started, 4)
    return answer


async def run(args) -> int:
    if platform.system() != "Linux" or not args.rented_host_id:
        raise ValueError("named_rented_linux_host_required")
    if args.out.exists():
        raise ValueError("fresh_output_path_required")
    if (args.concurrency < 1 or args.concurrency > 64 or args.timeout <= 0 or args.max_wall_seconds <= 0
            or not math.isfinite(args.timeout) or not math.isfinite(args.max_wall_seconds)
            or args.base_model == args.adapter_model):
        raise ValueError("invalid_runtime_arguments")
    endpoint = local_endpoint(args.base_url)
    if digest(args.manifest) != VALIDATION_SHA256 or digest(args.train_manifest) != TRAIN_SHA256:
        raise ValueError("frozen_q30_public_manifest_mismatch")
    from training.qwen30_lora_train import Qwen30Collator, validate_rows, quiet_libraries, SYSTEM, DESK_Q
    rows = load_manifest(args.train_manifest) + load_manifest(args.manifest)
    lineage = validate_rows(rows)
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate_composition_identity")
    validation = [row for row in rows if row["split"] == "validation"]
    if len(validation) != 128 or Counter(row["condition"] for row in validation) != Counter(dict.fromkeys(CONDITIONS, 32)):
        raise ValueError("frozen_q30_condition_inventory_mismatch")
    if any(row.get("json_schema") != OUTPUT_SCHEMA for row in rows):
        raise ValueError("pipeline_desk_schema_mismatch")
    root = (args.images_root or args.manifest.parent).resolve()
    cropper = Qwen30Collator(None, root)
    with quiet_libraries():
        import httpx
    started = time.monotonic()
    aliases = {"base": args.base_model, "adapter": args.adapter_model}
    observations = {phase: [failed_row(row) for row in validation] for phase in aliases}
    result = {"schema": "forte.q30-public-desk-paired.v1", "status": "processing",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "reused_public_compositions_transfer_only;partial_known_boolean_supervision",
        "bank_video_accuracy_claim": False, "fresh_lockbox_claim": False, "cross_validation_claim": False,
        "checkpoint_selection_performed": False, "violation_enablement_claim": False,
        "provenance": {"rented_host_id": args.rented_host_id, "endpoint": endpoint, "model_aliases": aliases,
            "base_precision_declared": "bfloat16", "manifest_sha256": VALIDATION_SHA256,
            "train_manifest_sha256": TRAIN_SHA256, "code_sha256": digest(Path(__file__)),
            "renderer_code_sha256": digest(Path(__file__).resolve().parents[1] / "training/qwen30_lora_train.py"),
            "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "desk_prompt_sha256": hashlib.sha256(DESK_Q.encode()).hexdigest(),
            "group_overlap_audit": lineage, "expected_views_per_model": 128, "expected_known_values_per_model": 224,
            "server_provenance_sha256": digest(args.server_provenance) if args.server_provenance else None},
        "request": {"transport": "loopback_http_no_proxy_no_redirect", "retries": 0,
            "same_lossless_png_bytes_both_aliases": True, "max_tokens": 128, "temperature": 0, "seed": 0,
            "concurrency_total": args.concurrency, "order": "frozen_manifest_interleaved_model_pairs",
            "exact_pipeline_system_and_desk_prompt": True, "strict_response_schema": OUTPUT_SCHEMA},
        "scoring_policy": {"fields": list(SCORED_FIELDS), "other_four_fields_scored": False,
            "note_only_id_scored": False, "failed_values_remain_in_denominator": True}, "runs": []}
    write_json(args.out, result)
    verified, prepared = {}, []
    for index, row in enumerate(validation):
        try:
            for component in row["composition"]["components"]:
                path = (root / component["image_path"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("public_component_outside_root")
                if path not in verified:
                    verified[path] = digest(path)
                if verified[path] != component["image_sha256"]:
                    raise ValueError("public_component_hash_mismatch")
            # Composition geometry/masks are renderer inputs; expected booleans are not.
            with quiet_libraries(), cropper._crop({"composition": row["composition"]}) as canvas:
                payload = io.BytesIO()
                canvas.save(payload, format="PNG")
            image = "data:image/png;base64," + base64.b64encode(payload.getvalue()).decode("ascii")
            prepared.append((index, image))
        except Exception as exc:
            for phase in observations:
                observations[phase][index]["failure"] = "public_input:" + type(exc).__name__
    result["preparation_s"] = round(time.monotonic() - started, 4)
    result["provenance"]["public_source_images_checked"] = len(verified)
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=endpoint[:-3], trust_env=False, follow_redirects=False,
                                timeout=httpx.Timeout(args.timeout, connect=5), limits=limits) as client:
        readiness_failure = None
        try:
            response = await client.get("/v1/models")
            if response.status_code != 200:
                readiness_failure = "model_inventory_http_" + str(response.status_code)
            elif set(aliases.values()) - {row["id"] for row in response.json()["data"]}:
                readiness_failure = "required_alias_missing"
        except Exception as exc:
            readiness_failure = "model_inventory:" + type(exc).__name__
        async def one_request(phase, name, index, image):
            answer = await infer(client, sem, name, image, SYSTEM, DESK_Q)
            observations[phase][index].update(answer)
        jobs = []
        if readiness_failure:
            for phase in observations:
                for row in observations[phase]:
                    if row["failure"] == "not_processed":
                        row["failure"] = readiness_failure
        else:
            jobs = [asyncio.create_task(one_request(phase, name, *item))
                    for item in prepared for phase, name in aliases.items()]
        remaining = max(0, args.max_wall_seconds - (time.monotonic() - started))
        done, pending = await asyncio.wait(jobs, timeout=remaining) if jobs else (set(), set())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    for phase, name in aliases.items():
        score_rows = observations[phase]
        for row in score_rows:
            if row["failure"] == "not_processed":
                row["failure"] = "evaluation_deadline"
        result["runs"].append({"phase": phase, "model": name, "metrics": metrics(score_rows),
            "metrics_by_condition": {condition: metrics([row for row in score_rows if row["condition"] == condition])
                                     for condition in CONDITIONS}, "results": score_rows})
    result["paired"] = paired_metrics(observations["base"], observations["adapter"])
    result["wall_s"] = round(time.monotonic() - started, 4)
    result["status"] = "done" if all(run["metrics"]["failed"] == 0 for run in result["runs"]) else "failed"
    write_json(args.out, result)
    print(json.dumps({"status": result["status"], "wall_s": result["wall_s"], "paired": result["paired"]}), flush=True)
    return 0 if result["status"] == "done" else 2


def self_test():
    prediction = dict.fromkeys(FIELDS, False)
    assert strict_desk(json.dumps(prediction)) == prediction
    for bad in ('{"cash":true}', json.dumps({**prediction, "cash": 1}),
                'prefix ' + json.dumps(prediction), '{"cash":true,"cash":false}'):
        try:
            strict_desk(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("strict_desk_fixture_failed")
    source = {"example_id": "synthetic", "source_group": "synthetic", "condition": "note_only",
              "target": {**prediction, "cash": True}, "supervision_fields": ["cash"]}
    failed = failed_row(source)
    passed = {**failed, "status": "done", "failure": None, "predicted": {**prediction, "cash": True,
                                                                                  "drink": True}}
    score = metrics([failed, passed])
    assert score["known_values"] == 2 and score["known_value_accuracy"] == .5
    assert score["per_field"]["id_or_document"]["known_values"] == 0
    assert "drink" not in score["per_field"] and known_correct(passed)
    assert paired_metrics([failed], [passed])["known_field_values"]["adapter_only_correct"] == 1
    body = request_body("synthetic", "fake", "fixed_system", "fixed_prompt")
    assert "target" not in json.dumps(body) and "condition" not in json.dumps(body)
    assert local_endpoint("http://localhost:8000/v1") == "http://127.0.0.1:8000/v1"
    try:
        local_endpoint("http://example.com/v1")
    except ValueError:
        pass
    else:
        raise AssertionError("nonlocal_endpoint_fixture_failed")
    print(json.dumps({"status": "passed", "synthetic_only": True,
        "checks": ["strict_six_booleans", "partial_supervision", "failure_denominator", "unscored_fields_ignored",
                   "paired_metrics", "scorer_metadata_absent", "loopback_only"]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rented-host-id")
    parser.add_argument("--server-provenance", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--base-model", default="q30test")
    parser.add_argument("--adapter-model", default="q30desk")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-wall-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not all((args.manifest, args.train_manifest, args.out, args.rented_host_id)):
        parser.error("--manifest --train-manifest --out --rented-host-id are required")
    existed = args.out.exists()
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        failure = {"status": "failed", "failure": type(exc).__name__}
        if not existed:
            try:
                partial = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {
                    "schema": "forte.q30-public-desk-paired.v1", "bank_video_accuracy_claim": False,
                    "fresh_lockbox_claim": False}
                write_json(args.out, {**partial, **failure})
            except (OSError, ValueError):
                pass
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
