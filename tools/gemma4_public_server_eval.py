"""Paired public object diagnostic through one local vLLM FP8/LoRA server.

Requires the frozen public-v1 train/validation inventories. Reads public images
only on a named rented Linux host; never displays pixels or model prose. This
tool reports transfer metrics, never a Forte video score or promotion decision.
It does not start a model, train, contact a cloud API, or modify production.
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
import sys
import time
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.gemma4_public_eval import (  # noqa: E402
    CLASSES, GROUP_FIELDS, _write, digest, metrics, paired_metrics, read_manifest,
    strict_class,
)

VALIDATION_SHA256 = "e2c7c3c9fb070e8f078c93414005e484f389ea74cc387a6219b11eba747c5fbe"
TRAIN_SHA256 = "69c5a6ecfde839fdd50678ddae2802950fe13a9ea16b0e70c621e322388adbf8"
EXPECTED_VIEWS = 192
EXPECTED_GROUPS = 48
SCHEMA = {"type": "object", "properties": {
    "object_class": {"type": "string", "enum": list(CLASSES)}},
    "required": ["object_class"], "additionalProperties": False}
PROTOCOL_SOURCE = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/"
    "vllm/entrypoints/openai/chat_completion/protocol.py"
)


def local_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("local_vllm_v1_endpoint_required")
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        raise ValueError("literal_loopback_address_required") from None
    if not address.is_loopback:
        raise ValueError("nonlocal_model_endpoint_forbidden")
    host = f"[{address}]" if address.version == 6 else str(address)
    return urlunsplit(("http", f"{host}:{parsed.port or 8000}", "/v1", "", ""))


def request_body(model: str, image_url: str, prompt: str) -> dict:
    # Neither the public target nor identifiers/grouping metadata enter this body.
    return {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt}]}],
        "temperature": 0, "seed": 0, "max_tokens": 32, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"max_soft_tokens": 280},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "public_object_class", "strict": True, "schema": SCHEMA}}}


def failed_observation(row: dict, failure: str) -> dict:
    return {"example_id": row["example_id"], "source_group": row["source_group"],
            "source_dataset": row["source_dataset"],
            "short_side": row.get("transform", {}).get("short_side", 0),
            "expected": row["target"]["object_class"], "predicted": None,
            "status": "failed", "failure": failure, "wall_s": 0.0,
            "calls": 0, "prompt_tokens": 0, "generated_tokens": 0}


async def infer(client, semaphore, model: str, image_url: str, prompt: str) -> dict:
    observation = {"status": "failed", "failure": None, "predicted": None,
                   "calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "wall_s": 0.0}
    started = time.monotonic()
    try:
        async with semaphore:
            observation["calls"] = 1
            response = await client.post("/v1/chat/completions", json=request_body(model, image_url, prompt))
        if response.status_code != 200:
            observation["failure"] = "http_" + str(response.status_code)
            return observation
        data = response.json()
        if data.get("model") != model:
            observation["failure"] = "served_model_alias_mismatch"
            return observation
        usage = data.get("usage") or {}
        observation["prompt_tokens"] = max(0, int(usage.get("prompt_tokens") or 0))
        observation["generated_tokens"] = max(0, int(usage.get("completion_tokens") or 0))
        choices = data["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            observation["failure"] = "incomplete_or_ambiguous_completion"
            return observation
        message = choices[0]["message"]
        if (message.get("reasoning") or message.get("reasoning_content") or message.get("refusal")
                or message.get("tool_calls") or message.get("function_call")):
            observation["failure"] = "unexpected_reasoning_or_nonanswer"
            return observation
        observation["predicted"] = strict_class(message["content"])
        observation["status"] = "done"
    except asyncio.CancelledError:
        observation["failure"] = "evaluation_deadline"
    except (ValueError, KeyError, TypeError, IndexError):
        observation["failure"] = "invalid_closed_json_or_response"
    except Exception as exc:
        # Exception strings may contain a request/response: retain the type only.
        observation["failure"] = "technical:" + type(exc).__name__
    finally:
        observation["wall_s"] = round(time.monotonic() - started, 4)
    return observation


def phase_metrics(observations: list[dict]) -> dict:
    score = metrics(observations)
    score.update(calls=sum(row["calls"] for row in observations),
                 prompt_tokens=sum(row["prompt_tokens"] for row in observations),
                 failure_codes=dict(Counter(row["failure"] for row in observations if row["failure"])))
    return score


async def run(args) -> int:
    if platform.system() != "Linux" or not args.rented_host_id:
        raise ValueError("named_rented_linux_host_required")
    if args.out.exists():
        raise ValueError("output_exists_use_fresh_record_path")
    endpoint = local_endpoint(args.base_url)
    if args.base_model == args.adapter_model or not args.base_model or not args.adapter_model:
        raise ValueError("two_distinct_served_aliases_required")
    if (args.concurrency < 1 or args.concurrency > 64 or args.timeout <= 0 or args.max_wall_seconds <= 0
            or not math.isfinite(args.timeout) or not math.isfinite(args.max_wall_seconds)):
        raise ValueError("invalid_runtime_limits")
    if digest(args.manifest) != VALIDATION_SHA256 or digest(args.train_manifest) != TRAIN_SHA256:
        raise ValueError("frozen_public_v1_manifest_hash_mismatch")
    rows = read_manifest(args.manifest, args.train_manifest)
    validation = [row for row in rows if row["split"] == "validation"]
    if (len(validation) != EXPECTED_VIEWS
            or len({row["source_group"] for row in validation}) != EXPECTED_GROUPS):
        raise ValueError("frozen_public_v1_inventory_mismatch")
    root = (args.images_root or args.manifest.parent).resolve()
    from training.gemma4_lora_train import Gemma4Collator, quiet_libraries
    with quiet_libraries():
        import httpx
    cropper = Gemma4Collator(None, root, max_soft_tokens=280)
    started = time.monotonic()
    result = {
        "schema": "forte.gemma4-public-server-paired.v1", "status": "processing",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "reused_public_validation_transfer_only;published_object_class_component",
        "fresh_lockbox_claim": False, "bank_video_accuracy_claim": False,
        "cross_validation_claim": False, "violation_enablement_claim": False,
        "checkpoint_selection_performed": False,
        "provenance": {"rented_host_id": args.rented_host_id, "endpoint": endpoint,
                       "base_alias": args.base_model, "adapter_alias": args.adapter_model,
                       "expected_views_per_model": EXPECTED_VIEWS, "source_groups": EXPECTED_GROUPS,
                       "manifest_sha256": VALIDATION_SHA256, "train_manifest_sha256": TRAIN_SHA256,
                       "code_sha256": digest(Path(__file__)),
                       "metric_code_sha256": digest(Path(__file__).with_name("gemma4_public_eval.py")),
                       "crop_code_sha256": digest(Path(__file__).resolve().parents[1] / "training/gemma4_lora_train.py"),
                       "group_fields_audited": [*GROUP_FIELDS, "image_sha256"],
                       "prompt_sha256": hashlib.sha256(validation[0]["prompt"].encode()).hexdigest(),
                       "licenses": sorted({row["license_spdx"] for row in rows}),
                       "protocol_source": PROTOCOL_SOURCE,
                       "server_provenance_file_sha256": digest(args.server_provenance)
                           if args.server_provenance else None},
        "request": {"transport": "loopback_http_no_proxy_no_redirect",
                    "image_encoding": "lossless_png_same_bytes_both_models",
                    "order": "frozen_manifest_interleaved_model_pairs",
                    "concurrency_total": args.concurrency, "retries": 0,
                    "temperature": 0, "seed": 0, "max_tokens": 32,
                    "mm_processor_kwargs": {"max_soft_tokens": 280},
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_schema": SCHEMA, "response_format_strict": True},
        "runs": [],
    }
    _write(args.out, result)
    aliases = {"base": args.base_model, "adapter": args.adapter_model}
    observations = {phase: [failed_observation(row, "not_processed") for row in validation] for phase in aliases}
    prepared = []
    image_hashes = {}
    verified_images = set()
    for index, row in enumerate(validation):
        try:
            path = (root / row["image_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("public_image_outside_root")
            if path not in image_hashes:
                image_hashes[path] = digest(path)
            if image_hashes[path] != row["image_sha256"]:
                raise ValueError("public_image_hash_mismatch")
            verified_images.add(path)
            visual = {key: row[key] for key in ("image_path", "crop_box_xyxy", "transform") if key in row}
            with quiet_libraries(), cropper._crop(visual) as crop:
                payload = io.BytesIO()
                crop.save(payload, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(payload.getvalue()).decode("ascii")
            prepared.append((index, url, row["prompt"]))
        except Exception as exc:
            for phase in aliases:
                observations[phase][index]["failure"] = "public_image_input:" + type(exc).__name__
    result["preparation_s"] = round(time.monotonic() - started, 4)
    result["provenance"]["public_source_image_hashes_verified"] = len(verified_images)
    semaphore = asyncio.Semaphore(args.concurrency)
    client_limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    # Base URL has no prefix so absolute request paths cannot drop /v1 accidentally.
    origin = endpoint[:-3]
    async with httpx.AsyncClient(base_url=origin, trust_env=False, follow_redirects=False,
                                timeout=httpx.Timeout(args.timeout, connect=5.0), limits=client_limits) as client:
        ready_started = time.monotonic()
        readiness_failure = None
        try:
            ready = await client.get("/v1/models")
            if ready.status_code != 200:
                readiness_failure = "local_model_inventory_http_" + str(ready.status_code)
            else:
                served = {row["id"] for row in ready.json()["data"] if isinstance(row, dict)}
                if set(aliases.values()) - served:
                    readiness_failure = "required_local_model_alias_missing"
        except Exception as exc:
            readiness_failure = "model_inventory:" + type(exc).__name__
        result["readiness_s"] = round(time.monotonic() - ready_started, 4)

        async def one_request(phase: str, name: str, index: int, image_url: str, prompt: str) -> None:
            response = await infer(client, semaphore, name, image_url, prompt)
            observations[phase][index].update(response)

        jobs = []
        if readiness_failure:
            for phase in observations:
                for row in observations[phase]:
                    if row["failure"] == "not_processed":
                        row["failure"] = readiness_failure
        else:
            jobs = [asyncio.create_task(one_request(phase, name, *prepared_row))
                    for prepared_row in prepared for phase, name in aliases.items()]
        remaining = max(0, args.max_wall_seconds - (time.monotonic() - started))
        done, pending = await asyncio.wait(jobs, timeout=remaining) if jobs else (set(), set())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
        if pending:
            # Canceled pairs may not have reached either request; keep their full denominator.
            for phase in observations:
                for row in observations[phase]:
                    if row["failure"] == "not_processed":
                        row["failure"] = "evaluation_deadline"
    for phase, name in aliases.items():
        score_rows = observations[phase]
        result["runs"].append({"phase": phase, "model": name,
            "metrics": phase_metrics(score_rows), "results": score_rows,
            "metrics_by_short_side": {str(side): phase_metrics([row for row in score_rows if row["short_side"] == side])
                                      for side in sorted({row["short_side"] for row in score_rows})},
            "metrics_by_dataset": {source: phase_metrics([row for row in score_rows if row["source_dataset"] == source])
                                   for source in sorted({row["source_dataset"] for row in score_rows})}})
    result["paired"] = paired_metrics(observations["base"], observations["adapter"])
    result["wall_s"] = round(time.monotonic() - started, 4)
    result["status"] = "done" if all(run["metrics"]["failed"] == 0 for run in result["runs"]) else "failed"
    _write(args.out, result)
    print(json.dumps({"status": result["status"], "calls": sum(run["metrics"]["calls"] for run in result["runs"]),
                      "wall_s": result["wall_s"], "paired": result["paired"]}), flush=True)
    return 0 if result["status"] == "done" else 2


def self_test() -> None:
    assert local_endpoint("http://localhost:8000/v1") == "http://127.0.0.1:8000/v1"
    for invalid in ("https://127.0.0.1/v1", "http://example.com/v1", "http://10.0.0.1/v1",
                    "http://127.0.0.1:8000/v1?secret=a", "http://name:password@127.0.0.1/v1"):
        try:
            local_endpoint(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("loopback_guard_failed")
    body = request_body("synthetic", "data:image/png;base64,", "fixed")
    assert set(body["messages"][0]) == {"role", "content"}
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "example_id" not in json.dumps(body) and "target" not in json.dumps(body)
    assert strict_class('{"object_class":"other"}') == "other"

    class Response:
        status_code = 200

        def __init__(self, content, *, model="synthetic", reasoning=None):
            self.data = {"model": model, "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                         "choices": [{"finish_reason": "stop", "message": {
                             "content": content, "reasoning": reasoning}}]}

        def json(self):
            return self.data

    class Client:
        def __init__(self, response):
            self.response = response

        async def post(self, path, **kwargs):
            assert path == "/v1/chat/completions"
            assert kwargs["json"]["model"] == "synthetic"
            return self.response

    async def mock_checks():
        sem = asyncio.Semaphore(1)
        valid = await infer(Client(Response('{"object_class":"id_card"}')), sem, "synthetic", "fake", "fixed")
        assert valid["status"] == "done" and valid["predicted"] == "id_card" and valid["calls"] == 1
        for response in (Response('{"object_class":"id_card"}', model="wrong_alias"),
                         Response('{"object_class":"id_card"}', reasoning="synthetic_reasoning"),
                         Response('prefix {"object_class":"banknote"}')):
            bad = await infer(Client(response), sem, "synthetic", "fake", "fixed")
            assert bad["status"] == "failed" and bad["predicted"] is None and bad["calls"] == 1

        class SlowClient:
            async def post(self, *_args, **_kwargs):
                await asyncio.sleep(100)

        task = asyncio.create_task(infer(SlowClient(), sem, "synthetic", "fake", "fixed"))
        await asyncio.sleep(0)
        task.cancel()
        canceled = await task
        assert canceled["failure"] == "evaluation_deadline" and canceled["calls"] == 1

    asyncio.run(mock_checks())
    print(json.dumps({"status": "passed", "synthetic_only": True,
                      "checks": ["loopback_only", "closed_wire_schema", "scorer_metadata_absent",
                                 "response_alias", "reasoning_rejection", "canceled_call_accounting"]}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rented-host-id")
    parser.add_argument("--server-provenance", type=Path, help="Optional serving receipt, hashed only")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--base-model", default="gemma4fp8")
    parser.add_argument("--adapter-model", default="gemma4ft")
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
                    "schema": "forte.gemma4-public-server-paired.v1",
                    "bank_video_accuracy_claim": False, "fresh_lockbox_claim": False}
                _write(args.out, {**partial, **failure})
            except (OSError, ValueError):
                pass
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
