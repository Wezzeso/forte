"""Private liveness/readiness and synthetic representative-shape warmup.

No original media is opened. No model output is logged or persisted. Warmup
proves successful local inference, not event accuracy or production capacity.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "redirect refused", headers, fp)


def request_local(port: int, path: str, *, payload=None, timeout: float = 5):
    # Endpoint is always this container's loopback; ignore environment proxies.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers)
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("model control response is too large")
        return json.loads(raw) if raw else None


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def is_ready(config: dict, path: Path, *, check_health: bool = True) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["config_sha256"] != config_digest(config) or not record["prewarm_completed"]:
            return False
        if record["model_revision"] != config["model_revision"] or not record.get("model_files_verified"):
            return False
        if record.get("vllm_version") != config["vllm_version"]:
            return False
        verification = record.get("snapshot_verification", {})
        if (verification.get("verified") is not True
                or verification.get("model_id") != config["model_id"]
                or verification.get("revision") != config["model_revision"]
                or verification.get("processor_revision") != config["model_revision"]
                or not re.fullmatch(r"[0-9a-f]{64}", verification.get("manifest_sha256", ""))):
            return False
        pid = int(record["server_pid"])
        if pid < 1:
            return False
        os.kill(pid, 0)
        if check_health:
            request_local(config["port"], "/health")
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _synthetic_payload(config: dict, shape: dict, seed: int) -> dict:
    # Pillow is provided by the pinned GPU image. Keep scheduler imports stdlib.
    from PIL import Image, ImageDraw

    content = [{"type": "text", "text": "Synthetic warmup only. Return JSON {\"ok\":true}."}]
    for index in range(shape["frames"]):
        image = Image.new("RGB", (shape["width"], shape["height"]),
                          ((seed * 17 + index * 31) % 256, 70, 140))
        ImageDraw.Draw(image).rectangle((20, 20, 80, 80), fill=(220, 180, 50))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        }})
    return {
        "model": config["served_model_name"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 32,
        "structured_outputs": {"json": {
            "type": "object", "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"], "additionalProperties": False,
        }},
    }


def prewarm(config: dict) -> list[dict]:
    models = request_local(config["port"], "/v1/models")
    if config["served_model_name"] not in {row["id"] for row in models.get("data", [])}:
        raise ValueError("expected model alias unavailable")

    def one(shape: dict, seed: int) -> dict:
        payload = _synthetic_payload(config, shape, seed)
        started = time.perf_counter()
        output = request_local(config["port"], "/v1/chat/completions", payload=payload,
                               timeout=config["request_timeout_seconds"])
        choice = output["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("warmup generation did not finish normally")
        decision = json.loads(choice["message"]["content"])
        if set(decision) != {"ok"} or decision["ok"] is not True:
            raise ValueError("warmup schema/content failed")
        usage = output.get("usage", {})
        return {"shape": shape, "elapsed_seconds": time.perf_counter() - started,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    records = [one(shape, index) for index, shape in enumerate(config["prewarm_shapes"])]
    # Exercise simultaneous largest requests before advertising max_num_seqs.
    largest = config["prewarm_shapes"][-1]
    with ThreadPoolExecutor(max_workers=config["prewarm_concurrency"]) as pool:
        futures = [pool.submit(one, largest, 100 + i) for i in range(config["prewarm_concurrency"])]
        records.extend(future.result() for future in futures)
    return records


def publish_ready(config: dict, path: Path, server_pid: int, warmup: list[dict], *,
                  verification: dict, schema_smoke: dict) -> None:
    if (verification.get("verified") is not True
            or verification.get("revision") != config["model_revision"]
            or verification.get("processor_revision") != config["model_revision"]
            or verification.get("model_id") != config["model_id"]
            or not re.fullmatch(r"[0-9a-f]{64}", verification.get("manifest_sha256", ""))):
        raise ValueError("snapshot verification does not match the selected baseline")
    if (schema_smoke.get("synthetic_only") is not True
            or schema_smoke.get("real_local_inference") is not True
            or schema_smoke.get("schema_and_semantic_validation_passed") is not True):
        raise ValueError("actual decision schema smoke did not complete")
    packages = {}
    for name in ("vllm", "torch", "torchvision", "transformers", "pillow", "jsonschema"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    record = {
        "prewarm_completed": True,
        "server_pid": server_pid,
        "config_sha256": config_digest(config),
        "model_revision": config["model_revision"],
        "vllm_version": config["vllm_version"],
        "package_versions": packages,
        "model_files_verified": True,
        "snapshot_verification": verification,
        "ready_at": datetime.now(timezone.utc).isoformat(),
        "warmup": warmup,
        "schema_smoke": schema_smoke,
        "capacity_benchmark_completed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Private local model readiness")
    parser.add_argument("--config", type=Path, default=Path("configs/serving/baseline.json"))
    parser.add_argument("--state", type=Path, default=Path("/run/forte/ready.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    return 0 if is_ready(config, args.state) else 1


if __name__ == "__main__":
    raise SystemExit(main())
