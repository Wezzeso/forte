"""One-use, generated-only full-schema endpoint gate; no private video input.

Prepare mode makes no HTTP/model calls. Execute mode sends exactly one model
POST, with no retry/repair. Run only after root's GPU handoff/startup reservation.
This proves structure/integration, never nineteen visual truths or accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlsplit


class GateError(RuntimeError):
    pass


def need(value, code):
    if not value:
        raise GateError(code)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def schema_definition(config, root):
    from serving.qwen38_identity import load_json
    from forte.decision.structured import DecisionConfig, _schema

    rules_path = root / "rules/catalog.json"
    need(sha(rules_path) == "3ba1c79c00f39a40addf4e1d2299c5bbade26c7949aeee10c382b9322fdde1dd",
         "generated_gate_public_rules_mismatch")
    catalog = load_json(rules_path)
    enabled = [rule for rule in catalog["rules"] if rule["enabled"] is True]
    fact_ids = sorted({fid for rule in enabled for fid in catalog["bindings"][rule["rule_id"]]["required_fact_states"]})
    events = sorted({catalog["bindings"][rule["rule_id"]]["event_type_id"] for rule in enabled})
    need(len(enabled) == 4 and len(fact_ids) == 19 and len(events) == 3
         and config["limits"]["max_images"] == 16
         and config["limits"]["max_pixels_per_image"] == 262144
         and config["limits"]["max_context_tokens"] == 32768
         and config["limits"]["request_timeout_seconds"] == 240
         and config["generation"]["max_new_tokens"] == 4096
         and config["generation"]["seed"] == 20261017,
         "generated_gate_full_public_profile_mismatch")
    schema = _schema([f"f{index:02d}" for index in range(16)], fact_ids, events,
                     DecisionConfig(compact_frame_ids=True, concise_observations=True))
    return catalog, fact_ids, events, schema


def check_grammar(ready, config):
    from serving.qwen38_grammar import POLICY
    value = ready.get("structured_decoding", {})
    warmup = value.get("public_schema_warmup", {})
    need(all(value.get(key) == item for key, item in POLICY.items())
         and value.get("fresh_matcher_per_generate") is True
         and value.get("model_vocab_size") == 248320 and value.get("stop_token_ids") == [248046, 248044]
         and warmup.get("fact_count") == 19 and warmup.get("rule_count") == 4
         and warmup.get("event_type_count") == 3 and warmup.get("frame_count") == 16
         and warmup.get("model_calls") == 0 and warmup.get("private_media_read") is False,
         "generated_gate_actual_grammar_not_verified")
    return value


async def execute(config, config_path, base_url, output, release_id, root, *,
                  expected_profile_sha256="4bbe4f79281d1b30ab2d45af539308bc57a4f911e5aad5d845c7b41eff9f07ab"):
    import httpx
    from PIL import Image, ImageDraw
    from forte.decision import build_prompt, validate_decisions
    from forte.decision.structured import DecisionConfig
    from forte.model.adapter import LocalModelConfig, LocalModelProvider, validate_private_endpoint
    from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef
    from serving.qwen38_identity import load_expected_runtime_identity
    from serving.qwen38_comparison_profile import runtime_profile_sha256

    validate_private_endpoint(base_url, ("model", "model-1", "model-2", "model-3"))
    parsed = urlsplit(base_url)
    readiness_url = f"{parsed.scheme}://{parsed.netloc}/readyz"
    key = os.environ.get("VLLM_API_KEY")
    headers = {"Authorization": "Bearer " + key} if key else {}
    catalog, fact_ids, event_ids, schema = schema_definition(config, root)
    from serving.qwen38_grammar import checked_schema, trusted_fact_order_schema
    derived_schema, derived_count = trusted_fact_order_schema(schema)
    config_sha = canonical(config)
    profile_sha = runtime_profile_sha256(config)
    need(re.fullmatch(r"[a-f0-9]{64}", expected_profile_sha256) is not None
         and profile_sha == expected_profile_sha256,
         "generated_gate_actual_comparison_profile_mismatch")
    expected = None
    if config["backend"] == "transformers_base_diagnostic":
        from serving.qwen38_untuned import expected_untuned_identity, validate_untuned_readiness
        from serving.qwen38_untuned_provider import UntunedModelProvider
        expected = expected_untuned_identity(config)
        factory = lambda media: UntunedModelProvider(config, media_root=media, base_url=base_url, api_key=key)
    else:
        expected = load_expected_runtime_identity(config_path)
        factory = lambda media: LocalModelProvider(LocalModelConfig(media_root=media, base_url=base_url,
            model=config["served_model_name"], expected_revision=config["model_revision"], api_key=key,
            max_frames=16, max_tokens=4096, seed=20261017, request_timeout_sec=config["limits"]["request_timeout_seconds"],
            max_attempts=1, readiness_cache_sec=0, expected_runtime_identity=tuple(expected.items()),
            chat_template_kwargs=(("enable_thinking", False), ("preserve_thinking", False))))

    async def read_ready():
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            result = await client.get(readiness_url, headers=headers, timeout=8)
            need(result.status_code == 200 and len(result.content) <= 1024 * 1024, "generated_gate_not_ready")
            value = result.json()
        need(all(value.get(name) == item for name, item in expected.items()) and value.get("ready") is True,
             "generated_gate_runtime_identity_mismatch")
        if config["backend"] == "transformers_base_diagnostic":
            validate_untuned_readiness(value, config)
        check_grammar(value, config)
        return value

    before = await read_ready()
    before_count = before["structured_decoding"]["requests_bound_to_grammar"]
    receipt = {"schema_version": "forte.qwen38.full19-generated-endpoint-gate.v1", "status": "started",
        "root_release_id": release_id, "source_archive_sha256": config["source_archive"]["sha256"],
        "runtime_image_sha256": config["runtime_image_sha256"], "config_file_sha256": sha(config_path),
        "config_sha256": config_sha, "runtime_profile_sha256": profile_sha,
        "expected_runtime_identity": expected, "actual_ready_identity": {name: before[name] for name in expected},
        "model_bearer_auth_configured": bool(key), "public_schema_sha256": canonical(schema),
        "original_grammar_schema_sha256": hashlib.sha256(checked_schema(schema).encode()).hexdigest(),
        "derived_grammar_schema_sha256": hashlib.sha256(json.dumps(derived_schema, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(), "canonical_fact_count": derived_count,
        "private_media_used": False, "generated_structure_only": True, "private_quality_measured": False,
        "visual_truth_measured": False, "event_accuracy_measured": False, "model_post_attempts": 0,
        "automatic_retry": False, "strict_json_schema_passed": False, "semantic_validation_passed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    # Claim the one-use slot before any generation request. An error consumes
    # this invocation; rerunning requires a new explicit root attempt identity.
    write_new(output / "generated.gate.attempt.json", receipt)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="full19_generated_", dir=output) as temporary:
            media = Path(temporary)
            owned = media / "full19_generated"
            owned.mkdir(mode=0o700)
            frames = []
            for index in range(16):
                path = owned / f"f{index:02d}.jpg"
                image = Image.new("RGB", (448, 448), (225, 225, 225))
                draw = ImageDraw.Draw(image)
                draw.rectangle((50 + index * 2, 100, 260, 300), fill=(180, 30, 90))
                draw.ellipse((270, 100, 380, 210), fill=(30, 100, 180))
                draw.text((15, 15), "GENERATED STRUCTURE TEST ONLY", fill=(0, 0, 0))
                image.save(path, "JPEG", quality=95)
                image.close()
                frames.append(FrameRef(frame_id=f"f{index:02d}", pts_sec=index * .5, path=path,
                    width=448, height=448, job_id="full19_generated", jpeg_sha256=sha(path)))
            facts = [FactDefinition(fact_id=fid, description=catalog["facts"][fid]["question_ru"] + " "
                                    + catalog["facts"][fid]["definition_ru"]) for fid in fact_ids]
            clip = ClipRequest(job_id="full19_generated", clip_id="full19_generated_clip",
                               frame_refs=frames, fact_definitions=facts)
            decision_config = DecisionConfig(compact_frame_ids=True, concise_observations=True)
            bundle = build_prompt(clip, reviewed_event_ids=event_ids,
                event_definitions={event: catalog["event_types"][event] for event in event_ids}, config=decision_config)
            need(bundle.response_schema == schema, "generated_gate_schema_factory_mismatch")
            async with factory(media) as provider:
                provider.config = replace(provider.config, max_attempts=1)
                original_request = provider._request
                async def counted_request(method, *args, **kwargs):
                    if method == "POST":
                        need(receipt["model_post_attempts"] == 0, "generated_gate_second_post_forbidden")
                        receipt["model_post_attempts"] += 1
                    return await original_request(method, *args, **kwargs)
                provider._request = counted_request
                response = await provider.infer(clip, prompt=bundle.prompt, response_schema=schema,
                    deadline=time.monotonic() + config["limits"]["request_timeout_seconds"])
            batch = validate_decisions(response.data, clip, reviewed_event_ids=event_ids,
                                       config=decision_config, finish_reason=response.finish_reason)
            need(len(batch.facts) == 19 and {fact.fact_id for fact in batch.facts} == set(fact_ids),
                 "generated_gate_fact_population_mismatch")
            need(receipt["model_post_attempts"] == 1, "generated_gate_one_post_not_executed")
            after = await read_ready()
            after_count = after["structured_decoding"]["requests_bound_to_grammar"]
            need(after_count == before_count + 1, "generated_gate_one_post_grammar_count_mismatch")
            receipt.update(status="passed", strict_json_schema_passed=True, semantic_validation_passed=True,
                facts_validated=19, rules_requested=4, event_types_requested=3, frames_generated=16,
                candidates_validated=len(batch.candidates), fact_state_counts=dict(Counter(fact.state for fact in batch.facts)),
                actual_grammar_requests_before=before_count, actual_grammar_requests_after=after_count,
                metrics=asdict(response.metrics), elapsed_seconds=time.monotonic() - started,
                actual_ready_record_sha256=canonical(after))
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__,
                       reason_code=str(error) if isinstance(error, GateError) else "generated_full19_endpoint_failed",
                       elapsed_seconds=time.monotonic() - started,
                       classification="generated_structure_gate_failure_not_private_quality")
        write_new(output / "full19generated.gate.receipt.json", receipt)
        raise
    write_new(output / "full19generated.gate.receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/app"))
    parser.add_argument("--base-url", default="http://model-3:8000/v1")
    parser.add_argument("--mode", choices=("prepare", "execute"), default="prepare")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root-release-id")
    parser.add_argument("--expected-runtime-profile-sha256",
                        default="4bbe4f79281d1b30ab2d45af539308bc57a4f911e5aad5d845c7b41eff9f07ab")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source_root), str(args.source_root / "src")]
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    _, facts, events, schema = schema_definition(config, args.source_root)
    if args.mode == "prepare":
        print(json.dumps({"status": "prepared", "facts": len(facts), "event_types": len(events),
                          "schema_sha256": canonical(schema), "planned_model_posts": 1, "actual_model_posts": 0,
                          "private_media_used": False, "gpu_access": False}))
        return 0
    need(args.output is not None and not args.output.exists(), "generated_gate_new_private_output_required")
    need(args.root_release_id and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", args.root_release_id),
         "generated_gate_root_release_id_required")
    args.output.mkdir(mode=0o700)
    result = asyncio.run(execute(config, args.config, args.base_url, args.output, args.root_release_id,
                                args.source_root, expected_profile_sha256=args.expected_runtime_profile_sha256))
    print(json.dumps({key: result[key] for key in ("status", "facts_validated", "model_post_attempts", "elapsed_seconds",
                                                  "strict_json_schema_passed", "semantic_validation_passed")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "reason_code": str(error) if isinstance(error, GateError)
                          else "generated_gate_control_error", "error_type": type(error).__name__,
                          "generated_descriptions_logged": False}))
        raise SystemExit(2)
