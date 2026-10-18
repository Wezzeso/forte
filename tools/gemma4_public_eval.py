"""Offline paired Gemma/LoRA object-class evaluation on public validation groups.

Run on the user's rented Linux GPU host only. Images are read by the local
processor, never displayed, and model text is reduced to a closed class enum.
This is a component diagnostic, not a bank-video verdict or release gate.
Use --self-test for synthetic, image-free checks on any machine.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time


CLASSES = ("id_card", "banknote", "other", "unclear")
GROUP_FIELDS = (
    "source_group", "source_image_group", "template_group",
    "physical_document_group", "scene_group",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def _unique_pairs(pairs):
    answer = {}
    for key, value in pairs:
        if key in answer:
            raise ValueError("duplicate_json_key")
        answer[key] = value
    return answer


def _reject_constant(_value):
    raise ValueError("nonfinite_json")


def strict_class(value: str) -> str:
    """Accept only a complete object; never repair fences or thinking text."""
    parsed = json.loads(value, object_pairs_hook=_unique_pairs,
                        parse_constant=_reject_constant)
    if (not isinstance(parsed, dict) or set(parsed) != {"object_class"}
            or not isinstance(parsed["object_class"], str)
            or parsed["object_class"] not in CLASSES):
        raise ValueError("invalid_closed_schema")
    return parsed["object_class"]


def read_manifest(path: Path, train_path: Path | None = None) -> list[dict]:
    rows = []
    for source in ([train_path, path] if train_path else [path]):
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line, object_pairs_hook=_unique_pairs,
                                           parse_constant=_reject_constant))
    validate_manifest(rows)
    return rows


def validate_manifest(rows: list[dict]) -> None:
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("empty_or_invalid_manifest")
    ids = set()
    prompts = set()
    for row in rows:
        identifier = row.get("example_id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("missing_or_duplicate_example_id")
        ids.add(identifier)
        if row.get("split") not in ("train", "validation"):
            raise ValueError("unsupported_manifest_split")
        if not isinstance(row.get("source_dataset"), str) or not row["source_dataset"]:
            raise ValueError("missing_public_source_dataset")
        if any(row.get(field) is not None and not isinstance(row[field], str)
               for field in GROUP_FIELDS):
            raise ValueError("invalid_public_group_identifier")
        if not isinstance(row.get("source_group"), str) or not row["source_group"]:
            raise ValueError("missing_source_group")
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            raise ValueError("missing_public_prompt")
        prompts.add(row["prompt"])
        if (not isinstance(row.get("target"), dict)
                or set(row["target"]) != {"object_class"}
                or row["target"]["object_class"] not in CLASSES):
            raise ValueError("invalid_public_target")
        if row.get("label_source") != "published_object_class_and_box":
            raise ValueError("not_published_object_annotation")
        if not isinstance(row.get("license_spdx"), str) or not row["license_spdx"]:
            raise ValueError("missing_source_license")
        if not isinstance(row.get("image_path"), str) or not row["image_path"]:
            raise ValueError("missing_image_path")
        expected_hash = row.get("image_sha256")
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                or any(c not in "0123456789abcdef" for c in expected_hash)):
            raise ValueError("missing_or_invalid_image_sha256")
        box = row.get("crop_box_xyxy")
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("invalid_public_crop_box")
    if len(prompts) != 1:
        raise ValueError("public_prompt_must_be_fixed_for_every_class")
    for field in (*GROUP_FIELDS, "image_sha256"):
        train = {row[field] for row in rows if row["split"] == "train" and row.get(field)}
        validation = {row[field] for row in rows
                      if row["split"] == "validation" and row.get(field)}
        if train & validation:
            raise ValueError("public_train_validation_group_overlap:" + field)
    if not any(row["split"] == "train" for row in rows):
        raise ValueError("missing_train_inventory_for_overlap_audit")
    if not any(row["split"] == "validation" for row in rows):
        raise ValueError("missing_validation_inventory")


def metrics(rows: list[dict]) -> dict:
    """Invalid/failed rows stay in every applicable accuracy denominator."""
    confusion = {truth: {pred: 0 for pred in (*CLASSES, "failed")} for truth in CLASSES}
    for row in rows:
        confusion[row["expected"]][row["predicted"] or "failed"] += 1
    class_metrics = {}
    for name in CLASSES:
        support = sum(confusion[name].values())
        true_positive = confusion[name][name]
        predicted = sum(confusion[truth][name] for truth in CLASSES)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else None
        f1 = (2 * precision * recall / (precision + recall)
              if recall is not None and precision + recall else 0.0)
        class_metrics[name] = {"support": support, "precision": precision,
                               "recall": recall, "f1": f1 if support else None}
    groups = {}
    for row in rows:
        groups.setdefault(row["source_group"], []).append(row)
    valid = sum(row["status"] == "done" for row in rows)
    correct = sum(row["expected"] == row["predicted"] for row in rows)
    walls = [row["wall_s"] for row in rows]
    supported_f1 = [row["f1"] for row in class_metrics.values() if row["support"]]
    return {
        "examples": len(rows), "processed": valid, "failed": len(rows) - valid,
        "strict_json_rate": valid / len(rows) if rows else None,
        "correct": correct, "accuracy": correct / len(rows) if rows else None,
        "macro_f1_supported_classes": statistics.mean(supported_f1) if supported_f1 else None,
        "per_class": class_metrics, "confusion": confusion,
        "banknote_to_id_card": confusion["banknote"]["id_card"],
        "source_groups": len(groups),
        "group_macro_accuracy": statistics.mean(
            sum(row["expected"] == row["predicted"] for row in values) / len(values)
            for values in groups.values()) if groups else None,
        "all_views_correct_groups": sum(all(row["expected"] == row["predicted"]
                                             for row in values) for values in groups.values()),
        "latency_s": {"p50": statistics.median(walls) if walls else None,
                      "p95": sorted(walls)[max(0, math.ceil(.95 * len(walls)) - 1)] if walls else None,
                      "max": max(walls) if walls else None},
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
    }


def paired_metrics(base: list[dict], adapter: list[dict]) -> dict:
    baseline = {row["example_id"]: row for row in base}
    candidate = {row["example_id"]: row for row in adapter}
    if set(baseline) != set(candidate):
        raise ValueError("paired_examples_differ")
    counts = Counter()
    for key, old in baseline.items():
        new = candidate[key]
        old_ok = old["predicted"] == old["expected"]
        new_ok = new["predicted"] == new["expected"]
        counts[(old_ok, new_ok)] += 1
    return {"examples": len(base), "both_correct": counts[(True, True)],
            "both_wrong_or_failed": counts[(False, False)],
            "adapter_only_correct": counts[(False, True)],
            "base_only_correct": counts[(True, False)],
            "accuracy_delta": ((counts[(False, True)] - counts[(True, False)]) / len(base)
                               if base else None)}


def _write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def self_test() -> None:
    assert strict_class('{"object_class":"id_card"}') == "id_card"
    for invalid in ('```json\n{"object_class":"id_card"}\n```',
                    '<think>x</think>{"object_class":"banknote"}',
                    '{"object_class":"banknote","extra":true}',
                    '{"object_class":"id_card","object_class":"banknote"}',
                    '{"object_class":true}', '{"object_class":NaN}'):
        try:
            strict_class(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("strict_json_rejected_fixture_failed")
    row = {"example_id": "a", "source_group": "a", "source_dataset": "synthetic",
           "split": "train", "prompt": "fixed",
           "target": {"object_class": "id_card"}, "license_spdx": "MIT",
           "label_source": "published_object_class_and_box", "image_path": "fake.jpg",
           "image_sha256": "a" * 64, "crop_box_xyxy": [0, 0, 2, 2]}
    other = {**row, "example_id": "b", "source_group": "b", "split": "validation",
             "image_sha256": "b" * 64}
    validate_manifest([row, other])
    try:
        validate_manifest([row, {**other, "source_group": "a"}])
    except ValueError:
        pass
    else:
        raise AssertionError("group_leakage_fixture_failed")
    good = {"example_id": "a", "source_group": "a", "expected": "id_card",
            "predicted": "id_card", "status": "done", "wall_s": 1.0, "generated_tokens": 8}
    failed = {**good, "example_id": "b", "predicted": None, "status": "failed"}
    score = metrics([good, failed])
    assert score["accuracy"] == .5 and score["failed"] == 1
    assert score["strict_json_rate"] == .5
    assert paired_metrics([failed], [{**failed, "predicted": "id_card", "status": "done"}])[
        "adapter_only_correct"] == 1
    print(json.dumps({"status": "passed", "synthetic_only": True,
                      "checks": ["strict_json", "group_overlap", "failure_denominator", "paired_delta"]}))


def run(args) -> int:
    if platform.system() != "Linux" or not args.rented_host_id:
        raise ValueError("run_only_on_named_rented_linux_host")
    if args.out.exists():
        raise ValueError("output_already_exists_use_new_record_path")
    if not args.model.is_dir() or not args.adapter.is_dir():
        raise ValueError("local_base_and_adapter_directories_required")
    if args.repeats < 1 or args.max_new_tokens < 1 or args.max_length < 1:
        raise ValueError("invalid_generation_limits")
    rows = read_manifest(args.manifest, args.train_manifest)
    validation = [row for row in rows if row["split"] == "validation"]
    root = args.images_root.resolve() if args.images_root else args.manifest.resolve().parent
    checked = {}
    for row in validation:
        path = (root / row["image_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("public_image_outside_declared_root")
        if path in checked and checked[path] != row["image_sha256"]:
            raise ValueError("contradictory_public_image_digest")
        if path not in checked:
            if digest(path) != row["image_sha256"]:
                raise ValueError("public_image_digest_mismatch")
            checked[path] = row["image_sha256"]
    config_path = args.adapter / "adapter_config.json"
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if adapter_config.get("bias", "none") != "none" or adapter_config.get("modules_to_save"):
        raise ValueError("paired_disable_adapter_requires_lora_only_without_bias")
    adapter_files = {path.name: digest(path) for path in sorted(args.adapter.iterdir())
                     if path.is_file() and path.suffix in (".json", ".safetensors")}
    if not any(name.endswith(".safetensors") for name in adapter_files):
        raise ValueError("missing_safetensors_adapter_weights")
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.gemma4_lora_train import Gemma4Collator, quiet_libraries
    with quiet_libraries():
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    if not torch.cuda.is_available():
        raise ValueError("cuda_gpu_required")
    started = time.monotonic()
    with quiet_libraries():
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                                 trust_remote_code=False)
        base = Gemma4ForConditionalGeneration.from_pretrained(
            str(args.model), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": args.device}, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False,
                                         local_files_only=True)
        model.eval()
    collator = Gemma4Collator(processor, root, max_length=args.max_length,
                              max_soft_tokens=args.max_soft_tokens)
    result = {
        "schema": "forte.gemma4-public-component-eval.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "processing",
        "scope": "reused_public_validation_source_groups;object_class_component_only",
        "fresh_lockbox_claim": False, "bank_video_accuracy_claim": False,
        "violation_enablement_claim": False, "cross_validation_claim": False,
        "provenance": {"rented_host_id": args.rented_host_id,
                       "manifest_sha256": digest(args.manifest),
                       "train_manifest_sha256": digest(args.train_manifest) if args.train_manifest else None,
                       "code_sha256": digest(Path(__file__)),
                       "collator_code_sha256": digest(Path(__file__).resolve().parents[1] / "training/gemma4_lora_train.py"),
                       "base_revision_declared": args.base_revision, "base_path": str(args.model),
                       "base_metadata_sha256": {name: digest(args.model / name) for name in
                           ("config.json", "generation_config.json", "model.safetensors.index.json",
                            "processor_config.json", "preprocessor_config.json", "tokenizer_config.json")
                           if (args.model / name).is_file()},
                       "base_full_weight_hash_verified_by_this_evaluator": False,
                       "adapter_file_sha256": adapter_files, "public_image_hashes_verified": len(checked),
                       "licenses": sorted({row["license_spdx"] for row in rows}),
                       "source_datasets": sorted({row["source_dataset"] for row in rows}),
                       "prompt_sha256": hashlib.sha256(rows[0]["prompt"].encode()).hexdigest(),
                       "train_examples": sum(row["split"] == "train" for row in rows),
                       "validation_examples": len(validation),
                       "group_fields_audited": [*GROUP_FIELDS, "image_sha256"],
                       "runtime_versions": {name: importlib.metadata.version(name) for name in
                                            ("torch", "transformers", "peft", "accelerate", "pillow")},
                       "python_version": platform.python_version(),
                       "gpu_name": torch.cuda.get_device_name(args.device),
                       "cuda_version": torch.version.cuda},
        "generation": {"do_sample": False, "seed": 0, "batch_size": 1,
                       "max_new_tokens": args.max_new_tokens, "max_length": args.max_length,
                       "max_soft_tokens": args.max_soft_tokens, "dtype": "bfloat16",
                       "attention": "sdpa", "strict_json_forced_during_generation": False,
                       "class_enum": list(CLASSES)},
        "load_s": round(time.monotonic() - started, 4), "runs": [],
    }
    for repeat in range(args.repeats):
        pair = {}
        for phase in ("base", "adapter"):
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
            observations = []
            phase_started = time.monotonic()
            context = model.disable_adapter() if phase == "base" else nullcontext()
            with context, torch.inference_mode():
                for number, row in enumerate(validation, start=1):
                    observation = {"example_id": row["example_id"], "source_group": row["source_group"],
                                   "source_dataset": row["source_dataset"],
                                   "short_side": row.get("transform", {}).get("short_side", 0),
                                   "expected": row["target"]["object_class"], "predicted": None,
                                   "status": "failed", "failure": None, "generated_tokens": 0}
                    tick = time.monotonic()
                    try:
                        # Only visual inputs, a fixed prompt and geometry reach perception.
                        # Public target labels and grouping metadata remain scorer-only.
                        with quiet_libraries():
                            inputs = collator.prompt_inputs({key: row[key] for key in
                                ("image_path", "crop_box_xyxy", "transform", "prompt") if key in row})
                            inputs = {key: value.to(args.device) for key, value in inputs.items()}
                            generated = model.generate(**inputs, do_sample=False,
                                                       max_new_tokens=args.max_new_tokens)
                            continuation = generated[0, inputs["input_ids"].shape[-1]:]
                            observation["generated_tokens"] = int(continuation.numel())
                            raw = processor.tokenizer.decode(continuation, skip_special_tokens=True)
                            observation["predicted"] = strict_class(raw)
                        observation["status"] = "done"
                    except torch.cuda.OutOfMemoryError:
                        observation["failure"] = "cuda_out_of_memory"
                        torch.cuda.empty_cache()
                    except (ValueError, KeyError, TypeError, IndexError):
                        observation["failure"] = "invalid_input_or_closed_json"
                    except Exception as exc:
                        # Exception messages may contain response text. Only the type is retained.
                        observation["failure"] = "technical:" + type(exc).__name__
                    observation["wall_s"] = round(time.monotonic() - tick, 4)
                    observations.append(observation)
                    if number % 10 == 0 or number == len(validation):
                        print(json.dumps({"phase": phase, "repeat": repeat + 1,
                                          "completed": number, "expected": len(validation),
                                          "failed": sum(item["status"] != "done" for item in observations)}), flush=True)
            entry = {"phase": phase, "repeat": repeat + 1,
                     "wall_s": round(time.monotonic() - phase_started, 4),
                     "gpu_memory_peak_bytes": torch.cuda.max_memory_allocated(),
                     "metrics": metrics(observations), "results": observations}
            entry["metrics_by_short_side"] = {
                str(short_side): metrics([row for row in observations if row["short_side"] == short_side])
                for short_side in sorted({row["short_side"] for row in observations})}
            entry["metrics_by_dataset"] = {
                source: metrics([row for row in observations if row["source_dataset"] == source])
                for source in sorted({row["source_dataset"] for row in observations})}
            result["runs"].append(entry)
            pair[phase] = observations
            _write(args.out, result)
        result.setdefault("paired", []).append({"repeat": repeat + 1,
                                                 **paired_metrics(pair["base"], pair["adapter"])})
        _write(args.out, result)
    result["wall_s"] = round(time.monotonic() - started, 4)
    result["status"] = ("done" if all(entry["metrics"]["failed"] == 0 for entry in result["runs"])
                        else "failed")
    _write(args.out, result)
    print(json.dumps({"status": result["status"], "paired": result["paired"]}), flush=True)
    return 0 if result["status"] == "done" else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--train-manifest", type=Path,
                        help="Frozen train inventory, required when --manifest contains validation only")
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rented-host-id")
    parser.add_argument("--base-revision", default="842da3794eaa0b77d5f08bae87a17459d91ff475")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-soft-tokens", type=int, default=280)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not all((args.manifest, args.model, args.adapter, args.out, args.rented_host_id)):
        parser.error("--manifest --model --adapter --out --rented-host-id are required")
    output_existed = args.out.exists()
    try:
        return run(args)
    except Exception as exc:
        # Avoid traceback strings containing images, model text, secrets or raw records.
        failure = {"status": "failed", "failure": type(exc).__name__}
        if not output_existed:
            try:
                partial = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {
                    "schema": "forte.gemma4-public-component-eval.v1",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "fresh_lockbox_claim": False, "bank_video_accuracy_claim": False,
                }
                _write(args.out, {**partial, **failure})
            except (OSError, ValueError):
                pass
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
