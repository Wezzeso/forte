"""Offline BF16 Qwen30 LoRA on independent synthetic typed timelines.

This teaches reasoning over symbolic evidence, not better visual perception.
No bank media, bank labels, public images, or model scene descriptions are read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any

from training import qwen30_lora_train as base
from training.data import masked_completion_labels
from training.guards import TrainingGateError, canonical_sha256, file_sha256, require

SCHEMA = "forte.qwen30-procedure-lora.run.v1"
TYPES = frozenset({"NO_RECOUNT", "KEY_LEFT_IN_SAFE", "COUNT_WITHOUT_CLIENT", "NO_CLIENT_ID",
    "NO_AUTHENTICITY_CHECK", "CASH_LEFT_UNATTENDED", "COLLECTOR_NO_DOCUMENTS",
    "CASH_HANDOVER_IRREGULAR", "WORKSTATION_UNLOCKED"})
DECISIONS = frozenset({"supported", "not_supported", "unassessable"})
RESPONSE_FIELDS = ("type_id", "decision", "anchor_index")
USER_PREFIX = "Evaluate the requested check using only this structured event timeline:\n"
RESPONSE_SCHEMA = {"type": "object", "properties": {
    "type_id": {"type": "string", "enum": sorted(TYPES)},
    "decision": {"type": "string", "enum": sorted(DECISIONS)},
    "anchor_index": {"type": "integer", "minimum": -1}},
    "required": list(RESPONSE_FIELDS), "additionalProperties": False}


def render_messages(row: dict, include_target: bool = False) -> list[dict]:
    messages = [{"role": "system", "content": row["system"]},
                {"role": "user", "content": USER_PREFIX + json.dumps(row["input"], sort_keys=True,
                    separators=(",", ":"), allow_nan=False)}]
    if include_target:
        text, _ = target_text(row)
        messages.append({"role": "assistant", "content": text})
    return messages


def target_text(row: dict) -> tuple[str, list[tuple[int, int]]]:
    text, spans = "{", []
    for index, name in enumerate(RESPONSE_FIELDS):
        if index:
            text += ","
        text += json.dumps(name) + ":"
        start = len(text)
        text += json.dumps(row["target"][name], separators=(",", ":"), allow_nan=False)
        spans.append((start, len(text)))
    return text + "}", spans


def _typed(value: Any, depth: int = 0) -> None:
    require(depth <= 8, "procedure_input_too_deep")
    if isinstance(value, dict):
        require(len(value) <= 64 and all(isinstance(k, str) and re.fullmatch(r"[A-Za-z0-9_]+", k)
                for k in value), "procedure_typed_keys_required")
        for item in value.values():
            _typed(item, depth + 1)
    elif isinstance(value, list):
        require(len(value) <= 64, "procedure_timeline_too_long")
        for item in value:
            _typed(item, depth + 1)
    elif isinstance(value, str):
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value)), "procedure_enums_not_free_text_required")
    else:
        require(value is None or type(value) in {bool, int, float}, "procedure_typed_scalar_required")
        if type(value) is float:
            require(math.isfinite(value), "procedure_finite_values_required")


def validate_rows(rows: list[dict]) -> dict:
    require(bool(rows), "procedure_manifest_empty")
    groups, inputs, counts = defaultdict(set), defaultdict(set), Counter()
    policies = set()
    for row in rows:
        require(isinstance(row, dict) and row.get("origin") == "synthetic"
                and row.get("modality") == "text" and row.get("split") in {"train", "validation"},
                "procedure_independent_synthetic_text_required")
        group = row.get("template_group")
        require(isinstance(group, str) and bool(group), "procedure_template_group_required")
        groups[group].add(row["split"])
        require(isinstance(row.get("system"), str) and 30 <= len(row["system"]) <= 24000,
                "procedure_fixed_policy_required")
        policies.add(row["system"])
        source = row.get("input")
        require(isinstance(source, dict) and set(source) == {"requested_type", "timeline", "coverage"}
                and source["requested_type"] in TYPES and isinstance(source["timeline"], list)
                and bool(source["timeline"]) and isinstance(source["coverage"], dict),
                "procedure_typed_evidence_schema_required")
        _typed(source)
        inputs[canonical_sha256(source)].add(row["split"])
        target = row.get("target")
        require(isinstance(target, dict) and set(target) == set(RESPONSE_FIELDS)
                and target["type_id"] == source["requested_type"] and target["decision"] in DECISIONS
                and type(target["anchor_index"]) is int, "procedure_response_schema_invalid")
        if target["decision"] == "supported":
            require(0 <= target["anchor_index"] < len(source["timeline"]), "procedure_supported_anchor_invalid")
        else:
            require(target["anchor_index"] == -1, "procedure_unconfirmed_anchor_must_be_minus_one")
        counts[(row["split"], target["type_id"], target["decision"])] += 1
    require(len(policies) == 1, "procedure_per_example_policy_forbidden")
    require(all(len(value) == 1 for value in groups.values()) and all(len(value) == 1 for value in inputs.values()),
            "procedure_template_or_input_leakage")
    require({split for values in groups.values() for split in values} == {"train", "validation"},
            "procedure_train_validation_required")
    return {"rows": len(rows), "template_groups": len(groups),
        "groups_by_split": dict(Counter(next(iter(values)) for values in groups.values())),
        "counts": {f"{split}/{kind}/{decision}": count for (split, kind, decision), count in sorted(counts.items())},
        "system_sha256": canonical_sha256(next(iter(policies)))}


def resolve_config(path: Path) -> dict:
    config = base.read_json(path)
    require(isinstance(config, dict) and config.get("schema_version") == "forte.qwen30-procedure-lora.config.v1",
            "procedure_config_schema_invalid")
    for key in ("model_dir", "model_download_receipt", "manifest", "output_dir"):
        value = config.get(key)
        require(isinstance(value, str) and bool(value) and "://" not in value, "procedure_local_paths_required")
        config[key] = str((path.resolve().parent / value).resolve())
    require(config.get("model_id") == base.MODEL_ID and config.get("revision") == base.REVISION
            and config.get("quantization") == "bf16" and config.get("runtime_versions") == base.PINS,
            "procedure_pinned_bf16_base_required")
    require(config.get("training_data_policy") == "independent_synthetic_typed_timelines"
            and config.get("target_loss_policy") == "procedural_response_values_only"
            and config.get("adapter_scope") == "text_only_procedure_reasoning"
            and config.get("bank_video_labels_used") is False and config.get("production_enabled") is False,
            "procedure_no_bank_data_or_visual_claims")
    require(config.get("optimizer_steps") == 100 and config.get("max_training_seconds") == 600
            and config.get("pilot_steps") == 10 and config.get("pilot_max_seconds") == 90
            and config.get("seed") == 0 and config.get("lora_rank") == 8,
            "procedure_predeclared_steps_budget_seed_rank_required")
    require(re.fullmatch(r"[0-9a-f]{64}", str(config.get("expected_manifest_sha256"))) is not None,
            "procedure_manifest_hash_pin_required")
    require(type(config.get("grad_accum_steps")) is int and 1 <= config["grad_accum_steps"] <= 8
            and type(config.get("max_length")) is int and 128 <= config["max_length"] <= 8192,
            "procedure_batch_budget_invalid")
    require(0 < config["learning_rate"] <= .001 and 0 <= config["lora_dropout"] <= .2
            and 0 < config["max_grad_norm"] <= 10 and 0 <= config["warmup_steps"] < 100
            and config["checkpoint_every"] in {10, 20, 25, 50, 100}, "procedure_hyperparameters_invalid")
    require(base.deadline(config).utcoffset().total_seconds() == 0, "procedure_utc_deadline_required")
    return config


def preflight(config: dict) -> dict:
    require(file_sha256(Path(config["manifest"])) == config["expected_manifest_sha256"],
            "procedure_manifest_hash_mismatch")
    rows = base.load_rows(Path(config["manifest"]))
    stats = validate_rows(rows)
    root = Path(config["model_dir"])
    model_config, receipt = base.read_json(root / "config.json"), base.read_json(Path(config["model_download_receipt"]))
    require(file_sha256(root / "config.json") == base.CONFIG_SHA256
            and model_config.get("architectures") == [base.MODEL_CLASS]
            and model_config.get("model_type") == "qwen3_vl_moe"
            and not model_config.get("auto_map") and not model_config.get("quantization_config"),
            "procedure_base_architecture_mismatch")
    weights = receipt.get("weight_manifest")
    require(receipt.get("repo") == base.MODEL_ID and receipt.get("revision") == base.REVISION
            and receipt.get("complete") is True and receipt.get("license") == "apache-2.0"
            and isinstance(weights, list) and bool(weights)
            and receipt.get("weight_manifest_sha256") == canonical_sha256(weights), "procedure_base_receipt_mismatch")
    for item in weights:
        require(item["name"] == Path(item["name"]).name and item["name"].endswith(".safetensors")
                and (root / item["name"]).stat().st_size == item["bytes"], "procedure_base_weight_inventory_invalid")
        require(file_sha256(root / item["name"]) == item["sha256"], "procedure_base_weight_hash_mismatch")
    helpers = {Path(module.__file__).name: file_sha256(Path(module.__file__)) for module in (base,)}
    helpers["data.py"] = file_sha256(Path(__file__).with_name("data.py"))
    helpers["guards.py"] = file_sha256(Path(__file__).with_name("guards.py"))
    return {"rows": rows, "stats": stats, "model_config": model_config, "weight_receipt": receipt,
            "source_sha256": file_sha256(Path(__file__)), "helper_source_sha256": helpers}


class ProcedureCollator:
    def __init__(self, processor: Any, max_length: int):
        self.processor, self.tokenizer, self.max_length = processor, processor.tokenizer, max_length

    def prompt_inputs(self, row: dict) -> dict:
        text = self.processor.apply_chat_template(render_messages(row), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return dict(self.tokenizer(text, add_special_tokens=False, return_tensors="pt"))

    def __call__(self, row: dict) -> dict:
        import torch
        prompt = self.processor.apply_chat_template(render_messages(row), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        full = self.processor.apply_chat_template(render_messages(row, True), tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        encoded = dict(self.tokenizer(full, add_special_tokens=False, return_tensors="pt"))
        prefix = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        require(encoded["input_ids"].shape[-1] <= self.max_length, "procedure_sequence_too_long_no_truncation")
        answer, spans = target_text(row)
        labels = masked_completion_labels(encoded["input_ids"][0].tolist(), prefix,
            {151652, 151653, 151655, 151656}, tokenizer=self.tokenizer,
            prompt_text=prompt, full_text=full, target_text=answer, target_spans=spans,
            attention_mask=encoded["attention_mask"][0].tolist())
        encoded["labels"] = torch.tensor([labels], dtype=torch.long)
        require(set(encoded) <= {"input_ids", "attention_mask", "labels", "token_type_ids"},
                "procedure_text_only_batch_required")
        encoded.pop("token_type_ids", None)
        return encoded


def checkpoint(model: Any, optimizer: Any, output: Path, run: dict) -> str:
    import torch
    name = f"checkpoint-{run['optimizer_steps']:04d}"
    destination, staging = output / name, output / (name + ".partial")
    require(not destination.exists() and not staging.exists(), "procedure_checkpoint_already_exists")
    staging.mkdir()
    model.save_pretrained(staging, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, staging / "optimizer.pt")
    base.write_json(staging / "trainer_state.json", {key: run[key] for key in (
        "schema_version", "optimizer_steps", "source_sha256", "helper_source_sha256", "config_sha256",
        "manifest_sha256", "weight_manifest_sha256", "initial_adapter_sha256", "frozen_before_sha256", "steps")})
    base.write_json(staging / "checkpoint_complete.json", {"schema_version": "forte.qwen30-procedure-checkpoint.v1",
        "complete": True, "files": {p.name: file_sha256(p) for p in staging.iterdir() if p.is_file()}})
    staging.rename(destination)
    return name


def train(config: dict, checked: dict) -> dict:
    output = Path(config["output_dir"])
    require(not output.exists(), "procedure_run_output_exists")
    output.mkdir(parents=True)
    run = {"schema_version": SCHEMA, "state": "loading", "model_id": base.MODEL_ID,
        "mode": "train", "qualification_state": "not_evaluated",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision": base.REVISION, "base_dtype": "bfloat16", "quantization": "bf16",
        "config_sha256": canonical_sha256(config), "manifest_sha256": config["expected_manifest_sha256"],
        "source_sha256": checked["source_sha256"], "helper_source_sha256": checked["helper_source_sha256"],
        "weight_manifest_sha256": checked["weight_receipt"]["weight_manifest_sha256"],
        "training_data_policy": config["training_data_policy"], "target_loss_policy": config["target_loss_policy"],
        "adapter_scope": config["adapter_scope"], "bank_video_labels_used": False, "production_enabled": False,
        "visual_perception_improvement_claimed": False, "public_data": checked["stats"],
        "runtime_versions": base.PINS, "optimizer_steps": 0, "steps": [],
        "checkpoint_selection": "fixed_final_step_or_resource_budget_no_bank_scores",
        "training_hyperparameter_count": 11, "training_parameters_tuned_to_bank_labels": 0,
        "training_hyperparameters_predeclared": ["optimizer_steps", "grad_accum_steps", "learning_rate",
            "warmup_steps", "weight_decay", "max_grad_norm", "lora_rank", "lora_alpha", "lora_dropout", "max_length", "seed"]}
    base.write_json(output / "run.json", run)
    optimizer = None
    try:
        require(datetime.now(timezone.utc) < base.deadline(config), "procedure_deadline_passed")
        base.emit({"phase": "procedure_model_load"})
        with base.quiet_libraries():
            model, processor, trainable, targets = base.load_training_model(config, checked)
        import torch
        collator = ProcedureCollator(processor, config["max_length"])
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config["learning_rate"], weight_decay=config["weight_decay"])
        run.update(initial_adapter_sha256=base.adapter_digest(model), trainable_parameters=sum(p.numel() for _, p in trainable),
                   target_modules=targets, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
        base.emit({"phase": "procedure_frozen_parameter_digest"})
        run["frozen_before_sha256"] = base.frozen_digest(model)
        rows = [row for row in checked["rows"] if row["split"] == "train"]
        buckets = defaultdict(list)
        for index, row in enumerate(rows):
            buckets[(row["target"]["type_id"], row["target"]["decision"])].append(index)
        rng, cursor = random.Random(config["seed"]), Counter()
        keys = sorted(buckets)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        order = []
        for i in range(config["optimizer_steps"] * config["grad_accum_steps"]):
            key = keys[i % len(keys)]
            order.append(buckets[key][cursor[key] % len(buckets[key])])
            cursor[key] += 1
        model.train()
        for name, module in model.named_modules():
            if name.endswith(".visual"):
                module.eval()
        start = time.monotonic()
        run["state"] = "training"
        for step in range(config["optimizer_steps"]):
            optimizer.zero_grad(set_to_none=True)
            losses, tick, interrupted = [], time.monotonic(), False
            lr = config["learning_rate"] * min(1, (step + 1) / max(1, config["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            for micro in range(config["grad_accum_steps"]):
                elapsed = time.monotonic() - start
                if datetime.now(timezone.utc) >= base.deadline(config) or elapsed >= config["max_training_seconds"]:
                    run["state"], interrupted = "stopped_budget", True
                    break
                if step < 10 and elapsed >= 90:
                    run["state"], interrupted = "stopped_throughput_pilot", True
                    break
                with base.quiet_libraries():
                    batch = {key: value.to("cuda:0") for key, value in collator(rows[order[step * config["grad_accum_steps"] + micro]]).items()}
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = base.completion_loss(model, batch)
                    require(bool(torch.isfinite(loss).item()), "procedure_nonfinite_loss")
                    losses.append(float(loss.detach().item()))
                    (loss / config["grad_accum_steps"]).backward()
                del loss, batch
            if interrupted:
                optimizer.zero_grad(set_to_none=True)
                break
            grads = [p.grad for _, p in trainable if p.grad is not None]
            require(bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
                    and any(bool(torch.count_nonzero(g)) for g in grads), "procedure_invalid_gradients")
            norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["max_grad_norm"]).item())
            require(math.isfinite(norm), "procedure_nonfinite_gradient_norm")
            elapsed = time.monotonic() - start
            if datetime.now(timezone.utc) >= base.deadline(config) or elapsed >= 600 or (step < 10 and elapsed >= 90):
                run["state"] = "stopped_throughput_pilot" if step < 10 and elapsed >= 90 else "stopped_budget"
                optimizer.zero_grad(set_to_none=True)
                break
            optimizer.step()
            torch.cuda.synchronize()
            require(all(bool(torch.isfinite(p).all()) for _, p in trainable), "procedure_nonfinite_adapter")
            metric = {"step": step + 1, "mean_loss": sum(losses) / len(losses), "gradient_norm": norm,
                "learning_rate": lr, "seconds": time.monotonic() - tick,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2}
            run["optimizer_steps"] = step + 1
            run["steps"].append(metric)
            base.write_json(output / "run.json", run)
            base.emit(metric)
            if (step + 1) % config["checkpoint_every"] == 0:
                with base.quiet_libraries():
                    run["last_checkpoint"] = checkpoint(model, optimizer, output, run)
        require(run["optimizer_steps"] > 0, "procedure_no_completed_step")
        optimizer.zero_grad(set_to_none=True)
        run["final_adapter_sha256"] = base.adapter_digest(model)
        require(run["final_adapter_sha256"] != run["initial_adapter_sha256"], "procedure_adapter_unchanged")
        base.emit({"phase": "procedure_frozen_parameter_digest_final"})
        run["frozen_after_sha256"] = base.frozen_digest(model)
        require(run["frozen_after_sha256"] == run["frozen_before_sha256"], "procedure_frozen_parameters_changed")
        if not (output / f"checkpoint-{run['optimizer_steps']:04d}").exists():
            with base.quiet_libraries():
                run["last_checkpoint"] = checkpoint(model, optimizer, output, run)
        with base.quiet_libraries():
            model.save_pretrained(output / "adapter", safe_serialization=True)
            processor.tokenizer.save_pretrained(output / "tokenizer")
        run["adapter_files"] = {p.name: file_sha256(p) for p in (output / "adapter").iterdir() if p.is_file()}
        run["tokenizer_files"] = {p.name: file_sha256(p) for p in (output / "tokenizer").iterdir() if p.is_file()}
        run["training_seconds"] = time.monotonic() - start
        if run["state"] == "training":
            run["state"] = "trained_not_qualified"
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        base.write_json(output / "run.json", run)
        return {"ok": True, "state": run["state"], "optimizer_steps": run["optimizer_steps"],
                "adapter_sha256": run["final_adapter_sha256"], "trainable_parameters": run["trainable_parameters"]}
    except Exception as error:
        run.update(state="failed", error_type=type(error).__name__)
        if isinstance(error, TrainingGateError):
            run["error_code"] = str(error)
        base.write_json(output / "run.json", run)
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def synthetic_check() -> dict:
    system = "Use only the typed evidence. Coverage is required for absence conclusions."
    rows = [{"origin": "synthetic", "modality": "text", "split": split, "template_group": split,
        "system": system, "input": {"requested_type": "NO_RECOUNT", "coverage": {"complete": True},
            "timeline": [{"index": 0, "event": "cash_accepted"}, {"index": 1, "event": split}]},
        "target": {"type_id": "NO_RECOUNT", "decision": "supported", "anchor_index": 0}}
        for split in ("train", "validation")]
    require(validate_rows(rows)["rows"] == 2, "procedure_synthetic_rows_failed")
    try:
        validate_rows([*rows, dict(rows[0], split="validation")])
        raise AssertionError("leakage accepted")
    except TrainingGateError as error:
        require(str(error) == "procedure_template_or_input_leakage", "procedure_wrong_leakage_gate")
    text, spans = target_text(rows[0])
    require([text[a:b] for a, b in spans] == ['"NO_RECOUNT"', '"supported"', "0"],
            "procedure_value_spans_failed")
    bad = dict(rows[0], target={"type_id": "NO_RECOUNT", "decision": "unassessable", "anchor_index": 0})
    try:
        validate_rows([bad, rows[1]])
        raise AssertionError("unassessable anchor accepted")
    except TrainingGateError as error:
        require(str(error) == "procedure_unconfirmed_anchor_must_be_minus_one", "procedure_wrong_anchor_gate")
    require("target" not in render_messages(rows[0])[1]["content"], "procedure_target_leaked_into_prompt")
    return {"ok": True, "checks": 5, "pixels_read": 0, "bank_labels_read": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("preflight", "train", "synthetic-check"), default="preflight")
    args = parser.parse_args()
    try:
        if args.mode == "synthetic-check":
            base.emit(synthetic_check())
            return 0
        require(args.config is not None, "procedure_config_required")
        base.set_offline()
        config = resolve_config(args.config)
        checked = preflight(config)
        if args.mode == "preflight":
            base.emit({"ok": True, "state": "preflight_passed_not_trained", "data": checked["stats"]})
        else:
            base.emit(train(config, checked))
        return 0
    except Exception as error:
        result = {"ok": False, "state": "failed", "error_type": type(error).__name__}
        if isinstance(error, TrainingGateError):
            result["error_code"] = str(error)
        base.emit(result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
