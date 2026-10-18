"""Native HF XGrammar decoding for the audited decision-schema subset.

Compiled immutable grammars may be cached; matcher/logits-processor state is
always new for each generate call. Unsupported schema features fail before
generation instead of silently degrading to unconstrained JSON generation.
"""

from collections import OrderedDict
from copy import deepcopy
from importlib.metadata import version
import json
import hashlib
from pathlib import Path
import re
import threading
import time

from serving.qwen38_identity import CandidateError, require


POLICY = {"backend": "xgrammar_hf", "version": "0.2.7",
          "schema_policy": "forte.decision-json-subset.v1", "any_order": False,
          "fact_order": "canonical_requested_fact_ids.v1",
          "any_whitespace": False, "strict_mode": True,
          "max_cache_entries": 8, "max_cache_bytes": 67108864}
VOCAB_SIZE = 248320
STOP_TOKEN_IDS = (248046, 248044)
ANNOTATIONS = {"title", "description", "default", "$schema"}
SUPPORTED = ANNOTATIONS | {"$defs", "$ref", "type", "properties", "required", "additionalProperties",
                           "items", "minItems", "maxItems", "minLength", "maxLength", "pattern", "enum", "anyOf"}


def checked_schema(schema: dict) -> str:
    """Reject ignored/ambiguous keywords, remote refs and unbounded complexity."""
    require(isinstance(schema, dict), "candidate_grammar_schema_object_required")
    try:
        raw = json.dumps(schema, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise CandidateError("candidate_grammar_schema_invalid") from None
    require(len(raw.encode()) <= 65536, "candidate_grammar_schema_too_large")
    definitions = schema.get("$defs", {})
    require(isinstance(definitions, dict), "candidate_grammar_definitions_invalid")
    count = 0

    def visit(node, depth=0):
        nonlocal count
        count += 1
        require(isinstance(node, dict) and count <= 2048 and depth <= 32,
                "candidate_grammar_schema_complexity_exceeded")
        require(set(node) <= SUPPORTED, "candidate_grammar_unsupported_schema_keyword")
        if "$ref" in node:
            reference = node["$ref"]
            require(isinstance(reference, str) and re.fullmatch(r"#/\$defs/[A-Za-z0-9_-]+", reference)
                    and reference.split("/")[-1] in definitions
                    and set(node) <= ANNOTATIONS | {"$ref"}, "candidate_grammar_reference_unsupported")
        if "type" in node:
            require(node["type"] in {"object", "array", "string", "boolean", "null"},
                    "candidate_grammar_type_unsupported")
        if "properties" in node:
            properties = node["properties"]
            require(node.get("type") == "object" and isinstance(properties, dict) and len(properties) <= 64
                    and all(isinstance(name, str) and 0 < len(name) <= 128 for name in properties),
                    "candidate_grammar_properties_invalid")
            for child in properties.values():
                visit(child, depth + 1)
        if "required" in node:
            required = node["required"]
            require(isinstance(required, list) and len(set(required)) == len(required)
                    and set(required) <= node.get("properties", {}).keys(), "candidate_grammar_required_invalid")
        if "additionalProperties" in node:
            require(node["additionalProperties"] is False, "candidate_grammar_additional_properties_unsupported")
        if "items" in node:
            require(node.get("type") == "array", "candidate_grammar_items_invalid")
            visit(node["items"], depth + 1)
        for key in ("minItems", "maxItems", "minLength", "maxLength"):
            if key in node:
                require(type(node[key]) is int and 0 <= node[key] <= 4096,
                        "candidate_grammar_bound_invalid")
        for low, high in (("minItems", "maxItems"), ("minLength", "maxLength")):
            require(low not in node or high not in node or node[low] <= node[high], "candidate_grammar_bound_invalid")
        if "pattern" in node:
            # This is the only expression in the trusted candidate schema.
            # Do not claim arbitrary ECMA/Python regex semantics match XGrammar.
            require(node.get("type") == "string" and node["pattern"] == r"^[A-Za-z0-9_-]{1,96}$"
                    and not {"minLength", "maxLength", "enum"}.intersection(node),
                    "candidate_grammar_pattern_unsupported")
        if "enum" in node:
            values = node["enum"]
            require(isinstance(values, list) and 0 < len(values) <= 256
                    and all(isinstance(item, str) and 0 < len(item) <= 256 for item in values)
                    and len(set(values)) == len(values)
                    and all(node.get("minLength", 0) <= len(item) <= node.get("maxLength", 4096)
                            for item in values), "candidate_grammar_enum_unsupported")
        if "anyOf" in node:
            options = node["anyOf"]
            require(isinstance(options, list) and len(options) == 2
                    and all(isinstance(option, dict) for option in options)
                    and {option.get("type") for option in options} == {"null", "string"},
                    "candidate_grammar_union_unsupported")
            for child in options:
                visit(child, depth + 1)
        if "$defs" in node:
            require(depth == 0 and all(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name)
                                      for name in definitions), "candidate_grammar_definitions_invalid")
            for child in definitions.values():
                visit(child, depth + 1)

    visit(schema)
    return raw


def trusted_fact_order_schema(schema: dict) -> tuple[dict, int]:
    """Narrow only the known fact array to canonical requested-ID order.

    The original schema must pass checked_schema first. Arbitrary public
    prefixItems is never accepted. All state/support/notes constraints remain
    unchanged; only valid fact permutations are narrowed to sorted-ID order.
    """
    checked_schema(schema)
    properties = schema.get("properties", {})
    if "facts" not in properties:
        return schema, 0
    facts = properties["facts"]
    definitions = schema.get("$defs", {})
    fact = definitions.get("_RawFact", {})
    fields = fact.get("properties", {})
    fact_ids = fields.get("fact_id", {}).get("enum", [])
    require(facts.get("type") == "array" and facts.get("items") == {"$ref": "#/$defs/_RawFact"}
            and type(facts.get("minItems")) is int and facts.get("minItems") == facts.get("maxItems")
            and facts["minItems"] == len(fact_ids) and 1 <= len(fact_ids) <= 32
            and fact.get("type") == "object" and fact.get("additionalProperties") is False
            and set(fields) == {"fact_id", "state", "supporting_frame_ids", "notes"}
            and set(fact.get("required", [])) == {"fact_id", "state", "supporting_frame_ids"}
            and fields["fact_id"].get("type") == "string"
            and fields["state"].get("type") == "string"
            and set(fields["state"].get("enum", [])) == {"present", "absent", "unclear"}
            and fields["supporting_frame_ids"].get("type") == "array"
            and fields["supporting_frame_ids"].get("items", {}).get("type") == "string"
            and isinstance(fields["supporting_frame_ids"].get("items", {}).get("enum"), list)
            and bool(fields["supporting_frame_ids"]["items"]["enum"])
            and len(set(fact_ids)) == len(fact_ids), "candidate_grammar_trusted_fact_shape_mismatch")
    derived = deepcopy(schema)
    derived_facts = derived["properties"]["facts"]
    derived_facts["prefixItems"] = []
    for fact_id in sorted(fact_ids):
        item = deepcopy(fact)
        item["properties"]["fact_id"]["enum"] = [fact_id]
        derived_facts["prefixItems"].append(item)
    derived_facts["items"] = False
    return derived, len(fact_ids)


class NativeGrammar:
    """One model/tokenizer compiler, bounded immutable cache, fresh matchers."""

    def __init__(self, tokenizer, model, policy: dict):
        started = time.perf_counter()
        require(policy == POLICY and version("xgrammar") == POLICY["version"],
                "candidate_grammar_runtime_policy_mismatch")
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor

        text_config = model.config.text_config
        require(text_config.vocab_size == VOCAB_SIZE
                and model.get_output_embeddings().weight.shape[0] == VOCAB_SIZE,
                "candidate_grammar_model_vocab_mismatch")
        eos = model.generation_config.eos_token_id
        require(isinstance(eos, list) and tuple(eos) == STOP_TOKEN_IDS
                and model.generation_config.pad_token_id == 248044
                and tokenizer.convert_tokens_to_ids("<|im_end|>") == STOP_TOKEN_IDS[0]
                and tokenizer.convert_tokens_to_ids("<|endoftext|>") == STOP_TOKEN_IDS[1],
                "candidate_grammar_stop_token_mismatch")
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=VOCAB_SIZE,
                                                         stop_token_ids=list(STOP_TOKEN_IDS))
        require(tokenizer_info.vocab_size == VOCAB_SIZE
                and set(tokenizer_info.stop_token_ids) == set(STOP_TOKEN_IDS),
                "candidate_grammar_tokenizer_info_mismatch")
        self.compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=2, cache_enabled=False)
        self.processor_class = LogitsProcessor
        self.cache = OrderedDict()
        self.schema_bindings = OrderedDict()
        self.cache_bytes = 0
        self.lock = threading.Lock()
        self.requests = 0
        self.initialization_seconds = time.perf_counter() - started
        self.public_schema_warmup = None

    def compile_schema(self, schema: dict):
        raw = checked_schema(schema)
        with self.lock:
            compiled = self.cache.get(raw)
            if compiled is None:
                derived, count = trusted_fact_order_schema(schema)
                derived_raw = json.dumps(derived, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
                try:
                    compiled = self.compiler.compile_json_schema(derived_raw, any_whitespace=False,
                        separators=(",", ":"), strict_mode=True, any_order=False)
                except Exception:
                    raise CandidateError("candidate_grammar_schema_compilation_failed") from None
                size = compiled.memory_size_bytes
                require(type(size) is int and 0 < size <= POLICY["max_cache_bytes"],
                        "candidate_grammar_compiled_memory_exceeded")
                while self.cache and (len(self.cache) >= POLICY["max_cache_entries"]
                                      or self.cache_bytes + size > POLICY["max_cache_bytes"]):
                    old_key, old = self.cache.popitem(last=False)
                    self.cache_bytes -= old.memory_size_bytes
                    self.schema_bindings.pop(old_key, None)
                self.cache[raw] = compiled
                self.schema_bindings[raw] = {
                    "original_schema_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "derived_schema_sha256": hashlib.sha256(derived_raw.encode()).hexdigest(),
                    "canonical_fact_count": count, "fact_order": POLICY["fact_order"],
                    "valid_fact_permutations_narrowed": count > 1,
                    "truth_states_preserved": True, "original_post_validation_required": True}
                self.cache_bytes += size
            else:
                self.cache.move_to_end(raw)
            return compiled

    def new_processor(self, schema: dict):
        compiled = self.compile_schema(schema)
        self.requests += 1
        # Only compiled immutable grammar is cached; this object contains
        # mutable matcher/prefill state and must never cross requests.
        return self.processor_class(compiled)

    def precompile_public_schema(self, *, max_images: int, rules_path: Path | None = None):
        """Compile exact public full-family shape/aliases; no media/model call."""
        from serving.qwen38_identity import load_json, file_sha256
        from forte.decision.structured import DecisionConfig, _schema

        started = time.perf_counter()
        rules_path = rules_path or Path(__file__).resolve().parents[1] / "rules/catalog.json"
        require(file_sha256(rules_path) == "3ba1c79c00f39a40addf4e1d2299c5bbade26c7949aeee10c382b9322fdde1dd",
                "candidate_public_grammar_rules_pin_mismatch")
        catalog = load_json(rules_path)
        enabled = [rule["rule_id"] for rule in catalog["rules"] if rule["enabled"] is True]
        fact_ids = sorted({fact_id for rule_id in enabled
                           for fact_id in catalog["bindings"][rule_id]["required_fact_states"]})
        event_ids = sorted({catalog["bindings"][rule_id]["event_type_id"] for rule_id in enabled})
        require(max_images == 16 and len(enabled) == 4 and len(fact_ids) == 19 and len(event_ids) == 3,
                "candidate_public_grammar_full_family_mismatch")
        # compact_frame_request is the authoritative actual request mapping.
        # It always aliases chronologically ordered frames as f00..f15; no
        # private source IDs, frame pixels or timestamps enter this compile.
        schema = _schema([f"f{index:02d}" for index in range(max_images)], fact_ids, event_ids,
                         DecisionConfig(compact_frame_ids=True, concise_observations=True))
        self.compile_schema(schema)
        self.public_schema_warmup = {
            "schema_sha256": hashlib.sha256(checked_schema(schema).encode()).hexdigest(),
            "derived_schema_sha256": self.schema_bindings[checked_schema(schema)]["derived_schema_sha256"],
            "fact_order": POLICY["fact_order"],
            "fact_count": 19, "rule_count": 4, "event_type_count": 3, "frame_count": 16,
            "compact_aliases": "f00_through_f15", "concise_observations": True,
            "compile_seconds": time.perf_counter() - started, "model_calls": 0, "private_media_read": False}
        return self.public_schema_warmup

    def receipt(self) -> dict:
        return {**POLICY, "model_vocab_size": VOCAB_SIZE, "stop_token_ids": list(STOP_TOKEN_IDS),
                "fresh_matcher_per_generate": True, "requests_bound_to_grammar": self.requests,
                "initialization_seconds": self.initialization_seconds,
                "public_schema_warmup": self.public_schema_warmup,
                "schema_bindings": list(self.schema_bindings.values()),
                "compiled_cache_entries": len(self.cache), "compiled_cache_bytes": self.cache_bytes,
                "strict_post_generation_validation": True, "gpu_overhead_measured": False}
