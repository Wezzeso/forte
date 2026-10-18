"""Native grammar contracts without importing torch or initializing CUDA."""

from collections import OrderedDict
import json
import threading
import unittest

from serving.qwen38_grammar import NativeGrammar, POLICY, checked_schema, trusted_fact_order_schema
from serving.qwen38_identity import CandidateError


class NativeGrammarTests(unittest.TestCase):
    def test_real_full_decision_schema_is_in_explicit_subset(self):
        from forte.decision.structured import DecisionConfig, _schema
        schema = _schema([f"f{index}" for index in range(16)], [f"fact_{index}" for index in range(19)],
                         ["event1", "event2", "event3", "event4"],
                         DecisionConfig(compact_frame_ids=True, concise_observations=True))
        self.assertEqual(json.loads(checked_schema(schema)), schema)
        generic = _schema(["f0"], ["fact"], [], DecisionConfig())
        self.assertEqual(json.loads(checked_schema(generic)), generic)

    def test_ignored_or_remote_schema_features_fail(self):
        for schema in ({"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                       {"type": "object", "if": {"properties": {}}, "then": {}},
                       {"type": "string", "not": {"enum": ["x"]}},
                       {"oneOf": [{"type": "string"}, {"type": "string"}]},
                       {"$ref": "https://example.invalid/schema"},
                       {"type": "number", "minimum": 0},
                       {"type": "string", "pattern": r"^[A-Za-z0-9_-]{1,96}$", "maxLength": 2},
                       {"type": "string", "pattern": ".*"}):
            with self.subTest(schema=schema), self.assertRaises(CandidateError):
                checked_schema(schema)

    def runtime(self, size=100):
        class Compiled:
            memory_size_bytes = size
        class Compiler:
            calls = 0
            def compile_json_schema(self, schema, **options):
                self.calls += 1
                self.options = options
                return Compiled()
        class Processor:
            def __init__(self, compiled):
                self.compiled = compiled
                self.mutable_state = []
        runtime = NativeGrammar.__new__(NativeGrammar)
        runtime.compiler, runtime.processor_class = Compiler(), Processor
        runtime.cache, runtime.cache_bytes, runtime.requests = OrderedDict(), 0, 0
        runtime.schema_bindings = OrderedDict()
        runtime.lock = threading.Lock()
        runtime.initialization_seconds, runtime.public_schema_warmup = 0.0, None
        return runtime

    def test_compiled_cache_is_bounded_and_matcher_state_always_fresh(self):
        runtime = self.runtime()
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                  "required": ["ok"], "additionalProperties": False}
        one, two = runtime.new_processor(schema), runtime.new_processor(schema)
        one.mutable_state.append("used")
        self.assertIsNot(one, two)
        self.assertIs(one.compiled, two.compiled)
        self.assertEqual(two.mutable_state, [])
        self.assertEqual(runtime.compiler.calls, 1)
        self.assertEqual(runtime.compiler.options, {"any_whitespace": False, "separators": (",", ":"),
                                                  "strict_mode": True, "any_order": False})
        for index in range(15):
            runtime.new_processor({"type": "string", "enum": [str(index)]})
        self.assertEqual(len(runtime.cache), POLICY["max_cache_entries"])
        self.assertLessEqual(runtime.cache_bytes, POLICY["max_cache_bytes"])

    def test_compile_errors_and_memory_limits_never_fallback(self):
        runtime = self.runtime(POLICY["max_cache_bytes"] + 1)
        with self.assertRaisesRegex(CandidateError, "compiled_memory_exceeded"):
            runtime.new_processor({"type": "string"})
        def fail(*args, **kwargs):
            raise ValueError("private schema must not be exposed")
        runtime.compiler.compile_json_schema = fail
        with self.assertRaisesRegex(CandidateError, "schema_compilation_failed"):
            runtime.new_processor({"type": "string"})

    def test_actual_public_full_schema_is_compiled_before_any_model_request(self):
        runtime = self.runtime()
        proof = runtime.precompile_public_schema(max_images=16)
        self.assertEqual((proof["fact_count"], proof["rule_count"], proof["event_type_count"], proof["frame_count"]),
                         (19, 4, 3, 16))
        self.assertEqual(proof["model_calls"], 0)
        self.assertEqual(runtime.requests, 0)
        self.assertEqual(len(runtime.cache), 1)
        # Same immutable schema uses the startup compiler cache; mutable
        # matcher construction still happens only at the request boundary.
        schema = json.loads(next(iter(runtime.cache)))
        runtime.new_processor(schema)
        self.assertEqual(runtime.compiler.calls, 1)
        self.assertEqual(runtime.requests, 1)
        self.assertFalse(runtime.receipt()["gpu_overhead_measured"])

    def test_canonical_fact_transform_preserves_truth_states_and_original_contract(self):
        from forte.decision.structured import DecisionConfig, _schema
        from jsonschema import Draft202012Validator
        schema = _schema(["f00", "f01"], ["z_fact", "a_fact"], ["event"],
                         DecisionConfig(concise_observations=True))
        original_bytes = json.dumps(schema)
        derived, count = trusted_fact_order_schema(schema)
        self.assertEqual(count, 2)
        self.assertEqual(json.dumps(schema), original_bytes)
        self.assertEqual([item["properties"]["fact_id"]["enum"] for item in
                          derived["properties"]["facts"]["prefixItems"]], [["a_fact"], ["z_fact"]])
        self.assertIs(derived["properties"]["facts"]["items"], False)
        validator = Draft202012Validator(derived)
        for state in ("present", "absent", "unclear"):
            result = {"facts": [{"fact_id": fact, "state": state,
                                  "supporting_frame_ids": ["f00"], "notes": None}
                                 for fact in ("a_fact", "z_fact")], "candidates": []}
            validator.validate(result)
            Draft202012Validator(schema).validate(result)
        duplicate = {"facts": [{"fact_id": "a_fact", "state": "unclear",
                               "supporting_frame_ids": [], "notes": None}] * 2, "candidates": []}
        self.assertFalse(validator.is_valid(duplicate))
        self.assertTrue(Draft202012Validator(schema).is_valid(duplicate))
        with self.assertRaisesRegex(CandidateError, "unsupported_schema_keyword"):
            checked_schema(derived)


if __name__ == "__main__":
    unittest.main()
