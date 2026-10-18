"""Generated protocol metadata only; no labels, source clips or runtime calls."""
from copy import deepcopy

import pytest

from benchmarks.performance import qwen38_accuracy_diagnostic as original
from benchmarks.performance import qwen38_native_fullsix as native
from qa.system.test_qwen38_accuracy_diagnostic import inputs


def fixture(monkeypatch):
    family, actual = inputs()
    parent = original.freeze_protocol(family, actual)
    policy = {"schema_version": "forte.qwen38-accuracy-selection-policy.v2", "comparison_arms": native.ARMS.copy(),
        "source_population_denominator": 6, "diagnostic_protocol_semantic_sha256": original.digest(parent),
        "no_model_outputs_or_verdicts_read_for_policy_freeze": True, "final_check_used": False,
        "metrics": {"generated_test_definitions": "unchanged failure-inclusive definitions"},
        "ranking_lexicographic": [
            {"metric": native.RANKING[0], "direction": "higher"},
            {"metric": native.RANKING[1], "direction": "higher", "denominator": 3},
            {"metric": native.RANKING[2], "direction": "higher", "undefined_rule": "Keep null; ranks below any defined numeric value for this tie-break."},
            {"metric": native.RANKING[3], "direction": "higher"},
            {"metric": native.RANKING[4], "direction": "lower", "available_only_if": "Both arms have all six genuine complete/evidence-valid cases and valid monotonic timings for all six; otherwise omit this tie-break."},
            {"metric": native.RANKING[5], "winner": "qwen38_untuned"}]}
    monkeypatch.setattr(native, "PARENT_PROTOCOL_SHA256", original.digest(parent))
    monkeypatch.setattr(native, "SELECTION_POLICY_RAW_SHA256", "a"*64)
    monkeypatch.setattr(native, "SELECTION_POLICY_SHA256", original.digest(policy))
    monkeypatch.setattr(native, "METRIC_DEFINITIONS_SHA256", original.digest(policy["metrics"]))
    return parent, policy


def test_separate_full_six_is_frozen_before_native_outputs_and_preserves_parent(monkeypatch):
    parent, policy = fixture(monkeypatch)
    before = original.digest(parent)
    protocol = native.freeze_protocol(parent, policy, policy_raw_sha256="a"*64)
    assert original.digest(parent) == before
    assert protocol["run_all_sources_after_source_failure"] is True
    assert protocol["pilot_advancement_required"] is False and protocol["is_HF_pilot_expansion_or_retry"] is False
    assert protocol["comparison_arms"] == ["qwen38_untuned", "qwen38_tuned"]
    assert protocol["source_attempts_required_per_arm"] == 6 and protocol["source_attempts_required_total"] == 12
    assert protocol["runtime_binding_status"] == "pending_actual_native_GPU6_and_HTTP_gates"
    assert not any(key in protocol for key in ("runtime_bindings", "expected_model_identity"))
    budgets = native.case_budgets(protocol)
    assert [row["original_case_ordinal"] for row in budgets] == [5,1,2,3,4,6]
    assert [row["diagnostic_deadline_seconds"] for row in budgets] == [960,1980,1080,1080,1080,2040]
    assert sum(row["maximum_model_requests"] for row in budgets) == 208
    assert sum(row["diagnostic_deadline_seconds"] for row in budgets) == 8220
    assert sum(row["initial_requests"] for row in budgets) == 56


@pytest.mark.parametrize("mutation", [
    lambda p: p.__setitem__("run_all_sources_after_source_failure", False),
    lambda p: p.__setitem__("pilot_advancement_required", True),
    lambda p: p.__setitem__("reference_or_verdict_agreement_used_for_execution_order_or_continuation", True),
    lambda p: p.__setitem__("maximum_source_pipeline_attempts_per_case", 2),
    lambda p: p.__setitem__("single_source_failure_is_global_runtime_failure", True),
    lambda p: p["ordinary_source_failures_continue"].remove("invalid_model_semantics"),
    lambda p: p["global_halt_conditions"].append("first_wrong_verdict"),
    lambda p: p["execution_order"].reverse(),
    lambda p: p.__setitem__("maximum_model_requests_per_arm", 24),
    lambda p: p.__setitem__("source_attempts_required_per_arm", 1),
    lambda p: p.__setitem__("final_check_used", True),
    lambda p: p.__setitem__("final_check_authorized", True),
    lambda p: p.__setitem__("execution_authorized_by_protocol", True),
    lambda p: p.__setitem__("upload_or_usable_UI_latency_claim", True),
    lambda p: p["native_arithmetic_policy"].__setitem__("base_quantization", "nf4"),
    lambda p: p.__setitem__("runtime_bindings", {"source_archive_sha256": None}),
    lambda p: p["runtime_prerequisites"].remove("actual_native_HTTP_gate_passed"),
    lambda p: p["population"][0].__setitem__("split", "final_check"),
    lambda p: p["selection"]["ranking_lexicographic"].reverse(),
    lambda p: p["selection"].__setitem__("source_population_denominator", 5),
    lambda p: p["selection"]["ranking_lexicographic"][1].__setitem__("denominator", 2),
    lambda p: p["selection"]["metrics"].__setitem__("generated_test_definitions", "treat failures as clean negatives"),
    lambda p: p["selection"]["ranking_lexicographic"][5].__setitem__("winner", "qwen38_tuned"),
])
def test_protocol_rejects_adaptive_pilot_retry_scope_metric_and_arithmetic_changes(monkeypatch, mutation):
    parent, policy = fixture(monkeypatch)
    protocol = native.freeze_protocol(parent, policy, policy_raw_sha256="a"*64)
    mutation(protocol)
    with pytest.raises(ValueError):
        native.validate_protocol(protocol)


def test_other_population_or_selection_policy_cannot_replace_root_frozen_references(monkeypatch):
    parent, policy = fixture(monkeypatch)
    changed = deepcopy(policy)
    changed["metrics"]["new"] = "post-outcome definition"
    with pytest.raises(ValueError, match="exact_preexisting"):
        native.freeze_protocol(parent, changed, policy_raw_sha256="a"*64)
    with pytest.raises(ValueError, match="exact_preexisting"):
        native.freeze_protocol(parent, policy, policy_raw_sha256="b"*64)
