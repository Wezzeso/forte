"""Independent metadata-only checks of the genuine frozen native full-six plan."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.performance import qwen38_native_fullsix as native


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmarks/private/qwen38/native-bf16-fullsix-a12/protocol.frozen.json"
RAW = "2b5e22c90040084f694eab9d0dab675c9efdede8c200cdbf985a461160ef9ea3"
SEMANTIC = "f7566a354e514f29cbb75803d1ba11fe1d2b2ff30b5b7caa29823a23e3585477"


def frozen():
    raw = PROTOCOL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RAW
    document = json.loads(raw)
    assert native.parent_contract().digest(document) == SEMANTIC
    return document


def test_genuine_frozen_protocol_keeps_six_and_conservative_slot_caps_without_authority():
    document = frozen()
    before = PROTOCOL.read_bytes()
    native.validate_protocol(document)
    budgets = native.case_budgets(document)
    assert len(document["population"]) == 6
    assert [row["original_case_ordinal"] for row in budgets] == [5, 1, 2, 3, 4, 6]
    assert [row["maximum_model_requests"] for row in budgets] == [24, 52, 26, 26, 26, 54]
    assert [row["diagnostic_deadline_seconds"] for row in budgets] == [960, 1980, 1080, 1080, 1080, 2040]
    assert sum(row["maximum_model_requests"] for row in budgets) == 208
    assert sum(row["diagnostic_deadline_seconds"] for row in budgets) == 8220
    assert sum(row["initial_requests"] for row in budgets) == 56
    assert document["source_attempts_required_total"] == 12
    assert document["maximum_source_pipeline_attempts_per_case"] == 1
    assert document["provider_max_attempts"] == 2  # A conservative cap, not an instruction to retry.
    assert document["runtime_binding_status"] == "pending_actual_native_GPU6_and_HTTP_gates"
    assert document["execution_authorized_by_protocol"] is False
    assert PROTOCOL.read_bytes() == before


@pytest.mark.parametrize("path,value", [
    (("population",), []),
    (("execution_order",), [1, 2, 3, 4, 5, 6]),
    (("source_attempts_required_per_arm",), 1),
    (("maximum_model_requests_per_arm",), 209),
    (("maximum_execution_seconds_per_arm",), 8221),
    (("maximum_source_pipeline_attempts_per_case",), 2),
    (("run_all_sources_after_source_failure",), False),
    (("pilot_advancement_required",), True),
    (("ordinary_source_failures_continue",), ["source_timeout"]),
    (("single_source_failure_is_global_runtime_failure",), True),
    (("global_halt_conditions",), ["first_wrong_verdict"]),
    (("reference_or_verdict_agreement_used_for_execution_order_or_continuation",), True),
    (("all_failed_missing_invalid_or_halted_sources_remain_in_six_denominator",), False),
    (("native_arithmetic_policy", "base_quantization"), "nf4"),
    (("runtime_binding_status",), "passed"),
    (("runtime_bindings",), {"source_archive_sha256": None}),
    (("expected_model_identity",), {}),
    (("runtime_prerequisites",), []),
    (("selection", "source_population_denominator"), 5),
    (("selection", "provided_positive_denominator"), 2),
    (("selection", "automatic_promotion_authorized"), True),
    (("selection", "threshold_or_source_label_changes_allowed"), True),
    (("selection", "ranking_lexicographic", 5, "winner"), "qwen38_tuned"),
    (("selection", "ranking_lexicographic", 4, "available_only_if"), "ignore incomplete sources"),
    (("final_check_used",), True),
    (("execution_authorized_by_protocol",), True),
    (("upload_or_usable_UI_latency_claim",), True),
    (("historical_HF_NF4_protocols_or_failures_modified",), True),
])
def test_actual_frozen_policy_rejects_changed_scope_denominator_continuation_or_promotion(path, value):
    document = deepcopy(frozen())
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises((ValueError, KeyError, TypeError)):
        native.validate_protocol(document)


def test_frozen_metric_bytes_and_undefined_precision_ranking_cannot_be_replaced():
    document = frozen()
    assert native.parent_contract().digest(document["selection"]["metrics"]) == native.METRIC_DEFINITIONS_SHA256
    assert document["selection"]["ranking_lexicographic"][2]["undefined_rule"].startswith("Keep null;")
    for change in ("metrics", "precision"):
        altered = deepcopy(document)
        if change == "metrics":
            altered["selection"]["metrics"]["pretend_failure_is_negative"] = True
        else:
            altered["selection"]["ranking_lexicographic"][2]["undefined_rule"] = "Use zero"
        with pytest.raises(ValueError):
            native.validate_protocol(altered)
