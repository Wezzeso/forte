"""Typed static diagnostics preserve failure categories without private details."""

import pytest

from forte.decision import DecisionConfigurationError, DecisionOutputError
from forte.worker import safe_error_code


@pytest.mark.parametrize("error,expected", [
    (DecisionOutputError("schema_mismatch"), "invalid_model_output"),
    (DecisionOutputError("unknown_support_id"), "invalid_model_output"),
    (DecisionConfigurationError("enabled_rule_facts_not_requested"), "decision_configuration_error"),
])
def test_typed_decision_failure_uses_bounded_static_diagnostic(error, expected):
    assert safe_error_code(error) == expected


def test_arbitrary_exception_code_and_message_are_not_trusted():
    error = RuntimeError("synthetic private observation")
    error.code = "synthetic-private-path-and-caption"
    assert safe_error_code(error) == "processing_failed"
