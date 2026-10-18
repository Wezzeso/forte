"""Generated terminal counters and continuation only; no native execution."""
import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evaluation.private import native_fullsix_runner as runner


def plan():
    cases = [{"attempt_id": "generated-independent-" + str(n), "original_case_ordinal": n}
             for n in (5, 1, 2, 3, 4, 6)]
    return {"cases": cases, "source_attempts": cases,
        "attempts": [{"attempt_id": "generated-slot-" + str(i), "source_attempt_id": cases[i % 6]["attempt_id"]}
                     for i in range(208)],
        "native_protocol_sha256": "a" * 64, "recipe_sha256": "b" * 64,
        "expected_model_identity": {"model_id": "generated", "model_revision": "c" * 40, "adapter_sha256": None}}


def run_generated(tmp_path, executor):
    release = {"expires_at_utc": (runner._now() + timedelta(hours=3)).isoformat()}
    with patch.object(runner, "validate_execution"), patch.object(runner, "make_settings", return_value=SimpleNamespace()):
        return asyncio.run(runner.run_sources(plan(), release, {}, tmp_path, plan_sha256="d" * 64,
            release_sha256="e" * 64, case_executor=executor))


def test_escaped_source_local_io_failure_does_not_skip_remaining_sources(tmp_path):
    observed = []
    async def executor(plan, case, *args):
        observed.append(case["original_case_ordinal"])
        if len(observed) == 1:
            raise OSError("generated source-local I/O failure")
        return {"status": "failed", "public_contract_validated": False}, []
    result = run_generated(tmp_path, executor)
    assert observed == [5, 1, 2, 3, 4, 6]
    assert result["source_terminal_counts"] == {"failed": 6}
    assert result["model_slot_terminal_counts"] == {"not_attempted": 208}
    assert result["technically_complete_sources"] == 0 and result["global_halt_code"] is None


def test_failed_publication_or_cleanup_cannot_keep_a_technical_success_count(tmp_path):
    async def executor(*args):
        return {"status": "failed", "public_contract_validated": False,
                "server_receipt_verified": True, "technical_completion": {"generated_pre_cleanup_success": True},
                "cleanup_error_type": "GeneratedCleanupFailure"}, []
    result = run_generated(tmp_path, executor)
    assert result["source_terminal_counts"] == {"failed": 6}
    assert result["technically_complete_sources"] == 0


def test_missing_actual_native_plan_factory_fails_before_config_or_runtime_access():
    with patch.object(runner, "contract", return_value=SimpleNamespace()), patch.object(runner, "file_sha",
            side_effect=AssertionError("No actual file may be read")):
        with pytest.raises(ValueError, match="native_fullsix_actual_native_plan_factory_not_released"):
            runner.validate_execution({}, {}, {}, plan_sha256="a" * 64, release_sha256="b" * 64)
