"""Independent synthetic metadata boundaries for the separate accuracy stages."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.performance.qwen38_accuracy_diagnostic import (bind_plan, freeze_protocol,
    validate_retained_pilot)
from evaluation.io import digest
from qa.system.test_qwen38_accuracy_diagnostic import inputs, retained, completed, terminals


def plans():
    family, actual = inputs()
    protocol = freeze_protocol(family, actual)
    pilot = bind_plan(protocol, actual)
    remaining = bind_plan(protocol, actual, stage="remaining", retained=retained())
    return family, protocol, pilot, remaining


def test_independent_exact_caps_population_and_failure_inclusive_pilot_boundary():
    family, protocol, pilot, remaining = plans()
    original = digest(family)
    assert protocol["execution_order"] == [5, 1, 2, 3, 4, 6]
    assert len(pilot["source_attempts"]) == len(remaining["source_attempts"]) == 6
    assert (len(pilot["attempts"]), len(remaining["attempts"])) == (24, 184)
    assert pilot["maximum_execution_seconds"] + remaining["maximum_execution_seconds"] == 8220
    assert sum([len(pilot["attempts"]), len(remaining["attempts"])] ) == 208
    row = completed(pilot)
    row["verdict"] = "deliberate_reference_disagreement"
    assert validate_retained_pilot(remaining, pilot, terminals(pilot, row)) is row
    row["status"] = "timeout"
    with pytest.raises(ValueError):
        validate_retained_pilot(remaining, pilot, terminals(pilot, row))
    assert digest(family) == original


@pytest.mark.parametrize("field,value", [
    ("provider", {"launch_config": "/generated/changed-runtime.json"}),
    ("effective_client_request_timeout_seconds", 239),
    ("comparison_runtime_profile", {"unreviewed": True}),
    ("comparison_runtime_profile_sha256", "0" * 64),
    ("source_equivalence", {"unreviewed": True}),
    ("source_equivalence_sha256", "0" * 64),
    ("evaluator_source_archive_sha256", "0" * 64),
    ("serving_source_archive_sha256", "0" * 64),
    ("audited_common_source_deltas", [{"unreviewed": True}]),
])
def test_remaining_cannot_switch_any_actual_runtime_or_config_binding(field, value):
    _, _, pilot, remaining = plans()
    remaining[field] = deepcopy(value)
    with pytest.raises(ValueError):
        validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))


def test_retained_pilot_keeps_exact_one_success_and_five_unattempted_sources():
    _, _, pilot, remaining = plans()
    rows = terminals(pilot, completed(pilot))
    assert sum(row["status"] == "done" for row in rows) == 1
    for index in [0, 1, 2, 3, 5]:
        bad = deepcopy(rows)
        bad[index]["status"] = "done"
        with pytest.raises(ValueError):
            validate_retained_pilot(remaining, pilot, bad)


def test_metadata_role_config_raw_digest_can_refresh_without_switching_model_runtime():
    # A09 independently verifies unchanged role policy/assignments, genuine
    # exposure ancestry and fresh snapshots. The metadata gate must not force
    # reuse of an expired pilot snapshot by demanding equal raw config hashes.
    _, _, pilot, remaining = plans()
    remaining["role_config_sha256"] = "0" * 64
    assert validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))


def test_retained_pilot_role_digest_must_remain_its_actual_original_pin():
    _, _, pilot, remaining = plans()
    remaining["retained_pilot_role_config_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))


def test_host_contract_import_is_stdlib_only_under_isolated_python_without_site_packages():
    contract = Path(__file__).resolve().parents[2] / "benchmarks/performance/qwen38_accuracy_diagnostic.py"
    child = subprocess.run([sys.executable, "-I", "-S", "-c",
        "import importlib.util,sys,json; spec=importlib.util.spec_from_file_location('isolated_contract',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        "print(json.dumps({n:n in sys.modules for n in ['torch','PIL','httpx','numpy','forte','evaluation']}))", str(contract)],
        capture_output=True, text=True, timeout=10)
    assert child.returncode == 0
    assert json.loads(child.stdout) == dict.fromkeys(["torch", "PIL", "httpx", "numpy", "forte", "evaluation"], False)


@pytest.mark.parametrize("pin", ["expected_assignments_sha256", "expected_source_manifest_sha256",
    "expected_automated_split_sha256", "expected_scenario_groups_sha256"])
def test_role_snapshot_refresh_cannot_switch_any_frozen_assignment_or_policy_pin(pin):
    _, _, pilot, remaining = plans()
    remaining["role_config_sha256"] = "0" * 64
    remaining["role_policy"][pin] = "e" * 64
    with pytest.raises(ValueError):
        validate_retained_pilot(remaining, pilot, terminals(pilot, completed(pilot)))
