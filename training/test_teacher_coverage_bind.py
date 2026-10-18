"""Synthetic metadata-only root release bounds; no reservation or model calls."""
from copy import deepcopy

import pytest

from training.guards import TrainingGateError
from training.qwen38_preflight import FIXED_STOP_UTC
from training.teacher_coverage_bind import validate_release


def fixture():
    now = FIXED_STOP_UTC.timestamp() - 10000
    plan = {"plan_sha256": "0" * 64, "records": [{"generated": True} for _ in range(17)]}
    attempts = {"attempts": [{"generated": True} for _ in range(34)]}
    release = {"schema_version": "forte.teacher-execution-release.v1", "released": True, "released_by": "root",
        "reference": "generated-root-release", "purpose": "source_bound_teacher_annotation", "plan_sha256": plan["plan_sha256"],
        "teacher_plan_file_sha256": "a" * 64, "attempt_plan_file_sha256": "b" * 64,
        "max_model_requests": 34, "max_records": 17, "record_offset": 0, "anchor_focus_only": True,
        "model_manifest_sha256": "249977e15be3d72425155475e59adf2385747b16e28654c9d13d45761abf75cc",
        "native_pts_ledger_verified": True, "native_pts_ledger_sha256": "c" * 64,
        "idle_handoff_receipt_sha256": "d" * 64, "exclusive_idle_gpu_handoff_verified": True,
        "fresh_role_exposure_verified": True, "producer_source_archive_sha256": "e" * 64,
        "producer_runtime_image_sha256": "f" * 64, "coverage_guard_sha256": "1" * 64,
        "max_runtime_sec": 1800, "issued_at_unix": now - 1, "expires_at_unix": now + 3600}
    kwargs = {"plan_file_sha256": "a" * 64, "attempt_file_sha256": "b" * 64,
              "native_sha256": "c" * 64, "handoff_sha256": "d" * 64, "now": now}
    return plan, attempts, release, kwargs


def test_exact34_scope_and_fresh_root_pins_required():
    plan, attempts, release, kwargs = fixture()
    validate_release(release, plan, attempts, **kwargs)
    for key, value in (("released", False), ("max_model_requests", 35), ("native_pts_ledger_verified", False),
                       ("producer_source_archive_sha256", None), ("expires_at_unix", kwargs["now"] + 60)):
        with pytest.raises(TrainingGateError):
            validate_release({**release, key: value}, plan, attempts, **kwargs)


def test_source_plan_native_and_handoff_substitutions_rejected():
    plan, attempts, release, kwargs = fixture()
    for key in ("plan_file_sha256", "attempt_file_sha256", "native_sha256", "handoff_sha256"):
        wrong = {**kwargs, key: "9" * 64}
        with pytest.raises(TrainingGateError):
            validate_release(release, plan, attempts, **wrong)
    altered = deepcopy(attempts)
    altered["attempts"].pop()
    with pytest.raises(TrainingGateError, match="scope_changed"):
        validate_release(release, plan, altered, **kwargs)
