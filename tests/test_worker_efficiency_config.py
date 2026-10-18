"""Opt-in worker efficiency settings reach actual component constructors."""

from dataclasses import replace
import os

import pytest

from forte.config import WorkerSettings
from forte.pipeline import AnalysisPipeline
from rules.catalog import load_catalog


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    # Test only declared configuration, independent of a developer's runtime.
    for name in list(os.environ):
        if name.startswith("FORTE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FORTE_DATA_DIR", str(tmp_path))
    return monkeypatch


def test_efficiency_settings_preserve_conservative_defaults(worker_env):
    settings = WorkerSettings.from_env()
    assert settings.compact_frame_ids is False
    assert settings.concise_observations is False
    assert settings.staged_screening is False
    assert settings.max_inflight_per_job == 1
    assert settings.max_concurrent_clips == 2


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("TRUE", True), (" true ", True),
    ("false", False), ("FALSE", False), (" false ", False),
])
def test_compact_frame_environment_accepts_explicit_boolean_only(worker_env, raw, expected):
    worker_env.setenv("FORTE_COMPACT_FRAME_IDS", raw)
    assert WorkerSettings.from_env().compact_frame_ids is expected


@pytest.mark.parametrize("raw", ["", "1", "0", "yes", "no", "on", "off", "truthy", "null"])
def test_compact_frame_environment_rejects_ambiguous_flags(worker_env, raw):
    worker_env.setenv("FORTE_COMPACT_FRAME_IDS", raw)
    with pytest.raises(ValueError, match="FORTE_COMPACT_FRAME_IDS must be true or false"):
        WorkerSettings.from_env()


@pytest.mark.parametrize("raw", ["0", "-1", "0.5", "nan", "inf", "", "invalid"])
def test_per_job_concurrency_environment_requires_positive_integer(worker_env, raw):
    worker_env.setenv("FORTE_MAX_INFLIGHT_PER_JOB", raw)
    with pytest.raises(ValueError):
        WorkerSettings.from_env()


def test_per_job_concurrency_cannot_exceed_total_concurrency(worker_env):
    worker_env.setenv("FORTE_MAX_INFLIGHT_PER_JOB", "3")
    with pytest.raises(ValueError, match="cannot exceed max_concurrent_clips"):
        WorkerSettings.from_env()
    worker_env.setenv("FORTE_MAX_CONCURRENT_CLIPS", "3")
    assert WorkerSettings.from_env().max_inflight_per_job == 3


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_direct_settings_do_not_coerce_compact_flags(worker_env, value):
    with pytest.raises(ValueError, match="compact_frame_ids must be a boolean"):
        replace(WorkerSettings.from_env(), compact_frame_ids=value)


@pytest.mark.parametrize("field,environment", [
    ("concise_observations", "FORTE_CONCISE_OBSERVATIONS"),
    ("staged_screening", "FORTE_STAGED_SCREENING"),
])
def test_optional_analysis_modes_require_explicit_boolean_configuration(worker_env, field, environment):
    worker_env.setenv(environment, "true")
    assert getattr(WorkerSettings.from_env(), field) is True
    worker_env.setenv(environment, "1")
    with pytest.raises(ValueError, match=environment):
        WorkerSettings.from_env()
    worker_env.delenv(environment)
    with pytest.raises(ValueError, match=field):
        replace(WorkerSettings.from_env(), **{field: 1})


@pytest.mark.parametrize("value", [True, 1.0, "1", 0, -1, 3])
def test_direct_settings_reject_invalid_per_job_limits(worker_env, value):
    with pytest.raises(ValueError):
        replace(WorkerSettings.from_env(), max_inflight_per_job=value)


@pytest.mark.parametrize("compact,inflight", [(False, 1), (True, 2)])
def test_pipeline_receives_explicit_decision_and_scheduler_settings(worker_env, compact, inflight):
    worker_env.setenv("FORTE_COMPACT_FRAME_IDS", str(compact).lower())
    worker_env.setenv("FORTE_MAX_INFLIGHT_PER_JOB", str(inflight))
    worker_env.setenv("FORTE_MAX_CONCURRENT_CLIPS", "2")
    settings = WorkerSettings.from_env()
    synthetic_provider = object()  # Constructor-only fixture; no model request.
    pipeline = AnalysisPipeline(settings, provider=synthetic_provider, catalog=load_catalog())
    assert pipeline.engine.config.compact_frame_ids is compact
    assert pipeline.provider.scheduler.max_inflight_per_job == inflight
    assert pipeline.provider.scheduler.concurrency == settings.max_concurrent_clips
    assert pipeline.provider.delegate is synthetic_provider
    assert pipeline.engine.provider is pipeline.provider
