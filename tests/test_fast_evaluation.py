"""Synthetic scoring tests: no private media, model calls, or real annotations."""

from __future__ import annotations

import json
import math
import asyncio
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tools.fast_eval import main, run_one
from tools.fast_validation import grouped_folds, json_hash, regression_gate, score_results, validate_labels


def event(kind="TEST", t=5.0):
    return {"type_id": kind, "t": t}


def result(name, events=None, **extra):
    return {"name": name, "events": events or [], "wall_s": 1.0, **extra}


LABELS = {"clean": {"violation": False}, "positive": {"violation": True, "t": 5.0, "type_id": "TEST"}}


@pytest.mark.parametrize("name", ["clean", "positive"])
def test_processing_failure_is_always_incorrect(name):
    other = "positive" if name == "clean" else "clean"
    rows = [{"name": name, "error": "decode failed"}, result(other, [event()] if other == "positive" else [])]
    report = score_results(LABELS, rows)
    failed = next(r for r in report["results"] if r["name"] == name)
    assert failed["correct"] is False and failed["prediction"] is None
    assert report["summary"]["accuracy"] == 0.5
    assert report["summary"]["failed"] == 1
    assert report["summary"]["tn"] == (name != "clean")
    assert report["summary"]["recall"] == (name != "positive")
    assert "processing_failures" in regression_gate(report)["reasons"]


@pytest.mark.parametrize("mutation,integrity", [
    (lambda rows: rows[:1], "missing"),
    (lambda rows: rows + [result("extra")], "unexpected"),
    (lambda rows: rows + [result("clean")], "duplicates"),
    (lambda rows: rows + [{}], "invalid_rows"),
])
def test_gate_requires_exact_names(mutation, integrity):
    report = score_results(LABELS, mutation([result("clean"), result("positive", [event()])]))
    assert report["integrity"][integrity]
    assert report["summary"]["n"] == 2
    assert "result_names_not_exact" in regression_gate(report)["reasons"]


@pytest.mark.parametrize("events", [None, "[]", [event(t=float("nan"))], [event(t=-1)], [{"type_id": "TEST"}], [event(t=True)]])
def test_malformed_result_cannot_be_credited_as_clean(events):
    row = {"name": "clean"}
    if events is not None:
        row["events"] = events
    report = score_results({"clean": {"violation": False}}, [row])
    assert report["summary"]["accuracy"] == 0 and report["summary"]["tn"] == 0
    assert report["results"][0]["error"] == "invalid_or_missing_events"


def test_untrusted_cached_scores_labels_and_time_are_recomputed():
    report = score_results(LABELS, [result("clean", [event()], correct=True, label={"violation": True}),
                                    result("positive", [], correct=True, timecode_err=0, type_correct=True)])
    assert report["summary"]["accuracy"] == 0
    assert report["summary"]["fp"] == report["summary"]["fn"] == 1
    assert report["summary"]["timecode_scored"] == 0
    assert "false_alarms" in regression_gate(report)["reasons"]


def test_wrong_type_never_gets_timestamp_credit():
    report = score_results(LABELS, [result("clean"), result("positive", [event("WRONG", 5)])])
    row = report["results"][1]
    assert row["correct"] is True  # Jury's binary video verdict is scored separately.
    assert row["type_correct"] is False and "timecode_err" not in row
    assert row["event_time_matches"] == 0
    assert not regression_gate(report, require_type_time=True)["passed"]


def test_event_matching_is_one_to_one_and_type_specific():
    labels = {"clip": {"violation": True, "events": [event(t=4), event(t=6), event("OTHER", 10)]}}
    report = score_results(labels, [result("clip", [event(t=5), event("OTHER", 5)])], time_tolerance=2)
    assert report["summary"]["event_time_labels"] == 3
    assert report["summary"]["event_time_matches"] == 1
    assert not report["results"][0]["event_correct"]


def test_event_interval_matching_handles_case_greedy_closest_would_lose():
    labels = {"clip": {"violation": True, "events": [event(t=4), event(t=6)]}}
    report = score_results(labels, [result("clip", [event(t=2), event(t=5)])], time_tolerance=2)
    assert report["results"][0]["event_correct"]
    assert report["summary"]["event_time_matches"] == 2


def test_type_and_time_labels_are_optional_and_legacy_clean_time_is_ignored():
    labels = {"a": {"violation": True}, "b": {"violation": False, "t": None}}
    report = score_results(labels, [result("a", [event()]), result("b")])
    assert report["summary"]["accuracy"] == 1
    assert report["summary"]["type_labeled"] == report["summary"]["time_labeled_videos"] == 0
    assert regression_gate(report, require_type_time=True)["passed"]


def test_gate_checks_count_clean_latency_and_accuracy():
    report = score_results(LABELS, [result("clean"), result("positive", [], wall_s=241)])
    gate = regression_gate(report, expected_count=40, expected_clean=19, min_accuracy=0.85, max_wall_s=240)
    assert set(gate["reasons"]) == {"unexpected_video_count", "unexpected_clean_count", "accuracy_below_minimum", "latency_budget_exceeded"}
    report["summary"]["wall_scored"] = 1
    assert "latency_missing" in regression_gate(report, max_wall_s=240)["reasons"]


def test_baseline_blocks_per_video_regression_even_when_aggregate_improves():
    labels = {"a": {"violation": True}, "b": {"violation": True}, "c": {"violation": True}}
    baseline = score_results(labels, [result("a", [event()]), result("b"), result("c")])
    current = score_results(labels, [result("a"), result("b", [event()]), result("c", [event()])])
    gate = regression_gate(current, baseline=baseline)
    assert gate["regressed_videos"] == ["a"]
    assert gate["reasons"] == ["previously_correct_video_regressed"]


def test_invalid_baseline_cannot_authorize_a_gate():
    good = score_results(LABELS, [result("clean"), result("positive", [event()])])
    bad = score_results(LABELS, [result("clean")])
    assert not regression_gate(good, baseline=bad)["passed"]


@pytest.mark.parametrize("payload", [{}, [], {"a": {"violation": 0}}, {"a": {"violation": True, "t": math.inf}},
                                     {"a": {"violation": True, "events": []}}])
def test_invalid_labels_are_rejected(payload):
    with pytest.raises(ValueError):
        validate_labels(payload)


def manifest():
    return {"provenance": {"source": "synthetic_test", "grouping_method": "same actors/setup grouped"},
            "clips": [{"name": f"clip{i}", "source_id": f"src{i}", "group_id": f"group{i}"} for i in range(5)] + [
                {"name": "clip0_flip", "source_id": "src0", "group_id": "group0", "augmentation": "mirror", "parent_name": "clip0"},
                {"name": "untouched_camera", "source_id": "src_holdout", "group_id": "new_camera", "role": "holdout"}]}


def test_folds_are_deterministic_independent_of_input_order_and_preserve_group():
    data = manifest()
    split = grouped_folds(data)
    data["clips"].reverse()
    other = grouped_folds(data)
    assert split["folds"] == other["folds"]
    assert split["untouched_holdout"] == ["untouched_camera"]
    assert split["heldout_accuracy_claim"] is False
    for fold in split["folds"]:
        assert not set(fold["train"]) & set(fold["evaluate"])
        assert "untouched_camera" not in fold["train"] + fold["evaluate"]
        assert ("clip0" in fold["evaluate"]) == ("clip0_flip" in fold["evaluate"])
    assert sorted(n for fold in split["folds"] for n in fold["evaluate"]) == sorted(c["name"] for c in data["clips"] if c.get("role") != "holdout")


@pytest.mark.parametrize("change", [
    lambda m: m.pop("provenance"),
    lambda m: m["clips"][1].update(source_id="src0"),
    lambda m: m["clips"][1].update(content_sha256="same") or m["clips"][2].update(content_sha256="same"),
    lambda m: m["clips"][5].update(role="holdout"),
    lambda m: m["clips"][5].pop("parent_name"),
    lambda m: m["clips"][5].update(parent_name="missing"),
    lambda m: m["clips"][1].update(name="clip0"),
])
def test_fold_planning_rejects_missing_provenance_or_leakage(change):
    data = manifest()
    change(data)
    with pytest.raises(ValueError):
        grouped_folds(data)


def test_hash_canonicalizes_object_keys_but_not_values():
    assert json_hash({"a": 1, "b": 2}) == json_hash({"b": 2, "a": 1})
    assert json_hash({"a": 1}) != json_hash({"a": 2})


@pytest.mark.asyncio
async def test_offline_cli_rescores_report_and_marks_no_inference(tmp_path):
    labels = tmp_path / "labels.json"
    source = tmp_path / "recorded.json"
    output = tmp_path / "out.json"
    labels.write_text(json.dumps(LABELS), encoding="utf-8")
    source.write_text(json.dumps({"config": {"value": 1}, "results": [result("clean"), result("positive", [event()])]}), encoding="utf-8")
    code = await main(["--labels", str(labels), "--report", str(source), "--out", str(output), "--gate", "--expect-count", "2", "--expect-clean", "1"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0 and report["gate"]["passed"]
    assert report["provenance"]["mode"] == "cached_report"
    assert report["provenance"]["inference_executed"] is False
    assert report["provenance"]["cross_validation_claim"] is False
    assert report["provenance"]["config_sha256"] == json_hash({"value": 1})
    assert report["summary"]["accuracy"] == 1


@pytest.mark.asyncio
async def test_offline_cli_failure_exits_nonzero(tmp_path):
    labels = tmp_path / "labels.json"
    source = tmp_path / "recorded.json"
    labels.write_text(json.dumps(LABELS), encoding="utf-8")
    source.write_text(json.dumps({"results": [result("clean"), {"name": "positive", "error": "timeout"}]}), encoding="utf-8")
    assert await main(["--labels", str(labels), "--report", str(source), "--out", str(tmp_path / "out.json"), "--gate"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("extra,payload", [(["--only", "clean", "--gate"], {"results": []}),
                                          (["--only", "unknown"], {"results": []}),
                                          ([], {"summary": {"accuracy": 1}})])
async def test_offline_cli_rejects_subset_gate_unknown_names_or_aggregate_only(tmp_path, extra, payload):
    labels = tmp_path / "labels.json"
    source = tmp_path / "recorded.json"
    labels.write_text(json.dumps(LABELS), encoding="utf-8")
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        await main(["--labels", str(labels), "--report", str(source), "--out", str(tmp_path / "out.json"), *extra])
    assert error.value.code == 2


@dataclass
class FakeSpecialistAnalysis:
    approved_types: tuple[str, ...] = ()
    provenance: dict | None = None


class FakeImage:
    closed = False

    def close(self):
        self.closed = True


def fake_analysis():
    return SimpleNamespace(events=[], candidates=[], states=[], verifications=[], layout={}, timings={}, audit={})


@pytest.mark.asyncio
async def test_live_run_uses_service_native_decode_and_candidate_override_only_in_evaluator(monkeypatch, tmp_path):
    image = FakeImage()
    observed = {}
    original = FakeSpecialistAnalysis(provenance={"checkpoint_sha256": "a" * 64})

    def decode(path, cfg, runtime):
        observed["decode"] = (path, cfg, runtime)
        return SimpleNamespace(duration=10, frames=[SimpleNamespace(image=image)]), original

    async def analyze(client, video, cfg, *, specialist_analysis):
        observed["specialist"] = specialist_analysis
        return fake_analysis()

    monkeypatch.setitem(sys.modules, "forte.fast.service", SimpleNamespace(decode_for_analysis=decode))
    monkeypatch.setitem(sys.modules, "forte.fast.analyzer", SimpleNamespace(analyze=analyze))
    runtime = object()
    config = {"specialists": {"enabled": True}}
    row = await run_one("synthetic", tmp_path / "synthetic.mov", object(), config,
                        runtime=runtime, runtime_lock=asyncio.Lock(), candidate_types=("NO_CLIENT_ID",))
    assert observed["decode"] == (str(tmp_path / "synthetic.mov"), config, runtime)
    assert observed["specialist"].approved_types == ("NO_CLIENT_ID",)
    assert original.approved_types == ()
    assert row["specialist_provenance"] == {**original.provenance, "evaluation_only": True}
    assert image.closed


@pytest.mark.asyncio
async def test_live_run_records_swallowed_model_failure_instead_of_clean(monkeypatch, tmp_path):
    image = FakeImage()

    def decode(path, cfg):
        return SimpleNamespace(duration=10, frames=[SimpleNamespace(image=image)]), None

    class Client:
        async def chat_json(self, *_args, **_kwargs):
            raise TimeoutError("synthetic model timeout")

    async def analyze(client, video, cfg, **_kwargs):
        try:
            await client.chat_json([])
        except TimeoutError:
            pass
        return fake_analysis()

    monkeypatch.setitem(sys.modules, "forte.fast.service", SimpleNamespace(decode_for_analysis=decode))
    monkeypatch.setitem(sys.modules, "forte.fast.analyzer", SimpleNamespace(analyze=analyze))
    row = await run_one("synthetic", tmp_path / "synthetic.mov", Client(), {})
    assert row["error"] == "model_call_failed:TimeoutError"
    report = score_results({"synthetic": {"violation": False}}, [row])
    assert report["summary"]["accuracy"] == report["summary"]["tn"] == 0
    assert image.closed


@pytest.mark.asyncio
async def test_shared_specialist_decode_never_interleaves_videos(monkeypatch, tmp_path):
    active = 0
    peak = 0
    images = []

    def decode(path, cfg, runtime):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(0.01)
        image = FakeImage()
        images.append(image)
        active -= 1
        return SimpleNamespace(duration=10, frames=[SimpleNamespace(image=image)]), FakeSpecialistAnalysis()

    async def analyze(*_args, **_kwargs):
        await asyncio.sleep(0)
        return fake_analysis()

    monkeypatch.setitem(sys.modules, "forte.fast.service", SimpleNamespace(decode_for_analysis=decode))
    monkeypatch.setitem(sys.modules, "forte.fast.analyzer", SimpleNamespace(analyze=analyze))
    lock = asyncio.Lock()
    await asyncio.gather(*(run_one(str(i), tmp_path / f"{i}.mov", object(), {}, runtime=object(), runtime_lock=lock) for i in range(4)))
    assert peak == 1 and len(images) == 4 and all(image.closed for image in images)


@pytest.mark.asyncio
async def test_candidate_cli_binds_actual_runtime_identity_and_loads_it_once(monkeypatch, tmp_path):
    from tools import fast_eval

    labels = tmp_path / "labels.json"
    config = tmp_path / "config.json"
    output = tmp_path / "out.json"
    labels.write_text(json.dumps(LABELS), encoding="utf-8")
    config.write_text(json.dumps({"specialists": {"enabled": True}}), encoding="utf-8")
    for name in LABELS:
        (tmp_path / f"{name}.mov").write_bytes(b"synthetic fixture; never decoded")
    gate = {"checkpoint_sha256": "a" * 64, "dataset_lock_sha256": "b" * 64, "config_sha256": "c" * 64}
    runtime = SimpleNamespace(gate=gate, provenance={"gate_sha256": "d" * 64})
    calls = []

    class RuntimeFactory:
        @classmethod
        def from_config(cls, cfg):
            calls.append(cfg)
            return runtime

    class Client:
        calls = prompt_tokens = 0

        def __init__(self, *_args, **_kwargs):
            pass

        async def aclose(self):
            pass

    async def run(name, path, client, cfg, **kwargs):
        assert kwargs["runtime"] is runtime
        assert kwargs["candidate_types"] == ("NO_CLIENT_ID",)
        return result(name, [event()] if name == "positive" else [])

    monkeypatch.setitem(sys.modules, "forte.fast.analyzer", SimpleNamespace(DEFAULT_CONFIG={}))
    monkeypatch.setitem(sys.modules, "forte.fast.specialists", SimpleNamespace(SpecialistRuntime=RuntimeFactory))
    monkeypatch.setitem(sys.modules, "forte.fast.vlm", SimpleNamespace(VLMClient=Client))
    monkeypatch.setattr(fast_eval, "run_one", run)
    code = await main(["--labels", str(labels), "--videos", str(tmp_path), "--base-url", "http://synthetic.invalid/v1",
                       "--config", str(config), "--candidate-types", "NO_CLIENT_ID", "--out", str(output), "--gate"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0 and len(calls) == 1
    assert report["provenance"]["specialist_identity"] == {**gate, "gate_sha256": "d" * 64}
    assert report["provenance"]["evaluation_mode"] == "candidate_only"
    assert report["provenance"]["provisional_types"] == ["NO_CLIENT_ID"]
    assert all("sha256" in row for row in report["provenance"]["inputs"])


@pytest.mark.asyncio
async def test_baseline_cannot_silently_change_label_identity(tmp_path):
    labels = tmp_path / "labels.json"
    source = tmp_path / "recorded.json"
    previous = tmp_path / "baseline.json"
    labels.write_text(json.dumps(LABELS), encoding="utf-8")
    rows = [result("clean"), result("positive", [event()])]
    source.write_text(json.dumps({"results": rows}), encoding="utf-8")
    previous.write_text(json.dumps({"results": rows, "labels_sha256": "changed"}), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        await main(["--labels", str(labels), "--report", str(source), "--baseline", str(previous),
                    "--out", str(tmp_path / "out.json"), "--gate"])
    assert error.value.code == 2
