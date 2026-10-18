"""Synthetic metadata tests; no media, model, host or network operation."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("personal_items_probe", Path(__file__).parents[1] / "tools/personal_items_probe.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def source():
    rows = [{"name": f"X-{i}", "processed": True, "candidates": [], "layout": {"desk": [10, 20, 800, 900]}}
            for i in range(1, 41)]
    rows[4]["candidates"] = [{"type_id": "PERSONAL_ITEMS", "t": 3.0, "note": "private synthetic text"}]
    rows[7]["candidates"] = [{"type_id": "OTHER_TYPE", "t": 1.0}, {"type_id": "PERSONAL_ITEMS", "t": 5.0}]
    return {"results": rows}


def answer(**change):
    return {"object_kind": "drink_container", "recognizable": True, "on_work_surface": True,
            "consistent_across_frames": True, "fixed_equipment": False, "touched_or_lifted": False, **change}


def raw_report(expected):
    rows = []
    for row in expected["results"]:
        proposals = [{"t": item["t"], "start": 0, "end": 10, "status": "done", "answers": [answer(), answer()],
                      "supported": True, "full_frames": 6, "desk_crops": 3, "private_extra": "never exported"}
                     for item in row["proposals"]]
        rows.append({"run": row["run"], "name": row["name"], "proposals": proposals, "private_extra": "never exported"})
    return {"schema": "forte.personal-items-probe.v1", "model": "q30test", "policy": T.POLICY,
            "results": rows, "private_extra": "never exported",
            "summary": {"calls": 8, "attempted_calls": 8, "prompt_tokens": 100, "completion_tokens": 100,
                        "timed_out": False, "wall_s": 1.0}}


def test_all_pair_candidates_uniform_and_labels_never_projected():
    first, second = source(), source()
    expected = T.cached_pair(first, second, 4)
    for index, row in enumerate(first["results"]):
        row.update(label={"violation": bool(index % 2)}, correct=False,
                   events=[{"observation_ru": "private synthetic text"}])
    assert T.cached_pair(first, second, 4) == expected
    assert len(expected["results"]) == 80 and expected["proposals"] == 4
    assert sum(len(row["proposals"]) for row in expected["results"]) == 4
    assert "private" not in str(expected)


@pytest.mark.parametrize("change", [
    {"recognizable": False}, {"on_work_surface": False}, {"consistent_across_frames": False},
    {"fixed_equipment": True}, {"object_kind": "fixed_equipment"}, {"object_kind": "unclear"},
])
def test_object_checks_are_conjunctive(change):
    assert not T.supports([answer(), answer(**change)])


def test_touched_or_lifted_is_diagnostic_and_category_agreement_required():
    assert T.supports([answer(), answer()])
    assert not T.supports([answer(), answer(object_kind="coat_or_handbag")])


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_model_booleans_are_strict(bad):
    with pytest.raises(ValueError):
        T.project_answer(answer(recognizable=bad))


def test_collection_is_a_strict_projection():
    expected = T.cached_pair(source(), source(), 4)
    clean = T.project_report(raw_report(expected), expected)
    assert "private" not in str(clean)
    assert clean["pipeline_change_qualified"] is False and clean["type_enablement_qualified"] is False
    labels = {f"X-{i}": {"violation": i % 2 == 0, "private_extra": "ignored"} for i in range(1, 41)}
    score = T.score_after_inference(clean, labels)
    assert score["per_run"]["a"]["supported_on_negative_videos"] == 1
    assert score["per_run"]["a"]["supported_on_positive_videos"] == 1
    assert score["pipeline_accuracy_claim"] is False


def test_failed_proposal_remains_failed_and_cannot_be_credited_clean():
    expected = T.cached_pair(source(), source(), 4)
    raw = raw_report(expected)
    row = next(row for row in raw["results"] if row["proposals"])
    at = row["proposals"][0]["t"]
    row["proposals"][0] = {"t": at, "status": "failed", "error_class": "TimeoutError"}
    clean = T.project_report(raw, expected)
    labels = {f"X-{i}": {"violation": False} for i in range(1, 41)}
    assert T.score_after_inference(clean, labels)["per_run"]["a"]["failed_proposed_videos"] == 1
    assert row["proposals"][0]["status"] == "failed"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed_source", "call_budget", "nonfinite"])
def test_invalid_inventory_or_budget_refuses_inference(mutation):
    a, b = source(), source()
    if mutation == "missing":
        a["results"].pop()
    elif mutation == "duplicate":
        a["results"][1]["name"] = a["results"][0]["name"]
    elif mutation == "failed_source":
        a["results"][0]["processed"] = False
    elif mutation == "call_budget":
        for row in a["results"]:
            row["candidates"] = [{"type_id": "PERSONAL_ITEMS", "t": 1}]
    else:
        a["results"][4]["candidates"][0]["t"] = float("nan")
    with pytest.raises(ValueError):
        T.cached_pair(a, b, 4)


def test_host_wrapper_python_compiles_and_user_mount_guards_are_present():
    path = Path(__file__).parents[1] / "tmp/agent-bridge/payload/q30_personal_items_probe.remote.sh"
    text = path.read_text(encoding="utf-8")
    body = text.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(body, str(path), "exec")
    for token in ("'--user', user", "'--cap-drop', 'ALL'", "'container:' + MODEL_ID", "'--read-only'",
                  "labels_opened_only_after_all_queries=True", "'maximum_wrapper_seconds': 178"):
        assert token in body
