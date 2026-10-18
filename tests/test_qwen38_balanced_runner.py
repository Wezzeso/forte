"""Optional balanced objective integration checks; synthetic TRAIN facts only."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.qwen38_silver import (BALANCED_LOSS, CANONICAL_FRAMES, COMPACT_FRAMES, UNWEIGHTED_LOSS,
    apply_supervision_policy, collect_release_bindings, collate_objective_batch, objective_loss,
    supervision_summary)


def rows_fixture():
    result = []
    for index in range(4):
        content = str(index + 1) * 64
        first, focal = str(index + 5) * 64, str(index + 1) * 63 + "f"
        states = {"CASH_VISIBLE": "present" if index < 2 else "absent",
                  "CASH_HANDLER_PRESENT": "present" if index % 2 == 0 else "absent",
                  "CASH_PLACED_IN_CLOTHING_POCKET": "absent"}
        result.append({"sample_id": "sample-" + str(index), "content_sha256": content,
            "focal_frame_id": focal, "frames": [{"frame_id": first}, {"frame_id": focal}],
            "prompt": "Inspect exact focal " + focal, "split": "train", "annotation_origin": "local_model",
            "human_reviewed": False, "target_scope": "exact_anchor_frame",
            "fact_supervision_mask": dict.fromkeys(states, True),
            "target": {"facts": [{"fact_id": fact, "state": state, "supporting_frame_ids": [focal]}
                                  for fact, state in states.items()], "candidates": []}})
    return result


def config_fixture():
    return {"silver_loss_policy": BALANCED_LOSS, "silver_balance_minimum_class_contents": 2,
            "frame_alias_policy": COMPACT_FRAMES}


def pin_plan(config, rows, path):
    effective, report = apply_supervision_policy(config, rows, require_pins=False)
    path.write_text(json.dumps(report["balance_plan"], sort_keys=True, indent=2) + "\n")
    config.update(silver_balance_plan=str(path), expected_silver_balance_plan_sha256=file_sha256(path),
        expected_silver_balance_plan_semantic_sha256=report["balance_plan_semantic_sha256"],
        expected_effective_rows_sha256=report["effective_rows_sha256"],
        expected_effective_fact_weights_sha256=report["effective_fact_weights_sha256"])
    return effective, report


def test_unweighted_default_keeps_identical_rows_and_targets():
    rows = rows_fixture()
    effective, report = apply_supervision_policy({}, rows)
    assert effective is rows
    assert report["effective_rows_sha256"] == canonical_sha256(rows)
    assert report["balance_plan"] is None and report["alias_receipt"] is None
    assert report["objective"]["loss_policy"] == UNWEIGHTED_LOSS


def test_balance_then_alias_is_pinned_and_raw_bytes_not_changed(tmp_path):
    rows, config = rows_fixture(), config_fixture()
    original = canonical_sha256(rows)
    expected, report = pin_plan(config, rows, tmp_path / "balance.json")
    effective, checked = apply_supervision_policy(config, rows)
    assert canonical_sha256(rows) == original
    assert effective == expected and checked["effective_rows_sha256"] == report["effective_rows_sha256"]
    assert checked["canonical_target_rows_sha256"] == original
    assert checked["alias_receipt"]["balance_must_precede_aliasing"]
    assert all(row["focal_frame_id"] == "f01" for row in effective)
    for row in effective:
        assert row["canonical_focal_frame_id"] not in row["prompt"]
        pocket = next(fact for fact in row["target"]["facts"] if fact["fact_id"] == "CASH_PLACED_IN_CLOTHING_POCKET")
        assert pocket["state"] == "unclear" and pocket["supporting_frame_ids"] == []
        assert row["fact_loss_weights"][pocket["fact_id"]] == 0
    for key in ("expected_effective_rows_sha256", "expected_effective_fact_weights_sha256"):
        with pytest.raises(TrainingGateError, match="effective_input_pin"):
            apply_supervision_policy({**config, key: "a" * 64}, rows)


def test_plan_floats_masks_and_alias_policy_cannot_change_after_pin(tmp_path):
    config, rows = config_fixture(), rows_fixture()
    pin_plan(config, rows, tmp_path / "balance.json")
    with pytest.raises(TrainingGateError, match="effective_input_pin"):
        apply_supervision_policy({**config, "frame_alias_policy": CANONICAL_FRAMES}, rows)
    path = Path(config["silver_balance_plan"])
    plan = json.loads(path.read_text())
    plan["sample_fact_weights"]["sample-0"]["CASH_VISIBLE"] *= 2
    path.write_text(json.dumps(plan))
    config["expected_silver_balance_plan_sha256"] = file_sha256(path)
    with pytest.raises(TrainingGateError, match="exact_plan_pin"):
        apply_supervision_policy(config, rows)


def test_release_binding_collector_includes_exact_effective_objective(tmp_path):
    config, rows = config_fixture(), rows_fixture()
    pin_plan(config, rows, tmp_path / "balance.json")
    _, report = apply_supervision_policy(config, rows)
    model, runtime = tmp_path / "base.json", tmp_path / "runtime.json"
    model.write_text("{}"); runtime.write_text("{}")
    config.update(model_manifest=str(model), runtime_lock=str(runtime), quantization="none",
                  expected_runtime_lock_sha256=file_sha256(runtime))
    audit = {"coverage": {"target_rows_sha256": canonical_sha256(rows)}, "supervision": report}
    bindings = collect_release_bindings(config, {"machine_annotations_sha256": "a" * 64}, audit)
    assert bindings["target_rows_sha256"] == canonical_sha256(rows)
    for key in ("effective_rows_sha256", "effective_fact_weights_sha256", "supervision_objective_sha256",
                "balance_plan_semantic_sha256", "balance_plan_file_sha256"):
        assert bindings[key] == report[key]
    summary = supervision_summary(report)
    assert "sample_fact_weights" not in json.dumps(summary) and "mappings" not in summary
    assert summary["deferred_fact_ids"] == {"CASH_PLACED_IN_CLOTHING_POCKET": "missing_state"}


def test_weighted_runner_loss_uses_only_verified_fact_tokens_and_roundtrips():
    torch = pytest.importorskip("torch")
    class Collator:
        def collate_with_fact_spans(self, rows):
            return {"input_ids": torch.tensor([[0, 1, 2, 3, 4]]),
                    "labels": torch.tensor([[-100, 1, 2, -100, 1]])}, {"CASH_VISIBLE": [1, 2], "CASH_HANDLER_PRESENT": [4]}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.arange(15, dtype=torch.float32).reshape(1, 5, 3) / 10)
        def forward(self, input_ids, labels):
            assert labels is None
            return SimpleNamespace(logits=self.logits)
    row = {"fact_loss_weights": {"CASH_VISIBLE": .2, "CASH_HANDLER_PRESENT": .3,
                                  "CASH_PLACED_IN_CLOTHING_POCKET": 0.0}}
    batch, weights = collate_objective_batch(Collator(), row, device="cpu", balanced=True)
    assert weights.tolist()[0] == pytest.approx([0, .1, .1, 0, .3])
    model = Model()
    loss = objective_loss(model, batch, weights, example_count=4)
    loss.backward()
    assert torch.equal(model.logits.grad[:, 2], torch.zeros_like(model.logits.grad[:, 2]))
    assert torch.equal(model.logits.grad[:, 4], torch.zeros_like(model.logits.grad[:, 4]))
    assert model.logits.grad[:, 0].abs().sum() > 0 and model.logits.grad[:, 3].abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    smoke = objective_loss(model, batch, weights, example_count=4).detach()
    serialized = BytesIO()
    torch.save(model.state_dict(), serialized)
    serialized.seek(0)
    reloaded = Model(); reloaded.load_state_dict(torch.load(serialized, weights_only=True))
    training = objective_loss(reloaded, batch, weights, example_count=4).detach()
    assert torch.equal(smoke, training)
    assert torch.allclose(training, objective_loss(reloaded, batch, weights, example_count=1) * 4)


def test_unweighted_runner_uses_original_model_loss():
    assert objective_loss(lambda **kwargs: SimpleNamespace(loss=3.5), {"labels": "unchanged"}, None,
                          example_count=4) == 3.5
