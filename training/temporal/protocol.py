"""Bind W2 runtime settings to its predeclared objective, before validation."""
from pathlib import Path

from training.guards import canonical_sha256, file_sha256, load_json, require

SETTINGS = ("experiment_id", "seed", "epochs", "max_steps", "learning_rate", "weight_decay", "hidden_dim",
            "mil_temperature", "teacher_weight", "component_loss_weights", "coarse_anchor_weight",
            "coarse_anchor_tolerance_sec", "freeze_unsupervised_branches", "mask_cross_window_target_conflicts",
            "expected_initial_checkpoint_manifest_sha256", "expected_feature_cache_index_sha256",
            "expected_fact_axis_manifest_sha256", "validation_reuse_disclosure")


def validate_protocol_config(config: dict) -> dict | None:
    if config.get("experiment_id") != "W2" and not config.get("protocol_manifest"):
        return None
    path = Path(config["protocol_manifest"]).resolve()
    require(path.is_file() and file_sha256(path) == config.get("expected_protocol_manifest_sha256"),
            "weak_protocol_manifest_pin_mismatch")
    protocol = load_json(path)
    require(protocol.get("schema_version") == "forte.predeclared-weak-protocol.v1"
            and all(config.get(key) == protocol.get(key) for key in SETTINGS), "weak_predeclared_protocol_settings_changed")
    require(protocol.get("binary_negative_instance_weight") == 1.0 and protocol.get("gradient_clip_norm") == 1.0
            and protocol.get("optimizer") == "AdamW_fresh_state_parent_weights_only"
            and protocol.get("video_threshold") == {"score_axis": "violation_log_odds", "minimum_video_score": 0.0,
                 "comparison": "greater_than_or_equal", "selection_method": "predeclared_fixed_zero"}
            and bool(config.get("initial_checkpoint_dir")), "weak_predeclared_protocol_objective_invalid")
    return protocol


def require_protocol_supervision(protocol: dict | None, coverage: dict) -> None:
    if protocol is not None and protocol.get("experiment_id") == "W2":
        require(coverage.get("recording_bags") == 28
                and coverage.get("silver_frame_fact_labels", 0) > 0
                and coverage.get("silver_support_negative_labels", 0) > 0
                and coverage.get("provided_coarse_positive_anchor_bags", 0) == 15,
                "weak_W2_declared_supervision_not_available")


def validate_question_policy(config: dict) -> dict | None:
    if config.get("experiment_id") != "W2" and not config.get("anchor_question_policy"):
        return None
    path = Path(config["anchor_question_policy"])
    require(file_sha256(path) == config.get("expected_anchor_question_policy_sha256"),
            "weak_question_policy_file_pin_mismatch")
    policy = load_json(path)
    payload = {key: value for key, value in policy.items() if key != "policy_sha256"}
    require(policy.get("schema_version") == "forte.anchor-question-policy.v1"
            and policy.get("policy_sha256") == canonical_sha256(payload)
            and policy.get("question_count") == 8 and len(set(policy.get("fact_ids", []))) == 8
            and policy.get("scope") == "exact_anchor_frame"
            and policy.get("derivation") == "direct_anchor_teacher_question", "weak_question_policy_binding_invalid")
    return policy
