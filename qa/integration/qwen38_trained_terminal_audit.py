"""Read safe terminal projections only; never access source footage/model outputs."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A09 = ROOT / "benchmarks/private/qwen38/accuracy-diagnostic-a09/trained-pilot-r1"
A13 = ROOT / "benchmarks/private/qwen38/accuracy-diagnostic-a13/trained-pilot-terminal"
checks, hashes, formatting = {}, {}, {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = path.read_bytes()
    hashes[path.relative_to(ROOT).as_posix()] = sha(raw)
    return json.loads(raw)


def check(name, condition):
    checks[name] = bool(condition)
    assert condition, name


def main():
    s = read(A09 / "terminal-settlement.a09.json")
    check("genuine_settlement_raw_pin", hashes[(A09 / "terminal-settlement.a09.json").relative_to(ROOT).as_posix()]
          == "3356e4b8ddedbee173197cd3951dcd52986f7d0f32a9ff8365bb21055b93df45")
    source = read(A09 / "actual-source-technical-projection.a09.json")
    slots = read(A09 / "actual-model-slot-projection.a09.json")
    summary = read(A09 / "actual-helper.output-summary.original.json")
    controller = read(A09 / "actual-helper.controller.original.json")
    plan, reservation, release = (read(A09 / name) for name in ("model-plan.json", "reservation.json", "root-release.json"))
    handoff = read(A13 / "handoff.json")
    aggregate = read(A13 / "technical-native-audit.aggregate.json")
    export_source = read(A13 / "source-terminal-projection.json")
    export_slots = read(A13 / "model-slot-terminals.json")
    check("a09_exact_projection_raw_pins", sha((A09 / "actual-source-technical-projection.a09.json").read_bytes())
          == s["sanitized_source_projection_sha256"] and sha((A09 / "actual-model-slot-projection.a09.json").read_bytes())
          == s["sanitized_model_slot_projection_sha256"])
    original = s["host_original_metadata_sha256"]
    check("original_summary_controller_raw_pins", sha((A09 / "actual-helper.output-summary.original.json").read_bytes())
          == original["output/results/summary.json"] and sha((A09 / "actual-helper.controller.original.json").read_bytes())
          == original["controller.json"] == handoff["controller_raw_sha256"])
    check("actual_plan_release_reservation_raw_pins", sha((A09 / "model-plan.json").read_bytes())
          == s["attempt_plan_sha256"] == reservation["attempt_plan_sha256"] == controller["plan_raw_sha256"]
          == handoff["plan_raw_sha256"] == original["metadata/model-plan.json"]
          and sha((A09 / "root-release.json").read_bytes()) == s["root_release_sha256"]
          == reservation["release_sha256"] == controller["release_raw_sha256"] == handoff["release_raw_sha256"]
          == original["metadata/root-release.json"]
          and sha((A09 / "reservation.json").read_bytes()) == s["attempt_reservation_sha256"]
          == handoff["reservation_raw_sha256"] == original["metadata/reservation.json"])
    check("original_source_slot_raw_pins", original["output/results/source-terminals.json"]
          == handoff["original_artifact_raw_sha256"]["source-terminals.json"]
          and original["output/results/model-slot-terminals.json"]
          == handoff["original_artifact_raw_sha256"]["model-slot-terminals.json"])
    for name, pin in (("source-terminal-projection.json", handoff["exported_source_projection_raw_sha256"]),
                      ("model-slot-terminals.json", handoff["exported_slots_raw_sha256"])):
        raw = (A13 / name).read_bytes()
        formatting[name] = {"local_raw_sha256": sha(raw), "declared_remote_LF_raw_sha256": pin,
            "local_CRLF_to_LF_sha256": sha(raw.replace(b"\r\n", b"\n")), "actual_local_raw_matches_declared": sha(raw) == pin}
        check("a13_export_LF_pin_" + name, sha(raw.replace(b"\r\n", b"\n")) == pin)
    check("summary_export_semantically_equal", summary == read(A13 / "summary.json"))
    rows, posts = source["records"], slots["records"]
    check("six_unique_source_identities_match_plan", len(rows) == len({r["attempt_id"] for r in rows}) == 6
          and {r["attempt_id"] for r in rows} == {r["attempt_id"] for r in plan["source_attempts"]})
    check("24_unique_slots_match_plan_reservation", len(posts) == len({r["attempt_id"] for r in posts}) == 24
          and {r["attempt_id"] for r in posts} == {r["attempt_id"] for r in reservation["attempts"]}
          == {r["attempt_id"] for r in plan["attempts"]})
    exported_posts = {r["attempt_id"]: r for r in export_slots["records"]}
    check("slot_projection_shared_fields_equal", set(exported_posts) == {r["attempt_id"] for r in posts}
          and all(all(exported_posts[r["attempt_id"]].get(k) == v for k, v in r.items()) for r in posts)
          and all(set(exported_posts[r["attempt_id"]]) - set(r) <= {"error_type"} for r in posts)
          and [r.get("error_type") for r in export_slots["records"] if r["status"] == "failed"] == ["CancelledError"])
    export_by_id = {r["attempt_id"]: r for r in export_source["records"]}
    check("source_projection_shared_fields_equal", all(all(r[k] == export_by_id[r["attempt_id"]][k]
          for k in r.keys() & export_by_id[r["attempt_id"]].keys()) for r in rows))
    check("one_failed_five_unreleased", dict(Counter(r["status"] for r in rows))
          == s["source_terminal_counts"] == {"failed": 1, "not_attempted": 5}
          and all(r.get("release_state") == "not_released" for r in rows if r["status"] == "not_attempted"))
    pilot = next(r for r in rows if r["status"] == "failed")
    check("all_slots_join_one_failed_pilot_content", {r["source_attempt_id"] for r in posts} == {pilot["attempt_id"]}
          and {r["content_sha256"] for r in posts} == {pilot["content_sha256"]})
    counts = dict(Counter(r["status"] for r in posts))
    check("seven_done_one_failed_sixteen_unused", counts == s["reconciliation"]["status_counts"]
          == s["model_slot_terminal_status_counts"] == summary["model_slot_terminal_counts"]
          == handoff["model_slot_terminal_counts"] == {"done": 7, "failed": 1, "not_attempted": 16})
    check("eight_started_seven_http200", sum(bool(r.get("started_at_utc")) for r in posts)
          == s["actual_model_posts"] == handoff["actual_generation_posts"] == aggregate["model_posts_started"] == 8
          and sum(r.get("http_status") == 200 for r in posts) == s["actual_http200_completed_posts"]
          == aggregate["model_posts_http200"] == 7)
    by_id = {r["attempt_id"]: r for r in posts}
    check("settlement_outcomes_join_exact", len(s["outcomes"]) == 24
          and {r["attempt_id"] for r in s["outcomes"]} == set(by_id)
          and all(all(by_id[r["attempt_id"]].get(k) == v for k, v in r.items()) for r in s["outcomes"]))
    check("all_spent_no_refund_retry_or_expansion", s["all_slots_consumed"] is True
          and s["exposure_refunded"] is False and s["retry_authorized"] is False
          and s["new_attempt_authority"] is False and s["expansion_or_promotion_released_by_this_settlement"] is False
          and s["reconciliation"]["complete"] is True and s["unresolved_attempts"] == 0
          and s["reconciliation"]["recorded_attempts"] == s["reconciliation"]["reserved_attempts"] == 24)
    check("zero_results_native_proofs_and_technical_completion", s["source_publication_count"]
          == s["technically_complete_sources"] == summary["technically_complete_sources"]
          == aggregate["server_native_evidence_receipts"] == 0
          and aggregate["publication_ready"] is False and aggregate["independent_full_native_export_created"] is False
          and all(r.get("public_contract_validated") is False for r in rows))
    check("helper_terminal_but_server_drain_not_inferred", controller["container_id"] == s["actual_helper_container_id"]
          and controller["host_pid"] == s["actual_helper_historical_host_pid"] == 599477
          and s["actual_helper_terminal_state"]["Running"] is False
          and s["actual_helper_terminal_state"]["Pid"] == 0 and s["actual_helper_terminal_state"]["ExitCode"] == 0
          and s["actual_helper_terminal_state"]["OOMKilled"] is False and aggregate["server_idle_verified"] is False
          and aggregate["server_active_generation"] == "unknown_after_client_deadline_cancellation")
    check("static_failure_and_denominator_consistent", pilot["error_type"] == "TimeoutError"
          and aggregate["static_error_code"] == "refinement_model_incomplete"
          and s["source_population_denominator"] == summary["source_attempts_required"] == 6
          and s["source_pipeline_count"] == s["actual_video_attempts"] == summary["released_source_attempts"] == 1
          and s["source_not_released"] == aggregate["other_sources_unreleased"] == 5
          and summary["final_check_used"] is False and summary["automatic_private_retry_or_advance"] is False)
    receipt = {"schema_version": "forte.qwen38.independent-trained-terminal-audit.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "status": "verified_terminal_failure",
        "checks": checks, "artifact_sha256": hashes, "a13_text_export_formatting": formatting,
        "source_denominator": 6, "failed_sources": 1, "other_sources_unreleased_not_attempted": 5,
        "reserved_slots_consumed": 24, "started_model_slots": 8, "done_http200_slots": 7,
        "failed_started_slots": 1, "unused_notattempted_slots_consumed": 16,
        "terminal_error_type": "TimeoutError", "terminal_static_code": "refinement_model_incomplete",
        "publication_validated_results": 0, "verified_server_native_evidence_receipts": 0, "technical_completions": 0,
        "pipeline_native_pts_refs_reported_by_a13": 85, "full_native_export_or_evidence_schema_success_claimed": False,
        "server_drain_at_helper_exit": "unknown_after_client_deadline_cancellation",
        "projection_note": "A09 safe slot projection omits one CancelledError class retained by A13; all shared fields and IDs match. An initially overstrict whole-record equality assertion was corrected without changing either artifact.",
        "transport_starts_are_not_all_server_completed_posts": True, "binary_or_event_accuracy_claimed": False,
        "failed_source_is_not_negative_prediction": True, "retry_or_promotion_authorized": False,
        "private_predictions_or_media_exported_by_qa": False, "model_gpu_http_service_exposure_operations_by_qa": 0}
    output = ROOT / "qa/integration/qwen38_trained_terminal_audit.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "receipt_sha256": sha(output.read_bytes())}))


if __name__ == "__main__":
    main()
