"""New experimental rule paths use only generated source IDs and observations."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from forte.events.experimental import (
    ExperimentalEpisode, ExperimentalTemporalContext, ExperimentalTrack,
    build_experimental_events, validate_experimental_vocabulary,
)
from forte.shared.schemas import CandidateEvent, FactDecision, FrameRef, VideoAsset
from rules.catalog import load_catalog
from rules.experimental_catalog import RULE_IDS, load_experimental_catalog


@pytest.fixture
def source(tmp_path):
    asset = VideoAsset(job_id="experimental-synthetic", owned_path=tmp_path / "generated.mov",
                       content_hash="0" * 64, duration_sec=10.0, width=64, height=48, time_base="1/10")
    frames = {}
    for point in range(11):
        source_pts = point * 10
        frame_id = sha256(f"{asset.content_hash}:{source_pts}:{asset.time_base}".encode()).hexdigest()
        frames[frame_id] = FrameRef(frame_id=frame_id, pts_sec=float(point), source_pts=source_pts,
                                    path=tmp_path / f"original-{point}.jpg", width=64, height=48,
                                    job_id=asset.job_id)
    report = {"duration_sec": asset.duration_sec, "complete": True, "gaps": [], "missing_window_indices": [],
              "inspected_intervals": [{"start_sec": 0.0, "end_sec": 10.0, "frame_ids": list(frames)}]}
    return asset, frames, report


def _episode(catalog, rule_id, frames, *, context_id="episode", candidate_id="candidate", clip_id="window"):
    binding = catalog.bindings[rule_id]
    policy = binding["experimental_temporal"]
    phases = policy["ordered_phases"]
    points = [0, 2, 4, 6] if len(phases) == 4 else [0, 3, 6]
    by_time = {frame.pts_sec: frame.frame_id for frame in frames.values()}
    scope = tuple(by_time[float(point)] for point in range(7))
    phase_frames = {phase["phase_id"]: (by_time[float(point)],) for phase, point in zip(phases, points)}
    states = binding["required_fact_states"]
    facts = {key: FactDecision(fact_id=key, state=state, supporting_frame_ids=list(scope))
             for key, state in states.items()}
    phase_support = {}
    for phase in phases:
        for fact_id in phase["fact_ids"]:
            phase_support.setdefault(fact_id, set()).update(phase_frames[phase["phase_id"]])
    properties = set(binding["context_fact_ids"]) | set(policy["track_fact_ids"].values())
    for key, support in phase_support.items():
        if key not in properties and states.get(key, "present") == "present":
            facts[key] = FactDecision(fact_id=key, state="present", supporting_frame_ids=sorted(support))
    for key, state in states.items():
        if state == "absent" and not policy["absence_must_cover_procedure"]:
            ids = {frame_id for phase in policy["absence_phase_ids"] for frame_id in phase_frames[phase]}
            facts[key] = facts[key].model_copy(update={"supporting_frame_ids": sorted(ids)})
    tracks = {}
    for kind in policy["required_track_kinds"]:
        ids = {frame_id for phase in policy["track_phase_ids"][kind] for frame_id in phase_frames[phase]}
        tracks[kind] = ExperimentalTrack(track_id=f"local-{kind}-1", frame_ids=tuple(sorted(ids)))
        key = policy["track_fact_ids"][kind]
        facts[key] = facts[key].model_copy(update={"supporting_frame_ids": sorted(ids)})
    current, prior = "procedure-current", "procedure-prior" if policy["distinct_procedure_ids"] else None
    procedure_ids = {phase["phase_id"]: prior if policy["phase_procedure_roles"][phase["phase_id"]] == "prior" else current
                     for phase in phases}
    event_phase = ("count_with_client_area_empty" if "count_with_client_area_empty" in phase_frames else
                   "no_count_handover" if "no_count_handover" in phase_frames else phases[-1]["phase_id"])
    anchor_id = phase_frames[event_phase][0]
    anchor = frames[anchor_id].pts_sec
    candidate = CandidateEvent(candidate_id=candidate_id, event_type_id=binding["event_type_id"],
        start_sec=anchor, end_sec=anchor, supporting_facts=[facts[key] for key in sorted(facts)],
        supporting_frame_ids=[anchor_id], clip_id=clip_id)
    context = ExperimentalTemporalContext(context_id=context_id, procedure_id=current, prior_procedure_id=prior,
        full_context_start_frame_id=by_time[0.0], full_context_end_frame_id=by_time[6.0],
        inspected_frame_ids=scope, phase_frame_ids=phase_frames, phase_procedure_ids=procedure_ids, tracks=tracks)
    return ExperimentalEpisode(rule_id=rule_id, vocabulary_version=catalog.vocabulary_version,
                                candidate=candidate, context=context)


def _verify(source, episodes, catalog=None):
    asset, frames, coverage = source
    return build_experimental_events(episodes, asset=asset, catalog=catalog or load_experimental_catalog(),
                                     frames=frames, coverage=coverage)


def _replace_fact(episode, fact_id, *, state=None, support=None):
    values = []
    for fact in episode.candidate.supporting_facts:
        if fact.fact_id == fact_id:
            update = {}
            if state is not None:
                update["state"] = state
            if support is not None:
                update["supporting_frame_ids"] = list(support)
            fact = fact.model_copy(update=update)
        values.append(fact)
    return episode.model_copy(update={"candidate": episode.candidate.model_copy(update={"supporting_facts": values})})


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_four_priority_branches_produce_source_grounded_experimental_events(source, rule_id):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, rule_id, source[1])
    result = _verify(source, [episode], catalog)
    assert not result.pending, result.pending
    assert len(result.verified_events) == 1
    event = result.verified_events[0]
    assert event.rule_id == rule_id and event.anchor_frame_id in event.evidence_frame_ids
    assert 1 <= len(event.evidence_frame_ids) <= 3
    reference = result.source_references[0]
    assert reference.source_url == catalog.rules[rule_id].source_url
    assert reference.clause == catalog.rules[rule_id].clause and len(reference.source_sha256) == 64
    assert result.runtime_enabled is False and not hasattr(result, "verdict")
    if catalog.bindings[rule_id]["is_omission"]:
        assert event.anchor_frame_id == episode.context.full_context_end_frame_id


def test_identical_overlap_proposal_deduplicates_without_hiding_separate_episodes(source):
    catalog = load_experimental_catalog()
    one = _episode(catalog, RULE_IDS[2], source[1])
    two = one.model_copy(update={"candidate": one.candidate.model_copy(update={"candidate_id": "overlap", "clip_id": "other-window"})})
    separate = _episode(catalog, RULE_IDS[2], source[1], context_id="another-receipt", candidate_id="repeat")
    result = _verify(source, [one, two, one, separate], catalog)
    assert not result.pending and len(result.verified_events) == 2


def test_reversed_source_phase_order_stays_pending(source):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[0], source[1])
    phases = dict(episode.context.phase_frame_ids)
    names = list(phases)
    phases[names[0]], phases[names[1]] = phases[names[1]], phases[names[0]]
    episode = episode.model_copy(update={"context": episode.context.model_copy(update={"phase_frame_ids": phases})})
    result = _verify(source, [episode], catalog)
    assert not result.verified_events and result.pending


def test_prior_cash_requires_two_real_procedure_scopes(source):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[0], source[1])
    episode = episode.model_copy(update={"context": episode.context.model_copy(update={
        "prior_procedure_id": episode.context.procedure_id})})
    result = _verify(source, [episode], catalog)
    assert result.pending[0].reason == "experimental_distinct_procedures_not_established"


def test_phase_from_another_procedure_cannot_fill_a_gap(source):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[2], source[1])
    ids = dict(episode.context.phase_procedure_ids)
    ids["procedure_complete"] = "unrelated-procedure"
    episode = episode.model_copy(update={"context": episode.context.model_copy(update={"phase_procedure_ids": ids})})
    assert _verify(source, [episode], catalog).pending[0].reason == "experimental_phase_procedure_mismatch"


def test_client_track_is_required_before_exit_and_not_during_absence(source):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[1], source[1])
    track = episode.context.tracks["contributor"]
    absence = episode.context.phase_frame_ids["count_with_client_area_empty"]
    assert not set(track.frame_ids).intersection(absence)
    assert _verify(source, [episode], catalog).verified_events
    tracks = dict(episode.context.tracks)
    tracks.pop("contributor")
    episode = episode.model_copy(update={"context": episode.context.model_copy(update={"tracks": tracks})})
    assert _verify(source, [episode], catalog).pending[0].reason == "experimental_track_continuity_missing"


@pytest.mark.parametrize("rule_id", RULE_IDS[2:])
def test_sparse_absence_does_not_establish_whole_procedure_omission(source, rule_id):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, rule_id, source[1])
    episode = _replace_fact(episode, "CASH_COUNT_VISIBLE", support=episode.candidate.supporting_frame_ids)
    result = _verify(source, [episode], catalog)
    assert result.pending[0].reason == "experimental_absence_scope_incomplete"


@pytest.mark.parametrize("fact_id", ["PROCEDURE_CONTEXT_COMPLETE", "ALL_COUNTING_PATHS_VISIBLE", "WORK_AREA_CONTINUOUSLY_VISIBLE"])
def test_unknown_or_incomplete_context_never_turns_into_absence(source, fact_id):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[2], source[1])
    episode = _replace_fact(episode, fact_id, state="unclear", support=())
    result = _verify(source, [episode], catalog)
    assert not result.verified_events and result.pending


@pytest.mark.parametrize("rule_id,fact_id,state", [
    (RULE_IDS[0], "EXP_PRIOR_CASH_IDLE_ON_ORDINARY_COUNTER", "absent"),
    (RULE_IDS[1], "EXP_ANY_CLIENT_AREA_PERSON_PRESENT", "present"),
    (RULE_IDS[1], "EXP_ALL_CLIENT_AREAS_CONTINUOUSLY_VISIBLE", "unclear"),
    (RULE_IDS[2], "EXP_LOOSE_COINS_RECEIVED", "unclear"),
    (RULE_IDS[3], "EXP_BUNDLE_WRAPPER_DAMAGE_CLEAR", "absent"),
    (RULE_IDS[3], "EXP_BANKNOTE_BUNDLE_IDENTIFIABLE", "unclear"),
])
def test_normal_lookalikes_and_clause_exceptions_do_not_verify(source, rule_id, fact_id, state):
    catalog = load_experimental_catalog()
    episode = _replace_fact(_episode(catalog, rule_id, source[1]), fact_id, state=state)
    result = _verify(source, [episode], catalog)
    assert not result.verified_events and result.pending


def test_missing_declared_frames_or_technical_coverage_blocks_temporal_verification(source):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[2], source[1])
    episode = episode.model_copy(update={"context": episode.context.model_copy(update={
        "inspected_frame_ids": episode.context.inspected_frame_ids[:-1]})})
    assert not _verify(source, [episode], catalog).verified_events
    failed = (source[0], source[1], {**source[2], "complete": False})
    episode = _episode(catalog, RULE_IDS[2], source[1])
    assert _verify(failed, [episode], catalog).pending[0].reason == "experimental_incomplete_analysis_coverage"


def test_source_frame_identity_and_opaque_track_contracts_are_enforced(source):
    asset, frames, coverage = source
    bad = dict(frames)
    key = next(iter(bad))
    bad[key] = bad[key].model_copy(update={"source_pts": 999})
    with pytest.raises(ValueError, match="source PTS"):
        build_experimental_events([], asset=asset, catalog=load_experimental_catalog(), frames=bad, coverage=coverage)
    with pytest.raises(ValueError):
        ExperimentalTrack(track_id="person real name", frame_ids=(key,))


def test_legacy_axes_and_unversioned_episode_cannot_enter_experimental_consumer(source):
    catalog = load_experimental_catalog()
    axes = catalog.vocabulary["feature_axis_order"]
    validate_experimental_vocabulary(catalog, catalog.vocabulary_version, axes)
    with pytest.raises(ValueError, match="vocabulary"):
        validate_experimental_vocabulary(catalog, catalog.vocabulary_version, axes[:19])
    with pytest.raises(ValueError, match="vocabulary"):
        validate_experimental_vocabulary(catalog, "forte.legacy.fact_axes.v1", axes)
    episode = _episode(catalog, RULE_IDS[2], source[1]).model_copy(update={"vocabulary_version": "legacy-19"})
    assert _verify(source, [episode], catalog).pending[0].reason == "experimental_episode_vocabulary_mismatch"


def test_default_catalog_cannot_accidentally_use_experimental_consumer(source):
    with pytest.raises(ValueError, match="dedicated explicit loader"):
        _verify(source, [], load_catalog())


def test_loading_experiment_leaves_default_catalog_and_axis_order_unchanged(source):
    path = Path("rules/catalog.json")
    before = sha256(path.read_bytes()).hexdigest()
    baseline_axes = [fact.fact_id for fact in load_catalog().fact_definitions()]
    catalog = load_experimental_catalog()
    result = _verify(source, [_episode(catalog, RULE_IDS[2], source[1])], catalog)
    assert result.verified_events
    assert before == sha256(path.read_bytes()).hexdigest()
    assert [fact.fact_id for fact in load_catalog().fact_definitions()] == baseline_axes
    assert len(baseline_axes) == 19 and catalog.vocabulary["feature_axis_order"][:19] == baseline_axes


def test_procedure_end_and_candidate_seconds_must_reference_actual_source_pts(source):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[2], source[1])
    episode = episode.model_copy(update={"candidate": episode.candidate.model_copy(update={"start_sec": 2.5, "end_sec": 2.5})})
    assert _verify(source, [episode], catalog).pending[0].reason == "experimental_candidate_boundary_not_source_pts"


@pytest.mark.parametrize("rule_id", RULE_IDS[:2])
def test_action_anchor_comes_from_event_phase_not_earlier_context(source, rule_id):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, rule_id, source[1])
    episode = episode.model_copy(update={"candidate": episode.candidate.model_copy(update={"start_sec": 0.0, "end_sec": 6.0})})
    result = _verify(source, [episode], catalog)
    event = result.verified_events[0]
    phase_id = catalog.bindings[rule_id]["experimental_temporal"]["event_phase_id"]
    expected = episode.context.phase_frame_ids[phase_id][0]
    assert event.anchor_frame_id == expected
    assert event.start_sec == event.end_sec == source[1][expected].pts_sec


def test_changed_cash_track_across_same_episode_is_not_two_supported_events(source):
    catalog = load_experimental_catalog()
    one = _episode(catalog, RULE_IDS[2], source[1])
    tracks = dict(one.context.tracks)
    tracks["cash"] = tracks["cash"].model_copy(update={"track_id": "unrelated-cash"})
    two = one.model_copy(update={"context": one.context.model_copy(update={"tracks": tracks}),
                                 "candidate": one.candidate.model_copy(update={"candidate_id": "other-window"})})
    result = _verify(source, [one, two], catalog)
    assert not result.verified_events
    assert all(item.reason == "experimental_context_procedure_or_tracks_conflict" for item in result.pending)


def test_disjoint_inspection_windows_cannot_support_complete_procedure(source):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[2], source[1])
    coverage = deepcopy(source[2])
    coverage["inspected_intervals"] = [{"start_sec": 0.0, "end_sec": 2.0, "frame_ids": []},
                                       {"start_sec": 4.0, "end_sec": 10.0, "frame_ids": []}]
    result = _verify((source[0], source[1], coverage), [episode], catalog)
    assert result.pending[0].reason == "experimental_context_coverage_gap"


def test_four_required_distinct_images_exceeds_public_evidence_budget(source):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[1], source[1])
    by_time = {frame.pts_sec: frame.frame_id for frame in source[1].values()}
    # Four global affirmative predicates have four exclusive source images.
    for fact_id, point in [("BANK_CASH_WORKSTATION", 0.0), ("CASH_RECEIPT_OPERATION", 2.0),
                           ("CASH_VISIBLE", 4.0), ("CASH_COUNT_VISIBLE", 5.0)]:
        episode = _replace_fact(episode, fact_id, support=(by_time[point],))
    phases = dict(episode.context.phase_frame_ids)
    phases["count_with_client_area_empty"] = (by_time[4.0], by_time[5.0])
    episode = episode.model_copy(update={"context": episode.context.model_copy(update={"phase_frame_ids": phases})})
    # The absence/track phase scope includes both actual count frames.
    episode = _replace_fact(episode, "EXP_ANY_CLIENT_AREA_PERSON_PRESENT", support=(by_time[4.0], by_time[5.0]))
    episode = _replace_fact(episode, "EXP_CASH_TRACK_CONTINUOUS", support=(by_time[0.0], by_time[4.0], by_time[5.0], by_time[6.0]))
    tracks = dict(episode.context.tracks)
    tracks["cash"] = tracks["cash"].model_copy(update={"frame_ids": (by_time[0.0], by_time[4.0], by_time[5.0], by_time[6.0])})
    episode = episode.model_copy(update={"context": episode.context.model_copy(update={"tracks": tracks})})
    result = _verify(source, [episode], catalog)
    assert not result.verified_events
    assert result.pending[0].reason == "experimental_evidence_support_exceeds_frame_budget"


def test_private_episode_cli_runs_without_inference_or_default_writes(source, tmp_path):
    catalog = load_experimental_catalog()
    episode = _episode(catalog, RULE_IDS[2], source[1])
    input_path, output_path = tmp_path / "private-pack.json", tmp_path / "private-result.json"
    payload = {"schema_version": "forte.experimental.temporal_input.v1", "asset": source[0].model_dump(mode="json"),
               "frames": [frame.model_dump(mode="json") for frame in source[1].values()],
               "coverage": source[2], "episodes": [episode.model_dump(mode="json")]}
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path("src").resolve()) + os.pathsep + str(Path.cwd())
    completed = subprocess.run([sys.executable, "-m", "forte.events.experimental", "--input", str(input_path),
                                "--output", str(output_path)], check=False, capture_output=True, text=True,
                               env=environment, timeout=10)
    assert completed.returncode == 0
    summary = json.loads(completed.stdout)
    assert summary == {"episodes_checked": 1, "verified_events": 1, "pending": 0, "runtime_enabled": False}
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == "forte.experimental.temporal_result.v1"
    assert result["verified_events"] and "verdict" not in result
    assert "procedure-current" not in completed.stdout and str(input_path) not in completed.stdout
