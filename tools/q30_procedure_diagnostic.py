"""Text-only procedural-head diagnostic from whitelisted observed facts.

No media, scene descriptions, evidence, labels, filenames, verdicts, durations or
per-video identifiers enter model input. Existing sampled visual reports cannot
prove continuous action coverage: their unknown coverage is preserved. This tool
does not enable violations or establish a whole-pipeline accuracy result.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.q30_public_eval import digest, local_endpoint, write_json
from tools.q30_procedure_eval import load_rows, query

COVERAGE = ('transaction_complete', 'observed_fraction', 'critical_regions_visible', 'same_cash_track',
            'same_key_track', 'same_lock_track', 'hand_withdrawal_observed', 'cashier_departure_observed',
            'client_region_continuously_visible', 'responsible_region_continuously_visible',
            'client_cash_link_visible', 'authentication_route_known', 'offered_cash_group_track_visible')


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def event(name, t, object_token='unknown', role='cashier'):
    return {'event': name, 'role': role, 'object_token': object_token, 't_sec': t, 'observed': True}


def coarse_events(states):
    """Presence is not an action completion; false samples never imply omission."""
    found, previous = [], set()
    for state in states:
        if not isinstance(state, dict) or not number(state.get('t')):
            raise ValueError('invalid_structured_state')
        present = set()
        if (state.get('desk') or {}).get('cash') is True: present.add('cash_on_desk')
        if state.get('client_at_desk') is True or state.get('client_seat') is True: present.add('client_present')
        # Counting/device/ID booleans do not establish completed actions. The
        # curriculum has completed-step tokens, so no such token is invented.
        for name in sorted(present - previous):
            found.append(event(name, state['t'], role='client' if name == 'client_present' else 'cashier'))
        previous = present
    return found


def dense_events(report, names):
    if report.get('schema') != 'forte.procedure-visible-protocol-smoke.v2':
        raise ValueError('affirmative_protocol_v2_required')
    rows = report.get('results', [])
    if {row['name'] for row in rows} != names or len(rows) != 40:
        raise ValueError('dense_inventory_mismatch')
    mapped = {}
    # The other local action enums do not prove completion of the corresponding
    # curriculum step; forwarding them as completed checks would invent facts.
    actions = {'cash_accepted': 'cash_accepted', 'cash_stored': 'cash_stored'}
    for row in rows:
        facts = []
        if row.get('status') == 'done':
            for sample in row.get('consensus', []):
                if not number(sample.get('t')): raise ValueError('invalid_dense_time')
                visibility = sample.get('visibility', {})
                if visibility.get('cashier_hands_visibility') != 'clear': continue
                for key, token in actions.items():
                    if sample.get('actions', {}).get(key) != 'yes': continue
                    if key == 'banknote_to_detector' and visibility.get('detector_visibility') != 'clear': continue
                    if key in ('card_to_cashier', 'cash_accepted') and not (
                        visibility.get('desk_visibility') == 'clear'
                        and visibility.get('visitor_hands_visibility') == 'clear'): continue
                    facts.append(event(token, sample['t']))
        mapped[row['name']] = facts
    return mapped


def safe_events(report, names):
    from tools.safe_key_sequence_probe import project_report
    clean = project_report(report, names)  # Recomputes paired resolved sequence support.
    mapped = {}
    for row in clean['results']:
        facts = []
        for window in row['windows'] if row['status'] == 'done' else []:
            if window.get('key_remains_after_departure_support') is True:
                facts.append(event('key_visible_after_departure', window['support_end_sec']))
            for item in window.get('observations', []):
                if item.get('lock_visibility') != 'resolved' or not number(item.get('t')): continue
                if item.get('key_state') == 'empty': facts.append(event('lock_observed_empty', item['t']))
        mapped[row['name']] = facts
    return mapped


def project(args):
    if args.out.exists(): raise ValueError('fresh_projection_output_required')
    if digest(args.policy_manifest) != args.policy_sha256: raise ValueError('frozen_policy_hash_mismatch')
    policy = load_rows(args.policy_manifest)[0]['system']
    from training.qwen30_procedure_lora_train import TYPES
    source = json.loads(args.states.read_text())
    if args.run_name: source = source['runs'][args.run_name]
    rows = source.get('results', source.get('videos'))
    if not isinstance(rows, list) or len(rows) != 40 or len({row['name'] for row in rows}) != 40:
        raise ValueError('exact_40_structured_source_rows_required')
    names = {row['name'] for row in rows}
    dense = dense_events(json.loads(args.dense.read_text()), names) if args.dense else {}
    safe = safe_events(json.loads(args.safe_key.read_text()), names) if args.safe_key else {}
    queries, failures = [], []
    for row in rows:
        failed = row.get('processed') is False or row.get('status') == 'failed' or bool(row.get('error'))
        facts = [] if failed else coarse_events(row.get('states', [])) + dense.get(row['name'], []) + safe.get(row['name'], [])
        facts.sort(key=lambda item: (item['t_sec'], item['event']))
        facts = [item for index, item in enumerate(facts) if index == 0 or item != facts[index - 1]]
        overflow = len(facts) > 64
        if overflow or not facts:
            facts = [{'event': 'transaction_started', 'role': 'cashier', 'object_token': 'unknown',
                      't_sec': 0.0, 'observed': False}]
        if failed or overflow: failures.append({'name': row['name'], 'source_failed': failed, 'projection_overflow': overflow})
        for kind in sorted(TYPES):
            queries.append({'name': row['name'], 'system': policy, 'input': {'requested_type': kind,
                'timeline': facts, 'coverage': dict.fromkeys(COVERAGE)},
                'projection': {'source_failed': failed, 'overflow': overflow,
                    'coarse_states_never_establish_continuous_coverage': True,
                    'sampled_dense_actions_never_establish_complete_transaction': True}})
    report = {'schema': 'forte.q30-procedure-bank-inputs.v1', 'diagnostic_only': True,
        'labels_used': False, 'continuous_visibility_proven': False, 'videos': 40,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': digest(args.states), 'dense_source_sha256': digest(args.dense) if args.dense else None,
        'safe_source_sha256': digest(args.safe_key) if args.safe_key else None,
        'policy_manifest_sha256': args.policy_sha256, 'failures': failures, 'queries': queries}
    write_json(args.out, report)
    print(json.dumps({'status': 'projected', 'videos': 40, 'queries': len(queries), 'coverage': 'unknown',
                      'no_omission_support_claim': True}))
    return 0


def coverage_guard(input_row, prediction):
    """Unknown continuous evidence cannot become a supported omission/condition."""
    if prediction is None: return None
    if any(value is None for value in input_row['coverage'].values()):
        # Every legacy source currently projected here has unproven continuous
        # coverage; no new claim may bypass that fact via a language-head output.
        return {'type_id': input_row['requested_type'], 'decision': 'unassessable', 'anchor_index': -1}
    return prediction


async def infer(args):
    if platform.system() != 'Linux' or not args.rented_host_id or args.out.exists():
        raise ValueError('fresh_named_rented_host_run_required')
    report = json.loads(args.inputs.read_text())
    if report.get('schema') != 'forte.q30-procedure-bank-inputs.v1' or report.get('videos') != 40:
        raise ValueError('strict_projected_bank_inputs_required')
    from training.qwen30_procedure_lora_train import render_messages, TYPES, RESPONSE_SCHEMA
    import httpx
    rows = report['queries']
    if len(rows) != 40 * len(TYPES) or len({row['name'] for row in rows}) != 40:
        raise ValueError('all_40_all_types_required')
    result = {'schema': 'forte.q30-procedure-bank-diagnostic.v1', 'diagnostic_only': True,
        'whole_pipeline_qualification': False, 'bank_video_accuracy_claim': False, 'labels_used': False,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': digest(args.inputs),
        'model': args.model, 'runs': []}
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=local_endpoint(args.base_url)[:-3], trust_env=False,
                                follow_redirects=False, timeout=httpx.Timeout(90, connect=5)) as client:
        for repeat in (1, 2):
            observations = [{'name': row['name'], 'requested_type': row['input']['requested_type'],
                'status': 'failed', 'failure': 'not_processed', 'prediction': None, 'calls': 0,
                'prompt_tokens': 0, 'completion_tokens': 0, 'wall_s': 0.0} for row in rows]
            async def one(index, row):
                answer = await query(client, sem, args.model, row, render_messages, RESPONSE_SCHEMA, TYPES)
                observations[index].update(answer)
                observations[index]['guarded_prediction'] = coverage_guard(row['input'], answer['prediction'])
                observations[index]['unsupported_support_guarded'] = bool(
                    answer['prediction'] and answer['prediction']['decision'] == 'supported')
            jobs = [asyncio.create_task(one(index, row)) for index, row in enumerate(rows)]
            done, pending = await asyncio.wait(jobs, timeout=args.max_wall_seconds)
            for job in pending: job.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for job in done: job.result()
            result['runs'].append({'repeat': repeat, 'videos': 40, 'queries': len(rows),
                'failed_queries': sum(row['status'] != 'done' for row in observations),
                'guarded_support_queries': sum(row.get('unsupported_support_guarded', False) for row in observations),
                'results': observations})
            write_json(args.out, result)
    result['status'] = 'done' if all(run['failed_queries'] == 0 for run in result['runs']) else 'failed'
    write_json(args.out, result)
    print(json.dumps({'status': result['status'], 'runs': 2, 'videos_per_run': 40,
                      'whole_pipeline_qualification': False}))
    return 0 if result['status'] == 'done' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('project')
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--run-name')
    p.add_argument('--dense', type=Path)
    p.add_argument('--safe-key', type=Path)
    p.add_argument('--policy-manifest', type=Path, required=True)
    p.add_argument('--policy-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('infer')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rented-host-id', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--model', default='q30procedure')
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--max-wall-seconds', type=float, default=180)
    args = parser.parse_args()
    try: return project(args) if args.mode == 'project' else asyncio.run(infer(args))
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        return 2


if __name__ == '__main__': raise SystemExit(main())
