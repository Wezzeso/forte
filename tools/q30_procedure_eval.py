"""Paired local text-timeline procedural-head evaluation; no images or bank data.

Every synthetic validation example remains in the denominator. The distinct
template groups establish synthetic training separation, not CCTV competence.
No metric selects a checkpoint or promotes a service configuration.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.q30_public_eval import digest, local_endpoint, unique_pairs, reject_constant, write_json


def load_rows(path):
    return [json.loads(line, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
            for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def strict_prediction(text, requested_type, timeline_length, types):
    result = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    if (not isinstance(result, dict) or set(result) != {'type_id', 'decision', 'anchor_index'}
            or result['type_id'] not in types or result['type_id'] != requested_type
            or result['decision'] not in {'supported', 'not_supported', 'unassessable'}
            or type(result['anchor_index']) is not int):
        raise ValueError('strict_procedure_output_required')
    anchor = result['anchor_index']
    if ((result['decision'] == 'supported' and not 0 <= anchor < timeline_length)
            or (result['decision'] != 'supported' and anchor != -1)):
        raise ValueError('decision_anchor_inconsistent')
    return result


async def query(client, sem, model, row, render_messages, response_schema, types):
    outcome = {'status': 'failed', 'prediction': None, 'failure': None, 'calls': 0,
               'prompt_tokens': 0, 'completion_tokens': 0, 'wall_s': 0.0}
    tick = time.monotonic()
    try:
        messages = render_messages({'system': row['system'], 'input': row['input']}, include_target=False)
        body = {'model': model, 'messages': messages, 'temperature': 0, 'seed': 0,
                'max_tokens': 96, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False},
                'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': 'procedure', 'strict': True, 'schema': response_schema}}}
        async with sem:
            outcome['calls'] = 1
            response = await client.post('/v1/chat/completions', json=body)
        if response.status_code != 200:
            outcome['failure'] = 'http_' + str(response.status_code)
            return outcome
        raw = response.json()
        if raw.get('model') != model:
            raise ValueError('served_alias_mismatch')
        usage = raw.get('usage') or {}
        outcome['prompt_tokens'] = max(0, int(usage.get('prompt_tokens') or 0))
        outcome['completion_tokens'] = max(0, int(usage.get('completion_tokens') or 0))
        choices = raw['choices']
        if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
            raise ValueError('incomplete_completion')
        message = choices[0]['message']
        if any(message.get(key) for key in ('reasoning', 'reasoning_content', 'refusal', 'tool_calls', 'function_call')):
            raise ValueError('unexpected_nonanswer')
        outcome['prediction'] = strict_prediction(message['content'], row['input']['requested_type'],
                                                  len(row['input']['timeline']), types)
        outcome['status'] = 'done'
    except asyncio.CancelledError:
        outcome['failure'] = 'evaluation_deadline'
    except (ValueError, TypeError, KeyError, IndexError):
        outcome['failure'] = 'invalid_response_or_closed_schema'
    except Exception as error:
        outcome['failure'] = 'technical:' + type(error).__name__
    finally:
        outcome['wall_s'] = round(time.monotonic() - tick, 4)
    return outcome


def metrics(rows):
    completed = [row for row in rows if row['status'] == 'done']
    confusion = {truth: {pred: 0 for pred in ('supported', 'not_supported', 'unassessable', 'failed')}
                 for truth in ('supported', 'not_supported', 'unassessable')}
    for row in rows:
        confusion[row['target']['decision']][row['prediction']['decision'] if row['status'] == 'done' else 'failed'] += 1
    exact = sum(row['status'] == 'done' and row['prediction'] == row['target'] for row in rows)
    return {'assessment': 'evaluated' if rows else 'not_evaluated',
            'examples': len(rows), 'processed': len(completed), 'failed': len(rows) - len(completed),
            'exact_correct': exact, 'exact_accuracy': exact / len(rows) if rows else None,
            'decision_correct': sum(row['prediction']['decision'] == row['target']['decision'] for row in completed),
            'strict_schema_rate': len(completed) / len(rows) if rows else None,
            'decision_confusion': confusion,
            'unsupported_support': sum(row['prediction']['decision'] == 'supported' and
                                       row['target']['decision'] != 'supported' for row in completed),
            'anchor_correct_supported': sum(row['target']['decision'] == 'supported' and
                                            row['prediction'] == row['target'] for row in completed),
            'expected_supported': sum(row['target']['decision'] == 'supported' for row in rows),
            'calls': sum(row['calls'] for row in rows),
            'prompt_tokens': sum(row['prompt_tokens'] for row in rows),
            'completion_tokens': sum(row['completion_tokens'] for row in rows),
            'failures': dict(Counter(row['failure'] for row in rows if row['failure']))}


async def run(args):
    if platform.system() != 'Linux' or not args.rented_host_id or args.out.exists():
        raise ValueError('fresh_named_rented_host_run_required')
    endpoint = local_endpoint(args.base_url)
    if digest(args.train) != args.train_sha256 or digest(args.validation) != args.validation_sha256:
        raise ValueError('frozen_synthetic_manifest_mismatch')
    from training.qwen30_procedure_lora_train import render_messages, TYPES, RESPONSE_SCHEMA, validate_rows
    train, validation = load_rows(args.train), load_rows(args.validation)
    shared_audit = validate_rows(train + validation)
    if len(train) != 512 or len(validation) != 128:
        raise ValueError('fixed_synthetic_inventory_required')
    groups = [{row['template_group'] for row in rows} for rows in (train, validation)]
    if len(groups[0]) != 32 or len(groups[1]) != 8 or groups[0] & groups[1]:
        raise ValueError('synthetic_template_group_leakage')
    for split, rows in (('train', train), ('validation', validation)):
        for row in rows:
            if row.get('origin') != 'synthetic' or row['split'] != split:
                raise ValueError('synthetic_origin_and_split_required')
            strict_prediction(json.dumps(row['target']), row['input']['requested_type'],
                              len(row['input']['timeline']), TYPES)
    if len({row['system'] for row in train + validation}) != 1:
        raise ValueError('fixed_procedure_system_required')
    import httpx
    started = time.monotonic()
    aliases = {'base': args.base_model, 'adapter': args.adapter_model}
    if args.smoke:
        # A fixed training fixture is a compatibility smoke, never heldout evaluation.
        row = train[0]
        sem = asyncio.Semaphore(1)
        async with httpx.AsyncClient(base_url=endpoint[:-3], trust_env=False, follow_redirects=False,
                                    timeout=httpx.Timeout(90, connect=5)) as client:
            outcomes = {phase: await query(client, sem, model, row, render_messages, RESPONSE_SCHEMA, TYPES)
                        for phase, model in aliases.items()}
        smoke = {'schema': 'forte.q30-procedure-synthetic-smoke.v1',
                 'fixture_split': 'train', 'fixture_index': 0, 'synthetic_only': True,
                 'actual_adapter_inference_requested': True, 'aliases': aliases,
                 'expected': row['target'], 'results': outcomes,
                 'passed': all(value['status'] == 'done' for value in outcomes.values()),
                 'correctness_not_a_promotion_gate': True, 'bank_video_accuracy_claim': False,
                 'train_sha256': args.train_sha256, 'validation_sha256': args.validation_sha256,
                 'generated_at_utc': datetime.now(timezone.utc).isoformat()}
        write_json(args.out, smoke)
        print(json.dumps({'passed': smoke['passed'], 'actual_adapter_inference_requested': True}))
        return 0 if smoke['passed'] else 2
    observations = {phase: [{'example_index': index, 'template_group': row['template_group'],
        'target': row['target'], 'status': 'failed', 'prediction': None, 'failure': 'not_processed',
        'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'wall_s': 0.0}
        for index, row in enumerate(validation)] for phase in aliases}
    report = {'schema': 'forte.q30-procedure-synthetic-paired.v1', 'status': 'processing',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'synthetic_template_group_heldout_procedural_logic_only',
        'synthetic_train_validation_groups_disjoint': True, 'bank_video_accuracy_claim': False,
        'perception_improvement_claim': False, 'checkpoint_selection_performed': False,
        'provenance': {'train_sha256': args.train_sha256, 'validation_sha256': args.validation_sha256,
            'train_groups': 32, 'validation_groups': 8, 'train_examples': 512, 'validation_examples': 128,
            'code_sha256': digest(Path(__file__)), 'rented_host_id': args.rented_host_id,
            'aliases': aliases, 'endpoint': endpoint, 'shared_typed_group_audit': shared_audit}, 'runs': []}
    write_json(args.out, report)
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=endpoint[:-3], trust_env=False, follow_redirects=False,
                                timeout=httpx.Timeout(90, connect=5)) as client:
        async def one(phase, model, index, row):
            result = await query(client, sem, model, row, render_messages, RESPONSE_SCHEMA, TYPES)
            observations[phase][index].update(result)
        jobs = [asyncio.create_task(one(phase, model, index, row))
                for index, row in enumerate(validation) for phase, model in aliases.items()]
        done, pending = await asyncio.wait(jobs, timeout=args.max_wall_seconds)
        for job in pending: job.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for job in done: job.result()
    for phase in aliases:
        rows = observations[phase]
        report['runs'].append({'phase': phase, 'model': aliases[phase], 'metrics': metrics(rows), 'results': rows,
            'per_type': {name: metrics([row for row in rows if row['target']['type_id'] == name]) for name in TYPES}})
    pair = Counter()
    for old, new in zip(observations['base'], observations['adapter']):
        pair[(old['status'] == 'done' and old['prediction'] == old['target'],
              new['status'] == 'done' and new['prediction'] == new['target'])] += 1
    report['paired'] = {'both_correct': pair[(True, True)], 'adapter_only_correct': pair[(False, True)],
                        'base_only_correct': pair[(True, False)], 'both_wrong_or_failed': pair[(False, False)]}
    report['wall_s'] = round(time.monotonic() - started, 4)
    report['status'] = 'done' if all(run['metrics']['failed'] == 0 for run in report['runs']) else 'failed'
    write_json(args.out, report)
    print(json.dumps({'status': report['status'], 'paired': report['paired'], 'wall_s': report['wall_s']}))
    return 0 if report['status'] == 'done' else 2


def self_test():
    types = ('NO_CLIENT_ID',)
    target = {'type_id': types[0], 'decision': 'unassessable', 'anchor_index': -1}
    assert strict_prediction(json.dumps(target), types[0], 0, types) == target
    for decision, anchor in (('supported', -1), ('not_supported', 0)):
        try: strict_prediction(json.dumps({**target, 'decision': decision, 'anchor_index': anchor}), types[0], 1, types)
        except ValueError: pass
        else: raise AssertionError('anchor_fixture_failed')
    failed = {'target': target, 'status': 'failed', 'prediction': None, 'calls': 0,
              'prompt_tokens': 0, 'completion_tokens': 0, 'failure': 'technical'}
    passed = {**failed, 'status': 'done', 'prediction': target, 'failure': None}
    assert metrics([failed, passed])['exact_accuracy'] == .5
    print(json.dumps({'status': 'passed', 'synthetic_only': True,
                      'checks': ['decision_anchor_schema', 'failure_denominator']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Fixed train row0 compatibility, no heldout score')
    for name in ('train', 'validation', 'out'): parser.add_argument('--' + name, type=Path)
    for name in ('train-sha256', 'validation-sha256', 'rented-host-id'): parser.add_argument('--' + name)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--base-model', default='q30test')
    parser.add_argument('--adapter-model', default='q30procedure')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--max-wall-seconds', type=float, default=150)
    args = parser.parse_args()
    if args.self_test: self_test(); return 0
    if not all((args.train, args.validation, args.out, args.train_sha256, args.validation_sha256, args.rented_host_id)):
        parser.error('frozen train/validation paths and SHA256 values, output and rented-host-id required')
    try: return asyncio.run(run(args))
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        return 2


if __name__ == '__main__': raise SystemExit(main())
