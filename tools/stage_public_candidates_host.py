"""CPU-only copy into one fresh, disabled candidate release; never activates it.

Invoke on the host with --inputs /absolute/staging-inputs.json and its exact
--expected-inputs-sha256. --print-input-template is read-only and shows all pins
the operator must supply. --check-only verifies all files without creating a
release. No model/runtime imports, subprocesses, service calls, or network code.
Failures retain a partial release and never remove or overwrite existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path('/data/claude/training/public-20261018')
PUBLIC_RUNTIME = Path('/data/forte-ml/public')
RELEASE = Path('/home/ubuntu/forte/deploy/releases/public-specialists-20261018-0ab06fd0')
COHORT_SHA = '0ab06fd0e9c17c259b8464fc43d7466f8b55cc2e2bfd8182df5ab73942a0a437'
COHORT_PATH = ROOT / 'test-exposure-registry/public-sealed-6fd01ecf63dceb738a069626bdebf59e286565e73c0c38b67a9715097b474a29.json'
IMAGES = {'detector': 'sha256:2a7af0c822817bda42d8eabba5085b1756ffd051d37f2c3aa3ec138a166462ba',
          'crop_vlm': 'sha256:2be4db9629837c6a3c0e252d2a4b45b54bd0a0d25d41e04274dbff0dd2755712'}
DESCRIPTORS = {
    'detector': {'path': str(ROOT / 'manifests/detector-descriptor-v1.json'),
                 'sha256': '0e923a37a10344ac00e0686b2f5d4ca8e1fc40e21b9fd5db171228b7191e8e5a'},
    'crop_vlm': {'path': str(ROOT / 'manifests/crop-descriptor-v1.json'),
                 'sha256': '6ea88b19f84082848091bffa9c1485b55763d854881c39aa69f92f470246fa11'},
}
SHA = re.compile(r'[a-f0-9]{64}\Z')
MEMBERS = {'detector', 'crop_vlm'}
NATIVE_ADAPTER = {'README.md', 'adapter_config.json', 'adapter_model.safetensors'}
RESERVE = 4 * 1024**3


class StageError(ValueError):
    pass


def require(ok, code):
    if not ok:
        raise StageError(code)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value, *, file=False):
    require(isinstance(value, str) and value.startswith('/') and not any(c in value for c in ('\0', '\n', '\r')),
            'absolute_local_path_required')
    path = Path(value)
    require(path == path.resolve() and not path.is_symlink(), 'canonical_nonsymlink_path_required')
    if file:
        require(path.is_file(), 'required_file_missing')
    return path


def pin(value, bindings):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'exact_file_pin_required')
    path = canonical(value['path'], file=True)
    require(path.is_relative_to(ROOT) or path.is_relative_to(PUBLIC_RUNTIME), 'input_outside_public_training_scope')
    expected = value['sha256']
    require(isinstance(expected, str) and SHA.fullmatch(expected) and sha(path) == expected, 'file_hash_mismatch')
    if str(path) in bindings:
        require(bindings[str(path)] == expected, 'conflicting_file_pins')
    bindings[str(path)] = expected
    return path


def read_json(path):
    require(path.stat().st_size <= 16 * 1024**2, 'json_receipt_oversize')
    def invalid(_):
        raise StageError('nonfinite_json_receipt')
    try:
        value = json.loads(path.read_text(), parse_constant=invalid)
    except (ValueError, UnicodeError):
        raise StageError('json_receipt_invalid') from None
    require(isinstance(value, dict), 'json_object_required')
    return value


def template():
    result = {'schema_version': 'forte.public-candidate-stage-inputs.v1',
              'cohort': {'path': COHORT_PATH.as_posix(), 'sha256': COHORT_SHA}, 'members': {}}
    for member, name in [('detector', 'rf-small512-v1'), ('crop_vlm', 'crop2b-fit-v1')]:
        run = ROOT / 'runs' / name
        output = run / 'output'
        final = (output / 'cohort-detector-evaluation/result.json' if member == 'detector'
                 else output / 'sealed-test-results.json')
        local_attempt = (output / 'cohort-detector-evaluation/attempt.json' if member == 'detector'
                         else output / 'sealed-test-access.json')
        attempt = COHORT_PATH.with_name(COHORT_PATH.stem + f'.member-{member}.attempt.json')
        result['members'][member] = {
            'training_host': {'path': (run / 'host-report.json').as_posix(), 'sha256': 'FILL_SHA256'},
            'final_host': {'path': (ROOT / 'reports' / ('rf-small512-final-v2-host.json' if member == 'detector'
                                                          else 'crop2b-final-v1-host.json')).as_posix(), 'sha256': 'FILL_SHA256'},
            'final_result': {'path': final.as_posix(), 'sha256': 'FILL_SHA256'},
            'cohort_attempt': {'path': attempt.as_posix(), 'sha256': 'FILL_SHA256'},
            'evaluation_attempt': {'path': local_attempt.as_posix(), 'sha256': 'FILL_SHA256'},
        }
    result['optional_receipts'] = []  # Each item: {name: safe-name, path: absolute, sha256: exact}.
    return result


def prepare(inputs_path, expected_inputs_sha):
    require(sys.platform == 'linux', 'linux_host_required')
    bindings = {}
    inputs_path = pin({'path': str(inputs_path), 'sha256': expected_inputs_sha}, bindings)
    inputs = read_json(inputs_path)
    require(set(inputs) <= {'schema_version', 'cohort', 'members', 'optional_receipts'}
            and inputs.get('schema_version') == 'forte.public-candidate-stage-inputs.v1'
            and isinstance(inputs.get('members'), dict) and set(inputs['members']) == MEMBERS,
            'staging_inputs_schema_invalid')
    require(inputs.get('cohort') == {'path': str(COHORT_PATH), 'sha256': COHORT_SHA}, 'exact_registered_cohort_required')
    cohort = read_json(pin(inputs['cohort'], bindings))
    require(cohort.get('schema_version') == 'forte.public-eval-cohort.v1' and cohort.get('state') == 'registered'
            and set(cohort.get('members', {})) == MEMBERS, 'registered_cohort_required')
    require(cohort.get('member_ids') == ['crop_vlm', 'detector']
            and cohort.get('registry_dir') == str(COHORT_PATH.parent), 'cohort_scope_mismatch')
    copies = {}

    def add(relative, value):
        path = pin(value, bindings)
        require(relative not in copies and not Path(relative).is_absolute()
                and '..' not in Path(relative).parts, 'release_relative_path_invalid')
        copies[relative] = {'path': str(path), 'sha256': value['sha256'], 'bytes': path.stat().st_size}
        return path

    add('receipts/staging-inputs.json', {'path': str(inputs_path), 'sha256': expected_inputs_sha})
    add('cohort/registered-cohort.json', inputs['cohort'])
    add('cohort/root-authorization.json', cohort['authorization'])
    pin(cohort['dataset_lock'], bindings)  # Bind provenance, never copy dataset/images.
    model_summary = {}
    for member in sorted(MEMBERS):
        declared = cohort['members'][member]
        descriptor = read_json(add(member + '/descriptor.json', DESCRIPTORS[member]))
        require(descriptor == declared, 'descriptor_cohort_mismatch')
        for field in ('run', 'checkpoint', 'config', 'protocol'):
            pin(declared[field], bindings)
        artifacts = declared.get('artifacts')
        require(isinstance(artifacts, dict), 'cohort_artifacts_required')
        for value in artifacts.values():
            pin(value, bindings)  # Includes frozen base snapshot; does not copy it.
        run = read_json(Path(declared['run']['path']))
        config = read_json(Path(declared['config']['path']))
        protocol = read_json(Path(declared['protocol']['path']))
        output = Path(declared['run']['path']).parent
        require(config.get('output_dir') == str(output)
                and run.get('config_sha256') == declared['config']['sha256']
                and run.get('evaluation_protocol_sha256') == declared['protocol']['sha256']
                and run.get('dataset_lock_sha256') == cohort['dataset_lock']['sha256'], 'frozen_run_binding_mismatch')
        require(type(run.get('optimizer_steps')) is int and run['optimizer_steps'] > 0
                and run.get('approved_types', []) == [] and not run.get('promotion_eligible', False)
                and not run.get('serving_eligible', False), 'real_unpromoted_fit_required')
        require(protocol.get('selection_scope') == 'public_train_dev_only', 'frozen_protocol_scope_mismatch')
        if member == 'detector':
            require(run.get('schema_version') == 'forte.specialists.public-run.v1'
                    and run.get('state') == 'trained_unselected_pending_test'
                    and run.get('checkpoint_sha256') == declared['checkpoint']['sha256'], 'detector_run_invalid')
            add(member + '/checkpoint_best_total.pth', declared['checkpoint'])
            launch_config_pin = {'path': str(ROOT / 'configs/rf-small512-v1.json'),
                                 'sha256': '4287852e1bf43d1c9865e2fb86416602436d3926dd6aaf9c93383865cb67e45e'}
            launch_config = read_json(add(member + '/launch-config.json', launch_config_pin))
            require(launch_config == config, 'rf_launch_and_saved_config_objects_differ')
            expected_result = output / 'cohort-detector-evaluation/result.json'
            expected_local_attempt = output / 'cohort-detector-evaluation/attempt.json'
        else:
            require(run.get('schema_version') == 'forte.public-crop-lora.run.v1'
                    and run.get('state') == 'trained_unselected' and run.get('training_completed') is True
                    and not run.get('memory_smoke_only'), 'crop_run_invalid')
            adapter = run.get('adapter_files')
            require(isinstance(adapter, dict) and set(adapter) == NATIVE_ADAPTER, 'exact_native_three_adapter_files_required')
            require({path.name for path in (output / 'adapter').iterdir()} == NATIVE_ADAPTER,
                    'native_adapter_inventory_mismatch')
            for name in sorted(NATIVE_ADAPTER):
                value = {'path': str(output / 'adapter' / name), 'sha256': adapter[name]}
                require(artifacts.get('adapter_' + name) == value, 'native_adapter_cohort_pin_mismatch')
                add(member + '/adapter/' + name, value)
            require(adapter['adapter_model.safetensors'] == declared['checkpoint']['sha256'], 'selected_adapter_pin_mismatch')
            expected_result = output / 'sealed-test-results.json'
            expected_local_attempt = output / 'sealed-test-access.json'
            add(member + '/base-model-manifest.json', artifacts['model_manifest'])
        for field, name in [('run', 'run.json'), ('config', 'config.json'), ('protocol', 'evaluation-protocol.json')]:
            add(member + '/' + name, declared[field])
        for field, name in [('source_inventory', 'source.manifest.json'), ('source_archive', 'source.tar.gz'),
                            ('dependency_freeze', 'pip-freeze.txt'), ('dependency_resolver', 'pip-resolver.json'),
                            ('runtime_versions', 'runtime-versions.json'), ('training_guard', 'training-host.py'),
                            ('evaluation_guard', 'evaluation-host.py'), ('evaluation_source', 'evaluation-source.py')]:
            add(member + '/' + name, artifacts[field])
        supplied = inputs['members'][member]
        require(set(supplied) == {'training_host', 'final_host', 'final_result', 'cohort_attempt', 'evaluation_attempt'},
                'member_receipts_required')
        receipts = {name: read_json(add(member + '/receipts/' + name.replace('_', '-') + '.json', value))
                    for name, value in supplied.items()}
        require(supplied['final_result']['path'] == str(expected_result)
                and supplied['evaluation_attempt']['path'] == str(expected_local_attempt), 'final_receipt_path_mismatch')
        expected_attempt = COHORT_PATH.with_name(COHORT_PATH.stem + f'.member-{member}.attempt.json')
        require(supplied['cohort_attempt']['path'] == str(expected_attempt), 'cohort_attempt_path_mismatch')
        attempt = receipts['cohort_attempt']
        require(attempt.get('schema_version') == 'forte.public-eval-cohort-attempt.v1'
                and attempt.get('member_id') == member and attempt.get('cohort_sha256') == COHORT_SHA
                and attempt.get('failure_consumes_attempt') is True
                and all(attempt.get(key) == declared[field]['sha256'] for key, field in
                        [('run_sha256', 'run'), ('checkpoint_sha256', 'checkpoint'), ('config_sha256', 'config'),
                         ('protocol_sha256', 'protocol')]), 'retained_cohort_attempt_mismatch')
        result = receipts['final_result']
        require(result.get('run_sha256') == declared['run']['sha256'], 'final_result_run_mismatch')
        if member == 'detector':
            require(result.get('schema_version') == 'forte.specialists.public-final-evaluation.v1'
                    and result.get('state') == 'complete' and result.get('member_id') == 'detector'
                    and result.get('test_evaluation_count') == 1 and result.get('cohort_receipt_sha256') == COHORT_SHA
                    and result.get('checkpoint_sha256') == declared['checkpoint']['sha256'], 'detector_final_not_complete')
            require(receipts['evaluation_attempt'].get('state') == 'complete'
                    and receipts['evaluation_attempt'].get('test_evaluation_count') == 1
                    and receipts['evaluation_attempt'].get('result_sha256') == supplied['final_result']['sha256'],
                    'detector_completion_attempt_mismatch')
        else:
            require(result.get('schema_version') == 'forte.public-crop-lora.sealed-evaluation.v1'
                    and result.get('promotion_eligible') is False and result.get('test_reuse_for_tuning_allowed') is False
                    and result.get('shared_test_exposure_receipt') == str(expected_attempt)
                    and isinstance(result.get('adapter'), dict), 'crop_final_not_complete')
            require(receipts['evaluation_attempt'].get('run_sha256') == declared['run']['sha256']
                    and receipts['evaluation_attempt'].get('state') == 'access_started', 'crop_access_marker_mismatch')
        final_host, training_host = receipts['final_host'], receipts['training_host']
        require(final_host.get('schema_version') == 'forte.public-evaluation-host.v1'
                and final_host.get('phase') == ('rf-final' if member == 'detector' else 'crop-final')
                and final_host.get('passed') is True and final_host.get('evaluation_exit_code') == 0
                and final_host.get('production_unchanged') is True and final_host.get('immutable_inputs_unchanged') is True
                and final_host.get('owned_containers_retained_stopped') is True
                and final_host.get('result', {}).get('result_sha256') == supplied['final_result']['sha256'],
                'successful_guarded_final_required')
        require(training_host.get('schema_version') == 'forte.public-training-host.v1'
                and training_host.get('phase') == ('rf-fit' if member == 'detector' else 'crop-fit')
                and training_host.get('training_exit_code') == 0
                and training_host.get('final_progress', {}).get('optimizer_steps', 0) > 0,
                'native_training_success_required')
        require(training_host.get('image_id') == IMAGES[member] and final_host.get('image_id') == IMAGES[member]
                and final_host.get('config_sha256') == declared['config']['sha256'], 'runtime_image_or_config_mismatch')
        training_config_sha = launch_config_pin['sha256'] if member == 'detector' else declared['config']['sha256']
        require(training_host.get('config_sha256') == training_config_sha, 'training_config_identity_mismatch')
        if member == 'detector':
            observed = result.get('fixed_detection_metrics', {}).get('per_class', {}).get('id_card', {}).get('recall')
            metric_name = 'fixed_confidence_id_card_recall'
        else:
            observed = result['adapter'].get('macro_accuracy')
            metric_name = 'class_and_resolution_macro_accuracy'
        require(type(observed) in (float, int) and math.isfinite(observed) and 0 <= observed <= 1,
                'primary_quality_metric_missing_or_invalid')
        model_summary[member] = {'training_image_id': IMAGES[member], 'run_sha256': declared['run']['sha256'],
            'launch_config_sha256': training_config_sha, 'saved_config_sha256': declared['config']['sha256'],
            'launch_and_saved_config_objects_equal': True,
            'checkpoint_sha256': declared['checkpoint']['sha256'], 'protocol_sha256': declared['protocol']['sha256'],
            'source_archive_sha256': artifacts['source_archive']['sha256'],
            'source_manifest_sha256': artifacts['source_inventory']['sha256'],
            'final_result_sha256': supplied['final_result']['sha256'], 'final_execution_complete': True,
            'training_host_passed': training_host.get('passed'), 'training_host_failure': training_host.get('failure'),
            'training_host_postcheck_failures': training_host.get('postcheck_failures', []),
            'quality_gate': {'metric': metric_name, 'value': observed, 'target': 0.95,
                             'passed': observed >= 0.95, 'status': 'failed' if observed < 0.95 else 'passed_unapproved',
                             'reason': 'primary_metric_below_predeclared_target' if observed < 0.95 else 'activation_unapproved'},
            'quality_gate_approval': False, 'activation': False, 'approved_types': [],
            'serving_eligible': False, 'promotion_eligible': False}
    optional = inputs.get('optional_receipts', [])
    require(isinstance(optional, list) and len(optional) <= 16, 'optional_receipts_invalid')
    for row in optional:
        require(isinstance(row, dict) and set(row) == {'name', 'path', 'sha256'}
                and isinstance(row['name'], str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}', row['name']),
                'optional_receipt_name_invalid')
        read_json(add('receipts/optional-' + row['name'] + '.json', {'path': row['path'], 'sha256': row['sha256']}))
    require(RELEASE.is_absolute() and RELEASE == RELEASE.resolve() and not RELEASE.is_symlink()
            and not RELEASE.exists() and RELEASE.parent.is_dir(), 'fresh_exact_release_directory_required')
    return {'copies': copies, 'bindings': bindings, 'models': model_summary,
            'inputs_sha256': expected_inputs_sha, 'cohort_sha256': COHORT_SHA}


def copy_verified(target, entry):
    source_path = canonical(entry['path'], file=True)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    value = hashlib.sha256()
    with source_path.open('rb') as source:
        with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as destination:
            for block in iter(lambda: source.read(4 * 1024**2), b''):
                destination.write(block)
                value.update(block)
            destination.flush()
            os.fsync(destination.fileno())
    require(value.hexdigest() == entry['sha256'] and target.stat().st_size == entry['bytes']
            and sha(target) == entry['sha256'], 'copied_payload_hash_mismatch')
    os.chmod(target, 0o444)


def stage(plan):
    total = sum(row['bytes'] for row in plan['copies'].values())
    require(shutil.disk_usage(RELEASE.parent).free - total - 16 * 1024**2 >= RESERVE, 'release_disk_reserve_insufficient')
    RELEASE.mkdir(mode=0o700)
    for relative, entry in sorted(plan['copies'].items()):
        copy_verified(RELEASE / relative, entry)
    # Recheck immutable metadata/selected files after copying; do not rewrite sources.
    for path, expected in plan['bindings'].items():
        require(sha(Path(path)) == expected, 'source_changed_during_staging')
    manifest = {'schema_version': 'forte.public-specialist-candidate-release.v1', 'state': 'staged_disabled',
        'activation': False, 'approved_types': [], 'serving_eligible': False, 'promotion_eligible': False,
        'quality_gate_approval': False, 'cohort_sha256': COHORT_SHA, 'models': plan['models'],
        'inputs_sha256': plan['inputs_sha256'], 'files': {name: {'sha256': row['sha256'], 'bytes': row['bytes']}
                                                     for name, row in sorted(plan['copies'].items())},
        'original_bindings': plan['bindings'], 'release_dir': str(RELEASE),
        'base_snapshot_copied': False, 'dataset_or_images_copied': False,
        'gpu_or_test_inference_performed': False, 'live_configuration_changed': False,
        'remote_originals_retained': True, 'retained_training_guard_failures': True}
    target = RELEASE / 'release-manifest.json'
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as output:
        output.write(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n')
        output.flush()
        os.fsync(output.fileno())
    os.chmod(target, 0o444)
    for directory in sorted((path for path in RELEASE.rglob('*') if path.is_dir()), key=lambda p: len(p.parts), reverse=True):
        os.chmod(directory, 0o555)
    os.chmod(RELEASE, 0o555)
    return {'state': 'staged_disabled', 'activation': False, 'approved_types': [], 'serving_eligible': False,
            'promotion_eligible': False, 'release_dir': str(RELEASE), 'files': len(plan['copies']),
            'copied_bytes': total, 'manifest_sha256': sha(target), 'gpu_or_test_inference_performed': False,
            'live_configuration_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--expected-inputs-sha256')
    parser.add_argument('--print-input-template', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.print_input_template:
            result = template()
        else:
            require(args.inputs is not None and args.expected_inputs_sha256 is not None, 'pinned_staging_inputs_required')
            plan = prepare(args.inputs, args.expected_inputs_sha256)
            result = ({'verified': True, 'copy_performed': False, 'planned_files': len(plan['copies']),
                       'planned_bytes': sum(row['bytes'] for row in plan['copies'].values()),
                       'activation': False, 'release_dir': str(RELEASE)} if args.check_only else stage(plan))
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except StageError as error:
        print(json.dumps({'state': 'staging_failed', 'activation': False, 'failure': str(error)}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
