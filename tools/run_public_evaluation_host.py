"""Separate, guarded development/final evaluation of immutable public runs.

The fitting launcher is imported only after its externally supplied SHA256 pin
is verified. It is never edited. Source inventory and archive, run, selected
weights, config, protocol, image, and (for finals) both cohort members are pinned.
CPU preflight cannot claim test access. Only the GPU phase invokes final APIs.
Failed final attempts remain consumed; this tool never removes their markers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/data/claude/training/public-20261018')
DEADLINE = datetime(2026, 10, 18, 5, 45, tzinfo=timezone.utc)
SHA = re.compile(r'[a-f0-9]{64}\Z')
PHASES = ('rf-dev', 'rf-final', 'crop-final')


class EvaluationError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise EvaluationError(code)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def load_common(path, expected):
    require(sys.platform == 'linux', 'linux_host_required')
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and path.is_file() and not path.is_symlink(),
            'canonical_common_launcher_required')
    raw = path.read_bytes()
    require(isinstance(expected, str) and SHA.fullmatch(expected) and hashlib.sha256(raw).hexdigest() == expected,
            'common_launcher_pin_mismatch')
    spec = importlib.util.spec_from_file_location('_forte_pinned_training_host_guard', path)
    require(spec is not None and spec.loader is not None, 'common_launcher_unloadable')
    module = importlib.util.module_from_spec(spec)
    # Execute the exact verified bytes, never an unpinned cached .pyc; importing
    # the common tool also must not write __pycache__ into its frozen bin folder.
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require(module.HARD_STOP == DEADLINE and module.LOCK_PATH == ROOT / 'gpu-training.lock'
            and module.MIN_START_FREE_MIB == 6656 and module.MIN_FREE_MIB == 1536
            and module.MAX_JOB_MIB == 5632, 'common_guard_contract_changed')
    return module


def pin_file(common, raw_path, expected, pins):
    path = common.absolute_path(str(raw_path), kind='file')
    require(isinstance(expected, str) and SHA.fullmatch(expected) and digest(path) == expected,
            'evaluation_artifact_pin_mismatch')
    if str(path) in pins:
        require(pins[str(path)] == expected, 'evaluation_artifact_conflicting_pins')
    pins[str(path)] = expected
    return path


def authorized_write(path):
    require(path != ROOT and path.is_relative_to(ROOT), 'evaluation_writable_scope_forbidden')


def verify_mount_isolation(phase, readonly, writable, run_parent):
    """Only the frozen crop adapter may be RO below its writable final run."""
    for write in writable:
        for read in readonly:
            if read.is_dir():
                frozen_adapter_submount = (phase == 'crop-final' and read == run_parent / 'adapter'
                                           and write == run_parent)
                require(not write.is_relative_to(read)
                        and (not read.is_relative_to(write) or frozen_adapter_submount),
                        'evaluation_writable_input_directory_overlap')
            elif phase == 'rf-dev':
                require(not read.is_relative_to(write), 'development_writable_input_file_overlap')


def verify_cohort_pins(common, cohort, checked, args):
    """Expose all registered peer pins at identical paths, with no write access."""
    require(cohort.get('schema_version') == 'forte.public-eval-cohort.v1'
            and cohort.get('state') == 'registered'
            and cohort.get('member_ids') == ['crop_vlm', 'detector'], 'cohort_receipt_invalid')
    registry = common.absolute_path(cohort.get('registry_dir'), kind='dir')
    require(registry == checked['registry'] and checked['cohort_path'].parent == registry,
            'cohort_registry_mismatch')
    members = cohort.get('members')
    require(isinstance(members, dict) and set(members) == {'detector', 'crop_vlm'}, 'cohort_members_invalid')
    required = []

    def pin(value):
        require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'cohort_artifact_pin_invalid')
        path = pin_file(common, value['path'], value['sha256'], checked['pins'])
        require(path.is_relative_to(ROOT) or path.is_relative_to(Path('/data/forte-ml/public')),
                'cohort_artifact_scope_forbidden')
        required.append(path)
        return path

    dataset_lock = pin(cohort.get('dataset_lock'))
    require(dataset_lock == checked['dataset'] / 'dataset-lock.json'
            and cohort['dataset_lock']['sha256'] == checked['config']['expected_dataset_lock_sha256'],
            'cohort_dataset_namespace_mismatch')
    pin(cohort.get('authorization'))
    for member_id, member in members.items():
        require(isinstance(member, dict) and set(member) == {'run', 'checkpoint', 'config', 'protocol', 'artifacts'},
                'cohort_member_fields_invalid')
        for field in ('run', 'checkpoint', 'config', 'protocol'):
            pin(member[field])
        require(isinstance(member['artifacts'], dict) and 'evaluation_source' in member['artifacts'],
                'cohort_member_sources_missing')
        for value in member['artifacts'].values():
            pin(value)
    member_id = 'detector' if args.phase == 'rf-final' else 'crop_vlm'
    selected = members[member_id]
    for field, path, expected in (
        ('run', checked['run_path'], args.expected_run_sha256),
        ('config', checked['run_config'], args.expected_config_sha256),
        ('protocol', checked['protocol'], args.expected_protocol_sha256),
        ('checkpoint', checked['checkpoint'], checked['checkpoint_sha256']),
    ):
        require(selected[field] == {'path': str(path), 'sha256': expected}, 'cohort_selected_member_mismatch')
    attempt = checked['cohort_path'].with_name(checked['cohort_path'].stem + f'.member-{member_id}.attempt.json')
    require(not attempt.exists() and not attempt.is_symlink(), 'cohort_member_attempt_already_exists')
    checked['attempt_marker'] = attempt
    checked['readonly'].extend(required)


def validate_inputs(args, common, *, now=None):
    require(args.phase in PHASES, 'evaluation_phase_invalid')
    require(re.fullmatch(r'sha256:[a-f0-9]{64}', args.image) is not None, 'immutable_image_required')
    require(common.NAME.fullmatch(args.name) is not None, 'isolated_container_name_required')
    require(1 <= args.max_wall_seconds <= 3600, 'evaluation_wall_limit_invalid')
    stop = common.utc(args.stop_at_utc)
    require(stop <= DEADLINE and stop > (now or datetime.now(timezone.utc)), 'evaluation_deadline_invalid')
    source = common.absolute_path(str(args.source_dir), kind='dir')
    manifest = common.absolute_path(str(args.source_manifest), kind='file')
    source_receipt = common.verify_sources(source, manifest, args.expected_source_manifest_sha256)
    pins = {}
    archive = pin_file(common, args.source_archive, args.expected_source_archive_sha256, pins)
    require(not archive.is_relative_to(source), 'source_archive_must_be_external')
    pin_file(common, args.common_launcher, args.expected_common_sha256, pins)
    pin_file(common, manifest, args.expected_source_manifest_sha256, pins)
    config_path = pin_file(common, args.config, args.expected_config_sha256, pins)
    config = common.pinned_json(config_path, args.expected_config_sha256)
    require(common.utc(config.get('stop_at_utc')) <= stop, 'config_deadline_exceeds_evaluation')
    require(config.get('gpu_memory_budget_mib') == 5120 and config.get('device', 'cuda:0') == 'cuda:0'
            and type(config.get('minimum_free_margin_mib')) is int
            and config['minimum_free_margin_mib'] >= 1024, 'evaluation_bounded_gpu_recipe_required')
    run_path = pin_file(common, args.run, args.expected_run_sha256, pins)
    require(run_path.name == 'run.json' and run_path.is_relative_to(ROOT / 'runs'), 'authorized_existing_run_required')
    run = common.pinned_json(run_path, args.expected_run_sha256)
    run_config = pin_file(common, run_path.parent / 'config.json', args.expected_config_sha256, pins)
    require(run.get('config_sha256') == args.expected_config_sha256
            and config.get('output_dir') == str(run_path.parent), 'evaluation_run_config_mismatch')
    require(type(run.get('optimizer_steps')) is int and run['optimizer_steps'] > 0
            and not run.get('memory_smoke_only') and not run.get('serving_eligible', False)
            and not run.get('promotion_eligible', False) and run.get('approved_types', []) == [],
            'evaluation_real_unpromoted_fit_required')
    protocol = pin_file(common, args.protocol, args.expected_protocol_sha256, pins)
    require(protocol == run_path.parent / 'evaluation-protocol.json'
            and run.get('evaluation_protocol_sha256') == args.expected_protocol_sha256,
            'evaluation_protocol_mismatch')
    dataset = common.absolute_path(config.get('dataset_dir'), kind='dir')
    registry = common.absolute_path(config.get('test_exposure_registry_dir'), kind='dir')
    authorized_write(registry)
    require(run.get('dataset_lock_sha256') == config.get('expected_dataset_lock_sha256'),
            'evaluation_dataset_pin_mismatch')
    pin_file(common, dataset / 'dataset-lock.json', config.get('expected_dataset_lock_sha256'), pins)
    readonly = [source, manifest, archive, config_path, run_path, run_config, protocol, dataset]
    if args.phase.startswith('rf-'):
        require(run.get('schema_version') == 'forte.specialists.public-run.v1'
                and run.get('state') == 'trained_unselected_pending_test'
                and run.get('defer_sealed_test') is True and run.get('test_evaluation_count') == 0,
                'evaluation_detector_pending_run_required')
        require(run.get('checkpoint') == 'checkpoint_best_total.pth', 'evaluation_selected_checkpoint_invalid')
        checkpoint_sha = run.get('checkpoint_sha256')
        checkpoint = pin_file(common, run_path.parent / run['checkpoint'], checkpoint_sha, pins)
        if config.get('pretrain_weights'):
            readonly.append(pin_file(common, config['pretrain_weights'],
                                     config.get('expected_pretrain_weights_sha256'), pins))
        output = (common.absolute_path(str(args.output)) if args.phase == 'rf-dev' and args.output
                  else run_path.parent / 'cohort-detector-evaluation')
        if args.phase == 'rf-dev':
            require(args.output is not None, 'development_output_required')
        elif args.output:
            require(str(args.output) == str(output), 'final_output_override_forbidden')
        require(not output.exists() and not output.is_symlink() and output.parent.is_dir(),
                'fresh_evaluation_output_required')
        result_path = output / ('receipt.json' if args.phase == 'rf-dev' else 'result.json')
    else:
        require(run.get('schema_version') == 'forte.public-crop-lora.run.v1'
                and run.get('state') == 'trained_unselected' and run.get('training_completed') is True,
                'evaluation_crop_trained_run_required')
        output = run_path.parent
        require(not args.output or str(args.output) == str(output), 'final_output_override_forbidden')
        require(not any(path.exists() or path.is_symlink()
                        for path in (output / 'sealed-test-access.json', output / 'sealed-test-results.json')),
                'crop_final_already_attempted')
        adapter = common.absolute_path(str(output / 'adapter'), kind='dir')
        files = run.get('adapter_files')
        require(isinstance(files, dict) and {'adapter_model.safetensors', 'adapter_config.json'} <= set(files),
                'evaluation_adapter_files_missing')
        require({p.name for p in adapter.iterdir()} == set(files), 'evaluation_adapter_inventory_mismatch')
        for name, expected in files.items():
            require(isinstance(name, str) and name == Path(name).name and '\\' not in name
                    and ':' not in name, 'evaluation_adapter_name_invalid')
            pin_file(common, adapter / name, expected, pins)
        checkpoint = adapter / 'adapter_model.safetensors'
        checkpoint_sha = files['adapter_model.safetensors']
        readonly.append(adapter)
        model_dir = common.absolute_path(config.get('model_dir'), kind='dir')
        readonly.extend([model_dir, pin_file(common, config.get('model_manifest'),
                                           config.get('expected_model_manifest_sha256'), pins)])
        result_path = output / 'sealed-test-results.json'
    readonly.append(checkpoint)
    writable = [output.parent if args.phase == 'rf-dev' else run_path.parent]
    authorized_write(writable[0])
    require(writable[0].is_dir(), 'existing_result_parent_required')
    checked = {'source': source, 'manifest': manifest, 'source_receipt': source_receipt, 'pins': pins,
               'archive': archive, 'config_path': config_path, 'config': config, 'run_path': run_path,
               'run': run, 'run_config': run_config, 'protocol': protocol, 'dataset': dataset,
               'registry': registry, 'readonly': readonly, 'writable': writable, 'output': output,
               'result_path': result_path, 'checkpoint': checkpoint, 'checkpoint_sha256': checkpoint_sha,
               'stop': stop, 'attempt_marker': None, 'cohort_path': None}
    if args.phase != 'rf-dev':
        require(args.cohort_receipt and args.expected_cohort_sha256, 'final_joint_cohort_required')
        cohort_path = pin_file(common, args.cohort_receipt, args.expected_cohort_sha256, pins)
        checked['cohort_path'] = cohort_path
        readonly.append(cohort_path)
        cohort = common.pinned_json(cohort_path, args.expected_cohort_sha256)
        verify_cohort_pins(common, cohort, checked, args)
        writable.append(registry)
    else:
        require(not args.cohort_receipt and not args.expected_cohort_sha256, 'development_cannot_claim_cohort')
    # Frozen files nested inside a writable final run/registry are overmounted RO.
    # Entire source, dataset, model and adapter directories may never be writable.
    verify_mount_isolation(args.phase, readonly, writable, run_path.parent)
    report = common.absolute_path(str(args.report))
    require(not report.exists() and report.parent.is_dir() and not report.is_relative_to(source),
            'fresh_evaluation_report_required')
    authorized_write(report)
    report_exclusions = [path for path in readonly if path.is_dir()] + writable + [registry, run_path.parent]
    require(not any(report.is_relative_to(path) for path in report_exclusions),
            'evaluation_report_must_be_separate_from_inputs_and_results')
    checked['report'] = report
    checked['readonly'] = sorted(set(readonly + [Path(p) for p in pins]), key=lambda p: (len(p.parts), str(p)))
    return checked


def phase_argv(args, checked):
    if args.phase == 'rf-dev':
        return ['training.specialists.public_dev_validate', '--run', str(checked['run_path']),
                '--expected-run-sha256', args.expected_run_sha256, '--output', str(checked['output'])]
    cohort = ['--cohort-receipt', str(checked['cohort_path']), '--expected-cohort-sha256', args.expected_cohort_sha256]
    if args.phase == 'rf-final':
        return ['training.specialists.public_train', '--config', str(checked['config_path']),
                '--final-eval-run', str(checked['run_path']), '--expected-run-sha256', args.expected_run_sha256,
                '--protocol', str(checked['protocol'])] + cohort
    return ['training.public_crop_lora', '--config', str(checked['config_path']),
            '--final-eval-run-sha256', args.expected_run_sha256] + cohort


def bootstrap(args, checked, *, preflight):
    if not preflight:
        argv = phase_argv(args, checked)
        return ("import runpy,sys,torch; "
                "torch.cuda.set_per_process_memory_fraction(5120*1024**2/torch.cuda.get_device_properties(0).total_memory,0); "
                f"sys.argv={argv!r}; runpy.run_module(sys.argv[0],run_name='__main__')")
    values = {'phase': args.phase, 'config': str(checked['config_path']), 'run': str(checked['run_path']),
              'run_sha': args.expected_run_sha256, 'cohort': str(checked['cohort_path']) if checked['cohort_path'] else None,
              'cohort_sha': args.expected_cohort_sha256}
    return f'''import json,os,torch
v={values!r}
def check(ok):
    if not ok: raise RuntimeError('cpu_evaluation_preflight_contract_failed')
check(not any(x.startswith('nvidia') for x in os.listdir('/dev')))
check(not torch.cuda.is_initialized())
from pathlib import Path
if v['phase']=='rf-dev':
    from training.specialists.public_dev_validate import pending_inputs,preflight,development_view
    run,config,checkpoint=pending_inputs(Path(v['run']),v['run_sha'])
    root,lock=preflight(config)
    development_view(lock)
elif v['phase']=='rf-final':
    from training.specialists.public_train import preflight
    config=json.loads(Path(v['config']).read_text())
    preflight(config)
else:
    from training.public_crop_lora import resolve_config,cohort_descriptor
    config=resolve_config(Path(v['config']))
    cohort_descriptor(config,v['run_sha'])
if v['cohort']:
    from training.specialists.public_cohort import verify_cohort
    verify_cohort(Path(v['cohort']),v['cohort_sha'])
check(not torch.cuda.is_initialized())
print(json.dumps({{'ok':True,'gpu_initialized':False,'test_access_claimed':False}}))
'''


def container_args(args, checked, common, *, preflight):
    name = args.name + '-preflight' if preflight else args.name
    result = ['docker', 'create', '--name', name, '--user', f'{os.getuid()}:{os.getgid()}',
              '--network', 'none', '--read-only', '--cpus', '4', '--memory', '24g', '--shm-size', '1g',
              '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
              '--tmpfs', '/tmp:rw,size=2147483648,mode=1777', '--workdir', str(checked['source'])]
    if args.phase == 'crop-final':
        result += ['--tmpfs', '/triton-cache:rw,nosuid,nodev,exec,size=536870912,mode=1777']
    if not preflight:
        result += ['--gpus', 'device=0']
    env = {'NVIDIA_VISIBLE_DEVICES': 'void' if preflight else '0',
           'CUDA_VISIBLE_DEVICES': '' if preflight else '0', 'HOME': '/tmp', 'HF_HOME': '/tmp/hf', 'RF_HOME': '/tmp/rf',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
           'WANDB_MODE': 'disabled', 'DO_NOT_TRACK': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(checked['source']), 'OMP_NUM_THREADS': '2',
           'PYTORCH_ALLOC_CONF': 'backend:native'}
    if args.phase == 'crop-final':
        env['TRITON_CACHE_DIR'] = '/triton-cache'
    for key, value in env.items():
        result += ['--env', f'{key}={value}']
    if not preflight:
        for path in checked['writable']:
            result += ['--mount', f'type=bind,src={path},dst={path}']
    else:
        path = checked['registry']
        result += ['--mount', f'type=bind,src={path},dst={path},readonly']
    for path in checked['readonly']:
        result += ['--mount', f'type=bind,src={path},dst={path},readonly']
    return result + ['--entrypoint', 'python3', args.image, '-c', bootstrap(args, checked, preflight=preflight)]


def unchanged(args, checked, common):
    common.verify_sources(checked['source'], checked['manifest'], args.expected_source_manifest_sha256)
    for path, expected in checked['pins'].items():
        require(digest(path) == expected, 'evaluation_immutable_input_changed')


def verify_result(args, checked, common):
    path = common.absolute_path(str(checked['result_path']), kind='file')
    require(path.stat().st_size <= 32 * 1024**2, 'evaluation_result_too_large')
    result = json.loads(path.read_text())
    require(isinstance(result, dict) and result.get('run_sha256') == args.expected_run_sha256,
            'evaluation_result_run_mismatch')
    if args.phase == 'rf-dev':
        require(result.get('schema_version') == 'forte.specialists.public-dev-rehearsal.v1'
                and result.get('state') == 'complete' and result.get('scope') == 'development_only'
                and result.get('test_evaluation_count') == 0 and result.get('sealed_pixels_accessed') is False
                and result.get('checkpoint_selection_changed') is False, 'development_completion_invalid')
    elif args.phase == 'rf-final':
        require(result.get('schema_version') == 'forte.specialists.public-final-evaluation.v1'
                and result.get('state') == 'complete' and result.get('member_id') == 'detector'
                and result.get('test_evaluation_count') == 1
                and result.get('cohort_receipt_sha256') == args.expected_cohort_sha256,
                'detector_final_completion_invalid')
    else:
        require(result.get('schema_version') == 'forte.public-crop-lora.sealed-evaluation.v1'
                and result.get('promotion_eligible') is False
                and result.get('test_reuse_for_tuning_allowed') is False
                and isinstance(result.get('adapter'), dict), 'crop_final_completion_invalid')
        marker = json.loads((checked['output'] / 'sealed-test-access.json').read_text())
        require(marker.get('run_sha256') == args.expected_run_sha256, 'crop_access_marker_mismatch')
    if args.phase.startswith('rf-'):
        require(result.get('checkpoint_sha256') == checked['checkpoint_sha256']
                and result.get('config_sha256') == args.expected_config_sha256
                and result.get('evaluation_protocol_sha256') == args.expected_protocol_sha256,
                'evaluation_result_artifact_mismatch')
    if checked['attempt_marker']:
        marker = json.loads(checked['attempt_marker'].read_text())
        require(marker.get('cohort_sha256') == args.expected_cohort_sha256
                and marker.get('run_sha256') == args.expected_run_sha256
                and marker.get('checkpoint_sha256') == checked['checkpoint_sha256'],
                'cohort_attempt_marker_mismatch')
    metrics = {}

    def numeric(value, prefix, depth=0):
        if depth > 8 or len(metrics) >= 512:
            return
        if type(value) in (float, int):
            require(math.isfinite(value), 'evaluation_nonfinite_metric')
            metrics[prefix] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and re.fullmatch(r'[a-zA-Z0-9_./:+() -]{1,100}', key):
                    numeric(item, prefix + '/' + key, depth + 1)

    for key in ('metrics', 'fixed_detection_metrics', 'native_selected_dev_metrics', 'fixed_development_metrics', 'adapter'):
        if key in result:
            numeric(result[key], key)
    require(bool(metrics), 'evaluation_numeric_metrics_missing')
    return {'result_sha256': digest(path), 'state': 'complete', 'numeric_metrics': metrics}


def emit(kind, value):
    print('PUBLIC_EVALUATION_' + kind + ' ' + json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def execute(args):
    import fcntl
    common = load_common(args.common_launcher, args.expected_common_sha256)
    try:
        checked = validate_inputs(args, common)
    except common.GuardError as exc:
        raise EvaluationError(str(exc)) from None
    started = time.monotonic()
    report = {'schema_version': 'forte.public-evaluation-host.v1', 'phase': args.phase, 'passed': False,
              'failure': None, 'image_id': args.image, 'run_sha256': args.expected_run_sha256,
              'config_sha256': args.expected_config_sha256, 'checkpoint_sha256': checked['checkpoint_sha256'],
              'protocol_sha256': args.expected_protocol_sha256, 'source': checked['source_receipt'],
              'source_archive_sha256': args.expected_source_archive_sha256,
              'common_launcher_sha256': args.expected_common_sha256, 'launcher_sha256': digest(__file__),
              'network': 'none', 'promotion_eligible': False, 'gpu_samples': 0,
              'maximum_sampled_training_mib': 0, 'minimum_sampled_global_free_mib': 2**31 - 1,
              'limits': {'wall_seconds': args.max_wall_seconds, 'stop_at_utc': args.stop_at_utc,
                         'allocator_mib': 5120, 'start_free_mib': 6656, 'running_free_mib': 1536,
                         'actual_evaluation_max_mib': 5632}}
    if checked['cohort_path']:
        report['cohort_receipt_sha256'] = args.expected_cohort_sha256
    owned, before, lock = [], None, None
    previous = {}

    def interrupt(signum, _frame):
        if signum == signal.SIGALRM:
            raise EvaluationError('absolute_deadline_reached' if datetime.now(timezone.utc) >= checked['stop']
                                  else 'hard_wall_time_exceeded')
        raise EvaluationError('evaluation_host_signal_interruption')

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        previous[sig] = signal.signal(sig, interrupt)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, max(.001, min(args.max_wall_seconds,
                     (checked['stop'] - datetime.now(timezone.utc)).total_seconds())))
    try:
        before = common.production_bindings()
        report['production_before'] = before
        report['readiness_before'] = common.readiness()
        require(common.inspect_container(args.name) is None and common.inspect_container(args.name + '-preflight') is None,
                'evaluation_container_name_exists')
        image = json.loads(common.command(['docker', 'image', 'inspect', args.image]))[0]
        require(image['Id'] == args.image, 'evaluation_image_mismatch')
        emit('START', {'phase': args.phase, 'state': 'cpu_preflight', 'run_sha256': args.expected_run_sha256})
        preflight_id = common.command(container_args(args, checked, common, preflight=True))
        require(SHA.fullmatch(preflight_id), 'evaluation_container_id_invalid')
        owned.append((args.name + '-preflight', preflight_id))
        log_path = checked['report'].with_name(checked['report'].stem + '.preflight.private.log')
        with common.private_file(log_path) as log:
            state = common.wait_container(args.name + '-preflight', preflight_id, log, deadline=checked['stop'],
                                          wall_started=started, wall_seconds=min(args.max_wall_seconds, 900))
        report['cpu_preflight_exit_code'] = state['State']['ExitCode']
        require(state['State']['ExitCode'] == 0, 'evaluation_cpu_preflight_failed')
        unchanged(args, checked, common)
        lock = common.LOCK_PATH.open('a')
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EvaluationError('evaluation_gpu_lock_busy') from None
        report['gpu_lock_held'] = True
        require(before == common.production_bindings(), 'production_changed_during_evaluation_preflight')
        common.readiness()
        report['queue_before'] = common.queue_state()
        require(report['queue_before'] == {'active': 0, 'queued': 0}, 'live_queue_not_idle')
        report['gpu_before'], _ = common.gpu_state()
        require(report['gpu_before']['free_mib'] >= 6656, 'evaluation_insufficient_spare_gpu')
        if checked['attempt_marker']:
            require(not checked['attempt_marker'].exists() and not checked['attempt_marker'].is_symlink(),
                    'cohort_member_attempt_already_exists')
        require(not checked['result_path'].exists(), 'evaluation_result_created_during_preflight')
        require(time.monotonic() - started < args.max_wall_seconds and datetime.now(timezone.utc) < checked['stop'],
                'evaluation_time_budget_exhausted')
        container_id = common.command(container_args(args, checked, common, preflight=False))
        require(SHA.fullmatch(container_id), 'evaluation_container_id_invalid')
        owned.append((args.name, container_id))
        report['container_id'] = container_id
        emit('START', {'phase': args.phase, 'state': 'isolated_gpu_evaluation', 'gpu_lock_held': True,
                        'gpu_free_before_mib': report['gpu_before']['free_mib']})
        log_path = checked['report'].with_name(checked['report'].stem + '.evaluation.private.log')
        with common.private_file(log_path) as log:
            state = common.wait_container(args.name, container_id, log, deadline=checked['stop'],
                                          wall_started=started, wall_seconds=args.max_wall_seconds,
                                          monitor_gpu=True, receipt=report)
        report['evaluation_exit_code'] = state['State']['ExitCode']
        require(state['State']['ExitCode'] == 0, 'evaluation_container_failed')
        report['result'] = verify_result(args, checked, common)
        report['passed'] = True
    except (EvaluationError, common.GuardError) as exc:
        report['failure'] = str(exc)
    except BaseException:
        report['failure'] = 'unexpected_evaluation_host_failure'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        failures = []
        for name, container_id in reversed(owned):
            try:
                common.stop_owned(name, container_id)
            except BaseException:
                failures.append('evaluation_owned_cleanup_failed')
        report['owned_containers_retained_stopped'] = bool(owned) and not failures
        try:
            unchanged(args, checked, common)
            report['immutable_inputs_unchanged'] = True
        except BaseException:
            failures.append('evaluation_immutable_inputs_changed')
        if before is not None:
            try:
                report['production_after'] = common.production_bindings()
                report['production_unchanged'] = before == report['production_after']
                require(report['production_unchanged'], 'evaluation_production_binding_changed')
                report['readiness_after'] = common.readiness()
                report['queue_after'] = common.queue_state()
                report['gpu_after'], _ = common.gpu_state()
            except BaseException:
                failures.append('evaluation_production_postcheck_failed')
        if lock:
            lock.close()
        if failures:
            report['postcheck_failures'] = failures
            report['passed'] = False
        report['maximum_sampled_evaluation_mib'] = report.pop('maximum_sampled_training_mib')
        if not report['gpu_samples']:
            report['minimum_sampled_global_free_mib'] = None
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        common.write_report(checked['report'], report)
        emit('FINAL', report)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
    return 0 if report['passed'] else 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--phase', choices=PHASES, required=True)
    for field in ('run', 'config', 'source-dir', 'source-manifest', 'source-archive', 'common-launcher', 'protocol', 'report'):
        value.add_argument('--' + field, type=Path, required=True)
    for field in ('run', 'config', 'source-manifest', 'source-archive', 'common', 'protocol'):
        value.add_argument('--expected-' + field + '-sha256', required=True)
    value.add_argument('--cohort-receipt', type=Path)
    value.add_argument('--expected-cohort-sha256')
    value.add_argument('--output', type=Path)
    value.add_argument('--image', required=True)
    value.add_argument('--name', required=True)
    value.add_argument('--max-wall-seconds', type=int, default=1800)
    value.add_argument('--stop-at-utc', default='2026-10-18T05:45:00Z')
    return value


def main():
    args = parser().parse_args()
    try:
        return execute(args)
    except EvaluationError as exc:
        emit('FINAL', {'passed': False, 'failure': str(exc), 'gpu_container_started': False})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
