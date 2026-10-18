"""Evaluation orchestration contracts; no dataset pixels, Docker or GPU access."""
from argparse import Namespace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_public_evaluation_host as evaluation
from tools import run_public_training_host as common


@pytest.fixture(autouse=True)
def portable_identity(monkeypatch):
    if not hasattr(os, 'getuid'):
        monkeypatch.setattr(evaluation.os, 'getuid', lambda: 1000, raising=False)
        monkeypatch.setattr(evaluation.os, 'getgid', lambda: 1000, raising=False)


def command_fixture(tmp_path, phase='rf-dev'):
    args = Namespace(phase=phase, name='forte-public-eval-contract', image='sha256:' + 'a' * 64,
        expected_run_sha256='b' * 64, expected_cohort_sha256='c' * 64)
    source = tmp_path / 'repo'
    run = tmp_path / 'run'
    checked = {'source': source, 'run_path': run / 'run.json', 'config_path': run / 'config.json',
        'protocol': run / 'evaluation-protocol.json', 'output': tmp_path / 'newdev',
        'cohort_path': tmp_path / 'registry/cohort.json' if phase != 'rf-dev' else None,
        'registry': tmp_path / 'registry', 'readonly': [source, run / 'run.json'],
        'writable': [tmp_path / 'newdev-parent']}
    return args, checked


def test_unpinned_guard_code_is_never_imported(tmp_path, monkeypatch):
    path = tmp_path / 'untrusted.py'
    path.write_text("raise RuntimeError('must never execute')\n")
    monkeypatch.setattr(evaluation.sys, 'platform', 'linux')
    with pytest.raises(evaluation.EvaluationError, match='common_launcher_pin_mismatch'):
        evaluation.load_common(path, '0' * 64)


@pytest.mark.parametrize('phase', evaluation.PHASES)
def test_cpu_preflight_is_gpu_hidden_and_every_host_mount_readonly(tmp_path, phase):
    args, checked = command_fixture(tmp_path, phase)
    command = evaluation.container_args(args, checked, common, preflight=True)
    assert '--gpus' not in command and 'NVIDIA_VISIBLE_DEVICES=void' in command
    assert 'CUDA_VISIBLE_DEVICES=' in command and 'PYTHONDONTWRITEBYTECODE=1' in command
    assert 'PYTORCH_ALLOC_CONF=backend:native' in command
    mounts = [command[i + 1] for i, value in enumerate(command) if value == '--mount']
    assert mounts and all(value.endswith(',readonly') for value in mounts)
    bootstrap = command[-1]
    compile(bootstrap, '<synthetic-cpu-preflight>', 'exec')
    assert 'claim_cohort_member' not in bootstrap and 'final_evaluate' not in bootstrap
    assert 'torch.cuda.is_initialized()' in bootstrap


@pytest.mark.parametrize('phase', evaluation.PHASES)
def test_gpu_command_is_offline_capped_and_uses_exact_phase_api(tmp_path, phase):
    args, checked = command_fixture(tmp_path, phase)
    command = evaluation.container_args(args, checked, common, preflight=False)
    assert command[command.index('--network') + 1] == 'none' and '--read-only' in command
    assert command[command.index('--gpus') + 1] == 'device=0'
    assert 'HF_HUB_OFFLINE=1' in command and 'TRANSFORMERS_OFFLINE=1' in command
    assert 'set_per_process_memory_fraction(5120*1024**2/' in command[-1]
    compile(command[-1], '<synthetic-gpu-bootstrap>', 'exec')
    argv = evaluation.phase_argv(args, checked)
    assert '--defer-test' not in argv and '--smoke' not in argv
    if phase == 'rf-dev':
        assert argv[0] == 'training.specialists.public_dev_validate'
        assert '--cohort-receipt' not in argv and '--output' in argv
    else:
        assert '--cohort-receipt' in argv and '--expected-cohort-sha256' in argv
        assert '--final-eval-run' in argv if phase == 'rf-final' else '--final-eval-run-sha256' in argv


@pytest.mark.parametrize('preflight', [True, False])
def test_crop_final_cache_is_only_separate_bounded_exec_mount(tmp_path, preflight):
    args, checked = command_fixture(tmp_path, 'crop-final')
    command = evaluation.container_args(args, checked, common, preflight=preflight)
    cache = '/triton-cache:rw,nosuid,nodev,exec,size=536870912,mode=1777'
    tmpfs = [command[i + 1] for i, value in enumerate(command) if value == '--tmpfs']
    assert tmpfs == ['/tmp:rw,size=2147483648,mode=1777', cache]
    assert command.count('TRITON_CACHE_DIR=/triton-cache') == 1
    assert command[command.index('--memory') + 1] == '24g'
    assert command[command.index('--cpus') + 1] == '4'
    assert command[command.index('--shm-size') + 1] == '1g'
    assert command[command.index('--cap-drop') + 1] == 'ALL'
    assert command[command.index('--security-opt') + 1] == 'no-new-privileges'
    prefix = command[:command.index('--entrypoint')]
    for flag, value in [('--tmpfs', cache), ('--env', 'TRITON_CACHE_DIR=/triton-cache')]:
        index = prefix.index(value)
        assert prefix[index - 1] == flag
        del prefix[index - 1:index + 1]
    for phase in ('rf-dev', 'rf-final'):
        args.phase = phase
        rf = evaluation.container_args(args, checked, common, preflight=preflight)
        assert prefix == rf[:rf.index('--entrypoint')]
        assert '/triton-cache' not in ' '.join(rf) and 'TRITON_CACHE_DIR' not in ' '.join(rf)


def test_development_receipt_requires_no_sealed_access_and_exact_artifacts(tmp_path):
    args, checked = command_fixture(tmp_path)
    args.expected_config_sha256 = 'd' * 64
    args.expected_protocol_sha256 = 'e' * 64
    checked.update(result_path=tmp_path / 'receipt.json', checkpoint_sha256='f' * 64, attempt_marker=None)
    receipt = {'schema_version': 'forte.specialists.public-dev-rehearsal.v1', 'state': 'complete',
        'scope': 'development_only', 'test_evaluation_count': 0, 'sealed_pixels_accessed': False,
        'checkpoint_selection_changed': False, 'run_sha256': args.expected_run_sha256,
        'checkpoint_sha256': checked['checkpoint_sha256'], 'config_sha256': args.expected_config_sha256,
        'evaluation_protocol_sha256': args.expected_protocol_sha256,
        'native_selected_dev_metrics': {'val/AP': .2},
        'fixed_development_metrics': {'recall': .3, 'predictions': [{'filename': 'NEVER-EMIT'}]}}
    checked['result_path'].write_text(json.dumps(receipt))
    portable_common = SimpleNamespace(absolute_path=lambda path, **kwargs: Path(path))
    result = evaluation.verify_result(args, checked, portable_common)
    assert result['numeric_metrics'] == {'native_selected_dev_metrics/val/AP': .2,
                                         'fixed_development_metrics/recall': .3}
    assert 'NEVER-EMIT' not in json.dumps(result)
    receipt['sealed_pixels_accessed'] = True
    checked['result_path'].write_text(json.dumps(receipt))
    with pytest.raises(evaluation.EvaluationError, match='development_completion_invalid'):
        evaluation.verify_result(args, checked, portable_common)


def test_final_run_writable_mount_keeps_adapter_directory_and_files_readonly(tmp_path):
    run_parent = tmp_path / 'existing-run'
    adapter = run_parent / 'adapter'
    adapter.mkdir(parents=True)
    checkpoint = run_parent / 'checkpoint.pth'
    checkpoint.write_bytes(b'synthetic')
    evaluation.verify_mount_isolation('crop-final', [adapter, checkpoint], [run_parent], run_parent)
    evaluation.verify_mount_isolation('rf-final', [checkpoint], [run_parent], run_parent)
    with pytest.raises(evaluation.EvaluationError, match='development_writable_input_file_overlap'):
        evaluation.verify_mount_isolation('rf-dev', [checkpoint], [run_parent], run_parent)
    with pytest.raises(evaluation.EvaluationError, match='evaluation_writable_input_directory_overlap'):
        evaluation.verify_mount_isolation('crop-final', [adapter], [adapter], run_parent)
    source = run_parent / 'forbidden-source'
    source.mkdir()
    with pytest.raises(evaluation.EvaluationError, match='evaluation_writable_input_directory_overlap'):
        evaluation.verify_mount_isolation('crop-final', [source], [run_parent], run_parent)


@pytest.mark.skipif(os.name == 'nt', reason='Canonical Linux cohort artifact paths')
def test_final_mounts_all_peer_pins_but_never_claims_cohort_access(tmp_path, monkeypatch):
    root = tmp_path / 'authorized'
    root.mkdir()
    monkeypatch.setattr(evaluation, 'ROOT', root)
    registry = root / 'registry'
    registry.mkdir()
    cohort_path = registry / 'public-sealed-inventory.json'
    cohort_path.write_text('{}')
    dataset = root / 'public-data'
    dataset.mkdir()

    def pin(path, payload=b'synthetic-metadata'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {'path': str(path), 'sha256': evaluation.digest(path)}

    dataset_pin = pin(dataset / 'dataset-lock.json', b'{}')
    members = {}
    for name in ('detector', 'crop_vlm'):
        run_parent = root / 'runs' / name / 'output'
        members[name] = {key: pin(run_parent / filename) for key, filename in
            [('run', 'run.json'), ('config', 'config.json'), ('checkpoint', 'selected.pth'),
             ('protocol', 'evaluation-protocol.json')]}
        members[name]['artifacts'] = {'evaluation_source': pin(root / 'repo-v2' / (name + '.py'))}
    selected = members['detector']
    checked = {'registry': registry, 'cohort_path': cohort_path, 'dataset': dataset,
        'config': {'expected_dataset_lock_sha256': dataset_pin['sha256']}, 'pins': {}, 'readonly': [],
        'run_path': Path(selected['run']['path']), 'run_config': Path(selected['config']['path']),
        'protocol': Path(selected['protocol']['path']), 'checkpoint': Path(selected['checkpoint']['path']),
        'checkpoint_sha256': selected['checkpoint']['sha256']}
    args = Namespace(phase='rf-final', expected_run_sha256=selected['run']['sha256'],
        expected_config_sha256=selected['config']['sha256'], expected_protocol_sha256=selected['protocol']['sha256'])
    cohort = {'schema_version': 'forte.public-eval-cohort.v1', 'state': 'registered',
        'member_ids': ['crop_vlm', 'detector'], 'registry_dir': str(registry), 'dataset_lock': dataset_pin,
        'authorization': pin(root / 'manifests/authorization.json'), 'members': members}
    evaluation.verify_cohort_pins(common, cohort, checked, args)
    assert Path(members['crop_vlm']['checkpoint']['path']) in checked['readonly']
    assert Path(members['crop_vlm']['artifacts']['evaluation_source']['path']) in checked['readonly']
    assert not checked['attempt_marker'].exists()
    checked['attempt_marker'].write_text('{}')
    with pytest.raises(evaluation.EvaluationError, match='cohort_member_attempt_already_exists'):
        evaluation.verify_cohort_pins(common, cohort, checked, args)


@pytest.mark.skipif(os.name == 'nt', reason='Canonical Linux host paths')
def test_actual_common_guard_pin_loads_without_changes():
    path = Path(common.__file__).resolve()
    loaded = evaluation.load_common(path, evaluation.digest(path))
    assert loaded.LOCK_PATH == common.LOCK_PATH and loaded.HARD_STOP == evaluation.DEADLINE


@pytest.mark.skipif(os.name == 'nt', reason='Canonical Linux host paths')
def test_dev_host_inputs_are_pinned_and_writable_parent_must_be_disjoint(tmp_path, monkeypatch):
    root = tmp_path / 'authorized'
    root.mkdir()
    monkeypatch.setattr(evaluation, 'ROOT', root)
    source = root / 'repo-v2'
    (source / 'training').mkdir(parents=True)
    (source / 'training/__init__.py').write_text('')
    manifest = root / 'source.manifest.json'
    manifest.write_text(json.dumps({'files': {'training/__init__.py': evaluation.digest(source / 'training/__init__.py')}}))
    archive = root / 'source.tar.gz'
    archive.write_bytes(b'synthetic-audit-archive')
    common_path = Path(common.__file__).resolve()
    dataset = root / 'public-data'
    dataset.mkdir()
    lock = dataset / 'dataset-lock.json'
    lock.write_text('{}')
    registry = root / 'registry'
    registry.mkdir()
    run_root = root / 'runs/rf-one/output'
    run_root.mkdir(parents=True)
    weights = root / 'official-small.pth'
    weights.write_bytes(b'synthetic-official-weights')
    checkpoint = run_root / 'checkpoint_best_total.pth'
    checkpoint.write_bytes(b'synthetic-selected-weights')
    config = run_root / 'config.json'
    config.write_text(json.dumps({'output_dir': str(run_root), 'dataset_dir': str(dataset),
        'expected_dataset_lock_sha256': evaluation.digest(lock), 'test_exposure_registry_dir': str(registry),
        'stop_at_utc': '2026-10-18T05:45:00Z', 'gpu_memory_budget_mib': 5120, 'minimum_free_margin_mib': 1024,
        'pretrain_weights': str(weights), 'expected_pretrain_weights_sha256': evaluation.digest(weights)}))
    protocol = run_root / 'evaluation-protocol.json'
    protocol.write_text('{}')
    run = run_root / 'run.json'
    run.write_text(json.dumps({'schema_version': 'forte.specialists.public-run.v1',
        'state': 'trained_unselected_pending_test', 'defer_sealed_test': True, 'test_evaluation_count': 0,
        'optimizer_steps': 8, 'config_sha256': evaluation.digest(config),
        'evaluation_protocol_sha256': evaluation.digest(protocol),
        'dataset_lock_sha256': evaluation.digest(lock), 'checkpoint': checkpoint.name,
        'checkpoint_sha256': evaluation.digest(checkpoint)}))
    dev_parent = root / 'dev/rehearsal-one'
    dev_parent.mkdir(parents=True)
    args = Namespace(phase='rf-dev', image='sha256:' + 'a' * 64, name='forte-public-eval-test',
        max_wall_seconds=1800, stop_at_utc='2026-10-18T05:45:00Z', source_dir=source,
        source_manifest=manifest, expected_source_manifest_sha256=evaluation.digest(manifest),
        source_archive=archive, expected_source_archive_sha256=evaluation.digest(archive),
        common_launcher=common_path, expected_common_sha256=evaluation.digest(common_path),
        config=config, expected_config_sha256=evaluation.digest(config), run=run,
        expected_run_sha256=evaluation.digest(run), protocol=protocol,
        expected_protocol_sha256=evaluation.digest(protocol), output=dev_parent / 'output',
        cohort_receipt=None, expected_cohort_sha256=None, report=root / 'host-report.json')
    now = datetime(2026, 10, 18, 1, tzinfo=timezone.utc)
    checked = evaluation.validate_inputs(args, common, now=now)
    assert checked['writable'] == [dev_parent] and checked['checkpoint'] == checkpoint
    args.report = dataset / 'host-report.json'
    with pytest.raises(evaluation.EvaluationError, match='evaluation_report_must_be_separate'):
        evaluation.validate_inputs(args, common, now=now)
    args.report = run_root / 'cohort-detector-evaluation.json'
    with pytest.raises(evaluation.EvaluationError, match='evaluation_report_must_be_separate'):
        evaluation.validate_inputs(args, common, now=now)
    args.report = root / 'host-report.json'
    args.output = dataset / 'new-dev'
    with pytest.raises(evaluation.EvaluationError, match='evaluation_writable_input_directory_overlap'):
        evaluation.validate_inputs(args, common, now=now)
    args.output = dev_parent / 'output'
    args.stop_at_utc = '2026-10-18T05:45:01Z'
    with pytest.raises(evaluation.EvaluationError, match='evaluation_deadline_invalid'):
        evaluation.validate_inputs(args, common, now=now)


@pytest.mark.skipif(os.name == 'nt', reason='Linux flock/SIGALRM supervisor')
@pytest.mark.parametrize('failure', ['cpu', 'gpu', 'signal', 'postcheck'])
def test_failure_stops_only_owned_jobs_and_preserves_postchecks(tmp_path, monkeypatch, failure):
    args, checked = command_fixture(tmp_path)
    args.common_launcher = tmp_path / 'common.py'
    args.expected_common_sha256 = '1' * 64
    args.expected_config_sha256 = '2' * 64
    args.expected_protocol_sha256 = '3' * 64
    args.expected_source_archive_sha256 = '4' * 64
    args.expected_source_manifest_sha256 = '5' * 64
    args.max_wall_seconds = 10
    args.stop_at_utc = '2026-10-18T05:45:00Z'
    checked.update(stop=datetime.now(timezone.utc) + timedelta(seconds=30), report=tmp_path / 'report.json',
        checkpoint_sha256='6' * 64, source_receipt={}, attempt_marker=None,
        result_path=tmp_path / 'new-result.json')
    monkeypatch.setattr(evaluation, 'load_common', lambda *a: common)
    monkeypatch.setattr(evaluation, 'validate_inputs', lambda *a: checked)
    monkeypatch.setattr(evaluation, 'unchanged', lambda *a: None)
    monkeypatch.setattr(evaluation, 'verify_result', lambda *a: {'state': 'complete'})
    monkeypatch.setattr(common, 'LOCK_PATH', tmp_path / 'gpu-training.lock')
    monkeypatch.setattr(common, 'inspect_container', lambda _: None)
    calls = []
    binding_reads = []

    def bindings():
        binding_reads.append(True)
        return {'all_five': 'changed' if failure == 'postcheck' and len(binding_reads) > 2 else 'original'}

    monkeypatch.setattr(common, 'production_bindings', bindings)
    monkeypatch.setattr(common, 'readiness', lambda: {'status': 'ready'})
    monkeypatch.setattr(common, 'queue_state', lambda: {'active': 0, 'queued': 0})
    monkeypatch.setattr(common, 'gpu_state', lambda: ({'free_mib': 7000}, {}))
    monkeypatch.setattr(common, 'stop_owned', lambda name, cid: calls.append((name, cid)))
    creates = iter(['a' * 64, 'b' * 64])

    def command(command_args, **unused):
        if command_args[:3] == ['docker', 'image', 'inspect']:
            return json.dumps([{'Id': args.image}])
        assert command_args[:2] == ['docker', 'create']
        return next(creates)

    monkeypatch.setattr(common, 'command', command)

    def wait(name, *unused, **kwargs):
        if name.endswith('-preflight'):
            return {'State': {'ExitCode': 1 if failure == 'cpu' else 0}}
        if failure == 'signal':
            evaluation.signal.raise_signal(evaluation.signal.SIGTERM)
        if failure == 'gpu':
            raise common.GuardError('global_free_memory_guard')
        return {'State': {'ExitCode': 0}}

    monkeypatch.setattr(common, 'wait_container', wait)
    assert evaluation.execute(args) == 1
    report = json.loads(checked['report'].read_text())
    assert report['owned_containers_retained_stopped'] is True
    if failure == 'cpu':
        assert calls == [(args.name + '-preflight', 'a' * 64)]
    else:
        assert calls == [(args.name, 'b' * 64), (args.name + '-preflight', 'a' * 64)]
    if failure == 'postcheck':
        assert report['production_unchanged'] is False and report['postcheck_failures']
    else:
        assert report['production_unchanged'] is True and report['queue_after'] == {'active': 0, 'queued': 0}
