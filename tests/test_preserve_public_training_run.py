"""Selected-artifact preservation contracts using synthetic bytes, no SSH/GPU."""
from argparse import Namespace
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from tools import preserve_public_training_run as helper


ROOT = '/data/claude/training/public-20261018/runs/rf-small512-v1/output'


def index_fixture(kind='rf'):
    run = {'schema_version': 'forte.specialists.public-run.v1' if kind == 'rf' else 'forte.public-crop-lora.run.v1',
           'state': 'trained_unselected_pending_test' if kind == 'rf' else 'trained_unselected',
           'optimizer_steps': 8, 'config_sha256': 'a' * 64, 'evaluation_protocol_sha256': 'b' * 64,
           'approved_types': [], 'promotion_eligible': False, 'test_evaluation_count': 0}
    if kind == 'rf':
        run.update(mode='fixed_recipe_train', defer_sealed_test=True,
                   checkpoint='checkpoint_best_total.pth', checkpoint_sha256='c' * 64)
        names = {'checkpoint_best_total.pth': 'c' * 64}
    else:
        run.update(training_completed=True, memory_smoke_only=False,
                   adapter_files={'adapter_model.safetensors': 'c' * 64, 'adapter_config.json': 'e' * 64,
                                  'README.md': 'f' * 64})
        names = {'adapter/adapter_model.safetensors': 'c' * 64, 'adapter/adapter_config.json': 'e' * 64}
    return {'schema_version': helper.SCHEMA, 'kind': kind, 'host_alias': helper.HOST_ALIAS,
            'remote_run_root': ROOT, 'remote_run_sha256': 'd' * 64, 'run_state': run['state'],
            'artifacts': [{'relative_path': name, 'remote_path': ROOT + '/' + name,
                           'bytes': 16, 'sha256': pin} for name, pin in names.items()],
            'receipts': [{'relative_path': 'receipts/' + name, 'remote_sha256': pin, 'content': content}
                         for name, pin, content in [('run.json', 'd' * 64, run),
                            ('config.json', 'a' * 64, {'schema_version': 'synthetic-config'}),
                            ('evaluation-protocol.json', 'b' * 64, {'schema_version': 'synthetic-protocol'})]],
            'receipt_scope': 'frozen_training_metadata_only',
            'remote_originals_retained': True, 'images_or_datasets_included': False, 'promotion_eligible': False}


@pytest.mark.parametrize('kind', ['rf', 'crop'])
def test_exact_successful_artifact_index_accepted(kind):
    index = index_fixture(kind)
    assert helper.validate_index(index, run_root=ROOT, kind=kind, expected_run_sha256='d' * 64) == 16 * len(index['artifacts'])


@pytest.mark.parametrize('path', ['/home/ubuntu/forte/run', ROOT + '/../other', ROOT + ';cat /etc/passwd',
                                 ROOT + '/bad name', 'relative/path', str(helper.REMOTE_RUNS)])
def test_unscoped_or_shell_shaped_remote_paths_rejected(path):
    with pytest.raises(helper.PreservationError):
        helper.remote_path(path, run=True)


@pytest.mark.parametrize('name', ['image.jpg', '../adapter_model.safetensors', 'sub/adapter_config.json',
                                 'archive.tar.gz', 'private.log'])
def test_adapter_never_accepts_media_logs_archive_or_traversal(name):
    run = index_fixture('crop')['receipts'][0]['content']
    run['adapter_files'][name] = '1' * 64
    with pytest.raises(helper.PreservationError):
        helper.artifact_names(run, 'crop')


@pytest.mark.parametrize('size', [0, -1, 2 * 1024**3 + 1, True])
def test_artifact_size_bounds_rejected(size):
    index = index_fixture()
    index['artifacts'][0]['bytes'] = size
    with pytest.raises(helper.PreservationError, match='artifact_size_or_hash_invalid'):
        helper.validate_index(index, run_root=ROOT, kind='rf', expected_run_sha256='d' * 64)


@pytest.mark.parametrize('change, code', [('run-pin', 'run_receipt_hash_mismatch'),
    ('checkpoint-pin', 'selected_artifact_pins_mismatch'), ('config-pin', 'training_receipt_pin_mismatch')])
def test_index_artifacts_and_receipts_must_match_pinned_run(change, code):
    index = index_fixture()
    if change == 'run-pin':
        index['receipts'][0]['remote_sha256'] = '1' * 64
    elif change == 'checkpoint-pin':
        index['artifacts'][0]['sha256'] = '1' * 64
    else:
        index['receipts'][1]['remote_sha256'] = '1' * 64
    with pytest.raises(helper.PreservationError, match=code):
        helper.validate_index(index, run_root=ROOT, kind='rf', expected_run_sha256='d' * 64)


def test_secret_keys_unknown_fields_nonfinite_numbers_and_urls_are_sanitized():
    result = helper.sanitized_receipt({'schema_version': 'synthetic', 'password': 'NEVER',
        'unknown': 'NEVER', 'training': {'api_token': 'NEVER', 'access_token': 'NEVER',
            'wandb_api_key': 'NEVER', 'lr': .001, 'url': 'https://user:secret@example.test'},
        'recent_losses': [float('nan'), 1.25]})
    assert result == {'schema_version': 'synthetic', 'training': {'lr': .001, 'url': None},
                      'recent_losses': [None, 1.25]}
    assert 'NEVER' not in json.dumps(result)


@pytest.mark.parametrize('field,value', [('state', 'training'), ('optimizer_steps', 0),
    ('test_evaluation_count', 1), ('serving_eligible', True), ('approved_types', ['violation'])])
def test_incomplete_test_exposed_or_promoted_runs_are_rejected(field, value):
    run = copy.deepcopy(index_fixture()['receipts'][0]['content'])
    run[field] = value
    with pytest.raises(helper.PreservationError):
        helper.verify_success(run, 'rf')


def test_remote_script_compiles_without_future_import_order_bug(monkeypatch):
    captured = []

    def fake_command(command, *, input_text=None, **unused):
        captured.append((command, input_text))
        compile(input_text, '<synthetic-ssh-wrapper>', 'exec')
        return json.dumps(index_fixture())

    monkeypatch.setattr(helper, 'run_command', fake_command)
    args = Namespace(run_root=ROOT, kind='rf', expected_run_sha256='d' * 64, extra_json=[])
    assert helper.fetch_remote_index(args)['kind'] == 'rf'
    assert captured[0][0][-3:] == [helper.HOST_ALIAS, 'python3', '-']
    assert "'__file__':'/tmp/preserve_public_training_run.py'" in captured[0][1]


def reviewed_args(tmp_path, monkeypatch):
    root = tmp_path / 'private runs with spaces'
    root.mkdir()
    monkeypatch.setattr(helper, 'LOCAL_ROOT', root)
    index = index_fixture()
    payload = b'synthetic-weight'
    row = index['artifacts'][0]
    row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    index['receipts'][0]['content']['checkpoint_sha256'] = row['sha256']
    path = tmp_path / 'reviewed-index.json'
    path.write_text(json.dumps(index))
    args = Namespace(run_root=ROOT, kind='rf', expected_run_sha256='d' * 64, extra_json=[],
                     remote_index=path, expected_index_sha256=helper.file_sha256(path), destination=root / 'one')
    return args, payload


def test_insufficient_disk_reserve_prevents_scp(tmp_path, monkeypatch):
    args, payload = reviewed_args(tmp_path, monkeypatch)
    monkeypatch.setattr(helper.shutil, 'disk_usage', lambda _: SimpleNamespace(free=helper.RESERVE_BYTES + len(payload)))
    calls = []
    monkeypatch.setattr(helper, 'run_command', lambda *a, **kw: calls.append(a))
    with pytest.raises(helper.PreservationError, match='local_disk_reserve_would_be_violated'):
        helper.preserve(args)
    assert not calls and not args.destination.exists()


def test_reviewed_preservation_verifies_bytes_and_keeps_only_selected_payload(tmp_path, monkeypatch):
    args, payload = reviewed_args(tmp_path, monkeypatch)
    monkeypatch.setattr(helper.shutil, 'disk_usage', lambda _: SimpleNamespace(free=5 * 1024**3))
    incoming = []
    commands = []

    def fake_command(command, **unused):
        commands.append(command)
        if 'wslpath' in command:
            assert '\\' not in command[-1]
            assert command[-1] == Path(command[-1]).as_posix()
            assert 'private runs with spaces' in command[-1]
            assert command[:6] == ['wsl', '-d', 'Ubuntu', '--', 'wslpath', '-u']
            incoming.append(Path(command[-1]))
            return '/mnt/c/synthetic-incoming'
        assert 'scp' in command
        target = incoming[-1] if os.name == 'nt' else Path(command[-1])
        target.write_bytes(payload)
        return ''

    monkeypatch.setattr(helper, 'run_command', fake_command)
    result = helper.preserve(args)
    assert result['verified'] is True and result['remote_originals_retained'] is True
    assert result['minimum_reserve_bytes'] == 4 * 1024**3
    assert (args.destination / 'checkpoint_best_total.pth').read_bytes() == payload
    assert all('rm' not in command and 'images' not in ' '.join(command) for command in commands)
    assert set(p.suffix for p in args.destination.rglob('*') if p.is_file()) == {'.json', '.pth'}
    assert not list(args.destination.rglob('*.incoming'))


def test_transport_keeps_space_containing_path_in_one_argument_without_shell(monkeypatch):
    command = ['wsl', '-d', 'Ubuntu', '--', 'wslpath', '-u', '-a',
               'C:/private runs with spaces/a.pth']
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='/mnt/c/private runs with spaces/a.pth')

    monkeypatch.setattr(helper.subprocess, 'run', execute)
    assert helper.run_command(command).endswith('spaces/a.pth')
    assert calls[0][0] == command
    assert calls[0][1].get('shell', False) is False


def test_corrupt_scp_never_produces_success_receipt(tmp_path, monkeypatch):
    args, payload = reviewed_args(tmp_path, monkeypatch)
    monkeypatch.setattr(helper.shutil, 'disk_usage', lambda _: SimpleNamespace(free=5 * 1024**3))
    incoming = []

    def fake_command(command, **unused):
        if 'wslpath' in command:
            incoming.append(Path(command[-1]))
            return '/mnt/c/synthetic-incoming'
        (incoming[-1] if os.name == 'nt' else Path(command[-1])).write_bytes(b'corrupt')
        return ''

    monkeypatch.setattr(helper, 'run_command', fake_command)
    with pytest.raises(helper.PreservationError, match='copied_artifact_size_or_hash_mismatch'):
        helper.preserve(args)
    assert not (args.destination / 'preservation-report.json').exists()
    assert list(args.destination.rglob('*.incoming'))
