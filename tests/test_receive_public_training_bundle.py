"""Synthetic streamed payloads only; no images, remote files or training."""
import hashlib
import io
import json
import tarfile

import pytest

from tools import receive_public_training_bundle as receiver


def bundle(tmp_path, monkeypatch, extra_files=None):
    monkeypatch.setattr(receiver, 'AUTHORIZED_ROOT', tmp_path)
    payload = b'synthetic-public-payload'
    files = {'train/sample.txt': hashlib.sha256(payload).hexdigest()}
    files.update(extra_files or {})
    lock = tmp_path / 'external-lock.json'
    lock.write_text(json.dumps({'schema_version': 'forte.specialists.public-dataset-lock.v1',
                               'state': 'frozen', 'origin': 'public', 'files': files}))
    raw = lock.read_bytes()
    members = [('train/sample.txt', payload, tarfile.REGTYPE),
               ('dataset-lock.json', raw, tarfile.REGTYPE)]
    return lock, hashlib.sha256(raw).hexdigest(), members


def stream(members):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode='w') as archive:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.REGTYPE:
                info.size = len(data)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = '/etc/passwd'
            archive.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    result.seek(0)
    return result


def test_stream_matches_exact_lock_and_writes_fresh_external_receipt(tmp_path, monkeypatch):
    lock, pin, members = bundle(tmp_path, monkeypatch)
    destination = tmp_path / 'received'
    report = tmp_path / 'receipt.json'
    result = receiver.receive(lock, pin, destination, report, stream(members))
    assert result['verified'] is True and result['files'] == 2
    assert (destination / 'train/sample.txt').read_bytes() == members[0][1]
    assert json.loads(report.read_text()) == result
    assert set(p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()) == {
        'train/sample.txt', 'dataset-lock.json'}


@pytest.mark.parametrize('kind', [tarfile.DIRTYPE, tarfile.SYMTYPE, tarfile.LNKTYPE,
                                tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_nonregular_headers_rejected(tmp_path, monkeypatch, kind):
    lock, pin, members = bundle(tmp_path, monkeypatch)
    with pytest.raises(receiver.TransferError, match='only_regular_payload_files_allowed'):
        receiver.receive(lock, pin, tmp_path / 'received', tmp_path / 'receipt.json',
                         stream([('train/sample.txt', b'', kind)] + members))
    assert not (tmp_path / 'receipt.json').exists()


@pytest.mark.parametrize('name', ['../escape.txt', '/absolute.txt', 'extra.txt', 'train/sample.txt'])
def test_extra_traversal_and_duplicate_payloads_rejected(tmp_path, monkeypatch, name):
    lock, pin, members = bundle(tmp_path, monkeypatch)
    with pytest.raises(receiver.TransferError, match='unexpected_or_duplicate_payload'):
        receiver.receive(lock, pin, tmp_path / 'received', tmp_path / 'receipt.json',
                         stream(members + [(name, b'extra', tarfile.REGTYPE)]))


def test_changed_payload_rejected_and_partial_directory_retained(tmp_path, monkeypatch):
    lock, pin, members = bundle(tmp_path, monkeypatch)
    members[0] = (members[0][0], b'changed', tarfile.REGTYPE)
    with pytest.raises(receiver.TransferError, match='payload_hash_mismatch'):
        receiver.receive(lock, pin, tmp_path / 'received', tmp_path / 'receipt.json', stream(members))
    assert (tmp_path / 'received').is_dir()
    assert not (tmp_path / 'receipt.json').exists()


def test_missing_payload_rejected(tmp_path, monkeypatch):
    lock, pin, members = bundle(tmp_path, monkeypatch)
    with pytest.raises(receiver.TransferError, match='payload_set_incomplete'):
        receiver.receive(lock, pin, tmp_path / 'received', tmp_path / 'receipt.json', stream(members[:1]))


def test_report_cannot_modify_outside_authorized_root_or_bundle(tmp_path, monkeypatch):
    lock, pin, members = bundle(tmp_path, monkeypatch)
    with pytest.raises(receiver.TransferError, match='path_outside_authorized_root'):
        receiver.receive(lock, pin, tmp_path / 'received', tmp_path.parent / 'escape-receipt.json', stream(members))
    with pytest.raises(receiver.TransferError, match='external_report_parent_required'):
        receiver.receive(lock, pin, tmp_path / 'received', tmp_path / 'received/receipt.json', stream(members))


def test_existing_receipt_is_preserved(tmp_path, monkeypatch):
    lock, pin, members = bundle(tmp_path, monkeypatch)
    report = tmp_path / 'receipt.json'
    report.write_text('preserve')
    with pytest.raises(receiver.TransferError, match='fresh_path_required'):
        receiver.receive(lock, pin, tmp_path / 'received', report, stream(members))
    assert report.read_text() == 'preserve'

