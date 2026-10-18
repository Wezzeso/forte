"""Receive a streamed tar of an exact public bundle into a fresh Linux folder.

The separately transferred dataset lock pins every payload. Symlinks, hardlink
headers, special files, extra payloads and path traversal are rejected. Failed
receipts retain the partial isolated destination for audit; nothing is deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

AUTHORIZED_ROOT = Path('/data/claude/training/public-20261018')


class TransferError(ValueError):
    """Finite validation codes, independent of Python optimization settings."""


def require(condition, code):
    if not condition:
        raise TransferError(code)


def receive(lock_path: Path, expected_lock_sha256: str, destination: Path,
            report_path: Path, input_stream) -> dict:
    raw = lock_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_lock_sha256, 'lock_hash_mismatch')
    lock = json.loads(raw)
    require(isinstance(lock, dict), 'lock_object_required')
    files = lock.get('files')
    require(lock.get('schema_version') == 'forte.specialists.public-dataset-lock.v1'
            and lock.get('state') == 'frozen' and lock.get('origin') == 'public', 'public_frozen_lock_required')
    require(isinstance(files, dict) and bool(files) and 'dataset-lock.json' not in files,
            'lock_files_invalid')
    expected = {**files, 'dataset-lock.json': expected_lock_sha256}
    for path in (destination, report_path):
        require(path.is_absolute() and path == path.resolve() and not path.is_symlink(),
                'canonical_absolute_path_required')
        require(path.is_relative_to(AUTHORIZED_ROOT) and path != AUTHORIZED_ROOT,
                'path_outside_authorized_root')
        require(not path.exists(), 'fresh_path_required')
    require(not report_path.is_relative_to(destination) and report_path.parent.is_dir(),
            'external_report_parent_required')
    for name, digest in expected.items():
        require(isinstance(name, str), 'payload_path_invalid')
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and name == rel.as_posix() and '\\' not in name
                and '\x00' not in name and all(part not in ('', '.', '..') for part in name.split('/')),
                'payload_path_invalid')
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest), 'payload_hash_invalid')
    destination.mkdir(parents=True, mode=0o700)
    received = set()
    total_bytes = 0
    with tarfile.open(fileobj=input_stream, mode='r|*') as stream:
        for member in stream:
            # Explicit --no-recursion file lists produce no directory headers.
            require(member.isfile() and not member.issym() and not member.islnk(),
                    'only_regular_payload_files_allowed')
            name = member.name.removeprefix('./')
            require(name in expected and name not in received, 'unexpected_or_duplicate_payload')
            require(0 <= member.size <= 2 * 1024**3, 'payload_size_exceeds_limit')
            path = destination / name
            require(path.resolve().is_relative_to(destination), 'payload_destination_escape')
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source = stream.extractfile(member)
            require(source is not None, 'payload_stream_missing')
            digest = hashlib.sha256()
            remaining = member.size
            # Exclusive creation also rejects a concurrent file/symlink replacement.
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as out:
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    require(bool(block), 'payload_truncated')
                    out.write(block)
                    digest.update(block)
                    remaining -= len(block)
            source.close()
            require(digest.hexdigest() == expected[name], 'payload_hash_mismatch')
            received.add(name)
            total_bytes += member.size
    require(received == set(expected), 'payload_set_incomplete')
    report = {'schema_version': 'forte.public-training-bundle-transfer.v1', 'verified': True,
              'dataset_lock_sha256': expected_lock_sha256, 'files': len(received),
              'bytes': total_bytes, 'destination': str(destination),
              'transport': 'tar stream; every destination payload SHA256 verified',
              'no_image_decoding': True, 'no_training_performed': True}
    with os.fdopen(os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--expected-lock-sha256',required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    report = receive(args.lock, args.expected_lock_sha256, args.destination, args.report, sys.stdin.buffer)
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
