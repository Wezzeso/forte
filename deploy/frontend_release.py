"""Guarded frontend-only release helper; settings live in sibling release-config.json.

Run validate-local anywhere. Run prepare/activate/rollback only on the configured host.
This helper never uses SSH or uploads files itself.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import uuid


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True,
                            timeout=kwargs.pop('timeout', 60), **kwargs)
    require(result.returncode == 0,
            'command failed: ' + args[0] + ' (exit ' + str(result.returncode) + ')')
    return result.stdout.strip()


def read_regular(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
            'expected regular file: ' + str(path))
    require(not getattr(info, 'st_file_attributes', 0) &
            getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0),
            'reparse point rejected: ' + str(path))
    return path.read_bytes()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fingerprint(container):
    config = container['Config']
    fields = {key: config.get(key) for key in
              ['Env', 'Cmd', 'Entrypoint', 'User', 'WorkingDir', 'Healthcheck']}
    # Compose can reorder environment entries while preserving every value.
    fields['Env'] = sorted(fields['Env'] or [])
    fields['mounts'] = sorted((m['Destination'], m['Source'], m['RW'])
                              for m in container['Mounts'])
    host = container['HostConfig']
    fields['host'] = {key: host.get(key) for key in
                      ['ReadonlyRootfs', 'NetworkMode', 'CapDrop', 'CapAdd',
                       'SecurityOpt', 'Tmpfs', 'PortBindings', 'RestartPolicy',
                       'DeviceRequests', 'Memory', 'NanoCpus', 'Init']}
    return sha(json.dumps(fields, sort_keys=True).encode())


class FrontendRelease:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config_bytes = read_regular(self.config_path)
        self.config = json.loads(self.config_bytes)
        c = self.config
        require(c['schema_version'] == 1, 'unsupported release config schema')
        for key in ['archive_sha256', 'manifest_sha256', 'profile_sha256']:
            require(re.fullmatch('[0-9a-f]{64}', c[key]), 'invalid ' + key)
        for key in ['parent_api_image', 'expected_worker_image']:
            require(re.fullmatch('sha256:[0-9a-f]{64}', c[key]), 'invalid ' + key)
        for key in ['image_tag', 'parent_image_tag']:
            require(re.fullmatch('[a-z0-9][a-z0-9:._/-]*', c[key]), 'invalid ' + key)
        root = PurePosixPath(c['deployment_root'])
        release = PurePosixPath(c['release_directory'])
        require(root.is_absolute() and release.is_absolute() and
                '..' not in root.parts and '..' not in release.parts and
                release.is_relative_to(root / 'releases') and release != root / 'releases',
                'invalid deployment/release path')
        for name in c['compose_files'] + [c['env_file'], c['canonical_wrapper']]:
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts,
                    'deployment path must stay below deployment root')
        for key in ['archive_filename', 'manifest_filename']:
            require(PurePosixPath(c[key]).name == c[key], 'artifact must be a sibling file')
        require(c['api_container'] in c['guarded_containers'] and
                c['worker_container'] in c['guarded_containers'] and
                len(set(c['guarded_containers'])) == len(c['guarded_containers']),
                'invalid guarded container list')
        for key in ['readiness_url', 'application_url']:
            url = urllib.parse.urlsplit(c[key])
            require(url.scheme == 'http' and url.hostname == '127.0.0.1' and not url.username,
                    'health/assets URLs must be host loopback HTTP')
        require(c['expected_canonical_wrapper_text'].startswith('#!/usr/bin/env bash\n'),
                'expected old canonical wrapper bytes are required')
        self.root = Path(c['deployment_root'])
        self.release = Path(c['release_directory'])
        self.artifacts = self.config_path.parent
        self.api = c['api_container']
        self.services = c['guarded_containers']
        self.backend_services = [name for name in self.services if name != self.api]
        self.canonical = self.root / c['canonical_wrapper']
        self.old_dispatch = c['expected_canonical_wrapper_text'].encode()
        self.new_dispatch = ('#!/usr/bin/env bash\nset -euo pipefail\nexec ' +
                             shlex.quote(str(self.release / 'compose-active.sh')) +
                             ' "$@"\n').encode()

    def compose(self, extra=None):
        c = self.config
        files = [self.root / name for name in c['compose_files']]
        if extra is not None:
            files.append(extra)
        return ['docker', 'compose', '--env-file', str(self.root / c['env_file'])] + [
            item for path in files for item in ['-f', str(path)]]

    def inspect(self, name):
        return json.loads(run(['docker', 'inspect', name]))[0]

    def snapshot(self):
        result = {}
        for name in self.services:
            container = self.inspect(name)
            require(container['State']['Running'], 'container is not running: ' + name)
            result[name] = {'id': container['Id'], 'image': container['Image'],
                            'started': container['State']['StartedAt'],
                            'fingerprint': fingerprint(container)}
        return result

    def unchanged(self, before, after, include_api=True):
        for name in self.services if include_api else self.backend_services:
            require(before[name] == after[name], 'concurrent runtime change: ' + name)

    def profile(self):
        check = ('import pathlib,hashlib;print(hashlib.sha256(pathlib.Path(' +
                 repr(self.config['profile_path']) + ').read_bytes()).hexdigest())')
        return run(['docker', 'exec', self.api, 'python', '-c', check])

    def pending(self):
        # In Forte, queued and active jobs both have status=processing.
        c = self.config
        value = run(['docker', 'exec', c['database_container'], 'psql', '-U',
                     c['database_user'], '-d', c['database_name'], '-Atc',
                     "SELECT count(*) FROM jobs WHERE status='processing'"])
        require(value.isdecimal(), 'unexpected job count result')
        return int(value)

    def ready(self):
        with urllib.request.urlopen(self.config['readiness_url'], timeout=8) as response:
            data = json.load(response)
            components = data.get('components', {})
            return response.status == 200 and data.get('status') == 'ready' and bool(
                components) and all(components.values())

    def wait_ready(self):
        deadline = time.monotonic() + self.config.get('readiness_timeout_seconds', 90)
        while time.monotonic() < deadline:
            try:
                if self.ready():
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError('readiness timeout')

    def runtime_guard(self, before=None):
        current = self.snapshot()
        if before is not None:
            self.unchanged(before, current)
        c = self.config
        require(current[self.api]['image'] == c['parent_api_image'], 'active API base changed')
        require(current[c['worker_container']]['image'] == c['expected_worker_image'],
                'worker base changed')
        require(self.profile() == c['profile_sha256'] and self.ready(),
                'profile/readiness guard failed')
        require(self.pending() == 0, 'queued or active jobs; retry when idle')
        return current

    def canonical_guard(self, expected):
        require(read_regular(self.canonical) == expected,
                'canonical wrapper bytes changed; inspect before deployment')

    def frozen_files(self):
        c = self.config
        paths = c['compose_files'] + [c['env_file']]
        return {name: sha(read_regular(self.root / name)) for name in paths}

    def load_artifact(self):
        c = self.config
        manifest_bytes = read_regular(self.artifacts / c['manifest_filename'])
        require(sha(manifest_bytes) == c['manifest_sha256'], 'manifest hash differs')
        manifest = json.loads(manifest_bytes)
        archive = self.artifacts / c['archive_filename']
        require(manifest['archive']['sha256'] == c['archive_sha256'] and
                sha(read_regular(archive)) == c['archive_sha256'], 'archive hash differs')
        require(manifest['archive']['bytes'] == archive.stat().st_size,
                'archive byte length differs')
        expected = {}
        for entry in manifest['files']:
            path = PurePosixPath(entry['path'])
            require(len(path.parts) > 1 and not path.is_absolute() and '..' not in path.parts and
                    path.parts[0] == 'web-dist' and path.as_posix() == entry['path'],
                    'unsafe artifact path')
            require(entry['path'] not in expected and isinstance(entry['bytes'], int) and
                    entry['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}', entry['sha256']),
                    'invalid or duplicate manifest file')
            expected[entry['path']] = entry
        require('web-dist/index.html' in expected and len(expected) == manifest['totalFiles'] and
                sum(item['bytes'] for item in expected.values()) == manifest['totalBytes'],
                'manifest totals or index missing')
        payload = {}
        with tarfile.open(archive, 'r:gz') as archive_file:
            for member in archive_file:
                require(member.isfile() and member.name in expected and member.name not in payload,
                        'unexpected or non-regular archive member')
                data = archive_file.extractfile(member).read()
                entry = expected[member.name]
                require(len(data) == entry['bytes'] and sha(data) == entry['sha256'],
                        'archive file differs: ' + member.name)
                payload[member.name] = data
        require(set(payload) == set(expected), 'archive files missing')
        return manifest, payload

    def validate_local(self):
        manifest, _ = self.load_artifact()
        result = {'stage': 'artifact_verified', 'archive_sha256': self.config['archive_sha256'],
                  'file_count': manifest['totalFiles'], 'file_bytes': manifest['totalBytes']}
        print(json.dumps(result))
        return result

    @contextmanager
    def locked_host(self):
        require(os.name == 'posix', 'host deployment actions require the configured Linux host')
        require(self.release.resolve() == self.artifacts.resolve(),
                'config/artifacts must be in the configured host release directory')
        import fcntl
        lock_path = self.root / '.frontend-release.lock'
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(descriptor)

    def replace_dispatch(self, expected, replacement, mode=0o755):
        self.canonical_guard(expected)
        temporary = self.canonical.with_name('.' + self.canonical.name + '.' + uuid.uuid4().hex)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(replacement)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), mode)
            self.canonical_guard(expected)
            os.replace(temporary, self.canonical)
            directory = os.open(self.canonical.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self.canonical_guard(replacement)
        finally:
            if temporary.exists():
                temporary.unlink()

    def prepared(self):
        report = json.loads(read_regular(self.release / 'prepared.json'))
        require(report['config_sha256'] == sha(self.config_bytes), 'release config changed')
        require(report['frozen_files'] == self.frozen_files(), 'frozen Compose/env files changed')
        require(read_regular(self.release / 'compose.frontend.json') ==
                (json.dumps({'services': {self.config['api_service']: {'image': report['image']}}},
                            indent=2) + '\n').encode(), 'frontend override changed')
        require(read_regular(self.release / 'compose.rollback.json') ==
                (json.dumps({'services': {self.config['api_service']: {
                    'image': self.config['parent_api_image']}}}, indent=2) + '\n').encode(),
                'rollback override changed')
        require(read_regular(self.release / 'compose-active.sh') == self.wrapper('compose.frontend.json'),
                'active release wrapper changed')
        require(read_regular(self.release / 'rollback.sh') == self.rollback_wrapper(),
                'rollback release wrapper changed')
        return report

    def wrapper(self, override):
        return ('#!/usr/bin/env bash\nset -euo pipefail\nexec ' +
                shlex.join(self.compose(self.release / override)) + ' "$@"\n').encode()

    def rollback_wrapper(self):
        command = ['python3', str(self.release / 'frontend_release.py'), 'rollback',
                   '--config', str(self.release / 'release-config.json')]
        return ('#!/usr/bin/env bash\nset -euo pipefail\nexec ' +
                shlex.join(command) + ' "$@"\n').encode()

    def prepare(self):
        with self.locked_host():
            before = self.runtime_guard()
            self.canonical_guard(self.old_dispatch)
            frozen = self.frozen_files()
            manifest, payload = self.load_artifact()
            context = self.release / 'context'
            context.mkdir(exist_ok=False)
            for name, data in payload.items():
                destination = context.joinpath(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                destination.write_bytes(data)
                destination.chmod(0o644)
            c = self.config
            run(['docker', 'tag', c['parent_api_image'], c['parent_image_tag']])
            (context / 'Dockerfile').write_text(
                'FROM ' + c['parent_image_tag'] + '\n' +
                'LABEL forte.frontend.archive.sha256="' + c['archive_sha256'] + '"\n' +
                'LABEL forte.frontend.parent.image="' + c['parent_api_image'] + '"\n' +
                'COPY --chown=10001:10001 web-dist/ /app/web/dist/\n', encoding='utf-8')
            environment = os.environ.copy()
            environment['DOCKER_BUILDKIT'] = '0'
            output = run(['docker', 'build', '--pull=false', '--network=none',
                          '--cpu-quota=100000', '--memory=536870912', '-t', c['image_tag'],
                          str(context)], env=environment, timeout=180)
            (self.release / 'build.log').write_text(output, encoding='utf-8')
            image = json.loads(run(['docker', 'image', 'inspect', c['image_tag']]))[0]
            parent = json.loads(run(['docker', 'image', 'inspect', c['parent_api_image']]))[0]
            require(image['Config']['Labels']['forte.frontend.archive.sha256'] == c['archive_sha256'],
                    'child image label differs')
            require(image['RootFS']['Layers'][:-1] == parent['RootFS']['Layers'],
                    'child image is not exactly one frontend layer above the current API image')
            check = ("import sys,json,pathlib,hashlib;m=json.load(sys.stdin);r=pathlib.Path('/app/web/dist');"
                     "bad=[f['path'] for f in m['files'] if hashlib.sha256((r/pathlib.PurePosixPath(f['path']).relative_to('web-dist')).read_bytes()).hexdigest()!=f['sha256']];"
                     "sys.exit('frontend mismatch: '+repr(bad)) if bad else print('verified',len(m['files']),'frontend files')")
            validation = run(['docker', 'run', '--rm', '-i', '--network', 'none', '--read-only',
                              '--cpus', '1', '--memory', '268435456', '--cap-drop', 'ALL',
                              '--security-opt', 'no-new-privileges:true', '--entrypoint', 'python',
                              image['Id'], '-c', check], input=json.dumps(manifest))
            self.runtime_guard(before)
            self.canonical_guard(self.old_dispatch)
            require(frozen == self.frozen_files(), 'frozen Compose/env files changed during build')
            for name, image_id in [('compose.frontend.json', image['Id']),
                                   ('compose.rollback.json', c['parent_api_image'])]:
                write_json(self.release / name, {'services': {c['api_service']: {'image': image_id}}})
            for name, data in [('compose-active.sh', self.wrapper('compose.frontend.json')),
                               ('rollback.sh', self.rollback_wrapper())]:
                path = self.release / name
                path.write_bytes(data)
                path.chmod(0o755)
            report = {'stage': 'prepared', 'config_sha256': sha(self.config_bytes),
                      'archive_sha256': c['archive_sha256'], 'image': image['Id'],
                      'base_image': c['parent_api_image'], 'file_count': len(payload),
                      'validation': validation, 'before': before, 'frozen_files': frozen,
                      'profile_sha256': c['profile_sha256'],
                      'canonical_wrapper_mode': stat.S_IMODE(self.canonical.stat().st_mode)}
            write_json(self.release / 'prepared.json', report)
            print(json.dumps({key: value for key, value in report.items()
                              if key not in ['before', 'frozen_files']}))

    def verify_after(self, prepared, expected_image):
        self.wait_ready()
        after = self.snapshot()
        self.unchanged(prepared['before'], after, include_api=False)
        require(after[self.api]['image'] == expected_image, 'API image differs')
        require(after[self.api]['fingerprint'] == prepared['before'][self.api]['fingerprint'],
                'API configuration changed')
        require(self.profile() == self.config['profile_sha256'], 'profile changed')
        require(self.pending() == 0, 'queued or active jobs after cutover')
        return after

    def serve_check(self, manifest):
        count = 0
        for entry in manifest['files']:
            relative = PurePosixPath(entry['path']).relative_to('web-dist').as_posix()
            suffix = '' if relative == 'index.html' else urllib.parse.quote(relative, safe='/')
            url = urllib.parse.urljoin(self.config['application_url'], suffix)
            with urllib.request.urlopen(url, timeout=10) as response:
                require(response.status == 200 and sha(response.read()) == entry['sha256'],
                        'served file differs: ' + relative)
            count += 1
        return count

    def restore(self, prepared):
        # Roll back API even when activation failed halfway through Compose replacement.
        runtime_error = pointer_error = None
        restored = None
        try:
            output = run(self.compose(self.release / 'compose.rollback.json') +
                         ['up', '-d', '--no-deps', '--no-build', '--timeout', '60',
                          self.config['api_service']], timeout=150)
            (self.release / 'rollback.log').write_text(output, encoding='utf-8')
            restored = self.verify_after(prepared, self.config['parent_api_image'])
        except Exception as error:
            runtime_error = error
        # Restore the dispatcher independently even if Compose or readiness verification failed.
        # Never overwrite a dispatcher introduced by another operator.
        try:
            current = read_regular(self.canonical)
            if current == self.new_dispatch:
                self.replace_dispatch(self.new_dispatch, self.old_dispatch,
                                      prepared['canonical_wrapper_mode'])
            else:
                require(current == self.old_dispatch,
                        'canonical wrapper changed concurrently; old pointer not overwritten')
        except Exception as error:
            pointer_error = error
        require(runtime_error is None and pointer_error is None,
                'rollback verification failed; runtime=' + str(runtime_error) +
                '; canonical=' + str(pointer_error))
        self.canonical_guard(self.old_dispatch)
        require(prepared['frozen_files'] == self.frozen_files(), 'frozen Compose/env files changed')
        result = {'stage': 'rolled_back', 'image': self.config['parent_api_image'],
                  'ready': True, 'profile_sha256': self.config['profile_sha256'],
                  'api_configuration_unchanged': True, 'unchanged_services': self.backend_services,
                  'canonical_wrapper_restored': True, 'after': restored}
        write_json(self.release / 'rollback-verified.json', result)
        return result

    def activate(self):
        with self.locked_host():
            prepared = self.prepared()
            manifest, _ = self.load_artifact()
            self.runtime_guard(prepared['before'])
            self.canonical_guard(self.old_dispatch)
            deployed = False
            try:
                deployed = True
                # Pre/post idle snapshots plus the existing Uvicorn 60-second drain.
                output = run(self.compose(self.release / 'compose.frontend.json') +
                             ['up', '-d', '--no-deps', '--no-build', '--timeout', '60',
                              self.config['api_service']], timeout=150)
                (self.release / 'activate.log').write_text(output, encoding='utf-8')
                after = self.verify_after(prepared, prepared['image'])
                count = self.serve_check(manifest)
                require(prepared['frozen_files'] == self.frozen_files(),
                        'frozen Compose/env files changed during deployment')
                # Repoint only after readiness, unchanged-runtime, profile and exact asset checks.
                self.replace_dispatch(self.old_dispatch, self.new_dispatch)
                result = {'stage': 'active', 'image': prepared['image'],
                          'archive_sha256': self.config['archive_sha256'], 'ready': True,
                          'unchanged_services': self.backend_services,
                          'api_configuration_unchanged': True,
                          'profile_sha256': self.config['profile_sha256'],
                          'served_files_verified': count, 'after': after,
                          'active_wrapper': str(self.release / 'compose-active.sh'),
                          'canonical_wrapper': str(self.canonical),
                          'rollback_wrapper': str(self.release / 'rollback.sh')}
                write_json(self.release / 'deployed.json', result)
                print(json.dumps({key: value for key, value in result.items() if key != 'after'}))
            except Exception as original:
                if deployed:
                    try:
                        result = self.restore(prepared)
                        result['error'] = str(original)
                        write_json(self.release / 'failed.json', result)
                    except Exception as rollback_error:
                        write_json(self.release / 'failed.json', {
                            'stage': 'rollback_failed', 'error': str(original),
                            'rollback_error': str(rollback_error)})
                        raise RuntimeError('deployment failed and rollback verification failed') from rollback_error
                raise

    def rollback(self):
        with self.locked_host():
            prepared = self.prepared()
            current = self.snapshot()
            self.unchanged(prepared['before'], current, include_api=False)
            require(current[self.api]['image'] == prepared['image'] and
                    current[self.api]['fingerprint'] == prepared['before'][self.api]['fingerprint'],
                    'active API no longer matches this release')
            require(self.profile() == self.config['profile_sha256'] and self.ready() and
                    self.pending() == 0, 'rollback profile/readiness/idle guard failed')
            self.canonical_guard(self.new_dispatch)
            result = self.restore(prepared)
            print(json.dumps({key: value for key, value in result.items() if key != 'after'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate-local', 'prepare', 'activate', 'rollback'])
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('release-config.json'))
    args = parser.parse_args()
    publisher = FrontendRelease(args.config)
    getattr(publisher, args.action.replace('-', '_'))()


if __name__ == '__main__':
    main()
