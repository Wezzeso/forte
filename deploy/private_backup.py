"""Restricted, operator-run backup. Never prints private content or credentials.

Run `check` during normal service. Run `create --quiesce-idle` only after the
exclusive benchmark/review window is released. It briefly stops API/worker,
requires no processing jobs, and always attempts to resume those services.
No GPU/model call, account switch, rental, deletion or provider action occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tarfile
from uuid import uuid4

REMOTE_ROOT = "/home/ubuntu/forte"
ARTIFACTS = ("database.dump", "data.tar.gz", "runtime-config.tar.gz", "source.tar.gz")
OPTIONAL_ARTIFACTS = {"experiments.tar.gz"}
EXPERIMENT_DENIED_SUFFIXES = (".mov", ".mp4", ".mkv", ".avi", ".webm", ".tar", ".tar.gz", ".tgz", ".zip", ".7z")


def ssh_prefix(host: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
        raise ValueError("invalid_ssh_alias")
    prefix = ["wsl", "-d", "Ubuntu", "--"] if os.name == "nt" else []
    return prefix + ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host]


def command(host: str, arguments: list[str], *, output=None, input_file=None,
            timeout: int = 1200) -> bytes:
    result = subprocess.run(ssh_prefix(host) + [shlex.join(arguments)],
                            stdin=input_file, stdout=output or subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode:
        # Provider/DB/tar errors may include private names. Keep stderr local to
        # the process; never forward it to a tool, log, or exception message.
        raise RuntimeError("remote_backup_command_failed")
    return result.stdout or b""


def active_jobs(host: str) -> int:
    raw = command(host, ["docker", "exec", "forte-db-1", "psql", "-U", "forte",
                         "-d", "forte", "-Atqc",
                         "SELECT count(*) FROM jobs WHERE status IN ('queued','processing')"])
    return int(raw.strip())


def isolated_running(host: str) -> bool:
    raw = command(host, ["docker", "ps", "--format", '{{.Label "com.docker.compose.project"}}'])
    return bool({line.strip() for line in raw.decode().splitlines()} & {"forte-ml", "forte-teacher"})


def restrict_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    if os.name == "nt":
        script = """
$ErrorActionPreference='Stop'
$p=$env:FORTE_PRIVATE_BACKUP_DIRECTORY
$sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl=New-Object System.Security.AccessControl.DirectorySecurity
$acl.SetAccessRuleProtection($true,$false)
$acl.SetOwner($sid)
$inherit=[System.Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit'
$prop=[System.Security.AccessControl.PropagationFlags]::None
foreach($who in @($sid,[System.Security.Principal.SecurityIdentifier]'S-1-5-18')) {
  $rule=New-Object System.Security.AccessControl.FileSystemAccessRule($who,'FullControl',$inherit,$prop,'Allow')
  $acl.AddAccessRule($rule)
}
Set-Acl -LiteralPath $p -AclObject $acl
$observed=Get-Acl -LiteralPath $p
if(-not $observed.AreAccessRulesProtected){throw 'ACL is not protected'}
foreach($rule in $observed.Access){
  $identity=$rule.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
  if($identity -notin @($sid.Value,'S-1-5-18')){throw 'Unexpected ACL principal'}
}
"""
        env = dict(os.environ, FORTE_PRIVATE_BACKUP_DIRECTORY=str(path))
        # Avoid inheriting another PowerShell major version's module path.
        env.pop("PSModulePath", None)
        shell = shutil.which("pwsh") or "powershell"
        result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", script],
                                env=env, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("restricted_backup_acl_failed")
    else:
        path.chmod(0o700)
        if path.stat().st_mode & 0o077:
            raise RuntimeError("restricted_backup_mode_failed")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_archive(path: Path, *, reject_experiment_media: bool = False) -> int:
    count = 0
    with tarfile.open(path, "r:gz") as archive:
        for item in archive:
            relative = PurePosixPath(item.name)
            if relative.is_absolute() or ".." in relative.parts or item.isdev() or item.issym() or item.islnk():
                raise RuntimeError("archive_contains_unsafe_entry")
            if reject_experiment_media and item.isfile() and item.name.lower().endswith(EXPERIMENT_DENIED_SUFFIXES):
                raise RuntimeError("experiment_archive_contains_media_or_nested_archive")
            if item.isfile():
                stream = archive.extractfile(item)
                if stream is None:
                    raise RuntimeError("archive_member_unreadable")
                for _ in iter(lambda: stream.read(1024 * 1024), b""):
                    pass
                count += 1
    return count


def verify_local(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    names = set(manifest["artifacts"])
    if not set(ARTIFACTS).issubset(names) or names - set(ARTIFACTS) - OPTIONAL_ARTIFACTS:
        raise RuntimeError("backup_artifact_names_invalid")
    for name in sorted(names):
        source = path / name
        expected = manifest["artifacts"][name]
        if source.stat().st_size != expected["bytes"] or digest(source) != expected["sha256"]:
            raise RuntimeError("backup_integrity_mismatch")
        if name.endswith(".tar.gz"):
            inspect_archive(source, reject_experiment_media=name == "experiments.tar.gz")
    with (path / "database.dump").open("rb") as stream:
        if stream.read(5) != b"PGDMP":
            raise RuntimeError("postgres_custom_dump_header_invalid")
    return {"verified": True, "artifact_count": len(names),
            "bytes": sum(item["bytes"] for item in manifest["artifacts"].values()),
            "private_content_printed": False}


def create_backup(host: str, destination: Path, *, include_experiments: bool = False) -> dict:
    workspace = Path(__file__).resolve().parents[1]
    if destination == workspace or workspace in destination.parents:
        raise ValueError("private_backup_must_be_outside_source_workspace")
    if active_jobs(host):
        raise RuntimeError("processing_jobs_present_wait_for_terminal_window")
    experiment_source = None
    teacher_paths: list[str] = []
    if include_experiments:
        if isolated_running(host):
            raise RuntimeError("isolated_experiment_still_running")
        # Parse only the public code-directory setting; never print runtime.env.
        script = "from pathlib import Path; p=Path('/data/forte-ml/private/runtime.env'); print(next(x.split('=',1)[1] for x in p.read_text().splitlines() if x.startswith('FORTE_ML_SOURCE=')))"
        experiment_source = command(host, ["python3", "-c", script]).decode().strip()
        if not re.fullmatch(r"/home/ubuntu/forte-ml-code-[0-9a-f]{12}", experiment_source):
            raise RuntimeError("experiment_source_directory_not_verified")
        script = """
import json
from pathlib import Path
p=Path('/data/forte-teacher/runtime.env')
keys={'FORTE_TEACHER_CODE','FORTE_TEACHER_PRIVATE'}
values={}
if p.is_file():
 values={line.split('=',1)[0]:line.split('=',1)[1] for line in p.read_text().splitlines() if line.split('=',1)[0] in keys}
print(json.dumps(values))
"""
        teacher = json.loads(command(host, ["python3", "-c", script]))
        if teacher:
            code, private = teacher.get("FORTE_TEACHER_CODE", ""), teacher.get("FORTE_TEACHER_PRIVATE", "")
            if not re.fullmatch(r"/data/forte-teacher/code-[0-9a-f]{12}", code) or not re.fullmatch(r"/data/forte-teacher/private-v[0-9]+/metadata", private):
                raise RuntimeError("teacher_snapshot_paths_not_verified")
            teacher_paths = ["data/forte-teacher/runtime.env", "data/forte-teacher/compose.teacher.yaml",
                             "data/forte-teacher/w2", code.lstrip("/"), private.lstrip("/")]
    mounts = json.loads(command(host, ["docker", "inspect", "--format", "{{json .Mounts}}", "forte-api-1"]))
    volumes = [mount["Name"] for mount in mounts
               if mount.get("Type") == "volume" and mount.get("Destination") == "/var/lib/forte"]
    if len(volumes) != 1 or not re.fullmatch(r"[A-Za-z0-9_.-]+", volumes[0]):
        raise RuntimeError("application_data_volume_not_verified")
    image = command(host, ["docker", "inspect", "--format", "{{.Image}}", "forte-api-1"]).decode().strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise RuntimeError("application_image_not_verified")
    restrict_directory(destination)
    compose = ["docker", "compose", "--project-directory", REMOTE_ROOT + "/deploy",
               "--env-file", REMOTE_ROOT + "/deploy/.env", "-f", REMOTE_ROOT + "/deploy/compose.yaml"]
    # Public source packages are immutable. Give each snapshot its own remote
    # path rather than overwriting an earlier rollback/source archive.
    source_snapshot = "/home/ubuntu/forte-backup-source-" + uuid4().hex + ".tar.gz"
    stopped = False
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_host_alias": host, "app_image": image,
                "data_restore_target": "/var/lib/forte", "postgres_major": 18,
                "snapshot_consistent": False, "artifacts": {}}
    try:
        stopped = True
        command(host, compose + ["stop", "api"])
        if active_jobs(host):
            raise RuntimeError("new_processing_job_detected_backup_deferred")
        command(host, compose + ["stop", "worker"])
        if active_jobs(host):
            raise RuntimeError("active_job_detected_after_worker_stop_backup_deferred")
        command(host, ["python3", REMOTE_ROOT + "/deploy/package_code.py", "--output", source_snapshot])
        steps = {
            "database.dump": ["docker", "exec", "forte-db-1", "pg_dump", "-U", "forte", "-d", "forte", "-Fc", "--no-owner", "--no-privileges"],
            "data.tar.gz": ["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--entrypoint", "tar", "--mount", f"type=volume,source={volumes[0]},target=/data,readonly", image, "-C", "/data", "-czf", "-", "."],
            "runtime-config.tar.gz": ["tar", "-C", REMOTE_ROOT, "-czf", "-", "deploy/.env", "deploy/secrets", "configs", "models/manifest.json"],
            "source.tar.gz": ["cat", source_snapshot],
        }
        if include_experiments:
            exclusions = ["--exclude=*" + suffix for suffix in EXPERIMENT_DENIED_SUFFIXES]
            steps["experiments.tar.gz"] = ["tar", "-C", "/", "--ignore-case", *exclusions, "-czf", "-",
                "data/forte-ml/private", "data/forte-ml/runs", "data/forte-feature-cache",
                experiment_source.lstrip("/"), *teacher_paths]
            manifest["isolated_experiments_included"] = True
            manifest["experiment_public_encoder_weights_included"] = False
            manifest["experiment_original_movs_included"] = False
            manifest["teacher_private_outputs_included"] = bool(teacher_paths)
        for name, arguments in steps.items():
            if name == "experiments.tar.gz":
                if isolated_running(host):
                    raise RuntimeError("isolated_experiment_started_backup_deferred")
            path = destination / name
            with path.open("xb") as stream:
                if os.name != "nt":
                    path.chmod(0o600)
                command(host, arguments, output=stream)
            files = inspect_archive(path, reject_experiment_media=name == "experiments.tar.gz") if name.endswith(".tar.gz") else None
            manifest["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": digest(path), "file_count": files}
        with (destination / "database.dump").open("rb") as stream:
            command(host, ["docker", "exec", "-i", "forte-db-1", "pg_restore", "--list"],
                    input_file=stream, output=subprocess.DEVNULL)
        manifest["snapshot_consistent"] = True
        manifest["postgres_toc_verified"] = True
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return verify_local(destination)
    finally:
        if stopped:
            command(host, compose + ["up", "-d", "--no-deps", "api", "worker"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "create", "verify"))
    parser.add_argument("--host", default="forte-a100-smoke-20261017")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--quiesce-idle", action="store_true")
    parser.add_argument("--include-experiments", action="store_true",
                        help="After all isolated runs finish, include private W1 metadata, feature caches and checkpoints")
    args = parser.parse_args()
    try:
        if args.action == "check":
            result = {"check_only": True, "processing_jobs": active_jobs(args.host), "service_changes": False}
        elif args.action == "verify":
            if not args.destination:
                parser.error("verify requires --destination")
            result = verify_local(args.destination.resolve())
        else:
            if not args.quiesce_idle:
                parser.error("create requires --quiesce-idle after the benchmark/review window is released")
            destination = args.destination or Path.home() / "FortePrivateBackups" / datetime.now().strftime("%Y%m%d-%H%M%S")
            result = create_backup(args.host, destination.resolve(), include_experiments=args.include_experiments)
        print(json.dumps(result))
        return 0
    except Exception as error:
        print(json.dumps({"backup_failed": True, "error_type": type(error).__name__, "private_content_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
