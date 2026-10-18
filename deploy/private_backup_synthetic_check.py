"""Local-only restricted backup verification with synthetic PostgreSQL data."""
from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import tarfile
from uuid import uuid4

from private_backup import ARTIFACTS, digest, inspect_archive, restrict_directory, verify_local


def docker(*args, stdin=None, stdout=None):
    result = subprocess.run(["docker", "exec", *( ["-i"] if stdin else []), "forte-db-1", *args],
                            stdin=stdin, stdout=stdout or subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("synthetic_postgres_check_failed")
    return result.stdout or b""


def main():
    token = uuid4().hex[:12]
    root = Path.home() / "FortePrivateBackups" / ("synthetic-check-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + token)
    restrict_directory(root)
    source, target = "qa_backup_source_" + token, "qa_backup_target_" + token
    created = []
    try:
        for name in (source, target):
            docker("createdb", "-U", "forte", name)
            created.append(name)
        docker("psql", "-U", "forte", "-d", source, "-v", "ON_ERROR_STOP=1", "-Atqc",
               "CREATE TABLE backup_probe(id integer PRIMARY KEY,note text); INSERT INTO backup_probe VALUES(1,'synthetic');")
        with (root / "database.dump").open("xb") as output:
            docker("pg_dump", "-U", "forte", "-d", source, "-Fc", "--no-owner", "--no-privileges", stdout=output)
        with (root / "database.dump").open("rb") as stream:
            docker("pg_restore", "-U", "forte", "-d", target, "--exit-on-error", "--no-owner", "--no-privileges", stdin=stream)
        observed = docker("psql", "-U", "forte", "-d", target, "-Atqc",
                          "SELECT count(*)=1 AND min(note)='synthetic' FROM backup_probe").strip()
        assert observed == b"t"
        for name in ARTIFACTS[1:]:
            with tarfile.open(root / name, "w:gz") as archive:
                payload = b"synthetic backup fixture only"
                member = tarfile.TarInfo("synthetic-fixture.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        manifest = {"artifacts": {name: {"bytes": (root / name).stat().st_size,
                                         "sha256": digest(root / name)} for name in ARTIFACTS}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        assert verify_local(root)["verified"]
        with (root / "data.tar.gz").open("ab") as stream:
            stream.write(b"synthetic corruption")
        try:
            verify_local(root)
            raise AssertionError("corruption was not rejected")
        except RuntimeError:
            pass
        with tarfile.open(root / "unsafe-synthetic.tar.gz", "w:gz") as archive:
            archive.addfile(tarfile.TarInfo("../escape"))
        try:
            inspect_archive(root / "unsafe-synthetic.tar.gz")
            raise AssertionError("unsafe path was not rejected")
        except RuntimeError:
            pass
        print(json.dumps({"synthetic_only": True, "restricted_directory_verified": True,
                          "postgres_custom_dump_actual_restore_passed": True,
                          "archive_hash_verification_passed": True, "corruption_rejected": True,
                          "unsafe_path_rejected": True, "private_data_used": False,
                          "remote_service_changes": False}))
    finally:
        # Only these newly created, uniquely named local synthetic QA databases
        # are removed. Current application data is never modified or deleted.
        for name in created:
            docker("dropdb", "-U", "forte", name)


if __name__ == "__main__":
    main()
