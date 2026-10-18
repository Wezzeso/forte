"""Safe binary/package license inventory; never inspects media or job data."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import ssl
import subprocess


def output(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    if result.returncode:
        return {"available": False}
    return {"available": True, "text": result.stdout + result.stderr}


def distribution_licenses(name):
    dist = importlib.metadata.distribution(name)
    records = []
    for entry in dist.files or ():
        if any(word in entry.name.lower() for word in ("license", "copying", "notice")):
            path = dist.locate_file(entry)
            if path.is_file():
                records.append({"distribution_relative_path": str(entry), "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"version": dist.version, "license_files": records}


def main():
    import psycopg.pq
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Observed CPU image package/build/license metadata only; no legal conclusion",
        "ffmpeg_version_build": output(["ffmpeg", "-version"]),
        "ffprobe_version_build": output(["ffprobe", "-version"]),
        "ffmpeg_license_banner": output(["ffmpeg", "-L"]),
        "system_package_versions": {package: output(["dpkg-query", "-W", "-f=${Package} ${Version}\n", package])
                                    for package in ("ffmpeg", "libpq5", "libssl3t64")},
        "python_ssl_runtime": ssl.OPENSSL_VERSION,
        "psycopg_pq_implementation": psycopg.pq.__impl__,
        "psycopg_pq_runtime_version": psycopg.pq.version(),
        "python_distributions": {name: distribution_licenses(name) for name in ("psycopg", "psycopg-binary", "av")},
        "system_license_files": [],
        "media_or_job_data_accessed": False,
    }
    for package in ("ffmpeg", "libpq5", "libssl3t64"):
        path = Path("/usr/share/doc") / package / "copyright"
        if path.is_file():
            payload["system_license_files"].append({"path": str(path), "bytes": path.stat().st_size,
                                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
