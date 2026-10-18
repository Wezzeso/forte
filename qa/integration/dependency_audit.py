"""Export installed package versions/license metadata without reading media."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform


def collect() -> dict:
    packages = []
    for distribution in sorted(distributions(), key=lambda item: item.metadata["Name"].lower()):
        metadata = distribution.metadata
        license_classifiers = [
            value for value in metadata.get_all("Classifier", [])
            if value.startswith("License ::")
        ]
        packages.append({
            "name": metadata["Name"],
            "version": distribution.version,
            "license_expression": metadata.get("License-Expression"),
            "legacy_license": (metadata.get("License") or "")[:300] or None,
            "license_classifiers": license_classifiers,
            "license_files": metadata.get_all("License-File", []),
            "requires_python": metadata.get("Requires-Python"),
            "review_status": "metadata_recorded_not_redistribution_clearance",
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "scope": "installed CPU development environment; optional GPU packages and system binaries excluded",
        "packages": packages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    data = collect()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"packages_recorded": len(data["packages"]), "python": data["python"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
