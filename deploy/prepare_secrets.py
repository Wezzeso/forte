"""Create local random deployment secrets without printing secret values."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "secrets")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in ("postgres_password", "api_keys"):
        path = directory / name
        # Exclusive creation preserves existing DB/API credentials on rerun.
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Compose mounts local secret files without UID remapping. Their
            # parent stays 0700 on POSIX, while the mounted file is readable by
            # the dedicated non-root service UID, as with Docker secret mode.
            if os.name == "posix":
                path.chmod(0o444)
            print(f"Preserved existing {name} file")
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
        if os.name == "posix":
            path.chmod(0o444)
        print(f"Created {name} file")
    print("Secret values remain in local files and are never included in this output.")


if __name__ == "__main__":
    main()
