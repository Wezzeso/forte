"""Read mounted secrets without persisting or logging their values."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from urllib.parse import quote


def configure_environment() -> None:
    # Applications receive the DSN only at process start. Compose config and image
    # history contain file paths, never a database password.
    if not os.environ.get("FORTE_DATABASE_URL"):
        password_file = os.environ.get("FORTE_DB_PASSWORD_FILE")
        if password_file:
            password = Path(password_file).read_text(encoding="utf-8").strip()
            if not password:
                raise RuntimeError("Database password file is empty")
            user = quote(os.environ.get("FORTE_DB_USER", "forte"), safe="")
            host = os.environ.get("FORTE_DB_HOST", "db")
            port = int(os.environ.get("FORTE_DB_PORT", "5432"))
            database = quote(os.environ.get("FORTE_DB_NAME", "forte"), safe="")
            os.environ["FORTE_DATABASE_URL"] = (
                f"postgresql+psycopg://{user}:{quote(password, safe='')}@{host}:{port}/{database}"
            )
    model_key_file = os.environ.get("FORTE_MODEL_API_KEY_FILE")
    if model_key_file and not os.environ.get("FORTE_MODEL_API_KEY"):
        os.environ["FORTE_MODEL_API_KEY"] = Path(model_key_file).read_text(encoding="utf-8").strip()


def main() -> None:
    configure_environment()
    if len(sys.argv) < 2:
        raise SystemExit("A service command is required")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
