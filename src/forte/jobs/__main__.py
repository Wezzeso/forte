"""Schema bootstrap/recovery: python -m forte.jobs init|recover."""

import argparse
import json
import os

from .store import JobStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Forte durable job administration")
    parser.add_argument("command", choices=("init", "recover", "counts"))
    parser.add_argument("--database-url", default=os.environ.get("FORTE_DATABASE_URL"))
    parser.add_argument("--result-root", default=os.environ.get("FORTE_RESULT_DIR", "var/results"))
    arguments = parser.parse_args()
    if not arguments.database_url:
        parser.error("--database-url or FORTE_DATABASE_URL is required")
    store = JobStore(arguments.database_url, arguments.result_root)
    try:
        if arguments.command == "recover":
            print(json.dumps({"recovered": store.recover()}))
        elif arguments.command == "counts":
            print(json.dumps(store.counts()))
        else:
            print(json.dumps({"schema": "ready", "database": "sqlite-development" if store.is_sqlite else "postgresql"}))
    finally:
        store.close()


if __name__ == "__main__":
    main()
