"""A local heartbeat freshness check; never returns job or media content."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path


def main() -> int:
    try:
        heartbeat = Path(os.environ.get("FORTE_WORKER_HEARTBEAT_FILE", "/var/lib/forte/worker-heartbeat.json"))
        record = json.loads(heartbeat.read_text(encoding="utf-8"))
        updated_at = datetime.fromisoformat(record["updated_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - updated_at).total_seconds()
        return 0 if 0 <= age <= float(os.environ.get("FORTE_WORKER_HEARTBEAT_MAX_AGE_SECONDS", "30")) else 1
    except (OSError, ValueError, KeyError, TypeError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
