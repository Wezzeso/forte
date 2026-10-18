"""Nonsensitive TCP routing probe, intended inside the private model container.

This checks network isolation only. Pair it with a real synthetic inference
request in the same container; a blocked connection alone is not proof of
offline inference. No video, frame, prompt, token or description is transmitted.
"""

from datetime import datetime, timezone
import json
import os
import socket


def main() -> int:
    reachable = False
    error_type = None
    try:
        # Fixed public address avoids DNS and sends no application payload.
        with socket.create_connection(("1.1.1.1", 443), timeout=3):
            reachable = True
    except OSError as error:
        error_type = type(error).__name__
    print(json.dumps({
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "external_tcp_reachable": reachable,
        "target": "1.1.1.1:443",
        "payload_sent": False,
        "failure_type": error_type,
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE") == "1",
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "VLLM_NO_USAGE_STATS": os.environ.get("VLLM_NO_USAGE_STATS") == "1",
        },
        "inference_verified_by_this_probe": False,
    }))
    return int(reachable)


if __name__ == "__main__":
    raise SystemExit(main())
