"""Safe command-line snapshot verifier. Success is validation, not inference."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict

from .manifest import ManifestError, verify_local_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="verify all pinned local weights and processor checksums")
    verify.add_argument("--snapshot", required=True)
    verify.add_argument("--manifest", default="models/manifest.json")
    stage = commands.add_parser("stage", help="explicit provisioning-only download of pinned public model artifacts")
    stage.add_argument("--snapshot", required=True)
    stage.add_argument("--manifest", default="models/manifest.json")
    stage.add_argument("--workers", type=int, default=4)
    stage.add_argument("--timeout", type=float, default=1800)
    smoke = commands.add_parser("smoke", help="real private model request using generated synthetic media only")
    smoke.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    smoke.add_argument("--media-root", default="/tmp/forte-model-smoke")
    smoke.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        if args.command == "smoke":
            from .adapter import ModelError
            from .smoke import synthetic_smoke
            from forte.decision.structured import DecisionConfigurationError, DecisionOutputError
            try:
                result = asyncio.run(synthetic_smoke(base_url=args.base_url, media_root=args.media_root,
                    timeout_sec=args.timeout, api_key=os.environ.get("FORTE_MODEL_API_KEY")))
            except (ModelError, DecisionConfigurationError, DecisionOutputError, TimeoutError):
                print(json.dumps({"real_local_inference": False, "error_code": "synthetic_smoke_failed"}))
                return 1
        elif args.command == "stage":
            from .staging import stage_local_snapshot
            result = stage_local_snapshot(args.snapshot, args.manifest,
                                          max_workers=args.workers, timeout_sec=args.timeout)
        else:
            result = {"verified": True, **asdict(verify_local_snapshot(args.snapshot, args.manifest))}
    except ManifestError as exc:
        print(json.dumps({"verified": False, "error_code": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
