"""Synthetic-only process fixture. This module is never a production runner."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from forte.config import WorkerSettings
from forte.worker import Worker


class SyntheticFaultPipeline:
    def __init__(self, mode, marker):
        self.mode, self.marker = mode, marker

    async def analyze(self, job, *, stage, deadline, cancel_event):
        await stage("screening")
        self.marker.write_text(json.dumps({"attempt": job.attempt}), encoding="utf-8")
        if self.mode == "stall":
            await asyncio.sleep(120)
        payload = json.dumps({"job_id": job.job_id, "status": "done", "verdict": "no_violation",
                              "violations": []}, separators=(",", ":")).encode()
        return SimpleNamespace(json_bytes=payload,
                               timings={"clock": "process_monotonic", "stages_ms": {}},
                               metadata_path=self.marker)

    async def aclose(self):
        pass


async def run(args):
    worker = Worker(WorkerSettings.from_env(), pipeline=SyntheticFaultPipeline(args.mode, args.marker))
    try:
        result = await worker.run_once()
        print(json.dumps({"status": result.get("status") if result else "idle",
                          "attempt": result.get("attempt") if result else None}))
    finally:
        await worker.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stall", "complete"), required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("FORTE_SYNTHETIC_FAULT_FIXTURE") != "explicitly-enabled-for-qa":
        parser.error("synthetic fault fixture must be explicitly enabled")
    asyncio.run(run(args))
