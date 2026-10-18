"""Actual HTTP smoke using generated media; prints aggregate checks only."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit

import av
import httpx
from PIL import Image

from forte.shared.schemas import AnalyzeResponse, CompletedResponse


def synthetic_video(path: Path) -> None:
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=4)
        stream.width, stream.height = 128, 96
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            with Image.new("RGB", (128, 96), (20 + 20 * index, 60, 100)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = index, Fraction(1, 4)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def check(base_url: str, api_key: str, deadline_seconds: float) -> dict:
    report = {
        "pass": False,
        "scope": "actual HTTP with generated synthetic MOV; no accuracy or required-video latency claim",
        "checks": {}, "completed_schema_valid": False,
        "synthetic_upload_start_to_result_sec": None,
    }
    with tempfile.TemporaryDirectory(prefix="forte-synthetic-smoke-") as temporary:
        path = Path(temporary) / "synthetic.mov"
        synthetic_video(path)
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=20, trust_env=False, follow_redirects=False) as client:
            for label, headers in (("missing_key", {}), ("wrong_key", {"X-API-Key": "synthetic-wrong-key"})):
                post = client.post("/api/v1/analyze", headers=headers, content=b"synthetic-unparsed-body")
                get = client.get("/api/v1/jobs/synthetic-unknown", headers=headers)
                report["checks"][f"{label}_post_rejected"] = post.status_code == 401
                report["checks"][f"{label}_get_rejected"] = get.status_code == 401
            headers = {"X-API-Key": api_key}
            report["checks"]["unknown_job_404"] = client.get("/api/v1/jobs/synthetic-unknown", headers=headers).status_code == 404
            started = time.monotonic()
            with path.open("rb") as media:
                uploaded = client.post("/api/v1/analyze", headers=headers,
                                       files={"file": ("synthetic.mov", media, "video/quicktime")})
            report["checks"]["upload_202"] = uploaded.status_code == 202
            if uploaded.status_code != 202:
                report["error_code"] = "synthetic_upload_rejected"
                return report
            job = AnalyzeResponse.model_validate(uploaded.json())
            report["job_id"] = job.job_id  # Opaque trace ID; no observations.
            while time.monotonic() - started < deadline_seconds:
                response = client.get(f"/api/v1/jobs/{job.job_id}", headers=headers)
                if response.status_code != 200:
                    report["error_code"] = "polling_http_failure"
                    return report
                payload = response.json()
                if time.monotonic() - started > deadline_seconds:
                    report["error_code"] = "synthetic_analysis_deadline"
                    return report
                if payload.get("status") == "done":
                    report["terminal_status"] = "done"
                    CompletedResponse.model_validate(payload)
                    report["completed_schema_valid"] = True
                    report["synthetic_upload_start_to_result_sec"] = round(time.monotonic() - started, 4)
                    report["synthetic_upload_start_to_terminal_sec"] = report["synthetic_upload_start_to_result_sec"]
                    repeated = client.get(f"/api/v1/jobs/{job.job_id}", headers=headers)
                    report["checks"]["immutable_completed_bytes"] = repeated.status_code == 200 and repeated.content == response.content
                    report["checks"]["schema_valid_completed_result"] = True
                    report["pass"] = all(report["checks"].values())
                    return report
                if payload.get("status") == "failed":
                    report["terminal_status"] = "failed"
                    report["error_code"] = "synthetic_analysis_failed"
                    report["synthetic_upload_start_to_terminal_sec"] = round(time.monotonic() - started, 4)
                    return report
                if payload != {"job_id": job.job_id, "status": "processing"}:
                    report["error_code"] = "invalid_processing_payload"
                    return report
                time.sleep(min(1, max(.01, deadline_seconds - (time.monotonic() - started))))
    report["error_code"] = "synthetic_analysis_deadline"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="FORTE_API_KEY")
    parser.add_argument("--deadline-seconds", type=float, default=285)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    target = urlsplit(args.base_url)
    if (target.scheme not in {"https", "http"} or not target.hostname
            or target.username or target.password or target.query or target.fragment
            or target.path not in {"", "/"} or not 0 < args.deadline_seconds <= 300):
        parser.error("Use a bare authorized service origin and a deadline from 0 to 300 seconds")
    key = os.environ.get(args.api_key_env)
    if not key:
        parser.error("The specified API-key environment variable is unset")
    try:
        report = check(args.base_url, key, args.deadline_seconds)
    except Exception:
        # Never echo a response, frame, source path, credential, or URL query.
        report = {"pass": False, "error_code": "http_smoke_or_schema_failure"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report.get("pass", False), "checks": len(report.get("checks", {})),
                      "error_code": report.get("error_code")}))
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
