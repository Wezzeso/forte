"""Regenerate versioned public/internal schemas and synthetic fixtures locally.

Run from the repository root: PYTHONPATH=src python contracts/generate.py.
Never import or read the private dataset while generating contract fixtures.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from forte.shared import schemas


ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    schema_dir = ROOT / "schemas"
    fixture_dir = ROOT / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for name in schemas.__all__:
        model = getattr(schemas, name)
        if isinstance(model, type) and issubclass(model, schemas.BaseModel):
            write_json(schema_dir / f"{name}.schema.json", model.model_json_schema())
    public_schema = schemas.JOB_RESPONSE_ADAPTER.json_schema()
    public_schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    public_schema["$id"] = f"urn:forte:contracts:{schemas.CONTRACT_VERSION}:job-response"
    write_json(schema_dir / "JobResponse.schema.json", public_schema)

    image = Image.new("RGB", (128, 96), color=(227, 232, 238))
    canvas = ImageDraw.Draw(image)
    canvas.rectangle((10, 10, 117, 85), outline=(50, 65, 80), width=2)
    canvas.text((18, 42), "SYNTHETIC", fill=(20, 25, 30))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=False)
    jpeg_bytes = buffer.getvalue()
    (fixture_dir / "synthetic-evidence.jpg").write_bytes(jpeg_bytes)
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    positive = schemas.CompletedResponse(
        job_id="synthetic-job-001",
        verdict="violation",
        violations=[
            schemas.PublicViolation(
                type="Synthetic contract event",
                timecode_sec=2.5,
                description=(
                    "Synthetic image used only to validate the API contract. "
                    "Synthetic fixture document, section TEST-1; no real procedure or legal finding."
                ),
                evidence_frames=[schemas.EvidenceFrame(timestamp_sec=2.5, image_base64=encoded)],
            )
        ],
    )
    fixtures = {
        "accepted": schemas.AnalyzeResponse(job_id="synthetic-job-001"),
        "processing": schemas.ProcessingResponse(job_id="synthetic-job-001"),
        "failed": schemas.FailedResponse(job_id="synthetic-job-001"),
        "negative": schemas.CompletedResponse(
            job_id="synthetic-job-001", verdict="no_violation", violations=[]
        ),
        "positive": positive,
    }
    for name, model in fixtures.items():
        payload = model.model_dump(mode="json")
        if name != "accepted":
            schemas.JOB_RESPONSE_ADAPTER.validate_python(payload)
        write_json(fixture_dir / f"{name}.json", payload)
    write_json(
        ROOT / "version.json",
        {
            "contract_version": schemas.CONTRACT_VERSION,
            "python_module": "forte.shared.schemas",
            "timestamp_unit": "seconds",
            "timestamp_origin": "original video start, subtracting source_start_sec",
            "public_routes": ["POST /api/v1/analyze", "GET /api/v1/jobs/{job_id}"],
            "authentication_header": "X-API-Key",
            "multipart_field": "file",
            "fixture_provenance": "programmatically generated synthetic image; no dataset access",
        },
    )
    print(f"Generated contracts {schemas.CONTRACT_VERSION}; {len(fixtures)} synthetic JSON fixtures")


if __name__ == "__main__":
    main()
