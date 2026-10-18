# Forte contracts, version 1.1.0

Import from `forte.shared.schemas`. The canonical definition is
`src/forte/shared/schemas.py`; generated files in `schemas/` use JSON Schema.
`version.json` declares version, exact routes, authentication and timing units.

The public endpoint paths are `POST /api/v1/analyze` and
`GET /api/v1/jobs/{job_id}`. Both require `X-API-Key`; upload is multipart field
`file`. Accepted upload returns `AnalyzeResponse`. Polling returns the
`JobResponse` union, discriminated by `status` through
`JOB_RESPONSE_ADAPTER.validate_python(payload)` or `.validate_json(bytes)`.

Public statuses are `processing`, `done`, `failed`. The only completed verdicts
are `violation` and `no_violation`; positive results require at least one event,
negative results require zero. Every event requires one to three raw base64 JPEG
images, a source timestamp, a type, and a description including the applicable
source. Rich UI diagnostics do not add extra jury fields. Multiple events are
supported. `processing` and `failed` contain exactly `job_id` and `status`.

## Source timeline and validation boundaries

All `*_sec` fields are finite nonnegative seconds relative to the original video
start. The decoder subtracts `VideoAsset.source_start_sec` from stream PTS; that
origin may itself be negative. Width/height are after applying recorded rotation.
`FrameRef.source_pts` preserves optional raw stream PTS and `VideoAsset.time_base`
records its rational unit. FPS is metadata and never the evidence timestamp clock.

Models reject invalid enum values, reversed intervals, duplicate IDs, invalid
JPEGs, inconsistent verdict/event lists and unsupported enabling of rules.
`ClipRequest.validate_decisions` binds outputs to requested fact and existing
frame IDs, rejecting missing required facts and unrequested IDs.
`FrameRef.validate_timeline(asset)` and
`CompletedResponse.validate_timeline(asset)` check known source duration and job.
`VerifiedEvent.validate_context(asset, frames, rule)` checks source frame IDs,
rule applicability/enabling, present required facts, timeline and anchor interval.
Call these context-dependent methods at the relevant pipeline boundary; standalone
model parsing cannot know a video's duration or verify frame identity.

The evidence assembler must additionally verify owned paths, file integrity and
that each JPEG actually depicts its claimed source PTS. Structured validation
cannot establish observation truth, visual sufficiency, legal applicability,
temporal conditions or exceptions. Rule source verification is distinct from
applicability verification; enabling a rule requires both. A score requires
explicit provenance and is not a calibrated probability.

The Python models are validated working objects. Immutable completion means a
canonical validated JSON byte artifact published atomically and served unchanged;
do not expose a mutable cached Python object as that artifact. The worker must
assemble evidence successfully before marking a job done. A decoder/model error
or incomplete coverage must never be converted into a negative result.

## Synthetic fixtures and regeneration

`fixtures/accepted.json`, `processing.json`, `failed.json`, `negative.json`, and
`positive.json` parse against the public models. `synthetic-evidence.jpg` is a
programmatically drawn image and is also embedded in the positive fixture. It
does not depict private footage or make a real rule claim. The negative fixture
is a contract example, not a model inference result.

From the repository root, with the source directory on Python's import path,
run `python contracts/generate.py`. The generator reads no media or labels.
Pillow checks real JPEG decoding. Runtime validators implement cross-field and
context invariants that a JSON Schema consumer must separately enforce.

## Change policy

A01 owns the public shape and this version. Coordinate changes before modifying
field names, enum values, required fields or validation boundaries. A breaking
public or internal shape change increments the major version. Additive optional
internal changes increment the minor version. Corrections preserving the
documented shape increment the patch version. Regenerate schemas and fixtures
after an approved change, rerun the contract suite and notify affected owners.

Pydantic v2 model validators, field validators and discriminated unions follow
the [official validator documentation](https://docs.pydantic.dev/latest/concepts/validators/)
and [union documentation](https://docs.pydantic.dev/latest/concepts/unions/).

## Dataset split names

Version 1.1.0 adds the canonical `final_check` split name agreed by dataset,
training and evaluation owners. `EvaluationRecord` uses `train`, `validation`,
or `final_check`; `RecordingRecord` additionally allows `unassigned`. For
compatibility with the initial draft only, Python validation accepts `test`
and immediately normalizes it to `final_check`. Serialized records and emitted
JSON schemas use only the canonical name. This local team split is distinct
from the organizer's unseen jury test videos and does not authorize training
on either final-check data or jury data.
