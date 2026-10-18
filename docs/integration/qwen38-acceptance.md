# Qwen3.8 fresh HTTP acceptance

`benchmarks/performance/qwen38_acceptance.py` implements a new, failure-inclusive
measurement path. The implementation and its generated fixtures are not a
measurement of the real service. No release, reservation, private model run or
upload is created merely by importing the module or running `prepare`.

Each frozen case has at least three fresh multipart uploads. Each upload polls
only its own newly returned job ID and downloads the complete canonical JSON,
including every base64 JPEG. The client uses a monotonic clock immediately
before POST and stops its primary timer after the completed response's final
byte. Contract validation decodes the actual JPEGs; its additional elapsed time
is retained separately. Polling delay is included. POST acceptance time is not
the full result time.

The browser has a further readiness boundary: all evidence images decoded,
source player `loadeddata`, and the result rendered. HTTP final-byte timing
does not measure that boundary; the UI's upload-origin timer supplies it. A
restored page cannot reconstruct a previous document's monotonic upload clock.

## Prerequisites

- Use the project Python environment with `httpx`, `pydantic`, and Pillow.
- Pin the actual deployed model revision and adapter hash. A plan's expected
  model identity is checked against each actual private server receipt.
- Freeze the shared three-arm comparison plan and its recipe digest with the
  quality evaluator. Use separate acceptance plans for `baseline_8b`,
  `qwen38_untuned`, and `qwen38_tuned`; never switch a server mid-plan.
- Select only root-authorized TRAIN/VALIDATION contents. Keep FINAL_CHECK blind.
- Refresh source-role and central service-exposure inventory before reserving.
- Obtain the exact root acceptance release after exclusive idle GPU handoff.
  The release cannot exceed 18 October 2026, 11:01:25 Asia/Qyzylorda
  (`2026-10-18T06:01:25Z`).
- Store all plans, sources, receipts, full responses and terminal exports below
  `benchmarks/private`. They include private source and evidence identities.
- Supply actual server stage/native/evidence receipts after execution. Their
  consistency is checked; the operator must establish their server provenance.

## Frozen input and release

Write a private spec, for example `benchmarks/private/qwen38/acceptance-spec.json`:

```json
{
  "schema_version": "forte.qwen38-acceptance-spec.v1",
  "arm": "qwen38_tuned",
  "recipe_sha256": "<shared-recipe SHA-256>",
  "comparison_plan_sha256": "<shared-comparison-plan digest>",
  "expected_model_identity": {
    "model_id": "Qwen/Qwen3.8-27B",
    "model_revision": "<actual pinned revision>",
    "adapter_sha256": "<actual adapter SHA-256, or null for an untuned arm>"
  },
  "settings": {
    "base_url": "<root-authorized direct API origin>",
    "whole_result_cache": false,
    "concurrency": 1,
    "normalization": ["job_id"],
    "deadline_seconds": 300,
    "request_timeout_seconds": 60,
    "poll_seconds": 0.25
  },
  "cases": [{
    "case_id": "case_001",
    "sample_id": "<sample ID shared with the comparison plan>",
    "owned_path": "<absolute source path>",
    "content_sha256": "<source SHA-256>",
    "source_bytes": 123,
    "split": "train"
  }]
}
```

The template is deliberately non-executable until actual pins are supplied.
`prepare` hashes the source bytes, freezes random unique attempt IDs and refuses
an existing output. It never decodes the source or opens a model.

```powershell
.venv/Scripts/python.exe -m benchmarks.performance.qwen38_acceptance prepare --spec benchmarks/private/qwen38/acceptance-spec.json --output benchmarks/private/qwen38/acceptance-plan.json --repeats 3
```

The root then creates a `forte.qwen38-root-release.v1` release with phase
`acceptance`. Its `attempt_plan_sha256` is the SHA-256 of the exact plan file,
not the plan's internal `conditions_sha256`. It must bind `config_sha256`,
`source_archive_sha256`, `runtime_image_sha256`, `base_manifest_sha256`, exact
`maximum_attempts`, a bounded `maximum_execution_seconds`, an exclusive idle
handoff receipt and expiry. Acceptance additionally requires
`uploads_released: true`, `not_before_utc`, and the exact
`expected_model_identity` from the plan.

The central exposure gate consumes that release and reserves every attempt,
including attempts that might fail or be cancelled:

```powershell
.venv/Scripts/python.exe -m training.qwen38_preflight --config training/private/qwen38-role-config.json --reserve-plan benchmarks/private/qwen38/acceptance-plan.json --release benchmarks/private/qwen38/acceptance-release.json --output benchmarks/private/qwen38/acceptance-reservation.json
```

This produces `forte.qwen38-reservation.v1`. Reservation does not launch a model.
It is a separate required step; a command-line declaration cannot replace it.

## Released execution and private server receipts

Only after the root grants this exact release and reservation, set the API key
through the normal secret-management path. Do not put the key in a command or
report. The following is the actual upload command, not a preparation command:

```powershell
.venv/Scripts/python.exe -m benchmarks.performance.qwen38_acceptance execute --plan benchmarks/private/qwen38/acceptance-plan.json --release benchmarks/private/qwen38/acceptance-release.json --reservation benchmarks/private/qwen38/acceptance-reservation.json --output benchmarks/private/qwen38/acceptance-run --api-key-env FORTE_API_KEY
```

The executor refuses an existing run directory, checks source bytes before
each attempt and rechecks root release/reservation before each POST. A durable
one-use claim beside the plan prevents replaying the same reserved attempts
into a different output folder. Do not remove that claim to retry a run.
A durable submission intent precedes each POST. Each complete attempt is written to its
own immutable `attempt.json`, including failures and timeouts. An interrupted
attempt with only an intent is unknown, never a success or a free retry slot.
Accepted/uncertain jobs must be settled in the central exposure inventory
before another release is considered.

For each completed attempt, export actual server metadata to
`<run>/<attempt_id>/server-receipt.json` with schema
`forte.qwen38-acceptance-server-receipt.v1`:

- `attempt_id`, `job_id`, `conditions_sha256`, `recipe_sha256`, `content_sha256`,
  `normalized_result_sha256`, and `actual_model_identity` bind the exact job,
  source, result and actual model/adapter.
- `cache_hits` must equal
  `{"decode":0,"model":0,"evidence":0,"whole_result":0}` and `stage_calls`
  must contain positive actual call counts for `decode`, `model`, and `evidence`.
- `native_source` contains `content_sha256`, `source_complete: true`,
  `tail_included: true`, integer `origin_pts`, positive integer `time_base_num`
  and `time_base_den`, `decoded_source_frame_count`, and chronological `frames`.
  Each frame contains the actual `frame_id`, integer `source_pts`, and
  `timestamp_sec = (source_pts - origin_pts) * time_base_num / time_base_den`.
- `evidence_frames` contains an entry per returned evidence JPEG, in response
  order: `event_index`, `frame_index`, `frame_id`, `source_pts`, `timestamp_sec`,
  and SHA-256 of the actual decoded JPEG bytes as `jpeg_sha256`. Each frame must
  occur in the native ledger. An empty list is correct for no evidence.
- `decision_projection` preserves actual facts, candidate events, rule IDs,
  legal/procedure conditions, final events, uncertainty and limitations from
  the private analysis audit. Do not replace this with only a binary verdict.

These receipts require an actual server artifact exporter or an operator
export from the corresponding job's private artifacts. The HTTP client cannot
observe the server's loaded weights, cache activity or native-frame ledger.
Absent/invalid receipts keep acceptance unverified.

## Offline report and interpretation

```powershell
.venv/Scripts/python.exe -m benchmarks.performance.qwen38_acceptance report --run benchmarks/private/qwen38/acceptance-run --private-output benchmarks/private/qwen38/acceptance-terminals.json --public-output benchmarks/performance/qwen38-acceptance-observed.json
.venv/Scripts/python.exe -m pytest qa/system/test_qwen38_acceptance.py -q
```

The report keeps the frozen expected denominator. Missing, timed-out, failed,
duplicate or receipt-unverified attempts cannot disappear. Three uploads must
return distinct job IDs. Every pair within each case compares the complete
public result, native-source identity, private decision projection, and native
evidence hashes. **Only `job_id` keys are normalized**. Event IDs, descriptions,
event ordering, timestamps, rule references, uncertainty, limits and JPEG bytes
remain part of comparison. Unknown projection fields remain in the hash.

The ten-second condition is strict (`<10000 ms`); exactly ten seconds fails.
Failures fail observed acceptance. Missing attempts/receipts make it unverified.
Public output contains aggregate counts, timings and comparison outcomes;
private terminal exports retain full identities, responses and hashes for the
quality evaluator. `comparison_plan_sha256`, `recipe_sha256`, arm, sample,
content, split and zero-based repeat index are echoed on every terminal slot.
`expected_model_identity` is retained even for failures before inference;
`model_identity` is only asserted after verification of the actual server receipt.

Repeated content is deliberately reused for repeatability and is not a set of
independent novel latency samples. Tiny samples do not produce a p95. Passing
this finite receipt-backed test establishes neither universal accuracy nor
human-reviewed type/legal/boundary correctness. Browser-ready latency and
reviewed-gold evaluation remain separate measured requirements.
