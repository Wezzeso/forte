# Three-arm Qwen3.8 development comparison

`python -m evaluation.qwen38_comparison` is an **offline aggregator** for baseline
8B, untuned Qwen3.8, and tuned Qwen3.8 results. It makes no model/API/GPU call,
reads no video, and never trains, selects, promotes or deploys a model. A genuine
run requires the separately authorized compute/exposure gates and actual model
identity receipts. Prepared code and synthetic tests are not model measurements.

The present validation references contain six unique contents, three supplied
positive labels and three supplied negative labels. The coarse validation file
contains four approximate points over the three positive contents: one content
retains the two disagreeing points supplied for its duplicate aliases. Those
points are not four independent events. The six contents are reused development
diagnostics. Scenario independence and human-reverified event/fact/type/support,
legal and boundary truth remain unestablished.

Only `provided_binary_labels.validation.jsonl` and
`user_coarse_anchors.validation.jsonl` were inspected for this implementation.
FINAL_CHECK labels, media and predictions were not opened or scored. No real
three-arm result has been generated.

## Run an already completed, authorized comparison

Run from the workspace root after the private plan, identity/exposure receipts
and real terminal artifacts exist:

```powershell
python -m evaluation.qwen38_comparison `
  --plan evaluation/private/qwen38-comparison-plan.json `
  --results evaluation/private/qwen38-baseline-terminals.jsonl `
  --results evaluation/private/qwen38-untuned-terminals.jsonl `
  --results evaluation/private/qwen38-tuned-terminals.jsonl `
  --labels data/manifests/provided_binary_labels.validation.jsonl `
  --coarse-anchors data/manifests/user_coarse_anchors.validation.jsonl `
  --metadata evaluation/private/qwen38-comparison-metadata.json `
  --output benchmarks/quality/qwen38-development-comparison.json

python -m unittest discover -s tests -p test_qwen38_comparison.py -v
```

The three `--results` flags may also point at one combined export. Individual arm
runs may target different services or time windows, but they must bind the same
comparison plan and recipe. Preserve original terminal artifacts. Do not import
the old staged-concise trial as this baseline: it was measured under a different
recipe. All three arms require actual runs under the newly frozen conditions.

Exit `0` means the supplied artifacts support a development diagnostic with
complete provenance, including accounted-for failures; it does **not** mean a
quality/latency target passed. Exit `2` means synthetic, unmeasured, partial or
incompletely verified evidence. Exit `1` means malformed or incomparable input.
The CLI prints only aggregate status/counts, never private source identities,
results, filenames, descriptions, pixels or exception details.

## Frozen plan contract

Construct the plan from A09's audited development-only role export. Do not copy
FINAL_CHECK records into it. Freeze and retain the exact plan before any attempts
are made; `evaluation.io.digest(plan)` is the shared `comparison_plan_sha256`.
Hashes establish supplied artifact integrity and binding, not independent proof
that a source is eligible or a model actually ran.

The plan has these fields:

| Field | Required value |
| --- | --- |
| `schema_version` | `forte.qwen38_comparison_plan.v1` |
| `records` | One `{sample_id, content_sha256, split, duration_sec?}` per distinct development content; split is explicit `train` or `validation` |
| `arms` | Object with exactly `baseline_8b`, `qwen38_untuned`, `qwen38_tuned` |
| Each arm | Actual `model_id`, immutable `model_revision`, explicit `adapter_sha256`; baseline/untuned use null, tuned requires the real adapter SHA-256 |
| `recipe` | Shared `sampling`, `prompt_sha256`, `generation`, `evidence`, `timing`, `concurrency`, `supported_envelope` |
| `recipe_sha256` | `evaluation.io.digest(recipe)` |
| `provided_labels_sha256` | `evaluation.weak_supervision.provided_label_digest(labels)` for the complete supplied binary input rows in CLI concatenation order |
| `coarse_anchors_sha256` | Same digest function over supplied coarse rows, when those diagnostics are requested |
| `attempts` | Predeclared list of `{attempt_id, arm, sample_id, content_sha256, split, repeat_index}` |

Untuned and tuned must bind the same base model revision. No floating
`main`/`latest` or unknown revision is accepted. Immutable identity must come
from actual acquisition/runtime evidence, not a fabricated hash. Use the
verified target model ID rather than treating the arm label as a model identity.

Every arm has the same `(sample_id, repeat_index)` population; repeat indices
start at zero without gaps. Every planned content appears in every arm. Content
aliases cannot increase the population. Freeze all attempts, including repeated
uploads, before execution. Fresh repeatability cases need at least three runs
under the separate A12 policy; a one-repeat comparison does not establish that
policy. No best-of-retries substitution is allowed.

The shared recipe must identify effective frame/PTS selection, complete prompt
bytes, generation settings, evidence policy, both timer endpoints, concurrency,
network/duration/size/codec envelope, cache/warm-service policy and relevant
runtime configuration. Differences in a model's inherent processor and exact
runtime identity belong in its immutable arm receipt; disclose them. Changed
sampling/prompt/quantization/caching conditions require a newly predeclared
comparison instead of reusing a recipe hash from another run.

Role labels in a caller-authored plan cannot prove eligibility. Retain A09's
fresh content-role/exposure proof outside this report and bind its actual hash
in metadata. The evaluator rejects nondevelopment roles on records, attempts,
labels, coarse references and results, as well as paths containing `final_check`
(including punctuation/case variants) before opening input files. Renaming a
file is not authorization. No final-check option exists.

## Terminal export and provenance metadata

Each result echoes every identity field from its planned attempt plus:

- `comparison_plan_sha256` and `recipe_sha256` from the shared comparison plan.
- `model_identity:{model_id,model_revision,adapter_sha256}` from the actual loaded
  arm's verified receipt. These three fields can instead be top-level.
- `status`. Successful results require `done`, `public_contract_validated:true`
  and the complete `complete_result` object, or top-level verdict/violations.
  Failure terminals may use `failed`, `timeout`, `upload_failed`,
  `admission_rejected`, `invalid_json`, `insufficient_analysis`, `not_attempted`
  or another explicit failure status.
- On success, `upload_start_to_full_result_ms` with
  `timing_origin:"client_upload_start_to_full_result_byte"`.
- When actually observed in the browser,
  `upload_start_to_usable_result_ms` with
  `ui_timing_origin:"client_upload_start_to_usable_rendered_result"`.

Pre-inference failures may have null/absent `model_identity`: retain that actual
identity as unknown and supply a separate `expected_model_identity` matching the
frozen arm. Failed uploads need no fictional model load. A claimed success with
only expected identity receives failure-inclusive unprocessed counts and no
latency credit; it also blocks provenance readiness.

The complete useful public result must pass the real upstream contract and
original-JPEG/source provenance validator. The comparator also rejects obvious
verdict/event inconsistencies, invalid points and out-of-duration events as
failed results. `public_contract_validated:true` is an upstream attestation; this
script does not decode or independently inspect evidence pixels. Result/model
identity or recipe mismatches, unexpected attempts and duplicate terminal IDs
invalidate the whole input rather than being silently discarded.

Metadata requires the following **actual** verification exports:

```json
{
  "inference_kind": "real_local",
  "attempt_plan_frozen_before_run": true,
  "validation_reused_for_development": true,
  "final_check_used": false,
  "execution_verified": true,
  "exposure_audit_passed": true,
  "fresh_upload_decode_inference_verified": true,
  "coarse_anchors_used_to_condition_inference": false,
  "comparison_plan_sha256": "actual shared plan digest",
  "exposure_audit_sha256": "actual current A09 exposure audit digest",
  "verified_model_identity_receipts": {
    "baseline_8b": "actual receipt digest",
    "qwen38_untuned": "actual receipt digest",
    "qwen38_tuned": "actual receipt digest"
  }
}
```

The displayed strings describe required sources; they are deliberately invalid
placeholders, not usable attestations. Unknown provenance stays unknown. A
synthetic fixture remains `SYNTHETIC_ONLY` even if every software condition is
satisfied. A report cannot prove these flags independently: retain actual runner,
image/base/adapter, source-clock/evidence, fresh decode, exposure and frozen-plan
receipts with the private run.

## Denominators and scope

Per-attempt completion includes all frozen attempts. Whole-video TP/FP/FN/TN are
reported only for completed results with independently supplied binary labels;
failed/missing results become explicit `unprocessed_positive` or
`unprocessed_negative`. Effective correctness includes those failures in its
denominator. A failed upload without a durable job is still an attempt.
Unprocessed negatives are never counted as clean TNs.

Per-content confusion uses repeat zero, even when a later repeat succeeds.
Per-attempt confusion includes every repeat and reports disagreement separately.
TRAIN and VALIDATION confusion matrices are additionally reported separately;
their combined development counts do not turn training data into validation.
The report includes per-class deltas against the baseline; no larger-model-name
preference, automatic winner or promotion is inferred. Quality degradation and
completion failures remain visible.

Optional approximate-point bands use the first frozen attempt per content and
the full supplied-point denominator, including failed results. They report
distance to supplied coarse points, predicted-point count and failure-inclusive
hit rates. Repeated model points cannot create reference events. Unknown source
uncertainty remains unknown; the 1/2/5-second bands are separately declared team
diagnostics. Scoring many points can inflate hits, so no exact event precision,
recall/F1, type, boundary or millisecond-onset accuracy is inferred.

Fact, evidence support, event type, legal applicability and exact timing metrics
remain explicitly `UNMEASURED_NO_GENUINE_REFERENCE`. Silver teacher agreement is
not independent accuracy. If real human-reviewed references later exist, use
the separately documented human-gold evaluators with their real review/source
bindings; do not relabel workbook points or teacher records to bypass them.

Latency p50/max describe completed measurements with the correct upload-start
clock and must be read beside failure counts and timing coverage. P95 is omitted
for this small reused development population. Exactly 10,000 ms fails the strict
target. Bytes-only measurements never establish a usable-rendered-result target;
missing browser timing remains unmeasured. Loading/warmup and human deliberation
need their own clocks. A finite development comparison cannot establish universal
100% correctness, scenario generalization or an untouched held-out result.
