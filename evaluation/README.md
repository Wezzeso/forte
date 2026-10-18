# Local evaluation

This harness produces internal diagnostics, never an invented official score. It uses Python's standard library and runs independently of GPU dependencies. Supplied videos, captions, per-video predictions and annotations stay in local private files. Commands print only aggregate summaries; parse failures do not echo source records or paths.

The authorized automatic first pass has a separate operational track in `evaluation/WEAK-SUPERVISION.md`. It can train/evaluate/select a provisional weak video model against independent provided binary labels without manual event annotation. The human-gold workflow below remains separate and is not a blocker for that authorized experiment.

Run from the repository root:

```powershell
python -m unittest discover -s tests -p test_evaluation.py -v
python -m evaluation.synthetic --output-dir benchmarks/quality/synthetic
python -m evaluation audit-splits --manifest data/splits/candidate_split.json --output benchmarks/quality/split-audit.json
```

The proposed split returns exit code 2 until A20 locks reviewed scenario groups. An integrity pass alone never authorizes training. Exit codes: `0` successful evaluation/authorized audit/earned candidate selection; `2` valid report with a failed release/selection gate; `1` malformed input or I/O failure.

For real predictions, after local human review and an actual local model run:

```powershell
python -m evaluation evaluate --truth annotations/private/reviewed.jsonl --predictions evaluation/private/baseline.jsonl --splits data/splits/locked_split.json --metadata evaluation/private/baseline-metadata.json --rules evaluation/private/verified-rule-ids.json --split validation --output benchmarks/quality/baseline.json
python -m evaluation evaluate --truth annotations/private/reviewed.jsonl --predictions evaluation/private/candidate.jsonl --splits data/splits/locked_split.json --metadata evaluation/private/candidate-metadata.json --rules evaluation/private/verified-rule-ids.json --split validation --output benchmarks/quality/candidate.json
python -m evaluation select --baseline benchmarks/quality/baseline.json --candidate benchmarks/quality/candidate.json --output benchmarks/quality/selection.json
```

Paths above describe the real workflow; missing human review or model access is not replaced with synthetic data. Synthetic reports cannot earn checkpoint selection. Baseline remains the default when improvement is unestablished. A run with no required samples writes an aggregate report and exits 2.

## Local input contracts

All record files accept JSONL, a JSON array, or `{"records": [...]}`. A02 uses `sample_id`/`recording_id` aliases and `content_sha256`/`content_hash` aliases; both are supported. Do not display private input records in cloud tools.

Truth records follow A03's `forte.annotations.v1` export:

```json
{
  "sample_id": "opaque-sample-id",
  "duration_sec": 10,
  "review_status": "human_reviewed",
  "reviewer_id": "local-reviewer-reference",
  "reviewed_at": "2026-10-17T00:00:00Z",
  "verdict": "violation",
  "events": [{"event_type_id": "reviewed-taxonomy-id", "anchor_sec": 4.2, "rule_id": "verified-rule-id"}],
  "scenario_tags": ["multiple_events"]
}
```

Unresolved verdicts, missing human-review provenance and local-teacher annotations do not enter accuracy denominators. Their samples remain in completion denominators. An unresolved truth `rule_id` is an explicit grounding gap, never a correct mapping by assumption. Negative review requires A03's complete local review protocol; the evaluator cannot independently establish that a supposed negative is visually normal.

Prediction records contain `sample_id`, `status` (`done`, `processing`, or `failed`), `verdict` for done, `events` (or public `violations`) and `upload_start_to_result_sec` (alias `elapsed_sec`, always measured from upload start). Event aliases are `event_type_id`/`type` and `anchor_sec`/`timecode_sec`/`timestamp_sec`. Events may carry `rule_id` and `local_review`:

```json
{
  "sample_id": "opaque-sample-id",
  "status": "done",
  "verdict": "violation",
  "upload_start_to_result_sec": 21.5,
  "events": [{
    "event_type_id": "reviewed-taxonomy-id",
    "anchor_sec": 4.1,
    "rule_id": "verified-rule-id",
    "local_review": {"review_status": "human_reviewed", "reviewer_id": "local-reviewer-reference", "reviewed_at": "2026-10-17T00:00:00Z", "evidence_supported": true}
  }]
}
```

`local_review` is an independent local human assessment of the returned evidence, not a model confidence. The harness checks event/verdict coherence and timeline range; A01/A12 own full JPEG/public-payload validation. A matching timestamp alone does not validate the image, visibility or legal applicability. Rule catalog input is `{"verified_rule_ids": ["..."]}` from A04's reviewed applicable catalog; model output cannot mark its own rule verified.

Metadata contains `model_id`, immutable `model_revision`, optional `adapter_sha256`, `inference_kind` (`synthetic` or `real_local`), `conditions_sha256`, `rules_sha256`, `taxonomy_sha256`, and `final_check_used_for_tuning` (explicit `false` for clean validation). Conditions hash identifies identical sampling, prompt/config, evidence and timing conditions excluding the checkpoint under comparison. Do not fill unknown hashes with invented values in real runs.

## Split and checkpoint gates

Split manifests contain `records`, `state`, `scenario_groups_reviewed`, `approved_by`, `locked_at`, `source_manifest_sha256`, `records_sha256`. Records require nonempty IDs/groups, 64-character content hashes and split `train`, `validation` or `final_check`. Optional `scenario_group_id` and `parent_sample_id`/`source_recording_id` enable related-scenario/window audits. All duplicates, scenarios and descendants stay in one split.

Use `evaluation.audit.records_digest(records)` for the lock digest: records sorted by sample ID, sorted JSON keys, UTF-8, compact separators, no nonfinite numbers. A02's `assignments_sha256` is not silently substituted for this canonical lock digest. A20 approves the lock after scenario review. The audit detects content/group/scenario leakage, conflicting duplicate labels, duplicate sample rows, missing parent IDs, derived split changes and stale lock digests.

Selection compares identical sample-set, ground-truth, protocol, split, rule, taxonomy and conditions hashes. It requires real local inference, reviewed positive and negative coverage, at least six reviewed cases (a team policy, not a statistical guarantee), a valid split lock, timing coverage and evidence/grounding review. Candidate must reduce at least one guarded error count with no guarded regression. Axes are video FP, missed positives including failures, event FP/FN, unprocessed/late results, wrong mappings, unverified sources and unsupported evidence. Final-check remains untouched until selection; disclose any later reuse for tuning.

## Denominators and interpretation

Whole-video TP/FP/FN/TN apply only to completed, human-reviewed cases. Unprocessed positives and negatives are reported separately instead of treating failures as clean negatives. Effective recall is `TP/(TP+FN+unprocessed_positive)` and effective correct rate is `(TP+TN)/all_reviewed`. Completion and deadline-completion denominators include every required sample, including missing predictions and missing annotations. Timing without upload-start origin must remain unknown.

`whole_video_within_deadline` additionally treats late and unknown-timing results as unprocessed. Selection uses these cutoff-aware video counts; `completion.unprocessed_at_deadline_including_unknown_timing` makes the five-minute risk explicit. Status completion and cutoff completion are separate denominators.

Events use deterministic maximum-cardinality, then minimum-anchor-error one-to-one matching of the same type. Wrong types yield FP plus FN; repeated predictions cannot match one event twice. A failed positive contributes all its reviewed events to FN. Tolerances 0.5/1/2 seconds are internal diagnostics; override with repeated `--tolerance`. Timing errors describe matched events with reliable annotated anchors only (optional `anchor_reliable:false` excludes an uncertain anchor), so read their count alongside unmatched events. There is no unsupported conversion to rubric points. Every report carries limitations about tiny samples, genuinely new angles, image correspondence and failure handling.

## Optional proposed-train exploration

`python -m evaluation.exploratory --terminals benchmarks/private/a18-terminal-verdicts.json` joins only A18's six selected distinct proposed-train content hashes to A02's organizer-workbook binary labels. It performs no network, inference or media operation, does not read the heldout split, and makes no training/split-lock claim. It emits aggregate diagnostics to `benchmarks/quality/proposed-train-provided-labels.json`; those are exploratory development observations against provided labels, never reviewed heldout accuracy. Timestamps, evidence and legal grounding remain unreviewed.

Private terminal records are an array or `records` object with `sample_id`, `status`, `verdict`, `public_contract_validated:true` for successful terminal results, and upload-origin `upload_start_to_result_sec`. The CLI waits for exactly six terminal records (`done`, `failed` or `timeout`) before writing any confusion report. Failed/missing/timeout cases stay in the required six-sample denominator and separate from clean negatives. Source labels are retained only for the permitted six content hashes; no filenames, action narratives or workbook timestamps enter the aggregate report.

## Optional strict targets: all attempts, 100% observed criteria, under 10 seconds

The current real target status is `benchmarks/quality/strict-target-unmeasured.json`. The review/data prerequisites are concrete in `qa/vision/strict-target-review-checklist.md`. This optional mode does not modify the immutable original all19 exploratory evidence and makes no training or performance claim.

```powershell
python -m evaluation.strict_targets --truth evaluation/private/reviewed-strict.jsonl --expected-attempts evaluation/private/expected-attempts.json --attempts evaluation/private/strict-attempt-results.jsonl --splits data/splits/locked_split.json --metadata evaluation/private/strict-run-metadata.json --rules evaluation/private/verified-rule-ids.json --output benchmarks/quality/strict-target-run.json
```

This is an offline aggregation command: it performs no API/model/GPU/media operation. Inputs must already exist from the authorized reviewed run. Exit 0 means observed-sample acceptance PASS, 2 means FAIL/UNMEASURED/SYNTHETIC_ONLY, and 1 means invalid input. `evaluation/strict-target-report.schema.json` defines the report.

Expected attempts contain `attempt_id` and `sample_id` and must be frozen before the run. Results contain both IDs, terminal status/verdict/events, `public_contract_validated`, `upload_start_to_full_result_ms` and `timing_origin:"client_upload_start_to_full_result_byte"`. Missing/failed attempts stay in the denominator. Exactly 10,000 ms fails; numerical display rounding never changes the latency comparison.

Truth extends A03 reviewed exports with `review_origin:"human"`, full review and genuine lock binding, `negative_verified:true` for full negatives, and original-frame anchor bindings. Provide `source_timing:{original_timeline_verified:true,original_frame_pts_sec:[...]}` from the actual original frame index; `frames` (or `source_frames`) maps frame IDs to raw `pts_sec`. Human-reviewed events require verified applicable rule IDs plus temporal/exception review. Strict prediction records need existing `anchor_frame_id`, numeric `timestamp_ms` equal to rounded original-frame PTS milliseconds, 1–3 original `evidence_frame_ids`, and independent `local_review` with human provenance plus `evidence_supported`, `timestamp_correspondence_verified`, `rule_applicability_verified` all true. The millisecond field belongs in this internal review/acceptance export; the mandatory jury `timecode_sec` contract remains under A01's ownership. Review flags must originate in real local review; model-assigned booleans do not establish review.

Run metadata records `inference_kind`, actual `execution_verified`, immutable model revision, conditions/rule hashes, `attempt_manifest_frozen_before_run`, `uncached_inputs_verified`, and `heldout_use_disclosed`. A02's latest full registry/scenario/accepted-job audit supplies `exposure_audit_passed` and `exposure_registry_sha256`; do not infer either from an old snapshot. The truth split hash must match the bytes of the supplied approved split. Only locked validation/final-check records qualify; previously used final-check must be disclosed and replaced before an untouched-set claim.

The report separates integer millisecond representation, adjacent original-frame PTS spacing, and error to human-selected original-frame anchors. Sparse inference sampling is not source-frame precision. Event matching defaults to the explicitly labeled internal 1,000 ms diagnostic; optional `--max-anchor-error-ms` adds a separately frozen source-anchor error requirement. No 1 ms action-time accuracy or universal 100% claim is permitted. Synthetic fixtures can satisfy software criteria and still remain `SYNTHETIC_ONLY`.

The named strict-10-second policy rejects a relaxed `deadline_ms` even through its Python API. Source maps/timing flags are trusted **upstream verification exports**, not independent proof from this offline script: A05 must validate canonical frame identity from content hash/raw source PTS/time base/origin and source geometry before the export is marked verified. Retain that audit with the run; model-assigned flags cannot establish identity.
