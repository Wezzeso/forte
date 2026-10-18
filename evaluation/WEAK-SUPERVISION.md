# Operational provided-label evaluation without manual annotation

This separate track implements the user's authorized automatic first pass. **Human event labels are not required to train/evaluate a weak video model under A02's explicit weak gate.** The original human-gold evaluator remains available for stronger claims; its absence never blocks this operational experiment.

A02 exports independent organizer-provided **whole-video binary labels** by partition. A03/local-teacher outputs may supervise masked silver facts and event brackets; agreement with those outputs is not fact/event accuracy. The evaluator uses only `provided_binary_labels.validation.jsonl`. It never joins teacher predictions as binary ground truth, and never reads final-check label/media files.

Actual metadata prepared: 28 train, 6 validation and 6 final-check unique contents under automatic provisional grouping. The six validation labels contain three positives and three negatives. Automatic groups have explicit machine provenance and unvalidated scenario independence. The weak authorization freezes assignment and per-role label hashes; it does not change `locked:false`, `training_authorized:false`, or human-review flags on the separate human-gold split.

## Actual runnable workflow

After A18/root releases compute and the accepted-job exposure snapshot is fresh, run the **offline** validation gate:

```powershell
python -m evaluation.weak_supervision gate --experiment-config data/splits/automated_experiment.run-config.json --provided-labels data/manifests/provided_binary_labels.validation.jsonl --output benchmarks/quality/weak-validation-gate.json
```

This calls A02's `validate_automated_experiment_inputs(..., label_split="validation")`, checking authorization, externally pinned automatic assignments, independent source-workbook and validation-role label hashes, latest registry/coverage and known group leakage. The first preflight failed only stale inventory; the next actual check passed after the fresh A18 handoff. A past passing proof is not continuing freshness authority: rerun at actual execution.

A10 has already frozen six expected validation attempts **before new weak-model inference** in `evaluation/private/weak-W1-validation.expected.json`. Its aggregate hash/count binding is `benchmarks/quality/weak-W1-validation.plan-binding.json`. The plan creates no inference/training permission and must not be rewritten after observing outcomes.

When A09's actual validation runner produces predictions and metadata with those exact attempt IDs:

```powershell
python -m evaluation.weak_supervision evaluate --provided-labels data/manifests/provided_binary_labels.validation.jsonl --expected-attempts evaluation/private/weak-W1-validation.expected.json --predictions training/runs/W1/validation/predictions.jsonl --splits data/splits/automated_provisional_split.json --experiment-config data/splits/automated_experiment.run-config.json --weak-gate-proof benchmarks/quality/weak-validation-gate.json --metadata training/runs/W1/validation/metadata.json --output benchmarks/quality/weak-W1-validation.json
python -m evaluation.weak_supervision select --baseline benchmarks/quality/weak-baseline-validation.json --candidate benchmarks/quality/weak-W1-validation.json --output benchmarks/quality/weak-W1-selection.json
```

The prediction/metadata paths are the agreed producer interface; do not create fake files or call synthetic output an actual run if training/inference has not occurred. Baseline comparison requires the same validation population/multiplicity, label artifacts, assignment freeze and conditions. Initial threshold is predeclared zero raw violation log-odds (`>= 0` means violation), frozen before examining validation results. Later same-validation threshold search must disclose in-sample calibration reuse; it is never pristine final-check accuracy.

## Producer fields

Prediction records require `attempt_id`, `sample_id`, `status`, binary `verdict`, and `prediction_validated:true` for a structurally checked head result. Failed/missing attempts stay in the denominator. Public-service completion additionally requires `public_contract_validated:true`; upload-start-to-last-byte latency additionally requires `upload_start_to_full_result_ms` and `timing_origin:"client_upload_start_to_full_result_byte"`. Worker/head compute time cannot satisfy the `<10000 ms` service target. Failure observation time may be retained separately as `terminal_observed_elapsed_ms`.

Metadata requires `inference_kind:"real_local"`, `execution_verified:true`, actual immutable `model_revision`, `conditions_sha256`, `comparison_scope` (`weak_model_only` or `full_service`), `expected_attempts_frozen_before_run:true`, and `threshold_selection_sha256` (SHA256 of actual frozen threshold-file bytes). Include `head_identity` as A08's exact JSON `model_dump(mode="json")` when requesting A08 machine readiness. It binds actual encoder/checkpoint/feature/taxonomy hashes and tensor-axis order.

The frozen threshold artifact uses `forte.weak_video_threshold.v1`, score axis `violation_log_odds`, finite `minimum_video_score`, comparison `greater_than_or_equal`, selection method, explicit `validation_labels_used_for_selection`, actual head/identity/feature bindings and provided validation-label hash. A08 independently checks its contents; A10's provided-label adapter checks only matching artifact bindings.

Provided label rows use A02's exact fields: `provided_video_label` boolean, `provided_label_origin:"organizer_workbook"`, source-workbook SHA, `target_scope:"video_binary_mil_only"`, `teacher_used_as_binary_ground_truth:false`, content/group identities and split `validation`. Binary labels must not contain facts, events or fact targets. Canonical label digest is exactly A02's `label_records_digest`: sorted JSON keys, UTF-8, one default-spaced record plus newline in source order. Assignment digest is A02's sorted-key, two-space-indented records JSON plus newline. Neither is silently replaced with the human-track `records_digest`.

## Metrics and selection

`forte.weak_operational_evaluation.v1` reports TP/FP/FN/TN and separate unprocessed positives/negatives. Unique-content confusion uses the **first frozen planned attempt** per content; repeated attempts never increase independent video sample size. All-attempt confusion/completion/latency remains separately visible, including rejected uploads and results with no durable job. Missing validation contents, changed role/source/assignment pins, unknown groups or a failing exposure gate prevent model-selection eligibility without erasing diagnostic counts.

Provisional head selection may proceed below the requested 100%/10-second targets when total unique-content binary errors (including failures) decrease without worsening completion/repeat consistency. This equal-count error policy is a team choice, not official rubric scoring; FP/FN tradeoffs remain explicit. `weak_model_only` selection makes no public-service/evidence-readiness claim; `full_service` selection also guards public completion and all-attempt 10-second failures. Requested binary and latency targets are explicit separate booleans. Fact/event/timing/evidence/legal accuracy stays **UNKNOWN**. Even a perfect provided-label validation result is a finite, automatically grouped development result, not universal or human-gold accuracy.

## A08 attestation API

```python
from evaluation.weak_supervision import (
    validate_weak_attestation,
    verify_provided_evaluation_for_manifest,
)

integrity = validate_weak_attestation(report, expected_bindings)
provided_proof = verify_provided_evaluation_for_manifest(report, machine_manifest, selection_report)
```

Reports bind `split_assignments_sha256`, `provided_labels_sha256`, `input_set_sha256`, `predictions_sha256`, expected-attempt/population digests, actual model/threshold identity, registry file/tip and coverage hashes. `provided_evaluation_attestation_sha256` is the canonical report digest excluding itself; a digest binds integrity, not external authorship. The A08 adapter checks model/threshold/exposure/selected-evaluation bindings and returns provided-binary/latency observations. It deliberately returns `teacher_support_agreement_verified:false` and `threshold_file_contents_verified:false`: A08 must compose real independent teacher-support and threshold-content checks before opening its full machine gate.

Schema: `evaluation/weak-operational-report.schema.json`. All operations above are file-only; they issue no model, API, media-decoding or GPU call.
