# W1: executable frozen-encoder candidate

W1 is a separate automated experiment. Organizer-provided whole-video binary
labels train one multiple-instance bag per complete recording. User-provided
approximate seconds train positive eventness bags within a predeclared ±1 s
tolerance. That tolerance is a modeling assumption, not an observed duration.
Frames outside an anchor remain unknown except where a provided negative video
label independently establishes negative eventness. Binary labels never create
nineteen fact states, legal event classes, or start/end boundaries.

`train_weak.py` uses AdamW, finite-gradient checks, clipping, and fixed seeds.
It saves the initial untrained head before fitting, then a real trained
`head.safetensors` and `checkpoint.json`. Each checkpoint binds the exact encoder,
preprocessing, native source PTS, split assignments, label source, cache index,
taxonomy, runtime and optional silver teacher export. It remains an unselected
candidate. The existing human-reviewed E1/LoRA gate is unchanged.

The initial encoder is the immutable SigLIP2 B16 manifest in
`configs/serving/encoders/siglip2-base-patch16-256.manifest.json`. A15's encoder
recipe is retained verbatim, including processor backend/version and no L2
normalization. Root's `extract_features.py` consumes all native source frames,
including the final frame/EOF interval, and publishes complete private cache
manifests. A 4,273-frame source remains one bag; no positive label is copied to
chunks. The first executable cache supports one global view per frame.

The runtime config must pin A20's byte-identical A02 experiment config with
`experiment_config` and `expected_experiment_config_sha256`. Relocated private
paths can override the original Windows paths. The authority and role-label
digests always come from the pinned original. Required runner fields are:

`fact-axis.v1.json` freezes the live catalog's sorted nineteen enabled facts
and three enabled event types. The generic annotation taxonomy contains
twenty-one available facts; taking its first nineteen would change the model
axis. A20 independently checked the frozen axis against the enabled catalog.

- `automated_split`, `scenario_groups`, `source_manifest`, `provided_train_labels`;
- `exposure_registry`, `exposure_inventory_status`, optional `source_inventory_path_override`;
- `feature_cache_index`, `expected_feature_cache_index_sha256`;
- `taxonomy_path`, `expected_taxonomy_sha256`, `fact_axis_manifest`,
  `expected_fact_axis_manifest_sha256`, `output_dir`;
- explicit `compute_release_authorization` with the A18/root release reference.

For coarse supervision also pin `provided_train_anchors`,
`expected_provided_train_anchors_sha256`, and
`expected_coarse_anchor_source_sha256`. Optional machine supervision requires
`machine_annotations` and its exact `expected_machine_annotations_sha256`.
Only train-role teacher records are considered. Incomplete window provenance is
counted and masked; organizer MIL can still train independently. Current
window facts/support do not establish per-frame fact-state labels. The
checkpoint reports actual supervision counts and eligibility separately for
window facts, frame facts, support, boundaries, and coarse eventness.

Run in the isolated cu129 runtime after the GPU owner hands it over:

```sh
python -m training.temporal.train_weak --config /private/W1.local.json --preflight
python -m training.temporal.train_weak --config /private/W1.local.json
```

`--preflight` checks metadata only. Real fitting rechecks the fresh exposure
snapshot and frozen role assignments, then verifies the pinned runtime and
CUDA. It never opens validation or final-check label manifests. No download,
remote model code, active-serving change, or automatic model selection occurs.

Validation is a separate command using `provided_validation_labels`, the frozen
six-attempt plan (`expected_attempts`, `expected_attempts_sha256`), and exact
`checkpoint_dir` / `expected_checkpoint_manifest_sha256`. Also supply
`video_threshold` / `expected_video_threshold_sha256` and a new
`validation_output_dir`. Training writes the predeclared raw-logodds threshold
0.0 before validation is opened. Use each initial/trained checkpoint's own
threshold file. The same frozen plan may be evaluated into separate directories.

```sh
python -m training.temporal.evaluate_weak --config /private/W1-untrained-validation.local.json
python -m training.temporal.evaluate_weak --config /private/W1-validation.local.json
```

Outputs are private `predictions.jsonl`, native-PTS
`temporal-predictions.jsonl`, A10 `metadata.json`, `gate-proof.json`, and
`evaluation.json`. Every planned attempt contributes, including failures.
Head-forward timing uses cached features and does **not** measure the upload to
complete-result target. Binary validation is against organizer labels; sparse
teacher agreement is not fact/event accuracy. Final-check data is excluded from
fitting and calibration.

`checkpoint.load_checkpoint(dir, expected_manifest_sha256=..., device=...)`
returns `(model, A08 HeadIdentity, metadata)` after pinned JSON and safetensors
verification. `cache.load_feature_cache(path)` returns
`(FeatureSequence, CPU_FP16_features[T,1,768], manifest)`. `sequence_inputs`
constructs only numeric feature/time/geometry inputs; filenames, groups, source
hashes and labels never enter the head. A08 must abstain on untrained or
uncalibrated fact/support/boundary axes.

Actual synthetic CPU checks:

```sh
python -m unittest training.test_weak_runtime training.test_feature_cache
```

These execute a real optimizer and strict checkpoint reload, full 4,273-frame
MIL, frozen-input gradients, NaN padding isolation, unknown masks, and cache
corruption/partial-stream rejection. They establish implementation behavior,
not quality or GPU performance on the supplied corpus.
