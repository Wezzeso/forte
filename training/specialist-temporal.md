# Tracked-object temporal specialist

This optional path trains a real, randomly initialized, small PyTorch Transformer
over detector counts per camera zone, hand/object **2-D proximity**, explicit seat
occupancy and optional cached local frame embeddings. It is separate from
`training/temporal/head.py`, which uses a different frozen-frame feature contract.
There is no trained checkpoint, measured detection gain or enabled inference type
provided by this implementation. It is not connected to the frozen fast candidate.

`src/forte/fast/tracked_head.py` imports torch only when constructing or loading a
model. `training/specialist_temporal.py --preflight` is CPU-only and performs no
model downloads, CUDA calls or video decoding. Training requires the configured
installed torch version and an explicitly selected CUDA device; CPU training is
possible only with `device: "cpu"` and `allow_cpu_training: true`.

## Inputs

First prepare the reviewed detector dataset with
`python -m training.specialists.prepare`. The pinned `dataset-lock.json` supplies
source recording hashes, ownership/synthetic provenance, consent records, source
sample IDs, scenario groups, splits and camera IDs. Organizer/jury media cannot
train this head. Source recordings and scenario groups cannot cross partitions;
one entire own-recording camera remains `holdout`. Synthetic ancestors cannot
come from organizer media or another partition.

Create a private schema with `default_feature_schema`:

```python
from forte.fast.tracked_head import default_feature_schema
schema = default_feature_schema(
    classes=["person", "id_card", "banknote_stack", "key_in_lock", "detector_device", "cup"],
    zones=["desk", "safe", "door"],
    contact_classes=["id_card", "banknote_stack", "key_in_lock", "detector_device"],
    seats=["cashier", "client"],
    producer={
        "local_only": True,
        "detector_checkpoint_sha256": "<exact hash>",
        "detector_gate_sha256": "<exact passed detector gate hash>",
        "tracking_config_sha256": "<exact tracker/sampling recipe hash>",
        "layout_sha256": "<exact camera zone layout hash>",
        "hand_model_sha256": "<exact local hand checkpoint hash>"
    }
)
```

All hashes must be lowercase, 64-character SHA-256 values. The producer descriptor
pins the detector, its gate, hand model and tracking/layout recipe that generated
these cached observations. Keep the same producer identity for inference. The
recipe must include preprocessing, calibration, confidence thresholds and track
association settings. The schema's ordered axes define each feature column:
counts by zone/class, proximity by zone/class, seats, then optional embeddings.
Counts are clipped and divided by the fixed `max_count` (default 8).

Optional embeddings require an explicit schema `embedding` descriptor containing
`model_id`, `dimension`, `local_only: true`, `checkpoint_sha256`,
`extractor_config_sha256`, `reviewed_by` and `reviewed_at`. This path has no encoder
loader or implicit SigLIP fallback. A feature row without a matching cached vector
has an unknown embedding mask. Do not insert fabricated vectors.

Each cached feature file has this form (placeholders are intentionally invalid):

```json
{
  "schema_version": "forte.tracked-head.sequence.v1",
  "feature_schema_sha256": "<schema file SHA-256>",
  "source_recording_sha256": "<original source recording SHA-256>",
  "complete_recording": true,
  "duration_sec": 1.0,
  "frames": [
    {
      "t": 0.0,
      "producer_identity_sha256": "<canonical_sha256(schema.producer)>",
      "counts": {"desk/id_card": 1, "desk/banknote_stack": 0},
      "contacts": {"desk/id_card": true},
      "seats": {"cashier": true, "client": null}
    },
    {
      "t": 0.5,
      "producer_identity_sha256": "<canonical_sha256(schema.producer)>",
      "counts": {"desk/id_card": null},
      "contacts": {"desk/id_card": null},
      "seats": {"cashier": true, "client": null}
    }
  ],
  "label_intervals": [
    {
      "type_id": "NO_CLIENT_ID",
      "start_sec": 0.0,
      "end_sec": 1.0,
      "label": 0,
      "human_reviewed": true,
      "reviewer": "<human reviewer>",
      "reviewed_at": "<timestamp>"
    }
  ]
}
```

Use exact source-relative timestamps `0, 0.5, 1.0, ...`, with exactly
`ceil(duration_sec / 0.5)` rows covering the full recording. Long clips exceeding
`model.max_steps` fail; they are never silently truncated. A positive count is an
observed track count, not the number of raw duplicate boxes. A **known zero** or
false proximity value requires a calibrated producer with adequate channel
visibility; missing detections, missing hands, occlusion, model failures and track
switches must be `null` or omitted. Missing values are numerical zero plus a false
observed mask, so they remain distinguishable from reviewed zero observations.
Seats accept only `true`, `false` or `null`; strings and numeric booleans fail.
People counts do not supply actor roles or seat occupancy.

Intervals are half-open `[start_sec, end_sec)` and cannot overlap for one type.
Only explicit human-reviewed labels `0` and `1` train the model. Everything outside
those intervals stays `-100` and is excluded from binary cross-entropy. Detector
boxes and a whole-video binary violation verdict do not create event intervals.
Every configured output type needs reviewed positive and negative training grid
points; a wholly unsupervised output cannot become supported from validation alone.
For calibration, fully review positive and clean validation videos per type;
partially reviewed videos cannot set a whole-video false-positive threshold.

The private manifest links every sequence to one exact dataset-lock source:

```json
{
  "schema_version": "forte.tracked-head.data.v1",
  "frozen": true,
  "reviewed_by": "<human reviewer>",
  "reviewed_at": "<timestamp>",
  "dataset_lock_sha256": "<dataset-lock.json file SHA-256>",
  "feature_schema_sha256": "<schema file SHA-256>",
  "records": [
    {
      "sequence_id": "<unique sequence ID>",
      "source_sample_id": "<exact dataset-lock sample ID>",
      "source_recording_sha256": "<exact source hash>",
      "scenario_group": "<exact source scenario group>",
      "camera_id": "<exact source camera ID>",
      "split": "train",
      "origin": "own",
      "feature_path": "train-take-001.json",
      "feature_sha256": "<feature file SHA-256>",
      "human_reviewed": true,
      "reviewer": "<human reviewer>",
      "reviewed_at": "<timestamp>"
    }
  ]
}
```

The real manifest needs `train`, `validation` and `holdout` records. The training
manifest accepts one complete sequence per source recording hash, so duplicated
exports cannot inflate independent video support. The training
CLI verifies holdout lineage and hash syntax but does not open holdout feature
files or labels. Run the camera holdout only after training, checkpoint selection
and threshold selection are frozen. Native source video is never opened by this
trainer. Human review covers feature availability and event intervals, as well as
their source association. IDs, hashes, filenames, groups and labels never enter
the model's numerical feature tensor.

## Commands and output

Fill in `configs/training/specialist-temporal.example.json`, then run:

```powershell
python -m training.specialist_temporal --config configs/training/specialist-temporal.example.json --preflight
python -m training.specialist_temporal --config configs/training/specialist-temporal.example.json
```

Preflight writes nothing. Training refuses to overwrite an existing run directory.
The model uses learned positions, a mask-aware linear projection, Transformer
encoder layers and per-timestep per-type logits. The optimizer sees training
sequences only; no hidden validation normalization is fitted. Fixed-step AdamW,
seeded shuffling, deterministic torch algorithms and gradient clipping are used.
Unsupported deterministic CUDA operations fail rather than silently change mode.

The run contains `head.pt` (tensor state dictionary), `model-config.json`,
`feature-schema.json`, `thresholds.json`, `validation-metrics.json`,
`training-config.json` and `artifact.json`. The artifact binds exact file hashes,
the input schema and manifest, dataset-lock hash, runtime version and config hash.
It always says `trained_unselected`, has empty `approved_types` and requires
independent verification. Validation metrics include exact predictions, known
frame precision/recall, full-video precision/recall and peak distance to the
nearest positive grid point. They are development measurements, not a public
evaluation or 95% generalization claim.

Each per-type threshold is at least 0.5 and strictly above the largest fully
reviewed clean validation video's peak. Types without both clean and positive
fully reviewed validation support remain unsupported. This is a zero false alarm
development calibration, not promotion. Unknown labels cannot create negatives.

## Runtime gate and API

The runtime is deliberately not automatically selected. Integration must first
run a frozen full-video baseline/candidate evaluation, grouped scenario folds,
the untouched camera holdout and timing checks. A separately reviewed gate must
use schema `forte.tracked-head.full-video-gate.v1`, `state: "passed"`, reviewer and
timestamp, and bind `artifact_sha256`, `checkpoint_sha256`, `model_config_sha256`,
`feature_schema_sha256`, `thresholds_sha256`, `evaluation_report_sha256` and
`evaluation_data_manifest_sha256`. It must explicitly contain:

- `full_video_type_evaluation: true`, `independent_vlm_verification_required: true`
  and `procedural_proof_rules_required: true`;
- `enabled_types` listing only measured, supported types;
- `required_channels_by_type` listing nonempty exact feature names returned by
  `feature_names(schema)`;
- `per_type` evidence with `passed: true`, positive and clean video support,
  `false_positive_videos: 0`, `recall >= 0.95`, `strict_improvement: true`,
  `no_type_regressions: true` and `timecode_gate_passed: true`.

These gate fields are human-reviewed evaluation assertions. This implementation
does not manufacture the full-video evaluation or a passed gate from training
validation statistics. The evaluator must compute type/time matches against
reviewed events with a predeclared time tolerance, preserve previously correct
types even on already-positive videos and retain exact reports. In particular,
the detector's object-class gate cannot approve the temporal event model.

```python
from pathlib import Path
from forte.fast.tracked_head import TrackedTemporalHead

head = TrackedTemporalHead.load(
    Path("/private/tracked-head/artifact.json"),
    expected_artifact_sha256="<exact artifact file hash>",
    gate_path=Path("/private/tracked-head/full-video-gate.json"),
    expected_gate_sha256="<exact separate gate file hash>",
    device="cuda",
)
proposals = head.propose(observed_half_second_rows, duration_sec=video_duration)
```

Loading verifies the complete checkpoint/config/schema/threshold/metrics inventory
and exact recorded torch version before importing torch, then uses
`torch.load(..., weights_only=True)` and strict
state-dictionary matching. No artifact discovery or model download occurs. Each
enabled type abstains if any of its required feature channels is unknown anywhere
in the recording. A proposal carries the peak time, probability, artifact/gate
identity and explicit independent-verification flags.

Proposals must pass the analyzer's VLM verification and existing procedural proof
rules. A probability does not create transaction boundaries, cashier identity,
complete visibility or a reviewed negative action interval. In particular,
`NO_CLIENT_ID`, `NO_AUTHENTICITY_CHECK` and `NO_RECOUNT` retain those requirements.
The trained head must not synthesize `TemporalObservation` or `CoverageInterval`
records to bypass them. A person still makes the final decision.

## Local checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_specialist_temporal_head.py -q
```

The synthetic tests cover mask semantics, strict seat booleans, producer/embedding
identity, interval review and overlap, camera/source leakage, exact feature hashes,
preflight isolation, no-torch failure, zero-FP threshold selection, runtime type
gates and unknown-channel abstention. They do not claim real training speed,
model accuracy, checkpoint quality or GPU availability.
