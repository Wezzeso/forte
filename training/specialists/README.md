# Small-object specialist workflow

This workflow prepares an RF-DETR object-detector candidate for the objects the
current video pipeline struggles to resolve. It separates preparation, fitting,
held-out evaluation, and release evidence. A completed fit has status
`trained_unselected`; it does not activate a detector in serving or establish an
accuracy improvement. The plan's 40/40 and 95% targets remain targets until an
eligible run measures them.

For licensed third-party image pretraining before own footage is available,
use the separate [public pretraining workflow](PUBLIC-PRETRAINING.md). Its
publisher provenance and public holdouts do not replace this workflow's
own-recording consent or domain-promotion evidence.

Run commands from the repository root. Keep private manifests, labels, images,
weights, and per-recording outputs in the project's ignored private storage.
Supply the actual reviewed manifest and configuration files in place of these
example paths:

```sh
python -m training.specialists.prepare --manifest training/private/specialists/recordings.json --output training/private/specialists/dataset
python -m training.specialists.train --config training/private/specialists/train.json --preflight
python -m training.specialists.train --config training/private/specialists/train.json
python -m training.specialists.evaluate --config training/private/specialists/evaluate.json
python -m training.specialists.gate --run training/private/specialists/run/run.json --metrics training/private/specialists/evaluation/metrics.json --output training/private/specialists/gate.json
```

The commands validate their own schemas. The filenames above are examples. Start
with the [source manifest](../../configs/training/specialists-manifest.example.json),
[training config](../../configs/training/specialists.example.json), and
[evaluation config](../../configs/training/specialists-evaluate.example.json).
Use those templates and command help for the exact fields supported by the
installed code; do not manufacture missing review, consent, source, split, or
hash evidence to make a preflight pass.

## Filming and labels

Use consenting participants and footage the team is permitted to use for model
training. Record consent and permitted use in the source manifest. Avoid real
customer documents, account details, and banknotes containing unnecessary
personal information; use approved props where possible. Keep originals and
consent records private.

Record a matching correct and incorrect action from each camera position. Start
with ID-card versus cash handovers, a key left in versus removed from a safe
lock, use versus omission of the authenticity detector, and desk objects that
can be confused with a cup. Include empty locks, occluded objects, blue cards
next to blue banknotes, hands holding both objects, and people standing in the
doorway. A doorway presence alone does not establish an unauthorized person.

Film at 1080p when available, then make 464-by-848 derivatives for the intended
input condition. Record source hashes, camera IDs, scenario IDs, timestamps,
and each derivative's relationship to its original. Preserve aspect ratio and
document any crop or padding. Higher-resolution originals help human review;
they do not make an invisible object visible in the model's low-resolution
input. Do not use generated super-resolution detail as evidence.

Reserve one whole camera position for the own-recording hold-out before
fitting. Group every original, extracted frame, crop, re-encode, and augmented
copy by source recording and scenario. All derivatives inherit their source's
split. A mirrored or rotated copy of a training clip is not independent
held-out evidence. Use remaining camera positions for training and development
validation. Augment training data only; choose settings and thresholds using
development data before opening the hold-out.

Human reviewers must verify boxes and class identities in the actual model
inputs. Pre-labels from SAM, a VLM, or the current service are proposals. They
remain machine labels until reviewed. Mark uncertain and occluded objects as
unknown rather than silently treating them as negatives. Keep a reviewed
annotation's identity, revision, source hash, reviewer, and review status with
the exported labels. For key-in-lock examples, label the visible lock state
consistently; a 4-by-6-pixel ambiguity may require abstention even after training.

Use stable definitions for `id_card` and `banknote_stack`, and record object
boxes separately from interaction and event labels. A box near a detector does
not prove a banknote was checked; a hand touching a safe does not prove a key
was removed. Temporal contact and recount labels need reviewed event intervals
and enough visible context. Do not infer those labels from a whole-video
violation flag.

Keep the 40 organizer-supplied clips for evaluation in this specialist track.
Do not include their frames or derivatives in specialist training, threshold
calibration, or synthetic examples. Never train on jury test videos. A supplied
evaluation result cannot substitute for the independent camera hold-out.

## Preparation and fitting

`prepare` validates the frozen manifest, source and annotation provenance, and
the expected organizer-inventory SHA-256. It produces the local dataset export
and `dataset-lock.json` with its hashes. Review failures at their source instead
of editing output metadata to bypass a check. Keep the prepared export immutable
for a run. RF-DETR's documented detection dataset layout uses COCO annotations
and split directories; the local preparation code is responsible for exporting
the contract accepted by the pinned dependency.

`train --preflight` checks the local configuration and required input metadata
without importing a model. It is preparation evidence, not training or
model-quality evidence. The dependencies are pinned to `rfdetr[train]==1.11.2`
and `supervision==0.29.1`, using
`RFDETRSmall`. Real training needs a compatible, isolated RF-DETR runtime, its
pinned initial checkpoint, and a GPU allocation that does not interfere with
serving or load tests. Pre-stage dependencies and initial weights; retain their provenance and
licenses. Do not upgrade the active serving environment to satisfy training.

The fitted `run.json` retains the checkpoint and SHA-256 provenance tying it to
the configuration, reviewed data, split, and runtime. The output remains
`trained_unselected`. Training loss, upstream mAP, and a saved checkpoint do not
select the candidate for Forte's violation decisions. Time and throughput
estimates in the roadmap need measurement on the actual GPU and workload.

Each attempt has local `run.json`/`config.json` provenance, including a failed
attempt after fitting starts. Install MLflow in the separate experiment runtime
to export selected hashes and scalar metrics to a local SQLite store:

```sh
python -m training.specialists.track --run training/private/specialists/run/run.json --metrics training/private/specialists/evaluation/metrics.json --tracking-dir training/private/mlflow
```

For a failed or incomplete fit, omit `--metrics`. This exporter uses a local
database and artifact directory, disables URL selection, and logs only selected
provenance fields and recomputed scalar metrics. It does not log media, labels,
frame detections, weight files, or enable autologging. JSON provenance remains
available if the optional `mlflow` dependency is missing.

## Evaluation and gate

Evaluate the exact trained checkpoint against the untouched own-recording
camera hold-out using thresholds fixed before that hold-out is read. Preserve
the checkpoint, dataset, split, threshold/configuration, prediction, and metrics
hashes so that the reported numbers can be traced to one immutable run. Report
per-class TP, FP, FN, precision, and recall in `metrics.json`, with detections in
`predictions.json`, including support counts and failures. A class with no
positive support cannot establish its recall. COCO category IDs are one-based;
the RF-DETR prediction ID to `class_names` mapping is zero-based. Preserve the
explicit mapping when comparing detections with labels.

The release evidence gate requires `id_card` recall of at least 0.95,
`key_in_lock` recall of at least 0.95, and zero total false positives on the
held-out set, with matching run and metrics hashes. Missing, stale, mismatched,
or failing evidence does not pass. Zero observed false positives on a finite
hold-out does not guarantee zero false alarms on new recordings.

A default gate pass approves object classes and keeps `approved_types` empty.
At least 20 positive instances for each required class and 20 completely
object-free own-recording hold-out frames are required. Because the configured
`person` class is exhaustive, empty-room frames are needed for this clean-image
count; negative lock/card crops alone do not count as object-free full frames.
Full video/API regression checks must then establish:
whole-video verdicts, type and timecode correctness, evidence frames, the clean
clips, and concurrent-upload latency. Object recall alone does not establish
violation accuracy. Keep the baseline available until those checks select a
candidate. Two signals that use the same visual evidence can share errors;
agreement is a decision rule to evaluate, not proof of statistical independence.

The evaluator can exercise newly proposed event types without activating them
in the service. After collecting reviewed per-type labels for all 40 evaluation
videos, run the real baseline and a candidate-only run with
`tools/fast_eval.py --candidate-types TYPE1,TYPE2` and the appropriate complete
configuration. The optional promotion command consumes the two live reports:

```sh
python -m training.specialists.gate --run training/private/specialists/run/run.json --metrics training/private/specialists/evaluation/metrics.json --video-validation benchmarks/private/candidate.json --baseline-video-validation benchmarks/private/baseline.json --approve-type TYPE1 --approve-type TYPE2 --output training/private/specialists/video-gate.json
```

This recomputes whole-video type results, requires the same exact 40 input
hashes/labels, 19 clean videos with zero alarms, no failed inference, per-type
recall at least 0.95, strict per-type recall gain over baseline, no previously
correct verdict becoming wrong, and the exact class-gated checkpoint identity.
Requested types also need explicit timestamp labels, one-to-one event matching
at the same predeclared tolerance (at most 2 seconds), event recall at least 0.95,
zero unmatched extra events, and no baseline event-match regression. Duplicate
events and a correct type with an incorrect timecode cannot satisfy this gate.
The promoted gate also binds the candidate's effective analyzer-policy hash;
changing decision thresholds or verifier policy requires fresh live validation.
Binary-only labels, cached reports, incomplete reports, and hand-edited summary
metrics cannot unlock a type. This finite-corpus gate does not establish unseen
video accuracy. The output contains only the explicitly requested passing types.

Do not tune a threshold after reviewing a failed hold-out and then reuse the
same set as an untouched test. Record that exposure, retain the failed result,
and collect a new camera/source hold-out for a new generalization claim.

## Existing LoRA and temporal commands

The reviewed VLM LoRA route remains documented in
[the training README](../README.md). Its entry point is
`python -m training.train --config PATH [--preflight]`, and
`python -m training.export_provider --config PATH --output PATH` prepares a
local provider-input export after the same review and split checks. That route
requires reviewed observation targets and its locked split/runtime contract;
specialist boxes do not automatically satisfy those requirements.

The separate [weak temporal workflow](../temporal/README.md) uses
`python -m training.temporal.train_weak --config PATH [--preflight]` and
`python -m training.temporal.evaluate_weak --config PATH`.
`python -m training.temporal.prepare_run train ...` and its `evaluate`
subcommand write pinned private runner configurations. Its whole-recording
binary labels and coarse eventness anchors do not create class-specific
fact states, contact labels, or event boundaries. The specialist policy above
does not retroactively characterize or authorize a previous temporal run.

## Remaining real-world inputs

An executable workflow cannot supply consenting actors, footage from new camera
positions, human-reviewed boxes/contact intervals, an untouched hold-out,
compatible GPU time, or a passing checkpoint. Missing inputs keep the candidate
unselected. No actual detector training, quality gain, unseen-video accuracy,
or serving speed is claimed by this guide.

Before submission, verify commercial-use permission for the exact dependency
versions, model weights, and any labeling aid used. Source-code and
checkpoint licenses can differ. The upstream documentation is useful API
guidance but can change: validate constructor, training arguments, checkpoint
names, and prediction behavior against the specific RF-DETR version pinned by
this repository, rather than installing the latest package into an existing
runtime.

Official references: [RF-DETR training guide](https://rfdetr.roboflow.com/learn/train/)
and [RF-DETR Python API](https://rfdetr.roboflow.com/reference/rfdetr/).
