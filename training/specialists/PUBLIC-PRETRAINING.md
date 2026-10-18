# Public object pretraining

Public pretraining is separate from the own-recording specialist workflow. It
can teach object appearance before the team records its own footage. It does
not establish ID-before-cash, authenticity checks, safe-key procedures, or a
banking violation. Its outputs use `forte.specialists.public-run.v1`, remain
unselected, and cannot pass the existing own-recording/video promotion gates.

The planned combined detector has exactly two labels, `single_banknote` and
`id_card`. Their prediction IDs are 0 and 1; COCO category IDs are 1 and 2.
Public banknotes are never relabelled as stacks, and a generic ID document is
never relabelled as a successful customer identity check.

## Sources and frozen groups

KZ-BD v1 comes from the author's [Mendeley deposit](https://data.mendeley.com/datasets/dny3dgvvw8/1)
under CC BY 4.0. Preserve the exact source archive hash, license evidence,
annotation-publication evidence, and attribution in its lock. The published
LabelImg annotations contain both the enclosing currency and internal feature
boxes. The importer derives a single unique enclosing currency box only when
the source annotations support it; ambiguous images are quarantined. Coin
images are background examples for the banknote class. Inner security-feature
boxes do not become separate banknotes.

Freeze the following denomination policy before any model metric is observed:

| Role | Paper denominations | Coin denominations |
| --- | --- | --- |
| Train | 500, 1,000, 2,000, 5,000 | 1, 2, 5, 10 |
| Development | 10,000 | 20, 50 |
| Sealed test | 20,000 | 100, 200 |

Only the publisher's original training portion of training denominations can
fit the model. The publisher's validation images of those denominations are
excluded from fitting and development. Reserved development and test
denominations use all available source images for support. Exact duplicates
and conservative dHash256 near-duplicate components cannot cross final roles;
cross-role components are quarantined. Each denomination is a document/design
group. The source does not provide trustworthy recording or scene identities;
the importer records that missing information explicitly. This protocol does
not prove independence of real camera scenes.

MIDV-Holo uses the official fictional-document annotations and exact template
quadrangles. Preserve its CC BY-SA 2.5 license, author and Generated Photos
attribution, source-member hashes, and transformation provenance. Template
types 1–6 train, 7–8 develop, and 9–10 remain sealed. All physical samples,
clip frames, threat variants, and template-derived composites inherit their
template's role. Passport pages are not silently converted to ID-card labels.
No organizer frame is a source image, synthetic background, training example,
or development example.

The public importers and merger retain separate source locks. They do not
invent own-recording consent or local human-review fields for publisher labels.
`forte.specialists.public-dataset-lock.v1` binds the actual source/license and
annotation bytes, class map, frozen split policy, derived-image transformations,
COCO exports, duplicate quarantine, and exported image hashes.

## Bounded training

Use the isolated RF-DETR environment and local verified weights. The source
recipe is pinned to RF-DETR 1.11.2. The selected recipe uses Small at 512 pixels,
a frozen encoder, bf16, physical/evaluation batch size 1, accumulation 8,
fixed image scale, no EMA, and no compilation/CUDA graphs. Full-fit resource
use still needs measurement. The isolated Small synthetic
forward/backward/AdamW smoke measured 568 MiB allocated, 684 MiB reserved and
1,078 MiB host-observed trainer usage; that establishes a short resource check,
not public-data accuracy or full-fit throughput. The selected allocator cap is
5,120 MiB (the runner permits at most 5,632 MiB), with at least 1,024 MiB
additional free memory required
before model allocation. A process allocator cap does not reserve memory or
control allocations made by other processes; serialize trainer jobs.

The predeclared loader gives banknote, ID-card, and verified coin-background
groups equal expected sampling mass. Replacement draws retain their original
source/template identities and add no independent evaluation support. For 80%
of positive training draws, a synthetic zoom-out places the image on a plain
canvas with its longest box drawn uniformly from 12–96 pixels at the 512-pixel
reference resolution. Boxes and areas follow the exact raster transform;
development and sealed examples retain their exported pixels.

Start from [the public training configuration](../../configs/training/specialists-public-train.example.json),
set exact dataset/weight hashes and the actual shared-GPU release, and retain
the predeclared recipe reference. Use distinct output directories for smoke
and fitting:

```sh
python -m training.specialists.public_train --config training/private/public-train.json --preflight
python -m training.specialists.public_train --config training/private/public-smoke.json --smoke
python -m training.specialists.public_train --config training/private/public-train.json
```

The initial public-development baseline is the pretrained backbone with a
new, untrained target-class head. COCO has no banknote class, so COCO class
outputs cannot be reused as banknote labels. The baseline uses an evaluation
trainer that cannot select the eventual training checkpoint. One fixed recipe
then fits training examples, with development-only early stopping and best
checkpoint selection. The sealed test is evaluated once after selection.
`sealed-test-attempt.json` records the attempt even if evaluation fails.

The current two-model experiment defers this step. Set `defer_sealed_test: true`
and use `--defer-test`: the detector fit returns
`trained_unselected_pending_test` with zero test evaluations. After the detector
and compact crop VLM both select their checkpoints using development data,
freeze both checkpoints, runs, configurations, and evaluation protocols in the
shared two-member cohort. The global sealed-inventory receipt and exclusive
participant attempt receipts prevent a different output directory from
reopening the same sealed set. The detector's separate final-evaluation command
requires the cohort receipt and exact hash, frozen run hash, and protocol path;
it leaves the selected run immutable and writes a separate result sidecar.

Real run evidence includes finite observed losses, optimizer-step counts,
changed trainable-parameter hashes, unchanged frozen-parameter hashes, the
selected checkpoint hash, per-class development/test metrics, and measured
peak allocator memory and elapsed time. A smoke result has a separate state
and does not count as completed training or a detection-accuracy gain.
The final protocol also predeclares object precision/recall and coin-negative
image false-positive rate at confidence 0.5 and IoU 0.5. These thresholds are
fixed before the cohort test and are not tuned from its result.

The runner uses the official Lightning training primitives because the old
high-level callback dictionary does not control the 1.11.2 training loop.
Lightning's timer and batch callback reserve evaluation time and stop fitting
near the explicit UTC deadline. They cannot interrupt a hung CUDA operation;
the job owner must retain an external process watchdog for the hard stop.
The current requested hard stop is 11:01 on 18 October 2026 in Asia/Qyzylorda
(06:01 UTC). The configuration must not extend it during a run.

## Limits of the result

Public heldout object accuracy measures transfer between these public groups.
It is not a measurement of unseen bank footage, banking rules, event timecodes,
or the supplied 40-video corpus. Keep `serving_eligible` false and
`approved_types` empty. Domain promotion still requires the untouched
own-camera holdout and the hash-bound full-video protocol in the
[specialist workflow](README.md). Inspect failures by cause, retain the failed
run, and collect new independent data before claiming generalization after
tuning on a previously sealed set.

API references: [RF-DETR 1.11.2 configuration](https://github.com/roboflow/rf-detr/blob/1.11.2/src/rfdetr/config.py)
and [official custom-training guide](https://github.com/roboflow/rf-detr/blob/1.11.2/docs/learn/train/customization.md).
