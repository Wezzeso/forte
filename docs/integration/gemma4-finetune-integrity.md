# Gemma 4 public-data adaptation: integrity gates

18 October 2026. This is a protocol audit, not a training-completion, accuracy or
deployment claim. The parent session owns execution and measured result records.

## Scope and selection

The proposed adaptation trains the local Gemma 4 model on licensed public object
data only. The supplied bank recordings, their frames/crops/backgrounds,
annotations, pseudo-labels and transformed copies do not enter this public-data
fit, calibration or checkpoint selection. The original brief permits supplied
labels in training/calibration only through true leave-one-video-out folds, with
every derivative of each original grouped in that original's fold. Replaying a
fixed fitted model is not LOVO.

The 40 supplied recordings remain exposed, in-sample development regressions.
Their labels are read by the scorer after inference. Bank benchmark results must
not choose adapter checkpoints, training duration, prompts or thresholds. Record
the complete attempt list and freeze a candidate on public development data
before running its bank regression twice.

The earlier public KZ-BD/MIDV-Holo sealed inventories have already been exposed.
Their failed results remain recorded in [public model cards](public-model-cards-20261018.md).
Those inventories cannot supply a new untouched holdout or a second attempt at
their consumed acceptance test. Reusing the existing public development split is
permitted when labelled reused development; it does not establish independent
generalization. A new independent acceptance claim requires fresh, unexposed
source groups reserved before feedback.

## Data and task checks

- Retain publisher/version, received hashes, license/attribution, annotation
  provenance, grouping and derivative lineage. Existing source terms and pending
  derived-artifact release obligations are in the [license register](../submission/licenses.md).
- Keep source recording, document design/template, near-duplicates and all
  derivatives in the same partition. Hash disjointness alone does not establish
  scene independence. Quarantine unresolved overlap. Known missing capture
  lineage remains a limitation.
- Exclude sealed rows from fitting and development views. Strip paths, IDs,
  publisher names, filenames, denomination strings and other source/class hints
  from model inputs; bookkeeping identifiers may remain in manifests only.
- Use published object labels only. Generic banknote/ID-card appearance labels
  do not become ID-presentation, recount, authentication, key, hand-contact or
  banking-procedure labels. A dataset without verified object-free examples
  cannot establish an absent-object false-alarm rate.
- No audio, face detection, face matching, OCR of personal details or identity
  targets. The previously admitted MIDV-Holo source contains author-declared
  fictional documents. Incidental artificial portraits do not authorize any
  face task. No new portrait-detector or masking requirement is imposed here.
- The coding agent sees only allowlisted structured metadata and metrics, never
  images, evidence JPEGs, scene descriptions or raw model completions.

## Training proof and artifact freeze

Before regression, retain hashes for the exact base snapshot, processor, runtime,
recipe, source, split membership, adapter and effective inference configuration.
Record the number of trainable parameters separately from the number of selected
or tuned hyperparameters. Report seeds, steps, learning rate, rank/alpha, target
modules, loss masking, resolutions and all attempted configurations.

A successful fit requires finite loss and gradients, actual completed optimizer
updates, changed trainable tensors and unchanged frozen tensors. A download,
import, memory smoke or saved zero-update adapter is not a completed fit. Verify
the saved adapter reload against its base/processor and record whether inference
uses NF4, BF16 or another representation; a quantized training result does not
prove equality to a different serving representation.

Evaluate base and adapter with the same public development preprocessing and
closed output schema. Retain invalid/missing outputs as failures, report unique
source-group support, per-class results and correlated degradation views
separately. Public crop improvements are component measurements only.

## Bank regression and release gate

Both frozen whole-pipeline runs must account for all 40 unique recordings.
Failures stay failed and are retained in the denominator. Zero false alarms
requires all 19 negative recordings to process successfully. Record model,
adapter, config/source hashes, verdicts, enum event types, numeric timestamps,
calls, timing and failures using a strict allowlist projection.

The current session's numerical target is at least 37/40 and zero false alarms
in each of two independent runs. This is an in-sample qualification target, not
an unseen-accuracy claim. The original brief's leakage/invariance and evidence
requirements continue to apply; absent tests cannot be represented as passing.

A newly enabled type still needs its stated visual principle before the
experiment, a threshold from the non-target distribution plus a margin, and
zero fires on the non-target videos and required augmented copies. Positive
labels cannot set that threshold. Omission claims require observable interval
coverage; a missing object detection is insufficient.

Production remains unchanged until the model passes staging, schema/evidence
checks and the six longest parallel jobs within 285 seconds, and the user
approves the switch. Record a one-step rollback and perform the public six-upload
check after promotion. Training success alone does not authorize a switch.

Binding sources: [original brief](../../Claude%20outputs/FORTE_AGENT_PROMPT.md),
[public training protocol](accuracy-public-training-protocol-20261018.md),
[session log](agent-log.md), and the [model cards](public-model-cards-20261018.md).
