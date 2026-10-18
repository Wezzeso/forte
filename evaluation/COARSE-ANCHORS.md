# Approximate validation-anchor diagnostics

The later supplied table adds coarse positive timing cues to the provided binary labels. A02 verified agreement with the original workbook, aliases and source durations and exported separate role artifacts. These points are **not** exact event starts/ends, event types, legal reasons or a complete event inventory. Their source uncertainty is explicitly unspecified (`null`), rather than invented as 100 ms or one source frame.

Training may use only `user_coarse_anchors.train.jsonl` as an existential class-agnostic eventness-bag cue. Outside those bags remains unknown. Validation anchors must not enter training, prompts, inference conditioning or threshold selection. Final-check anchors remain untouched. The fixed binary threshold and frozen split assignments stay unchanged.

The optional offline evaluator reads only `user_coarse_anchors.validation.jsonl` **after frozen temporal predictions exist**:

```powershell
python -m evaluation.coarse_anchors --anchors data/manifests/user_coarse_anchors.validation.jsonl --anchor-pins data/splits/user_coarse_anchors.pins.json --temporal-predictions training/runs/W1/validation/temporal-predictions.jsonl --expected-attempts evaluation/private/weak-W1-validation.expected.json --terminals training/runs/W1/validation/predictions.jsonl --splits data/splits/automated_provisional_split.json --metadata training/runs/W1/validation/metadata.json --threshold training/runs/W1/video-threshold.json --output benchmarks/quality/weak-W1-coarse-anchors.json
```

The concrete path above is A09's trained W1 threshold; the untrained reference uses `training/runs/W1-untrained/video-threshold.json`. Its raw file SHA256 must match the exported metadata. For another run, use the actual `video_threshold` path from its frozen evaluation config. Do not change a threshold after seeing anchor hits. The command issues no model/API/GPU calls, never opens source media and never supplies anchor values to the model.

The temporal producer exports each unique native frame with `frame_id`, integer raw `source_pts`, normalized `pts_sec` and finite `eventness_logit`, plus content-independent run identity, stream `time_base`, `source_start_sec`, frozen threshold and head/feature hashes. A10 recomputes canonical A05 frame identity from content hash/raw PTS/time base and verifies normalized seconds against the original source origin. Millisecond distances use raw-source PTS, not rounded display text or an invented fixed frame rate. Invalid references, missing scores and sampling gaps remain visible.

Metadata explicitly states `validation_anchors_used_for_training:false`, `validation_anchors_used_for_threshold_selection:false`, and `validation_anchors_used_for_prompting:false`. Missing or conflicting declarations produce a provenance-incomplete diagnostic. A02's source/role/assignment digests bind the anchor file, and the expected-attempt manifest prevents dropping failed predictions. Repeated attempts cannot cherry-pick the best localization: only the first frozen attempt per content is used, with point/video denominators separately reported.

Default **team diagnostic expansions** are 1, 2 and 5 seconds around each supplied point, extended by separately supplied source uncertainty only when known. They are not organizer scoring tolerances or asserted source precision; override with repeated `--tolerance-sec`. Every provided positive point stays in each hit-rate denominator, including failed/missing temporal predictions. Reports separately show whether any usable sampled frame covered the band, whether an above-threshold frame hit it, and nearest-frame distance to the **supplied approximate point**.

A model that scores every frame positive can hit every point trivially. The report therefore includes above-threshold frame fraction and never infers event precision/F1. One coarse point cannot label the rest of a positive video normal, identify an event class, prove why it violates a rule, establish exact onset or support a 100 ms accuracy claim. Local teacher explanations and verified rules remain a separate, explicitly provisional WHAT/WHY path rather than independent event ground truth.

Schema: `evaluation/coarse-anchor-report.schema.json`. Current actual role/pin readiness: `benchmarks/quality/coarse-anchor-validation.preflight.json`; actual hit/distance metrics remain pending the real temporal prediction export.
