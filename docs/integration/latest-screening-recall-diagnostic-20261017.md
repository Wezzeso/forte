# Latest completed-job screening diagnosis

Measured on the approved private worker at **17 October 2026, 18:34:36 +05**
(`13:34:36 UTC`), before the controlled API/worker quiescence. The database
contained 18 accepted jobs: 9 completed, 9 failed and none processing. The
diagnostic inspected the latest three completed private audits on the host;
only technical aggregates and static catalog identifiers were exported.
No footage, JPEGs, captions, observation notes or individual predictions were
exported. No model calls or remote mutations were made.

The immediate recall concern is the screening shortcut: **all 14 screened
windows refuted all four rule branches, so none received full fact inspection**.
All three jobs completed as `no_violation`. This establishes the executed
decision path; it does not establish that the model's visual absences were
correct or that these clips are human-reviewed negative examples.

| Aggregate across the three completed jobs | Measured value |
| --- | ---: |
| Screening model requests/windows | 14 |
| Windows considered to refute every enabled rule | 14 |
| Full inspections required/completed | 0 / 0 |
| Refinement requests | 0 |
| Candidate / verified events | 0 / 0 |
| Screen facts marked unclear | 0 |
| Unique sampled frames / source frame count | 206 / 1,912 |
| Repeated frame uses across model requests | 224 |
| Combined source duration | 64 seconds |
| Largest actual gap between sampled PTS | 0.533333 seconds |

| Screen fact | Present | Absent | Unclear |
| --- | ---: | ---: | ---: |
| `CASH_VISIBLE` | 3 | 11 | 0 |
| `CASH_HANDLER_LEAVES_WORKSTATION` | 5 | 9 | 0 |
| `CASH_PLACED_IN_CLOTHING_POCKET` | 0 | 14 | 0 |
| `CASH_HANDOVER_VISIBLE` | 0 | 14 | 0 |

All 14 handover-absence and all 14 pocket-absence records cited every sampled
frame in their respective windows. That is support over the sparse input
images, not a measurement of continuous procedural visibility. Each of the
four catalog branches was counted as refuted in all 14 windows. The four-fact
screen did not request the remaining procedure/context facts, and the full
19-fact passes were never reached.

The active profile recorded 2 fps, eight-second windows, four-second stride,
16 frames per window and eight permitted refinement passes. Stored JPEG
references retain 464 × 848 geometry. The model consumed 107,136 prompt tokens
and generated 6,066 tokens across 14 requests; per-request generation ranged
from 382 to 453 tokens. Coarse-model stage time ranged from 14.565 to 23.018
seconds per job. These stage times exclude upload and do not establish total
latency.

The audits bind the same verified baseline model/processor revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` and vLLM `0.31.0`. The latest three
audits contain no failure code. The complete 18-job inventory includes one
historical `invalid_model_output` failure; it must not be attributed to these
three successful completions. Raw finish reasons and actual processor grids
are not stored in these audits, so truncation internals and processed visual
pixels remain unmeasured. There is no measured evidence here of a 512-pixel
resize or a processor/token-cap truncation causing these negatives.

The immediate corrective experiment should disable screening for the quality
comparison and require the full fact contract. Keep unsupported visibility or
procedure context uncertain, preserve source frames carrying earlier positive
support during refinement, and compare fresh actual results against a frozen
development recipe. Increased density alone cannot repair a shortcut that
prevents full inspection. Any density change needs measured recall and runtime:
the separate generated-video sampling probe shows that a short action can fall
between the default samples, and wider capped refinement windows can become
sparser.

Private evidence: `qa/integration/private/latest-three-completed-recall-measured.json`.
This is an exact selected-field projection recovered from the successful tool
response. Two subsequent read attempts failed because the worker was being
quiesced; their failure receipts remain separate. The diagnostic script is
`qa/integration/private/aggregate_latest_jobs.py`. Its later single-fact replay
and implementation-hash extensions were **not** run successfully and supply
no evidence for this report.
