# Quality evidence

`synthetic/report.json` is a deliberately imperfect eight-sample fixture report. It demonstrates false positives, wrong types, multiple events, timing diagnostics, failed/missing predictions and one late result. Its locked split and human-review metadata are invented fixture values, labeled synthetic; they authorize no real training or model selection.

`split-audit.json` audits the real data manifest by aggregate metadata only. Proposed scenario groups are not a reviewed split lock. `release-status.json` records unmeasured real-quality items honestly. Replace status only with linked evidence from reviewed local inference; do not promote fixture results to model accuracy claims.

Run instructions and exact report denominators are in `evaluation/README.md`.

The later authorized automatic first pass uses `evaluation/WEAK-SUPERVISION.md`, `weak-validation-gate.json`, a frozen six-attempt plan and independent provided validation binary labels. It does not require human event review for provisional weak-model progress. `weak-operational-status.json` keeps pending actual weak predictions separate from human-gold accuracy. The reused sequential staged/concise result has its own provided-label diagnostic under `trials/staged-concise/`; it is not primary-SLO or heldout evidence.
