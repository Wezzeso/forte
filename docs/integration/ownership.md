# Integration ownership

Exactly 20 assignments implement the supplied plan. ROOT orchestrates them and owns `src/forte/config.py`, `pipeline.py`, and `worker.py`. A20 owns the root package manifest/lock, `src/forte/__init__.py`, README, integration and submission documents, integration QA, and `tests/test_integration.py`.

| Agent | Write ownership |
|---|---|
| A01 | `contracts/`, `src/forte/shared/schemas.py`, organizer requirements/rubric |
| A02 | `data/manifests/`, `data/splits/`, `scripts/audit_data.py` |
| A03 | `annotations/`, annotation guide, event taxonomy |
| A04 | `rules/`, legal-source register |
| A05 | video decode and probe |
| A06 | video sampling and crops |
| A07 | `src/forte/model/`, `models/manifest.json` |
| A08 | `src/forte/decision/`, `configs/decision/` |
| A09 | `training/` |
| A10 | `evaluation/`, quality/robustness benchmarks, vision QA |
| A11 | `src/forte/events/` |
| A12 | `src/forte/evidence/` |
| A13 | `src/forte/api/` |
| A14 | `src/forte/jobs/` |
| A15 | `serving/` |
| A16 | `frontend/` |
| A17 | `infra/`, `deploy/`, `.dockerignore`, GPU provisioning/deployment runbook |
| A18 | performance/reliability benchmarks, observability |
| A19 | brand evidence/tokens/assets, visual guidelines/QA |
| A20 | Root dependencies, integration/submission docs and QA |

Read across boundaries; request explicit handoff before editing another owner's files. Changes to frozen contracts require A01. Dependency additions go through A20. Shared GUI input goes through the exclusive lease.
