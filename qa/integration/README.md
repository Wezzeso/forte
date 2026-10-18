# Integration QA

Only synthetic media may be used by these checks. They must not print private predictions, annotation content, frames, or descriptions.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_integration.py -q
.venv\Scripts\python.exe qa/integration/dependency_audit.py --output docs/submission/installed-dependencies.json
```

The first five API integration cases use an actual generated MOV and real durable SQLite job storage, then publish explicitly synthetic results through the leased queue contract. Four further cases run the actual worker/pipeline with an injected test-only provider through decode, sampling, refinement, rules, original JPEG assembly and publication. The generated-color JPEGs are checked against their source timeline. All nine pass. These tests do not invoke the learned detector, validate a real legal implication, or measure production performance. SQLite is a local-development fallback; PostgreSQL and external deployment checks have separate acceptance rows.

`public_smoke.py` is prepared for an actual external authorized API. It generates its own MOV and keeps the API key in the environment, printing aggregate diagnostics only. It has not yet established a public model-backed deployment. `build_release_bundle.py` creates a reproducible code/operator archive with explicit private-media, record, credential and weight exclusions. The release manifest must retain unresolved real-world gates.
