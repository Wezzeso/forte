# Optional workplace public-law profile

This profile is **unselected and unmeasured**. It does not replace or modify the default legal catalogue, taxonomy, nineteen-axis head, prompts or deployed configuration. The [extended legal review](../docs/integration/qwen38-extended-legal-coverage-audit.md) records applicability and known limits.

The separate catalogue version is `2026-10-17.workplace.public.v1`, raw schema `forte.optional.workplace.rules.v1`. The default `rules.catalog.load_catalog()` intentionally rejects its JSON. Explicit use is required:

```python
from rules.workplace_catalog import load_workplace_catalog

catalog = load_workplace_catalog()  # only the three new optional branches
combined = load_workplace_catalog(include_baseline=True)  # explicit 4 + 3 composition
facts = catalog.axis_fact_definitions(trusted_scene_context="bank_cash_node")
catalog.validate_axis_order(
    [fact.fact_id for fact in facts],
    vocabulary_version="forte.optional.workplace.fact-axes.v1",
)
```

No production route or factory selects this loader. `include_baseline=True` remains an optional profile with seven branches; it does not enable anything globally. A separate runtime/configuration decision and actual local evaluation are required before any selection. A nineteen-output checkpoint cannot be relabeled: the new vocabulary has **32 axes**, with the original nineteen in their exact original order followed by thirteen `WP_` axes. The three optional events and all 32 definitions are in [the separate taxonomy](../annotations/workplace-taxonomy.v1.json). The existing strict physical-observation contract accepts 32 facts; synthetic prompt construction is tested without giving legal citations to the model.

| Rule | Exact official source | Narrow scope |
| --- | --- | --- |
| `OPT_RK_NBK231_P12_PERSONAL_STORAGE` | NB RK No. 231, paragraph 12, first part, subject to second part | Observed storage of affirmatively personal employee belongings inside a confirmed cash node; supported exclusion of medicines, medical devices and permitted communication equipment |
| `OPT_RK_NBK231_P12_RECORDING_COMM_DEVICE` | No. 231, paragraph 12, second part, read with first part | Tracked employee brings a proven personal communication device with **observably established** photo/video capability into that confirmed node |
| `OPT_RK_HEALTH110_5_9_WORKPLACE_TOBACCO` | Health Code article 110(5)(9), read with 110(6) | Clearly identified tobacco product is actually smoked inside an affirmatively established indoor workplace/work zone |

`bank_cash_workstation` alone **does not establish a cash node**. The new cash-node fact needs the separate `bank_cash_node` trusted setting plus the actual visible internal zone and boundary. That setting does not prove any worker role, personal provenance, camera, tobacco, action, continuity or exception. If the setting is absent, the fact question explicitly requires `unclear`. Scene attestation remains an operator responsibility outside the image model.

Personal-property provenance must come from positive observed personal-use context and continuity, never appearance, a pocket or location alone. Actual camera capability must be distinguishable on the same communication device or established by observed camera operation; a rectangle, generic phone gesture or unknown device is insufficient. Fixed bank CCTV and operational bank equipment are outside this narrow personal-device branch. A basic handset without a camera and medical devices remain excluded. Unknown bag contents stay unclear. Worn clothing is not by itself a completed storage episode.

Smoking requires a distinguishable tobacco product and actual consumption. Holding an unlit cigarette, a pen, a medical inhaler, unknown vapor, a generic vape or outdoor smoking does not satisfy this branch. Do not infer drugs, intoxication, age or disciplinary guilt. Article 110(6)'s specified-place exception concerns items 3 and 7 of paragraph 5; the cited workplace prohibition is item 9.

Sources remain offline during inference. The profile references the existing checksum-verified No. 231 archives and adds [official Health Code source archives](workplace_sources/manifest.json), including unmodified content/version responses, a deterministic article 110 extraction and an explicitly derived metadata snapshot. Current metadata/history were checked against the official API. No. 14, fire, collection and ATM proposals remain report-only.

Every required state—including exception absence—must have original source-frame support. The existing builder checks applicability, uncertain/missing/opposite facts, candidate-local action support, and its 1–3-frame evidence policy; renderer accepts reviewed Russian phrases and citations only. **Composite continuity, personal provenance, cash-node scope and device classification remain substantive observations.** The builder does not independently recover body/object tracks or inspect pixels. Synthetic fact agreement is not proof of actual continuity or camera capability. If local observations cannot establish those facts, the candidate remains unresolved; the profile must not invent them to obtain a verdict.

Reproduce and validate:

```powershell
.venv/Scripts/python.exe -m rules.workplace_catalog
.venv/Scripts/python.exe -m pytest rules/tests/test_workplace_catalog.py -q
```

`python -m rules.workplace_catalog --write` regenerates only the new optional catalogue and taxonomy. Hard-pinned source archives, reviewed branch definitions, baseline catalogue/taxonomy hashes and exact axis identity are checked before the explicit loader succeeds. Changed sources or branch gates require deliberate review. This work establishes no footage accuracy, class calibration, semantic timestamp accuracy, speed or production readiness claim.

