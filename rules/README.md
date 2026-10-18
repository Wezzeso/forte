# Offline legal catalog

`catalog.json` contains reviewed public Kazakhstan rules; `sources/manifest.json` records official source URLs, dates and hashes. The full source register is `docs/legal-source-register.md`.

```python
from rules.catalog import load_catalog

catalog = load_catalog()  # hashes and exact excerpts checked; no network
rules = catalog.enabled_rules()
facts = catalog.fact_definitions(trusted_scene_context=None)
```

Pass the operator-attested setting `bank_cash_workstation` only for footage within that scope. It does not bypass visual applicability or evidence checks. There are 4 enabled conservative branches and explicit unsupported/unresolved categories. All counting omissions require complete context and all counting paths visible. A missing visible action is not automatically an absent action.

Consumers must evaluate `bindings[rule_id].required_fact_states`, preserve explicit `unclear`, enforce temporal and context facts, and reject unsupported evidence IDs. `RuleRecord.required_facts` contains every required **present** fact; required **absent** facts live in the binding.

Descriptions use only reviewed phrases: `catalog.fact_description(id, state)` and `catalog.render_description(rule, phrases)`. Do not interpolate model notes or invented source text. Copy this package, JSON files and source archives into deployment; the CPU package wheel alone does not currently include root-level `rules/`.

Tests: `python -m pytest rules/tests -q`.
