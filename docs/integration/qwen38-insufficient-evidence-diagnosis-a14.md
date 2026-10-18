# Terminal pilot evidence-insufficiency diagnosis

**17 October 2026, Asia/Qyzylorda.** CPU-only inspection of a terminal, settled diagnostic run and generated public-code fixtures. No GPU operation, new inference, threshold/source/rule change or optional-profile activation occurred. Media, model text, labels, observation notes, source timestamps and per-fact/per-rule/per-zone states are excluded from this report and tool output.

## Untuned control: what actually failed

The settled untuned control's one attempted source ended with `insufficient_visible_evidence`. Its failure was preserved; five other sources remain unreleased within the six-source denominator. The diagnostic did not retry a source, reclaim a budget slot or create an independent native-video proof.

The original private analysis and settlement bytes were hashed before/after diagnosis and remained unchanged. A09 settlement SHA-256 is `c29fa7e37edd0c2305cc57be0839acb723c6aed447a3c454b35dbda966fd3d34`. Detailed semantic gate diagnostics remain in a mode-600 file inside the restricted terminal-run directory on the team host; they were not downloaded or printed. Only [global technical aggregates](../../qa/integration/private/qwen38-insufficiency-a14-untuned-v1/aggregate.safe.json) were returned and saved in a restricted local directory.

| Global technical measure | Observed count |
| --- | ---: |
| Original terminal analysis files inspected | 1 |
| Completed observation windows | 10: 4 coarse, 6 refinement |
| Structurally valid requested fact records | 190 |
| Missing requested records / invalid records | 0 / 0 |
| Source-support references / references outside requested frames | 400 / 0 |
| Fact records with source support / without source support | 190 / 0 |
| Fact records citing every requested sample frame | 0 |
| Window dispositions recorded as `unclear` | 10 |
| Proposed candidates / verified events | 0 / 0 |
| Aggregate branch evaluations / fully satisfied evaluations | 40 / 0 |
| Unresolved intervals | 10, all static reason `unclear_visual_context` |
| Initial planned interval coverage complete | Yes |
| Distinct source-frame references retained | 103 |
| Publication-ready result / independent full-native proof created | No / no |

This was **full nineteen-fact observation followed by refinement**, not the earlier four-fact screening shortcut. The bound worker configuration explicitly supplied `bank_cash_workstation`, disabled staged screening, used sixteen frames per request and enabled compact IDs/concise observations. The settings constructor forwards that configuration. A missing trusted-bank configuration is therefore not established by this run; actual private zone states are not disclosed here.

The immediate software path is falsifiable: no candidate enters positive event construction; no observation fully satisfies a rule branch; no fact cites the complete requested sample set needed for scoped opposite-state refutation; unresolved visual context therefore remains and publication fails. Structurally valid JSON and valid source IDs do not establish decisive visual evidence. With zero candidate events, the public one-to-three-JPEG event budget was not the cause of this result.

These totals establish neither that the model's uncertain observations are right nor that a real violation occurred. Private footage was not reviewed by the coding agent. They do establish that this completed inference did not supply the decisive facts/support required by the frozen pipeline. No incorrect rule implementation is proved by these aggregate values.

## Falsified hypothesis: internal support capped at three or four frames

The public-code and generated-fixture check **disproves** an impossible support-count contract:

- `forte.decision.structured._RawFact.supporting_frame_ids` permits **96 IDs**; raw candidate support also permits 96. Shared `FactDecision` adds no smaller count limit.
- `_schema` retains that support bound. Concise mode restricts notes, not factual support. HF `trusted_fact_order_schema` deep-copies support constraints while ordering fact IDs; it introduces no three/four-ID cap.
- The all-requested-frame condition in `analyze_rule_relevance` concerns using an **opposite physical state to refute an entire observation window**. It is not imposed on every positive fact or local event. Knowledge/visibility gates cannot be refuted merely by lack of knowledge; an observed count also cannot globally refute another procedure's count omission.
- The **one-to-three** bound belongs to public event evidence and its original-frame set-cover policy. Internal factual support and a public JPEG list are separate types.

[Nine generated regression checks](../../tests/test_internal_support_scope_contract.py) pass without models/media/GPU. Both concise and ordinary modes accept sixteen supports on each of nineteen facts through original JSON schema, the HF grammar's derived canonical schema, the decision validator and compact-ID restoration. Such supported opposite states can refute all four branches and yield a clean synthetic result. One/three/four/eight representative supports are structurally valid but cannot establish the opposite state over all sixteen requested samples. Both one-frame and sixteen-frame positive-action support produce a valid synthetic event while retaining one-to-three public evidence IDs.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_internal_support_scope_contract.py -q
```

Result: **9 passed**. This verifies representability and scope, not real visual truth. It does not justify padding support lists, turning point-local negatives into window negatives or relaxing legal visibility conditions.

## Remaining policy/input limitation

The public prompt explicitly says sampled images alone cannot establish continuous visibility between frames. All four default branches require supported continuous relevant visibility; omission branches additionally require complete procedure/counting opportunities. [The generated sampling audit](sampling-visibility-recall.md) independently shows complete interval metadata can coexist with a wholly missed brief action. A longer, fixed-sixteen-frame refinement may improve context while reducing temporal density.

There is consequently a sensing/proof limitation to measure, rather than a proved three-frame schema contradiction. Representative factual IDs do not silently become an exhaustive absence witness. Training point-local labels must remain point-local, and missing continuous/native proof must remain missing. Any future change to support-scope representation or visibility evidence requires a separately versioned, evaluated design; this diagnostic makes no such change.

## Trained comparison

A13 subsequently confirmed the trained caller was genuinely terminal and A09 settled it with SHA-256 `3356e4b8ddedbee173197cd3951dcd52986f7d0f32a9ff8365bb21055b93df45`. Only then did this diagnosis inspect its terminal audit. Its original audit and settlement were verified unchanged, detailed semantic values stayed on the restricted team host, and only [global technical aggregates](../../qa/integration/private/qwen38-insufficiency-a14-trained-v1/aggregate.safe.json) were downloaded.

| Global technical measure | Untuned control | Trained pilot |
| --- | ---: | ---: |
| Static terminal failure category | `insufficient_visible_evidence` | `refinement_model_incomplete` |
| Completed observation windows | 10: 4 coarse + 6 refinement | 7: 4 coarse + 3 refinement |
| Structurally valid requested fact records | 190 | 133 |
| Missing / invalid records | 0 / 0 | 0 / 0 |
| Source-support references / outside requested frames | 400 / 0 | 275 / 0 |
| Records citing every requested sample frame | 0 | 0 |
| Completed-window `unclear` dispositions | 10 | 7 |
| Candidates / verified events | 0 / 0 | 0 / 0 |
| Fully satisfied aggregate branch evaluations | 0 of 40 | 0 of 28 |
| Initial interval coverage complete | Yes | Yes |
| Distinct retained source-frame references | 103 | 85 |
| Publication / independent full-native proof | None / none | None / none |

A13's terminal accounting records seven completed HTTP requests followed by one client-side deadline failure in the trained pilot, with sixteen remaining slots unused. This is a **technical interruption of refinement**, distinct from the untuned completed-observation evidence-insufficiency outcome. Client termination alone does not prove a server-side in-flight generation ended; the root operator separately owns service/in-flight settlement.

The trained failure audit retained **four** `unclear_visual_context` intervals from the preceding provisional build. The pipeline writes `built or provisional` if a refinement model call fails; it does not recompute final events over the newly incomplete round. Consequently four retained unresolved intervals versus ten in the completed untuned build cannot be interpreted as a quality gain. The differing window/fact/support totals likewise reflect different completed work, not an accuracy improvement.

Neither pilot produced a usable result, and five population sources remained unreleased in each arm. No model winner, confusion matrix improvement, revised threshold, satisfied native proof or production selection follows from this diagnosis. The optional thirty-two-axis workplace profile remained unselected; both audited runs used the frozen four cash branches and nineteen facts.

