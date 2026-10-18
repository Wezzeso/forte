# Qwen3.8 deterministic rule integration audit

Reviewed **17 October 2026, Asia/Qyzylorda**. Scope: handoff sections 6 and Phase D, organizer [requirements](../requirements.md), [rubric](../rubric.md), frozen public sources, physical-observation contracts and synthetic event construction. No private recording, frame, annotation, supplied per-video label or model observation was inspected. No model inference or remote mutation was performed.

## Result and frozen compatibility boundary

The default catalogue remains **4 enabled branches, 3 event types and 19 ordered physical/context fact axes**. Its SHA-256 remains `3ba1c79c00f39a40addf4e1d2299c5bbade26c7949aeee10c382b9322fdde1dd`. This audit changes no catalogue, source archive, taxonomy, axis order, model configuration or production selection.

The new [focused regression suite](../../rules/tests/test_qwen38_rule_gates.py) checks the catalogue and taxonomy hashes against `training/temporal/fact-axis.v1.json`, exact fact order, explicit `absent/present/unclear` state order and preservation of the default prefix by the isolated 37-axis experimental vocabulary. A 19-output head cannot satisfy the experimental vocabulary contract. A different model family must produce the same complete structured facts and real source-frame references; a model name or binary score cannot replace those gates.

## Official source recheck

The primary authority is [National Bank Resolution No. 231 on Adilet](https://adilet.zan.kz/rus/docs/V1900019680). On this audit's date the public document page returned a JavaScript shell. A read-only request to the [official document API](https://adilet.zan.kz/api/documents/by-ngr/V1900019680?language=rus) returned document `c4cd4052-63e3-475b-ba9e-93f23de13fdb`, `status=upd` and `is_actual=true`.

The [official current content API](https://adilet.zan.kz/api/documents/c4cd4052-63e3-475b-ba9e-93f23de13fdb/content) returned a nonempty `text_content` string of **168,836 characters**, exactly equal to the frozen `text_content` in `rules/sources/nbk_231_content.json`. This comparison used the actual nonempty field, not metadata or a missing JSON property. The [official version history](https://adilet.zan.kz/api/documents/c4cd4052-63e3-475b-ba9e-93f23de13fdb/versions?page_size=100) identified the same document as current with revision boundary `2026-07-11T18:00:00Z`, matching the existing register's **12 July 2026** source boundary. The API's `end_date` is not independently interpreted as repeal. No frozen archive was refreshed or overwritten.

## Supported branches and ordinary lookalikes

These are the project's conservative executable interpretations of the cited official clauses, not a finding about any actual person or recording. See the [source register](../legal-source-register.md) for full applicability and exceptions.

| Branch and citation | Required observation/context | Unsupported shortcut that must fail closed |
| --- | --- | --- |
| Receipt departure, paragraph 22 | Same unfinished customer receipt, actual workstation departure and supported absence, cash and continuous relevant visibility | Bending, occlusion, a step within the workstation, an idle customer or a completed service does not establish departure during service |
| Previously accepted cash into clothing pocket, paragraph 20, second part | Completed earlier receipt, continuous tracking of the same accepted cash and visible placement into an actual clothing pocket | Unknown paper, a customer's own funds, a bag, cash cassette, drawer, inferred ownership or alleged theft is insufficient |
| Loose-note receipt without observed count, paragraph 19(2) | Actual completed receipt, loose notes, visible handover and the full opportunity across all manual and automatic counting paths | A single frame without counting, hidden machine, cropped hands, late recording start or collection bag is insufficient |
| Loose-note issue without observed count, paragraph 31(4), subject to 32 | Actual loose-note issue and complete preparation/issue/counting visibility | Intact packaged cash may follow paragraph 32's packaging branch; lack of visible hand counting alone does not establish omitted counting |

Phone use, clothing, gestures, unverified staff authority, an ordinary room's access, unseen paper identity documents, account correctness and reconciliation remain unsupported grounds. The default catalogue does not enable these findings. The trusted `bank_cash_workstation` setting supplies case scope; it does not supply visual presence, room classification, identity, authority or procedure completeness.

## Proven defect corrected

Two synthetic counterexamples established a concrete localization and false-association defect: a departure or pocket action supported only at second 16 could ground a candidate at seconds 4–6 using a context frame at second 5. Previously the builder checked that *some* support was inside the candidate, while required action support could be anywhere in the same analysis window. Likewise an earlier context frame could become the public action timestamp even when the action itself was supported only later.

The narrow correction in [the event builder](../../src/forte/events/builder.py) now requires each bound `violation_fact_ids` action to have original-frame support inside that candidate's interval. Missing local action support produces a refinement request with `action_support_outside_candidate`; it cannot quietly become `no_violation`. For supported action branches, the event anchor must be one of those action-support frames. Context evidence may still precede or follow the action. Omission branches retain their separate completed-procedure anchor convention.

The focused suite reproduces both failures for both default action branches and now passes without expected-failure markers. It also removes, makes unclear, and reverses **each of the 34 bound gate positions across all four branches**, checking that no unsupported event survives, and verifies deterministic reviewed Russian rendering with the frozen source/paragraph. The ordinary four rendered branch descriptions are at most **691 characters**, before any approved limitation. This is an observed template size, not a newly imposed legal or API length limit.

## Remaining limits for Qwen3.8 acceptance

- `present` and `absent` must have source support; missing or unclear facts cannot become negative labels. No observed counting at one point establishes only a point observation unless the complete procedure and every counting path are established.
- Actor/object continuity, one relevant procedure, and complete opportunity remain substantive fact prerequisites. The default fact contract does not independently encode general-purpose procedure IDs or track IDs. The new interval check does **not** prove temporal identity or eliminate contradictory or mistaken model predicates. The separate experimental wrapper must not be described as default runtime proof.
- Evidence assembly must still resolve the selected original frames to original JPEGs, keep 1–3 frames, preserve source PTS and validate the final binary jury result. These rule tests do not inspect pixels, prove evidence relevance, measure semantic onset error or measure video accuracy/latency.
- Only reviewed observation phrases, approved limitations and frozen citations enter public explanations. The model must not generate rule numbers, legal titles, verdicts, internal bank policies, guilt or motive. The explanations remain possible signs for human confirmation.
- An independent new-camera/video evaluation is still required before claiming model precision, recall, complete hidden-event coverage, repeatability or the user's ten-second objective. Synthetic rule success cannot establish any of those claims.

## Urgent recall follow-up

The later [aggregate screening diagnosis](latest-screening-recall-diagnostic-20261017.md) reported that the latest three completed jobs all returned `no_violation`: their 14 screening windows closed all four default branches, with **zero full primary passes, zero refinements and zero candidates**. This establishes the executed shortcut. It does not establish that those recordings legally contain violations, nor independently measure recall.

The screen asks four physical facts and does not ask procedure/visibility gates. Static code permits absent departure, pocket or handover predicates supported on all requested **sampled** frames to refute branches. In a synthetic example, cash remains visible while all three movement predicates are absent on the samples; all four branches close and no nineteen-fact pass is selected. As the separate [sampling counterexample](sampling-visibility-recall.md) shows, metadata coverage of all time intervals does not mean every native frame or brief action was observed. Root chose to disable staged screening in deployment configuration; this audit does not itself change or attest deployment configuration.

A separate, proven reconciliation defect could erase an earlier supported candidate when a wider capped refinement **omitted the original positive source frames** and reported absence over its different samples. The interval contained the earlier procedure, but the evidence was never reinspected. The correction is confined to `pipeline.py`'s `_explicit_type_refutation` and `_explicit_refutation`: negative refutation now needs the earlier candidate support and positive bound action support among the actual refinement frame IDs. Omission/generic branches without an affirmative action ID retain their original positive procedure support. Missing-proposal positives receive the same protection. Full reinspection may still refute the original; the change never forces a positive verdict or waives legal facts.

[Nine synthetic regressions](../../tests/test_recall_refutation_support.py) pass, including disjoint refinement samples, candidate versus action support, missing candidates, two distinct occurrences and legitimate reinspection/refutation. A diagnostic can be reproduced without media or inference:

```powershell
.venv/Scripts/python.exe -m qa.diagnostics.a14_recall_counterexamples
```

It reports the unchanged screen shortcut separately from corrected reconciliation: the missed-original-support example retains an unresolved interval and has no publishable verdict. A10 confirmed that refinement did not run in the three diagnosed jobs, so this reconciliation defect is a separate latent risk rather than the observed screen-only explanation.

## Validation

Use the workspace environment (`.venv/Scripts/python.exe`); the system `python` does not have pytest installed.

Initial existing rule/experimental/event checks: **53 passed**. Action-scope regression result: **279 passed and 51 subtests passed in 17.74 seconds**. After the urgent recall helper correction and its nine additional tests, the same focused rule, experimental, event, evidence, decision and pipeline run reported **288 passed and 51 subtests passed in 17.64 seconds**. No skipped or expected-failure checks were reported.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_recall_refutation_support.py rules/tests rules/experimental_test_catalog.py tests/test_events.py tests/test_pipeline_review.py tests/test_pipeline_sampling_review.py tests/test_staged_screening.py tests/test_evidence.py tests/test_decisions.py -q
```

