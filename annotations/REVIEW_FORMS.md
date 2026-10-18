# Human review using the local forms

Both updated local servers share the same database. Save any current draft before refreshing a tab to load the new static forms. Neither proposal import nor these static file updates closes or reloads an existing tab. Keep a separate private copy before switching records if a draft has not been saved.

For an ordinary review, open a queue entry, enter your local reviewer ID, watch the entire relevant video and use the form controls. `reviewed` is an explicit human choice; no model output or template selects it automatically.

For a fully checked negative recording:

1. Confirm that the entire recording and all potential episodes were reviewed.
2. Choose the no-signs verdict and add one or more explicit negative intervals. The whole-recording button creates only an **unverified template**: visibility starts unobservable and both context/result confirmations start false.
3. Select clear visibility and personally confirm full relevant context and absence of the reviewed signs for each interval. Intervals must cover the complete source timeline for a whole-video negative verdict.
4. Resolve and remove each outstanding question only after checking it. Add review notes where useful.
5. Choose the human-reviewed state and save. Existing backend rules reject gaps, unclear visibility, overlapping events or unresolved questions.

For an event, copy a local model proposal into a draft or add an empty event template. New facts start `unclear`; neither the expected state shown for a rule nor selection of that rule fills an observation. Select the event interval, an exact original source frame for the anchor, 1–3 evidence frames and supporting frames for each observed fact. Frame selections come from already registered source frames or imported proposals. With no source frames registered yet, the advanced JSON import path is still needed for events; negative-window review can use the local source player.

Choose the exact enabled rule and separately verify every required fact, temporal/full-context condition, exception and applicability. Selecting a rule leaves grounding unresolved and its explicit checks false. The backend still validates the authoritative checksum-verified rule catalog, every evidence ID and source PTS. The static chooser is generated from the public catalog by `python -m annotations.generate_ui_rules`; it cannot authorize an outdated or disabled mapping.

Scenario grouping uses a separate proposal field and an explicit reviewed checkbox, stored as `corrections[kind=scenario_group_proposal]`. It never modifies the protected content identity, actual group ID, split or lock. A02/A10 must still approve and lock scenario groups through their existing process.

Advanced JSON remains available. Use “Применить JSON к формам” to bring manual JSON edits into the form controls. Saving uses the existing optimistic revision guard; a rejected stale save keeps the draft data in the editor. Source records stay immutable, and raw model proposals remain separate from reviewed training/evaluation exports.

Synthetic form checks: `node annotations/ui/test-forms.mjs` uses the existing project's Playwright package and local Chrome, mocks every route, creates no screenshots, contacts no private service and performs no inference. Nine checks cover explicit negative review, safe defaults, scenario metadata and author preservation, source-frame selection, exact rule choices with unknown facts, advanced JSON and stale-save draft preservation. The Python workflow/proposal tests retain all authoritative evidence/grounding/export gates.
