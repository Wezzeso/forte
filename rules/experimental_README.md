# Optional priority-1 legal experiment

This experiment is isolated from `rules/catalog.json`, its source archives and the existing nineteen-axis model vocabulary. No default route, prompt, checkpoint, threshold or deployment is changed by importing the regular rules package.

Use the explicit loader:

```python
from rules.experimental_catalog import load_experimental_catalog

catalog = load_experimental_catalog()  # four new branches only
definitions = catalog.fact_definitions(trusted_scene_context="bank_cash_workstation")
axes = catalog.axis_fact_definitions(trusted_scene_context="bank_cash_workstation")
catalog.validate_axis_order(
    [axis.fact_id for axis in axes],
    vocabulary_version=catalog.vocabulary_version,
)
```

`include_baseline=True` explicitly combines the original four and experimental four branches. The resulting catalog remains experimental. Raw `experimental_catalog.json` uses a distinct schema; the ordinary production `load_catalog()` rejects it.

The experiment's vocabulary is **`forte.experimental.fact_axes.priority1.v1`**, with **37 axes**: the original 19 in their original order followed by 18 new `EXP_` axes. The manifest records the ordered-axis hash and baseline catalog hash. An existing nineteen-axis checkpoint cannot be silently relabeled or expanded to this vocabulary. `validate_axis_order()` rejects missing, reordered or differently versioned axes. The catalog version is **`2026-10-17.priority1.observable.v1`**.

## Four branches

| Rule ID suffix | Official clause | Observation |
| --- | --- | --- |
| `P20_PRIOR_CASH_OUTSIDE_STORAGE` | No. 231 paragraph 20, second part | Prior accepted cash remains idle on an ordinary counter, outside permitted storage, while a subsequent distinct receipt proceeds |
| `P20_RECEIPT_CLIENT_ABSENT` | No. 231 paragraph 20, first part | The same unfinished receipt is counted after the contributor visibly leaves and the entire client area is empty |
| `P19_2_COIN_RECEIPT_COUNT` | No. 231 paragraph 19(2) | Loose coins are received without observable manual or machine unit counting over the full procedure |
| `P32_DAMAGED_BUNDLE_COUNT` | No. 231 paragraph 32, read with 31(4) | An identifiable banknote bundle has visibly damaged bundle wrapping before issue and no leaf counting occurs over the full preparation/issue procedure |

All four use the existing verified official source archives and hashes. They add no source material or speculative authority, phone, clothing, identity or internal-policy rules. Bank-cash scope may come from the organizer's declared bank-cash scenario; source-frame visibility and actual procedure applicability still need automatic evidence.

## Deterministic temporal proof

Every binding includes `experimental_temporal` with schema **`forte.experimental.temporal.v1`**:

- `ordered_phases`: exact phase IDs and mandatory fact IDs; phase-only start/completion facts are requested alongside global legal facts.
- `phase_procedure_roles`: each phase explicitly belongs to the prior or current procedure; the prior-storage branch requires distinct procedure IDs.
- `required_procedure_boundary_fact_ids`: actual source-frame proof of start and completion, rather than the video's beginning/end or an invented time.
- `required_track_kinds`, `track_phase_ids`, `track_fact_ids`: one cash track across the necessary source phases; an anonymous body track only from contribution through the client's exit. Face identification is excluded.
- `absence_must_cover_procedure` and `counting_paths_must_cover_procedure`: both true for the coin and damaged-bundle count omissions. Missing observations, sparse unclear frames, unseen devices or gaps never become a confirmed omission.
- `absence_phase_ids`: client absence applies only to `count_with_client_area_empty`; it cannot apply to the earlier contribution phase where the person is present.
- `strict_phase_order=true`, `max_visibility_gap_sec=null`: phases require actual ordered source PTS. No arbitrary statutory minimum duration or permitted visibility gap is invented.
- `event_phase_id`, `candidate_interval_policy`: action intervals/anchors derive from the actual violation phase, while omission intervals derive from the complete procedure. Earlier context frames cannot become a premature event timestamp.

Use A11's separate experimental temporal wrapper (`src/forte/events/experimental.py`), which consumes explicit automated local episodes and original FrameRefs. **Do not pass these branches directly to the ordinary co-occurrence event aggregator.** A correct list of present facts alone does not establish phase order, cash continuity, complete omission scope or a client's exit before counting.

Automatic local analysis may produce all facts, tracks and phase contexts; manual annotation is not a prerequisite. Every asserted phase, boundary, absence and track must still resolve to the inspected original source frames. Unclear or insufficient observations remain unresolved. Explanations retain possible-sign wording for human checking, as the organizer requires.

The public event contract remains seconds and 1-3 actual JPEG frames. Private temporal proof may retain additional source frames. Do not broaden factual support IDs, create montages or fabricate evidence to fit the public budget. If the required original factual support cannot fit the existing evidence policy, the experimental wrapper must defer rather than produce an unsupported grounded event.

## Checks and regeneration

```text
python -m pytest rules/experimental_test_catalog.py -q
python -m rules.experimental_catalog
```

`python -m rules.experimental_catalog --write` regenerates only the experimental JSON after a reviewed source/vocabulary change. It never writes the default catalog or source archives. A changed baseline hash/axis order forces review and regeneration, preventing an old experimental checkpoint from silently inheriting different facts.

These are source-grounded optional rule branches and synthetic checks. They establish no private-video detection accuracy, calibration, throughput or under-ten-second result. The existing A18 trial and nineteen-axis head remain separate.
