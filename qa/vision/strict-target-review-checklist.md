# Data and review checklist for the strict targets

The requested targets are **100% observed detection and grounding on the declared reviewed evaluation set**, and **every attempt strictly below 10,000 ms from upload start through receipt of the final result byte**. They are acceptance targets, not achieved performance or a universal accuracy guarantee. The original all19 exploratory report and 26 preserved proof artifacts remain unchanged.

## 1. Establish the evaluation population before model selection

- Preserve source files and content hashes. Resolve exact duplicates, related scenarios and all derived windows before splitting. A coding agent may inspect opaque metadata; supplied frames and action descriptions stay in the private local review environment.
- A human completes scenario grouping and approves it. The A02 human bridge writes an unlocked reviewed proposal; A20 alone creates a genuine approved lock after checks. Keep final-check groups unseen during training, prompt changes and checkpoint selection.
- Validate the latest hash-chained exposure registry and complete accepted-job inventory. Conservatively include attempted/failed/uncertain submissions and manual uploads. Exclude the entire reviewed scenario containing any model-exposed recording from final-check. A digest of an old registry does not establish current safety.
- A02's registry records exposure, not training authority. Any final-check data used for model selection or iterative changes is development data thereafter; disclose it and collect a new genuinely unseen final-check set. Do not reclassify consulted final-check data into authorized training silently.
- Lock content/group membership, annotation export hash, rule/taxonomy versions, selected checkpoint and all pipeline settings. Record actual deployed runtime overrides. A post-run local default file is not proof of the measured runtime.

## 2. Complete genuinely reviewed labels and events

- A local human reviews the full original timeline, every event and every supposed normal interval. Retain reviewer identity/reference, timestamp, review origin `human`, complete review state and an empty unresolved-items list only when resolved.
- A positive needs all relevant reviewed events, each with type, interval, human-selected original-frame anchor, observable supporting facts, context, visibility and exceptions. One supplied positive workbook timestamp does not make the remaining timeline negative.
- A negative needs complete visible procedure/context review and verified negative coverage. Provided workbook binary labels and local-teacher suggestions are not substitutes for this review.
- For a learned fact head, label each fact/time interval explicitly as present, absent or unknown. Unknown/unreviewed intervals stay masked. Never infer all fact labels, omission labels or frame-level timing from a whole-video binary label; sparse labels are not dense ground truth.
- Mark every applicable legal mapping independently: current verified source, relevant clause/section, applicability, required visual facts, temporal preconditions and exceptions. A verified source title alone does not establish that the observed procedure violates it.

## 3. Bind timing and evidence to original frames

- Preserve the original timeline origin, actual source-frame PTS, VFR behavior and rotation/crop provenance. Do not substitute model-sampled times for source timestamps.
- The strict evaluator consumes a trusted A05 verification export. Before setting `original_timeline_verified`, upstream must run canonical source-reference validation binding content hash, raw `source_pts`, stream time base and source origin (and dimensions/rotation) to each returned frame. A copied map or a model-assigned boolean is not that proof. This offline evaluator does not reread media or independently reconstruct canonical frame identity; preserve the upstream technical audit hash with the run.
- Bind each human-selected anchor and each predicted anchor to an existing original frame ID. Supply an independently verified original-frame PTS index (or equivalent exact adjacent original PTS metadata), not only the sparse model-sampling list.
- Report milliseconds as a numeric representation, e.g. `round(source_pts_sec * 1000)`. Keep raw source PTS for error calculation. At exactly 30 fps adjacent frames are about 33.333 ms apart; writing `33 ms` does not create 1 ms action-time accuracy. VFR spacing must come from actual PTS.
- Report error relative to the human-selected original-frame anchor and the adjacent original-frame spacing separately from display rounding. A frame-selected anchor is not the unknowable exact physical action onset between frames. Human ambiguity belongs in the interval/limitations, not an invented precision claim.
- The matching tolerance is an explicit team diagnostic (default 1,000 ms, inherited from the internal 1-second analysis tolerance), not an organizer tolerance. If a separate maximum original-frame-anchor error is wanted, freeze `--max-anchor-error-ms` before evaluation; it has no assumed 1 ms default.
- Independently review every returned evidence frame locally for visible support, exact source/timestamp correspondence, original-frame identity and rule applicability. Public JPEG decoding/schema checks alone do not establish any of these. Every predicted event needs 1–3 distinct original evidence IDs including its anchor.

## 4. Freeze all attempts and measure the complete path

- Before requests, write an expected-attempt manifest with unique `attempt_id` and `sample_id`. Include repeats, rejected multipart uploads, failures before job creation and timeouts. Repeated attempts increase service measurements, not independent video sample size.
- Measure monotonic time at client upload start through receipt of the complete validated result body. Include upload, queue, decode, features/model, temporal decisions, rule/evidence assembly, polling and complete base64 payload transfer. Worker compute time, POST acknowledgment and first-byte time are separate diagnostics.
- Require strictly `< 10000 ms`; exactly 10,000 ms fails. Missing results, invalid result contracts, missing timing or an incorrect timing origin fail. Failed jobs remain separate from clean negatives and count against every all-attempt target.
- Use uncached distinct inputs and the intended concurrency/hardware, with actual weight/model/config/rule versions. Record all retries and operational failures. Do not stop after successful examples or report a successful subset as 100%.

## 5. Claim only what the reviewed run establishes

- All-attempt acceptance requires every attempt to complete, meet the strict latency bound, detect all and only reviewed events/verdicts, use verified applicable grounding, and pass independent evidence/timestamp review under the declared matching/anchor policy.
- `SYNTHETIC_ONLY` validates software behavior. `UNMEASURED` means prerequisites are absent. Neither is real target achievement. A finite successful run can support “all N reviewed attempts met the declared criteria,” never universal 100% future accuracy.
- Report the reviewed distinct-video/group counts, positive/negative and per-type coverage, every failure and unsupported mapping, source-frame error distributions and absent robustness scenarios. New-angle robustness still requires genuinely different footage.

The optional offline evaluator is `evaluation/strict_targets.py`; report shape is `evaluation/strict-target-report.schema.json`. No training, split lock, model call, API request, GPU operation or private media review is performed by that evaluator.
