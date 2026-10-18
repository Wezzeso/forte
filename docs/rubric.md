# ForteBank rubric and evidence plan

Status: organizer weights verified from the supplied brief; actual accuracy, latency, deployment, and score remain unmeasured here. Reviewed on 17 October 2026, Asia/Qyzylorda.

Source: [Кейс_ForteBank.docx.pdf](../Кейс_ForteBank.docx.pdf), section 9 and FAQ 4, 12-20. SHA-256: `376d6fc5970f406a29211cc00ccd732d837eea900ca79f5a6dfa016f00aadf8f`. The local seven-page PDF was read through full text extraction, including the table and all FAQ items. [Requirements and source qualifications](requirements.md) distinguish organizer facts from execution-plan recommendations.

## Published weights: exactly 100 points

| # | Organizer criterion | Maximum points | What the organizer assesses | Evidence the team should retain | Primary owners |
|---|---|---:|---|---|---|
| 1 | Detection quality / Качество обнаружения | 25 | Correct presence or absence of a visible violation. FAQ 15 says the verdict is scored for the whole video. | Group-held-out whole-video TP, FP, FN, TN; inspected error records; handling of normal actions. | A07, A08, A09, A10 |
| 2 | Localization quality / Качество локализации | 15 | Timestamp accuracy and relevance of the evidence frame. FAQ 17 requests a concrete frame and timecode. | Original-source timestamps; frame identity; local relevance review; localization-error distribution where reviewed ground truth exists. | A05, A06, A11, A12 |
| 3 | Classification quality / Качество классификации | 15 | Correct event type. | Reviewed taxonomy; consistent event names; per-type errors; handling of multiple events. | A03, A08, A10, A11 |
| 4 | Rule justification / Обоснованность правил | 15 | Source and interpretation accuracy; no invented references or unsupported legal claims. | Verified source catalog; applicability and exact clauses when available; observable facts linked to each event; explicit incomplete grounding. | A04, A08, A11, A12 |
| 5 | Robustness / Устойчивость | 10 | Performance on previously unseen videos, including a different angle. | Locked unseen recording/scenario groups; angle checks; difficult normal lookalikes; visibility and uncertainty cases. | A06, A10, A18, A20 |
| 6 | Product readiness / Готовность продукта | 10 | Interface usability, reliable upload/processing, understandable results. | Public UI and API smoke checks; progress/results; authentication checks; corrupt-file and recovery behavior; local human UI review. | A13, A14, A16, A17, A19 |
| 7 | Technical solution / Техническое решение | 10 | Justified architecture, practicality, capacity for development, and locality compliance. | Reproducible deployment; exact model/runtime versions; architecture and resource rationale; egress-disabled analysis; measured performance. | A01, A15, A17, A18, A20 |
| | **Total** | **100** | | | A20 assembles evidence |

The first four criteria sum to **70 points**. Prioritize correct whole-video decisions, relevant original evidence, correct event types, and defensible source linkage. This is a team execution priority, not an organizer allocation formula.

## Published evaluation facts and unknowns

- The jury checks through the public asynchronous JSON API and automatically evaluates the prescribed schema (section 10; FAQ 13, 18). Public UI availability remains mandatory.
- At least five final videos are described (FAQ 12). The organizer does not specify simultaneous concurrency.
- A final check uses another camera angle (FAQ 4); robustness also covers previously unseen video (section 9).
- Detection criterion 1 uses a whole-video verdict (FAQ 15). The organizer does not publish a complete TP/FP/FN/TN-to-points formula.
- FAQ 16 asks for as few false alarms as possible and says missing a violation is unacceptable. It supplies no numerical FP/FN cost, threshold, or precision/recall target.
- FAQ 17 requires a concrete frame and timestamp. It supplies no numeric timestamp tolerance, temporal overlap rule, event matching algorithm, or evidence relevance formula.
- Section 10 and FAQ 14 impose a five-minute cutoff. No additional latency points or sub-second bonus is published.
- FAQ 15 mentions unjustified insufficient-data responses, while the mandatory API permits only `violation` and `no_violation`. Preserve uncertainty internally; do not introduce a third jury verdict without organizer clarification.
- Pitching is for shortlisted teams: five minutes plus two minutes of questions; slides optional; ties are resolved through pitching (FAQ 19-20).

No target, benchmark, hidden-test distribution, or guaranteed score should be inferred from these facts.

## Team measurement plan

This section specifies desired evidence, not organizer scoring rules and not achieved results.

| Evidence area | Required report content | Interpretation limit |
|---|---|---|
| Detection | Number of held-out independent groups/videos, TP/FP/FN/TN, per-class support, configuration versions, every failure. | A small local set cannot establish hidden-test accuracy or guaranteed recall. |
| Localization | Source time origin, event anchor convention, evidence frame IDs, reviewed timing errors, omission-event interval definition. | Do not invent an organizer tolerance; keep the full error distribution. |
| Classification | Taxonomy version, per-type support/errors, unsupported and ambiguous categories. | Training labels alone do not establish a complete event taxonomy; the organizer supplies no closed list. |
| Rule grounding | Source title/URL, jurisdiction, retrieval/verification date, current status, clause when verified, applicability, observable conditions, reviewer status. | Public availability does not prove legal applicability. Other-bank documents are not ForteBank policy. |
| Robustness | Group/angle separation, lighting/occlusion/lookalike slices when available, unseen cases, false positive/negative records. | Synthetic or local-teacher data is not independently verified ground truth. |
| Product | External UI/API check, key rejection, positive/negative/multiple-event flows, status/failure/recovery outcomes. | A working local screen does not prove public availability or reliable jury operation. |
| Technical | Exact package lock/model revision/weight hash, hardware, warm/cold configuration, local weights, egress-disabled result, resource use. | A current model card or newer library version does not establish task accuracy or deployment performance. |
| Latency | Upload start/completion, durable acceptance, queue/worker stages, result publication, first/full result-byte receipt; sample count, p50/p95 if meaningful, maximum, all timeouts. | Report inference, polling, transfer, and full-result cache modes separately. No cache-hit timing may be represented as uncached analysis. |

P1 proposes a 240-second safety objective, a 90-second warm full-video target, stretch goals, and five simultaneous distinct uploads. Those are team targets only. Replace claims with measurements or label them unmeasured. Publish content-safe aggregate metrics to coding agents; keep private frames, detailed per-video findings, annotations, and raw predictions inside the team's private environment.

## Scoring traceability checklist

- [ ] **25 / detection:** versioned whole-video confusion matrix on held-out recording/scenario groups; all failed jobs included in the accounting.
- [ ] **15 / localization:** original timeline and evidence-frame identity verified; concrete frames/times retained; review status recorded.
- [ ] **15 / classification:** reviewed taxonomy and per-type errors; multiple-event behavior checked.
- [ ] **15 / rule justification:** enabled rules have current verified public RK provenance and observable support; no invented source/section references.
- [ ] **10 / robustness:** unseen groups and another-angle testing reported honestly, including unavailable or provisional checks.
- [ ] **10 / product:** externally reachable UI/API and end-to-end progress/results, key checks, failure recovery, and readable explanations.
- [ ] **10 / technical:** reproducible locked deployment, local inference without internet, commercial-use component records, measured hardware/performance and stated limitations.
- [ ] Required compliance gates in [requirements.md](requirements.md) pass independently of likely rubric points.
- [ ] A20 submission evidence distinguishes completed validation, planned targets, and unresolved work; no projected point total is presented as an achieved score.
