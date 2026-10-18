# Experimental native-frame scope plan

**17 October 2026, Asia/Qyzylorda. Status: generated metadata only, not released or selected.** No private media, actual media generation/decoding, processor/model forward, GPU call, production source edit or protocol change was performed by A14. The frozen four cash branches/nineteen facts remain unchanged.

The concrete first experiment is **exact native-PTS image chunks of at most sixteen original frames**, with explicit source coverage and shared seam frames. It reuses the existing image input type. A separate native-video-tensor experiment could reduce visual token cost, but the current HTTP service does not support it. Complete recorded-frame delivery supplies input-coverage evidence; it never automatically supplies semantic continuity, adequate visibility or a complete banking procedure.

The [machine-readable plan](../../configs/experiments/qwen38-native-scope.generated-plan.v1.json) includes all nineteen exact public fact definitions, fact-specific minimum witnesses, source pins, four generated PTS fixtures and six token-budget estimates. [Generated check receipt](../../qa/integration/qwen38-native-scope.generated-feasibility.v1.json): four ledger partition checks pass; no media or model was involved.

## Positive observations and complete-scope claims are different

There is **no universal all-native-frame prerequisite for every positive action**. In particular:

- `CASH_HANDLER_LEAVES_WORKSTATION` asks for an observed transition across a reliably known boundary. Its exact definition is: «Виден переход от рабочего места за его надежно определенную границу. Наклон, поворот, заслонение и шаг внутри рабочего места уходом не считаются.» An unambiguous same-handler crossing can establish this positive observation; neither all native frames nor three frames within 1.1 seconds is specified by the catalogue.
- `CASH_COUNT_VISIBLE` asks for a visible handling sequence or device operation: «Наблюдается последовательное перебирание наличных либо их помещение в счетное устройство и его работа; точность суммы не утверждается.» A static hand/stack pose is insufficient, but this positive definition does not demand the entire counting opportunity or every native frame.

These action predicates alone do not satisfy their legal branches. For example, an actual departure still needs the same unfinished receipt, cash and the separate required visibility facts. A positive observed count is also not universal proof that no different procedure omitted counting.

The following distinctions come directly from [the frozen public fact definitions](../../rules/catalog.json). Exact Russian questions/definitions and absence requirements are copied into the separate plan for auditability; the English descriptions below are implementation interpretations.

| Fact | Minimum positive witness and scope | What must not be promoted from sparse points |
| --- | --- | --- |
| `CASH_VISIBLE` | Distinguishable cash at a labelled instant, with sufficient resolution | A sampled set's absence does not cover unobserved times or surfaces |
| `CASH_ON_WORK_SURFACE` | Cash and the surface supporting it visibly coincide at the instant; no ownership claim | Point geometry does not prove continuous storage/absence |
| `CASH_HANDLER_PRESENT` | Physically present person whose cash-handling role is established by observed action; no identification | Absence additionally needs sufficient continuous work-zone/occlusion coverage over its claimed scope |
| `BANK_CASH_WORKSTATION` | Trusted bank-cash setting plus visibly established workstation | A logo/clothing or camera frame alone cannot supply trust; more frames cannot supply an absent operator attestation |
| `CASH_HANDLER_LEAVES_WORKSTATION` | Unambiguous same-handler transition across a known boundary; enough observed context to reject bending/turning/occlusion/within-zone movement | No universal frame-count/duration rule; a before/after disappearance alone may remain ambiguous |
| `CASH_COUNT_VISIBLE` | Actual sequential handling, or cash feeding plus visible counting-device operation; no amount correctness | A hand pose, static stack or motion cue does not establish count; absence needs the full relevant opportunity |
| `CASH_HANDOVER_VISIBLE` | Explicit **continuous** movement of the same cash from one hand region to another; no transaction type inferred | Three separated poses or matching endpoints do not prove the required path |
| `CASH_PLACED_IN_PERSONAL_CONTAINER` | Explicit **continuous** original-object/motion/destination view; bag purpose/permission not assumed | Unknown contents, offscreen motion and a destination pose cannot prove the movement |
| `CASH_PLACED_IN_CLOTHING_POCKET` | Discernible cash followed **continuously** into the actual visible clothing pocket | Bag/cassette/drawer/paper lookalikes and an unknown gap cannot prove pocket placement |
| `PREVIOUSLY_ACCEPTED_CASH_TRACKED` | Same cash tracked **continuously** from completed prior acceptance to placement | Position cannot prove ownership/history; a dense short clip alone cannot establish the prior receipt |
| `CASH_RECEIPT_OPERATION` | Direction and purpose established by the full visible actual client receipt stage | A generic handover is not automatically receipt; exclude issue/exchange/internal/client-owned cash |
| `CASH_ISSUE_OPERATION` | Full visible actual cashier-to-client issue stage | A generic handover is not automatically issue |
| `LOOSE_BANKNOTES_RECEIVED` | Actual receipt and distinguishable loose notes; sealed bag/bundle excluded | Packaging at one instant does not establish receipt purpose |
| `LOOSE_BANKNOTES_ISSUED` | Actual issue and distinguishable loose notes; intact packaged-amount branch excluded | Packaging at one instant does not establish issue purpose or negate the paragraph 32 exception |
| `CUSTOMER_SERVICE_CONTINUES` | Same receipt stage visibly continues before/after departure; relevant service-completion boundary resolved | A client's presence or pause alone is insufficient; an unknown finish boundary remains unclear |
| `PREVIOUS_RECEIPT_COMPLETE` | Start, acceptance and completion of the relevant previous service established in context | A client leaving or a pause does not establish completion |
| `PROCEDURE_CONTEXT_COMPLETE` | Actual relevant start/end and zones with no unknown fragments | The video start/end or complete sparse request schedule does not establish procedure boundaries |
| `ALL_COUNTING_PATHS_VISIBLE` | Full relevant preparation/receipt/issue, including hands, cash and manual/automatic routes | A hidden device, time before recording, blind zone or unknown fragment remains unclear |
| `WORK_AREA_CONTINUOUSLY_VISIBLE` | Key relevant actions traceable without cuts, occlusion or leaving frame | All recorded frames may still contain obscured zones, unreadable detail, cuts or unobserved off-camera actions |

For explicitly continuous predicates, showing every recorded native frame in the actual witness interval is a concrete, testable **coverage strategy**, not a newly invented legal frame-rate requirement. Adequate resolution, no occlusion/cut, object/body continuity and actual scene scope must still be answered. A native PTS chain proves representation of the recorded signal; it cannot prove infinitely continuous real-world visibility between camera exposures or recover source recording gaps. Any unresolved source/visibility gap stays unresolved.

## What the current code can consume

- `iter_rgb_batches(sample_fps=None)` in [video batches](../../src/forte/video/batches.py) can exhaust the original decoded frame chain, including actual tail, source origin, rational time base, native PTS and source hashes. `iter_boundary_batches` can deliver every native frame of a bounded bracket and verify its exact PTS set against a completed native index. Supply that completed decoded index; do not treat compressed-packet counts alone as a universally valid frame-count proof.
- The ordinary [sampler](../../src/forte/video/sampling.py) asks for nearest frames at a finite list of times. Current refinement uses `min(4,16/span)`, so widening a request reduces density. It does not automatically request every native PTS. Explicit `CoverageWindow` targets at actual indexed PTS can be bound to original JPEG refs, but the production pipeline does not currently plan those native cohorts.
- The existing provider transmits only text plus original JPEG `image_url` blocks. The [HF service parser](../../serving/qwen38_service.py) rejects a `video_url` block, calls the processor with `images=...`, and requires `pixel_values` plus `image_grid_thw`. Native video therefore needs an isolated new provider/service/manifest path; sending a different message block is not sufficient.
- Current serving uses sixteen images, 32,768 total tokens and 4,096 maximum generated tokens, with actual processor-length checks and no truncation. Generic validation accepting up to 64 configured images does not override the active sixteen-image manifest or its full-family grammar warmup. Those remain frozen.

The official model describes native image/video capability, but that is not proof of this project's deployed route. Its video examples warn that default sampling is two fps with frame sampling enabled. The proposed video mode must disable that default and audit what is actually presented. [Official Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B), [official processor implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py).

## Concrete bounded input strategies

**A — exact native-image chunks, first feasibility target.** Build a complete native ledger once; select a short event bracket by source ticks; divide its actual ordered frames into at most sixteen per request with one original seam frame shared. Send full scene context at adequate resolution with exact source IDs. Request fact-specific positive witnesses separately from procedure/visibility questions. Keep the original coarse/prior context available; do not call a short event bracket a complete receipt by itself.

Generated metadata checks:

| Generated source ledger | Native frames | Sixteen-image calls with one seam frame | Meaning |
| --- | ---: | ---: | --- |
| 8 seconds at 25 fps | 200 | 14 | Complete native presentation is much more costly than one sparse window |
| 2 seconds at 30 fps, nonzero origin | 60 | 4 | Origin is preserved in integer ticks, not frame-index/FPS substitution |
| 16 frames at 25 fps | 16 | 1 | 0.64-second recorded cohort; first-to-last PTS span is 0.60 seconds |
| 17-frame VFR ledger, nonzero origin | 17 | 2 | Variable frame gaps and the final frame are preserved |

The seam proves original-frame identity across chunks. It does not independently prove that an ambiguous object is the same object; the tracker/teacher must resolve that. A transfer crossing a seam may need the two neighbouring native cohorts considered together with explicit track state. An unconfirmed seam cannot be silently treated as continuity. Purely nonoverlapping requests provide weaker transition context; overlap or a dedicated seam witness is preferable.

**B — separate native-video tensor mode.** Present a manually decoded, ordered native RGB tensor, not a path/URL that may be resampled implicitly. Set `do_sample_frames=False`; record exact frame count, processor grid, any odd-frame padding/duplication and the effective pixel/token policy. Initially bound each experimental video cohort to **64 original frames**; this also stays inside the existing abstract 96-frame fact-reference ceiling, although the HTTP/video integration is new. Longer source procedures need multiple linked cohorts or an independently designed larger frame-reference contract.

The frozen public processor configuration uses patch size 16, merge size 2 and temporal patch size 2. A video temporal token can combine two original frames; that does not justify dropping one from the source ledger or locating an event from a rounded processor midpoint. Actual video processor behavior and grid accounting must be checked in the exact candidate image. A08 owns that separately authorized generated CPU processor audit; its result is not presumed here.

## Clock, codec and evidence requirements

Preserve original `source_pts`, reduced rational `time_base`, nonzero `source_origin_pts`, exact next-PTS relation, source dimensions/rotation, stream identity and content hash. Normalized time is `(source_pts - origin_pts) * numerator / denominator`; evidence uses that original relation. Decode presentation order, not packet/decode order or nominal FPS. B-frame reordering, variable frame rate and orthogonal display rotation must be handled. Missing/corrupt/duplicate/nonmonotonic PTS, incomplete decode or unsupported transformations must fail closed. No interpolation, generated frames, audio or face identification is introduced.

For processor metadata a candidate exact clock encoding is `frames_indices=(source_pts-origin_pts)*numerator`, `fps=denominator`, while retaining the integer/rational ledger independently in text. This avoids rounding to a millisecond index. The processor can still average temporal pairs and print timestamps to tenths; those displayed values are not the evidence clock. Odd padding repeats must be labelled as padding and excluded from original-frame counts/support provenance.

Future generated codec checks should include H.264/MOV with reordered frames, a VFR/nonzero-origin fixture, rotated source and deliberately missing/duplicate/invalid PTS failures. No codec fixture or actual media was created by this planning step. Source facts remain visual-only. Public evidence remains one to three actual original JPEGs; private native input may contain many more original frames. If the final required evidence cannot fit the existing policy, defer rather than fabricate a montage or alter evidence pixels.

## Token budget: estimates, then actual processor checks

For processed dimensions divisible by 32, a planning estimate is `H/32 * W/32` tokens per image, and `ceil(N/2) * H/32 * W/32` for a native video with temporal patch size two. These follow the frozen patch/merge configuration; actual dynamic resizing, grids, timestamps, prompt text and padding must be measured. Reserve 6,144 text/clock tokens, 4,096 generation tokens and 1,024 headroom in this plan. Those reserves are hypotheses, not processor receipts.

| Proposed processed input | Estimated visual tokens | Estimated total including reserves | Within 32,768 estimate? |
| --- | ---: | ---: | --- |
| 16 images, 1024×1024 | 16,384 | 27,648 | Yes |
| 16 images, 672×384 | 4,032 | 15,296 | Yes |
| 64 native video frames, 512×512 | 8,192 | 19,456 | Yes |
| 200 native video frames, 448×448 | 19,600 | 30,864 | Yes, **token estimate only; exceeds initial 64-frame experiment cap** |
| 200 video frames, 512×512 | 25,600 | 36,864 | No |
| 240 video frames, 448×448 | 23,520 | 34,784 | No |

Actual `input_ids` plus requested generation must fit the active context without truncation; padding, prompt/clock ledger, every video temporal group and actual grid must be accounted for. A token fit is not a memory/latency/visual-detail fit. Downscaling that removes cash/device/object detail cannot establish the corresponding fact merely because the request fits.

## Separately measured generated CPU processor result

A08 subsequently ran the exact candidate image's processor on generated RGB arrays, with no weights, GPU, network or private media. The [seven-case receipt](../../infra/qwen38-native-video-cpu-processor-audit-20261017.json) has SHA-256 `fa00b9af1d265ebcc922afd8abe98a70fc4beaa90efe6f5710e173be98456b63`. This is a separate processor-only experimental prompt/schema contract; it does not reuse or expand acceptance under the frozen sixteen-image profile.

All seven cases recovered every original RGB hash by inverse patchification. Explicit `do_sample_frames=False`, `do_resize=False` and fail-on-call sampler/resize guards recorded zero sampling or resizing. The seventeen-frame VFR case padded to eighteen only by duplicating the final frame; its external source ledger still contains seventeen original frames. Exact integer/rational clock ratios were preserved independently of rounded displayed temporal-patch midpoints.

| Actual generated video input | Actual prompt plus 4,096 output tokens | Remaining tokens within 32,768 | Result |
| --- | ---: | ---: | --- |
| 16 frames, 512×512, 25 fps ledger | 8,899 | 23,869 | Fits |
| 16 frames, 512×512, 30 fps/nonzero-origin ledger | 9,030 | 23,738 | Fits |
| 64 frames, 512×512 | 16,051 | 16,717 | Fits |
| 200 frames, 448×448 | 30,515 | 2,253 | Fits processor token bound only |
| 200 frames, 512×512 | 36,515 | −3,747 | Reject |
| 240 frames, 448×448 | 35,355 | −2,587 | Reject |
| 17 VFR frames, 448×448 | 8,714 | 24,054 | Fits |

These measured values supersede estimates only for those exact generated cases and their pinned experimental compact clock-ledger prompt. The processor's ordinary total-video-pixel policy may otherwise reduce spatial detail; explicitly disabling resizing here does not establish that an unmodified service or other processor invocation preserves the arrays. The 200/240-frame tests deliberately bypass the frozen compact alias validator and remain outside the initial sixty-four-frame integration proposal. No model memory, generation latency, visual correctness, continuity judgment, deployed-video route or public evidence was established by this CPU receipt. A08 is separately checking the exact vLLM interface; a raw `video_url` decoder that silently samples is not an equivalent input contract.

## Teacher and acceptance gates for a future released experiment

1. Preserve three separate notions: **input cohort**, **target scope** and **support frames**. A point label stays point-local even if sixteen context frames accompany it. All sixteen sparse sample labels matching can describe that requested set; it does not become continuous interval/procedure truth. The current production `FactDecision` has no explicit scope field, so exporting cohort-only absence under the production complete-procedure prompt would be a semantic error. Keep such targets masked at that boundary, or use an explicitly separate scoped training prompt/protocol.
2. For `LEAVES` and `COUNT` positives use their minimum public definitions above. Do not invent a universal number of before/during/after frames, duration or all-native requirement. Clear direct positive witnesses remain machine SILVER. Ambiguous poses stay unclear/masked.
3. For explicitly continuous paths, require a direct scoped teacher question about the same object/path, an independently source-bound coverage receipt, all necessary original frames actually presented, adequate visibility and no unresolved cut/occlusion. A receipt is necessary input evidence for this chosen strategy, not a label generator.
4. For counting omission, observe the entire relevant actual receipt/issue/preparation from defensible start to completion, all manual/automatic paths and relevant exceptions. Count-negative snapshots and a low proposal score cannot replace this. Hidden machines or a start before recording remain unclear, even with every recorded frame decoded.
5. A generated CPU stage may check processor inputs/PTS/grids/padding/token lengths without loading weights. Any later generated-model or TRAIN-only teacher stage requires its own explicit release, exact source/profile pins, attempted-call accounting, cancellation/terminal settlement and a bounded budget. Do not touch validation/final-check material to create scope truth or tune thresholds.
6. Measure positive-witness recovery, whole-procedure coverage/uncertainty, repeatability and upload-to-useful-result cost separately. Fourteen image calls for eight seconds is a cost warning, not a sub-ten-second claim. If one proposal stage is used, it must supplement complete coverage and cannot make empty/unsupported candidates clean negatives.

The next decision is between a minimal short native-image witness experiment and a separately audited native-video prototype. Neither is activated by this plan, and neither guarantees that existing footage contains the visibility and procedural context required for all nineteen facts.

