# Gemma 4 public-object adaptation: transfer limits and integration contract

18 October 2026. This is a source-based engineering assessment. It does not
establish completed Gemma training, improved bank-video accuracy, or deployment
eligibility. The execution owner records those results separately. No supplied
label file, image, model scene description or remote host was accessed for this
review.

## Expected transfer

Public banknote/ID-card QLoRA is a plausible appearance warm start. It is not a
directly supervised solution to the missing banking procedures. Its usefulness
to Forte must be measured with the frozen adapted model under the actual
pipeline prompts. The current pipeline never asks the earlier compact
classifier's question, `Identify the visible object. Answer exactly one label:
single_banknote or id_card.`

The relevant differences are visible in
[prompts.py](../../src/forte/fast/prompts.py) and
[analyzer.py](../../src/forte/fast/analyzer.py):

| Task | What the model receives and returns | What public object labels can establish |
| --- | --- | --- |
| Public positive-crop classifier | A publisher-boxed object crop; one mutually exclusive class | Appearance of an object known to be present inside that box |
| `STATE_PROMPT` / `STATE_SCHEMA` | A complete bank-room frame; eleven fields for roles, people, objects and actions | An indirect appearance representation; no complete eleven-field training target |
| `DESK_Q` / `DESK_SCHEMA` | An expanded desk crop; six independent booleans, with multiple objects potentially present | Individual object presence only when the corresponding label is exhaustive for that image; no justified values for the other fields |
| Layout | Full frame; eight normalized boxes or nulls | No desk, room, role, safe, laptop or door supervision from a two-object crop label |
| Verification | Several ordered frames and crops; visible/confidence/time fields | No action, temporal omission or violation supervision from a static object label |

`STATE.id_card_shown` means a document is being handed over or held and looked
at. `DESK.id_or_document` means an ID card or document lies on the desk. They
are different observations, even though current state assembly ORs the desk
answer into `id_card_shown`. A generic visible card does not establish identity
verification. Likewise, a visible banknote does not establish counting,
authenticity checking, cash acceptance, a completed handover or a cash stack.

Training a positive-crop class answer does not provide absent-object
calibration. It also teaches a different output format from either production
schema. Do not manufacture complete `STATE`/`DESK` JSON by setting unannotated
fields to false, translate a two-class prediction into all six booleans, or
use a banknote box to declare that no ID appears elsewhere. A frame may contain
both cash and documents.

## Where a change can help or hurt

The current rule code uses desk cash/document answers in enabled
`CASH_LEFT_UNATTENDED` and `PREVIOUS_CASH_ON_DESK` candidates. Cash/document
observations also contribute to client-session and collector logic. Better
appearance perception could therefore affect real candidates without a new
type. That is a possible mechanism, not a measured gain.

The same propagation creates regressions: a false desk document can become
`id_card_shown` and suppress `COLLECTOR_NO_DOCUMENTS`; a missed desk item can
remove an unattended-cash candidate; a false object can create one. This is why
two-class crop accuracy alone cannot qualify the model. A globally applied
language adapter may also alter layout, role recognition, laptop/seat answers,
verification confidence, JSON compliance and answer length. Narrow training
does not isolate its effect to the two object fields.

The disabled `NO_AUTHENTICITY_CHECK` and `NO_RECOUNT` types need action and
interval coverage; `KEY_LEFT_IN_SAFE` needs lock/key visibility; handover
irregularity needs object continuity through the exchange. Public static
banknote/card labels supply none of these. Their thresholds and enablement
must remain unchanged in an appearance-only model comparison. More model
parameters cannot recover details absent from the original pixels.

The earlier compact public-data fit improved its development crop macro score
from 92.1875% to 96.09375%, while its frozen sealed primary result was
94.53125%; the sealed evaluation had no base-model comparison. These numbers
demonstrate a limited component task, not a bank-video improvement. They do not
predict a Gemma result. The source/class confounding, small document-design
support, consumed sealed inventory and unknown upstream pretraining overlap
remain relevant. See the [measured public model cards](public-model-cards-20261018.md).

## Recipe advice before freezing

Use the admitted public training/development groups only and freeze the recipe
and final candidate before the 40-video regression. Keep the task name and
artifact card explicit: public object-appearance adaptation. A checkpoint
selected by public crop results can be tested on bank videos; bank scores
must not select steps, adapter rank, prompt variants or thresholds. Repeated
bank feedback does not become an independent test.

If public annotations genuinely support a task-aligned partial target, supervise
only those labelled facts with a documented loss mask. That requires a model
and evaluation contract that handles the partial labels explicitly; it must
not silently produce a fabricated complete production-schema target. Merely
wrapping the two-class answer in JSON changes syntax, not the supervised task.
Occlusion, small-scale and compression variants are correlated appearance
stress tests, not temporal procedure labels or independent camera support.

Avoid inventing a richer training task to fill the missing labels before the
deadline. A small, reproducible, frozen appearance experiment plus an honest
whole-pipeline comparison provides more useful evidence than unverified
procedure targets. Required lineage, training-update proof and evaluation
separation are in [the integrity protocol](gemma4-finetune-integrity.md).

## Default-off whole-model integration

Recommended implementation contract; this note does not implement a new flag:

1. Add an explicit proposed `gemma4_public_adapter_enabled: false` profile
   control with an immutable adapter/base/processor manifest binding. The off
   path must retain the configured baseline model exactly. The on path must
   require a verified local candidate endpoint and artifact; missing or invalid
   bindings fail readiness rather than silently selecting another model.
2. Resolve that choice once when constructing the staging pipeline's single
   `VLMClient`. Existing service endpoint/name controls are
   `FORTE_FAST_MODEL_BASE_URL` and `FORTE_FAST_MODEL_NAME`. The resulting adapted
   Gemma selection applies uniformly to layout, full-frame state, every crop,
   door questions and verification. Do not choose a model by filename, score,
   candidate outcome or known video. A field-specific specialist route would
   be a different experiment with its own flag and qualification.
3. Preserve exact prompts, schemas, input frames/crops, sample selection,
   rules, disabled types and thresholds for the first base-versus-adapter
   comparison. Preserve the image-token budget and serving settings or record
   a separately named candidate. Do not claim a server command or metadata
   annotation alone implements the required profile flag.
4. Record whether the adapted endpoint uses runtime LoRA or a merged model,
   plus the actual base representation, adapter and processor hashes. A fit
   under NF4 does not establish BF16/FP8 serving equivalence. Smoke every
   production schema and multi-image verification using synthetic inputs
   before real evaluation; malformed outputs and technical failures remain
   failures under the service failure contract.
5. Freeze and run the complete model twice on all 40 unique supplied videos,
   with labels available only to the scorer. Measure agreement, all failures,
   false alarms, events, timecodes, calls and latency. A full-model comparison
   is necessary even if only public crop accuracy improved.

The session's promotion target remains at least 37/40 and zero false alarms in
each of two runs, with all 19 negatives successfully processed. Those are
in-sample results. Evidence/schema/invariance requirements, isolated staging,
the six longest clips within 285 seconds, an exact one-step rollback, user
approval and the public upload check remain separate requirements. Passing
public crop metrics, successful optimizer steps or a server startup satisfies
none of those deployment gates by itself.

Review changes in this document: no trained weights, tuned parameters,
thresholds, model routing or live configuration changes.

## Qwen30 BF16 pivot: immediate recipe recommendation

The later user instruction selects actual Qwen3-VL-30B-A3B BF16 fine-tuning.
This addendum applies the task-alignment review to that candidate. The earlier
Gemma whole-model contract is historical advice for the separate Gemma
experiment; it does not require applying a narrow Qwen desk adapter globally.
The proposed rank-eight language q/v LoRA with frozen base and vision weights
is an actual weight update, but remains object-presence adaptation. It adds
no missing temporal or role ground truth.

The smallest defensible pipeline-aligned training task uses the exact
`SYSTEM`, `DESK_Q` and `DESK_SCHEMA`. The model sees only the public image or
its declared derivative; filenames, source, class and grouping metadata stay
out of the request. Supervision must distinguish known facts from missing
annotations:

| Input provenance | Supported target | Unsupported inference |
| --- | --- | --- |
| A publisher-labelled banknote object crop | `cash=true` while the labelled object remains visibly represented | `id_or_document=false`, or all four unrelated fields false, from the banknote label alone |
| A publisher-labelled document object crop | `id_or_document=true` while the labelled document remains visibly represented | `cash=false`, or all four unrelated fields false, from the document label alone |
| A generated empty neutral canvas | Empty-object targets by generation; useful cash/document negative | A real-desk false-alarm rate or unseen-camera validation |
| A composition using certified object masks/polygons on a known empty canvas | Presence of the inserted objects, including simultaneous cash and document when both are inserted | Treating an ordinary rectangular source crop as a certified foreground-only mask |
| An ordinary public background or a coin-labelled photograph | Only its explicitly exhaustive annotations | Assuming the absence of every unlabelled object |

If labels are partial, mask loss at the exact supervised boolean value tokens.
Masking the prompt and supervising the entire JSON completion still teaches
fabricated negatives. A trainer optimized for a contiguous answer-token suffix
must be changed and checked for discontiguous masks. Placeholder values for
unknown fields also occur in autoregressive context even when their loss is
masked; record their policy. Do not report those placeholders as ground truth.
Severe degradation that destroys recognizability cannot inherit a confident
positive target without a separately justified visibility policy.

For full-schema training, a useful optional retention alternative is frozen
Qwen30 BF16 inference on the same public images with the exact desk prompt.
Its unrelated four booleans may supply low-weight distillation targets or KL
retention. Mark them as base-model pseudo-labels, not human truth. Override only
the facts justified by source labels. This helps avoid explicitly training
universal absence of drinks, bags, coats and coins. It does not establish that
the base answers are correct. Teacher inference stays local and exposes only
closed structured answers to the agent.

Use positive, certified-negative and co-presence cases in the public
development scorer; a positive-only classifier score cannot select a desk
presence model responsibly. Composition assets, document designs, source
recordings and derivatives must remain within their source group and split.
Generated backgrounds must not encode class through a fixed colour, scale or
layout. Report real public crops and synthetic composition/degradation views
separately. A synthetic-only negative gain is limited evidence, and existing
consumed sealed material cannot be called a fresh holdout.

Select the frozen checkpoint using public development measurements only:
supported cash/document per-field precision and recall, negative false
positives, invalid outputs, and co-presence accuracy. Use a predeclared macro
summary and preserve class-specific regressions. Report teacher retention for
the four unlabelled fields separately from supervised accuracy. The 40 supplied
bank labels remain scorer-only; their scores do not select rank, steps,
prompts, routing or operating points.

For this narrow candidate, prefer an explicit default-off desk adaptation flag
over global adapter application. A desk-only endpoint still emits all six
fields and can suppress drinks or bags after narrow training. The conservative
field-limited route keeps the base desk answer's four unrelated fields and
allows the adapter to refine only validated cash/document presence. That
requires an extra desk call; measure its latency and record this as a separate
pipeline candidate. A single adapted all-six-field desk call needs measured
retention validation. A globally adapted model needs complete layout, state,
seat, laptop and verification regression in addition.

No route may turn missing, invalid or failed adapter output into negative
findings. The off path retains the baseline. The on path requires exact
base/adapter/runtime bindings and the service's failure semantics. No production
switch follows from public-object gains. In particular, nothing in this recipe
alone establishes at least 37/40 or repairs the disabled omission/key types.

### Selected Qwen desk candidate and synthetic-label conditions

The execution owner selected four generated public-only conditions: banknote
only, document only, both objects, and blank. The selected integration asks the
base and adapted desk models in parallel and replaces only `cash` and
`id_or_document` in the base answer. The other four base booleans remain exactly
unchanged. Training uses NF4; intended inference uses BF16. This distinction
must remain in the artifact/evaluation record. The extra desk request increases
logical state-stage requests from five to six when all four normal crops are
present, or 20% for those states; whole-video call and latency overhead depend
on actual layout availability, verification and batching and must be measured.

Complete six-field targets are justified for a synthetic image only when its
known empty generated background and its actual pasted foreground pixels
establish the relevant absence. An annotation bbox is not an object mask:
corner pixels can contain another item, and eroding a bbox does not prove they
are gone. A document quadrilateral can define a foreground mask; its validity,
coordinate convention and transformed support must be checked. For banknotes,
retain the documented isolation method and any uncertain-source exclusions.
Do not describe a heuristic isolation as ground-truth certified segmentation.

For `both`, the final image must retain visible pixels from both inserted
objects; record per-object surviving support after transforms and compositing.
Keep them non-overlapping for the minimal experiment, clip no object outside
the canvas, and do not reduce either to a few indistinguishable pixels while
keeping an unquestioned positive target. Use the same declared size,
position, background and photometric distributions across conditions so the
generator does not reveal the answer through a fixed layout or colour. A
blank case contains no pasted source pixels. Source image/document groups and
all generated descendants remain in their assigned public partition.

The six false/true labels in these generated scenes do not certify six-field
accuracy on actual desks. In particular, no generated condition supplies a
positive cup, cash bag, coat or coin example. Keeping those four outputs from
the base model limits this training-induced absence bias at integration.
The copied document's fictional portrait remains incidental pixels, with no
face, OCR or identity objective.

This generated-domain task removes the old classifier's forced exclusivity
and matches the desk prompt, but still omits real clutter, hand occlusion,
scene geometry and procedure supervision. Existing training inventory lists
36 ID-positive source images, not 36 independent document designs or people.
Repeated crops, scales, canvases and pairs do not increase that independent
support. Generated blank success likewise supplies no bank-negative gate.
Publish grouped public development results separately from the two complete
bank regressions and the measured latency qualification.

### Actual source-geometry finding

The data owner inspected the existing source metadata: MIDV retains the
publisher document polygon as `source_record.source_quad`; KZ-BD retains
normalized YOLO boxes in `source_record.source_objects`, with no banknote
polygon. The old positive-crop bundle carries boxes only. Consequently, treating
a pasted KZ rectangle as an isolated physical banknote would overstate the
available annotation. This is a target-support issue, not a request to acquire
another dataset.

The immediate existing-assets remedy is partial supervision: a note rectangle
supports `cash=true` but not `id_or_document=false`; a both-object composite
supports both true values because the labelled note and inserted document are
present. Mask the unrelated four fields for these cases. A blank generated
canvas supports both false values. A document scene constructed through the
actual polygon-isolation path must retain that provenance and explicitly record
the cash-absence assumption and any source-isolation limitations. If that
assumption cannot be supported, mask `cash` there too rather than invent a
negative; the blank cases still supply certified synthetic negatives. Per-field
development metrics must use only supported targets and disclose their
denominators.
