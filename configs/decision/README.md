# Local visual decisions

`forte.decision.build_prompt` creates a repeatable request-bound prompt and JSON
schema. The caller supplies trusted generic fact questions and allowed taxonomy
IDs. `event_definitions` accepts the reviewed taxonomy's Russian observable
criteria. Rule titles, URLs and clauses are intentionally excluded from prompts
and are never accepted in model output; verified legal matching follows in the
event module. These prompts query natural language criteria without requiring
trained classification heads.

The raw model object contains `facts` and `candidates`. Each requested fact has
`fact_id`, `state`, `supporting_frame_ids`, and optional `notes`. Each candidate
has `candidate_id`, `event_type_id`, `start_frame_id`, `end_frame_id`,
`supporting_fact_ids`, `supporting_frame_ids`, and `visibility_notes`. Candidate
seconds are derived exclusively from the requested decoded frames. Unsupported
IDs, duplicate JSON keys, missing facts, extra source fields, truncation and
schema errors raise safe static errors. No invalid output becomes a negative.

`DecisionEngine.analyze(request, deadline=...)` uses A07's local async provider
and returns a `DecisionBatch` with shared fact/candidate objects, prompt/config
versions, prompt hash and inference metrics. `disposition=unclear` preserves
visibility uncertainty even when there are no candidates. `complete=True` means
the model response was structurally complete; it does not certify full video
coverage or legal applicability. The caller must retain coverage metadata.

`DecisionConfig(compact_frame_ids=True)` opts into deterministic request-local
`f00`, `f01`, ... frame aliases for inference. It defaults to `false` pending
local measurement. The engine copies the request, asks and validates using only
the aliases, then restores every structured fact/candidate support to the
original canonical frame IDs and returns the original `ClipRequest`. Media
paths, checksums, timestamps and dimensions are unchanged. Unknown or mixed
canonical/alias outputs and changed request bindings fail. Downstream evidence
never uses aliases. No speedup is claimed without a real benchmark.

The same boundary is exposed for serving benchmarks and training:
`aliases = compact_frame_request(canonical_request)` creates `aliases.request`
and read-only `canonical_to_alias`/`alias_to_canonical` maps. Call `build_prompt`
and the provider with `aliases.request`, validate its output against that same
request, and finish with `restore_frame_ids(batch, aliases)`. Training targets,
if this mode is selected, must uniformly translate all fact supports, candidate
start/end frame IDs and candidate supports with the identical per-request map.
The map itself stays private and is never included in a prompt or response.
Equal frame counts and fact/event sets produce the same schema frame enums;
different frame counts or fact/event bundles still produce different schemas.

The worker exposes these measured-deployment options through
`FORTE_COMPACT_FRAME_IDS=true|false` (default `false`) and
`FORTE_MAX_INFLIGHT_PER_JOB` (default `1`). Boolean spelling is case insensitive;
other flag values fail. The per-job limit must be a positive integer no greater
than `FORTE_MAX_CONCURRENT_CLIPS`. Deployment manifests retain their defaults;
operators select overrides only after the relevant serving measurements.

`DecisionConfig(concise_observations=True)` is a separate opt-in generation
mode, disabled by default. It constrains fact notes to null and candidate
visibility notes to one entry of at most 160 characters. Semantic validation
enforces the same limits even if a provider bypasses the generation schema.
All facts, present/absent/unclear states, source references and candidates remain
required as before. Incomplete visibility or context still requires unclear;
null notes never establish full observation or a clean result. The mode has a
distinct prompt version/fingerprint. Selecting it requires the corresponding
worker/operator settings and a measured rollout. Any character/UTF-8 byte reduction
from synthetic JSON comparisons is a proxy and is not a measured token saving
or an inference speed measurement.

Text depicted in media is untrusted evidence. Prompts explicitly reject media
instructions, identity inference, audio, hidden system facts and legal claims.
Strict schemas and request-bound validation enforce the output boundary; model
instruction following alone is not treated as a security guarantee. Free text
notes must remain data, must be escaped in a UI, and must never be executed.

The generation schema avoids `uniqueItems`, which the current vLLM XGrammar
backend does not support. Duplicate identifiers are still rejected by the
mandatory local semantic validator. Backend compatibility must be confirmed
with a real local serving request before claiming readiness.
[vLLM backend source](https://docs.vllm.ai/en/latest/api/vllm/v1/structured_output/backend_xgrammar/).

Prompt version `forte.visual-observation.v1.0.1` requests compact JSON with no
padding outside strings and brief candidate visibility notes. The actual A100
synthetic smoke exposed generation whitespace runaway with the server's default
whitespace policy; production serving also constrains whitespace in its grammar
configuration. A completion ending by token limit still fails even if a JSON
prefix looks plausible. Prompt instructions do not replace the serving fix or
strict completion validation.

`default.json` specifies the limits represented by `DecisionConfig`. Override
limits only with the matching supported version; never log responses or prompts
containing private observations. Generation temperature and local serving limits
are controlled by the model provider. There is no logit/calibration claim and no
Jev-equivalence claim.

Implementation uses Pydantic's current V2 validation and generated JSON schema:
[models](https://docs.pydantic.dev/latest/concepts/models/),
[configuration](https://docs.pydantic.dev/latest/concepts/config/).

## Optional temporal head path

The default Qwen provider is unchanged by these optional modules. A head for 19
observed facts may be used only with pinned encoder, feature recipe,
taxonomy and safetensors checkpoint identities. `head_contract` retains a
separate strict human-reviewed calibration track. `machine_contract` explicitly
marks organizer-video-binary plus local-teacher-silver calibration provisional;
it never claims human fact/event gold or independent fact/event accuracy.
`assignments_frozen` is the machine experiment's independently pinned assignment
freeze, and does not mutate the human-track split's `locked=False` state.

`ArtifactMachineEvidenceVerifier(MachineEvidenceFiles(...))` composes the actual
local A02 frozen assignment/exposure proof, A10 provided-label evaluation and
selection attestation, raw threshold-file hash, and source-bound teacher support
comparison artifacts. Class thresholds are never supplied by defaults. Each
eligible fact state needs class alternatives, explicit positive and negative
support references, and agreement recomputed from the frozen comparison rows.
Teacher agreement remains silver agreement, not accuracy. Each comparison must
join the exact actual A03 validation record, window and original target hashes,
including canonical source PTS, state, support, emitted prompt and focal question.
A nonempty comparison bundle also requires the pinned A07 teacher model manifest;
`teacher_processor_artifacts_sha256` derives the processor/shared-config digest
from its existing artifact entries. Missing or mismatched receipts cannot unlock
a fact class. Explicit focal unclear can provide a source-bound no-decisive-support
negative, and never creates an absent fact. A binary-only threshold file cannot
authorize 19 fact or event thresholds. Missing artifacts remain
unavailable; the verifier never backfills observations or prompt hashes.

`CachedTemporalHeadBackend` is an offline proof adapter for a pinned A09 feature
cache. It is not the arbitrary-upload service. The optional live path uses A15's
`serving.fast_service` with a persistent encoder and `NativeHeadRunner`, keeping
Torch outside the CPU worker. `HttpNativeHeadClient` accepts only a private
endpoint and returns a strict `NativeHeadResult` bound to the actual uploaded
`VideoAsset`. Every native integer PTS, canonical source frame ID, original tail,
global/ROI view geometry, keyframe flag and upright source dimensions are retained.
There is no known-recording/filename/label lookup or external model API.

`map_native_observations(result, asset, manifest, ...)` verifies the complete
native source binding and returns a private `NativeObservationPlan`.
`plan.requested_frame_ids` selects source references for A05 to materialize into
actual original JPEGs; `plan.bind_evidence_frames(actual_refs)` validates their
original PTS, canonical ID and JPEG SHA and returns the existing `DecisionBatch`.
No JPEG `FrameRef` is invented for an unmaterialized native frame. The existing
96-reference observation limit remains; an over-budget projection fails instead
of silently losing support. Final jury evidence stays subject to A11/A12 limits.

Window and per-frame state logits, source-support scores and trained-component
masks are checked independently. Untrained or uncalibrated branches yield unclear.
Absence requires verified native continuity, every original frame passing the
selected absent state/support threshold, and every explicit visibility/procedure
context passing its present state/support threshold throughout. Feature delivery
alone does not prove visibility. `BANK_CASH_WORKSTATION` remains unclear unless
the caller supplies `trusted_scene_context="bank_cash_workstation"`.

An eligible trained boundary head can propose explicitly paired original-frame
brackets. When positive event-boundary targets are unavailable, the separate
opt-in physical fact-run policy can propose candidates from independently
calibrated physical states and full rule context on contiguous native frames.
`load_physical_fact_run_policy` compiles complete gates from the pinned,
source-verified baseline regulatory catalog; `physical_policy_observation_definitions`
returns shared event gates and absence-context mappings. Unknown gaps split runs.
These brackets describe observed source support and do not establish exact action
onset or legal applicability. MIL scores and approximate organizer anchors never
select a legal event type or timestamp. A11's procedure/temporal/source guards
remain mandatory, and the unsupported 37-axis experimental vocabulary is rejected.

These optional interfaces have synthetic CPU contract tests. That does not
establish model quality, a valid calibration bundle, selection, deployment,
native-video accuracy, or upload-start-to-full-result latency.
