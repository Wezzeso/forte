# ForteBank Datathon: 20-Agent Execution Plan

**Purpose:** give 20 future coding subagents a complete, coordinated implementation brief for a high-scoring, fast, locally executed video-analysis service.

**Status:** implementation plan only. The application, training runs, latency measurements, and legal mappings described here have not been implemented or validated by this document.

**Planning date:** 17 October 2026, Asia/Qyzylorda.
**Execution budget:** 20 hours from dispatch. Recalculate the gates if less time remains.
**Resources:** reuse team-controlled GPUs first. A17 may operate the user's project-related PC/browser workflow and prepare or provision NVIDIA Brev capacity under the execution authorization, spending policy, and current tool rules in section 20. Benchmark actual capacity.
**Brand direction:** A19 independently researches Forte's brandbook and official visual language; A16 implements the reviewed design system. There are still exactly 20 agents.
**Primary objective:** maximize coverage of the published 100-point rubric while meeting every mandatory interface and locality requirement.
**Performance objective:** a responsive upload/polling API and genuinely fast uncached video analysis.
**Scope:** exactly 20 implementation assignments, A01 through A20. This document does not launch them.

---

## Contents

1. [Strategy and fixed decisions](#1-strategy-and-fixed-decisions)
2. [Organizer requirements and score coverage](#2-organizer-requirements-and-score-coverage)
3. [Architecture and component boundaries](#3-architecture-and-component-boundaries)
4. [Public API contract](#4-public-api-contract)
5. [Internal contracts and repository ownership](#5-internal-contracts-and-repository-ownership)
6. [Latency objectives and measurement](#6-latency-objectives-and-measurement)
7. [Data, annotations, and evaluation](#7-data-annotations-and-evaluation)
8. [Rules, evidence, and uncertainty](#8-rules-evidence-and-uncertainty)
9. [Model and training plan](#9-model-and-training-plan)
10. [Serving and performance engineering](#10-serving-and-performance-engineering)
11. [Twenty-hour schedule and dependencies](#11-twenty-hour-schedule-and-dependencies)
12. [Shared instructions for every subagent](#12-shared-instructions-for-every-subagent)
13. [Twenty copy-ready assignments](#13-twenty-copy-ready-assignments)
14. [Integration protocol and decision gates](#14-integration-protocol-and-decision-gates)
15. [Acceptance matrix](#15-acceptance-matrix)
16. [Submission and pitch](#16-submission-and-pitch)
17. [Failure decisions and cut list](#17-failure-decisions-and-cut-list)
18. [Operator launch prompt](#18-operator-launch-prompt)
19. [Sources and evidence status](#19-sources-and-evidence-status)
20. [PC control and NVIDIA Brev GPU provisioning](#20-pc-control-and-nvidia-brev-gpu-provisioning)
21. [Forte brandbook research and design system](#21-forte-brandbook-research-and-design-system)

## 1. Strategy and fixed decisions

Build a narrow visual decision system around an existing open-weight vision-language model. The learned component observes actions and proposes events; ordinary software controls timestamps, rule applicability, evidence, persistence, and the jury response.

Use one selected checkpoint for both screening and refinement, replicated across GPUs. Keep the original untuned checkpoint as the release fallback. A fine-tune must earn selection through validation.

The Jev-inspired element is the interface: specific questions, short structured decisions, explicit uncertainty, and composition in code. This plan does not claim to reproduce Jev's proprietary architecture, reinforcement-learning method, parallel sampling, published speedups, or calibration.

### Decision register

| Area | Default choice | Alternative and switch condition |
|---|---|---|
| Visual model | Qwen3-VL-8B-Instruct | Test 4B only if measured latency or capacity remains inadequate and held-out quality remains acceptable |
| Model precision | BF16 when supported | Quantize only after a measured benefit and quality regression check |
| Adaptation | Small LoRA, frozen vision encoder initially | Release untuned baseline if the adapter does not improve meaningful validation errors |
| Decision output | Short schema-constrained facts and candidate events | A parallel classification head is a stretch experiment, never a release dependency |
| Model serving | Private vLLM service, exact tested version pinned | A validated Transformers runner if serving integration is still blocked at hour 2 |
| Public API | FastAPI, CPU-only web processes | Add API workers only after profiling; never duplicate GPU models in web workers |
| Durable jobs | PostgreSQL with leased jobs | One database serves job state and queue; do not add another broker without a demonstrated need |
| Files | Local NVMe on a single multi-GPU host | Cross-host storage only when deployment requires it and its performance has been tested |
| Frontend | Small React application, same-origin API | A plain web interface is acceptable if React threatens delivery |
| Packaging | Docker Compose, local weights, locked dependencies | Avoid Kubernetes and distributed training infrastructure unless already operational |
| Public deployment | TLS reverse proxy in front of the API | Model, database, metrics, and worker control ports remain private |
| GPU acquisition | A17 inventories servers and prepares a live NVIDIA Brev quote | Rent with explicit purchase authority, known spending caps, and required tool gates |
| PC/browser operation | A17 is the provisioning operator; A20 serializes shared-device input | Prefer an available provider CLI/API; use documented browser/native controls as appropriate |
| Forte identity | A19 supplies source-backed research, tokens, assets, and visual QA | If no authentic brandbook is found, label website-based guidance as inferred |
| UI language | Russian by default for the local case | Other languages are allowed; keep event terminology consistent |
| Extra models | None on the critical inference path | A licensed local teacher may help annotation only if setup and review fit the schedule |

Qwen's model card establishes video capabilities and an Apache-2.0 license for the reference checkpoint. It does not establish performance on this dataset. [S3]

### Priority order

1. Mandatory compliance, exact API, and end-to-end completion.
2. Detection, event localization, correct type, and defensible source linkage.
3. Robustness to another angle and normal lookalike actions.
4. Measured latency and reliable concurrent handling.
5. UI clarity and a reproducible deployment.
6. Additional experiments only after the preceding gates are stable.

No plan can guarantee 100 points. The organizers have not published a complete scoring formula, event-matching tolerance, or hidden test distribution. Do not invent them.

## 2. Organizer requirements and score coverage

### Published scoring

| Criterion | Points | Planned evidence | Primary owners |
|---|---:|---|---|
| Presence/absence detection | 25 | Whole-video TP, FP, FN, TN on held-out recordings | A07, A08, A09, A10 |
| Localization and evidence | 15 | Source timestamp, relevant original frames, localization error distribution | A05, A06, A11, A12 |
| Event classification | 15 | Reviewed taxonomy, per-type error analysis, consistent event names | A03, A08, A10, A11 |
| Rule justification | 15 | Verified applicable document, clause when available, visible support, no invented references | A04, A08, A11, A12 |
| Robustness | 10 | New-angle and unseen-scenario checks, uncertainty behavior | A06, A10, A18, A20 |
| Product readiness | 10 | Reliable upload, progress, clear result, Forte-aware design, public access | A13, A14, A16, A17, A19 |
| Technical approach | 10 | Practical local architecture, reproducibility, measured operation | A01, A15, A17, A18, A20 |
| Total | 100 | Complete evidence bundle | A20 |

The first four criteria account for 70 points. The brief contains no published extra points for millisecond latency; speed protects the processing cutoff, product quality, and demonstration reliability.

### Mandatory organizer facts

- A working frontend and backend accept a new video without manual code execution.
- A public UI URL and API URL are required.
- The jury supplies .mov files through multipart field file.
- Both jury endpoints use the X-API-Key header.
- Analysis is asynchronous: upload returns a job ID; polling returns status and result.
- Completed output has exactly two verdict values: violation or no_violation.
- Support more than one event in a video.
- Every event needs a type, timestamp in seconds, concise observable explanation, and 1-3 JPEG evidence frames encoded in base64.
- Show the relevant rule and source. The main scenario requests document and clause/section.
- If unavailable or no result is returned within five minutes, the video counts as unprocessed.
- Inference must run inside the team's deployed service and work without internet after weights are downloaded.
- External cloud AI calls must not receive videos, frames, or their descriptions.
- Analyze visuals only. Do not use audio or identify faces.
- Use public Kazakhstan normative sources relevant to the observed procedure.
- Do not assert facts requiring unavailable banking systems, logs, accounting entries, or intent.
- Results support human checking; use wording about possible signs rather than definitive disciplinary findings.
- Pretrained open-source components must have licenses allowing commercial use.
- Do not train on, publish, or transfer jury test videos to third parties.

### Ambiguities to handle conservatively

| Wording issue | Implementation decision |
|---|---|
| FAQ mentions insufficient data, but the mandatory schema defines two verdicts | Keep visibility and uncertainty internally and in the UI; do not add a third jury verdict |
| Main scenario requests exact clauses; FAQ allows less-specific references and other banks' open documents | Seek exact applicable RK clauses; label incomplete grounding; never present another bank's policy as ForteBank policy |
| Pretrained components are allowed; ready-made solutions are banned | Build the original case-specific pipeline and document component licenses and original work |
| Deadline is stated as 12:00 without a calendar date in the supplied brief | Operator confirms the actual event deadline and timezone; execution proceeds with relative time gates |
| Five-minute timing origin is not precisely specified | Measure upload start, upload completion, durable acceptance, and result availability; budget conservatively from upload start |

## 3. Architecture and component boundaries

~~~mermaid
flowchart TD
    U[Browser or jury client] --> P[TLS reverse proxy]
    P --> A[CPU-only FastAPI]
    A --> S[Durable upload on NVMe]
    A --> Q[PostgreSQL jobs and leases]
    Q --> W[Independent worker]
    W --> D[Decode source timestamps and rotation]
    D --> C[Coverage windows and motion cues]
    C --> M[Private warm model replicas]
    M --> E[Candidate events]
    E --> R[Detailed verification using same checkpoint]
    R --> T[Temporal logic and verified rule catalog]
    T --> F[Original JPEG evidence]
    F --> J[Immutable jury result JSON]
    J --> A
    A --> U
~~~

### Critical separations

- API processes never run model inference, video decoding, training, or evidence assembly.
- Workers never publish a successful result before all required evidence exists.
- The model never selects a legal source by inventing a title or clause.
- The production pipeline never reads the label workbook or uses dataset filenames to decide a verdict.
- Model replicas stay warm and independent of API worker restarts.
- Training has dedicated resources and must not compete with serving benchmarks.
- The public response is a projection of a richer internal event object; public schema changes require A01.
- One selected checkpoint may have several GPU replicas. Replication is capacity scaling, not an ensemble.

### Recommended resource reservation

A17 inventories existing resources before renting anything. Section 20 defines PC/browser operation, quote comparison, spending limits, authentication handoffs, provisioning checks, judging keep-alive, and cleanup. A19 researches the brand independently.

Begin with one working inference replica. Reserve capacity for up to four warm replicas on suitable 48/80 GB-class GPUs, or equivalent tested hardware, plus a separate training GPU allocation. These are provisioning suggestions, not measured memory requirements.

Prefer a single host with NVMe and several GPUs during this 20-hour build. It reduces file-distribution and network failure modes. If GPUs span hosts, A17 must establish and test the media transport contract before A15 advertises any latency target.

Avoid multi-node tensor parallelism when the selected model already fits comfortably on one GPU. Replica scaling is simpler for independent short video windows.

## 4. Public API contract

A01 freezes machine-readable schemas and fixtures within the first 45 minutes. Every other agent imports them.

### POST /api/v1/analyze

Request:

- Header: X-API-Key: valid team-issued key.
- Content-Type: multipart/form-data.
- Field: file.
- Input: a .mov upload, approximately 29 MB in the brief.
- Recommended response: HTTP 202 after the owned upload file and job record are durable.

~~~json
{
  "job_id": "3f2a9c1e"
}
~~~

HTTP 202 is a design choice; the brief requires the job ID, not a specific success status code.

Do not run a full ffprobe or model call before acknowledging a valid accepted upload. Header-level authentication, body limits, multipart handling, durable file storage, and job creation belong on the acceptance path. Detailed decoding belongs in the worker.

Missing keys and incorrect keys are rejected. Authenticate before expensive multipart processing where the chosen ASGI stack permits it.

Use a configurable upload limit of at least 256 MiB initially, as a team setting rather than an organizer limit. Do not introduce a restrictive 29 MB cutoff that rejects valid overhead or slightly larger examples.

### GET /api/v1/jobs/{job_id}

Require X-API-Key.

Processing response:

~~~json
{
  "job_id": "3f2a9c1e",
  "status": "processing"
}
~~~

The public processing status includes internal queued, claimed, decoding, screening, refining, and assembling stages.

Completed positive response:

~~~json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Human-readable event type from the reviewed taxonomy",
      "timecode_sec": 44.2,
      "description": "Visible observation; why it may meet the verified rule; document title and clause when available.",
      "evidence_frames": [
        {
          "timestamp_sec": 44.2,
          "image_base64": "REPLACE_WITH_VALID_RAW_JPEG_BASE64"
        }
      ]
    }
  ]
}
~~~

This example contains placeholders, not a completed event or a real JPEG fixture. A01 must create valid synthetic contract fixtures without copying private video content.

Completed negative response:

~~~json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "no_violation",
  "violations": []
}
~~~

Minimal failed response:

~~~json
{
  "job_id": "3f2a9c1e",
  "status": "failed"
}
~~~

The brief does not define a full error schema. Keep diagnostics in internal/UI records unless A01 confirms compatible public error fields. Unknown job IDs return HTTP 404; unauthorized requests return the documented authentication error.

### Public invariants

1. Status is processing, done, or failed.
2. Verdict is present for done results and is violation or no_violation.
3. Violation implies at least one event; no_violation implies an empty event list.
4. Each event has 1-3 decodable JPEGs, not URLs or data-URI prefixes.
5. Event timestamps and evidence timestamps lie within the original video's timeline.
6. The image actually corresponds to its declared timestamp.
7. One clear supported event makes the whole-video verdict positive even if another interval is unclear.
8. Decode errors, timeouts, or missing evidence do not become clean verdicts.
9. Descriptions distinguish visible observations from possible rule implications.
10. Rule references appear in the required description and in richer UI metadata; do not rely on extra jury fields to satisfy grounding.

### Fast completed-result serving

Assemble and validate the jury JSON once, write it atomically, then mark the job done. Repeated GET requests serve the immutable authorized result without model calls or repeated JPEG encoding.

A completed response can be large. Measure time to first byte and full response transfer separately.

## 5. Internal contracts and repository ownership

All repository paths below are proposed implementation paths, not claims that files already exist.

### Contract objects

| Object | Required fields and meaning |
|---|---|
| VideoAsset | job_id, owned_path, content_hash, duration_sec, stream metadata, source timeline origin |
| FrameRef | frame_id, pts_sec normalized to video start, local media reference, dimensions, rotation handling |
| ClipRequest | job_id, clip_id, frame_refs, fact definitions, optional supported rule definitions, analysis version |
| FactDecision | fact_id, state: present/absent/unclear, supporting_frame_ids, optional model score with score provenance |
| CandidateEvent | candidate_id, event_type_id, candidate interval, supporting facts, visibility notes |
| VerifiedEvent | event_type_id, rule_id, source interval, anchor frame ID, evidence IDs, observed facts, limitations |
| RuleRecord | rule_id, source details, applicability, required facts, temporal conditions, exceptions, verification state |
| JobRecord | public status, internal stage, accepted time, deadline, lease owner, lease expiry, attempt/fencing version, paths |
| ModelManifest | model identifier and revision, weight checksum, processor version, adapter checksum, license, serving config |
| EvaluationRecord | sample/group ID, split, predictions, matched errors, timings, model/config/rule versions |

Contracts must define units, timestamp origin, optionality, and validation. Use frame IDs to bind model outputs to existing decoded media. Unsupported frame IDs are invalid outputs, not approximate evidence.

### Proposed layout and write ownership

~~~text
contracts/                         A01
src/forte/shared/schemas.py         A01

data/manifests/                     A02
data/splits/                        A02
scripts/audit_data.py               A02

annotations/                       A03
docs/annotation-guide.md            A03
docs/event-taxonomy.md              A03

rules/                             A04
docs/legal-source-register.md       A04

src/forte/video/decode.py           A05
src/forte/video/probe.py            A05

src/forte/video/sampling.py         A06
src/forte/video/crops.py            A06

src/forte/model/                    A07
models/manifest.json                A07

src/forte/decision/                 A08
configs/decision/                  A08

training/                          A09

evaluation/                        A10
benchmarks/quality/                 A10
benchmarks/robustness/              A10
qa/vision/                         A10

src/forte/events/                   A11
src/forte/evidence/                 A12

src/forte/api/                      A13
src/forte/jobs/                     A14

serving/                           A15
configs/serving/                   A15

web/                               A16

deploy/                            A17
infra/                             A17
docs/gpu-provisioning.md            A17

src/forte/observability/            A18
benchmarks/performance/             A18
qa/system/                         A18

brand/                             A19
docs/forte-brand-research.md         A19
docs/forte-ui-guidelines.md          A19
docs/visual-qa.md                   A19

qa/integration/                    A20
docs/integration/                  A20
docs/submission/                   A20
README.md                          A20
pyproject.toml and lockfile         A20
~~~

A20 owns the root dependency manifest. Other agents request dependency additions with exact package/version justification. A17 owns deployment files, not the root dependency lock. No two agents edit the same file without an explicit ownership handoff.

Raw videos, labels, and model weights are read-only inputs. Do not rename supplied files, overwrite source data, commit large media, or embed training labels in production configuration.

### Private data boundary for cloud-backed coding agents

Coding agents may implement and invoke local processing code, but their tool output must not expose supplied video frames, per-video captions, observed-action descriptions, raw model predictions, or private annotation records to the cloud model. Do not use a cloud image viewer or return those values in terminal output.

Keep media, per-video annotations, local predictions, and detailed review screens inside the team's private environment. Coding-agent-facing diagnostics should use aggregate counts, timings, schema errors, and opaque sample IDs. Generic task definitions and normative criteria can come from public sources; they must not contain a narrative of a supplied clip.

A03, A10, A12, and A20 therefore need a human reviewing the private local interface, or a locally hosted model operating inside the service, for content-dependent work. Reserve a human reviewer for approximately the first 2-3 hours and the final evidence checks. If that reviewer is unavailable, record the annotation/evidence checks as provisional and do not present local-teacher outputs as independently verified ground truth.

## 6. Latency objectives and measurement

**Every number in this section is a proposed target, not a benchmark result or organizer promise.**

Fast acknowledgment does not mean fast analysis. Report both.

### Timing definitions

- T0: reverse proxy receives the first request-body byte.
- T1: API receives the last body byte.
- T2: owned file and job record are durable; accepted response can be returned.
- T3: worker begins analysis.
- T4: validated complete result is published.
- T5: client receives the final response byte.

Capture timestamps with a monotonic clock within one host; record clock synchronization when comparing services.

### Recommended objectives

| Measurement | Initial target | Measurement conditions |
|---|---:|---|
| POST acceptance overhead, T2-T1 | p95 <= 150 ms | 29 MB upload, warm API, NVMe, excluding network upload time |
| GET processing response | p95 <= 25 ms | Server-side duration, representative concurrent jobs |
| GET done time to first byte | p95 <= 50 ms | Authorized immutable result, warm local storage |
| Typical short uncached video | 10-30 seconds stretch | Actual quality-preserving configuration; report duration/resolution |
| Full 145-second uncached video | p95 <= 90 seconds target | Warm selected model, no whole-result cache, declared hardware |
| Accelerated full-video profile | <= 45 seconds stretch | Warm replicas, explicit GPU allocation, novel content, and unchanged quality; single-job and concurrent results reported separately |
| Complete processing safety objective | <= 240 seconds | Include queuing; also measure upload-start-to-result |
| Organizer cutoff | 300 seconds | Timing origin is unclear; retain conservative margin |
| Capacity check | 5 simultaneous distinct uploads | Team test assumption, not a published jury concurrency requirement |

At 100 Mbps, transmitting 29 MB alone takes about 2.32 seconds before protocol overhead. Do not advertise a sub-150 ms complete upload based on post-upload acceptance latency.

A small dataset cannot establish a statistically reliable p99. For each benchmark, report sample count, p50, p95 where meaningful, maximum, and every timeout. Distinguish observed maxima from population guarantees.

### Per-job compute budget for the 90-second target

| Stage | Planning budget |
|---|---:|
| Queue and dispatch | 5 s |
| Probe, decode, frame preparation | 10 s |
| Coarse coverage analysis | 40 s |
| Candidate refinement and additional context | 25 s |
| Rule application, evidence, result publication | 5 s |
| Margin | 5 s |
| Total | 90 s |

These budgets are provisional. A18 measures the bottleneck; A15 adjusts capacity or scheduling; A10/A20 veto quality-breaking shortcuts.

### Required benchmark modes

1. Warm model, previously unseen video content: principal analysis benchmark.
2. Warm API, repeat GET for the same completed job: polling benchmark.
3. Whole-result cache hit: report separately and never use to claim inference speed.
4. Five distinct uploads starting together: capacity benchmark.
5. Cold service startup: readiness measurement, not mixed into warm inference.
6. Egress-disabled execution with local weights.
7. Maximum-duration development clips, multiple events, and difficult normal actions.

A cache must not look up organizer labels. Filename changes or content-identical copies must not reveal ground truth to the production service.

## 7. Data, annotations, and evaluation

### Observed local inputs

The supplied files are a seven-page case PDF, a label workbook, and 42 .mov files.

The workbook has 22 positive and 20 negative entries. Hashing revealed two byte-identical pairs:

- Video IDs Х-4 and Х-47; positive labels refer to 57 and 68 seconds.
- Video IDs Х-5 and Х-48; both negative.

There are 40 unique video contents. These are metadata and label observations; video contents have not been visually classified as part of this planning document.

The filenames use both Latin X and Cyrillic Х. Preserve the original spelling and a content-hash identity. Unicode normalization alone does not equate those letters. Resolve joins explicitly and audit collisions.

### Annotation protocol

A human on the team reviews the media locally. A coding subagent backed by a cloud model must not receive provided frames or descriptions as an annotation shortcut.

For each event record:

- Recording ID and scenario group.
- Start, end, anchor frame, and timestamp.
- Event type and natural-language definition.
- Observable facts and the frames supporting them.
- Visibility limitations and relevant occlusion.
- Applicable rule ID, or an explicit unresolved grounding state.
- Whether the procedure was fully visible.
- Reviewer and review status.

For every supposed negative interval, confirm that no relevant event is visible. A single provided positive timestamp does not prove all remaining frames are negative.

### Split policy

1. A02 groups exact duplicates and related scenarios before window extraction.
2. A03 labels event types and scenario metadata.
3. A10 proposes a stratified group split, aiming at roughly 28 train, 6 validation, 6 final-check unique clips.
4. Group integrity takes priority over exact counts. Document rare types and absent coverage.
5. A20 locks split IDs and hashes before A09 starts training.
6. All windows, crops, and augmented copies inherit their recording's split.
7. The final-check set is used once after checkpoint selection. If errors drive further changes, disclose that it has become development data.

If available, collect allowed new footage at a genuinely different angle and reserve it for the last robustness check. Do not describe image rotation as equivalent to a new viewpoint.

### Training example construction

Create positive windows that actually include reviewed evidence, verified normal windows, confusing normal actions, and unclear views. Preserve before/after context. Balance by recording and scenario, not by generating thousands of correlated windows from one positive clip.

Mild appearance changes may be useful if they preserve the action. Horizontal flips, time reversal, arbitrary crops, and altered order can invalidate rules; do not apply them indiscriminately.

A local licensed teacher may suggest annotations inside the team's service. Human review is still required. Teacher agreement is not ground truth.

### Quality evaluation

Record whole-video confusion counts, precision, recall, and F1 as internal diagnostics. The official scoring formula is not specified.

Also record:

- Per-type event TP, FP, and FN.
- Unsupported or wrong legal mappings.
- Absolute anchor timestamp errors where the annotation is reliable.
- Whether evidence visibly supports the stated action.
- Duplicate-event overcounting.
- Misses of short actions and omissions.
- New-angle and held-out-scenario results.
- Completion rate and latency.

A10 may use internal localization tolerances such as 0.5/1/2 seconds for analysis, clearly labeled as team diagnostics. They are not organizer tolerances.

Do not convert these metrics into an invented official score. Show a rubric coverage table and actual errors.

## 8. Rules, evidence, and uncertainty

### Normative grounding

A04 starts with official Kazakhstan sources, including NB RK/Adilet publications relevant to cash operations, cash collection, and cash-room procedures.

One research starting point is NB RK Resolution No. 231 of 29 November 2019, concerning cash operations and cash collection, at [Adilet](https://adilet.zan.kz/rus/docs/V1900019680). The current text, exact clauses, legal status, and applicability have not been verified by this plan; A04 must verify them before enabling a corresponding rule.

Do not assume that phone use, a particular gesture, casual clothing, or the presence of another person automatically violates an applicable law. Read and validate the actual requirement.

A rule record needs:

~~~yaml
rule_id: stable_internal_id
title: reviewed_event_name
jurisdiction: Kazakhstan
source:
  document_title: verified_title
  issuing_body: verified_body
  clause: verified_clause_or_explicitly_unspecified
  url: official_source_url
  accessed_on: actual_date
  archived_copy_hash: recorded_hash
verification_state: verified_or_unresolved
applicability:
  procedure: described_procedure
  preconditions: reviewed_conditions
required_visual_facts: []
temporal_conditions: []
exceptions: []
visibility_requirements: []
explanation_template: reviewed_template
~~~

Only verified applicable rules may produce a grounded event in the final release. Unresolved mappings are coverage gaps requiring attention, not permission to fabricate a source.

### Fact-to-event logic

Use tri-state visual facts: present, absent, unclear. Unknown or occluded evidence must not silently become absent.

An event must establish the rule's applicability and required observations. For sequence rules, retain state across overlapping windows. For omissions, establish that the relevant process completed and that the allegedly omitted action was observable throughout the necessary interval.

When a procedure needs more than eight seconds of context, request a longer-context verification pass using the same checkpoint. Do not force a long procedure into a short-window label.

### Evidence policy

- Select actual decoded frame IDs, then retrieve the source image at its recorded presentation timestamp.
- Preserve original evidence; optional overlays must not replace the original frame.
- Use 1-3 JPEGs, preferably enough to show the action's temporal context.
- For an omission, show the visible procedure boundary and supporting context; document the anchor convention.
- Use compression that preserves small relevant details.
- Do not create, enhance by generation, or hallucinate evidence.
- Do not infer identity, audio content, system transactions, intent, or unseen actions.

### Uncertainty and failure

- One verified event means violation.
- Successful analysis with no supported event means no_violation, interpreted as no detected signs.
- Low visibility stays visible in the internal/UI explanation.
- Technical failure means failed.
- Coverage loss caused by timeout or decoder failure must not be disguised as a complete normal analysis.
- Rechecking with the same model can provide more visual context; it is not independent proof of correctness.
- Never claim self-reported model confidence is calibrated. Retain score provenance and evaluate any calibration separately.

## 9. Model and training plan

### Baseline first

A07 and A08 produce an untuned end-to-end baseline by hour 6. Store predictions and configuration hashes. The model sees video content and task definitions, not filenames, workbook labels, split names, or answers.

Start with eight-second windows and four-second stride, around two model-visible frames per second. Always include the tail of the video. A 145-second video needs roughly 36 such coverage windows depending on tail policy.

A06 can compute cheap higher-rate motion cues on small grayscale frames to trigger extra inspection. Motion cannot be the only coverage strategy: some important events involve little movement.

Keep full-scene context plus selected work-area crops. Avoid hard-coded camera coordinates that collapse under the final angle change. Set frame/resolution budgets through the actual processor and measured visual-token counts.

### Output contract

Return a compact schema containing facts, candidate event IDs, supporting frame IDs, and visibility limits. Use natural-language fact/rule definitions so that rules absent from training can still be queried.

Constrained JSON controls format. It does not prevent a wrong observation. vLLM supports JSON-schema constraints and finite choices. Use the supported syntax for the pinned version rather than deprecated guided_* arguments. [S5]

The output budget should be measured and kept small, initially around 256 generated tokens per modest fact bundle, adjusted when truncation appears. Do not force truncation into a valid-looking empty result.

### Training experiment budget

A09 trains only after splits, annotations, schemas, and the baseline evaluator are usable.

| Experiment | Proposed starting settings | Purpose |
|---|---|---|
| E0 | Untuned 8B baseline | Mandatory control |
| E1 | LoRA rank 8, alpha 16, dropout 0.05, adapter LR 1e-5, one epoch, BF16 where supported | Test whether adaptation helps the reviewed task |
| E2 | One focused change based on E1 errors; e.g. rank 16/alpha 32 or improved hard negatives | Investigate one observed failure cause |
| E3 | 4B candidate only if serving speed is still inadequate | Capacity alternative, not a required third training run |

These are experimental hypotheses, not official defaults or proven settings. A09 checks trainable parameter names and performs a tiny overfit/sanity exercise on training-only examples before a real run. Stop if gradients, labels, video formatting, or output masking are wrong.

The official Qwen repository documents video conversation datasets and LoRA options. Freeze the vision encoder initially and verify actual parameter selection instead of assuming flags do what their names imply. [S4]

Preserve general semantics by using natural-language tasks and avoiding repetitive filename-specific training. Do not train away unclear outcomes.

### Selection gate at hour 11

A10 compares E0 and E1/E2 on locked validation videos. A20 chooses a checkpoint using:

- Missed events and whole-video errors.
- False alarms and confusing normal actions.
- Correct type and grounding.
- Timestamp/evidence quality.
- Robustness.
- Measured runtime.

Training loss alone is not a selection criterion. A one-example improvement on six clips has substantial uncertainty; report the counts.

If no credible improvement exists, release E0. Keep the adapter and results as an experiment, not the production dependency.

### Optional System One fast path

After the baseline works, A08/A15 may test fixed-choice scoring for individual facts, using actual candidate-label logits where the backend supports it. Verify tokenization, score normalization, batching, latency, and factual errors.

Such scores are conditional model scores, not guaranteed calibrated probabilities. A custom temporal classifier or parallel head is a stretch item only after all release gates pass; it must not consume the hours reserved for deployment and validation.

## 10. Serving and performance engineering

### Public API path

FastAPI's UploadFile uses a spooled file and exposes file operations without requiring the complete upload as an application-level bytes object. Prefer it to reading an entire video into memory. [S6]

Implementation order:

1. Authenticate cheaply before expensive body handling.
2. Accept multipart input with a documented body limit.
3. Transfer the spooled upload into an owned durable location with bounded memory.
4. Record the job in the same durable system used by workers.
5. Return the job ID.
6. Let independent workers do all expensive analysis.

Do not promise zero-copy upload processing: the framework and reverse proxy may spool or copy data. Profile actual disk writes before optimizing them.

Use a separate worker service for heavy compute rather than FastAPI BackgroundTasks as the durable inference queue. FastAPI's documentation explicitly distinguishes heavier background computation from small in-process tasks. [S7]

### Durable queue design

Use PostgreSQL job records with transactional claims, lease expiry, and attempt fencing. SKIP LOCKED can support queue-like consumers; PostgreSQL documents its limitations for general-consistency reads. [S9]

- Do not claim a job until its uploaded file is complete.
- Keep the claim transaction short; never hold a database transaction during GPU inference.
- Heartbeat leases.
- Renew or recover expired work without accepting stale result publications.
- Write result files under attempt-specific temporary names.
- Validate and atomically publish only when the attempt still owns the lease/fencing version.
- Expose queued work as processing to the jury.
- Limit recovery retries by the absolute deadline.
- Do not acknowledge accepted jobs if persistence failed.

A14 owns the database state machine; A13 does not implement a separate in-memory queue.

### GPU path

- Pin a tested model/processor/serving combination.
- Keep model weights resident.
- Prewarm representative frame and token shapes before readiness.
- Separate training GPUs from serving.
- Replicate one selected checkpoint across GPUs.
- Start with modest microbatches, such as two clips, then tune against actual visual-token memory and latency.
- Schedule fairly across jobs; one video's many windows must not starve other uploads.
- Preserve per-job deadlines and cancellation.
- Bound CPU decoder and image-preparation concurrency to avoid oversubscription.
- Cache decoded media and processor results only through verified supported mechanisms.
- Measure prompt/prefill time as well as generation; short JSON does not eliminate visual encoding cost.
- Keep model endpoints private. Internal media references must be job-owned and restricted to configured roots.
- Use public API workers for CPU I/O, not one GPU model per process.
- Keep serving configuration stable during the final benchmark.

vLLM exposes request queue, prefill, decode, end-to-end, cache, and utilization metrics; metric names vary by version, so A18 verifies the pinned instance. [S8]

### Evidence and payload speed

Encode evidence once. Initially target JPEGs around 80-150 KiB where the relevant detail survives; this is a tuning range, not a quality guarantee. If text, bills, or small objects become unreadable, retain more pixels or less compression.

Base64 increases payload size by approximately one third. Include actual response bytes in transfer benchmarks. A quick status lookup does not establish a quick multi-megabyte completed response.

### Cache policy

- Per-job immutable result JSON is required.
- Whole-content result caching is optional and scoped by caller, model, adapter, processor, rule catalog, and analysis config.
- New job IDs must receive correct envelopes even when reusable result content exists.
- Never key a result solely by filename.
- Never use labels or precomputed organizer answers as a cache.
- Benchmark novel content with whole-result caching disabled.
- Do not assume that multimodal prefix caching or shared preprocessing is supported without checking the pinned stack.

### Performance changes in order

1. Eliminate GPU work from HTTP handlers.
2. Keep one warm model per intended replica.
3. Remove repeated decoding and repeated evidence encoding.
4. Bound and batch window work.
5. Add replicas and fair dispatch.
6. Profile resolution, frames, and contextual coverage.
7. Test smaller weights or quantization with quality gates.
8. Consider specialized heads only if substantial time remains.

Do not achieve speed by skipping every difficult interval, dropping additional events, replacing failures with normal verdicts, or reporting cache hits as uncached performance.

## 11. Twenty-hour schedule and dependencies

The 20 agents are not 20 independent applications. They work against common schemas, owned files, and a single integration branch.

### Timeline

| Time | Required output | Active focus | Hard decision |
|---|---|---|---|
| T+0:00-0:45 | Schemas, ownership, skeleton, GPU inventory/quote, brand-source search | A01, A02, A13, A14, A17, A19, A20 | Freeze API v1; identify GPU access/budget gaps |
| T+0:45-2:00 | Working GPU/model access, decoder, fixtures, legal research, annotation workflow, initial brand findings | A03-A08, A15-A19 | Use existing capacity while any purchase/login handoff is pending |
| T+2:00-3:00 | Reviewed initial events/rules, locked splits, complete job round trip | A02-A04, A10-A14, A20 | Training cannot start without split integrity |
| T+3:00-6:00 | Real untuned baseline, evidence, quality/latency records, initial branded UI | A05-A08, A10-A12, A15, A16, A18, A19 | Baseline must exist by hour 6 |
| T+6:00-9:00 | E1 adapter plus evaluation; product and performance continue | A09, A10, A13-A19 | Cancel training expansion if integration is unstable |
| T+9:00-11:00 | E2 if justified; checkpoint comparison | A08-A10, A15, A20 | Freeze model/checkpoint by hour 11 |
| T+11:00-14:00 | Correct event aggregation, rule references, source evidence, brand/accessibility review | A04, A11-A16, A19, A20 | Freeze public schema and brand tokens |
| T+14:00-16:00 | Public deployment, offline run, recovery, concurrency | A13-A19 | Stop adding features |
| T+16:00-18:00 | Final-check set, genuinely different angle when available, five-job burst, synthetic visual QA | A10, A18, A19, A20 | Only justified fixes; disclose reused test data |
| T+18:00-19:00 | Release bundle, API key handoff, docs, pitch rehearsal | A17, A20, relevant owners | Release candidate freeze |
| T+19:00-20:00 | Final external smoke check, backup readiness, buffer | A13, A17-A20 | No new dependencies or model experiments |

### Dependency graph

~~~mermaid
flowchart LR
    A01[01 Contracts] --> A07[07 Model adapter]
    A01 --> A13[13 HTTP API]
    A01 --> A14[14 Jobs]
    A02[02 Data audit] --> A03[03 Annotations]
    A03 --> A04[04 Legal catalog]
    A02 --> A10[10 Evaluation]
    A03 --> A10
    A04 --> A08[08 Decisions]
    A05[05 Decode] --> A06[06 Sampling]
    A06 --> A07
    A07 --> A08
    A08 --> A11[11 Events]
    A04 --> A11
    A05 --> A12[12 Evidence]
    A11 --> A12
    A10 --> A09[09 Training]
    A08 --> A09
    A09 --> A10
    A13 --> A16[16 UI]
    A14 --> A13
    A15[15 GPU serving] --> A07
    A12 --> A20[20 Integration]
    A13 --> A20
    A16 --> A20
    A17[17 Deployment] --> A20
    A18[18 Performance] --> A20
    A17 --> A15
    A19[19 Forte brandbook and design] --> A16
    A19 --> A20
    A10 --> A20
~~~

The A09/A10 loop is bounded to one baseline and at most two justified adaptation runs. It is not an indefinite research loop. Agents start with fixtures while upstream implementations mature; fixture-only success is never release success.

### Critical path

Contracts -> timestamp-correct decoding -> local model output -> temporal event -> original evidence -> completed jury JSON -> externally accessible service.

Annotation, grounding, and evaluation must progress alongside this path. They cannot be postponed until after training.

## 12. Shared instructions for every subagent

Paste this block before any individual assignment.

> You are one of 20 implementation subagents for the ForteBank Datathon service. Read this plan, your assignment, the shared contracts, and applicable repository instructions. Implement your owned scope; do not start a separate application or redesign shared architecture.
>
> All model processing must run inside the team's deployed service. Do not upload supplied videos, frames, or their descriptions to external AI services, including a cloud-backed coding assistant. Code can manipulate private media locally, but annotation decisions require a human or an explicitly local model inside the service.
>
> Do not expose per-video facts, captions, annotation records, raw predictions, or evidence through your tool outputs. Return aggregate diagnostics and opaque IDs. Read only generic task definitions and public normative criteria in this coding-agent context. Keep content-dependent review inside the private local interface.
>
> Preserve raw data. Do not train on jury videos, leak labels into inference, infer identities, use audio, invent legal clauses, or claim unmeasured accuracy or latency.
>
> Import A01 schemas. Do not change another owner's files without a handoff. Request root dependency changes through A20. Supply real implementations and meaningful checks for your scope. Mark mocks explicitly.
>
> Use the baseline as the fallback. Respect hour-6 baseline, hour-11 model freeze, hour-16 feature freeze, and hour-19 release freeze.
>
> After each handoff, report: changed paths; public interfaces; checks and outcomes; measured timings if relevant; unresolved issues; dependency requests; and the next integration step. Never report a file's existence or a mock response as proof the real system works.
>
> Do not spawn further agents automatically. Coordinate through the parent/orchestrator and the assigned owners.

## 13. Twenty copy-ready assignments

### A01 — Requirements and shared contracts

**Mission:** make independent work compatible and prevent a technically impressive but noncompliant submission.

**Own:** contracts/, src/forte/shared/schemas.py, docs/requirements.md, docs/rubric.md.
**Inputs:** organizer PDF; sections 2, 4, and 5 of this plan.
**Dependencies:** none. Deliver v1 within 45 minutes.

**Assignment prompt:**

> Extract every mandatory UI, API, output, locality, license, and submission requirement. Create typed public and internal schemas with timestamp units, enum values, and invariants. Publish valid synthetic positive, negative, processing, and failed fixtures. Preserve the exact two jury routes and required field names. Include the seven scoring weights and a traceability checklist. Resolve uncertainty wording conservatively without adding a third verdict. Describe exact-clause and ready-made-solution ambiguities without inventing organizer policy. Give all agents importable contracts and a version identifier.

**Acceptance:** fixtures parse; real JPEG fixture bytes decode; illegal verdict/status values fail validation; positive/negative event-list consistency is checked; timestamp constraints have a documented validation boundary.

**Handoff:** A05-A16 and A20 receive schema names, fixture paths, and change rules.

### A02 — Dataset audit, identity, and locked split inputs

**Mission:** prevent data leakage, silent file joins, and misleading evaluation.

**Own:** data/manifests/, data/splits/, scripts/audit_data.py.
**Inputs:** supplied workbook and videos as read-only local inputs.
**Dependencies:** A01 manifest schema; collaborate with A03 and A10.

**Assignment prompt:**

> Audit all filenames, content hashes, workbook rows, missing links, duration/codec/resolution metadata, and duplicate groups. Preserve Latin/Cyrillic filename distinctions and use stable content identity. Flag the duplicated positive pair with 57/68-second annotations for local review. Do not inspect or send video content to a cloud model. Create recording/scenario group inputs, deterministic split-generation support, and split-integrity checks. Ensure all derived windows inherit the original group. Exclude labels and annotation files from the production inference image and runtime mounts.

**Acceptance:** 42 supplied files and current label rows reconcile or discrepancies are explained; exact duplicates cannot cross splits; no path or Unicode join silently drops a video; source files remain unchanged.

**Handoff:** A03 gets review IDs; A10 gets groups; A09 gets only approved training manifests; A17 gets data-exclusion rules.

### A03 — Annotation protocol and event taxonomy

**Mission:** convert weak binary/timestamp labels into reviewed, trainable event evidence.

**Own:** annotations/, docs/annotation-guide.md, docs/event-taxonomy.md.
**Inputs:** A02 manifest and a human reviewer/local model workflow.
**Dependencies:** A01 schemas; A02 identities; A04 legal feedback.

**Assignment prompt:**

> Build a lightweight local annotation workflow and clear definitions for event types, facts, intervals, evidence frames, visibility, and omissions. Do not view private media through a cloud model. Support human review and any teacher model only inside the local service. Review all unique clips, every possible event, and normal lookalikes. Do not assume unmarked positive-video intervals are negative. Keep taxonomy descriptions understandable and consistent in Russian. Mark grounding gaps and uncertain observations. Correct ambiguous timestamps only after review, preserving original labels and an audit record.

**Acceptance:** every unique clip has a review state; known positives have reviewed intervals/facts or an explicit unresolved item; negative training windows are verified; unknown visibility is representable; taxonomy IDs are stable.

**Handoff:** A04 receives factual event descriptions locally; A08 receives definitions; A09/A10 receive split-safe reviewed annotations.

The factual event handoff remains inside the private processing environment. Cloud-backed agents receive public-source criteria, generic schemas, and aggregate review status, not per-video descriptions.

### A04 — Kazakhstan regulatory grounding

**Mission:** earn the rule-justification points with real, applicable sources.

**Own:** rules/, docs/legal-source-register.md.
**Inputs:** public official RK sources and the locally reviewed taxonomy.
**Dependencies:** A03 facts; A01 RuleRecord schema.

**Assignment prompt:**

> Research applicable cash-operation and cash-collection requirements using official Kazakhstan sources. Verify current document status, exact clauses where available, applicability conditions, exceptions, and what can be observed visually. Archive public source material and record URLs/access dates/checksums. Build a rule catalog mapping required visual facts and temporal conditions to supported event types. Do not import foreign regulations, invent a clause, assume another bank's policy binds ForteBank, or infer a violation from a gesture without a valid requirement. Mark unresolved mappings explicitly. Prepare short reviewed explanation templates.

Read public-source and generic taxonomy definitions in the coding context. Per-video legal matching is performed by local code with private annotations and local review; do not print clip descriptions into agent output.

**Acceptance:** every enabled rule has a traceable applicable source and necessary visible conditions; every event type has coverage or an explicit gap; source absence cannot be concealed by model-generated text.

**Handoff:** A08/A11 consume the catalog; A12/A16 use reviewed descriptions; A20 receives coverage limitations.

### A05 — Timestamp-correct video decoding

**Mission:** make every later timestamp and evidence reference trustworthy.

**Own:** src/forte/video/decode.py, src/forte/video/probe.py.
**Inputs:** owned upload paths and A01 objects.
**Dependencies:** A01 schemas; coordinate with A06/A12.

**Assignment prompt:**

> Implement bounded local probing and decoding for the provided MOV format. Handle rotation metadata, presentation timestamps, variable frame rate, timeline origin, short clips, and decoder failures. Normalize seconds to video start and expose stable FrameRef identities. Never use frame_index/nominal_fps as the only timestamp source. Ignore audio. Limit process/thread concurrency and propagate cancellation/deadlines. Allow sparse sampling and exact evidence retrieval without repeatedly decoding the entire file.

**Acceptance:** timestamp mapping is checked on known-timestamp synthetic clips and supplied-file metadata; rotated/VFR fixtures work; invalid videos fail honestly; frame IDs resolve to real source frames.

**Handoff:** A06 receives frame/timestamp access; A12 receives exact-source retrieval; A18 receives decode timings.

### A06 — Coverage sampling, crops, and motion cues

**Mission:** reduce visual compute while preserving action coverage and viewpoint robustness.

**Own:** src/forte/video/sampling.py, src/forte/video/crops.py.
**Inputs:** A05 frames and A03 task definitions.
**Dependencies:** A01, A05; coordinate with A07/A10.

**Assignment prompt:**

> Implement overlapping coverage windows, explicit final-tail coverage, processor-aware frame budgets, and optional motion-triggered extra inspection. Begin around 8-second windows, 4-second stride, and 2 model-visible fps, then benchmark. Preserve global scene context and adaptive work-area detail; do not rely solely on motion or fixed camera pixel coordinates. Support longer context for procedural rules. Record which intervals and frames were inspected. Never silently drop the difficult end of a video because a budget was exhausted.

**Acceptance:** no timeline gaps under the documented coverage policy; boundary and short-event fixtures are included; crops retain relevant context; budget changes are paired with quality results.

**Handoff:** A07 gets ClipRequests; A11 gets coverage metadata; A10 gets sampling stress cases.

### A07 — Local model adapter and baseline

**Mission:** expose one reliable local model interface before attempting training.

**Own:** src/forte/model/, models/manifest.json.
**Inputs:** Qwen3-VL-8B-Instruct or validated fallback; A01 ClipRequest; A15 private service.
**Dependencies:** A01, A05/A06, A15; initial synthetic runner can start immediately.

**Assignment prompt:**

> Implement the local inference adapter, preprocessing integration, structured response parsing, cancellation, bounded retries, and model manifest. Pin actual model and processor revisions. Use job-owned local media or a verified private transport; reject arbitrary filesystem/URL access. Keep filenames and ground-truth metadata out of prompts. Record processor output sizes, visual tokens when available, generated tokens, and inference timings. Deliver a real untuned baseline by hour 6, with the same checkpoint usable for coarse and refined views.

**Acceptance:** a real local clip produces valid decisions; missing weights and invalid output fail clearly; egress is unnecessary; no mock success remains on the production path; manifest identifies exact weights and license.

**Handoff:** A08 gets the provider; A09 gets a reproducible baseline; A18 gets timings; A20 gets a startup command.

### A08 — Structured decisions and System One interface

**Mission:** turn model capability into small, controlled visual judgments.

**Own:** src/forte/decision/, configs/decision/.
**Inputs:** A03 definitions, A04 catalog, A07 provider.
**Dependencies:** A01, A03, A04, A07.

**Assignment prompt:**

> Design short question bundles and constrained outputs for present/absent/unclear facts, candidate events, and supporting frame IDs. Separate observation from legal conclusion. Use natural-language criteria so untrained supported rules remain queryable. Validate that every frame ID exists in the request and every event ID exists in the reviewed taxonomy. Treat invalid or truncated responses as errors/uncertainty, not normal. Record prompts/config versions. Test a fixed-choice logit path only after the JSON baseline works and only if the backend exposes real scores correctly.

**Acceptance:** unclear is preserved; unsupported IDs and fabricated evidence references are rejected; the same factual test is repeatable under a fixed config; no unmeasured calibration or Jev-equivalence claim appears.

**Handoff:** A11 gets decisions; A09 gets training targets; A10 gets prompt/config versions.

### A09 — LoRA adaptation

**Mission:** improve the task without sacrificing the working baseline.

**Own:** training/.
**Inputs:** approved train/validation manifests, reviewed targets, baseline runner.
**Dependencies:** A02/A03/A08/A10 and locked splits; no training on final-check or jury data.

**Assignment prompt:**

> Build the smallest reproducible LoRA training pipeline using the selected open-weight model. Verify tokenizer/processor alignment, label masking, trainable parameters, gradients, and video formatting on training-only examples. Keep the vision encoder frozen initially. Run E1 and, only if justified by error analysis, one E2. Store config, seed, dataset hash, checkpoint checksum, and logs locally. Evaluate through A10, never choose by training loss alone. Do not generate thousands of correlated pseudo-examples and describe them as independent data. Stop new training work by hour 11 and hand off a loadable adapter or the recommendation to retain baseline.

**Acceptance:** the adapter loads in the serving path; split checks pass; training is reproducible; before/after predictions and errors exist; baseline remains selectable.

**Handoff:** A10 receives artifacts and reports; A15 receives the selected-load candidate; A20 receives the selection recommendation.

### A10 — Evaluation, vision robustness, and model selection

**Mission:** make honest selection decisions aligned with the rubric.

**Own:** evaluation/, benchmarks/quality/, benchmarks/robustness/, qa/vision/.
**Inputs:** A02 identities/groups, A03 annotations, A04 rules, all candidate predictions.
**Dependencies:** A01-A04; harness begins with fixtures before model output exists.

**Assignment prompt:**

> Create the evaluation harness for whole-video verdicts, per-event types, evidence support, timestamp errors, grounding mistakes, and completion rate. Lock group splits with A20 before training. Report TP/FP/FN/TN and denominators, per-type gaps, and actual failures. Label internal timestamp tolerances and metrics as diagnostics rather than official scoring. Compare baseline and adaptations on the same validation inputs/config conditions. Keep final-check data untouched until checkpoint selection. Produce a concise selection memo by hour 11 and a final results report with limitations.

**Acceptance:** no duplicate/scenario leakage; failed jobs are counted, not excluded; metric denominators are explicit; model selection is reproducible; small-sample uncertainty is acknowledged.

**Additional vision-QA work:** test new angles, brief actions, window boundaries, occlusion, confusing normal actions, omissions, multiple events, and unsupported facts. Check that onscreen instructions cannot override system rules. Keep media/predictions private and return aggregate diagnostics and opaque IDs.

**Handoff:** A09 gets actionable errors; A20 gets the checkpoint decision, missing robustness coverage, and vision-QA blockers.

### A11 — Temporal events and rule application

**Mission:** convert window-level facts into correctly separated grounded events.

**Own:** src/forte/events/.
**Inputs:** A08 decisions, A04 rules, A06 coverage metadata.
**Dependencies:** A01, A04, A06, A08.

**Assignment prompt:**

> Implement temporal state across overlapping clips, event grouping, duplicate suppression, omission conventions, and rule applicability. Preserve separate violations and do not stop after the first event. Request additional context or refinement when a rule's procedure exceeds the coarse window. Require visible support and verified grounding. Keep uncertainty explicit. Emit VerifiedEvents with actual anchor/evidence frame IDs, not generated seconds. Never force unknown model behavior into the nearest known violation category.

**Acceptance:** fixtures cover repeated overlap detections, two separate same-type events, multiple different types, normal actions, incomplete procedures, and one clear event plus an unclear interval.

**Handoff:** A12 receives complete events; A13/A16 receive versioned result semantics.

### A12 — Evidence extraction and explanation assembly

**Mission:** produce the concrete proof the jury and reviewer can inspect.

**Own:** src/forte/evidence/.
**Inputs:** A05 frame lookup, A11 events, A04 templates.
**Dependencies:** A01, A04, A05, A11.

**Assignment prompt:**

> Extract 1-3 original JPEGs per event at source timestamps, choose contextually meaningful before/during/after frames where useful, and verify readability of small relevant details. Generate descriptions from confirmed observations and verified rule templates. Encode evidence once, validate it, and assemble the immutable public result projection. Preserve original evidence separately from optional overlays. Reject mismatched frame/timestamp references. Measure encoding time and payload bytes.

**Acceptance:** all JPEGs decode; timestamp-to-image correspondence is verified; event support is visible in local review; no generated evidence or fabricated source enters the response; the jury payload satisfies A01.

**Handoff:** A14 gets the ready-to-publish result; A16 gets UI evidence metadata; A18 gets payload/encoding measurements.

### A13 — Fast public API and authentication

**Mission:** make upload and polling fast without hiding inference delay.

**Own:** src/forte/api/.
**Inputs:** A01 schemas, A14 job service, A12 final projection.
**Dependencies:** A01, A14; use declared fixtures until real worker integration.

**Assignment prompt:**

> Implement the two exact jury endpoints and X-API-Key handling, multipart file acceptance, bounded-memory durable upload, authorized result access, and honest HTTP errors. Keep inference, probing, and evidence work out of handlers. Authenticate early. Return a job ID only after file/job persistence succeeds. Serve completed immutable JSON without repeated encoding. Keep public queued work as processing. Provide private readiness/health endpoints and a separate UI adapter only if needed, using the same underlying job service. Do not ship the jury secret in browser JavaScript.

**Acceptance:** contract fixtures and real jobs work; wrong/missing keys fail; corrupt accepted media eventually fails honestly; GET never invokes a model; POST overhead and GET timings are measured with A18.

**Handoff:** A16 gets API examples; A17 gets routes/limits; A20 gets an external smoke procedure.

### A14 — Durable jobs, leases, and recovery

**Mission:** ensure every accepted job has an honest, recoverable lifecycle.

**Own:** src/forte/jobs/.
**Inputs:** PostgreSQL, A01 schemas, owned upload paths.
**Dependencies:** A01; coordinate with A13/A15/A17.

**Assignment prompt:**

> Implement the database schema, job repository, short transactional claims, lease heartbeats, attempt fencing, deadlines, and atomic result publication. Use one durable queue/state system. Never hold a transaction during model work. Recover worker crashes without stale attempts overwriting results. Bound retries by the job deadline. Separate internal stage from public status. Ensure accepted uploads are complete before a worker sees them. Retain diagnostics privately and preserve the required failed response.

**Acceptance:** restart and duplicate-claim checks pass; no acknowledged job disappears; stale workers cannot publish; file/database publication races are covered; cleanup does not delete active or jury-needed artifacts.

**Handoff:** A13 gets stable service functions; A15 gets claims/deadlines; A17 gets migrations and recovery instructions.

### A15 — GPU serving, batching, and scheduling

**Mission:** achieve fast uncached results under concurrent load.

**Own:** serving/, configs/serving/.
**Inputs:** selected model/adapter, local GPUs, A14 job deadlines.
**Dependencies:** A07 interface, A14 leases; serving baseline starts before training.

**Assignment prompt:**

> Deploy private warm model replicas with tested pinned settings. Start with one working instance, then scale independent replicas if profiling supports it. Keep training isolated. Add bounded microbatching, fair scheduling across jobs, cancellation, and deadline awareness. Prewarm representative shapes and publish real readiness. Measure visual-token memory rather than assuming a batch size fits. Avoid loading models in API worker processes. Use a smaller checkpoint, quantization, or tensor parallelism only after benchmarking quality and latency. Keep a working baseline serving configuration through the entire build.

**Acceptance:** multiple distinct jobs progress fairly; no cold download or first-use initialization occurs after readiness; OOM and failed requests propagate honestly; A18 provides uncached and five-upload measurements.

**Handoff:** A07 gets private endpoints and limits; A17 gets resource/port config; A20 gets validated capacity.

### A16 — Reviewer interface

**Mission:** make results clear and easy to verify.

**Own:** web/.
**Inputs:** A13 endpoints, A01 response examples, A12 evidence metadata, A19 brand tokens/assets/guidance.
**Dependencies:** A01/A13; begin with fixtures and neutral styling, then consume A19's output without blocking core behavior.

**Assignment prompt:**

> Build a concise Russian interface with video upload, real job status, final verdict, separate event cards, evidence images, timestamps that seek the source video, visible observations, rule references, and visibility limitations. Show failed processing honestly. Keep event wording consistent with the taxonomy. Use the same underlying result as the jury API. Preserve the jury key server-side through an appropriate UI adapter or separate limited credential; never hard-code it into the bundle. Avoid building account management, history, exports, or dashboards before core review works.

**Acceptance:** a person can upload an unseen clip and inspect each event without developer help; multiple events and normal/failed cases render correctly; status reflects actual backend progress; no mock result remains in production.

**Brand implementation:** A16 owns all frontend code; A19 owns research, tokens, permitted assets, and visual review. Consume source-backed or explicitly inferred brand choices. Package fonts/assets locally.

**Handoff:** A17 gets build/runtime requirements; A19 gets synthetic-fixture browser scenarios for visual QA; A20 gets a demo flow.

### A17 — PC operator, NVIDIA Brev GPU provisioning, and deployment

**Mission:** obtain usable GPU capacity through hands-on project-related PC/browser operation and deploy the service reproducibly.

**Own:** deploy/, infra/, docs/gpu-provisioning.md.
**Inputs:** user-provided server/account access; live inventory and quotes; spending policy; service entry points; dependency lock; model manifest.
**Dependencies:** inventory, tool-capability checks, and skeleton start immediately; coordinate capacity with A15 and PC-control access with A20.

**Assignment prompt:**

> Follow section 20. Reuse existing GPUs first. When more capacity is needed, use an available provider CLI/API or control the relevant browser tab to open NVIDIA Brev, inspect the correct organization, compare current options, and prepare the exact rental. Operate supported navigation and setup autonomously within this project. Do not interpret PC control as unlimited spending or unrelated-app access. Prepare a complete quote before asking for missing budget information or a tool-required transaction confirmation. Respect human authentication and action-specific tool gates. Verify actual GPU/VRAM, storage, connectivity, and a real model request before handing capacity to A15.
>
> Build Docker Compose deployment for TLS ingress, CPU API, PostgreSQL, workers, private model replicas, and static UI. Pre-stage weights/dependencies and validate runtime processing without external internet. Keep training labels out of the inference image. Record resources, live quoted costs, approved caps, expiry, storage behavior, judging keep-alive, cleanup responsibility, and rollback. Do not disable OS/browser security or type terminal commands through Windows UI automation.

**Acceptance:** a real GPU-verified access path exists, or a precise ready-to-act quote/handoff names the remaining dependency; no invented capability or unlimited-budget claim; deployment, external access, persistence, locality, and key-handling checks pass.

**Handoff:** A15 gets nonsecret connection metadata and verified capacity; A18 gets benchmark URL/failure controls; A20 gets the resource/cost ledger, keep-alive policy, runbook, and deployment manifest. Secrets stay outside Markdown, Git, and agent output.

### A18 — Performance, system reliability QA, and observability

**Mission:** prove speed with honest timing boundaries.

**Own:** src/forte/observability/, benchmarks/performance/, qa/system/.
**Inputs:** all component timings, A13 public API, A15 model service.
**Dependencies:** instrumentation skeleton starts immediately.

**Assignment prompt:**

> Implement monotonic stage timing and benchmark clients for T0-T5, queueing, decoding, prefill/inference, refinement, evidence, publication, and full payload transfer. Distinguish post-upload acceptance from total POST duration, model warmth from content cache hits, and status lookup from completed-response transfer. Test distinct uncached clips and a five-upload burst. Record hardware, versions, resolution/frame budgets, request counts, errors, and maximum latency. Verify available vLLM metric names on the pinned version. Keep telemetry inside the team's service and do not export private video content/descriptions.

**Acceptance:** a repeatable benchmark emits machine-readable results and a concise report; all failures are counted; no cache-hit inference claim is made; targets are labeled achieved/missed/unmeasured.

**Additional system-QA work:** test wrong/missing keys, large valid uploads, corrupt media, worker death, lease recovery, stale publication, cancellation, deadlines, restart, and concurrent requests. Ensure technical failures cannot become clean verdicts.

**Handoff:** A15 gets bottleneck evidence; A13 gets endpoint timing; A20 gets capacity/latency tables and reproducible reliability defects.

### A19 — Forte brandbook, design-system research, and visual QA

**Mission:** make the product intentionally designed for Forte using verifiable brand evidence and a clear evidence-review interface.

**Own:** brand/, docs/forte-brand-research.md, docs/forte-ui-guidelines.md, docs/visual-qa.md.
**Inputs:** public official Forte pages/assets; any organizer-provided brandbook; synthetic UI fixtures.
**Dependencies:** start immediately; deliver preliminary usable tokens by hour 2-3; A16 owns frontend implementation.

**Assignment prompt:**

> Follow section 21. Independently research Forte's authentic brandbook, logo usage, colors, typography, spacing, graphic language, components, and tone. Begin with official Forte sources and linked material; verify a manual's origin/version/applicability. If no authoritative brandbook is found, state that and distinguish observed website styling from mandatory rules. Never invent hex colors, font permissions, clear-space dimensions, or bank approval.
>
> Deliver a source ledger, permitted asset inventory, provenance-tagged tokens, typography/logo guidance, and component/UI specifications. Cover upload, actual processing, normal, possible-violation, failed, and low-visibility states. Adapt the brand to a working analyst tool with clear evidence, timestamps, rules, and readable Russian text. Give A16 a usable early handoff, then review synthetic-fixture screenshots for consistency, contrast, focus, responsiveness, and evidence legibility. Coordinate shared-PC access through A20/A17. Never inspect private video evidence through a cloud screenshot tool.

**Acceptance:** authentic sources cited; official/observed/inferred decisions distinguished; logo/fonts/assets have permitted use or a documented fallback; no invented palette is called official; tokens integrate with A16; core states receive synthetic visual QA; brand work does not block the baseline/API.

**Handoff:** A16 gets tokens, assets, component specs, and visual corrections; A20 gets brand rationale/limitations; A17 gets local asset/font packaging requirements. Vision robustness is assigned to A10 and system-failure QA to A18.

### A20 — Integration, release, and submission

**Mission:** deliver one coherent working service within 20 hours.

**Own:** docs/integration/, docs/submission/, qa/integration/, README.md, root dependency manifest and lockfile.
**Inputs:** all owner handoffs and organizer requirements.
**Dependencies:** all agents; coordinate from the first minute.

**Assignment prompt:**

> Maintain the integration branch, dependency lock, ownership map, milestone board, configuration versions, and decision log. Integrate small real vertical slices early. Enforce schema freeze, split lock, hour-6 baseline, hour-11 model selection, hour-16 feature freeze, and hour-19 release candidate. Resolve ordinary implementation choices using this plan and measured evidence; do not turn routine reversible work into repeated approval questions. Require real checks and explicit mocks/unmeasured claims in handoffs. Select the release checkpoint with A10, coordinate blockers, assemble the public URL/API key handoff, startup guide, licenses, source register, supported types, limitations, measured results, and five-minute pitch.

**Acceptance:** the release passes section 15; a clean deployment and external smoke test succeed; no unresolved blocker is hidden; documentation identifies actual versions and measured behavior; the fallback is runnable.

**Additional coordination:** serialize shared-PC input using an explicit control lease; ensure A17 provisioning and A19 public brand inspection never compete for focus. Track the GPU budget/expiry and brand handoff. A10 owns vision robustness, A18 owns system-failure QA, and A20 owns final cross-service checks.

**Handoff:** operator receives the release manifest, credentials privately, URLs, evidence, brand-source summary, GPU resource/cost ledger, judging keep-alive plan, and known limitations. Submit through the organizer portal only when within the authorized scope.

## 14. Integration protocol and decision gates

### Shared-workspace rules

- One owner per file; use the section-5 ownership map.
- Agents can read all relevant code and request changes from owners.
- Do not overwrite another agent's work or reset the shared checkout.
- A20 controls root manifests and lockfiles.
- A01 controls shared schemas.
- New dependencies require a small justification and compatibility check.
- Add meaningful interface/failure checks; avoid implementation-mirroring test volume.
- Deliver narrow, reviewable changes and integrate continuously.
- If worktrees are used, the operator deliberately chooses isolation and integration; this plan does not assume uncommitted work will be copied between them.

### Handoff template

~~~text
Agent:
Owned scope:
Changed paths:
Interfaces added/changed:
Real integration exercised:
Checks and results:
Measurements and timing boundaries:
Mocks/placeholders still present:
Known failures or limitations:
Required owner handoffs:
Next milestone:
~~~

### Gates

| Gate | Required proof | Action if missed |
|---|---|---|
| G0, T+0:45 | Shared contracts and ownership exist | A20/A01 resolve immediately; agents use frozen fixtures |
| G1, T+2 | Real local model call and decoder work | Use validated runner fallback; do not spend the entire build debugging serving |
| G2, T+3 | Reviewed initial taxonomy, rule coverage, split lock | Block training; continue baseline/product integration |
| G3, T+6 | Real video -> evidence -> jury JSON -> UI/API | Stop optional experiments and finish this vertical slice |
| G4, T+11 | Baseline/adaptation comparison and selected checkpoint | Freeze baseline if improvement is not established |
| G5, T+16 | Public service, locality test, required fields, stable failures | No feature additions; fix only release blockers |
| G6, T+19 | Complete acceptance evidence and runnable fallback | Release the most reliable compliant configuration |
| G7, before deadline | External smoke test, URLs/key/docs ready | Address only critical issues; preserve working service |

### Parent/orchestrator behavior

The parent tracks the critical path and file ownership. Agents report progress when outputs or blockers change. The parent consolidates status at milestones instead of accepting 20 incompatible final reports.

Do not let all agents independently choose model versions, schema shapes, queue systems, or legal taxonomies. Those decisions have named owners and explicit integration points.

## 15. Acceptance matrix

A20 marks every row PASS, FAIL, or UNMEASURED and links actual evidence.

| Check | Owner | Required proof |
|---|---|---|
| Exact POST route, multipart field, job ID | A01/A13 | Real request/response |
| Exact GET route and status enums | A01/A13 | Contract and real lifecycle |
| Missing/wrong X-API-Key rejected | A13/A18 | HTTP checks for both routes |
| Positive result contains complete event data | A11-A13 | Real reviewed example |
| Negative result has empty violations | A10/A13 | Real normal example |
| More than one event supported | A10/A11/A16 | Fixture plus real/allowed synthetic scenario |
| 1-3 valid JPEGs per event | A10/A12 | Decode and correspondence checks |
| Source timestamps preserved | A05/A12 | VFR/rotation fixtures and local source check |
| Rule references accurate and applicable | A04 | Source register and mapping review |
| No unsupported invisible conclusion | A04/A08/A10 | Error review |
| No audio or face identification | A07/A10 | Pipeline/prompt/config inspection |
| No external AI processing | A17/A18 | Network/config inspection and egress-disabled run |
| No labels in production decisions | A02/A17/A20 | Image/mount/prompt inspection |
| Group split integrity | A02/A10 | Manifest validation |
| Adapter selection justified | A09/A10 | Baseline comparison or documented fallback |
| Novel-angle evidence | A10 | Actual new-angle test or explicitly marked unmeasured |
| Public UI and API accessible | A16/A17/A20 | External client smoke run |
| Upload/polling latency measured honestly | A18 | T0-T5 report and sample counts |
| Distinct uncached analysis measured | A18 | Full stage timings, no result-cache shortcuts |
| Five simultaneous distinct uploads | A15/A18 | Completion/error/time records |
| All tested required videos under cutoff | A18/A20 | Worst-case observed latency and failures |
| Worker restart and stale attempt fencing | A14/A18 | Failure/recovery check |
| Offline model readiness and clean startup | A17 | Reproducible startup log |
| Commercial-use component licenses recorded | A07/A17/A20 | License manifest |
| GPU provisioning/access verified | A17/A15 | Instance ID, live quote, actual GPU check, private connection handoff |
| Rental budget and judging keep-alive recorded | A17/A20 | Known caps, quote, expiry/cleanup owner |
| Supported exclusive PC/browser operation | A17/A20 | Actual tool capability, observed target, control lease |
| Forte brand evidence distinguished from inference | A19 | Source ledger and authentic manual or explicit not-found status |
| Branded UI states and readability | A16/A19 | Source-backed tokens and synthetic visual QA |
| Fonts/logo/assets packaged locally | A16/A17/A19 | Asset manifest and permitted-use records |
| Submission package complete | A20 | Section-16 checklist |

A row marked UNMEASURED is not PASS. A tiny validation set is not evidence of production-wide accuracy.

## 16. Submission and pitch

### Submission bundle

- Public UI URL.
- Public jury API base URL.
- API key delivered privately.
- Exact sample upload and polling requests.
- Startup/use guide.
- Architecture and chosen-model description.
- Model revision, adapter revision, processor version, and weight/config hashes.
- Commercial-use license records.
- Supported event types and the evidence required for each.
- Rule/source register with applicable clauses where verified.
- Dataset/split methodology and duplicate handling.
- Measured quality, latency, concurrency, and failure behavior.
- Known limits: visibility, new types, angle change, tiny sample sizes, unresolved rule coverage.
- Reproducible local/offline deployment procedure.
- Recovery/fallback instructions.
- GPU resource/cost ledger, approved budget, judging keep-alive, and cleanup owner.
- Forte brand-source summary, token provenance, and asset/font usage records.

Keep organizer media out of public repos and public reports. Examples in externally shared documentation should be permitted synthetic fixtures or materials explicitly authorized for that use.

### Five-minute pitch

| Time | Content |
|---|---|
| 0:00-0:30 | Problem and the human-review workflow |
| 0:30-1:30 | Upload and one real result with timestamp, evidence, and source |
| 1:30-2:15 | Normal clip and false-alarm control |
| 2:15-3:00 | Multiple events or another camera angle, if actually demonstrated |
| 3:00-4:00 | Local architecture, original work, measured speed and completion |
| 4:00-4:40 | Validation counts, limitations, and what the fine-tune improved |
| 4:40-5:00 | Deployment readiness and practical next steps |

Prepare answers for: Why this model? What exactly was trained? How did you prevent leakage? Can it run with internet disabled? How do you ground rules? What happens when the action is unclear? Which latency was measured? What happens with five uploads? What remains unsupported?

Do not present Jev's published speedups as this system's results.

## 17. Failure decisions and cut list

### Mandatory fallbacks

| Problem | Decision |
|---|---|
| Adapter overfits or produces no credible improvement | Release the untuned baseline |
| vLLM integration blocked at hour 2 | Use the validated local runner; preserve the provider interface |
| Unresolved legal clause/applicability | Keep the mapping unresolved and repair grounding; never fabricate a citation |
| Model predicts a nonexistent frame | Reject the output and boundedly reprocess; never invent evidence |
| Decode fails | Mark failed; do not emit no_violation |
| Work cannot finish within deadline | Report honest failure; fix capacity/coverage before release |
| Long procedures do not fit coarse windows | Use temporal state and a targeted longer-context pass |
| Five-job queueing violates the budget | Add tested replicas or reduce measured inefficiency; do not hide queue time |
| Whole-response payload dominates transfer | Preserve evidence quality while reducing redundant images/bytes; report transfer separately |
| New angle breaks fixed crops | Retain full-scene context and adapt crop generation |
| Final-check results drive another change | Disclose that the set is now development data; seek a new independent check if feasible |

### Cut first

Account systems, historical dashboards, report exports, elaborate charting, animated UI, a new model architecture, RLCD reproduction, broad hyperparameter sweeps, multi-agent inference debates, untested quantization, automatic internet legal retrieval at runtime, and distributed infrastructure built from scratch.

### Add only after release gates

- A validated fixed-choice scoring path.
- A smaller distilled checkpoint with demonstrated quality.
- Better adaptive sampling backed by miss analysis.
- Additional verified rules.
- More allowed new-angle recordings.
- A concise comparison showing the practical benefit of adaptation.
- Optimized evidence transport that preserves the mandatory jury payload.

## 18. Operator launch prompt

Use this prompt in the future execution chat. It is an instruction template, not an action performed by this file.

~~~text
Implement the ForteBank Datathon service using FORTE_20_AGENT_EXECUTION_PLAN.md.

Create exactly 20 subagents named a01_contracts through a20_integration, mapped one-to-one to assignments A01-A20. Give each the shared instructions in section 12, its section-13 assignment, the ownership map, current milestone time, and the relevant shared interfaces. Do not allow them to spawn additional agents.

First establish A01 contracts, A02 data audit, A13/A14 API/job skeleton, A15 model readiness, A17 PC/browser capability and GPU access/provisioning, A19 Forte brandbook research, and A20 integration. Other agents may work against fixtures while dependencies mature. Integrate continuously.

A17 is the dedicated hands-on PC/browser and GPU operator. It may open NVIDIA Brev, inspect existing capacity, compare a live quote, and prepare or provision authorized resources under section 20. Respect actual tool capabilities, human authentication needs, spending caps, and action-specific gates. If caps/authority are missing, prepare the exact reviewable quote while other work proceeds.

A19 is a different dedicated agent for Forte's brandbook and design system under section 21. A10 owns vision robustness, A18 owns system-failure QA, and A20 owns final integration QA. A20 serializes shared-PC input so agents never compete for focus.

Honor the data boundary: provided videos, frames, and descriptions must not be sent to cloud AI models, including coding agents. Perform annotations through local human review or an approved local model inside the deployed service. Preserve raw inputs, licenses, and split integrity.

Use one selected open-weight visual checkpoint with short structured decisions, deterministic rule/evidence code, and independent GPU replicas. Retain the untuned baseline. Do not claim Jev-equivalent architecture, calibration, accuracy, or speed without direct evidence.

Enforce hour-6 end-to-end baseline, hour-11 checkpoint freeze, hour-16 feature freeze, and hour-19 release candidate. Maximize actual rubric coverage. Treat all latency numbers as targets until measured, separating upload, acknowledgment, queue, compute, publication, and response transfer.

Deliver one working public UI/API, exact jury JSON, local/offline startup, a complete acceptance report, private API-key handoff, and submission documentation. Keep limitations and unmeasured items explicit.
~~~

Agent names:

~~~text
a01_contracts
a02_dataset
a03_annotations
a04_regulations
a05_decoder
a06_sampling
a07_model_adapter
a08_decisions
a09_training
a10_evaluation
a11_events
a12_evidence
a13_api
a14_jobs
a15_serving
a16_frontend
a17_pc_gpu_deployment
a18_performance
a19_brandbook
a20_integration
~~~

## 19. Sources and evidence status

### Supplied local sources

- **S1 — Organizer case brief:** [ForteBank case PDF](<C:/Users/Wezeso/Desktop/Forte/Кейс_ForteBank.docx.pdf>), seven pages, especially sections 4-10 and FAQ A-G. Organizer requirements and rubric come from this file.
- **S2 — Supplied labels:** [Label workbook](<C:/Users/Wezeso/Desktop/Forte/open_labels.xlsx>), labels sheet, 42 data rows. Counts and original timestamps come from this workbook.
- Local file metadata/hash audit establishes the two duplicate pairs. It does not establish event contents or correct annotation times.

### Primary technical references

- **S3 — Reference model and license:** [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- **S3b — Smaller candidate:** [Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
- **S4 — Training implementation:** [Official Qwen VL training guide](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md).
- **S5 — Structured decisions:** [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).
- **S6 — Upload handling:** [FastAPI request files](https://fastapi.tiangolo.com/tutorial/request-files/).
- **S7 — Background compute boundary:** [FastAPI background tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/).
- **S8 — Serving measurements:** [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) and [optimization guidance](https://docs.vllm.ai/en/latest/configuration/optimization/).
- **S9 — Transactional queue building block:** [PostgreSQL SELECT and locking clauses](https://www.postgresql.org/docs/current/sql-select.html).
- **S10 — Original design inspiration:** [TypeSafe: System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

These pages establish available mechanisms and component behavior. They do not validate the proposed task accuracy, hardware sizing, fine-tuning settings, or latency targets.

### Research that remains part of execution

- Actual event taxonomy from permitted local review.
- Exact applicable and current Kazakhstan regulatory clauses.
- Real checkpoint comparison on locked groups.
- Real uncached latency and concurrent capacity.
- Actual new-angle robustness.
- Final event coverage and evidence readability.
- Live Brev availability, all-in quote, account quota, and authorized spending caps.
- Actual future-session browser/native-control capability and authenticated GPU access.
- Authentic Forte brandbook availability and approved logo, colors, fonts, and rules.

The release report must replace planned targets with measurements or explicitly label them unmeasured.

### GPU, computer-operation, and brand references

- **S11 — NVIDIA Brev:** [overview](https://docs.nvidia.com/brev/getting-started/overview), [quickstart](https://docs.nvidia.com/brev/getting-started/quickstart), [instance creation](https://docs.nvidia.com/brev/cli/instance-creation), [instance management](https://docs.nvidia.com/brev/cli/instance-management), and [official console](https://brev.nvidia.com/).
- **S12 — Brev CLI platform support:** [official repository](https://github.com/brevdev/brev-cli). Verify installed commands and current availability; examples do not establish today's prices or quota.
- **S13 — Official Forte research:** [Forte](https://forte.kz/), [retail site](https://bank.forte.kz/), and [business site](https://business.forte.kz/). These are research starting points, not an authenticated brandbook.
- **S14 — Current native Windows Computer Use instructions:** [SKILL.md](<C:/Users/Wezeso/.codex/plugins/cache/openai-bundled/computer-use/26.1013.40934/skills/computer-use/SKILL.md>), [runtime guidance](<C:/Users/Wezeso/.codex/plugins/cache/openai-bundled/computer-use/26.1013.40934/docs/guidance.md>), and [confirmation rules](<C:/Users/Wezeso/.codex/plugins/cache/openai-bundled/computer-use/26.1013.40934/docs/confirmations.md>). Re-read the current applicable skill at execution; future bundle paths and tools can differ.
- **S15 — Accessibility:** [W3C WCAG contrast minimum](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html). Accessibility targets are team design choices, not claims about Forte's brandbook.

## 20. PC control and NVIDIA Brev GPU provisioning

### Scope and platform

Interpret the user's reference to NVIDIA brew as **NVIDIA Brev**. Use the [official console](https://brev.nvidia.com/) and current official documentation.

A17 is explicitly assigned hands-on operation: navigate the relevant browser, inspect account capacity, compare live GPU choices, prepare a rental, connect to an authorized machine, and deploy this project. This is a future execution assignment. Editing the plan does not rent a GPU or start a PC-control session.

Reuse existing usable hardware before purchasing additional capacity. Brev also documents connecting existing hardware; use that option only if it helps the task and its access requirements are authorized.

Brev supports browser-console and CLI instance workflows. Its Windows CLI instructions use WSL; inspect what is actually installed rather than assuming it exists. [S11, S12]

### Capability and operating procedure

1. Read the active browser/computer-use tools and applicable skills in the execution session.
2. Prefer an available provider API/CLI for repeatable infrastructure actions; use supported browser control for console interaction.
3. If native Windows control is necessary, follow the current Computer Use skill's documented runtime and window selection. Do not invent native-control APIs or claim unrestricted PC access.
4. Select an observed Brev tab/window and verify the correct account/organization. Do not act from stale screenshots or guessed element positions.
5. Observe the current interface, take one targeted action, refresh, and verify the outcome. If a launch result is uncertain, inspect existing instances before retrying.
6. Use an available exec/SSH tool for terminal commands. Do not type shell commands through Windows terminal UI automation or the Run dialog.
7. Use an existing authenticated session where possible. Human login/OTP/CAPTCHA or required tool handoffs remain explicit.
8. Stop device input if the user takes control or stops automation. Keep unrelated apps outside the task scope.
9. If a capability is unavailable, continue public quote/research through supported tools and report the exact access dependency. Markdown does not create a missing control capability.

The current native Windows skill uses node_repl and @oai/sky. Its guidance prohibits terminal-app UI automation, changing security/privacy settings, and automating authentication dialogs. Read its API reference before using native methods. [S14]

For native Windows actions, its confirmation policy explicitly includes **"Confirm financial transactions"** at action time. Account/key creation and some other actions also have gates. Respect those requirements for that route. Apply the actual rules of other supported tools to their routes. Do not silently bypass a required gate.

When a gate is reached, finish the preparation first, then present the exact final action, quoted cost, and the rule requiring confirmation or human handoff. Routine browsing and quote preparation do not need repeated permission questions.

### One operator at a time

A20 maintains a control lease in docs/integration/pc-control.md:

~~~yaml
controller_agent: a17_pc_gpu_deployment
purpose: nvidia_brev_provisioning
target: observed_browser_tab_or_window
lease_started_at: actual_time
next_handoff: a19_public_brand_research_if_needed
~~~

Only one agent issues shared-PC GUI input at a time. Separate tabs do not make simultaneous native mouse/keyboard operations safe. A19 should use web fetch/search for public research when that avoids focus conflicts.

Targeted screenshots may show the provider console, public brand pages, or synthetic UI fixtures. They must not capture private video evidence, annotation screens, secrets, or unrelated windows. Do not use a full-desktop capture that would expose the provided dataset to a cloud model.

### Spending policy

Inventory and quote preparation proceed while budget fields are unresolved. A paid rental requires concrete purchase authority and meaningful caps. Abundant existing GPU resources do not imply unlimited spending authorization.

A17 prepares infra/spending-policy.example.yaml:

~~~yaml
provider: NVIDIA Brev
authorized_account_or_org: null
billing_currency: null
max_total_charge: null
max_hourly_charge: null
max_concurrent_instances: null
max_billable_runtime_hours: null
max_storage_gib: null
allowed_regions: []
judging_keepalive_until: null
cleanup_owner: null
purchase_authorization_reference: null
~~~

Null means unknown, not unlimited. Caps come from the user/operator's explicit authorization; do not fill them with invented acceptable prices.

Prepare a concrete comparison before committing money:

| Quote field | Required information |
|---|---|
| Identity | Provider, account/organization, region, proposed instance name |
| GPU | Exact GPU type/count and stated VRAM; verify actual hardware after launch |
| Host | CPU/RAM, disk type/capacity, available network/ports |
| Price | Current displayed hourly rate, billing currency/unit, minimum charge if any |
| Other charges | Storage, egress, public address, taxes, or unknown items |
| Capacity | Availability, quota, reservability, spot/preemptible behavior |
| Runtime | Setup, training, serving, judging keep-alive, contingency |
| Exposure | Expected total charge and maximum authorized cost |
| Lifecycle | Stop/delete behavior, retained disks and billing, cleanup owner |
| Existing resources | Whether the proposal duplicates usable capacity |

If caps or purchase authority are missing, prepare the exact reviewable quote and configuration, then ask only for the missing decision. Continue model/data/frontend work meanwhile. Once authorization exists, work within it without asking the same question repeatedly, except for any action-time gate imposed by the actual tool.

Do not add payment methods, change billing arrangements, accept a long-term commitment, or exceed instance/storage/region limits merely to finish this task.

### Provisioning steps

1. **Inventory:** inspect existing servers and Brev resources; identify reusable capacity.
2. **Initial shape:** obtain one suitable GPU for a real smoke run. Add independent inference replicas and training capacity only when measured demand and budget justify them.
3. **Compare options:** compare two or three actually available suitable 48 GB-class/80 GB-class configurations or equivalents. Consider setup time, memory, throughput, stability, disk, region, and total cost; a model name alone does not decide value.
4. **Prepare:** choose the organization, instance name, image/container, storage, network access, and connection path. Record enough state to avoid duplicate launches.
5. **Authenticate:** let the human complete required login/OTP steps. Never scrape a password manager, expose session cookies, or print access tokens.
6. **Final action:** apply budget/authorization and current tool gates, then record the actual instance ID.
7. **Connectivity:** verify actual access rather than accepting a console Running label.
8. **Hardware:** through exec/SSH, inspect nvidia-smi, actual GPU memory, disk, and a small CUDA operation.
9. **Application readiness:** load the pinned local weights and perform a real model request; then hand capacity to A15.
10. **Deployment:** stage the locked dependencies, private services, public TLS API/UI, and offline processing check.
11. **Record:** update costs, active resources, judging keep-alive, cleanup ownership, and remaining constraints.

The documented CLI includes login, instance discovery/refresh, creation, and connection workflows. Check the installed help and official version before using flags or GPU identifiers. Do not copy an old example and assume its hardware/price is available. [S11]

On Windows, use already functioning WSL or an existing supported SSH/exec path. Do not turn provisioning into BIOS changes, disabled security, terminal UI automation, or a reboot-dependent OS migration.

### Data, secrets, and infrastructure boundary

An authorized rented GPU may host the team's own weights and deployed service. This does not authorize a managed cloud foundation-model API, public notebook, visual assistant, or public video-sharing service.

Project/private media transfers must target the specifically authorized team-controlled deployment environment and respect organizer restrictions. Keep all model processing inside that environment.

Secrets stay outside Markdown, Git, screenshots, terminal output, and agent messages. Store only a local secret reference in the resource record.

### Resource and cost ledger

A17 owns infra/resources.json and docs/gpu-provisioning.md. Record:

- Provider/instance/storage IDs, organization, and role: training or inference.
- Exact GPU type/count and verified memory.
- Live quote timestamp, rate/currency, approved caps, expected/actual runtime where available.
- Storage/network/stop/delete billing behavior and unresolved items.
- Connection/readiness results and a nonsecret reference to credentials.
- Judging keep-alive time, cleanup owner, current lifecycle state, and last billing verification.

Do not shut down the public service at submission if judging follows later. Plan that window within the authorized budget. If judging time is unknown, record the unresolved availability/cost item rather than silently choosing infinite uptime.

After the authorized judging/demo window, follow the approved cleanup policy, preserving required artifacts first. Resource stop/delete/payment actions follow current tool rules. This plan does not itself schedule an automation or terminate an existing machine.

### Acceptance

- Correct account/organization and a real connection path verified.
- Paid launch, if any, within known limits and identified by instance ID.
- No duplicate accidental machine after retries.
- Actual GPU/VRAM, CUDA operation, and real local inference verified.
- Public API and private compute are correctly separated.
- Runtime processing works without external AI/internet dependencies.
- Cost, judging keep-alive, and cleanup responsibility are recorded.
- Opening the console or filling a form is not reported as successful provisioning.

## 21. Forte brandbook research and design system

### Dedicated owner and evidence hierarchy

A19 is separate from A17's PC/GPU operation and A16's frontend implementation. A19 owns brand research, provenance, tokens, assets, component specifications, and visual QA. Vision-model robustness is with A10, system reliability with A18, and final integration with A20.

Research in this order:

1. A brandbook/design package actually supplied by organizers.
2. An authentic current brandbook or official press/design assets published by Forte.
3. Official public pages: forte.kz, bank.forte.kz, business.forte.kz.
4. Official linked assets, product screens, and clearly attributable current materials.
5. Third-party portfolios as discovery leads only; they cannot establish current mandatory rules.

An authoritative public brandbook has **not been authenticated in this planning work**. Its availability, version, and applicability remain research tasks. Do not claim brandbook compliance merely from recognizing a logo or copying a homepage.

For every finding, use one provenance status:

- **Official rule:** verified applicable brand document with page/section.
- **Observed official styling:** measured from a dated official page/asset; not automatically mandatory.
- **Team design choice:** an explicitly inferred choice for this analyst workflow.
- **Unresolved:** insufficient evidence, uncertain version, or unclear permission.

### Research checklist

- Confirm the correct main ForteBank/Forte identity and distinguish business, Premier, mobile, historical, or campaign variants.
- Locate permitted original logo variants, vector files, backgrounds, proportions, and usage constraints.
- State clear space/minimum size/recoloring rules only when actually sourced.
- Identify typography, Russian/Kazakh glyph support, font licensing, redistribution rights, and a permitted fallback.
- Extract exact color values from verified manuals or official CSS/assets; do not guess hex codes.
- Inspect heading/body hierarchy, spacing, corner treatment, borders, shadows, icon style, navigation, buttons, forms, and cards.
- Study tone and consistent Russian/Kazakh labels.
- Record public reference URLs/dates and separate marketing composition from an analyst review workspace.
- Document allowed asset use; public availability alone does not establish unrestricted reuse.
- Use an allowed official logo asset or clearly labeled text treatment. Do not distort, redraw, or generate a substitute logo with ImageGen.

Research public sources without contacting bank staff or submitting messages/forms unless the user explicitly authorizes that communication.

### Required outputs

~~~text
brand/source-ledger.md
brand/tokens.json
brand/assets/manifest.json
brand/assets/                 permitted local assets only
brand/reference-captures/     public pages or synthetic UI only
docs/forte-brand-research.md
docs/forte-ui-guidelines.md
docs/visual-qa.md
~~~

Every token records role, value, provenance, source, and restrictions. This example is a record shape, not an official palette declaration:

~~~json
{
  "brand.primary": {
    "value": null,
    "status": "unresolved",
    "source_url": null,
    "source_section": null,
    "observed_at": null,
    "notes": "Verify from an applicable official source before calling this official."
  }
}
~~~

A16 can use a neutral fallback while research proceeds. An unresolved token must not silently become an official brand value in the release.

### Token/component specification

| Area | Decisions and evidence |
|---|---|
| Brand | Logo variant/placement, primary/accent colors, source status |
| Semantic status | Distinct normal, possible-violation, processing, failure, and low-visibility states |
| Typography | Permitted family/fallback, weights, heading/body/caption sizes, readable numerals |
| Surfaces | Page/panel/evidence backgrounds, borders, elevation, keyboard focus |
| Layout | Workspace width, hierarchy, spacing scale, panel alignment, responsive behavior |
| Actions | Primary/secondary buttons, disabled/loading/hover/focus states |
| Forms | Upload/dropzone, accepted type, help text, validation errors |
| Review | Video player, event list, timeline, evidence gallery, source/rule card |
| Language | Russian labels, cautious verdict wording, consistent event terms |
| Assets | Origin, permitted use, local path, checksum, license or unresolved state |

Keep brand accents separate from incident semantics. A branded button must not imply that an incident was detected.

### Screens and states

1. **Upload:** clear purpose, accepted format, one obvious action, concise local-processing explanation.
2. **Processing:** actual stage/status and elapsed time. No fake percentage or guaranteed completion time.
3. **Normal result:** no-detected-signs wording, source video access, coverage/visibility limits.
4. **Possible violations:** event type/time, selected evidence, visible observation, applicable rule/source.
5. **Multiple events:** stable ordering, selection, and preserved distinct events.
6. **Unclear view:** visible limitation without inventing a third jury verdict.
7. **Failed processing:** honest error styling, job reference, actionable retry information.
8. **Edge cases:** missing file, wrong type, lost connection, unavailable backend, narrow viewport.

Use a desktop evidence-review composition: video/evidence as the main surface, adjacent event navigation, and a readable explanation/source panel. Do not reproduce a consumer banking hero at the expense of reviewing footage.

### Readability and offline assets

- As a team target, use WCAG text contrast of at least 4.5:1 for ordinary text and 3:1 for large text. This is accessibility guidance, not a verified Forte rule. [S15]
- Provide labels, keyboard focus, and text/icon status cues; do not rely solely on color.
- Keep Russian/Kazakh strings, long rule titles, timestamps, and decimal values readable.
- Preserve source evidence aspect ratio/detail and allow enlargement.
- Review practical desktop sizes first, then a narrow viewport with no clipped actions.
- Use restrained motion that does not compete with evidence.
- Package fonts/icons/assets locally with documented permitted use; avoid runtime font/CDN dependencies.

### Handoff and timebox

- **By hour 2-3:** source-ledger status, short design brief, first usable tokens, logo/font decision, core component states.
- **By hour 6:** review the initial functional UI using synthetic fixtures.
- **By hour 14:** freeze source-backed tokens and fix material consistency/contrast/readability defects.
- **After feature freeze:** fix only release-blocking visual issues.

A19 provides assets/tokens and specific comments; A16 edits its own component files. Coordinate shared-browser input through A20's control lease. Private video/evidence review remains local and must not be shown to a cloud-backed screenshot tool.

### Acceptance

- Source ledger states whether an authentic current brandbook was found.
- Every claimed official rule has a supporting source/page.
- Website measurements and team inferences are labeled honestly.
- No invented palette, font permission, or clear-space rule is called official.
- Permitted assets are packaged locally.
- All core UI states pass documented synthetic visual QA.
- Focus, contrast, status semantics, responsiveness, and evidence readability are checked.
- The prototype does not imply that it is an officially deployed bank system.
- Brand polish does not delay detection, grounding, evidence, API reliability, or local execution.

