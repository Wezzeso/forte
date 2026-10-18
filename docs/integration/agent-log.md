# Agent log: speed program, 18 Oct 2026

Prompt: `Claude outputs/FORTE_AGENT_PROMPT.md`. The user chose **"speed, accuracy-safe"** for the remaining
GPU window (the A100 stops at 11:01:25 Asia/Qyzylorda), and pre-approved promotion if a candidate passes
the release gate with a one-step rollback. All numbers below come from
[`benchmarks/quality/speed-program-20261018.json`](../../benchmarks/quality/speed-program-20261018.json)
(structured fields only; no frames, evidence JPEGs or model free text). Development set, in sample.
No learned weights, thresholds or calibration constants were fitted to labels, so no LOVO claim applies
and none is made.

## Access

The coding session's own shells cannot reach the Brev host. The user started a file-queue bridge
(`tmp/agent-bridge/runner.sh`) in WSL, which runs reviewed scripts through the existing Brev SSH
configuration. Every job and log is kept in `tmp/agent-bridge/{done,out}`. Brev certificate minting
failed transiently once; jobs retry transport errors only.

## Harness built (prompt §6, minimal subset)

| Tool | Purpose |
|---|---|
| `tools/fast_agreement.py` | Label-free reference-vs-candidate comparison: verdicts, events (type, |Δt| ≤ 0.5 s), candidates, layout boxes, per-field state agreement, calls/tokens, stage timings. Never reads labels or free text. |
| `tools/fast_latency.py` | End-to-end latency, 202 → first `done`, polling every 0.2 s; sequential and N-parallel modes; validates schema, raw-JPEG evidence, 1–3 frames, time bounds; records arrival vs completion order. |
| `tools/fast_eval.py` | Now records per-video call counts and total completion tokens. |

Not built in this window: manifest v2, typed ground truth, augmented copies, lockbox, leakage tests,
synthetic resolution clips. Requirements A–H of §4.1 are therefore **not measured**.

## Finding 1: the live pipeline is not reproducible run to run

Two runs of the unchanged live image and profile on the same 40 videos (`ref1`, `ref2`):

| Comparison | Verdicts equal | Event sets equal | Layout boxes identical | State fields equal |
|---|---|---|---|---|
| ref1 vs ref2 (identical code/config) | 38/40 | 37/40 | 1/40 | 98.9 % |

Scores: ref1 33/40 (1 FP on X-19, 6 FN); ref2 33/40 (0 FP, 7 FN). The one-call layout step returns
slightly different boxes almost every run, so crops and downstream states shift. A replay of real
verification requests showed the first (cold-cache) answer can differ from later identical answers
(e.g. t_sec 32.5 vs 33.5), i.e. vLLM prefix-cache and batch composition change greedy outputs.
Consequence: the §8 agreement gate ("identical events and verdicts on all 40") cannot be met even by
the reference against itself. Candidates were therefore judged against the measured noise floor.

## Change S1+S3+S4c+S10 (flags `verify_concise`, `parallel_layout`, `job_priority`; worker poll 0.1 s)

Visual principle: *none changed* — every model request keeps the same prompt, images and response
grammar; only generation length, request order and pickup delay change.

- `verify_concise`: verification stops generating after `visible, confidence, t_sec` (same grammar,
  stop string `"observation`); the description is the catalog template (no model free text in results).
  Replay on four real candidates: decisions identical to the full path under identical cache state.
- `parallel_layout`: full-frame state questions start while the layout call runs; crop questions wait
  for it. Same requests (unit-tested).
- `job_priority`: client-side priority semaphore; earlier jobs get model slots first (FIFO among equals).
- `FORTE_WORKER_POLL_SECONDS=0.1` (was 1.0): worker pickup.

Defaults stay off in `DEFAULT_CONFIG`; the profile enables them.

Agreement (two candidate runs vs two reference runs):

| Pair | Verdicts | Events | State fields |
|---|---|---|---|
| ref1 vs ref2 (noise floor) | 38/40 | 37/40 | 98.9 % |
| ref1/ref2 vs candA/candA2 | 39/40 each | 35–37/40 | 98.6–98.7 % |
| candA vs candA2 | 40/40 | 39/40 | 98.4 % |

Scores: candA 32/40 and candA2 32/40 (1 FP on X-19, 7 FN). Single-job staging run of the same build:
X-15 detected, X-19 FP. All differences sit on videos that also flip between reference runs
(X-15, X-19, X-30, X-32, X-38). The per-field minimum is `client_seat` (82–87 % vs 93 % ref-ref;
76 % cand-cand), which is the chair crop that depends on the noisy layout box.

Latency, isolated staging stack (API → PostgreSQL → worker → existing q30), one job at a time, 40 videos:

| Build | p50 | p95 | max | ≤ 10 s |
|---|---|---|---|---|
| Live image `8c8b9255…` + live profile, poll 1.0 s | 9.689 s | 27.817 s | 32.899 s | 21/40 |
| `speed2-ede4958c` + candidate profile, poll 0.1 s | 8.367 s | 22.534 s | 27.173 s | 28/40 |

Every one of the 40 videos was faster (0.2–7.0 s). Four videos in parallel through the analyzer:
total 392.5 s → 342.2 s; verification p50 8.74 s → 3.94 s, max 32.3 s → 6.5 s.

Tuned parameters added: 0 fitted (poll 0.1 s chosen a priori).


## Promotion of speed candidate A (user-approved), 18 Oct 07:58 Asia/Qyzylorda

The user approved promoting A after seeing the noise-floor evidence above. Worker-only release
`deploy/releases/speed-20261018-ede4958c` on the Brev host:

| Item | Value |
|---|---|
| Worker image | `forte-review:speed2-ede4958c` = `sha256:26dc1184bb316d9ac7c55eb5a51d50ec472e5586002ad30dc37593ead5cbf4bc` (accepted `8c8b9255…` + ten `forte/fast` files) |
| Profile | `fast_config.json` sha256 `8fc400ec…40fd` (live profile + `parallel_layout`, `verify_concise`, `job_priority`) |
| Worker env | `FORTE_WORKER_POLL_SECONDS=0.1` |
| API / frontend | unchanged (`03304722…`, deployed by a concurrent frontend release `footer-crop-20261018-3ace4714`) |
| Rollback | `deploy/releases/speed-20261018-ede4958c/rollback.sh` restores profile `993d2f22…` and worker `8c8b9255…` with `up -d --no-deps worker` |

Activation required an idle queue, swapped the profile atomically (hash-checked), recreated only the
worker, and verified image, module hashes, effective flags and public readiness (200).

Staging, 6 longest clips in parallel (202 → done): 6/6 passed, p50 70.1 s, max 119.3 s.
Public HTTPS from the GPU host, 6 longest clips in parallel: **6/6 passed**, completed in arrival
order, 202 → done p50 64.1 s, **max 101.2 s**; uploads 22.3–43.5 s; total wall 144.8 s. The earlier
public record (142.6 s max) was measured from before upload, so compare it with the 144.8 s wall, not
with 101.2 s. The ≤ 60 s concurrency target is **not met**.

## Change S2 change gating (flag `change_gating`) — not qualified, stays off

Principle: a region whose pixels did not change keeps its previous answer (forced refresh every 5 s).
Calls 5,746 → 3,709 (−35 %), completion tokens −36 %, faster under load (365–372 s vs 613–719 s for
two concurrent runs). But state agreement fell to 97.4–97.6 % (below the 98 % floor; `cash_in_hands`
≈ 90 %, `hand_counting` ≈ 93 %), and verdicts became less stable: a new false alarm on X-5 in one run
and a lost X-4 detection in one run. Not promoted.

## Accuracy probes (flag `probe_types`, `src/forte/fast/probes.py`) — failed the negatives gate

User request at 07:56: at least 39/40. Probes ask closed questions over dense original frames of a
client-visit window, under two phrasings that must agree (prompt §9.4), using the 30B model:

| Type | Windows | Fires on the 19 negatives | Fires near a labelled positive |
|---|---|---|---|
| NO_AUTHENTICITY_CHECK | 17 | 6 (X-3, X-5, X-19, X-35, X-37, X-39 ×2) | 6 videos |
| NO_CLIENT_ID | 19 | 5 (X-5, X-35, X-37, X-39, X-45) | 3 videos |
| NO_RECOUNT | 3 | 0 | 0 |

By §5.4 a type that fires on any negative stays disabled; no threshold was tuned to rescue it.
None of the target misses (X-44, X-22, X-13) produced a window. Conclusion: at 464×848 the local
VLM cannot see ID cards versus banknotes or detector use, consistent with the earlier RF-DETR and
crop-LoRA results. 39/40 is not reachable honestly in this window.

## Layout consensus (flag `layout_consensus: 3`) — no net gain, stays off

Principle: fixed furniture does not move; the layout is the coordinate-wise median of boxes found on a
majority of 3 frames spread over the clip. Two runs: 33/40 and 33/40, **0 false alarms in both**
(X-19 no longer fires), 7 misses each; X-15 and X-4 still flip between runs. State agreement vs the
reference fell to 97.2–97.4 % (`client_seat` 52–57 %: the chair crop moved). Not promoted.

## ROI activity and teacher windows (diagnostics, no deployable change)

- `tools/roi_activity.py` (pixels only): motion seconds inside the layout's detector / counting machine /
  safe boxes per client visit. No separation of the misses: X-44 never shows a client visit and has
  no ROI motion; X-2 shows detector motion (so it is not "no authenticity check").
- `tools/teacher_windows.py` (enum answers only) at the labelled windows of the misses and at 38
  control windows on negatives: the 30B model names NO_CLIENT_ID on nearly every visit and
  NO_AUTHENTICITY_CHECK on most windows, negatives included; Qwen3-VL-32B-FP8 says "key in safe lock"
  on three negatives (X-21, X-25, X-27). No discriminative signal.

## Larger models on a second (old-account) H100, 18 Oct 08:35–09:15

Host `forte-h100-q32` (Brev hyperstack_H100, USD 3.00/h), vLLM v0.31.0-cu129, same pipeline/profile.

| Model (local, Apache-2.0) | Result on 40 videos |
|---|---|
| Qwen/Qwen3-VL-32B-Instruct-FP8 (sha `4bf2c2f3…`) | 32/40, 0 FP, 8 FN (also lost X-18, X-30) |
| Qwen/Qwen3.8-27B-FP8 (sha `017b9c7a…`) | run `q38a` started 09:12; see handoff prompt |

## Status at 09:13 and handoff

User decisions: keep production (33/40, speed release) on the old account; test new models on a new
NVIDIA/Brev account (USD 50 that must last until 18:00); try Gemma 4 31B. Remaining work is specified in
`Claude outputs/FORTE_AGENT_PROMPT_HANDOFF_20261018.md`.


## Continued session, 18 Oct 09:20 Asia/Qyzylorda

Production was read-only checked through the bridge: public readiness HTTP 200, speed2 worker and
profile hashes match the handoff, vlm30 running, no active jobs at observation. The user explicitly
approved disabling Windows task `Forte-Brev-budget-stop-wd88a7vvt`; its state is now Disabled.
The original `infra/budget_guard.py` cutoff remains unchanged, so manually invoking it with
`--execute` after 11:01 would still stop production. No live API, frontend, worker or profile changed.
Operational record: `benchmarks/quality/production-operations-20261018T0419Z.json`.
The old account's visible Brev billing Current Balance is USD 14.53; at USD 2.03/h this is about
7.16 compute hours after test-host cleanup, excluding storage and billing lag. It does not cover 18:00.

Visual principle before the next model comparison: stronger local perception may distinguish the
same original visual details under unchanged prompts, sample selection, temporal rules and thresholds.
Gemma 4 and Qwen3.8 are evaluation-only server selections; no new type, accuracy threshold or live
fast_config flag is enabled. Fitted/tuned accuracy parameters: 0. Synthetic schema and coordinate
checks precede real-data Gemma testing. All development results remain in-sample; no LOVO, robustness,
leakage, held-out, evidence-completeness or release qualification claim is made from these comparisons.

### T0 complete; new-account setup under way, 09:23 local

`benchmarks/quality/qwen-models-20261018T0422Z.json` records Qwen3-VL-32B-FP8:
32/40, 0 FP, 8 FN; Qwen3.8-27B-FP8 first run: 29/40, 0 FP, 11 FN, 40 processed,
0 failed, 401.295 s analyzer batch wall, 5,254 calls. Neither qualifies. The Qwen3.8 run
used the previous server command without an explicit disable-thinking default or reasoning parser;
new-account repetitions will record that serving difference rather than claim equivalence.
Structured states, layouts, events, timings and provenance were exported before deletion.
Raw reports/free text were never transferred and were deleted with the old test host.
Deletion of forte-h100-q32 / rrbfqp3zd is verified; old production wd88a7vvt remains RUNNING:
`benchmarks/quality/old-test-cleanup-20261018T0423Z.json`.

New isolated account login verified (HOME /home/wezeso/brev-new); new H100 plp5me7vz,
forte-new-h100, hyperstack_H100 at USD 3/h, is provisioning. New SSH config is at the expected
isolated path; default account/profile untouched. The user approved the prepared Windows budget
task; it is installed Ready for 17:55 local and binds only this new host and organization.
Guard dry-run is recorded in `benchmarks/quality/new-account-budget-guard-20261018T0423Z.json`.
It caps the budget at USD 50, keeps USD 2 reserve, and stops before 18:00. The PC must be available
for Windows scheduling; stop initiation can take provider time and storage remains billed until deletion.

### Gemma processor choice declared before inference

Visual principle: a larger vision-token budget can represent small original objects with less spatial
pooling. Gemma candidate uses the documented `--mm-processor-kwargs {"max_soft_tokens":560}`
for BOTH independent runs, while retaining original input pixels/prompts/rules/config. This selects
527 pooled tokens for a native 464x848 image (280-budget default would use 264); it does not recover
missing pixels. Its 32-image synthetic request fits the 32768 context. Primary source:
https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html#dynamic-vision-resolution .
This is one predeclared processor knob, chosen before real-data inference; zero parameters fitted to
labels. The serving arguments and payload digest must be recorded with results. No production flag
or profile changes. User's new testing/report deadline is 11:30 Asia/Qyzylorda.

Gemma serving capacity is also predeclared before real-data inference: max_num_seqs=16 and
max_num_batched_tokens=16384, GPU utilization remains 0.90. These are scheduler limits;
16 full 32768-token contexts are not reserved. vLLM must pass its allocator/readiness and
synthetic multi-image checks. This improves potential batching of short state queries;
latency remains unmeasured. Three explicitly chosen candidate knobs total (vision budget,
scheduler sequence cap, batch-token cap); zero parameters tuned to labels. The same serving
configuration is used for both independent runs. Offline and JSON/failure requirements remain.

Before any real inference or server start, pinned vLLM allocator review corrected the provisional
Gemma batch-token choice above: use max_num_seqs=16, max_num_batched_tokens=4096,
enable_chunked_prefill=true, async_scheduling=false. Sliding-window admission reserves in-flight
prefill tokens as well as its retained window; batch16384 would exceed the available BF16 headroom.
Source: https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/v1/kv_cache_interface.py .
The same final settings will be used for both runs. Five explicitly chosen candidate controls
(vision budget, sequence cap, batch-token cap, chunking, scheduling); zero accuracy parameters
fitted to labels. Readiness and synthetic checks remain required; no measured speed claim.

### Gemma synthetic preflight passed, 09:37 local

All 15 synthetic-only cases passed, including every prompts.py schema, strict plain JSON without
reasoning markup, exact concise-stop behavior, normalized-box calibration (maximum error 3/1000),
and distinct 14/18/32-image requests. Benchmark: `gemma4-smoke-20261018T043700Z.json` in
benchmarks/quality (the timestamped actual path is also in session-progress-20261018.json).
The box convention needs no adapter. Measured startup: ready after 175 s; available KV cache10.7GiB,
54,642 GPU cache tokens, recorded max32k-context concurrency1.67, GPU allocation73,019MiB.
These are startup measurements, not per-job latency or a 6-upload qualification.
First full Gemma 40-video evaluation has begun; second is sequentially queued. No accuracy claim yet.

### Parallel Qwen host and speed contingency, 09:43 local

The user explicitly approved a temporary second new-account H100 (USD3/h until11:20) and its
identity-checked Windows task. New host forte-new-q38 /xw5720fii is provisioning; the Ready task
Forte-Brev-new-budget-stop-xw5720fii targets11:20 only. Old production stays unchanged. Recorded
budget dry-run: parallel-q38-budget-20261018T0442Z.json. Compute ceilings through the two approved
cutoffs total USD30.75 (main25.75 +parallel5), leaving19.25 of the USD50 before storage/lag.
Qwen3.8 no-thinking repeats and diagnostics will run on this separate GPU in parallel with Gemma.

Label-free speed contingency declared before any completed Gemma score: Gemma BF16 has10.7GiB
KV pool and at04:39:54Z had15 active/48 waiting calls, cache91.5%, GPU94%. Online fp8_per_tensor
is supported for its language matrices (vision tower/connector remainBF16). A separate candidate
may be tested after Qwen finishes on the second GPU: context32768, imagebudget560, seq32,
batch8192, chunkedprefill enabled, asyncscheduling disabled, mem0.90. Both FP8 runs must be
independent and must not be combined with BF16 counts. No threshold/rule/label fitting is involved;
quantization is not presumed accuracy-safe. BF16 experiments continue unchanged.

### User-authorized Gemma fine-tuning and 20 parallel subagents, 09:54 local

The user requested Gemma fine-tuning and explicitly authorized20 parallel subagents. Work is divided
between public-data preparation, native crop manifests, QLoRA training/runtime, held-out public
comparison, adapter export/reload, integrity/licenses, model error/coverage/negative gates, source
provenance, failure semantics and speed/staging preparation. Cloud GPU actions remain centralized;
no extra GPU rental beyond the two approved H100s. Production remains untouched.

Visual principle: better discrimination of visible banknotes versus rigid ID cards from local public
object examples may improve Gemma's structured perception. Training is restricted to public train
sources (KZ-BD/MIDV-Holo) with frozen public group splits; supplied bankvideo labels are scorer-only,
no per-video conditions/identifierfeatures, no fitting against the40 verdicts. Checkpoint selection
uses public development data. This public-object skill does not itself justify omission violations.
Proposed small NF4 q/v LM LoRA: rank8,100 steps max,10-step throughput pilot and1500s total cap;
vision frozen. All hyperparameters are predeclared; actual optimizer steps/loss/weight changes and
adapter reload must be recorded. Training has not yet started on the GPU; preparation is under way.

### First Qwen3.8 no-thinking run, 18 Oct 10:08 local

The new-account Qwen3.8-27B-FP8 run used `enable_thinking=false` and
`--reasoning-parser qwen3`, preserving the speed profile and rules. It processed **40/40** with
**0 technical failures**, scoring **29/40, 0 FP, 11 FN** (10 TP, 19 TN). Total analyzer-evaluation
wall time was **406.664 s**, with **5,441 calls**. These are in-sample development results, not
public API latency. The initial server setup also scored 29/40; the matching total does not establish
identical events or states. No accuracy thresholds were fitted and no production changes were made.

Records: [structured result](../../benchmarks/quality/q38-first-no-thinking-20261018.json),
[pipeline steps](../../benchmarks/quality/q38-no-thinking-pipeline-steps-20261018T050825Z.json),
[qualification gate](../../benchmarks/quality/q38-no-thinking-qualification-20261018T050825Z.json).
The first observed gap was no candidate for **8** misses, and a candidate without an accepted event
for **3**; this is diagnostic localization, not proof of a visual cause. Unrecorded measurements
remain null. Labels were used only by the scorer and focus selection. The score gate failed;
the independent repeat is pending, so the model is **not qualified**. Gemma's first full result
remains pending and no actual GPU training steps have been recorded.


### Per-stage benchmark requested, 18 Oct 10:04 Asia/Qyzylorda

The user requested a benchmark for every pipeline step, emphasizing videos the model misses or cannot process. The structured-only replay tool 	ools/pipeline_step_benchmark.py now reports all 40 videos uniformly for each of q30 ref1/ref2, Qwen32 and the initial Qwen3.8 run (160 recorded evaluations), with a scorer-selected focus list for false negatives and technical failures. No new inference, accuracy change or parameter fitting was performed. Latest record: enchmarks/quality/pipeline-steps-enhanced-models-20261018T050317Z.json; the earlier reduced-export record remains for provenance.

All 160 evaluations processed; technical failures are zero. Scores remain 33/40 (FP1,FN6), 33/40 (FP0,FN7), 32/40 (FP0,FN8) and 29/40 (FP0,FN11). The first observed post-filter gap is no candidate for 6,7,6,9 misses respectively; another 2 Qwen32 and 2 Qwen3.8 misses have candidates without accepted events. These observations do not establish the visual cause. The enhanced q30 export records 17/15 visible=true verification decisions and 14/15 NO_CLIENT_ID proposals suppressed as type_not_promoted; those proposals also occur on 6/7 distinct negative videos, so suppression is not evidence that the disabled type qualifies.

Stage records contain available decode/layout/state/verification/analyzer times, crop boxes, state field coverage, unknown/missing values, suppressed proposal enums, candidates and events. Per-stage calls, rule computation times and final assembly times were not recorded and remain null; source timers may overlap and are not added. Analyzer wall time is not public API latency. Gemma stage reporting will follow its first complete structured export; no Gemma accuracy claim is made here.


### FT candidate protocol and isolation, predeclared 18 Oct 10:05 local

Visual principle: an incomplete, reasoning-prefixed or non-finite structured observation is not reliable visual evidence and must fail explicitly. New strict_model_protocol is default false; only the future Gemma FP8+public-LoRA candidate will enable it. It rejects non-stop completion, reasoning, non-object/prefixed JSON, duplicate keys and non-finite numbers, while preserving the intentional concise stop. No rule/type/score threshold changes; 47 focused local tests passed. Baseline and production profiles remain frozen.

The FP8+adapter candidate is a composite change relative to BF16, so its result will not isolate a causal fine-tuning benefit. Two complete independent 40-video runs and all operational gates remain mandatory. A successful queue exit means collection succeeded, never that the model qualifies. Trainer mounts are being narrowed to an isolated public-data/output sandbox so prior bank scoring reports are inaccessible to the training container. Actual GPU training has not yet begun.


### User pivot to Qwen30 BF16, 18 Oct 10:19 local

The user cancelled Gemma/Qwen3.8 experiments, asked which model achieved33/40, then explicitly dedicated both testH100s to Qwen3-VL-30B-A3B-Instruct BF16 fine-tuning/evaluation. Both old test model containers and incomplete second evaluations were stopped; queued Gemma training/diagnostics were cancelled before any GPU training. A provider instance-stop attempt failed; both new H100s remain RUNNING and are now intentionally reused under the new request. No production model/worker/profile/API change.

Q30 weights on both H100s are pinned to9c4b90e1e4ba969fd3b5378b57d966d725f1b86c; all13shards/62,141,618,016bytes verified against upstream LFS. CPU training runtime verified native Qwen3VLMoe support. Reference33/40 config was extracted from ref1/ref2, which match exactly; fileSHA b236180cd0ab3324c521f9f29a154bbcdb32220129c31bd905931ebdc73e08fe.

Visual principle: annotation-supported public banknote/document presence may improve desk cash/document perception. Train BF16 rank8 q/v LoRA (96modules), frozen base/vision,100steps/1500seconds maximum,10step/300second pilot. MoE fused expert tensors remain BF16; no NF4 claim. Training has1024 public compositions and128 reused public validation examples. KZ bbox note rows supervise cash=true only; unknown ID absence remains masked. Publisher-quad ID/blank/both examples provide supported cash/ID values. Loss applies only to these known boolean value tokens; other four DESK fields are unscored. No supplied bankvideo/labels mounted in training.

The default-off desk_adapter_model flag queries base+adapter in parallel and replaces onlycash/id_or_document; allfour other desk outputs remain from base.56focused tests passed. Additional call cost and two independent full40 results must be measured. This is an in-sample optimization attempt, not a37/40 claim; production requires qualification and user switch approval.

### Qwen30 GPU training in progress, 18 Oct 10:23:45 local

The [structured GPU snapshot](../../benchmarks/quality/q30-training-progress-20261018T052345Z.json)
from bridge job 177 recorded **46/100 completed optimizer steps**,
**3,342,336 trainable parameters**, **1.46–1.50 s/step**, finite gradients and no OOM. Peak allocated
memory was **59,548 MiB** (reserved **60,136 MiB**). The predeclared 100-step/public-data recipe is
unchanged: **12 declared recipe controls, zero tuned to bank-video labels**. This establishes actual ongoing GPU training; loss or step count does not establish
bank-video quality, and no complete adapter or two-run accuracy claim is made.

The main baseline launch initially failed because its pinned image provides `/usr/bin/python3`
without a `python` alias ([runtime record](../../benchmarks/quality/q30-image-runtime-20261018.json)).
The corrected restart is launched, with schema smoke and the two independent base evaluations queued.
Gemma/Qwen3.8 servers and their partial second runs remain stopped by user request; no Gemma GPU
optimizer steps occurred. Production model, worker and profile remain unchanged.

### Qwen30 BF16 training completed, 18 Oct 10:26:34 local

The [training summary](../../benchmarks/quality/qwen30-training-summary-20261018.json) confirms
**100/100 optimizer steps**, **3,342,336 trainable parameters**, **246.748572684 s recorded training
time**, changed adapter weights and unchanged frozen base-weight hashes. The predeclared 10-step
pilot passed in **20.1023 s**, and checkpoints at steps 25/50/75/100 were hash-verified. The recipe
retains **12 declared controls, zero tuned to bank-video labels**; no bank-video labels entered training.

The [execution receipt](../../benchmarks/quality/qwen30-training-execution-20261018.json) records exit 0
and `completed_training_not_qualified`. A final receipt permission problem was
[recovered without restarting training](../../benchmarks/quality/qwen30-training-recovery-20261018.json);
the recovery neither trained again nor changed model bytes. This confirms optimizer work and artifact
change, not accuracy. Trained-adapter loading and the corrected baseline server startup are in progress;
public validation, two independent full-video evaluations and operational qualification remain pending.
Training loss is not a bank-video quality score. Production remains unchanged.

### Fresh Qwen30 H100 base pair, 18 Oct 10:44 local

The [base-model pair](../../benchmarks/quality/q30-baseline-runs-20261018.json) processed 40/40 in
each run with zero technical failures: **33/40, FP1/FN6** and **32/40, FP1/FN7**. Zero false alarms
is mandatory; both runs also fall below 37/40, so the
[qualification gate failed](../../benchmarks/quality/q30-baseline-qualification-20261018T054438Z.json).
These are fresh in-sample measurements with the reference profile, not adapter scores or unseen accuracy.

At parallelism 4, total batch wall times were **223.345 / 217.891 s**; per-video analyzer p50/max
were **17.18/53.76 s** and **18.63/38.08 s**. The analyzer clock starts after semaphore acquisition;
these are not public API 202-to-done latencies. The
[step report](../../benchmarks/quality/pipeline-steps-q30-baseline-pair-20261018T054438Z.json) found
no candidate for all **6/7** misses. This is the first observed gap, not proof of the visual cause;
labels remain scorer/focus-only and no parameters were fitted by the report.

The corrected base server passed [15/15 synthetic checks](../../benchmarks/quality/q30test-smoke-20261018.json).
The trained adapter's [server loaded](../../benchmarks/quality/q30desk-server-20261018.json), but its
[synthetic preflight failed technically](../../benchmarks/quality/q30desk-smoke-debug-20261018T054500Z.json)
with `AssertionError`; debugging continues. The adapter's public validation and video evaluations
have not run. Its 100 completed training steps remain separate evidence, with no video accuracy
gain claimed. Production model, worker and profile remain unchanged.

### Adapter preflight/public component check and baseline load, 18 Oct 10:49 local

The permission wrapper was corrected without changing frozen source, profile or adapter bytes.
The [actual adapter synthetic smoke](../../benchmarks/quality/q30desk-smoke-20261018T054700Z.json)
passed **19/19**, including the production-like desk route and deterministic two-field merge.
The earlier technical preflight failure remains historical evidence; no model retraining was needed.

The [paired public component result](../../benchmarks/quality/q30-public-paired-20261018T054834Z-bf75ffe5.json)
used 128 reused validation compositions per model and 224 annotation-supported cash/ID values:
base **184/224 (82.1429%)**, adapter **219/224 (97.7679%)**. Adapter ID values were **96/96**;
cash values **123/128**, with **3 FP / 2 FN**. Both models processed 128/128 with zero schema failures.
This is reused public validation, not bank-video accuracy, cross-validation, a fresh lockbox or
qualification of omission types. The two full bank-video candidate runs have started; no adapter
video score is available. The mandatory zero-FP, ≥37/40 target remains unmet; fresh baseline scores
remain 33/40 and 32/40, each FP1.

The [baseline six-long staging check](../../benchmarks/quality/q30-baseline-stage-par6-20261018T054619Z.json)
completed **6/6**, zero failures, p50 **43.783 s**, max **74.316 s** from 202 to done. Original-JPEG
validation had zero mismatches/errors. The 285 s completion gate passed; ≤60 s and arrival-order
targets failed. Model/source identities stayed unchanged and temporary stage cleanup completed.
This is internal staging, not public HTTPS. The
[production health record](../../benchmarks/quality/production-health-20261018T054900Z.json) confirms
HTTP 200, public smoke **1/1 in 6.303 s**, and unchanged worker/profile. It is operational evidence,
not a fresh accuracy measurement. No production promotion occurred.

### PERSONAL_ITEMS diagnostic principle declared before inference, 18 Oct 2026

Visual principle: a personal drink container, coat or handbag must be physically recognizable on the cashier's work surface across the ordered proposal window; a bright static patch or fixed cash-desk equipment is not sufficient. The standalone diagnostic uniformly queries every cached PERSONAL_ITEMS proposal from both complete Q30 baseline runs, without label/correctness selection. It uses two independent closed phrasings over original full frames and unresized native desk crops; labels open only in the host scorer after inference ends. It enables no rule and makes no measured accuracy claim before execution. Eight preset resource/sampling controls: 6 full frames, 3 desk crops, 2 phrasings, 128 output tokens, 64 maximum requests, 2 parallel videos, 4 concurrent requests and 150 s tool budget (178 s wrapper ceiling). Proposal half-window inherits the frozen reference verify_window; JPEG quality 88 and decode 2 fps inherit existing utilities. Zero parameters fitted to bank labels; no confidence threshold changed. Object support requires paired kind agreement, recognizable/on-surface/consistent booleans and fixed_equipment=false; touched_or_lifted is recorded but not required. Full pipeline qualification remains separate.

### Contextual safe-key sequence principle declared before inference, 18 Oct 2026

Visual principle: the same physically distinguishable key must remain inserted in the same resolved safe lock after an observed hand withdrawal and cashier departure; an open safe, handle, highlight or customary action is never key evidence. A standalone diagnostic will apply one label-free policy to all 40 cached Q30 base videos: numeric safe-ROI activity and cashier-role departure proposals, before/during/after original full frames, independently localized native safe/lock crops, lossless PNG transport, and two closed sequence questions with explicit unknowns. All failures and proposal overflow remain in the 40-video denominator. This enables no type, changes no frozen source/profile, and makes no accuracy or release claim. Sixteen predeclared controls: retained 2 fps; motion intensity delta 20; active pixel fraction 0.08; activity gap merge 1 s; proposal separation 2 s; at most 2 uniformly spaced proposal windows; fixed six-frame offsets (-2,-1,0,+1,+2,+4 s); lock crop expansion 2; minimum lock crop side 24 pixels; 2 independent sequence phrasings; 900 maximum output tokens; request timeout 45 s; 2 parallel videos; 4 concurrent requests; 240 s tool ceiling; 298 s wrapper ceiling. Persistence support requires two resolved post-departure snapshots at least 0.5 s apart, a seventeenth physical sampling control. Zero controls fitted to bank labels; no learned weight or accuracy threshold is introduced. Host-only labels may be opened after inference solely for a contextual support/unknown/failure breakdown on all 19 negatives, never for selecting windows or tuning this diagnostic.


### Qwen30 DESK LoRA first bank-video run rejected, 18 Oct 10:54 local

The [first adapter benchmark](../../benchmarks/quality/q30-adapted-first-20261018T055500Z.json) processed all40 development videos with0 technical failures but regressed to **30/40, FP3, FN7** (14TP/16TN). The40-video batch took300.630s atpar4 and used7006calls. This is batch time, not individual API latency. Public cash/ID validation improvement did not establish bank-video transfer. The second fixed-profile run continues; the adapter is unqualified and production is unchanged. No parameter or checkpoint selection was changed based on these labels.

The user redirected attention to contextual micro-interactions: a visible key remaining in the lock after hand withdrawal and cashier departure; counting with the client absent; and storage without an observed recount. Standalone diagnostics are being prepared/run uniformly, with unresolved pixels or incomplete temporal coverage remaining unknown. No additional violation type has been enabled.

### Complete Qwen30 DESK pair rejected; contextual diagnostics, 18 Oct 11:00 local

The [complete fixed-profile adapter pair](../../benchmarks/quality/q30-adapted-runs-20261018.json)
processed 40/40 in each run, zero technical failures, but scored **30/40, FP3/FN7 in both runs**.
Batch times were **300.630 / 302.264 s**, with **7,006 / 7,024 calls** at parallelism 4.
These are whole-batch analyzer times, not per-job API latencies. The adapter is rejected; the
mandatory **≥37/40 and zero false alarms in two runs is not met**. The public component gain did
not establish bank-video transfer. No checkpoint, threshold or rule was changed to fit these labels;
production remains unchanged.

The [dense procedure diagnostic](../../benchmarks/quality/q30-procedure-visible-r2-20261018T055000Z.json)
recorded **21 done / 13 failed / 6 abstained** across 40 videos; the negative subset is
**9 done / 7 failed / 3 abstained**. Failures and abstentions do not establish clean negatives, and
continuous visibility is unproven. The
[contextual counting report](../../benchmarks/quality/q30-contextual-counting-r2-20261018T055902Z.json)
has no qualifying positive episode and incomplete negative coverage. Its single diagnostic duration
threshold is **2.8 s**, from the maximum observed negative episode **2.6 s** plus one untuned sample
margin **0.2 s**; no positive labels/times set that threshold. Calibration is incomplete, its gate
failed, and no type is enabled. The base-model safe-key sequence check is still running over
before/contact/release/departure/post observations; no completed support result is claimed.

The [PERSONAL_ITEMS diagnostic](../../benchmarks/quality/q30-personal-items-probe-20261018T060000Z.json)
completed 12 calls but also suppresses proposals on positive videos, so it does not qualify a
pipeline change. All diagnostics remain separate from frozen source/profile and production.


### Independent procedure curriculum and text-only LoRA, 18 Oct 11:15 local

Temporal principle (predeclared in the curriculum before GPU launch): a skipped step is supported only by a linked, completely observed operation; a persistent key needs the same visible key and lock after hand withdrawal and cashier departure. Unknown or occluded observations do not establish an omission. This log entry records the already frozen curriculum after the launch; the source/data/config were frozen before training.

The user requested dozens of procedural examples and actual training. The Russian [40-scenario curriculum](procedure-curriculum-20261018.md) expands to512train/128validation symbolic timelines in32/8disjoint template groups. No bank media, filenames or labels enter training. This teaches reasoning over observations, not additional visual recognition. All640 semantic checks and independent review passed. ManifestSHAa1f16975edee76cdcbc1acc71e93fbc532193ae75650b875aa16777a40c5d7b5. The new Qwen30BF16 rank8 q/v run has11declared training hyperparameters and0tuned to bank labels, fixed100-step cap, budget stop11:15 and hard11:16:30. Partial training remains partial_not_qualified.

A structured GPU snapshot at11:14:27 recorded27optimizer steps and3,342,336trainable parameters, about0.67-0.92s/step. This confirms actual updates, not37/40. Production remains unchanged. The initial transfer232failed before any installation because of a wrapper Python syntax error;233fixed the wrapper without changing frozen data/trainer/config.
