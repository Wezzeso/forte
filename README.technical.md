# Forte video review

Visual evidence review for the ForteBank datathon: a Russian review UI, authenticated asynchronous video API, durable leased jobs, local vision-model interface, source-timestamp sampling, deterministic rule grounding, original JPEG evidence, and evaluation/training tools.

**Live release (18 Oct 2026):** the old-account A100 runs local **Qwen3-VL-30B-A3B-Instruct** (`q30`) with worker `forte-review:speed2-ede4958c` (`sha256:26dc1184…f4bc`). The speed profile enables `parallel_layout`, `verify_concise` and `job_priority`; worker polling is 0.1 s. The worker release preserved the model, API and frontend. Its one-step rollback is `/home/ubuntu/forte/deploy/releases/speed-20261018-ede4958c/rollback.sh`. See the [release and experiment log](docs/integration/agent-log.md).

**Measured accuracy remains below 37/40.** In the [speed-program benchmark](benchmarks/quality/speed-program-20261018.json), two unchanged-reference runs each scored **33/40** (1 false alarm / 6 misses, then 0 / 7); two runs of the now-live speed profile each scored **32/40** (1 / 7). All four processed 40/40 with zero technical failures. These are results on the same 40 unique development videos, in sample. The historical **34/40** belongs to an [earlier qualification run](benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json). Unseen accuracy and event-type correctness remain unverified. A new model requires two runs at ≥37/40 with zero false alarms, service validation and user approval before switching production.

**Current experiment:** the user stopped Gemma/Qwen3.8 testing and redirected both test H100s to
**Qwen3-VL-30B-A3B-Instruct BF16** fine-tuning and evaluation. Production remains unchanged.
Both test hosts have the pinned base weights: **13 shards / 62,141,618,016 bytes**, verified against
upstream LFS ([main host](benchmarks/quality/q30-main-download-20261018.json),
[second host](benchmarks/quality/q30-training-download-20261018.json)). The public-data desk LoRA
uses 1,024 training compositions and 128 reused public validation examples; bank-video labels are
scorer-only. The recipe has **12 predeclared controls, zero tuned to bank-video labels**.
**GPU training completed 100/100 optimizer steps** at **10:26:34 local**, with **3,342,336 trainable
parameters** and **246.749 s recorded training time**. The adapter changed while frozen base-weight
hashes stayed identical. The 10-step throughput pilot passed in **20.102 s**; checkpoints at
25/50/75/100 steps were hash-verified. See the [training summary](benchmarks/quality/qwen30-training-summary-20261018.json),
[execution receipt](benchmarks/quality/qwen30-training-execution-20261018.json) and
[permission-only recovery](benchmarks/quality/qwen30-training-recovery-20261018.json), which did not
restart training. The adapter is **trained, not qualified**. The public component comparison below
is complete; its two independent 40-video candidate runs have started without a bank-video score yet.
Training loss is not a quality score.

On **128 reused public validation compositions**, the
[paired comparison](benchmarks/quality/q30-public-paired-20261018T054834Z-bf75ffe5.json) scored
**184/224 (82.14%)** supported cash/ID values for the base and **219/224 (97.77%)** for the adapter.
The adapter got **96/96** supported ID values and **123/128** cash values, with **3 false-positive
cash values and 2 misses**; both models processed all examples with zero schema failures.
This is reused public component validation, not bank-video accuracy, a fresh lockbox or permission
to enable a violation type. The ≥37/40, zero-false-alarm requirement remains unmet.

The [fresh H100 base-model pair](benchmarks/quality/q30-baseline-runs-20261018.json), using the
reference profile, processed all 40 videos in each run with **zero technical failures**:

| Base run | Correct | False alarms / misses | Batch wall time, parallelism 4 | Analyzer p50 / max |
|---|---|---|---|---|
| q30base20261018a | 33/40 | 1 / 6 | 223.345 s | 17.18 / 53.76 s |
| q30base20261018b | 32/40 | 1 / 7 | 217.891 s | 18.63 / 38.08 s |

**Zero false alarms is mandatory. Both runs failed that requirement and the ≥37/40 score gate**
([qualification](benchmarks/quality/q30-baseline-qualification-20261018T054438Z.json)). The
[step report](benchmarks/quality/pipeline-steps-q30-baseline-pair-20261018T054438Z.json) found no
candidate for all 6/7 misses; this is the first observed gap, not a proven visual cause. These are
in-sample base-model results. Analyzer timings exclude semaphore wait and are **not public API 202 → `done` latency**.

The staging flag `desk_adapter_model` defaults to an empty value. When selected, it replaces only
desk `cash` and `id_or_document` outputs with adapter observations, preserving the other four base
desk fields. The same evaluation harness/profile pair is installed on
[both](benchmarks/quality/q30-eval-install-plp5me7vz-20261018.json)
[hosts](benchmarks/quality/q30-eval-install-xw5720fii-20261018.json); no candidate has qualified.
The main Qwen server's initial startup failed because the pinned image exposes `/usr/bin/python3`
without a `python` command; the [runtime check](benchmarks/quality/q30-image-runtime-20261018.json)
identifies the path. The corrected base server passed [15/15 schema checks](benchmarks/quality/q30test-smoke-20261018.json)
before the runs above. The [adapter server loaded](benchmarks/quality/q30desk-server-20261018.json).
Its initial permission-wrapper failure was corrected without changing source, profile or adapter
bytes; the [actual adapter smoke passed 19/19](benchmarks/quality/q30desk-smoke-20261018T054700Z.json),
including the two-field adapter route and preservation of the four base fields. Current details:
[session record](benchmarks/quality/session-progress-20261018.json).

A separate [baseline staging check](benchmarks/quality/q30-baseline-stage-par6-20261018T054619Z.json)
completed **6/6** longest clips, zero failures, p50 **43.783 s**, max **74.316 s** from 202 to `done`.
Original-JPEG checks found zero mismatches/errors. It met the 285 s completion gate, but **missed
the ≤60 s and arrival-order targets**. Source/model identity stayed unchanged and staging cleanup
completed. This used internal staging transport, not public HTTPS. The
[production health check](benchmarks/quality/production-health-20261018T054900Z.json) returned
HTTP 200 and **1/1** public smoke in **6.303 s**, with the production worker/profile unchanged;
one smoke is not a new accuracy or full-set latency measurement.

Local model tests use the speed profile. Old-account H100 results are in the
[initial model benchmark](benchmarks/quality/qwen-models-20261018T0422Z.json); the first new-account
no-thinking Qwen3.8 result is in its [separate benchmark](benchmarks/quality/q38-first-no-thinking-20261018.json):

| Candidate | Completed runs | Correct | False alarms / misses | Processing failures |
|---|---|---|---|---|
| Qwen3-VL-32B-Instruct-FP8 | 1 | 32/40 | 0 / 8 | 0 |
| Qwen3.8-27B-FP8, initial server setup | 1 | 29/40 | 0 / 11 | 0 |
| Qwen3.8-27B-FP8, thinking disabled | 1; repeat interrupted | 29/40 | 0 / 11 | 0 |
| Gemma 4 31B-it, initial integration | 1; repeat interrupted | 26/40 | 1 / 13 | 0 |

Each completed Qwen run processed all 40 videos and none qualified. The initial Qwen3.8 server observed
at collection had no explicit disable-thinking or reasoning-parser flags. The new-account run used
`enable_thinking=false` and `--reasoning-parser qwen3`; it again scored **29/40**, with **406.664 s**
total analyzer-evaluation wall time and **5,441 calls**. Its [score gate failed](benchmarks/quality/q38-no-thinking-qualification-20261018T050825Z.json),
and the user interrupted its independent repeat. This is in-sample accuracy, not public API latency.
Historical serving attestation and weight-byte digests are incomplete in the initial tests.
No candidate has replaced the production model. The old test host was
[deleted and verified absent](benchmarks/quality/old-test-cleanup-20261018T0423Z.json); production stayed running.

Gemma 4 and the new-account Qwen3.8 setup each passed **15/15 synthetic schema checks**
([Gemma smoke](benchmarks/quality/gemma4-smoke-20261018T043700Z.json),
[Qwen3.8 smoke](benchmarks/quality/q38-smoke-20261018.json)). The [first Gemma run](benchmarks/quality/gemma4-first-run-20261018.json)
processed **40/40**, scoring **26/40, 1 false alarm, 13 misses, zero technical failures**, with
**1,911.778 s** total analyzer-evaluation wall time. A [structured geometry audit](benchmarks/quality/gemma4-layout-distribution-20261018T050927Z.json)
raised a strong coordinate-order hypothesis (YX versus expected XY), which was not fixed or re-tested
before the user stopped the experiment. This measures the tested integration; it does not establish
Gemma quality under a corrected adapter. Both model servers and partial second runs were stopped;
neither model completed two qualifying runs. Gemma GPU training was cancelled before any optimizer step
([stop receipt](benchmarks/quality/user-stopped-gemma4-20261018.json), [session record](benchmarks/quality/session-progress-20261018.json)).

The [pipeline-step tool](tools/pipeline_step_benchmark.py) audits recorded structured outputs from
decode → layout → states → rules/candidates → verification → final result. Its
[enhanced step report](benchmarks/quality/pipeline-steps-enhanced-models-20261018T050317Z.json)
covers **4 runs / 160 video evaluations, all processed, zero technical failures**:

| Run | Correct | False alarms / misses | Misses with no candidate | Misses with candidate but no accepted event |
|---|---|---|---|---|
| q30 ref1 | 33/40 | 1 / 6 | 6 | 0 |
| q30 ref2 | 33/40 | 0 / 7 | 7 | 0 |
| Qwen3-VL-32B q32a | 32/40 | 0 / 8 | 6 | 2 |
| Initial Qwen3.8 q38a | 29/40 | 0 / 11 | 9 | 2 |

The [first no-thinking Qwen3.8 step report](benchmarks/quality/q38-no-thinking-pipeline-steps-20261018T050825Z.json)
adds a separate completed 40-video run: **29/40, 0 false alarms, 11 misses, zero technical failures**;
**8** misses had no candidate and **3** had a candidate without an accepted event.

These are **first observed gaps, not proven causes**: no candidate can reflect earlier perception or
rule coverage; a candidate without an accepted event does not establish the rejection reason.
The report retains recorded box/state coverage, timings and verification decisions. Fields not
recorded remain `null`; overlapping stage times must not be added, and analyzer wall time is separate
from public 202 → `done` latency. Labels serve only the scorer and diagnostic focus selection;
the tool performs no inference or parameter fitting. Suppressed `NO_CLIENT_ID` proposals also occurred
on negatives, so this audit does not qualify the disabled type.

Identical reference runs agreed on **38/40 verdicts**, **37/40 event sets** and only **1/40 exact layouts** (overall state-field agreement **98.88%**). This variation limits conclusions from a one-point score change. Change gating and layout consensus remain disabled because they did not qualify. Dense omission probes failed the negative-video gate; protected types stay disabled.

**Measured speed:** isolated staging, one upload at a time, 202 accepted → first `done`: p50 **8.367 s**, p95 **22.534 s**, max **27.173 s**, with **28/40** under 10 s. The previous worker measured 9.689 / 27.817 / 32.899 s. The speed release's public HTTPS check completed **6/6** longest clips, p50 **64.050 s**, max **101.244 s** from 202 → `done`; including uploads the run took **144.830 s**. These measurements are in the [same benchmark](benchmarks/quality/speed-program-20261018.json). The all-videos ≤10 s and six-parallel ≤60 s goals remain unmet. Architecture, norms and limitations: [docs/fast-analyzer.md](docs/fast-analyzer.md).

The old-account shutdown task previously scheduled for **11:01:25 Asia/Qyzylorda** is now **Disabled**, with user approval; see the [production operations record](benchmarks/quality/production-operations-20261018T0419Z.json). The billing page showed **USD 14.53 at 09:19 local**, about **7.158 A100 hours** at USD 2.03/h if that host were the only charge: approximately **16:28 local**, before storage, other hosts and reporting lag. This does not fund production until 18:00. A separate [new-account guard](infra/new_account_budget_guard.py) uses isolated credentials, validates the exact test-host identity and becomes due no later than **18:00 local**, or earlier at its configured estimated budget limit; an operator or scheduler must invoke it.

The isolated new account now has test host `forte-new-h100` / `plp5me7vz` at USD 3/h. Its
[bound guard dry-run](benchmarks/quality/new-account-budget-guard-20261018T0423Z.json) passed with
a **17:55 local** cutoff, USD 50 ceiling and USD 2 reserve. With user approval, Windows task
`Forte-Brev-new-budget-stop-plp5me7vz` is now installed, **Ready and enabled**, for **18 Oct at 17:55 local**.
The task requires the user's PC and WSL to be available at that time; provider stop completion and
remaining storage charges still require verification.

**Public-data training (18 Oct):** both fits and their single frozen sealed attempts **completed**: RF-DETR Small performed 2,752 updates and compact crop-VLM LoRA 80. RF sealed ID recall was **13/16 (81.25%)**, below the 95% target, with six banknote false positives; compact crop macro accuracy was **94.53125%**, also below 95%. These remain inactive research artifacts. The 40 bank clips stayed outside their fitting/calibration. Source/class confounding, small grouped support and known-box crops limit transfer claims. The 100% target and faster processing remain goals. See [the result ledger](docs/integration/public-training-results-20261018.md) and [methods/accuracy/time benchmarks](docs/integration/accuracy-methods-benchmarks-20261018.md).

Both trained artifacts are [staged and hash-verified in a read-only versioned host release](infra/public-trained-candidates-release-verification-20261018.json), with activation disabled and failed quality gates retained; live configuration was not changed.

The [final machine-readable benchmark](benchmarks/quality/accuracy-final-benchmark-20261018.json) binds the bank regression, public-model tests, fresh API timing and staged-artifact receipts.

Open [hackathon.wezeso.me](https://hackathon.wezeso.me). Its runtime bootstrap sends authenticated uploads/polling to the [direct API gateway](https://forte-review-wd88a7vvt.gobrev.dev), with CORS restricted to the frontend origin. Use that direct gateway for jury scripts too: the original same-origin proxy still has an unresolved large-upload failure. The original [failure-inclusive GPU measurements](benchmarks/performance/gpu-performance-findings.json) and [later reused-input trial](benchmarks/quality/trials/staged-concise/paired.json) retain their separate image/profile bindings.

The separate [provided-label development diagnostic](benchmarks/quality/proposed-train-provided-labels.json) uses six proposed-train clips: one completed response disagreed with its provided binary label, and five attempts remained failed/unprocessed. Those labels have no completed human review, and this development check is not held-out accuracy.

The previous worker `sha256:8c8b92550c5dfc6702efbe266773ea4bfd86639b9f5edd8e86e6a0e3e45b6a3d` and profile `993d2f22…` are the speed release's rollback targets. Their [earlier qualification identities](benchmarks/quality/accuracy-upgrade-rollout-20261018.json), [01:03 UTC bindings](infra/public-training-final-postload-bindings-20261018.json), frontend overlays and earlier timing records remain historical evidence. They do not describe the current speed worker/profile. Specialists and all six protected types remain disabled.

The historical [source/operator bundle](dist/forte-release-source-20261018-final-trained.zip) has a [manifest](dist/forte-release-source-20261018-final-trained.manifest.json) and [verification record](dist/forte-release-source-20261018-final-trained.verification.json). It contains public source/operator documents, without private footage, weights or private state, and predates the speed release. The earlier `forte-release-source-20261018-accuracy-upgrade.zip` and other source ZIPs are also historical snapshots. Exact RF and compact fitting sources are preserved separately in the [training ledger](docs/integration/public-training-results-20261018.md).

## Start locally

Use the repository root as the working directory. Python 3.12–3.14 is supported for CPU components. Install [uv](https://docs.astral.sh/uv/) and put FFmpeg/FFprobe on `PATH` (`FORTE_FFPROBE` can select the probe binary). Node 22.12+ is required to build the UI.

```powershell
uv sync --locked
npm --prefix web ci
npm --prefix web run build
```

Set a private API key and the authorized private model address in each API/worker shell. The example key below is a placeholder for local development; do not deploy it.

```powershell
$env:FORTE_API_KEY = "replace-with-your-private-development-key"
$env:FORTE_DATA_DIR = "var/forte"
$env:FORTE_MODEL_BASE_URL = "http://127.0.0.1:8001/v1"
uv run --locked --no-sync uvicorn forte.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and enter the key. In a second shell with the same configuration, first verify model readiness, then run the independent worker:

```powershell
uv run --locked --no-sync python -m forte.worker --check
uv run --locked --no-sync python -m forte.worker
```

The worker requires real local weights and the private model service described in [serving/README.md](serving/README.md). Missing model readiness remains a startup dependency. SQLite is a local-development fallback; deployed workers/API share PostgreSQL through `FORTE_DATABASE_URL`, uploads and result storage. The API process loads no model or media decoder. `/healthz` checks liveness; `/readyz` checks components.

An authorized local operator must verify the footage's banking cash-workstation scope before setting `FORTE_TRUSTED_SCENE_CONTEXT=bank_cash_workstation`. The default leaves that applicability fact unclear; the model must not infer bank/procedure authority from appearance alone. This setting does not establish any specific visible action.

Run from a full checkout: the worker uses the reviewed `rules/` catalog, root `serving/` package, configs and model manifest alongside `src/forte`. A wheel alone is insufficient for this deployment layout. GPU-container/TLS startup belongs to [the deployment runbook](docs/gpu-provisioning.md) when its actual commands are verified.

## Jury API

Use `.mov` in multipart field `file`; both routes require `X-API-Key`. Upload returns HTTP 202 with a job ID. Poll returns `processing`, `done` or `failed`. Completed verdicts are exactly `violation` or `no_violation`; each event has its type, source time, cautious description/source and 1–3 raw-base64 JPEGs. The initial configurable upload cap is 256 MiB.

```powershell
curl.exe -H "X-API-Key: $env:FORTE_API_KEY" -F "file=@C:/path/to/authorized.mov" http://127.0.0.1:8000/api/v1/analyze
curl.exe -H "X-API-Key: $env:FORTE_API_KEY" http://127.0.0.1:8000/api/v1/jobs/JOB_ID
```

See [canonical contracts](contracts/README.md), [organizer requirements](docs/requirements.md), and [API operator examples](docs/submission/api-and-operations.md). A technical/visibility failure cannot become a clean negative result.

## Verify

```powershell
.venv\Scripts\python.exe -m pytest tests rules/tests serving/tests training/test_training.py training/test_alignment.py annotations/test_workflow.py annotations/test_rebase.py qa/system -q
.venv\Scripts\python.exe -m unittest discover -s scripts -p test_reviewed_split.py
.venv\Scripts\python.exe scripts/audit_data.py self-test
```

Synthetic integration uses an actual generated MOV through the API, leased store, decoder, sampling, injected test-only observations, decision/rule logic, refinement, original-source JPEG assembly and immutable polling. Positive/negative/unclear/provider-failure paths are covered. This proves software integration, not learned detection or legal correctness on supplied recordings. [Observed local HTTP reports](qa/system/OBSERVED.md) distinguish transport measurements from model analysis.

For an authorized external generated-media smoke during an available GPU window, the key comes from the environment and no returned frames or observations are printed:

```powershell
.venv\Scripts\python.exe qa/integration/public_smoke.py --base-url https://forte-review-wd88a7vvt.gobrev.dev --report qa/integration/public-smoke.json
```

## Review, evaluation and training

[Annotation guide](docs/annotation-guide.md), [event taxonomy](docs/event-taxonomy.md), [legal register](docs/legal-source-register.md), and [evaluation guide](evaluation/README.md) define the review process. Start the private human workbench with:

```powershell
.venv\Scripts\python.exe -m annotations.cli serve --port 8765
```

It binds to loopback; private media/annotations must remain local. Content-dependent review requires a local human or an explicitly local model inside the service. A cloud coding assistant does not view supplied recordings or evidence.

The authorized automated experiment uses frozen **provisional** assignments (28 train / 6 validation / 6 final-check contents), independent organizer binary labels and provenance-bound machine supervision. Manual annotation is not a prerequisite for this track. A20 independently pinned the files; fresh exposure coverage and the explicit serialized compute release remain enforced. Machine groups and teacher labels are not human-reviewed ground truth. See [the automated integration record](docs/integration/automated-experiment.md).

W1 trained a temporal head on frozen encoder features; it is not a Qwen or encoder fine-tune. [Optional training tooling](training/README.md) uses a separate Linux/Python 3.12 environment. The exact training Dockerfile inherits the tested CUDA 12.9 serving stack and adds hashed PEFT/accelerate packages; the root `--extra training` lock keeps CPU Pillow independent. Actual optimizer/checkpoint/validation proof must precede candidate selection. The active Qwen baseline stays selected until a concrete candidate passes its applicable checks. Actual W1 completed 336 optimizer steps, but matched 2/6 supplied validation labels versus 3/6 for the untrained comparator, so it was not selected. Parameter updates are not proof that every branch learned. Cached-head repeat matches are separate from fresh upload-to-result timing; see `qa/integration/actual-w1-aggregate.json`.

## Release records

- [Acceptance and blockers](docs/integration/acceptance.md), [ownership](docs/integration/ownership.md), [decision register](docs/integration/decisions.md), [PC control lease](docs/integration/pc-control.md).
- [Version provenance](docs/integration/dependency-versions.md), exact `uv.lock` / `web/package-lock.json`, [component licenses](docs/submission/licenses.md), and [model artifact manifest](models/manifest.json).
- [Forte brand research](docs/forte-brand-research.md), [UI guidance](docs/forte-ui-guidelines.md) and [visual QA](docs/visual-qa.md). Official web-brandbook rules and team design choices have separate provenance; the UI uses locally packaged Open Sans and a plain-text brand label.
- [Submission checklist](docs/submission/checklist.md) and [five-minute pitch](docs/submission/pitch.md).
- [Reproducibility and optional review](docs/submission/reproducibility-and-review.md), including separate source/model/checkpoint/repeat proofs and private operating boundaries.

Preserve raw inputs. Do not use audio, face identification, hidden banking-system facts, jury training data, filename labels or external AI processing. `.gitignore` and the deployment exclusions keep organizer media, labels, review records, credentials, weights and runtime state out of public source packages. Review the actual image/mount/network boundaries before claiming locality.
