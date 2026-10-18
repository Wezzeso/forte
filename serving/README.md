# Private local model serving

Implemented: a version and digest pinned GPU service, local-snapshot hash
verification, representative synthetic warmup, warm readiness, a bounded fair
clip scheduler and propagation of cancellations, deadlines and provider errors.

**Current status: real GPU startup and strict synthetic inference passed.**
The authorized Brev A100 80 GB PCIe uses driver 565.57.01. All 15 model artifacts
(17,545,914,364 bytes) passed checksum verification. The CUDA 13 default image
failed an actual tensor operation, including documented forward compatibility
mode. The same current vLLM release's CUDA 12.9 variant passed BF16 computation,
representative model warmup, and the full candidate-schema/known-marker smoke.
`configs/serving/startup-validation.json` records that observed startup; it is
an evidence record and cannot make a future process ready. Whole-video latency,
bank-event quality and five-job capacity require the separate A18 measurements.

## Exact interface

The internal model service is `model:8000`; `model-1`, `model-2`, etc. may be
separately configured replicas on other dedicated GPU devices. The worker uses:

* `POST /v1/chat/completions`, model alias `Qwen3-VL-8B-Instruct`.
* JPEG data URIs only, at most 16 images per clip request; no audio or raw video.
* `structured_outputs: {"json": <schema>}`; the removed `guided_json` API is not used.
* `GET /readyz` **and** `GET /health` must succeed before claiming GPU jobs.
* `GET /metrics` is private; inspect available metric names on the pinned image.

`/health` alone can succeed during warmup. `/readyz` remains 503 until the
launcher has verified local files and completed all configured synthetic warmup
requests. The Docker healthcheck also checks the live engine. No model port is
published. vLLM's API-key option does not protect every endpoint, so an internal
network is the security boundary. The supplied middleware prevents remote media
fetches and local file paths; it does not replace network isolation.

## Reproducible startup

`configs/serving/baseline.json` is the source of serving limits. The stable
vLLM 0.31.0 release was verified from the upstream GitHub release API on
2026-10-17. The Docker Registry returned a real multi-platform manifest digest
recorded in `configs/serving/image-lock.json`. Actual runtime checks are separate
in `gpu-validation.json` and `startup-validation.json`. The image reference is:

```text
vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb
```

Use A17's canonical `deploy/compose.gpu.yaml` overlay and
`deploy/Dockerfile.model`, together with `deploy/compose.yaml`. The derived image
removes unused TorchCodec because the upstream optional decoder links CUDA 13
libraries. This serving path accepts JPEG inputs; MOV decoding belongs to the
CPU worker. Do not substitute the unmodified base image for this validated build.
Set `FORTE_MODEL_SNAPSHOT` to the complete local snapshot. Its model, tokenizer
and processor all use revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
Mount source, serving config, A07's `models/manifest.json`, and weights read-only.
The image launches `python3 -m serving.launch` (this image has no `python` alias). Before starting the engine it
runs A07's stdlib verifier against every manifest artifact. Missing weights,
hash mismatches, wrong vLLM version and inference failures leave readiness false.

The Docker network is internal, Hub/Transformers offline modes are enabled,
telemetry is disabled, and local files are supplied directly. Pre-stage the
image and dependencies with authorized networking before the offline test. A
startup that needs to download another artifact must fail the offline check.
The first startup may compile CUDA kernels in the writable cache volume. The
overlay sets both `FLASHINFER_WORKSPACE_BASE` and `CUDA_CACHE_PATH`. Warm
replicas survive CPU API/worker restarts independently. Restarted GPU replicas
must complete preflight and warmup again before accepting clip work.

The initial BF16 config uses one GPU, context ceiling 32768, two simultaneous
sequences, 8192 batched prefill tokens, 0.85 GPU memory utilization and processor
`max_pixels=1048576`. These ceilings passed representative synthetic shapes on
the observed A100; they do not establish another GPU's capacity. Multimodal
memory profiling remains enabled.
Warmup exercises 1×448², 8×672² and 16×1024² JPEG inputs and simultaneous largest
requests, then A07's real adapter and A08's full decision schema/semantic validator
using three generated synthetic JPEGs and a required visible marker candidate.
Success only covers those representative
shapes. Per-request frame-ID enums may still require new grammar compilation;
monitor further JIT compilation and confirm real clip shapes with A18 before
publishing capacity.

The server uses xgrammar with `disable_any_whitespace=true`. Actual strict JSON
smoke exposed whitespace runaway before this setting; truncated outputs remained
failures. Per-request numeric metrics are enabled without request/output logging.

If a shape fails or OOMs, keep readiness false, record the failure, and jointly
adjust memory/request budgets with A06/A07/A18. Do not silently drop frames or
convert failures into normal results. Preserve the baseline when experimenting
with a selected adapter, smaller checkpoint or quantization.

## Fair dispatch and lease integration

`serving.scheduler.FairScheduler` is a transient clip-execution queue. A14's
PostgreSQL leases remain the only durable queue. Instantiate it once in each
independent worker process, then wrap the A07 provider:

```python
from serving.scheduler import FairScheduler, ScheduledProvider

async with FairScheduler(concurrency=2, max_inflight_per_job=2) as scheduler:
    provider = ScheduledProvider(local_model_provider, scheduler)
    # Same infer signature as A07; clip.job_id is the fairness key.
    response = await provider.infer(
        clip, prompt=prompt, response_schema=schema,
        deadline=asyncio.get_running_loop().time() + job.remaining_seconds(),
        cancel_event=lease_lost,
    )
```

The wrapper enforces 64 pending calls in total and 16 per job by default, FIFO
within a job and round robin across jobs. Expired queued calls never reach the
provider. `QueueFull` means the transient queue did not accept the request; the
worker may retry while its durable lease/deadline permits. A lost lease should
set the cancellation event or call `await scheduler.cancel_job(job_id)`. No
transaction stays open during inference. Provider OOMs, errors and timeouts
propagate; this layer never manufactures results. Cancellation of an HTTP call
is propagated to the provider. CPU tests cover cancellation and lease loss.
Two actual synthetic requests also exercised a direct cancellation signal and
nested shared deadline: running/waiting gauges returned to zero within observed
15.5 ms and 216.7 ms respectively. See
`configs/serving/cancellation-validation-before-patch.json` and
`serving/CANCELLATION.md` for scope and the still-unexplained earlier multi-job
tail; a database terminal state alone does not prove GPU abort.

vLLM continuously batches concurrent requests; the scheduler adds no delay to
fill a batch. For one worker/replica, the initial two-call limit bounds a
microbatch. Across multiple worker processes, account for the aggregate number
of connections and in-flight calls; per-process limits are not a global queue.

Add replicas only after measured demand and quality/latency checks. Each replica
must have its own assigned GPU, separate warm readiness and baseline config.
Do not use `docker compose --scale model=N` with the single default GPU ID:
that would load duplicate models on the same device. A17 must clone the service
with explicit devices/endpoint names. Keep training on separately assigned GPUs.

## Observed tuning and remaining measurement

The isolated synthetic comparison in
`configs/serving/synthetic-concurrency-validation.json` used 16 generated 672²
frames, all 19 production fact definitions, two requests per condition and one
trial. All eight responses passed strict schema and semantic validation.

| Frame IDs | Per-job in flight | Pair elapsed | Generated tokens, pair total |
|---|---:|---:|---:|
| Canonical | 1 | 53.565 s | 3978 |
| Canonical | 2 | 26.211 s | 3334 |
| Compact aliases | 1 | 25.336 s | 1814 |
| Compact aliases | 2 | 14.547 s | 1814 |

The recommended worker overrides are `FORTE_COMPACT_FRAME_IDS=true` and
`FORTE_MAX_INFLIGHT_PER_JOB=2`, keeping `FORTE_MAX_CONCURRENT_CLIPS=2` and the
4096 output-token ceiling. Aliases restore exact canonical source IDs before
decision publication. Compilation time is included: fresh raw-ID enums compile
per request, while compact aliases can reuse the grammar. Different generated
source identities and a single trial limit this comparison. It is a local tuning
observation, not an event-accuracy or sustained-throughput claim.

Run the synthetic checks from the repository root:

```text
python -m unittest discover -s serving/tests -v
python -m serving.launch --print-command
```

The tests cover distinct-job fairness, in-flight limits, bounded queue retention,
deadline expiry, cancellation/lease loss, provider failure propagation, private
readiness and media-fetch rejection. They use synthetic callbacks only.

The actual hardware, image, checksum and startup records are stored under
`configs/serving/`. A17 additionally verified an internal Docker network, no
published model ports, blocked external TCP with no payload, and successful
same-container synthetic inference. Visual-token counts remain unknown unless
the backend exposes them. A18 measures uncached end-to-end work and a five-job
burst; those results must establish the full-video release budget.

Primary references used for compatibility (accessed 2026-10-17):

* [vLLM official v0.31.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)
* [Versioned serving CLI](https://docs.vllm.ai/en/v0.31.0/cli/serve/)
* [Supported models](https://docs.vllm.ai/en/stable/models/supported_models/)
* [Structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/)
* [Multimodal media restrictions](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
