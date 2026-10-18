# Gemma 4 serving speed audit — 18 October 2026

Status at 04:55 UTC (09:55 Asia/Qyzylorda). This is a source and structured-metrics audit,
not a benchmark of the proposed settings. No hosts were contacted, no jobs were queued, and no
production or evaluation source was changed by this audit. Labels and media were not inspected.

## Measured bottleneck

The existing BF16 server uses the pinned `vllm/vllm-openai:v0.31.0-cu129` image,
`max_model_len=32768`, `max_num_seqs=16`, `max_num_batched_tokens=4096`, chunked prefill,
prefix caching, processor cache 4 GiB, vision budget 560, and async scheduling disabled.
Startup recorded weights 58.99 GiB, available KV cache 10.7 GiB / 54,642 tokens, and
1.67 maximum full-context concurrency. The latter is not a limit of 1.67 short requests:
14–15 requests were running in the observed workload.

Sources: `tmp/agent-bridge/payload/gemma4_serve.remote.sh`,
`tmp/agent-bridge/out/124-gemma4-startup-status.log`, and the synthetic smoke benchmark
`benchmarks/quality/gemma4-smoke-20261018T043700Z.json`.

| Structured observation | 04:39:54 UTC | 04:44:42 UTC |
|---|---:|---:|
| Running / waiting requests | 15 / 48 | 14 / 48 |
| KV cache use | 91.5% | 70.9% |
| GPU utilization | 94% | 93% |
| Completed requests since server start | 440 | 1,174 |
| Prompt / generated tokens since server start | 363,379 / 19,119 | 914,399 / 55,096 |

Source: `tmp/agent-bridge/out/126-gemma4-progress.log` and
`tmp/agent-bridge/out/136-gemma4-progress.log`. Their 288.016-second difference yields
**2.55 requests/s, 1,913 prompt tokens/s and 124.9 generated tokens/s** across all active
work. These are counter deltas, not per-video latency, full-run wall time or accuracy.
There is no evidence that increasing the already-64-request client cap will help a GPU
that is this busy. Request queue length alone does not prove KV thrashing; collect a
preemption counter before asserting recomputation.

## Best next experiments

The largest plausible speed improvement is LM quantization with increased batch capacity.
It changes numerical computation and cannot be described as accuracy-preserving before
the independent comparisons finish. For unchanged BF16 weights, encoder CUDA graphs are
a concrete optional acceleration that the current command does not enable. Neither has
a measured gain on this workload yet.

| Candidate | Change | Expected mechanism and limit |
|---|---|---|
| Existing FP8 contingency | `--quantization fp8_per_tensor`; existing plan uses seq 32, batch 8192 | Reduce language-matrix storage and use H100 FP8 computation; freed memory may admit more short requests. Actual KV capacity, speed and agreement must be measured. |
| Encoder CUDA graphs | `--compilation-config '{"cudagraph_mm_encoder":true}'` | Capture/replay the vision encoder without changing source pixels, token budget, prompts or weights. Additional graph memory can shrink KV capacity and erase the gain. |
| BF16 resource-only comparison | Keep 4096 batch tokens; test a predeclared higher memory fraction only after the baseline pair | More KV headroom may reduce admission stalls/preemptions. Busy GPU and substantial current batching limit the likely gain; no numerical multiplier is justified. |

For online conversion of BF16 weights, **use `fp8_per_tensor`, not `fp8`**. The latter
requires an already-FP8 checkpoint in this pinned release. Online FP8 uses lower-precision
weights and dynamically scaled activations, without training/calibration data. This is
not fine-tuning, and leaving `--dtype bfloat16` does not turn its matrix operations back
into BF16. Sources: [v0.31.0 online quantization](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/docs/features/quantization/online.md)
and [FP8 configuration](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/model_executor/layers/quantization/fp8.py).

Gemma's pinned implementation leaves the vision tower and connector unquantized when their
dimensions fail its divisibility check; the code specifically identifies vision intermediate
size 4304. Its language model receives the quantization configuration. The same implementation
explicitly supports encoder CUDA graphs and rejects externally supplied `image_embeds`.
Sources: [Gemma multimodal implementation](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/model_executor/models/gemma4_mm.py)
and [Gemma language implementation](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/model_executor/models/gemma4.py).

Encoder graph capture defaults off and is separate from `compile_mm_encoder`; do not substitute
the latter without checking support. The graph manager packs items into captured token budgets;
warm-up and allocator profiling must include the actual selected configuration. Sources:
[compilation configuration](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/config/compilation.py)
and [encoder graph manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/v1/worker/encoder_cudagraph.py).

## Why caching and batching alone do not promise a large gain

`VLMClient.chat()` places the common SYSTEM message first. Most analyzer requests then put
the image before the task text. Different images change multimodal prefix hashes; identical
task text after those images does not make the preceding KV blocks reusable. Prefix caching
is already enabled. Moving task text before images changes the prompt sequence and is outside
this unchanged-prompt audit. Source: [pinned prefix-cache design](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/docs/design/prefix_caching.md).

The 4 GiB processor cache avoids repeated preprocessing. GPU encoder-output caching is a
separate, hash-indexed mechanism. Increasing the CPU cache does not enlarge the GPU encoder
cache. In this release scheduler encoder compute/cache budgets are initially derived from
`max_num_batched_tokens`, then adjusted for the largest multimodal item. Sources:
[multimodal configuration](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/config/multimodal.py),
[scheduler configuration](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/config/scheduler.py)
and [encoder cache manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/v1/core/encoder_cache_manager.py).

Most state calls use a different original frame or ROI. Verification resizes full images to
768 whereas state calls allow 896; verification crops use a different resize cap as well.
Thus even overlapping source frames need not produce identical encoded image inputs. Exact
cache-hit fractions are unmeasured. Avoid approximate image matching, static-region answer
reuse, lower resolution, fewer frames, shorter JSON fields, or cross-video answer caches in
this experiment: those change the evaluated perception or decision inputs.

Larger prefill batches can increase throughput, but consume memory and can increase decode
latency. Generic recommendations for batches above 8192 do not establish feasibility for
this 58.99-GiB BF16 model with hybrid attention. Async scheduling stays disabled in both
baseline runs; changing it also changes in-flight memory requirements. Source:
[pinned optimization guide](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/docs/configuration/optimization.md).

## Minimal comparison and release gate

1. Finish both unchanged BF16 runs. Do not restart their server to trial a serving flag.
2. On an idle staging GPU, preserve original image bytes, token budget 560, prompts, schemas,
   sample selection, deterministic rules and thresholds. Keep each serving profile immutable
   across its two independent full runs. Record all argument/image/source digests.
3. Before real inference, rerun synthetic strict-JSON cases and distinct 14/18/32-image requests.
   For encoder graphs, include varied image dimensions and both cache-hit and cache-miss requests.
4. Record startup KV capacity, peak memory, preemptions, cache hit counters, request throughput,
   prompt/generated tokens and analyzer stage times. A GPU utilization increase alone is not a win.
5. Compare fields/events/verdicts label-free against both BF16 runs, then score separately.
   BF16 and FP8 observations are different candidates and must never be pooled as repeated runs
   of one model configuration. Batch/graph changes can also expose the measured baseline noise.
6. Only a candidate with at least 37/40 and zero false alarms in both full runs can proceed to
   the existing evidence/schema, six-longest-clips and production approval gates. Failed requests
   remain failures. Report any speed/quality tradeoff directly.

Tuned parameters in this audit: **0**. Implemented experiments: **0**. Proven Gemma throughput
improvement from these proposals: **none yet**. No score or latency qualification is claimed.
