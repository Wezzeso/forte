# Deadline cancellation investigation

Investigation date: 2026-10-17. This is a design record. No inference request or
running-service change was made for this investigation.

## Observed problem and evidence

A18 observed durable jobs become terminal at their deadlines while generation
continued for roughly 110 seconds before the engine's running/waiting gauges
returned to zero. A terminal database row therefore cannot establish GPU abort.

The installed `vllm==0.31.0+cu129` source has cancellation for **both** modes:

- `entrypoints/openai/chat_completion/api_router.py:52` decorates the route with
  `with_cancellation`.
- `entrypoints/serve/utils/api_utils.py:80` races the handler against an ASGI
  disconnect listener and cancels the other task. When the handler returns a
  `StreamingResponse`, that response takes over disconnect handling.
- `v1/engine/async_llm.py:747` catches `CancelledError` and `GeneratorExit`, then
  awaits `abort(q.request_id, internal=True)`.
- `async_llm.py:883` removes the request from the output processor and sends the
  corresponding engine-core abort.

Installed source SHA256 values, obtained without reading logs or media:

| File | SHA256 |
|---|---|
| `entrypoints/serve/utils/api_utils.py` | `ffb5e08c2b6cf6d0402c6ba31d0d05e53eb7e9e8c13b29700e84a4bfada420ba` |
| `entrypoints/openai/chat_completion/api_router.py` | `77f858fe0079b9dd131c26d22641c79d4ac0f2b22688d033863954a8a669d7d3` |
| `entrypoints/openai/chat_completion/serving.py` | `532dadea845c77a5a79011027b9989987c3e32467614e993f439f29c924de002` |
| `v1/engine/async_llm.py` | `ff933ed78aa3195782cf1c6d18c53897adfa5bf920421bd2bcc330f2adc84d1e` |

The matching upstream release sources are the
[route](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/chat_completion/api_router.py),
[disconnect wrapper](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/serve/utils/api_utils.py),
and [engine generator](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/async_llm.py).

`DataOnlyMiddleware` replays the request body once and then delegates to the
original ASGI `receive`; it preserves disconnect messages. The adapter uses an
HTTPX response-stream context even for a non-streaming JSON response. Its
`_bounded` helper cancels and gathers that transport task on deadline/lease loss.
Consequently `stream=false` alone is not an established root cause.

Several independent same-deadline scopes can cancel the same task chain:
worker, decision engine, scheduler submit, scheduler dispatch and adapter. The
scheduler caller can return before its provider task finishes asynchronous
cleanup. Repeated cancellation during cleanup is a possible race, not a proven
explanation of the observed 110-second tail. A CPU-only 30-trial nested-deadline
probe observed the caller return first, but every mocked transport completed its
20 ms cleanup without interruption.

The adapter also retries HTTP transport failures. A read timeout after the
backend accepted a POST has an ambiguous execution outcome; retrying may create
another backend request before the original generation has actually stopped.

## Proposed client change

Implement in A07's adapter, preserving its existing response interface:

1. Give each attempt an opaque request ID, unrelated to job identity or media.
   Record only safe cancellation timings and state, never request content.
2. Treat cancellation as idempotent. Cancel the operation once and retain a
   separate cleanup task which explicitly closes the response. Wait for cleanup
   under a short independent bounded grace period, shielding it from repeated
   parent cancellation. Retain and settle any task which outlives that grace;
   do not silently lose it or claim the GPU is idle.
3. Do not retry a POST after response headers/body were accepted or after an
   ambiguous read failure. Pre-admission connection failures and explicit
   retryable HTTP statuses can retain their bounded policy.
4. Add optional `stream=true` and `stream_options={"include_usage":true}`.
   Assemble SSE privately and return only the same final validated
   `ModelResponse`; do not publish partial decisions. Streaming permits response
   ownership before generation completes and makes per-token progress visible
   to transport timeouts. It still requires an actual abort test.
5. Bound individual event/line bytes, total wire bytes and assembled content
   independently. SSE overhead means the current 512 KiB JSON-body ceiling is
   not automatically a suitable total-wire ceiling. Ignore keepalive comments;
   require valid UTF-8 and bounded JSON events. Accept one stable response ID,
   expected model alias and choice index zero. Reject tools/refusals/unexpected
   deltas. Require an explicit `finish_reason="stop"`, complete terminal stream
   marker and final schema/semantic validation. EOF, deadline, `length`, malformed
   chunks or missing terminal metadata remain failures. Preserve usage and only
   backend-reported numeric metrics; unavailable fields remain null.

Scheduler cleanup should likewise become observable and idempotent. Keep a
dispatch slot occupied until the provider task has settled, and wait for the
specific provider cleanup before reporting a locally drained request. This does
not mean extending a video's successful-processing deadline. Add synthetic tests
for simultaneous cancellation sources, slow async close, close interrupted by a
second cancel, lease loss and scheduler shutdown.

## Decisive controlled test

After root coordinates an exclusive GPU window, start from actual running=0 and
waiting=0. Send a generated synthetic request long enough to still be generating
at cancellation, cancel after admission/first token, and measure the time until
both gauges return to zero. Exercise non-streaming and streaming separately,
plus lease-signal and task-cancel paths. Count every attempt and failure. Then
run one strict known-marker smoke to check recovery. Report only timings,
counters, statuses and token totals. Never infer successful abort from client
exceptions or durable job status alone.

Root authorized two such probes before any patch. Both actual non-streaming
requests were admitted/running and canceled 5–8 seconds afterward. The direct
provider signal path returned `ModelCancelled` after 6.4 ms and actual idle was
observed after 15.5 ms. The nested decision/scheduler/outer shared-deadline path
returned `TimeoutError` after 1.0 ms and actual idle was observed after 216.7 ms;
the scheduler had one active task at caller termination and zero by idle. These
are gauge observations including polling latency. No retries were allowed to
enforce the two-call budget. The earlier 110-second tail was **not reproduced**;
neither SSE nor a cancellation patch is established as necessary by this result.
See `configs/serving/cancellation-validation-before-patch.json` for all safe
measurements. Ambiguous read-timeout retries and multi-job behavior remain
separate cases to investigate before changing the serving transport.

## Four-sequence capacity feasibility

A read-only post-burst probe observed running=0, waiting=0, KV utilization=0,
GPU memory total/used/free of 81920/70829/10327 MiB. Free device memory excludes
the KV cache already reserved by the engine, so it does not directly establish
request capacity. The public model config has 36 layers, 8 KV heads, head
dimension 128. BF16 key/value storage is therefore 147456 bytes per token before
allocator/block overhead; four full 32768-token contexts would use 18 GiB of KV
data. This is a feasibility estimate, not a measured fit.

A coordinated `max_num_seqs=4` trial on this A100 is reasonable after cancellation
is proven. It requires an engine restart and fresh readiness, because setting
worker concurrency to four while the engine remains at two merely adds waiting
requests. Use a separate candidate config; preserve the verified two-sequence
baseline. Keep context, frame count, source resolution, dtype, pixel budget and
8192-token batched prefill limit unchanged. Warm four representative largest
requests, measure memory/preemptions/latency and strict validity, then compare the
same synthetic workload at two and four active sequences. Reject OOMs, invalid
outputs or latency regressions. Only root should coordinate deployment selection
and the later actual five-video measurement.
