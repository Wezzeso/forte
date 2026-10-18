# Performance evidence

Run from the installed project environment. Keep the API key in `FORTE_API_KEY`;
the clients never persist it, source paths, job IDs, response bodies, captions,
predictions, evidence images, or content hashes. Reports contain technical
measurements and opaque sample numbers. Do not print raw benchmark responses.

```powershell
python -m forte.observability.benchmark --base-url http://127.0.0.1:8000 --files <local-mov-path> --mode single --output benchmarks/performance/local-single.json
python -m forte.observability.benchmark --base-url http://127.0.0.1:8000 --files <five-distinct-local-mov-paths> --mode burst --model-warmth warm --cache-mode disabled --output benchmarks/performance/local-burst.json
python qa/system/run.py --base-url http://127.0.0.1:8000 --output qa/system/local-http.json
python qa/system/process_recovery.py --output qa/system/worker-process-recovery.json
python benchmarks/performance/readiness.py --base-url http://127.0.0.1:8000 --output benchmarks/performance/readiness.json
python -m pytest tests/test_observability.py qa/system/test_reliability.py -q
```

Single mode accepts one or more clips and processes them sequentially. Burst
requires exactly five different SHA-256 contents; changing filenames does not
establish distinctness. Hashes are checked locally and discarded. Both modes
read local inputs for a hash preflight before timing, so the client file cache
may be warm. This does not establish that the server cache is disabled or that
content is novel to the model service. Warmth/cache flags are operator
declarations; confirm them against serving configuration before making claims.
Use `--novel-content-declared` only after confirming the clips have not appeared
in earlier server runs. A local uniqueness check alone does not prove novelty.
Each completed job is polled another 20 times by default to measure warm
immutable GET lookup and full payload transfer. Set `--repeat-completed-get 0`
for upload-capacity isolation. Repeat-GET failures and timeouts are counted
separately from the job's analysis outcome. They make the benchmark exit fail.

The total POST duration includes upload. The client last-body-chunk-to-ACK
measurement starts when the multipart iterator yields its final chunk; it still
includes transport and is **not** post-upload server acceptance. The API's
`Server-Timing: acceptance` measures T2−T1 with one local monotonic clock after
the body is received and the owned upload plus job row are durable. T0 at the
reverse proxy is unmeasured unless separately instrumented. T3 worker start
and T4 publication require worker timing. T5 is measured by the client after
the full final JSON arrives. The upload-to-result value includes polling delay
and full JSON transfer, so it is an upper bound for result availability.

GET reports separate ASGI lookup, client first-body-byte, and complete transfer
times. First body byte is an application-level observation through httpx,
which can buffer; it is not a packet-capture TTFB. The immutable final JSON can
contain substantial base64 image data. Never equate lookup speed with transfer
speed. Queue times crossing processes require durable UTC timestamps and a
record of clock synchronization; never subtract two process monotonic clocks.

Reports include count, median, maximum, every failure, and descriptive p95 only
at 20 or more observations. A measured p95 is not a population guarantee. Speed
targets are achieved/missed/unmeasured within the declared observed sample.
No inference-speed claim follows from these HTTP timings. Whole-result-cache
hits must be reported separately; neither warm API GETs nor cache hits prove
novel video inference speed. Cold startup, egress-disabled real inference,
maximum-duration real clips, and actual five-job GPU capacity remain
unmeasured until executed on the deployment.
The readiness observer starts its clock when the client starts observing. It
does not claim to measure cold process startup without a launch boundary.
Use `--input-kind synthetic_transport` for generated non-media payloads; their
acceptance/lookup results cannot establish video or model processing capacity.

Record `--gpu-model`, `--gpu-count`, `--gpu-vram-gib`, `--model-revision`,
`--frame-budget`, and `--max-pixels` for real runs. Installed package versions
describe the benchmark client's environment, not the server. Archive the
server image digest, actual GPU inventory, exact model/processor version, video
duration/resolution aggregate, and local stage report alongside the JSON.
`--probe-inputs` records numeric duration/resolution aggregates locally before
timing. Probe failures are counted, and neither media pixels nor descriptions
are read into the report.

For private vLLM metrics, optionally supply `--metrics-url` to the team's
private endpoint. The tool records only actually exposed `vllm:` metric names;
all labels, values, model IDs, and other metrics are discarded. The pinned
candidate is vLLM 0.31.0; no specific metric name is assumed, and an unavailable
endpoint is reported failed or unmeasured. Exposing names is not model readiness.

`qa/system/failure_matrix.json` maps the complete reliability scope. Deterministic
SQLite lease tests establish store behavior; they do not establish PostgreSQL,
an actual killed worker, a successful model, or an externally accessible service.

For the first actual GPU pass, prepare inputs locally with
`python benchmarks/performance/select_inputs.py`. The selector uses technical
metadata from the proposed train subset, independently checks content hashes,
keeps all source paths in ignored `benchmarks/private`, and never opens final
check media. It prepares one shortest baseline plus five disjoint duration
quantiles including the proposed-train maximum. The split remains unlocked;
benchmark authorization does not authorize training or establish reviewed labels.

After the operator confirms full model readiness, run
`python benchmarks/performance/run_selected.py --phase baseline --base-url <team-api> --api-key-file <private-key-file> --output benchmarks/performance/gpu-baseline.json --model-warmth warm --cache-mode disabled --novel-content-declared --probe-inputs`.
Inspect completion/failure and the private local quality interface first. Run
the `burst` phase only after the baseline handoff. Record verified hardware,
model revision and frame/pixel budgets through the normal benchmark arguments.
Never set the novelty flag for a repeat of previously analyzed content.

The selected runner writes a private sample-to-job correlation file for each
phase. Save safe worker stdout locally under `benchmarks/private`, then run
`merge_stages.py --http-report <aggregate-http-json> --private-job-map <private-map-json> --worker-log <private-worker-jsonl> --output <aggregate-stage-json>`.
The merger exports fixed numeric stage fields and opaque sample numbers only.
Missing terminal worker records and missing stages remain unmeasured; HTTP
timeouts and failures remain in the denominators. Whole request HTTP elapsed
time does not become GPU compute time. Prefill/decode measurements require
actual verified model-server telemetry.

The first deployed, novel proposed-train baseline is recorded in
`gpu-baseline.json`, with safe worker/queue fields in `gpu-baseline-stages.json`
and terminal fixed audit fields in `gpu-baseline-terminal-technical.json`.
It failed with `insufficient_visible_evidence` after three completed model
requests; this does not establish a successful completed-video latency or
reviewed accuracy. `gpu-baseline-model-timing-delta.json` contains actual
server-reported prefill/decode/queue lifecycle counter deltas over a quiet
window with the same model process at both boundaries. Parallel request sums
are not job wall time or GPU-profiler compute time. No percentile is inferred
from counter sums or the single video.

`inspect_terminal.py` accepts only benchmark-owned opaque job mappings and
exports a fixed technical audit subset. Its optional private handoff remains
inside ignored `benchmarks/private`; audit contents, source references, and
proposal media never enter the public technical report. Annotation artifact
transfer must finish or defer before an isolated benchmark window.

The first five distinct upload burst is now recorded in `gpu-burst.json`.
Four uploads became durable jobs: one completed and three failed. A fifth
upload ended with a transport `ReadError`; a private exact-hash reconciliation
found no matching durable job. It remains a failed upload in the denominator.
`gpu-performance-findings.json` records the fixed profile, six-attempt outcome,
limits, and residual model work after deadline failures. The final attributable
model delta uses `gpu-burst-metrics-after-drain.json`, where running and waiting
were both zero; the earlier after snapshot still had running work and must not
be treated as a quiet final boundary. `gpu-observed-summary.json` is reproducible
with `python benchmarks/performance/summarize_observed.py`.

These observations missed successful five-upload capacity and the 240-second
completion safety target. The one validated completion arrived in 240.215 s;
successful video p95 and reviewed accuracy remain unmeasured. The completed GET
percentile comes from 21 reads of that same result and does not establish a
video throughput percentile. No provided-media result content is in these
technical reports.

Transport diagnosis uses generated FFprobe-invalid payloads of 28,718,927
bytes through authenticated team origins. `generated-transport.json` preserves
the initial normal/paced comparison: the public origin reset both requests
around 33 s before the multipart footer was yielded; direct Brev accepted the
normal upload in 2.868 s and returned 408 for the 54.793 s paced upload. This
reproduces the public transport failure without using provided videos; it does
not establish its cause. Byte counts record iterator handoff, not socket writes
or server receipt. Sanitized exception chains contain fixed class labels and
numeric errno only. Global model counters advanced during this window, so
`generated-transport-model-counters.json` explicitly refuses exclusive-window
or no-inference attribution. No model request was issued by this client.

For a separately authorized ingress hypothesis, run exactly one generated normal
upload with a new output and correlation path:
`python benchmarks/performance/generated_transport.py --api-key-file <private-key-file> --origins public --modes normal --condition-label <technical-label> --output <new-public-json> --private-job-map <new-benchmarks-private-json>`.
The runner refuses existing artifacts. Wait for the deployment owner's ready
handoff and finish artifact transfers before making an isolated comparison.

The prepared `staged-concise` paired plan is at
`benchmarks/quality/trials/staged-concise/paired.plan.json`. Preparation makes
no API/model requests. After an explicit release, use
`python benchmarks/performance/run_selected.py --phase paired --trial-id staged-concise --repeat-content --profile-label staged-concise --base-url <team-api> --api-key-file <private-key-file> --output <new-public-json> --model-warmth warm --cache-mode disabled --probe-inputs`.
Paired mode processes the same six contents sequentially, preserves stable
private sample IDs, isolates exports under `benchmarks/private/trials`, and
marks all primary SLO targets ineligible because the content is reused. The
original burst and sequential trial also have different load conditions.
`--novel-content-declared` is rejected for repeat-content trials. Original
artifacts and existing trial outputs are never overwritten.

Every actual selected-run submission now calls the append-only exposure
registry helper before network requests and fails closed if registration fails.
Prepare-only does not register an attempt. To refresh live service exposure
coverage, use
`python benchmarks/performance/collect_exposure_inventory.py --output <new-benchmarks-private-json> --aggregate-output <new-aggregate-json>`, then ingest it with A02's registry workflow. It reads only accepted content
hashes and aggregate active-job counts in a PostgreSQL repeatable-read snapshot.
Later uploads require a fresh inventory and ingestion before split locking.
