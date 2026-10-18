# Observed local system evidence — 17 October 2026

`local-http.json` records requests over real localhost TCP to the actual Uvicorn
API with a durable SQLite store. Six checks passed: missing/wrong keys for
POST and GET returned 401, unknown authorized job returned 404, and a synthetic
corrupt MOV upload returned 202. No worker was running during this observation;
its eventual failed terminal state is UNMEASURED. No media/model success is
inferred from durable acceptance. The temporary service has been stopped.

`local-http-worker.json` is the subsequent run against the actual independent
`python -m forte.worker` process and the actual Uvicorn API, with an isolated
durable SQLite store and the real catalog. All seven checks passed, including
synthetic corrupt upload → actual worker → failed public response without a
verdict or violations. No model was needed or called for the decode failure.
Both temporary processes have been stopped. This confirms real failure-path
wiring; it does not establish successful inference or crash recovery.

`worker-process-recovery.json` subsequently records actual abrupt OS termination
of a leased Worker process, real-time lease expiry, replacement Worker process
publication at attempt 2, and rejection of the old attempt's publication. This
check passed using the actual Worker and SQLite store with an explicitly
synthetic pipeline. It proves process-level recovery/fencing in that scope;
GPU/model pipeline recovery and deployment PostgreSQL remain separate checks.

`../../benchmarks/performance/local-transport-synthetic.json` records one actual
29,000,000-byte **synthetic zero-byte transport payload**, not a valid video.
Client total POST was 393.536 ms; ASGI post-body durable acceptance was
123.177 ms. This single measurement establishes neither p95 nor inference
performance. The client intentionally stopped observing after two seconds;
the job did not complete. Every speed target is UNMEASURED. The report includes
the measured client environment and an explicit no-GPU declaration. There is
no warm-model, real-video, accuracy, cutoff, or capacity claim.

`../../benchmarks/performance/local-burst-transport-synthetic.json` records five
simultaneous, locally hash-distinct 1 MiB **synthetic transport** uploads. All
five returned durable 202 responses. Total POST median/max were 233.727/664.587
ms; post-body durable acceptance median/max were 87.145/143.080 ms. With only
five uploads, acceptance p95 remains UNMEASURED. Across 29 measured processing
GETs, ASGI lookup p95 was 6.201 ms (max 13.017 ms), while client full-response
p95 was 69.205 ms (max 73.054 ms). The 25 ms server-side lookup target was met
in this synthetic observed sample. These GET timings demonstrate why lookup
and transfer must be reported separately. Both two-second observation windows
ended without a model result; five-real-video inference capacity remains
UNMEASURED. Request attempts and all client timeouts are retained in the JSON.

The command `python -m pytest tests/test_observability.py
qa/system/test_reliability.py -q` most recently passed 19 tests in 1.01 seconds. They cover
monotonic stage/error timing, accurate body/durable/transfer boundaries, auth
rejection without body consumption, bounded private-free telemetry, actual
metric-name discovery, small-sample reporting, streamed benchmark uploads,
absolute client deadlines, and actual SQLite lease recovery, stale attempt
fencing, cancellation, deadline failure, and database reopen. Test execution
time is not application latency. All payloads and verdict fixtures in these
tests are synthetic.

Still UNMEASURED: genuine five-video burst, valid large/maximum-duration video
completion, GPU inference, model stage budgets, GPU worker kill/replacement,
PostgreSQL deployment behavior, cold startup, clock synchronization, external
public access, egress-disabled local inference, and real vLLM 0.31.0 metrics.
The repeatable runners and complete failure matrix are supplied for those
checks when the deployment dependencies exist.

The actual public TLS gateway subsequently passed all five auth-only checks
in `public-auth-only.json`. No accepted media jobs were created. Host hardware
and NTP are independently observed in
`../../benchmarks/performance/gpu-hardware-observed.json`; actual vLLM metric
names are recorded in `gpu-vllm-metric-names.json`. These observations establish
neither successful private-video analysis nor quality. Six proposed-train
benchmark inputs have been selected and independently checked by A02; final
check media remain untouched. Private uploads are held until the model owners
complete schema validation and explicitly release the GPU.
# Actual GPU observation added 2026-10-17

The first proposed-train baseline and five distinct public-domain upload
attempts are complete. Six uploads produced five durable jobs, one validated
completion, four terminal job failures, and one transport failure with no
matching durable job. The one completed upload reached full JSON in 240.215 s.
Successful five-upload capacity and the 240 s safety target were missed;
successful-video p95 and reviewed event quality remain unmeasured.

Public technical findings and fixed runtime evidence are in
`benchmarks/performance/gpu-performance-findings.json`; the reproducible
aggregate is `benchmarks/performance/gpu-observed-summary.json`. Raw safe
HTTP/stage/queue reports retain all failures. Server lifecycle counter deltas
use quiet boundaries with an unchanged model process; they include residual
generation after durable deadline failure. Parallel request sums do not measure
GPU-profiler compute or job wall time. Private media and proposal review remain
outside these reports.

