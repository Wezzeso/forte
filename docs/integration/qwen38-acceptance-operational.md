# Six fresh uploads after model selection

This preparation has two fixed TRAIN contents, chosen by shortest recorded
duration within each organizer-provided binary class before any new outputs.
There are three fresh POSTs per content, six source jobs in total. The class
reference stays in a separate private file and never enters a model prompt.
These supplied classes are not reviewed event/type/legal ground truth.

Current local preparation is
`benchmarks/private/qwen38/acceptance-six-a12/`: `population.json`, two
hash-verified source copies, `reference.private.json`, the intended worker
recipe, and an intended model-request budget. No reservation, upload, model
call, serving activation or selected-runtime binding has been performed.
An actual trained adapter and selected endpoint are pending root's work.

The intended recipe preserves 8-second windows, 4-second stride, 2 fps,
16 frames, all nineteen facts and at most eight refinement requests. Screening
is disabled. There is one active job, one clip and one in-flight model request.
The existing HTTP provider has at most two transport attempts per logical
request. This is a maximum allocation, not an assertion that every request is
retried or that HTTP 429 is retried.

| Fixture | Duration | Coverage windows | Maximum logical requests/job | HTTP attempt slots/job | Fresh jobs |
|---|---:|---:|---:|---:|---:|
| Provided positive | 11.633333 s | 2 | 10 | 20 | 3 |
| Provided negative | 15.566667 s | 3 | 11 | 22 | 3 |

The end-aligned coverage tail is included. There are **126 maximum model POST
slots** and at least **15 initial full-fact requests** if all six jobs reach
their initial analysis. Startup/prewarm work is separate. A failure consumes
its source slot and every already attempted model slot. Unneeded model slots
are settled without refund or permission to retry a source job.

## Private operational bundle

The local preparation command has already run. A rerun must use a new output
directory, rather than replacing the frozen population:

```powershell
.venv/Scripts/python.exe -m benchmarks.performance.qwen38_acceptance_ops select --roles benchmarks/private/qwen38/development-roles-20261017-a09.json --labels data/manifests/provided_binary_labels.train.jsonl --recordings data/manifests/recordings.jsonl --output benchmarks/private/qwen38/acceptance-six-a12
```

Once A13's exporter/trace files are frozen, package only the operational modules
and these two source copies. The archive is private. It contains no API key,
model weights, new reservation or root release.

```powershell
.venv/Scripts/python.exe -m benchmarks.performance.qwen38_acceptance_ops bundle --population benchmarks/private/qwen38/acceptance-six-a12/population.json --output benchmarks/private/qwen38/acceptance-six.ops.final-a12.tar.gz
```

Transfer and extract it to a new protected directory on the existing host,
for example `/home/ubuntu/forte-acceptance-ops-a12`. Preserve the relative paths
from the archive. Its companion manifest records exact member and archive
hashes. The host launcher is Python 3.10 compatible; the actual project command
runs inside the existing API image's Python 3.12 environment. No GPU is attached
to these temporary operational containers.

The final local archive has 24 members, 5,730,261 bytes, SHA-256
`b4c2ba2df50dd32b7cc344fae73929ec41d4508888f276faf4f317e583f785b3`.
Its member hashes and isolated operational-module imports were verified.
The earlier interrupted archive remains historical and must not be deployed.

## Bind only actual runtime and recipe artifacts

Before binding, root supplies these files inside that new operational root:

- The actual A06 runtime config from a successful run and actual selected arm
  (`qwen38_tuned` or explicitly selected `qwen38_untuned`). Null template pins
  fail. The binder reads metadata, not model weights.
- The exact worker recipe from the configured traced worker. Include the
  A13 `WORKER_FIELDS` and `provider_max_attempts: 2`. It must match actual
  `WorkerSettings`, including job/clip concurrency and screening disabled.
- A frozen acceptance recipe envelope passed as `--comparison-plan`, with
  `arms.<selected arm>` containing actual model/revision/adapter, a complete
  `recipe.worker_settings`, `recipe.provider_max_attempts: 2`, and
  `recipe_sha256 = digest(recipe)`. Keep the actual prompt/generation/evidence
  and supported-envelope pins in this recipe. This selected-arm TRAIN
  acceptance envelope is distinct from A13's three-arm VALIDATION quality
  population. A13's final direct quality runner and this HTTP trace both
  preserve two provider attempt slots and count actual model POSTs. Their
  different client timing boundaries remain explicit.

Using paths relative to the operational root:

```bash
python3 /home/ubuntu/forte-acceptance-ops-a12/deploy/qwen38_acceptance_host.py bind \
  --ops-root /home/ubuntu/forte-acceptance-ops-a12 \
  --runtime-config benchmarks/private/qwen38/selected.runtime.actual.json \
  --worker-recipe benchmarks/private/qwen38/worker-recipe.actual.json \
  --comparison-plan benchmarks/private/qwen38/acceptance-envelope.actual.json \
  --arm qwen38_tuned
```

`bind` uses network `none` and produces `bound/plan.json` with six source
attempt IDs and `bound/model-attempt-plan.json` with 126 model slots. Each
model slot names its exact source attempt. It also writes the actual budget
and file-binding receipt. No model call or exposure reservation occurs.

## Separate releases and the genuine worker trace

Refresh the exposure inventory. A09 reserves the exact 126-slot model plan
using its own root release. The model release binds actual source/runtime/base
hashes, model identity, exact plan file hash, `maximum_attempts: 126`, expiry,
exclusive idle handoff and the bounded execution window. This is separate from
the later six-source-job release/reservation.

Copy the actual model release/reservation into the operational root. Preserve
the operational archive there too, so its hash can bind the wrapper source.
Create the trace binding without activating a worker:

```bash
python3 /home/ubuntu/forte-acceptance-ops-a12/deploy/qwen38_acceptance_host.py trace-bind \
  --ops-root /home/ubuntu/forte-acceptance-ops-a12 \
  --worker-recipe benchmarks/private/qwen38/worker-recipe.actual.json \
  --model-release benchmarks/private/qwen38/model-release.actual.json \
  --model-reservation benchmarks/private/qwen38/model-reservation.actual.json \
  --source-archive benchmarks/private/qwen38/acceptance-six.ops.final-a12.tar.gz
```

This writes `bound/trace-binding.json`. Its actual CPU image is read from the
API container. Its pipeline hash comes from that image's installed source,
and its wrapper hash comes from the frozen operational bundle. A13's
`evaluation.qwen38_worker_trace` validates those pins, the exact worker
settings, six source IDs and all 126 reserved model slots before processing.
It consumes a model slot **before** each actual `/chat/completions` POST,
including failed/uncertain transport attempts. Its witness preserves actual
stage counters, serving identity, audit/result hashes and original evidence
bindings. Historical audits cannot be upgraded into this witness.

Root configures the worker with A13's traced entrypoint and the operational
root mounted at `/ops`, using the same path names as the binding. Witnesses go
to `/var/lib/forte/trace-witnesses`. Normal deployment secret initialization
must still run. Set `FORTE_TRACE_APP_IMAGE_SHA256` to the actual CPU image hash
without `sha256:`. Root owns service activation, admission and GPU handoff;
the acceptance launcher changes none of them.

Next create and reserve the exact **six-source-job** release. In addition to
the existing acceptance requirements, it must contain:

```json
{
  "maximum_generation_http_attempts": 126,
  "model_budget_file_sha256": "<exact bound/model-budget.json file SHA-256>",
  "worker_trace_binding_file_sha256": "<exact bound/trace-binding.json file SHA-256>"
}
```

Neither a scalar budget nor a source-job receipt replaces the 126-slot model
reservation. Both plans and every attempted/unused/failed slot must be settled.

## Actual six-upload command

Run only after the selected model and traced worker are ready and root grants
the exact source release/reservation. The launcher verifies that the running
worker uses A13's trace entrypoint and the exact mounted binding. It discovers
the existing API key's secret mount from Docker metadata and mounts that file
read-only. Key bytes never enter command arguments, output, source, or plans.

```bash
python3 /home/ubuntu/forte-acceptance-ops-a12/deploy/qwen38_acceptance_host.py execute \
  --ops-root /home/ubuntu/forte-acceptance-ops-a12 \
  --release benchmarks/private/qwen38/source-release.actual.json \
  --reservation benchmarks/private/qwen38/source-reservation.actual.json
```

The direct API remains
`https://forte-review-wd88a7vvt.gobrev.dev`. Each response is fetched completely
before the next source is submitted. Queue time, transfer, rejection, timeout,
failure and full evidence bytes remain part of the original attempt outcome.
The launcher does not unpause admission or retry failed source jobs.

## Export actual artifacts and report

After the six terminal outcomes, this offline command mounts the existing
service data read-only, locates each actual fresh witness and immutable result
file, invokes A13's native-source exporter, then calls the original acceptance
reporter. Native forensic decoding is timed separately after the result and
never substituted for upload latency or inference sampling.

```bash
python3 /home/ubuntu/forte-acceptance-ops-a12/deploy/qwen38_acceptance_host.py report \
  --ops-root /home/ubuntu/forte-acceptance-ops-a12
```

Only aggregate fields leave the launcher. Complete responses, source IDs,
native PTS, JPEG hashes and witnesses stay private. Missing witnesses/exports
leave repeatability unverified. The separate class diagnostic retains all six
expected attempts and labels its agreement as organizer-provided binary
agreement, without claiming reviewed event/legal accuracy.

HTTP final-byte timing does not establish the browser's rendered usable-result
time. The UI timer and a real controlled browser run remain required for that
additional boundary. Passing two repeated TRAIN fixtures establishes only
that finite released scope.
