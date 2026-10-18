# Concrete private quality and acceptance operations

These commands prepare or execute the existing full analysis pipeline. Preparation
and generated CPU tests make no private model calls. Root owns actual service
switches, releases and inference. Keep detailed artifacts on the protected host;
the commands print aggregate counts/static errors only.

## Balanced six-source quality comparison

The frozen population draft is
`benchmarks/private/qwen38/quality-balanced-six.plan.draft-a13.json`; its source
map/operator draft is `quality-balanced-six.operator-spec.draft-a13.json` in the
same private directory. It contains all six reused VALIDATION contents, three
provided positives and three provided negatives. It supplies no individual
labels, coarse points or action narratives to the model. No FINAL_CHECK source,
target or label is needed.

Root's shared recipe is full nineteen-fact analysis, nonstaged, 8-second windows,
4-second stride, 2-fps sampling, at most sixteen frames, eight optional
refinements, one job, one model call in flight. Set identical
`recipe.worker_settings` (or `recipe.sampling`) and the operator spec's `worker`
object, plus `recipe.provider_max_attempts:2`. The exact worker fields are
`evaluation.qwen38_pipeline.WORKER_FIELDS`. Bind real prompt/source/rule,
generation/evidence/timer/envelope identities, actual tuned adapter hash, and
`recipe_sha256` before using the draft as a valid comparison plan. Do not use an
invented adapter to make preparation pass.

Existing metadata gives **56 initial generation requests per arm**. The
conservative reservation is **208 generation HTTP attempts per arm**:
`(56 + 6*8 possible refinements) * 2 provider attempts`. All three arms therefore
have at most 624 reserved generation attempts, not eighteen calls. The source
denominator remains six per arm. Preparation probes/hashes actual selected source
bytes again and recomputes these bounds. Root must stage only the six eligible
originals at opaque content-hash paths before the actual host preparation.

Create one actual private spec per arm, containing:

- `arm`, `worker`, `sources:[{sample_id,content_sha256,split,owned_path}]`.
- `provider:{base_url,served_model_name,launch_config?}`. Use actual strict tuned
  launch config for tuned, explicit diagnostic launch config for untuned; baseline
  uses its frozen manifest. Optional private key environment name is `api_key_env`.
- `runtime_bindings:{source_archive_sha256,runtime_image_sha256,base_manifest_sha256}`
  from the actual root release for that arm.

Then run on the private execution host (paths are operational examples):

```sh
python -m evaluation.qwen38_pipeline prepare --comparison-plan /private/quality/comparison.bound.json --spec /private/quality/untuned-spec.bound.json --role-config /private/quality/a09-role-config.json --output /private/quality/untuned-model-plan.json
```

The emitted plan uses the existing `forte.qwen38-attempt-plan.v1` schema with
`phase:evaluation`, six immutable `source_attempts`, and one `.attempts` entry for
every conservative model-call slot. Obtain the corresponding exact root release
and A09 reservation using the established preflight workflow. Release must bind
the raw plan-file SHA, metadata config SHA, actual model/base/adapter and
source/image/base pins, exclusive handoff, expiry and execution bound. Execute
only after those artifacts exist:

```sh
python -m evaluation.qwen38_pipeline execute --plan /private/quality/untuned-model-plan.json --release /private/quality/untuned-root-release.json --reservation /private/quality/untuned-reservation.json --output /private/quality/untuned-results
```

Repeat preparation/execution under the appropriate actual model for baseline
and tuned. No runner performs a model switch or deployment. Each source creates
a fresh opaque owned copy/job, uses the unchanged `AnalysisPipeline`, and retains
its raw result bytes and original audit. The explicit untuned provider remains
outside public production selection. Every actual generation HTTP attempt
consumes its slot before the request; failed requests and unused slots are
retained, with unused slots marked `not_attempted`. A reservation has a one-use
execution claim. All six source terminal slots exist before the first call, so a
process interruption does not remove later sources from the denominator.

The direct pipeline has no client upload or browser clock. Its
`pipeline_start_to_result_ms` is separately labeled; upload/full-byte/UI timing
fields remain null. The source terminals are compatible with
`evaluation.qwen38_comparison`. Each successful run also exports an original
source/native-frame/evidence receipt. The additional native ledger decode runs
after the result and has a separate duration; it does not pretend to be part of
the inference sampling or client latency.

## Genuine worker witnesses for six HTTP acceptance jobs

`evaluation.qwen38_worker_trace` wraps the unchanged worker/pipeline, not historical
audits. The API still handles actual fresh authenticated uploads, immutable
publication and exact model bytes. The wrapper records actual decoder,
preprocessing, generation HTTP and evidence-assembler invocations. It records
source JPEG joins from the real assembler and returns the exact same
`PipelineResult` object/bytes. A missing actual stage fails the witness; old
results cannot acquire fresh stage/cache claims.

A12's `qwen38_acceptance_ops.py trace-bind` produces schema
`forte.qwen38-worker-trace-binding.v1`. Bind the six source attempts separately
from the **126 exact model HTTP slots** for three uploads each of the two chosen
short TRAIN fixtures. Preserve provider attempts two, single job, clips/inflight
one, nonstaged 8/4/2/16/eight-refinement settings. The worker requires the exact
model slot plan/release/reservation and consumes the allocated source attempt's
slot before every POST. Other source contents are rejected before decode.

Required binding fields are:

```text
schema_version
source_attempts [{attempt_id, content_sha256, split, repeat_index}]
expected_model_identity {model_id, model_revision, adapter_sha256}
provider_max_attempts: 2
maximum_model_requests: 126
worker_settings (the exact actual common recipe)
model_attempt_plan {path, sha256}
model_release {path, sha256}
model_reservation {path, sha256}
wrapper_sha256 (actual evaluation/qwen38_worker_trace.py)
pipeline_sha256 (actual loaded forte/pipeline.py)
trace_source_archive {path, sha256}
trace_runtime_image_sha256 (actual CPU app image, separately from GPU model image)
```

The immutable new source bundle must contain `evaluation`, `src/forte`, `serving`,
`training` and other imported public dependencies/catalogs. Mount it on a separate
read-only path so the app image's `/app/.venv` remains available. The following
private compose override preserves the existing entrypoint's secret loading and
the existing data-volume mount. Exact source/runtime/config paths and digests
must match A12's bound files; this example does not activate anything:

```yaml
services:
  worker:
    image: ${FORTE_TRACE_APP_IMAGE:?Set exact CPU app repository@sha256 digest}
    working_dir: /trace-code
    entrypoint: [python, /app/deploy/entrypoint.py]
    command:
      [python, -m, evaluation.qwen38_worker_trace,
       --binding, /run/configs/qwen38-trace.bound.json,
       --witness-root, /var/lib/forte/trace-witnesses]
    environment:
      PYTHONPATH: /trace-code:/trace-code/src:/app:/app/src
      FORTE_TRACE_APP_IMAGE_SHA256: ${FORTE_TRACE_APP_IMAGE_SHA256:?Set independently inspected CPU image SHA}
      FORTE_MAX_CONCURRENT_JOBS: "1"
      FORTE_MAX_CONCURRENT_CLIPS: "1"
      FORTE_MAX_INFLIGHT_PER_JOB: "1"
      FORTE_STAGED_SCREENING: "false"
      FORTE_WINDOW_SECONDS: "8"
      FORTE_STRIDE_SECONDS: "4"
      FORTE_SAMPLE_FPS: "2"
      FORTE_MAX_FRAMES_PER_WINDOW: "16"
      FORTE_MAX_REFINEMENT_PASSES: "8"
      FORTE_COMPACT_FRAME_IDS: "true"
      FORTE_CONCISE_OBSERVATIONS: "true"
    volumes:
      - ${FORTE_TRACE_SOURCE_DIR:?Set verified public source tree}:/trace-code:ro
      - ${FORTE_TRACE_BINDING:?Set actual bound trace JSON}:/run/configs/qwen38-trace.bound.json:ro
      - ${FORTE_TRACE_SOURCE_ARCHIVE:?Set actual observer source archive}:/run/configs/trace-source.tar:ro
      - ${FORTE_TRACE_MODEL_PLAN:?Set exact model-slot plan}:/run/configs/model-plan.json:ro
      - ${FORTE_TRACE_MODEL_RELEASE:?Set exact root model release}:/run/configs/model-release.json:ro
      - ${FORTE_TRACE_MODEL_RESERVATION:?Set exact A09 model reservation}:/run/configs/model-reservation.json:ro
```

Bind paths inside the trace JSON to these in-container paths. Keep the selected
trained model's normal config mount and `FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG`
where applicable; the wrapper calls the unchanged strict selector. Do not use
this public worker path to introduce the separate untuned diagnostic. Its model
remains directly injected only for private quality runs.

After idle drain, verified backup, exact root releases and all ready gates, the
operator may apply the previously reviewed override with the existing compose
stack and `up -d --no-deps worker`. **No such command was run for implementation.**
The standalone process command is:

```sh
python /app/deploy/entrypoint.py python -m evaluation.qwen38_worker_trace --binding /run/configs/qwen38-trace.bound.json --witness-root /var/lib/forte/trace-witnesses
```

Witnesses remain at
`/var/lib/forte/trace-witnesses/{job_id}.attempt-{attempt}.json`, model-slot terminals
at `model-slot-terminals.json` in that directory, and the one-use model-plan claim
under `/var/lib/forte/trace-execution-claims/`. No prompts, observations, source
filenames or pixels enter returned worker logs. Backend attention KV internals
are separately unmeasured; cache-zero claims describe fresh pipeline
decode/model-response/evidence/whole-result stages.

## Export actual server receipts for A12

A12's new `export-report` operator joins each actual HTTP terminal to its genuine
worker witness and immutable stored result. The reusable exporter CLI is:

```sh
python -m evaluation.qwen38_server_export --audit /var/lib/forte/work/JOB/attempt-1/analysis.private.json --result /var/lib/forte/results/JOB.1.RESULTSHA.json --binding /private/acceptance/attempt-binding.json --witness /var/lib/forte/trace-witnesses/JOB.attempt-1.json --output /private/acceptance/server-receipt.json
```

`attempt-binding.json` contains `attempt_id`, actual `job_id`, `content_sha256`,
`conditions_sha256`, `recipe_sha256`, `expected_model_identity`, all copied from
the frozen acceptance plan and its actual terminal. The exporter verifies raw
result/audit hashes, exact witness source attempt, real stage trace, original
source ownership and clocks, full native PTS ledger and returned JPEG bytes. It
emits the existing `forte.qwen38-acceptance-server-receipt.v1` schema consumed by
`bind_server_receipt`. Detailed decisions stay private. The extra native CPU
decode is reported separately and never substituted for an upload/UI timer.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qwen38_operational_evaluation.py tests/test_qwen38_untuned.py tests/test_qwen38_comparison.py -q
```

The reference limits remain unchanged: provided binary labels are not
human-reverified truth, validation is reused development data, and coarse points
do not establish event type, support, legal applicability or exact boundaries.
