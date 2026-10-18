The optional private native-source backend lives in `serving.fast_service`. It
is separate from the active Qwen provider and is not enabled by default. It
holds one verified SigLIP vision encoder and one real temporal checkpoint in
memory. It has no source-name, known-file feature-cache or organizer-label
lookup. The CPU provider independently verifies any required fact calibration;
the backend always reports `candidate_calibration_verified: false`.

The source mount must be read-only and have the same absolute path as the CPU
worker's upload mount. Only the exact `<source-root>/<asset.job_id>.mov` path
is accepted, with a safe job identifier, no symlink, a bounded file size and a
full matching SHA-256. The source is decoded again for every request; tensors
and private metadata remain in request memory. Every native frame and the tail
must be present, with actual integer PTS, canonical source IDs, original source
dimensions, rotation, key-frame flags and global-view geometry. No JPEG paths
or invented training split are needed to invoke the head.

Startup requires an immutable encoder manifest, the actual checkpoint manifest
digest, a separately file-hashed `HeadIdentity` JSON, and the complete generic
feature-spec JSON with its canonical digest. The executed decoder/encoder,
sampling and global-only crop recipes must reproduce that exact feature spec.
This includes the frozen encoder/feature-stream source hashes and runtime
versions already bound by the encoder recipe. A mismatch fails startup.

A separate required source-code bundle pin covers packaged `src`, `serving`
and `training` Python source plus `pyproject.toml` and `uv.lock`; private output,
run and bytecode directories are excluded. Compute its canonical digest inside
the actual packaged runtime before selecting it:

```text
python3 -S -c "from serving.fast_service import source_code_bundle,_canonical_sha256; print(_canonical_sha256(source_code_bundle()))"
```

This execution pin can change for a new service build without changing the
completed W1 cache or its frozen semantic feature recipe. Startup binds fixed
batch size 32, explicit FRAME decoder threads (default 4), prefetch (default 2),
native/global-only coverage, resource limits, actual deterministic settings and
runtime packages, GPU identity, checkpoint and code bundle in a separate
execution manifest. Fixed settings support later repeatability measurements;
they do not themselves prove repeated bit-identical inference.

The service is a small ASGI application and imports no ML packages, decoder or
web server until explicitly loaded. Example operator invocation inside the
separate offline feature image, after a real candidate checkpoint is available:

```text
python3 -m serving.fast_service
  --source-root /var/lib/forte/uploads
  --snapshot /encoders/siglip2-base-patch16-256
  --encoder-manifest /app/configs/serving/encoders/siglip2-base-patch16-256.manifest.json
  --encoder-manifest-sha256 cd51863090e8c411a1227d8039e44118c44488243e3d615d73907106568877e7
  --feature-spec /candidate/feature-spec.json
  --feature-spec-sha256 <actual canonical feature-spec digest>
  --head-identity /candidate/head-identity.json
  --head-identity-sha256 <actual file digest>
  --checkpoint-dir /candidate/checkpoint
  --checkpoint-manifest-sha256 <actual checkpoint manifest file digest>
  --source-code-bundle-sha256 <actual packaged Python source bundle digest>
  --batch-size 32 --device cuda:0 --host 0.0.0.0 --port 8010
```

The deployment must use the internal private network with no published port,
read-only artifacts/source mounts, the unprivileged feature-runtime user, and
offline environment variables. No service deployment was performed as part of
the implementation or CPU tests. Do not start another GPU consumer without the
existing coordinated ownership handoff.

`GET /health` is liveness. `GET /ready` returns HTTP 200 only after the actual
encoder and head load, with top-level `ready`, `head_sha256`,
`feature_spec_sha256`, `encoder_manifest_sha256` and
`candidate_calibration_verified` fields, plus `execution_manifest_sha256`,
`source_code_bundle_sha256` and `batch_size`. `GET /execution-manifest` returns
the complete generic execution policy for pinning before/after fresh inference
trials. It contains no uploaded source identity or private path. This is backend
readiness, not a calibration, repeatability or task-accuracy claim.

`POST /v1/temporal/predict` accepts JSON
`{asset: VideoAsset, deadline_budget_sec: number, clip_id?: string}`. The budget
must be positive and at most 120 seconds; its clock begins when the service
receives the request, including body processing. The caller must deduct all
earlier upload/queue time from its own end-to-end deadline. Successful responses
are the A08 `NativeHeadResult` containing the bound complete source context,
numeric head output, video logit and fixed numeric timing fields. Results are
private and must not be included in access/error logs. The service disables
access logs and emits only static error codes.

Requests are bounded to 64 KiB; results to 64 MiB. One inference is admitted at
a time, with no hidden request queue. A busy backend responds 503. The decoder
and encoder overlap through the existing bounded two-batch producer. A deadline,
outer task cancellation or client disconnect signals the decoder/encoder, while
the GPU admission remains occupied until the actual worker and decoder drain.
Caller completion alone never declares the GPU free. Encoder elapsed time is
the measured sum of encoding calls; overlapping CPU decode time is not inferred
by subtracting this from wall time.

Synthetic CPU validation:

```text
python -m unittest serving.tests.test_fast_service -v
```

These tests check source/path/hash boundaries, request/response bounds, static
errors, readiness fields and actual worker-thread admission retention through
timeout/disconnect. They load no weights, GPU or private media. A real new-source
decode → encoder → trained-head run and fresh repeatability/latency measurements
remain required before selecting this backend for the live worker.
