# Qwen3.8 candidate serving

This is a separate executable candidate path. It does not alter
`serving/launch.py`, the baseline manifest or production selection. CPU tests
prove validation behavior; no A100 load, private inference, quality or latency
result is asserted by this document.

The backend uses Transformers `Qwen3_5ForConditionalGeneration` and PEFT. The
hybrid adapter is actually reloaded and every loaded LoRA tensor is compared
against the serialized tensor. This avoids treating a serving alias as proof
that hybrid GDN adapters were applied. BF16 and NF4/BF16 double quantization
are explicit separate identities. A trained NF4 adapter cannot silently run
under a BF16 identity, or the reverse.

Prerequisites:

1. Fully staged, separately verified pinned 18-shard public base and complete
   tokenizer/processor/template/index inventory from A01/A07.
2. Actual `forte.qwen38.silver-run.v1` completed run with at least two optimizer
   steps, finite gradients/loss, changed adapter tensors, frozen base/vision,
   exact base manifest and quantization, and successful adapter reload forward.
3. A bound copy of `configs/serving/qwen38/candidate.runtime.example.json`.
   Null hashes/versions intentionally fail. Pin the actual adapter tree/config,
   run receipt, source archive and exact installed package versions/image.
4. Candidate image containing `serving/`, `src/forte/`, pinned public manifest,
   the actual Transformers/PEFT/torch stack and FastAPI, uvicorn, Pillow and
   jsonschema. NF4 also requires an actually tested bitsandbytes installation.
5. Fresh exposure gates, accepted-call accounting/reservations, quiet drained
   queues and exclusive GPU handoff, within the unchanged rental stop/budget.
   Generated warmup invokes `len(prewarm_shapes)+2` actual synthetic model
   requests (one per shape, one largest-shape concurrency check, one decision
   smoke); reserve/account them as required by the central registry.
6. Isolated Linux internal Docker network with no default IPv4/IPv6 route,
   read-only model/adapter mounts, no model public port, and a deployment-level
   no-egress probe. Set `FORTE_RUNTIME_IMAGE_SHA256` from an independently
   inspected image digest. The launcher compares this assertion but cannot
   itself inspect Docker's host image store.

CPU-only full artifact check (all 55GB of base shards are read; CUDA is not
initialized):

```text
python -m serving.qwen38_launch --config /private/qwen38-runtime.json --verify-only
```

Print expected public identity without opening weights or claiming a load:

```text
python -m serving.qwen38_launch --config /private/qwen38-runtime.json --expectations-only
```

After the exclusive authorized handoff, run inside the pinned isolated image:

```text
python -m serving.qwen38_launch --config /run/configs/qwen38-runtime.json
```

The optional `deploy/compose.qwen38-candidate.yaml` is merged with
`deploy/compose.yaml`, using profile `qwen38`; do not combine the baseline GPU
override with it. Bind all required environment paths/digests before using it.
It does not authorize a cutover or run itself. It has no public model ports.

Startup hashes base/adapter/run/source, checks actual package versions and the
image assertion, then loads the model with `local_files_only=True` and
`trust_remote_code=False`. All requested LoRA tensors must load exactly.
Readiness stays false until generated JPEG warmup and the actual application's
decision-schema/visual sanity smoke both pass. `--check-ready` compares all
public expected identities and actual loaded/warm flags. A hash-only receipt
always has `model_loaded=false`.

Requests accept bounded local JPEG data URIs, a single pinned model alias,
greedy decoding with seed `20261017`, disabled thinking, one completion and
bounded output tokens. Remote media, paths, tools, request-side processor or
template changes and external JSON-schema references fail. Unknown/truncated,
nonfinite or invalid JSON/schema generations fail; they never become a clean
verdict. One inference uses the GPU at a time; extra requests return busy.
Disconnects/deadlines stop at the next generation boundary. A blocked CUDA
kernel still requires operational drain/termination before another GPU owner.

For API/worker integration, `load_expected_runtime_identity(config_path)` in
`serving.qwen38_identity` reads expected pins only; it never hashes the remote
base or asserts it is loaded. Bind these exact fields and the model alias,
frame/output limits, disabled thinking and seed in `LocalModelConfig`. Require
the candidate readyz response to match every pin, with `model_loaded=true`,
`prewarm_completed=true` and a positive `adapter_loaded_tensor_count`.
`FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG` is the explicit application opt-in.

The warm service is still an unselected candidate. Correctness, fresh-upload
repeatability, real browser upload-to-display timing and a separately bound
quality selection receipt remain mandatory before production activation.

## Native constrained JSON decoding

The new serving source uses installed XGrammar `0.2.7` through its native
Hugging Face `LogitsProcessor`. Both 27B arms use this same implementation;
the baseline common pipeline, prompts, rules and frozen quality family remain
unchanged. The actual decoder policy is pinned as `structured_decoding` in
both bound runtime configs and their matching runtime profile. Package
versions include `xgrammar=0.2.7` and any installed optional FLA kernels.

Every generation gets a fresh matcher/logits processor. Only immutable
compiled schemas are cached, with at most eight entries and 64 MiB total
compiled memory; the compiler's own hidden cache is disabled. Schema features
are explicitly allowed rather than assuming `strict_mode` rejects unknown
keywords. Unsupported keywords, external/unknown references, unsupported
patterns/unions and compilation failures stop the request. There is no
unconstrained fallback or JSON repair. Strict JSON parsing, the full output
schema and the application's semantic validation remain mandatory. The
trusted internal fact-array transform fixes the sorted requested-ID order:
each of the exact N entries receives a singleton fact-ID enum, and trailing
items are forbidden. This rejects duplicate/missing/unknown fact IDs while
preserving `present`, `absent`, `unclear` and every other payload field. The
original public schema and pipeline stay unchanged; valid fact permutations
are deliberately narrowed to one canonical order. Arbitrary public
`prefixItems` schemas remain unsupported. Readiness records the original and
derived schema hashes and count under policy
`canonical_requested_fact_ids.v1`. Grammar does not establish physical truth
or legal correctness, and full semantic postvalidation is retained.

Startup checks actual model/logit vocabulary size `248320`, both EOS IDs
`[248046,248044]` and padding `248044`, then initializes the tokenizer/compiler
and precompiles the exact public full-pipeline schema before readiness. That
schema is derived from the pinned rules catalogue: four enabled rules,
nineteen facts, three distinct event types and sixteen authoritative compact
frame aliases `f00` through `f15`, with concise observations enabled. No
private source IDs, timestamps, media or model calls enter this compilation.
Readiness includes initialization/compile time and the public schema hash.
Unseen schema shapes may still incur separately measured compilation work.

A08's actual image CPU audit verifies token masks and fresh matcher behavior,
including padded vocabulary/EOS handling and rejection of malformed JSON.
This is CPU grammar evidence; it does not establish CUDA masking overhead,
the actual model's full nineteen-fact output quality or endpoint latency.
After the root releases the GPU, run a separately counted generated-only full
nineteen-fact endpoint/schema/semantic smoke before private quality attempts.
The normal one-fact startup visual smoke remains a separate integration check.

## Actual a28 image and immutable source operation

The actual candidate image is
`sha256:a28a1d003c831907d5b8729426f376e7383ecf6c5eb347c4b8ab8c4300e32698`.
Its Python is `/opt/forte-features/bin/python`; it contains package/runtime
locks but no application source. The historical phase2 two-step controller mounts
`/home/ubuntu/forte-qwen38-smoke-code-df11c1e2fafa` read-only at `/app`.
That tree is bound to source archive SHA-256
`df11c1e2fafa5d44a7c431d1f6a09ec2fcaa68ffd5bef6d3d08889b25db14c7e`.
Keep this historical source immutable. The root's next run uses a separately
frozen phase3 source after the training fix; its path/hash remain pending until
the source owner freezes them. Never use the historical df11 source as an
implicit default for that new run. A code change requires a separately named archive,
source tree and new configuration/proof; a working-tree edit does not change
the code currently running in that mount.

The actual public A100 smoke found that an implicit `noexec` `/tmp` prevents
Torch 2.13's Triton-generated `cuda_utils.so` from being mapped. The candidate
GPU override therefore declares `/tmp` as `exec,nosuid,nodev` with UID/GID
1000; the API's temporary filesystem policy is separate. This is for trusted
JIT code generated from the pinned packages. `/run/forte` remains `noexec`.

Use `deploy/compose.qwen38-candidate-operational.yaml` after the generic
candidate override. It supplies the actual image/Python and requires an
explicit matching immutable source bind, runs as
`1000:1000` so mode-0600 adapter/run files can be read, and requires a fresh
UID-1000/GID-1000 mode-0700 cache bind rather than a root-owned named cache.
The root operator prepares that fresh cache/config-output directory before
launch, for example with `install -d -m 0700 -o 1000 -g 1000 <new-path>`.
Keep the source archive and completed run read-only, no public model ports,
and the existing internal private network (or `--network none` for isolated
loopback-only diagnosis). This preparation does not select production.

After the actual run completes, mount `deploy/qwen38_bind_runtime.py`
separately at `/ops/qwen38_bind_runtime.py`; it is an operational helper, not
a mutation of the frozen `/app` tree. In the exact image, without GPU access,
inspect packages first:

```text
/opt/forte-features/bin/python /ops/qwen38_bind_runtime.py --collect-packages
```

With the exact completed run's source archive at `/release/source.tar`, its
completed run directory mounted at `/candidate`, and a fresh private output
bind at `/output`, bind
the actual artifacts (all calls are CPU-only):

```text
/opt/forte-features/bin/python /ops/qwen38_bind_runtime.py \
  --code-root /app --source-archive /release/source.tar \
  --source-sha256 "$FORTE_QWEN38_SOURCE_SHA256" \
  --run-root /candidate \
  --image-sha256 a28a1d003c831907d5b8729426f376e7383ecf6c5eb347c4b8ab8c4300e32698 \
  --max-pixels 262144 --max-new-tokens 4096 --max-context-tokens 32768 \
  --max-images 16 --output-dir /output \
  --quality-family /quality/family.frozen.json \
  --quality-family-sha256 5942caee9cbe9001995bd2fb3a0dbdd07da6870eca938bbe6a6ff5d7e3a6e25f
```

The helper reads actual installed package metadata, run receipt and adapter
bytes; it rejects unfinished runs, changed source/image, mismatched processor
budgets, adapter bytes or hybrid coverage. It checks every archive file against
the mounted source and records exact core file hashes before importing the
immutable validators. It rejects unarchived Python or unreviewed bytecode in
the core source tree. It writes new mode-0600
`candidate.runtime.bound.json`, `untuned.runtime.bound.json` and
`runtime-binding.receipt.json`; existing outputs are never overwritten.
Both configs share the exact generation/processor/grammar/full comparison profile.
This binding does not load the model or rehash the base shards; the actual
candidate launcher performs the complete shard check before loading.

Mount `candidate.runtime.bound.json` at `/run/configs/qwen38-runtime.json`
in the GPU container and use the same bound configuration in API/worker
containers. Keep `production_selected=false`: a two-step reload checkpoint
and generated smoke are still an unselected candidate, with quality and
timing selection pending. Count the launcher's actual synthetic calls under
the fresh central attempt reservation before starting it.

Actual package metadata was collected from this exact a28 image on
17 October 2026 at 14:00:19 UTC in a CPU-only `runc` container with network
`none`, no GPU/device requests, no source/model/private-input mounts and a
single read-only public `/ops` mount. The temporary container was removed;
image and serving were unchanged. Exact versions are torch `2.13.0+cu129`,
Transformers `5.17.0`, PEFT `0.21.2`, FastAPI `0.136.3`, uvicorn `0.54.0`,
Pillow `12.3.0`, jsonschema `4.26.0` and bitsandbytes `0.50.2`.
Receipt: `infra/qwen38-candidate-serving-package-check-20261017.json`.

The following Linux shell commands are prepared for the root operator after
phase3 training completes. They do not provision compute, authorize GPU use
or alter the immutable sources. Bind the exact values from the completed run
before use; all directories must already exist with the required permissions.
The source archive remains immutable throughout.

```sh
IMAGE="sha256:$SELECTED_IMAGE_SHA256"
PYTHON=/opt/forte-features/bin/python
OPS="$FORTE_QWEN38_OPS_ROOT"
MODEL=/data/forte-models/Qwen3.8-27B-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
# SELECTED_IMAGE_SHA256 is independently inspected after actual GPU proof.
# SOURCE_ROOT, SOURCE_ARCHIVE, SOURCE_SHA256, RUN_ROOT, OUTPUT, CACHE and
# QUALITY_METADATA and OPS are explicit paths/pins; never infer them from an
# old run. Stage the exact new binder helper; the historical d417 helper does
# not implement the final native grammar/family binding policy.

# CPU-only binding: actual packages/adapter/run/source, no CUDA or model load.
docker run --rm --runtime runc --network none --read-only --user 1000:1000 \
  --cap-drop ALL --security-opt no-new-privileges:true --cpus 2 --memory 2g \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/src \
  -v "$OPS:/ops:ro" -v "$SOURCE_ROOT:/app:ro" \
  -v "$SOURCE_ARCHIVE:/release/source.tar:ro" -v "$RUN_ROOT:/candidate:ro" \
  -v "$QUALITY_METADATA:/quality:ro" \
  -v "$OUTPUT:/output" --entrypoint "$PYTHON" "$IMAGE" -B \
  /ops/qwen38_bind_runtime.py --code-root /app --source-archive /release/source.tar \
  --source-sha256 "$SOURCE_SHA256" --run-root /candidate \
  --image-sha256 "$SELECTED_IMAGE_SHA256" \
  --max-pixels 262144 --max-new-tokens 4096 --max-context-tokens 32768 \
  --max-images 16 --output-dir /output \
  --quality-family /quality/family.frozen.json \
  --quality-family-sha256 5942caee9cbe9001995bd2fb3a0dbdd07da6870eca938bbe6a6ff5d7e3a6e25f

# CPU-only full shard check. This really reads every base shard, plus all
# pinned config/tokenizer/processor/template/adapter/run/source bytes.
docker run --rm --runtime runc --network none --read-only --user 1000:1000 \
  --cap-drop ALL --security-opt no-new-privileges:true --cpus 4 --memory 4g \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/src \
  -v "$SOURCE_ROOT:/app:ro" -v "$SOURCE_ARCHIVE:/release/source.tar:ro" \
  -v "$MODEL:/models/Qwen3.8-27B:ro" -v "$RUN_ROOT:/candidate:ro" \
  -v "$OUTPUT/candidate.runtime.bound.json:/run/configs/qwen38-runtime.json:ro" \
  --entrypoint "$PYTHON" "$IMAGE" -B -m serving.qwen38_launch \
  --config /run/configs/qwen38-runtime.json --verify-only

# GPU start ONLY after the root's exclusive release and counted warmup
# reservation. No host ports; forte_private must have Internal=true.
docker run --rm --name forte-qwen38-candidate --network forte_private \
  --read-only --user 1000:1000 --gpus device=0 --cap-drop ALL \
  --security-opt no-new-privileges:true --cpus 8 --memory 90g --shm-size 2g \
  --tmpfs /tmp:size=6g,nosuid,nodev,exec,mode=1777,uid=1000,gid=1000 \
  --tmpfs /run/forte:size=2m,noexec,nosuid,nodev,mode=0700,uid=1000,gid=1000 \
  -e HOME=/cache -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/src \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e WANDB_DISABLED=true -e DO_NOT_TRACK=1 -e TOKENIZERS_PARALLELISM=false \
  -e XDG_CACHE_HOME=/cache/xdg -e HF_HOME=/cache/hf \
  -e TRITON_CACHE_DIR=/cache/triton -e CUDA_CACHE_PATH=/cache/cuda \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e "FORTE_RUNTIME_IMAGE_SHA256=$SELECTED_IMAGE_SHA256" \
  -v "$SOURCE_ROOT:/app:ro" -v "$SOURCE_ARCHIVE:/release/source.tar:ro" \
  -v "$MODEL:/models/Qwen3.8-27B:ro" -v "$RUN_ROOT:/candidate:ro" \
  -v "$OUTPUT/candidate.runtime.bound.json:/run/configs/qwen38-runtime.json:ro" \
  -v "$CACHE:/cache" --entrypoint "$PYTHON" "$IMAGE" -B \
  -m serving.qwen38_launch --config /run/configs/qwen38-runtime.json
```

The candidate backend has one nonblocking inference lock. Configure the
application for one active clip and one in-flight request to this model;
additional simultaneous model calls return busy. Do not enable staged
screening or a helper profile silently while claiming the full-factor
untuned/tuned comparison. Quality selection and cutover remain root-owned.
