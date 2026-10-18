# Isolated feature and W1 training runtime

The live API, worker, Qwen image and 18 October **11:01:25 +05:00** stop remain
unchanged. This is a separate executable research runtime on the existing rental;
it neither selects a new production model nor establishes the under-ten-second
target. W1 uses a frozen image encoder and a small temporal head. It does not
fine-tune the eight-billion-parameter Qwen model.

## Ownership and current capability

Serialize work: A18 releases the fully drained measurement window, A07 downloads
and hashes the two public encoders, A15 measures the encoder and creates dense
features, then A09 trains/checkpoints the head and evaluates the frozen validation
partition. A02's central exposure registry must record the frozen train/validation
members before feature calls. Final-check contents remain excluded. Only the
explicit machine-experiment release authorizes W1; machine targets remain silver
or unknown and do not become reviewed fact/event truth.

A07 verified eight artifacts totaling **3,008,466,624 bytes** under
`/data/forte-fast-models`; the data disk then had **252,658,270,208 bytes** free.
These snapshots are separate from the verified Qwen weights. No new rental or
storage allocation is required. Private feature caches, labels, source metadata,
checkpoints and run logs stay on the team host or in the restricted local backup.

The existing derived image
`sha256:046137dc151b7741b52fbb438de7d606fa74694969c7bb4ab6642d9408b9516e`
already supplies Torch **2.13.0+cu129**, Transformers **5.17.0**, Torchvision
**0.28.0+cu129**, Pillow **12.3.0**, NumPy **2.2.6** and Safetensors **0.8.0**.
It can run A15's generated encoder benchmark in a new one-shot container. Its
missing PyAV package prevents direct MOV decoding. Do not install packages into
the running model container or reuse the Windows/CPU application's environment.

`deploy/Dockerfile.features` preserves the exact tested CUDA12.9 image digest.
It adds an isolated system-site venv with only hashed PyAV **19.0.1**, PEFT
**0.21.2** and Accelerate **1.15.0**, using `--no-deps --only-binary :all:`.
The Linux CPython3.12+ ABI3 PyAV wheel is **34,983,451 bytes**; no Torch,
Transformers, CUDA or driver replacement is requested. PEFT/Accelerate keep the
existing training runtime check usable; the frozen head itself does not need
LoRA. Build checks inspect package metadata without initializing CUDA.
[PyAV release metadata](https://pypi.org/pypi/av/19.0.1/json),
[Docker GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).

## Prepare code and private inputs

From the local repository, after the code owners finish their interfaces:

```powershell
.venv/Scripts/python.exe deploy/package_ml_code.py --output tmp/forte-ml-code.tar.gz
```

The archive allowlist includes executable Python, generic taxonomy, immutable
encoder manifests and runtime locks. It excludes media, labels, review records,
secrets, model weights and checkpoints. Its adjacent integrity manifest records
every archived file hash. Transfer it to a new `/home/ubuntu/forte-ml-code`
directory and verify the outer SHA256 before extraction. Preserve that exact
manifest with each experiment; later code changes require a new package binding.

Root prepares a separate private TAR containing only flat `64hexsha.mov` regular
members and a source-bound input manifest. Original filenames never enter the
encoder. Transfer private bytes without printing them; verify the private outer
archive digest and each MOV's filename digest with:

```sh
umask 077
mkdir -p /data/forte-ml/private /data/forte-ml/runs /data/forte-feature-cache
python3 /home/ubuntu/forte-ml-code/deploy/stage_training_sources.py \
  --archive /data/forte-ml/private/sources.tar \
  --target /data/forte-training-sources \
  --expected-sha256 "$PRIVATE_SOURCE_ARCHIVE_SHA256"
```

The stager rejects links, traversal, duplicate names, mismatched hashes and files
outside its bounded flat layout. It keeps and re-verifies matching existing
sources. It never overwrites differing content. Its stdout contains aggregate
counts only. Do not delete the original sources or private archive before the
restricted backup has been verified.

CPU FFprobe creates the real `VideoAsset` metadata. A05's GPU-side PyAV batch
reader receives that metadata with its source hash and container-local path
binding, so the feature image does not need another FFprobe installation.
Preserve original integer PTS, complete native coverage, global views and the
tail. Never infer targets or presence from complete feature coverage.

## Build and run separately

Run the following on the team host only after A15 releases its synthetic window;
do not compete with the encoder download or a measured GPU run:

```sh
cd /home/ubuntu/forte-ml-code
docker build -f deploy/Dockerfile.features -t forte-features:cu129-w1 .
docker image inspect forte-features:cu129-w1 --format '{{.Id}}'
```

Set `FORTE_FEATURE_IMAGE` in a private operator copy of `deploy/ml.env.example`
to that inspected immutable image ID. For the initial synthetic-only encoder
benchmark, the existing derived image above is sufficient. The Compose file is
**standalone**: use project `forte-ml`, never merge it with production Compose,
and use `run --rm` rather than `up`. It has no published ports, `network_mode:
none`, a read-only root and code/weight/input mounts, writable separate caches
and checkpoints, dropped capabilities and no Docker log retention.

Container paths are fixed:

| Host input/output | Container path | Access |
|---|---|---|
| `/home/ubuntu/forte-ml-code` | `/app` | Read-only public code |
| `/data/forte-fast-models` | `/encoders` | Read-only verified weights |
| `/data/forte-ml/private` | `/private` | Read-only private config/labels/metadata |
| `/data/forte-training-sources` | `/private/sources` | Read-only original MOVs |
| `/data/forte-feature-cache` | `/cache` | Private feature cache writes |
| `/data/forte-ml/runs` | `/runs` | Private checkpoints/run logs |

With `/data/forte-ml/private/runtime.env` containing those operator paths and
image ID (keep this operator file outside the source/build context):

```sh
docker compose --env-file /data/forte-ml/private/runtime.env -f deploy/compose.ml.yaml \
  --profile metadata run --rm metadata -m deploy.feature_runtime_check --require-av --training

# Actual Torch/Safetensors CPU roundtrip, corruption and incomplete-tail checks.
# This service has no GPU reservation and this test allocates CPU tensors only.
docker compose --env-file /data/forte-ml/private/runtime.env -f deploy/compose.ml.yaml \
  --profile metadata run --rm metadata -m unittest training.test_feature_cache.CacheTensorTests -v

# A15 only, after measured headroom and explicit GPU ownership.
docker compose --env-file /data/forte-ml/private/runtime.env -f deploy/compose.ml.yaml \
  --profile encoder run --rm encoder-bench

# A09 metadata gate first; no training or CUDA from --preflight.
docker compose --env-file /data/forte-ml/private/runtime.env -f deploy/compose.ml.yaml \
  --profile head run --rm head-train

# A09 only after A15 extraction completion and cache/config checksum freeze.
docker compose --env-file /data/forte-ml/private/runtime.env -f deploy/compose.ml.yaml \
  --profile head run --rm head-train -m training.temporal.train_weak --config /private/W1.local.json
```

The `feature-cache` service deliberately defaults to a metadata check. Root's
explicit extraction CLI and pinned private config override its command only once
the source and exposure gates pass. Keep private diagnostic logs inside `/runs`,
not in public artifacts or terminal output. A09 freezes the candidate checkpoint
and predeclared binary threshold before validation labels are opened. A result
remains a candidate until quality and full-service latency are measured.

## VRAM and failure handling

The A100 has 81920 MiB. Qwen's `gpu_memory_utilization=0.85` is a configuration
budget of about 68 GiB, not an observed free-memory reading. After drain, A15 must
measure actual device free memory and baseline allocation before a new encoder
process. Start one B16 encoder at batch 32. Estimated vision weights are 177 MiB
(B32 about 180 MiB); a **1–4 GiB working allowance** and an initial **8 GiB maximum
planning reservation** for a feature/head run are unmeasured assumptions.
Retain at least 2 GiB measured margin; start no second encoder simultaneously.
Record actual allocated/reserved peak, not estimates, before increasing batches.

If the measured memory budget is insufficient, stop the isolated experiment and
report the resource need. Do not shrink/restart Qwen, change its maxseq2 setting,
or allocate another rental automatically. An OOM is a measured failed condition;
do not silently drop frames or change a benchmark batch. Full 8B LoRA training
has a different memory requirement and is outside this concurrent-head plan.

After extraction/training, verify no isolated GPU process remains and that the
unchanged public model attestation/readiness still passes. Root coordinates any
user upload pause. Only in that stable window create the prepared restricted
database/results/audit/config backup using the second-account handoff guide.
The existing 11:01 stop, current-account USD50 cap and separate future-account
budget remain in force. No source or jury data is automatically deleted.

Local preparation checks passed Compose parsing, Python syntax and seven
synthetic safe-staging cases (fresh/idempotent extraction, outer/member hash
mismatch, traversal, symlink and no escaped file). The actual feature image then
built as `sha256:1bcad5aaa9c13266e1adc9f9ce302e29aaaeacccbe1354a84cb984e611977b27`.
The inherited vendor `UV_OVERRIDE` required clearing for the three explicit
hashed `--no-deps` additions; inherited NCCL/Torch were preserved. All nine
pinned package versions passed inside the UID1000 container. Actual CPU
Torch/Safetensors roundtrip, corruption and incomplete-stream checks passed.

The final deterministic 32-generated-frame B16 smoke passed with a 373 MiB peak
allocated and 414 MiB reserved, with 9.16 GiB free. Its source recipe binds both
encoder/stream implementation files and the deterministic policy. This is a
small synthetic correctness check, not a full-service performance claim. Numeric
evidence is in `infra/feature-runtime-validation-20261017.json`.

The frozen 134-file public code package is SHA256
`dbf6b4c4ff49fb5f3820d27fca8027e003805b7a1202916d91d6d37a83b7e13a`.
Actual native extraction then completed all 34 authorized sources, 28 train and
6 validation, producing 35,186 frames in 84.214 seconds for the entire batch. Every
source was hash-verified and the cache requires complete original PTS/tail
coverage. The private index at `/cache/w1-native-b16/index.json` is SHA256
`61fca47d516e480bf137e703f5ea1c30624f12eb1cc78b638fc4d09f154a7d82`.
Aggregate proof is `infra/feature-cache-extraction-20261017.json`. Training,
candidate quality and full-service latency require their separate receipts;
the production Qwen service was unchanged.
