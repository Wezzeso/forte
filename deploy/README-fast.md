# Fast analyzer deployment

The fast profile runs one CPU worker with four analysis slots and a private
vLLM service with 64 request slots, bf16 weights, and explicit prefix caching.
These limits do not establish a latency result: upload and queue time share
the existing 285-second job deadline. Run the quality suite and the public API
parallel-upload benchmark on the destination host before changing routing.

## Existing provider access

A read-only check on **18 Oct 2026 at 02:35 Asia/Qyzylorda** found one
authenticated NVIDIA Brev organization and one reachable running instance,
`forte-a100-smoke-20261017`: A100 80 GB, 81,920 MiB total VRAM, 7,181 MiB free
while the existing model was loaded. API, worker, and database were healthy;
`/readyz` reported all components ready. This is a timestamped observation,
not a promise that the host remains running.

The second GPU account was **not available in the configured Brev CLI**.
An additional read-only audit found one local CLI credential set, one Brev
SSH target, and no Brev tab in the connected Chrome profile. No second
authenticated target was discoverable; unconnected browser or CLI accounts
cannot be inspected from this session.
The local budget-stop task remained scheduled for **18 Oct 2026 at
11:01:25 Asia/Qyzylorda**, with last task result zero. Bootstrap does not
switch accounts, rent resources, change that guard, or move public routing.
The operator must supply access to an already-authorized destination before
a second-account migration can run. Read-only inventory commands are:

```bash
brev org ls
brev ls
brev exec forte-a100-smoke-20261017 "nvidia-smi"
```

## Prepare an authorized destination

Use Linux, Docker Compose, NVIDIA Container Toolkit, and an idle NVIDIA GPU
with bf16 support. Preflight defaults require at least 75,000 MiB total and
70,000 MiB free VRAM for the current 30B bf16 profile. They deliberately fail
on the occupied source host. Different hardware or model sizes need their
own measured configuration; the script makes no 30-second guarantee.

Copy the repository to the target, then edit `deploy/.env` after copying
`deploy/.env.example`. Set `FORTE_FAST_MODEL_REVISION` to a reviewed immutable
40-character commit SHA from the official Hugging Face model repository.
No revision is supplied here because an invented or unreviewed pin would
defeat reproducibility. `main` and an empty revision fail before a download.
`FORTE_FAST_VLLM_SOURCE_IMAGE` can be set to a reviewed image digest.

```bash
cp deploy/.env.example deploy/.env
# Edit deploy/.env, including FORTE_FAST_MODEL_REVISION and the model path.
bash deploy/bootstrap_fast_host.sh
```

Bootstrap preserves existing credentials and the bind address. The default
bind is `127.0.0.1`; set `FORTE_BIND_HOST`, the site address, and exact CORS
origins for the intended ingress before exposing the destination. Initial
Docker pulls, application builds, and the model download require internet.
Weights download at the exact revision into a staging directory and are
promoted after validation. Interrupted downloads can resume in that staging
directory. A pre-existing serving snapshot with another revision, missing
receipt, or missing shard fails rather than silently reusing its files.
Use a fresh destination directory when the old snapshot's provenance is
unavailable.

The provenance receipt `.forte-model-receipt.json` records the exact revision
resolved by Hugging Face during download. It is an operational record;
preflight checks required files and shard readability, not a fresh hash of
every weight byte. Keep the receipt and reviewed upstream pin with backups.

## Preflight and readiness

Before starting services, bootstrap executes the equivalent of:

```bash
python3 tools/fast_host_preflight.py \
  --model-dir /data/models/Qwen3-VL-30B-A3B-Instruct \
  --model-revision "$FORTE_FAST_MODEL_REVISION" \
  --image forte-model:vllm-0.31.0 --gpu 0
```

The tool checks the local model receipt, metadata, and every referenced
safetensors shard; actual GPU memory; Docker/Compose access; the local image
identity; CUDA allocation and bf16 support inside that image; and model-file
readability inside the same read-only bind mount. The temporary probe has
no network, never loads the model, and never generates a verdict.
`--snapshot-only` skips GPU checks and reports `ready: false` even on success.
It is useful for copying weights, not evidence that a deployment is ready.

The running VLM has a read-only root filesystem and model mount, dropped
capabilities, writable temporary caches, and the internal Compose network.
Only Caddy has an external network route. Hugging Face offline and telemetry
flags are explicit. No remote-code loading flag is enabled. Do not attach
the model or worker to an external network for inference.

Bootstrap applies a bounded startup/readiness wait (default 900 seconds,
configurable with `FORTE_FAST_READY_TIMEOUT_SECONDS`). A startup failure or
timeout exits nonzero and preserves containers for inspection. A zero exit
requires `/readyz`, including model and worker checks, to succeed.

## Admission and acceptance

`Worker.run()` enforces four jobs **per process**. The fast overlay uses a
fixed container name and one replica so Compose rejects scaling that worker
to multiple replicas. Do not start another worker process or another Compose
project against the same database: that would multiply the limit. The VLM
request semaphore of 64 is separate from the four-video limit. Additional
uploads stay in the durable queue; queue expiry becomes an explicit failure
under the original job deadline.

After readiness, run the full 40-clip evaluation, then five- and six-parallel
uploads through the intended public API. Require no new false alarms and
the configured 240-second end-to-end acceptance bound. A readiness pass
does not prove accuracy, deadline compliance, or that a cutover occurred.
Keep the current public routing until the destination passes those checks.
