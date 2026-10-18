# GPU and deployment runbook

Updated 17 October 2026, Asia/Qyzylorda. A resource marked running is not proof of model readiness.

## Verified state

The Windows workstation has an AMD Ryzen 5 8645HS, 13.79 GiB RAM, and AMD Radeon 760M graphics. Windows CIM reports a 2 GiB adapter-memory field; this is not a measurement of dedicated VRAM. No NVIDIA GPU or `nvidia-smi` was found locally. WSL2 Ubuntu works. Docker Desktop was installed but initially stopped; it now runs Docker Engine 29.6.2 with Compose 5.3.1.

The user authorized a **USD 50 cap for the current Brev account** and separately described USD 50 on another account for demo day. That future allocation is not available to this deployment and must not be combined with the current cap. The user explicitly approved the reviewed Deploy action and provider terms before launch.

One Brev/CRUSOE instance was created at **2026-10-17 15:01:25 +05:00**:

| Field | Observed value |
| --- | --- |
| Name / ID | `forte-a100-smoke-20261017` / `wd88a7vvt` |
| Actual GPU | NVIDIA A100 80 GB PCIe, `81920 MiB` from remote `nvidia-smi` |
| Actual driver | `565.57.01` |
| Catalog host | 12 vCPUs, 120 GB RAM, x86_64 |
| Assigned region | Abilene, Texas, USA |
| Prepared disk / price | 256 GiB, USD 2.01/hour displayed |
| Running dashboard disk / price | 384 GiB, USD 2.03/hour: 1.98 compute + 0.05 storage |
| Host filesystems | 124 GiB root, 37 GiB free after images/weights; separate 251 GiB `/data` mount with 239 GiB available |
| Software | Raw VM with Python/CUDA/Docker; Jupyter disabled |
| Connectivity | Authenticated WSL SSH verified; no private dataset transferred |
| Runtime guard | At most 20 hours from creation; USD 40.60 at the observed rate, before unverified extra charges |
| Stop behavior | Provider says data persists; storage continues billing while stopped |
| Model state | 15 artifacts / 17,545,914,364 bytes verified; real CUDA/BF16, warmup and strict-schema synthetic inference passed |

The running disk/rate differs from the prepared quote. Use the actual dashboard rate for cost tracking. Taxes, egress, minimum billing increment and deletion billing remain unverified. Read-only billing inspection confirmed no connected payment method and USD 49.99 credit remaining at that check; no billing arrangement was changed. Do not add payment methods or enable automatic credit refill. A 20-hour run leaves USD 9.40 before unknown charges; this is a ceiling for preparation, not a reason to leave the GPU idle. The 20-hour guard expires **18 October 2026 at 11:01:25 +05:00**. The project operator/root integration owns cleanup; the judging date remains unresolved. Preserve required artifacts and stop idle compute earlier; stopped storage must also be cleaned up within the account cap.

The Windows scheduled task `Forte-Brev-budget-stop-wd88a7vvt` is installed and verified Ready for that cutoff. It invokes `infra/budget_guard.py` in the existing WSL profile, checks the exact organization/name/instance ID, and stops only that instance. An actual Task Scheduler dry run completed with `LastTaskResult=0`; the early-execution check prevented stopping the running instance. Observed settings are `WakeToRun=true`, `StartWhenAvailable=true`, a 30-minute execution limit, and ten retries at two-minute intervals. It survives closing Codex and requests wake from supported sleep; **the workstation must remain powered, connected to the network, and its Brev CLI must retain the current account login**. Wake depends on Windows power/firmware support and cannot recover a shut-down or disconnected PC. Use a separate profile for the future account. It does not delete data. Stopped storage still costs USD 0.05/hour, so export/retention and deletion must be handled within the remaining cap, conservatively within seven days after the stop. A workstation timer is not a provider-guaranteed billing cap.

The judging deadline is now known: **18 October at 12:00 noon +05:00**. The user explicitly chose to keep the 11:01:25 stop and rejected a proposed extension. The current public service will therefore stop before noon unless a separately authorized second-account deployment is verified and cut over first. No second rental or account switch has been performed. See `docs/second-account-demo-handoff.md` for the restricted backup/restore procedure and continuity gate.

Detailed nonsecret resource state is in `infra/resources.json`; the dated live comparison is in `infra/brev-quotes-20261017.json`. They contain no credentials. `infra/spending-policy.example.yaml` preserves unknown values as null rather than inventing authorization.

## Connection and ownership

The official Brev CLI `v0.6.337` is installed inside the existing WSL Ubuntu distribution at `/home/wezeso/.local/bin/brev`. Authentication remains in the user's local WSL profile. Never copy credentials into Git, docs, screenshots or agent messages.

From PowerShell:

```powershell
wsl -d Ubuntu -- ssh -o BatchMode=yes -o ConnectTimeout=12 forte-a100-smoke-20261017 'nvidia-smi'
wsl -d Ubuntu -- bash -lc '"$HOME/.local/bin/brev" ls --json'
```

SSH uses the CLI-managed configuration, remote user `ubuntu`, and a local identity-file reference under `~/.brev/ssh-certs/`. `brev refresh` eventually completed but was slower than direct `brev exec`; avoid duplicate retries while a launch/refresh is pending. There is exactly one instance.

A code-only archive was uploaded to `/home/ubuntu/forte-deploy.tar.gz` and extracted into `/home/ubuntu/forte`. Packaging uses `python deploy/package_code.py`, an explicit allowlist. It omits videos, labels, review records, secrets, weights and local caches. A07 owns downloading/verifying the pinned model into `/home/ubuntu/forte-models/Qwen3-VL-8B-Instruct`. A15 owns the GPU image pull, CUDA/BF16 operation, warm model launch and synthetic inference verification. A17 owns CPU deployment and ingress. Do not start a second weight download or model process.

## Local/remote Compose deployment

Run commands from the repository root. CPU API and worker use the same image and durable `/var/lib/forte` volume. PostgreSQL, workers and model have no published host ports and join an internal Docker network. Caddy alone joins ingress. The GPU profile uses a read-only verified local model snapshot and persistent compilation caches. The API never loads GPU weights.

```powershell
python deploy/prepare_secrets.py
python deploy/verify_config.py
docker compose --env-file deploy/.env.example -f deploy/compose.yaml build
docker compose --env-file deploy/.env.example -f deploy/compose.yaml up -d
```

`prepare_secrets.py` creates random local files once and preserves them on rerun; values are never printed. On POSIX its parent directory is private (`0700`), and mounted secret files use Docker-compatible read permissions so the non-root service UID can read them. The API key is a runtime secret file, not a frontend asset. The DB password is read at process startup by `deploy/entrypoint.py`; no DSN password appears in Compose configuration. Keep the files outside a release archive and retain them securely with the persistent database.

For the authorized GPU host, use a deployment-only `.env` with `FORTE_MODEL_SNAPSHOT=/home/ubuntu/forte-models/Qwen3-VL-8B-Instruct`, then:

```sh
docker compose --env-file deploy/.env -f deploy/compose.yaml -f deploy/compose.gpu.yaml --profile gpu up -d --build
docker compose --env-file deploy/.env -f deploy/compose.yaml -f deploy/compose.gpu.yaml --profile gpu ps
docker compose --env-file deploy/.env -f deploy/compose.yaml exec worker python -m forte.worker --check
```

The generic configuration leaves `FORTE_TRUSTED_SCENE_CONTEXT` empty. For uploads explicitly within the organizer's bank cash workstation case, the operator may set `FORTE_TRUSTED_SCENE_CONTEXT=bank_cash_workstation`. This scopes applicability; it does not identify the bank or establish visible facts. Unsupported domains must retain the empty default.

The live worker selects `FORTE_COMPACT_FRAME_IDS=true` and `FORTE_MAX_INFLIGHT_PER_JOB=2`, retaining two concurrent clips globally and the model's 4096-token generation limit. The repository defaults remain `false` and `1` until an operator chooses the measured profile. A15's eight synthetic 19-fact/16-frame requests all passed strict validation: the two-request pair took 53.565 seconds with raw IDs/inflight 1, 26.211 with raw IDs/inflight 2, 25.336 with compact IDs/inflight 1, and 14.547 with compact IDs/inflight 2. These are synthetic serving timings, not end-to-end video performance or accuracy.

The initial local binding is `127.0.0.1:8080`; it is not a public deployment. The GPU host's app ingress is bound for the provider gateway at `https://forte-review-wd88a7vvt.gobrev.dev`. Only app port 8080 is shared; API keys still protect job/upload routes. Model/database ports remain private. The existing Dokploy server has a separate `Forte Datathon` project and `Forte Ingress` Compose service for `hackathon.wezeso.me`, with Let's Encrypt TLS. `deploy/dokploy-ingress.yaml` routes through the verified-TLS Brev gateway without transferring GPU SSH credentials to Dokploy. No existing unrelated application was modified. The entire API/worker/database/model stack stays on the GPU host so uploaded media uses shared local storage.

The live frontend now uses `deploy/dokploy-ingress.direct-api.yaml`. Its secret-free `/runtime-config.json` returns the trusted Brev API origin with JSON content type and `Cache-Control: no-store`. Browser uploads, polling and protected media use **`https://forte-review-wd88a7vvt.gobrev.dev` directly**; use that base for jury/scripts too. The API permits CORS only from `https://hackathon.wezeso.me`, while actual API-key checks remain mandatory. Generated29MB uploads through the original public proxy still failed near32 seconds before sending the full body. An explicit HTTP/1.1 upstream test did not improve that failure and was rolled back. The direct-browser route bypasses that observed path; it does not establish its underlying cause.

## Versions and reproducibility

The CPU build pins Python `3.12.15-slim-trixie`, uv `0.13.1`, Node LTS `24.22.0-bookworm-slim`, PostgreSQL `18.6-trixie` and Caddy `2.11.7-alpine`. Application Python and npm dependencies use the repository lockfiles. PostgreSQL 19 RC and Node 26 Current were not selected for production. The GPU build extends vLLM `0.31.0-cu129` with the verified registry index digest; the model checkpoint and artifact hashes are pinned in the model manifest. The default CUDA 13 image failed on driver 565.57.01. The CUDA 12.9 variant passed an actual BF16 tensor operation; its unused optional `torchcodec` package was removed because that bundled binary required CUDA 13. JPEG inputs use the CPU decoder instead. Pinning establishes repeatability, not inference quality.

The Dockerfile builds UI assets and copies only generic taxonomy code plus the public legal source archives required by checksum verification. `.dockerignore` excludes the private dataset, labels, review records, secrets and weight formats. No runtime CDN or external AI endpoint is required.

## Checks and rollback

Client-side Compose validation passed for both CPU services and the optional GPU profile, including private networks, host ports and persistent volumes. Actual local PostgreSQL 18.6 QA passed 26 synthetic tests in 13.46 seconds under an isolated temporary schema, including concurrent claims, held-row `SKIP LOCKED`, lease expiry, fencing and restart behavior. The temporary loopback PostgreSQL port/test network was removed afterward. CPU image builds succeeded locally and remotely; API/database/worker health checks passed. The Brev app gateway rejects unauthenticated job requests with HTTP 401 and returns honest HTTP 503 readiness while the model warms. The provider reserves `/healthz` at its edge, so its plain `OK` response alone does not establish app health.

At 15:47 +05:00 the final CPU refresh recreated only API and worker with image `sha256:97fd2dd3a5dafd63309fab12868925a8c6c39b363703ae0a5809ad6ff28afdd9`. It includes the current client lock, compact-JSON prompt, pipeline relevance handling, worker error classification and UI readiness guard. Both containers were healthy, selected environment values were checked inside the worker, and public readiness returned HTTP 200. The GPU model's start time remained `2026-10-17T10:36:11.954566283Z`; the CPU update did not restart it.

At 16:30 +05:00 a later approved candidate became active: `sha256:c4528d19724dff4477f9c4f3e9bfb6c43b2075a5428eb5192e632b36283e48c8`. It adds direct-API CORS/bootstrap support and selects staged screening plus concise observations, retaining compact IDs/two inflight per job/two concurrent clips. Actual worker settings, source hashes, frontend asset bytes, strict readiness200, approved-origin preflight204, missing-key401 and forbidden-origin403 were verified. Model identity/start time stayed unchanged and running/waiting gauges were0/0 before A18's paired trial release. The original benchmark image and prior candidate remain available; no performance numbers from the old image are attributed to this one. See `infra/direct-api-deployment-20261017.json`; the tightened under10-second end-to-end target remains unverified until the real trial.

The current human-review release is
`sha256:64f56ae74e2c5824dbd20cac9a1540738d3467c0b57a0362ddfac0dd70ec39cc`.
It replaces only API/worker, adds the durable review ledger and serves the tested
human-confirm/reject interface. It preserves original jury-result bytes and
records a reviewer's self-reported name separately. The measured worker profile,
direct Brev API/CORS, Qwen image/start time and budget stop are unchanged. Live
checks passed exact frontend asset SHA256, strict readiness200, preflight204,
unauthenticated401, forbidden-origin403 and the real read-only review routes. A
negative result correctly rejects operational review with409, with its persisted
jury bytes verified unchanged. No real user review was written during checks.
Receipts are `infra/human-review-deployment-20261017.json` and its referenced
external/runtime reports; c452 remains the rollback image. A18's prior c452
paired trial completed 5 of 6 requests with median 26.485 seconds/max 149.597 seconds;
the under-10-second target was not achieved, and these numbers do not describe
an independent new benchmark of this human-review image.

The separate W1 encoder/head experiment completed native feature extraction,
actual optimizer steps and deterministic validation. Its candidate scored 2 of 6
versus the untrained reference 3 of 6 and remains unselected. The production model
stays on Qwen. The subsequent bounded 32-call TRAIN teacher pass uses a separate
private CPU runner on `forte_private`, pinned c452 and content-addressed code;
its outputs are unreviewed proposals. Do not use them as human truth or promote
a new model without the explicit evaluation/calibration gates.

At 15:38:56 +05:00 on 17 October, an actual fixed-IP TCP probe from `forte-model-1` could not connect to `1.1.1.1:443`; it sent no application payload. Docker inspection confirmed `forte_private` is internal and the model has no published ports. Offline environment variables were active. A paired request in the same container generated three synthetic JPEG frames and passed strict schema, semantics and known visual-marker validation in 1.9243 seconds (148 generated tokens). This establishes actual local synthetic inference while that service's external route is blocked; it does not measure event accuracy or video latency. Aggregate evidence is in `infra/offline-proof-20261017.json`. Public `https://hackathon.wezeso.me/readyz` returned HTTP 200 with every component true afterward.

Reproduce only outside a reserved video benchmark window, from the remote repository:

```sh
docker exec -i forte-model-1 python3 - < deploy/offline_probe.py
docker network inspect forte_private --format '{{.Internal}}'
docker inspect forte-model-1 --format '{{json .NetworkSettings.Ports}}'
docker exec forte-model-1 python3 -m forte.model smoke --base-url http://127.0.0.1:8000/v1 --media-root /tmp/forte-offline-smoke --timeout 120
```

Before release, verify one authenticated upload/poll cycle, evidence JSON validation, restart persistence and an uncached real authorized video locally. No private footage or content-dependent outputs may be shown in a cloud screenshot or returned in tool output. A15 owns the next synthetic serving benchmark; A18's subsequent private-video benchmark must have an exclusive measurement window without competing synthetic inference.

For rollback, retain the previously validated application image and manifest, stop only the affected worker/API, and recreate those services with the previous image. Named volumes preserve job state and uploads. A base-file-only CPU update may call the model an orphan; preserve it and use `--no-deps` rather than `--remove-orphans`. Never use `down -v` during routine rollback. Do not report processing as ready until an external authenticated upload/result smoke test passes; the live upload UI alone does not establish inference readiness.

## Primary references checked

The separate encoder/cache/W1 runtime and exact one-shot commands are documented
in [isolated-feature-training-runtime.md](isolated-feature-training-runtime.md).
It preserves the production model and budget stop, uses a code-only source
package and separate private mounts, and requires serialized GPU ownership.

- [Brev lifecycle and SSH refresh](https://docs.nvidia.com/brev/cli/instance-management), [official CLI installer/source](https://github.com/brevdev/brev-cli), and [live Brev console](https://brev.nvidia.com/).
- [Docker Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/) and [profiles](https://docs.docker.com/compose/how-tos/profiles/).
- [PostgreSQL supported stable versions](https://www.postgresql.org/support/versioning/) and [official container's PostgreSQL 18 volume change](https://hub.docker.com/_/postgres/).
- [uv container builds](https://docs.astral.sh/uv/guides/integration/docker/), [uv 0.13.1](https://github.com/astral-sh/uv/releases/tag/0.13.1), [Node release status](https://nodejs.org/en/about/previous-releases), [Caddy 2.11.7](https://github.com/caddyserver/caddy/releases/tag/v2.11.7), and [vLLM container deployment](https://docs.vllm.ai/en/stable/deployment/docker/).
