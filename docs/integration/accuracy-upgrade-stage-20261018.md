# Accuracy upgrade: isolated full-service stage, 18 Oct 2026

This record covers the verified VLM profile on the existing NVIDIA Brev host.
Second-account migration and new-footage training are deferred at the user's
request. No new violation detector is promoted by this change.

## Qualified artifacts

| Artifact | SHA-256 / identity |
| --- | --- |
| Existing CPU base image | `sha256:30691bd4230396c011f292beab7094bcdf160e7845c417e85bf8d11e8dcbdf32` (`forte-review:fast-v4`) |
| Verified CPU overlay image | `sha256:8c8b92550c5dfc6702efbe266773ea4bfd86639b9f5edd8e86e6a0e3e45b6a3d` (`forte-review:accuracy-v2-dcb5f165`) |
| Candidate source archive | `dcb5f16574df668aaaf6f6011cb0f13f70f0056a0c20027faa8777584dc21ce3` |
| Exact `fast_config.json` file bytes | `993d2f22867576be053d7a05663043077937d5cd749d67709dcab0d52016789c` |
| Effective config, sorted compact JSON | `113323095c2031430f0aff58f3aecc4fe28907dfbe84dfb0eea1d9545a7a7b29` |
| Worker audit config, sorted JSON with spaces | `0bc38acdb31659d0ee525b3c702d2516317e49420b0318237536c2fbccc3a8e2` |
| Existing PostgreSQL image | `sha256:29754c7520f4b9654cf8ff98e19ac73b88e255c3a53f6236ebe9d52249bb6df3` |

The two semantic config hashes differ only in JSON whitespace. Both bind the
same exact reviewed profile: `require_corroboration=true`, `use_verify=true`,
`specialists.enabled=false`, and the existing six disabled violation types.
The file is [the reviewed profile](../../benchmarks/quality/accuracy-upgrade-verified-config-20261018.json).
The source manifest lists all thirteen qualified archive files.

Direct evaluation of that frozen source/profile scored **34/40, zero false
alarms, zero processing errors, and zero previously correct binary verdicts
regressed**, compared with the fresh production baseline of 33/40 and one
false alarm. Six misses remain. This was development-set verification;
it establishes no unseen-video accuracy percentage.

## Isolation and checks

The stage uses brand-new PostgreSQL storage, database/user/password, API key,
upload/result/work directories, and uniquely named containers. API and worker
share only stage state. They reach the already-running `q30` model through
the existing internal `forte_private` network. No model or GPU container is
created, restarted, or reconfigured. The stage worker has no GPU reservation.

The frozen `src/forte/fast` directory is bound directly over the installed
package at `/app/.venv/lib/python3.12/site-packages/forte/fast`. This avoids the
namespace-package shadowing problem that a bare `PYTHONPATH` can introduce.
Imported `analyzer`, `service`, `video`, and `vlm` module paths and file hashes
are checked separately inside both API and worker containers. The exact
reviewed config bytes are written into the new stage volume before startup.

Checks include real PostgreSQL schema initialization; readiness with all five
components true; a protected unauthenticated request returning 401; the `q30`
model-list identity; the effective worker config and database host; and an
absent specialist runtime with neither `torch` nor `rfdetr` imported.

The actual production volume was checked independently: its
`fast_config.json` was absent and `load_config()` returned the baseline profile
with verification disabled. Production's API, worker, database, ingress, and
model container identities/start times, plus deployment-file hashes, are
captured before and after staging. Active and queued production counts are
captured separately without exposing job contents.

## Transport scope

The final stage client reaches the staged API by private Docker DNS. Ingress
is explicitly excluded. No stage port is publicly exposed; API and worker
retain internal-only networking.

Two setup attempts issued **no uploads**: publishing an internal-network API
port was refused by this host, and an isolated Caddy attempt could not execute
under the restricted capability set. Both attempts' containers were stopped
and production bindings stayed unchanged. Their sanitized reports are retained
beside the final report. The approved private-DNS fallback tests the full
API → upload/decoder → PostgreSQL queue → worker → existing model → durable
result/evidence path. Public HTTPS transport remains a separate post-rollout
check.

## Acceptance run

The stage submits the six longest supplied development clips simultaneously.
The client budget starts before upload and ends only after the completed
response validates. It validates matching job IDs, the public response schema,
every evidence JPEG, the 1–3 evidence-frame limit, and event/evidence timestamp
bounds against the original files' `ffprobe` durations. No images,
descriptions, API keys, database passwords, or complete result bodies are
copied into the public report.

PostgreSQL is sampled for actual active leases and queued jobs. The acceptance
gate requires six successful first-attempt durable results, every completed
worker audit bound to the reviewed profile, at most four active leases,
positive evidence-path coverage, and every request below 240 seconds.
The stage containers are stopped after the run; private state remains in the
stage-specific directory and PostgreSQL volume for controlled inspection.

Run result and exact cleanup identities are recorded in
[the sanitized stage report](../../benchmarks/quality/accuracy-upgrade-stage-20261018.json).

**Stage result: PASS.** Six of six uploads completed on the first attempt;
the slowest took **137.990 seconds**. Two detected events exercised six
evidence JPEGs. All JPEG/schema/time-bound checks passed. PostgreSQL sampling
observed no more than four active leases across 130 samples. Production had
zero active/queued jobs before and after, and all protected bindings were
unchanged. The three final stage containers were stopped. Its retained paths
are `/data/claude/staging/forte-accuracy-v2-20261018-223127` and PostgreSQL volume
`forte-accuracy-v2-20261018-223127-pgdata`.

The minimal overlay was then built with network access disabled from a
context containing only ten qualified Python files and its Dockerfile. An
offline import/config check in the built image matched every runtime file,
the four imported module bindings, and the reviewed effective config; no
specialist runtime or Torch/RF-DETR import occurred. See
[the image proof](../../benchmarks/quality/accuracy-upgrade-overlay-image-20261018.json).

## Reviewable rollout and rollback

Root integration accepted staging and authorized the exact-profile rollout.
The rollout/public acceptance runner records its final outcome in
[the rollout report](../../benchmarks/quality/accuracy-upgrade-rollout-20261018.json).
**Rollout result: PASS, completed 18 Oct 2026 at 03:42:52 Asia/Qyzylorda.**
API and worker run the verified overlay image above and the exact reviewed
config. All six public HTTPS uploads completed on the first attempt; the
slowest took **161.068 seconds**. Both detected events and all six evidence
JPEGs passed schema and original-video timestamp checks. Public readiness
returned all five components ready, and a protected unauthenticated request
returned 401. Final active and queued job counts were both zero. No rollback
was needed. Individual credential-file hashes are omitted from the public
artifact; it retains a `credentials_unchanged=true` result, while the private
receipt remains on the host.

Only the API and CPU worker change; the existing model, GPU flags, database,
public routing, API credentials, and 11:01:25 Asia/Qyzylorda budget guard stay
as configured.

1. Verify the frozen source/config hashes and the passing final stage report.
   Build an overlay image from the exact existing CPU base, copying only the
   qualified `forte/fast` files. Run an offline import/hash smoke check on that
   image and record its immutable Docker image ID. Do not use an unbounded
   repository build context containing private benchmark data or footage.
2. Record the current Compose files, CPU image ID, and the absence of the
   production `fast_config.json`. Preserve existing credentials in place.
   Prepare an image-only Compose override for API and worker plus a separate
   API maintenance override setting `FORTE_ADMISSION_PAUSED=true`.
3. Apply maintenance to the API only, then wait for production active and
   queued counts to reach zero. Stop the old worker gracefully. Do not add
   a second production worker against the same database.
4. Create the exact reviewed `fast_config.json` atomically in the existing
   application data volume, with ownership/read permissions for UID 10001.
   Refuse to overwrite an unexpected existing file. Recreate API and worker
   with the qualified overlay image while maintenance remains active, using
   `--no-deps --no-build` so PostgreSQL, ingress, and the model are untouched.
5. Verify both containers' imported module hashes and effective config. Remove
   the API maintenance override, recreate only the API if required, and wait
   for all `/readyz` components. Run the six-upload test through the existing
   public HTTPS origin with the original private API-key file, without logging
   the key or result evidence. Require all six valid results under 240 seconds.
6. If any check fails, pause admission again and drain/stop the candidate
   worker. Remove only the newly created config after checking its exact
   expected SHA-256, restoring the recorded baseline's absence; select the
   original CPU image `sha256:30691bd4230396c011f292beab7094bcdf160e7845c417e85bf8d11e8dcbdf32`.
   Recreate only API/worker, unpause admission, and verify readiness. Keep
   PostgreSQL, accepted jobs/results, the existing model, routing, secrets,
   and the budget guard intact throughout rollback.

Stage cleanup must target only the exact stage-labelled container names,
stage PostgreSQL volume, and stage directory recorded in its report. Never
run a project-wide `down -v` against the shared production Compose project.
Retained uploads/results contain bank footage; delete them only under the
project's authorized retention/cleanup decision.

Prepared release files are under
`/home/ubuntu/forte/deploy/releases/accuracy-v2-dcb5f165`:
`compose.image.json` selects the immutable candidate image;
`compose.rollback-image.json` selects the original image;
`compose.maintenance.json` pauses only API admission; and
`config_transition.py` atomically installs the exact reviewed config or
removes it on rollback only after its hash matches. The production data volume
is discovered from the existing worker mount and never replaced. Both the
stage and rollout scripts retain full result evidence only in the existing
private service storage; their public reports contain numeric/hash metadata.
Keep the release directory while Compose labels reference its override files.

## Active release persistence

The permanent wrapper
`/home/ubuntu/forte/deploy/releases/accuracy-v2-dcb5f165/compose-active.sh`
includes the base Compose file, the existing fast-live profile, and the exact
immutable image override. Use it for ordinary restart/recreate operations:

```bash
/home/ubuntu/forte/deploy/releases/accuracy-v2-dcb5f165/compose-active.sh restart api worker
/home/ubuntu/forte/deploy/releases/accuracy-v2-dcb5f165/compose-active.sh up -d --no-deps --no-build api worker
```

Docker's existing restart policy preserves the current container image across
a daemon/host restart. Compose `restart` also restarts existing containers
without selecting another image. For **recreation with `up`**, use the wrapper:
the unchanged historical `.env` alone still selects the old CPU image, so a
base-only `up` could revert source while leaving the verified config in the
data volume. The active release wrapper prevents that mismatch. Keep its
override files in place until an explicit future release or rollback replaces
them. The 11:01:25 Asia/Qyzylorda budget stop is preserved and still applies.
