# Demo handoff with the existing stop preserved

The user chose **KEEP THE 11:01 STOP**. The current account's instance `wd88a7vvt` remains scheduled to stop on **18 October 2026 at 11:01:25 Asia/Qyzylorda (+05:00), 06:01:25 UTC**. The proposed 12:30 extension was rejected and was never applied.

The judging deadline is **18 October at 12:00 noon +05:00**. The current public application depends on this GPU host, including its API, worker, database and static UI. **It will become unavailable before the noon deadline unless a separately authorized second-account deployment is ready and the proxy has been switched before 11:01:25.** A working domain or cached page does not keep inference available after the VM stops.

The current account has a USD50 cap. Its expected 20-hour cost is USD40.60 at the displayed USD2.03/hour, before unverified extra charges; an actual billed total has not been verified. Stopped storage continues at USD0.05/hour. The other account's USD50 is reserved and unavailable until its access and the actual new rental configuration are reviewed. Do not pool the two budgets, launch another rental, switch the current CLI account, or move the existing stop timer.

## Backup before any account handoff

Wait for measured trials, dense-feature extraction, training and the review handoff to release their quiet windows. No backup, deployment, model warmup or account change should interfere with an owned measurement. The generic procedure is implemented in `deploy/private_backup.py`.

From the current Windows repository, a check is read-only and prints only the combined number of queued and processing jobs:

```powershell
python deploy/private_backup.py check
```

After the window is released and all jobs are terminal, choose a new directory **outside the repository**, excluded from OneDrive/Dropbox/Git/public sharing. The tool creates it exclusively and protects its ACL for the current Windows user and SYSTEM. It refuses an existing destination and refuses a directory inside the source workspace.

```powershell
python deploy/private_backup.py create --quiesce-idle --destination C:/Users/Wezeso/FortePrivateBackups/demo-handoff-20261017
python deploy/private_backup.py verify --destination C:/Users/Wezeso/FortePrivateBackups/demo-handoff-20261017
```

Once the separate W1 experiment has finished, add `--include-experiments` to
`create`. It additionally requires that no `forte-ml` container is running and
captures `experiments.tar.gz`: private metadata, feature caches, candidate
checkpoints/run logs and the exact code version. When the separate teacher
runtime exists, its verified code/metadata snapshot and W2 outputs are included;
both training and teacher Compose projects must be idle. Public encoder weights and
original training MOVs are excluded. Media and nested source/archive formats
are excluded from this optional archive and rejected by its local verifier.
Preserve the already hash-verified original
dataset and private native-source TAR separately; this optional archive does not
replace them. Verification accepts this fifth artifact and checks it fully.

The first command checks queued/processing jobs, stops API admission, checks again for an admission race, stops the idle worker and checks once more. The database and warm model keep running. Freeze all deployment, configuration, database-admin and experiment writes for the snapshot as well; stopping API/worker does not fence other operators. In a `finally` block it attempts to restart API and worker even if backup fails. This creates brief application downtime; tell the operator before executing. If active jobs remain, defer instead of interrupting them. If restart fails, the tool exits nonzero; preserve partial files and inspect service state locally without printing private diagnostics.

Reserve the exclusive quiet window through every TAR stream and final integrity
check. The late isolated-container check narrows a race but is not a scheduler
lock: no other owner may start training, teacher inference, artifact mutation or
a measured upload until the backup owner releases the window.

The existing local human-review workbench and its private review records remain on the workstation and are not part of this remote-volume snapshot. Preserve them separately under their current private storage; do not publish them or automatically use unreviewed model proposals as training labels during restoration.

It streams binary data straight to the restricted local destination and emits aggregate status only:

| File | Contents and verification |
| --- | --- |
| `database.dump` | PostgreSQL custom-format dump, no owner/grant restoration; archive table of contents validated using PostgreSQL18 `pg_restore --list` without printing it |
| `data.tar.gz` | Entire durable `/var/lib/forte` volume: accepted uploads, immutable results, per-attempt private audits, work/evidence and heartbeat files |
| `runtime-config.tar.gz` | Actual deployment `.env`, runtime secrets, generic configs and pinned model manifest; this archive is sensitive |
| `source.tar.gz` | Explicit code-only allowlist package from the current remote source; excludes footage, labels, secrets and model weights |
| `manifest.json` | Byte counts, SHA256 hashes, generic file counts, app image identity and consistency flags; no content-dependent observations |

Verification checks every archive SHA256/size, reads compressed members fully, rejects absolute/traversal/device/link entries, and confirms the custom PostgreSQL header. A successful archive check proves integrity and dump readability; a restore drill on a new empty PostgreSQL18 database is still required before a live cutover. A failed backup never authorizes deleting anything. Never upload this private directory to a repository, Dokploy's compose editor, cloud screenshot, external AI service or public release archive.

Local synthetic procedure verification passed: protected Windows ACLs, actual PostgreSQL18 custom-dump restore into a fresh database, complete archive hash checks, corruption rejection and traversal rejection. A read-only preflight against the current remote database also passed with zero processing jobs; no services were changed. This is procedure validation, not a completed backup of current private data. Aggregate proof is in `infra/backup-procedure-check-20261017.json`.

An actual current-data backup was subsequently completed and independently verified on **17 October 2026 at 18:18 +05:00**. Its five protected artifacts total **353,429,782 bytes** and include the durable application data, PostgreSQL dump, sensitive runtime configuration, actual remote source and optional experiment metadata/cache/checkpoints. The sanitized receipt is `infra/qwen38-backup-verification-20261017T131821Z.json`; the protected directory is `C:/Users/Wezeso/FortePrivateBackups/qwen38-handoff-20261017T1310Z`. API/worker resumed successfully, active jobs remained zero, and internal/public strict readiness returned 200 with every component true. This snapshot captures the existing baseline deployment before candidate changes; later writes need a new snapshot if they must be part of the final handoff. This specific backup has not been restored on another target.

After three further user uploads, the current **18 accepted / 9 completed / 9 failed / 0 active** state was captured on-host at **18:35:48 +05:00** and copied to protected local storage without holding the GPU handoff. The four current database/data/runtime/source artifacts total **316,236,490 bytes**; on-host hashes, full archive reads and PostgreSQL TOC verification passed, followed by local hash/archive and restricted-ACL verification at **18:43:45 +05:00**. The current snapshot is `/data/forte-backups/qwen38-pretrain-20261017T133529Z-8c69c23d` and `C:/Users/Wezeso/FortePrivateBackups/qwen38-pretrain-20261017T133529Z-8c69c23d`; receipt: `infra/qwen38-onhost-pretrain-backup-20261017T133548Z.json`. API/worker were intentionally left paused for root's authorized training handoff, with the database/model preserved at that handoff; subsequent service control belongs to root. Both earlier and current backups remain preserved. No restore, account/rental change or stop-schedule edit was performed.

Take the final backup after any approved diagnostic CPU refresh. Reconcile its remote source hash/app image with the final A20 release manifest; the backup source describes that remote snapshot and must not be assumed identical to an older preliminary source ZIP.

Retain the current provider disk until local backup verification and the operator's cleanup decision. At the existing stop, a 72-hour retained-disk period through **21 October at 11:01:25 +05:00** adds USD3.60, for an estimated USD44.20 total and USD5.80 contingency before unknown charges. Review costs/retention before that point; indefinite disk retention cannot fit a fixed cap. No deletion is performed by this procedure.

## Second-account prerequisites

Root requests the second account only after the current burst and human-review handoff are ready. Until then this is a checklist, not authorization to deploy.

1. Obtain the user-authorized second account/organization and a live quote, account credit, launch configuration, stop behavior, retained-disk price and cleanup owner. Keep its maximum USD50 separate. Review any provider terms and required action-time confirmations before launch.
2. Use a separate OS/WSL profile or isolated CLI home for the second account. Preserve the current CLI credentials and Windows stop task until its exact instance has stopped. Do not reuse the current account's identity file or edit its managed SSH profile.
3. Choose a new SSH alias and project path. Verify the actual GPU VRAM, driver, Docker and disk before pulling the image. The current tested combination is A10080GB, driver565.57.01, vLLM0.31.0-cu129 digest `sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb`. A different GPU/driver combination needs its own BF16 and readiness checks.
4. Set a separate, explicit second-account stop cutoff and durable guard within its reviewed budget. The current account's 11:01:25 cutoff stays intact. Allow substantial time for a cold image pull, pinned17.5GB model download, validation and warmup; startup time on the new account is unmeasured. Aim to finish verification before the old host stops; do not promise continuity before actual checks pass.
5. Use the final reviewed source release and its recorded checksum from A20. The existing preliminary ZIP may be stale. Keep private backups and code release separate. Do not download footage into an image build context.

## Restore on an empty, authorized target

Use a fresh `/home/ubuntu/forte` deployment directory and empty `forte` Compose project. Verify the local backup before transfer, transfer it through the new authorized SSH connection, verify hashes again on the target, and restrict the remote backup parent to the operator (`0700`). Do not overwrite an existing database or data volume. Keep all application paths inside the containers at `/var/lib/forte` so database input/result paths remain valid.

1. Extract the verified `source.tar.gz` into the new repository. Extract `runtime-config.tar.gz` there privately. Check the secret directory remains `0700`; Compose-mounted secret files must be readable by service UID10001. Never print `.env`, passwords or API keys. Change host-specific snapshot/ingress settings and `FORTE_PUBLIC_API_BASE_URL` to the new trusted team gateway; retain the exact approved frontend CORS origin. Preserve the chosen measured worker profile and provided bank-case context if the same authorized case is being served.
2. Stage public pinned weights afresh using the model manifest. Do not use either account's authentication credentials as a model download token. Validate all15 artifacts and the model/processor revision before enabling the GPU service. Weights are intentionally absent from the private backup.
3. Build the CPU image from the verified source. Start only PostgreSQL (`docker compose ... up -d db`) and wait for its healthcheck. Restore into its fresh, empty `forte` database **before** running the migration/API/worker. Stream `database.dump` into `docker exec -i forte-db-1 pg_restore --exit-on-error --no-owner --no-privileges -U forte -d forte`. Do not use `--clean` against a populated database.
4. Restore `data.tar.gz` into a fresh `forte_forte_data` volume using the built CPU image with `--network none`, `--read-only`, `--cap-drop ALL`, `--security-opt no-new-privileges:true`, `--entrypoint tar` and stdin. Mount it at `/var/lib/forte` so the image initializes that directory with UID10001. Run extraction as the image's UID10001 with `tar --no-same-owner -C /var/lib/forte -xzf -`; verify the service UID can read uploads/results and create its heartbeat. Do not merge with an existing volume or use a root-owned extraction that blocks the worker.
5. Start the private GPU profile, perform an actual BF16 operation, artifact verification, full warm readiness and strict synthetic visual-marker smoke. Check internal network/no model host ports, the fixed-IP no-payload TCP probe and offline variables. Do this outside any reserved video benchmark window.
6. Start API/worker and run queue recovery for expired leases/deadlines. Restore never revives an expired job or changes its original deadline; new demo uploads get new jobs. Verify existing terminal result bytes against their persisted hashes without showing private contents. API/worker must see the same restored data volume.
7. Check `/healthz`, strict `/readyz`, missing-key rejection, an authorized synthetic upload/poll/result cycle and a private operator review of expected content. Record actual versions/image identities and an aggregate restore report. A successful startup alone is insufficient.

## Domain cutover and fallback

Keep the existing isolated Dokploy `Forte Datathon` / `Forte Ingress` service. Only after the new app is verified, replace both its Caddy frontend upstream and public `/runtime-config.json` API base with the new HTTPS team gateway, retaining the direct-API bootstrap in `deploy/dokploy-ingress.direct-api.yaml`. The current browser/API base bypasses an observed large-upload failure on the original same-origin proxy, so do not silently revert uploads to that route. Retain the API-key boundary, exact frontend CORS origin, private model ports and request-header-safe logging, then deploy that single ingress. Verify TLS, bootstrap JSON/no-store, all readiness components, allowed-origin preflight, unauthenticated rejection and the empty public UI on `https://hackathon.wezeso.me`. Do not copy SSH credentials, private backups or keys into Dokploy's compose editor. No DNS change or duplicate project should be needed.

Before 11:01, rollback can point the proxy to the still-running old gateway if needed. After the authorized stop it cannot provide service unless the user separately authorizes a restart. If the second account is not available or fails verification before 11:01, report the deployment unavailable before noon, preserve local submission/artifacts and existing data, and keep the chosen stop. Do not silently extend or restart the first rental.
