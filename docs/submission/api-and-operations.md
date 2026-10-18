# API and operator handoff

Run from the repository root after the locked CPU dependencies and reviewed runtime catalog are installed. `README.md`, `serving/README.md`, and A17's deployment runbook are authoritative for their current startup conditions.

## Requests

The UI origin is `https://hackathon.wezeso.me`; its runtime bootstrap now sends API traffic to `https://forte-review-wd88a7vvt.gobrev.dev`. Use the direct gateway for jury upload/polling scripts as well. CORS allows the exact frontend origin; authenticated script requests without an Origin header also work. The original same-origin proxy's large-upload failure is unresolved. Put the private jury key in `FORTE_API_KEY`; never commit or include it in screenshots/reports.

The source also provides opt-in `FORTE_CONCISE_OBSERVATIONS` and `FORTE_STAGED_SCREENING` profiles, both off by default. Staged screening requests four foundational facts without event proposals, then runs the full fact set on unresolved windows using the same source frames before context refinement. Its CPU tests cover complete-coverage and unresolved-branch guards. The original all-19-fact GPU benchmark does not validate these later profiles; use each run's recorded flags, prompt/config versions and hashes.

The current CPU image is `sha256:64f56ae74e2c5824dbd20cac9a1540738d3467c0b57a0362ddfac0dd70ec39cc`, with compact IDs, concise observations and staged screening unchanged, plus the optional review ledger and reproducibility audit. Strict readiness/auth/CORS, live review-route GET/negative-result rejection and unchanged persisted jury bytes were verified without review mutations or new model calls. The prior `c4528d…` image's sequential reused-input development trial had five completed results and one model-output failure; completion median 26.485 s, maximum 149.597 s, and zero of six attempts below 10 s. This does not replace the original image's benchmark or establish novel-content latency. New encoder/head experiment code remains separate from the active service.

```powershell
curl.exe -H "X-API-Key: $env:FORTE_API_KEY" -F "file=@C:/path/to/authorized.mov" https://hackathon.wezeso.me/api/v1/analyze
curl.exe -H "X-API-Key: $env:FORTE_API_KEY" https://hackathon.wezeso.me/api/v1/jobs/JOB_ID
```

POST returns `{"job_id":"..."}` once the owned file and durable record exist. GET returns `{"job_id":"...","status":"processing"}` until completion, a validated immutable completed object, or `{"job_id":"...","status":"failed"}`. Missing/invalid keys return 401; unknown IDs return 404. The upload limit starts at 256 MiB and is configurable; detailed decoding belongs to the worker.

The schema exposes no third verdict for unclear visibility. Low visibility can block a supported negative; technical failures remain failed. Rich internal review metadata stays outside the jury projection. Repeated completed GETs serve the same stored JSON/JPEG bytes.

## Locality and readiness

API processes are CPU-only. Workers send sanitized JPEGs only to the configured team-controlled private vision-model service. The database, model and control ports stay private. Set downloaded model/processor paths, exact artifact hashes, private network/configuration, and offline flags before the blocked-egress proof. A live health page is insufficient evidence of a warm model.

Do not put the label workbook, audit manifests, annotation ground truth, split truth or evaluation records into the serving image/mounts. The worker uses generic taxonomy/rules and accepted source media, with frame IDs bound to source hashes/PTS. Check the final built image and mounts rather than assuming ignore files prove the deployed state.

`FORTE_TRUSTED_SCENE_CONTEXT=bank_cash_workstation` is an explicit operator assertion about the authorized recording scope. Set it only after the local operator verifies that scope. With no trusted context, bank-specific applicability stays unclear; neither a visual appearance nor this setting establishes an individual action/procedure fact.

The selected deployed worker uses `FORTE_COMPACT_FRAME_IDS=true`, `FORTE_MAX_INFLIGHT_PER_JOB=2`, and `FORTE_MAX_CONCURRENT_CLIPS=2`. Short aliases are restored and checked against owned source IDs before event/evidence assembly. Generation still allows 4096 tokens; no source frame or fact was removed for this tuning. `configs/serving/synthetic-concurrency-validation.json` records all four ordered single-trial conditions (8 valid responses); it establishes a local tuning observation, not five-video capacity or accuracy. Generic code/example defaults remain false/one-per-job.

## Recovery

Accepted uploads and results live on persistent storage. Restart an API without reloading model weights; workers recover expired leases using fencing tokens. A stale attempt cannot replace the active result. Keep the same PostgreSQL/shared media locations across restarts. Verify actual database recovery and GPU-service cancellation on the selected deployment.

Keep the untuned pinned baseline as the fallback. If adapter improvement or serving compatibility is unverified, serve the baseline. A model failure or exhausted deadline is reported as failed; never publish an empty negative to conceal it.

## Public smoke

Run from an external machine after the actual TLS UI/API origin is ready:

```powershell
python qa/integration/public_smoke.py --base-url https://hackathon.wezeso.me --report qa/integration/public-smoke.json
```

The script creates synthetic MOV footage, verifies key rejection/unknown IDs, uploads, polls, validates completed JPEG/schema content and immutable bytes, and writes only aggregate diagnostics. Its completion timing describes synthetic media only. Separate real uncached required-video and five-distinct-video benchmarks remain necessary.
