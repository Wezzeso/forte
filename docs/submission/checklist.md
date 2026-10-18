# Submission checklist

Accepted live release (18 October): the VLM-confirmation profile is deployed on the current Brev host. Its direct analyzer run passed the binary development gate at **34/40, 0 false alarms, 6 misses and 0 processing errors**, with no previously-correct video verdict regressed. This was not a new 40-video public API run. The fresh unchanged-service baseline was **33/40 with 1 false alarm**; historical rv5/rv6 also scored 34/40. Median nearest-event timing worsened from **1.0 to 1.5 s**, with 11 rather than 12 detections within 2 s. Event-type labels and unseen-video accuracy are unverified. [Comparison](../../benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json), [implementation record](../integration/accuracy-upgrade-implementation-20261018.md).

The isolated full-service stage passed **6/6** longest-clip uploads in at most **137.990 s**, through internal Docker DNS with public ingress excluded. The accepted rollout's public HTTPS check passed **6/6** on the first attempt, maximum **161.068 s**, with evidence/JPEG/schema/timestamp checks and no more than four active leases. Readiness passed and the final queue was empty. [Stage evidence](../integration/accuracy-upgrade-stage-20261018.md), [accepted rollout](../../benchmarks/quality/accuracy-upgrade-rollout-20261018.json).

Final current check: **6/6 first attempts**, zero failures/retries, maximum **142.568 s**, median **92.076 s**. This warm GPU-host→public HTTPS check validates JPEG/schema/timeline handling; the 240-second gate passed and 30-second target failed. [Fresh load](../../benchmarks/quality/final-public-six-load-20261018T005948Z.json). [Post-load bindings](../../infra/public-training-final-postload-bindings-20261018.json) retain worker `8c8b9255…`, frontend-only API `088c4875…`, accepted analyzer bytes and unchanged protected model/database/ingress; readiness true, queue 0/0.

Both public-data fits and single frozen sealed attempts completed. RF ID recall **13/16 (81.25%)** and compact primary macro **94.53125%** failed 95% targets; six RF FP were observed. Their [versioned host release is verified and disabled](../../infra/public-trained-candidates-release-verification-20261018.json), without live configuration changes. Public-object evidence does not qualify bank violation types. [Methods and full benchmarks](../integration/accuracy-methods-benchmarks-20261018.md).

The accepted API/worker image is `sha256:8c8b92550c5dfc6702efbe266773ea4bfd86639b9f5edd8e86e6a0e3e45b6a3d`, over baseline `forte-review:fast-v4` (`sha256:30691bd4230396c011f292beab7094bcdf160e7845c417e85bf8d11e8dcbdf32`). Its source archive SHA256 is `dcb5f16574df668aaaf6f6011cb0f13f70f0056a0c20027faa8777584dc21ce3`, and effective profile SHA256 is `113323095c2031430f0aff58f3aecc4fe28907dfbe84dfb0eea1d9545a7a7b29`. Both running containers' imported sources/config were checked before and after public acceptance. The model, PostgreSQL, ingress, GPU flags, credentials, CORS and budget guard were preserved. Older `64f56a…` and `c4528d…` records describe previous profiles. No newly qualified weights are selected; specialists and six protected types remain disabled. Own-recording-dependent work and second-account migration remain deferred/missing. Only the current Brev account is configured; its **18 October 11:01:25 Asia/Qyzylorda** stop is preserved.

| Deliverable | Handoff location / state |
|---|---|
| Public UI URL and API origin | UI `https://hackathon.wezeso.me`; direct jury/browser API `https://forte-review-wd88a7vvt.gobrev.dev`; strict CORS/auth/readiness verified |
| Jury API key | Private operator handoff only; never in this bundle |
| Upload/polling examples | `api-and-operations.md`, exact contracts/fixtures |
| Startup/use/recovery | Root README, serving guide and A17 deployment runbook |
| Architecture and local model | Root README, execution plan, model/serving manifests |
| Model/processor revision and hashes | `models/manifest.json` and its 15 verified artifacts describe the historical 8B reference. Current 30B process/image/source bindings are in the baseline and accepted rollout reports; weight bytes were not hashed in these runs |
| Experimental model | Qwen baseline remains active; W1 trains a separate temporal head on frozen encoder features. Native cache: 34 sources / 35,186 frames. W1 completed 336 optimizer steps, but 2/6 supplied-label matches versus untrained 3/6; candidate not selected |
| Commercial-use component records | `licenses.md`, 39 CPU and 269 actual GPU distribution metadata entries; native OS/CUDA/image notices still need final inventory |
| Supported visual events | `src/forte/fast/catalog.py` and `disabled_types`; qualified profile retains all six disabled types and no new trained specialist. Whole-video development gate passed; per-type ground truth/visibility coverage incomplete |
| Applicable rule/source register | `docs/legal-source-register.md`, `rules/catalog.json` and archived public sources |
| Dataset/group splits and duplicates | Private records excluded; authorized machine-provisional 28/6/6 assignments and role-label/anchor pins frozen. Human-reviewed split remains a separate track |
| Quality, new-angle and evidence review | Qualified candidate: TP15/FP0/FN6/TN19, no processing errors or prior correct binary verdict regressions; baseline TP15/FP1/FN6/TN18. Nearest-event timing 1.5 s versus baseline 1.0 s. Historical transformed-image 10/10 is not a new physical camera angle; event-type/unseen accuracy unverified |
| Latency/concurrency/failure | Fresh current HTTPS: 6/6, zero failures/retries, maximum 142.568 s / median 92.076 s; warm GPU-host path, JPEG/schema/timeline valid. Historical accepted public 161.068 s and internal 137.990 s remain separate. No under-30-second, desktop/cold or novel-video p95 claim |
| Local/offline deployment proof | Actual artifacts, warm readiness, isolated synthetic-marker inference and one public real-video completed cycle recorded |
| GPU/budget/keep-alive/cleanup | Current account's 18 October 11:01:25 Asia/Qyzylorda stop preserved; second-account migration deferred. A17 private operator handoff governs retention/cleanup; no uptime extension claimed |
| Forte identity/assets | Brand source ledger, token provenance, local Open Sans/OFL; plain text label |
| Demonstration | Distinguish supplied-label machine outputs, optional human review and explicitly synthetic fixtures. Preserve failed-attempt denominators and current profile binding |

The current [source/operator bundle](../../dist/forte-release-source-20261018-final-trained.zip) has a [manifest](../../dist/forte-release-source-20261018-final-trained.manifest.json) and [verification](../../dist/forte-release-source-20261018-final-trained.verification.json). Preserve older bundles as historical snapshots. Reproduce with:

```powershell
python qa/integration/build_release_bundle.py --output dist/forte-release-source-20261018-final-trained.zip
```

The archive uses explicit code/public-document roots and excludes organizer media, spreadsheets, record JSONL, private directories, credentials, runtime state, weights, resource/spending/extension ledgers, provider quote records, benchmark selection/review files and all `infra/quotes/` provider/account screenshots. Supply deployment resource/cost details privately to the operator. Generic annotation review instructions are included; local review records are excluded. The archive includes reproducible source/configuration and local fonts; it is not a preloaded model image or a successful submission. Rebuild after final changes and use its neighboring SHA-256 manifest, which records exact configuration/lock/model/pipeline source digests.
