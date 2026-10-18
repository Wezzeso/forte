# Five-minute demo script

Label actual machine-generated results, optional human-reviewed results and synthetic fixtures separately. Manual review is not a prerequisite for the authorized automated demonstration or W1 experiment. Do not imply that machine proposals or fixtures establish event correctness. Keep source media inside the authorized service, and do not put private evidence on public slides or cloud screen tools.

For the 18 October demo, Forte runs the accepted fast analyzer/VLM-confirmation profile with local Qwen3-VL-30B-A3B on Brev. Its direct analyzer run passed the binary development gate at **34/40, 0 false alarms, 6 misses and 0 processing errors**, compared with **33/40 and 1 false alarm** for the fresh unchanged-service baseline. This was not a new 40-video public API run. It restores historical rv5/rv6 accuracy. Median nearest-event timing worsened from **1.0 to 1.5 s**; 11 rather than 12 detections were within 2 s. Labels lack event types, so no per-type or unseen-video accuracy is established.

The isolated full-service stage passed **6/6** longest-clip uploads, maximum **137.990 s**, through internal Docker DNS. The [accepted public HTTPS release](../../benchmarks/quality/accuracy-upgrade-rollout-20261018.json) passed **6/6**, maximum **161.068 s**. The [fresh current six-upload check](../../benchmarks/quality/final-public-six-load-20261018T005948Z.json) passed **6/6 first attempts**, zero failures/retries, maximum **142.568 s**, median **92.076 s**, with JPEG/schema/timeline validation. This warm GPU-host→public HTTPS result is not desktop/cold latency, new accuracy or proof of a trained-model speedup; the 30-second target is unmet. Own-recording-dependent work and second-account migration remain deferred/missing; the **11:01:25 Asia/Qyzylorda** stop is unchanged.

Licensed public RF-DETR and compact crop-LoRA fits and their single frozen sealed attempts completed. Quality targets failed: RF ID recall **81.25%** with six banknote false positives; compact primary macro **94.53125%**. Their [host release is staged with activation disabled](../../infra/public-trained-candidates-release-verification-20261018.json); live specialists/six protected types remain disabled. Show these limits alongside the [methods and benchmark tables](../integration/accuracy-methods-benchmarks-20261018.md).

| Time | Script / action |
|---|---|
| 0:00–0:30 | Explain the automated visual first pass: upload a recording, receive possible signs with source timestamps/evidence/rule references, and optionally record a separate operational review. |
| 0:30–1:30 | Upload a permitted clip through the UI in an available model window. Show the actual stage and terminal outcome. Identify automated results as machine outputs with their limitations; label synthetic demonstrations explicitly and preserve any failed outcome. |
| 1:30–2:15 | Select separate events. Show original JPEGs, timestamp, observable explanation, applicable source and visibility limitations. |
| 2:15–3:00 | Describe the accepted fast implementation: durable queue, local model, scene crops, temporal rules, VLM confirmation, original JPEG evidence and exact jury API. Optional detector/hand adapters and all six protected types remain disabled. |
| 3:00–3:45 | Show the direct 33/40 baseline and accepted profile's 34/40 development run, including six remaining misses and worse timing. Distinguish the internal stage (6/6, 137.990 s), accepted public check (6/6, 161.068 s), earlier public measurements and historical rv5/rv6/W1 results. Do not present operational checks as new accuracy results. |
| 3:45–4:30 | Demonstrate recovery/failure separation: a technical failure cannot become a clean negative. Explain local visual-only processing and offline deployment conditions. |
| 4:30–5:00 | State remaining limitations, enabled event types and unresolved ground truth, public URLs and privately supplied key. Identify the baseline fallback and judging availability owner. |

Current demo blockers and release state are recorded in `docs/integration/acceptance.md` and `release-manifest.json`. A prepared script is not a completed demonstration or submission.

The release [source/operator bundle](../../dist/forte-release-source-20261018-final-trained.zip) has a [manifest](../../dist/forte-release-source-20261018-final-trained.manifest.json) and [verification](../../dist/forte-release-source-20261018-final-trained.verification.json). Prior ZIPs remain historical; private footage/model weights are separate from this source bundle.
