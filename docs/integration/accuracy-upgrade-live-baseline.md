# Accuracy upgrade: live measurements, 18 October 2026

The unchanged live service scored **33/40 (82.5%)**, with **one false alarm**, in a fresh run at 02:40–02:46 UTC+05:00. This does not reproduce the retained rv6 result of 34/40 with zero false alarms. Treat the older 85% result as a recorded development run, not a guaranteed current accuracy level.

| Development run | Correct | TP | FN | FP | TN | Errors | Parallel clips | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained rv6 | 34/40 | 15 | 6 | 0 | 19 | 0 | 10 | 383.6 |
| Fresh unchanged service | 33/40 | 15 | 6 | 1 | 18 | 0 | 4 | 379.908 |

The new run's median analysis time was 31.539 seconds and maximum was 68.421 seconds. Its nearest-event median timecode error was 1 second, with 12 positive detections within 2 seconds. The original labels do not identify event types, so these timecode figures do not establish type correctness.

## Identity and scoring

- Public aggregate: [`accuracy-upgrade-baseline-20261018.json`](../../benchmarks/quality/accuracy-upgrade-baseline-20261018.json).
- Private gate input: `benchmarks/private/accuracy-upgrade-baseline-20261018.json`, SHA256 `df3f1f7e7eac2bb85f7e9502eeade91f67332307fc941a385b23d59a30c45ea3`. It contains clip names, scoring labels, event types/times and timing/error metadata. It contains no frames, scene states, observations, or verification bodies.
- Dataset: the 40 supplied development clips, 21 positive and 19 clean, 263,715,381 bytes. Labels SHA256: `3ade1ced5efe9d17549a15dd998298518fe420cafae47efcca5eef4152fff0a6`; canonical video-hash manifest SHA256: `9c361202645b419967d3d64cd8988f877e177d2649251e4430f154cd7a37934f`.
- Deployed evaluator image: `forte-review:fast-v4`, image ID `sha256:30691bd4230396c011f292beab7094bcdf160e7845c417e85bf8d11e8dcbdf32`. The installed fast module hashes exactly match the retained rv6 source. Their combined manifest SHA256 is `256c51ab3b063b30446985878bdeeaf0f71b63d7674b5e88faa063bd9ad53a0f`.
- Existing model: `q30`, A100 80GB, bf16, vLLM model image `sha256:046137dc151b7741b52fbb438de7d606fa74694969c7bb4ab6642d9408b9516e`. The server used `max-num-seqs=48`; the client allowed 64 VLM requests and at most four videos. No model-weight digest is available from either recorded run.
- Configuration: the original defaults with `{}` overrides; the six unreliable types remain disabled and `use_verify=false`. No new trained detector, hand tracker, temporal model, or VLM adapter was enabled.

Labels were used only after inference for scoring. A failed video, including a VLM exception swallowed inside the analyzer, is incorrect and remains in the accuracy denominator; it cannot become a true negative. All 40 completed without such errors. The first aggregate-only attempt was cancelled after 10 clips so the full run could retain the requested private gate metadata; its partial results are not scored.

This is an in-sample development measurement, with no held-out or cross-validation claim. The difference from rv6 also reflects a different batch schedule (four versus ten videos); identical source and seeds do not make live VLM outputs invariant to execution conditions.

## Public API load and candidate comparison

The public-load artifact is [`accuracy-upgrade-public-load-20261018.json`](../../benchmarks/quality/accuracy-upgrade-public-load-20261018.json). It measures the five and six longest supplied clips through the public HTTPS gateway from the GPU host, including upload, queue, polling and response-schema validation, with a 240-second budget. It cannot establish desktop upload latency or detection accuracy.

| Parallel uploads | Completed and schema-valid | Maximum upload-to-result seconds | Under 240 seconds |
| ---: | ---: | ---: | --- |
| 5 | 5/5 | 141.523851 | Yes |
| 6 | 6/6 | 167.542674 | Yes |

The proposed under-30-second target under this load has not been achieved.

An initial client attempt used the intentionally internal `forte_private` Docker network and failed transport before any upload was accepted. The client was moved to ordinary Docker bridge egress; no server network or service setting was changed. The load report retains this setup failure separately.

All evaluations used disposable CPU evaluator containers with existing images and read-only source/video mounts. During these measurements, the live API, worker, model, GPU account and service configuration remained unchanged.

## Candidate source comparison

The source-only v2 candidate archive contains 13 Python files and has SHA256 `dcb5f16574df668aaaf6f6011cb0f13f70f0056a0c20027faa8777584dc21ce3`. The evaluator explicitly loads the candidate fast package ahead of the installed package and verifies its source path and file hashes. Detector specialists remain disabled; there are no qualifying trained weights or verified recording labels to enable them.

With default `{}` overrides, the candidate also scored **33/40**, with TP15, FN6, FP1 and TN18; all 40 videos processed successfully. It added no wrong verdict on a video that the fresh baseline got right. Its gate still **failed** for `false_alarms` and `accuracy_below_minimum` (required 85%). Total runtime was 379.453 seconds. This is a regression measurement, not an accuracy improvement.

Generic verification with `require_corroboration=true` and `use_verify=true` then scored **34/40 (85%)**, with TP15, FN6, FP0, TN19 and no processing errors. It passed the binary verdict gate without regressing any previously correct video. Six misses remain unresolved. Its nearest-event timecode median worsened from 1 to 1.5 seconds, with 11 detections within 2 seconds instead of 12. The binary gate does not gate this timing tradeoff or type accuracy. Total runtime was 397.943 seconds; maximum analysis time excluding the evaluator queue was 97.71 seconds.

| Candidate profile | Correct | False alarms | Processing errors | Binary gate |
| --- | ---: | ---: | ---: | --- |
| Default | 33/40 | 1 | 0 | Failed |
| Generic verification | 34/40 | 0 | 0 | Passed |

The aggregate, source-bound report is [`accuracy-upgrade-candidate-comparison-20261018.json`](../../benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json); it references sanitized private per-video gate inputs by SHA256 and verifies the exact video hashes, label file and baseline identity. The complete tested profile is [`accuracy-upgrade-verified-config-20261018.json`](../../benchmarks/quality/accuracy-upgrade-verified-config-20261018.json), file SHA256 `993d2f22867576be053d7a05663043077937d5cd749d67709dcab0d52016789c`, canonical effective-config SHA256 `113323095c2031430f0aff58f3aecc4fe28907dfbe84dfb0eea1d9545a7a7b29`. Its sanitized private report SHA256 is `13c72070192920ba792ceefb60d52fcf9ba7ffc91a7b1a70decb67942177920f`.

This verification profile was explored after seeing development failures, so its result is not held out. There is **no 95% unseen-video accuracy claim**, and none of these measurements establish 40/40. Candidate runs did not deploy code or configuration. A subsequent read-only check of the actual live worker volume confirmed `/var/lib/forte/fast_config.json` was absent and its effective configuration still matched the original baseline with `use_verify=false`. The public-load figures above describe that unchanged service; they do not measure a deployed verification profile.

## Synthetic tracked-head CPU smoke

The inactive tracked-feature Transformer passed an actual CPU forward/backward/optimizer smoke in the existing model image with PyTorch `2.13.0+cu129`: output shape `[1,4,1]`, finite outputs and gradients, nonzero parameter gradients, and a parameter update. CUDA was unavailable and the container had no network. Source SHA256: `d15ef08d8d9f3eaa4665ef657338629ac760aaa29debf8023ab5e516f71b790a`.

The result is [`accuracy-upgrade-tracked-head-cpu-smoke-20261018.json`](../../benchmarks/quality/accuracy-upgrade-tracked-head-cpu-smoke-20261018.json). This used random synthetic tensors and a randomly initialized tiny head; no real-data model was trained, no checkpoint was saved, and it establishes no detection accuracy.
