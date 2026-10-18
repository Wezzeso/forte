# Accuracy upgrade: implementation and release runbook

Updated 18 October 2026. This runbook implements the actionable parts of the requested accuracy plan and corrects its evaluation and dependency assumptions. Recording new footage, human verification, GPU training, model acceptance and deployment are separate completion gates; adding code does not complete those gates.

## Evidence and acceptance

The existing `benchmarks/quality/fast-analyzer-eval.json` records two historical runs with 34/40 correct video verdicts, 15 true positives, 6 misses, 19 true negatives and no observed false positives. It also records 10 transformed-video checks. This is a stored summary, not a fresh reproduction, an accuracy improvement or proof of performance on new footage. The six misses need both object recognition and temporal coverage; the unidentified case has no justified new event type yet.

Targets are 40/40 on held-out development predictions and at least 95% on a separately collected test set. These are goals, not guarantees. Record sample counts, per-type precision/recall, observable/unclear cases and confidence intervals with every score. Zero false positives in 19 observed clean clips does not establish a zero population false-positive rate.

Release a type only when all of these pass:

1. Its observable facts and public norm are verified. Partial cash movement alone does not establish theft; an occluded card, key or device is `unclear` rather than absent.
2. A versioned specialist checkpoint has held-out class precision/recall and original-resolution evidence. Enlarging a 4×6-pixel object adds no missing information.
3. The complete video pipeline improves held-out results and adds no false alarms to the clean regression clips. Detector recall alone is insufficient to enable an event.
4. The type, timecode and 1–3 original evidence frames agree. Rejected candidates remain available in the private audit.
5. Public API checks pass with both five parallel uploads and the six-long-video stress case. Upload and queue time count toward each job's deadline.

Two checks can reduce errors but are not statistically independent merely because they use different model names. A temporal head consuming detector outputs inherits detector errors, and a VLM viewing the same crop shares visibility limits. Measure agreement, correlated misses and the recall cost of the confirmation gate on held-out clips.

## Data and evaluation rules

- Collect correct and incorrect versions of each scenario from several real camera positions. Preserve high-resolution originals, timestamped source frames and downscaled inference copies. Label ID cards separately from banknote stacks and notes; record hand/device contact and full action intervals explicitly.
- Keep source recording, content hash, actor/setup scenario group and every derivative in one partition. Adjacent frames, crops, mirrors and re-encodes are not independent samples. Record consent, footage provenance and human review; model-generated proposals remain unverified until corrected by a person. Existing annotation controls are described in [annotation-guide.md](annotation-guide.md).
- Train new specialists on authorized own recordings or permitted synthetic material. Keep supplied organizer clips out of weight training. Never train on jury test media or silently relabel organizer content as an own recording.
- Choose one honest development protocol. **Fixed evaluation:** freeze prompts, rules and thresholds using own train/validation recordings, then score the supplied clips once. **Grouped cross-validation:** if supplied labels influence tuning, call them development data; tune only inside each outer training fold, using an inner grouped validation split, and score its untouched outer fold. Save out-of-fold predictions, mean and worst fold. These results are not an untouched final test.
- The plan's “leave-one-out” and five-fold grouped CV are different protocols. Use grouped CV when related clips exist; do not label five-fold results leave-one-out. Reusing results to choose later changes also makes the suite a development regression set.
- Mirror, crop, zoom, rotation and perspective warps measure robustness to image transforms. They do not reproduce a new camera's occlusion or reveal hidden surfaces. Reserve one genuinely distinct camera position as a locked final check; do not train or choose thresholds using it.
- Freeze a per-type threshold using development data and an explicit false-positive budget. Report the precision/recall tradeoff instead of relaxing thresholds on the evaluation set until recall improves. Tiny probe sets such as 8/8 are useful regressions, not a substitute for representative validation.

## Dependency decisions

Record package version, exact checkpoint revision/hash, separate code and weight licenses, training-data terms and required notices in the release manifest. Repository licenses do not automatically settle every checkpoint. Sources below were checked on 18 October 2026.

| Component | Verified scope and action |
| --- | --- |
| RF-DETR | The package and Apache-designated models are Apache-2.0. Current upstream lists Nano, Small, Medium and Large detection models under Apache; Plus Atto/Femto/Pico/XL/2XL detection models use PML 1.0. Start with the pinned `RFDETRSmall` recipe. Segmentation and keypoint availability is version-specific; keypoints are currently preview, not a validated hand-contact model. [Upstream model/license table](https://github.com/roboflow/rf-detr), [pinned 1.11.2 license](https://github.com/roboflow/rf-detr/blob/1.11.2/LICENSE), [training API](https://rfdetr.roboflow.com/learn/train/). |
| SAM 3 | Code and weights use Meta's custom SAM License, which grants broad use and modification rights subject to its conditions and use restrictions. It is not Apache-2.0; commercial permission and compliance with a strict open-source-only rule are separate questions. Leave it optional until the exact license and competition requirement are accepted. It is a labelling aid, never an evidence source or automatic reviewer. [Official license](https://github.com/facebookresearch/sam3/blob/main/LICENSE), [official project](https://github.com/facebookresearch/sam3). |
| Qwen | `Qwen3-VL-30B-A3B-Instruct` and `Qwen3.6-27B` are published under Apache-2.0. Both support video: 3.6 does not introduce that capability for the first time. Treat 3.6 as an evaluated candidate; use its own processor, template and serving compatibility rather than substituting its model ID into another recipe. [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct), [Qwen3.6 model card and video example](https://huggingface.co/Qwen/Qwen3.6-27B). |
| Hands / MMPose | MMPose code is Apache-2.0 and supports hand pose estimation. Check the exact RTMPose checkpoint and source-dataset terms separately before adopting weights; do not infer approval from the code license. Body/wrist keypoints alone do not establish fingers touching a banknote or detector. [Official MMPose repository](https://github.com/open-mmlab/mmpose). |
| Serving | ONNX Runtime is MIT, not Apache-2.0. TensorRT's OSS repository covers only part of the full SDK; the runtime has NVIDIA SDK terms. Use the approved ONNX/PyTorch path unless the full SDK's terms are accepted. [ONNX Runtime license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE), [TensorRT OSS scope](https://github.com/NVIDIA/TensorRT), [SDK agreement and TensorRT supplement](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html). |
| CoTracker3 / VideoMAE | The official Meta CoTracker3 checkpoint and original MCG-NJU VideoMAE base checkpoint declare CC-BY-NC-4.0. Exclude these artifacts from a commercial-use stack. This finding does not license or disqualify every independent implementation sharing the architecture name. [Meta checkpoint card](https://huggingface.co/facebook/cotracker3/raw/main/README.md), [original VideoMAE checkpoint card](https://huggingface.co/MCG-NJU/videomae-base/raw/main/README.md). |
| Own temporal code | PyTorch's license does not automatically license Forte's original code. No root project `LICENSE` was identified during this audit; explicitly declare the project license before describing the entire submission as open source. Retain dependency notices. Learned event weights require their own provenance and acceptance run. |

Do not use generated super-resolution pixels as evidence. Preserve original source frames and report visibility limits. Pretrained tracking/temporal weights require an artifact-specific license check; a research repository or architecture name is insufficient.

## Implemented scope

The optional `forte.fast.specialists` runtime accepts local, hash-bound RF-DETR checkpoints and tracks detections with ByteTrack. The service sends each native decoded frame to that runtime, keeps bounded original frame samples for VLM verification/evidence, and records perception provenance. An optional local RTMPose backend separately checks its pose/detector configurations, checkpoints and gate. Its hand IDs are frame-local; persistent hand identity and actor roles are not established. A two-dimensional hand/object overlap is recorded as proximity; it is not proof of a physical authenticity check. Actual CUDA/checkpoint compatibility still needs a host run.

`forte.fast.temporal_events` provides deterministic candidates and abstention reasons. Calibrated persistent cups/coats in an explicit desk zone can produce a candidate. ID omissions, key removal, authenticity checks, recount omissions and cash handover sequences additionally need source-bound actor/action facts and complete relevant visibility. The generic detector does not produce those facts merely by missing an object. No trained transformer event head is supplied by this module. The existing weak temporal training workflow has a different contract; it is not a new class-specific action/visibility model.

Configure a staging worker through `FORTE_DATA_DIR/fast_config.json`. Its `specialists` object contains `enabled`, local `checkpoint`, local `gate_path`, `model_class`, explicit zero-based `class_names` and `device`. The gate binds model version/resolution, checkpoint, dataset and metric hashes, calibrated thresholds and an explicit type allowlist. Optional `hand_backend` uses `backend: "rtmpose"`, `pose_config`, `pose_checkpoint`, `detector_config`, `detector_checkpoint` and its own `gate_path`. Its separate hand gate hashes all four local artifacts, fixes `keypoint_threshold` and approves only `proximity_2d`. Use the training-generated class map exactly. Missing/invalid configured artifacts fail instead of becoming empty detections; absent optional models stay disabled. Do not manufacture type approvals or visibility metadata to bypass acceptance.

The new CPU-only scorer, split planner and HTTP load tool can validate their contracts with synthetic fixtures. They provide no new recognition score until real inference and reviewed labels are supplied.

## Execution order

1. Identify the configured NVIDIA Brev deployment, capture a read-only GPU/runtime inventory and preserve its current working image/config. The account name alone is not a deployable host address or verified GPU allocation. Follow [gpu-provisioning.md](gpu-provisioning.md) and [second-account-demo-handoff.md](second-account-demo-handoff.md) with the actual configured target.
2. Run a fresh baseline with code/config/model/data hashes before changing accepted behavior. Keep predicted and reference labels separate; preserve failed and timed-out clips in the denominator.
3. Prepare reviewed own-recording data, train the smallest detector, evaluate it on a distinct camera holdout and create a class-level gate. Class acceptance must not itself enable violation types.
4. Enable a specialist only in a staging configuration. Compare candidate and baseline video predictions under the protocol above, inspect misses and false alarms locally, then create a type-level release gate.
5. Complete five-upload and six-upload end-to-end checks, including queueing, result publication, evidence retrieval and deadline handling. Pin accepted artifacts before updating the deployed configuration.
6. Train LoRA and a temporal head only after reviewed crops/events exist. Use the same frozen splits. Replace rules one type at a time when complete-video results support it.

The proposed two-hour detector, three-hour LoRA, twenty-minute temporal training, three-millisecond frames, tenfold VLM reduction and twofold FP8 throughput are unmeasured estimates for this workload. Benchmark warm/cold latency, p50/p95/max, memory, decode cost and call counts on the actual GPU. A four-job worker cap limits simultaneous analysis; it cannot guarantee success for an unlimited queue. A100/H100 precision choices and batch size must be validated with the selected model and runtime.

Use a bounded decode/frame cache and preserve original presentation timestamps. Keeping every decoded frame of concurrent long videos in memory is not a safe default. A detector fed only the existing sampled frames does not implement every-frame or 30-fps hand-contact analysis; measure short-event coverage at the actual sampling rate before claiming that capability.

## Specialist training commands

The following Linux paths are examples for a private local GPU host. Replace placeholders with reviewed data; example manifests are not training data. Keep media, labels and generated crops inside the team's approved environment.

```bash
python -m training.specialists.prepare \
  --manifest /private/manifest.json --output /private/dataset
python -m training.specialists.train \
  --config /private/train.json --preflight
python -m training.specialists.train --config /private/train.json
python -m training.specialists.evaluate --config /private/eval.json
python -m training.specialists.gate \
  --run /private/run/run.json --metrics /private/eval/metrics.json \
  --output /private/gate.json
```

Start from `configs/training/specialists-manifest.example.json`, `specialists.example.json` and `specialists-evaluate.example.json`. The preparation path checks consent/provenance, organizer source hashes, recording/scenario separation and distinct holdout cameras. A generated object gate keeps `approved_types` empty until a full-video release gate is satisfied. Training success, passing object metrics and deployment acceptance are distinct artifacts.

## Video regression and API commands

Create reviewed source/scenario IDs before planning grouped folds:

```bash
python tools/fast_validation.py --manifest /private/scenarios.json \
  --folds 5 --seed 0 --out /private/splits.json
```

The manifest contains `provenance` with `source` and `grouping_method`, and `clips` with `name`, `group_id`, `source_id` and `role` (`development` or `holdout`). Derivatives additionally identify `parent_name` and `augmentation`; preserve `content_sha256` where available. This command writes a split plan. It does not fit models, tune thresholds or produce cross-validation scores. Every fold needs its own independently frozen model/configuration and held-out predictions.

Run fresh analysis against the configured **local model-serving endpoint**. This is separate from the public Forte API:

```bash
python tools/fast_eval.py --videos /private/videos \
  --labels /private/labels.json --base-url "$FORTE_FAST_MODEL_BASE_URL" \
  --model "$FORTE_FAST_MODEL_NAME" --config /private/fast_config.json \
  --par 4 --out /private/candidate.json
python tools/fast_eval.py --labels /private/labels.json \
  --report /private/candidate.json --baseline /private/baseline.json \
  --gate --expect-count 40 --expect-clean 19 --min-accuracy 0.85 \
  --require-type-time --out /private/scored.json
```

The count/accuracy flags above describe the current development regression floor; passing them alone is not a 40/40 or improvement claim. `--require-type-time` needs reviewed type/time labels. Omit that flag for a binary-only diagnostic and mark type/time acceptance pending. Failures and missing videos stay in the accuracy denominator. Offline `--report` re-scores that report only; it does not execute changed code or verify a new configuration. The aggregate historical summary cannot be replayed as complete per-video evidence.

Run the public API latency/schema checks with `FORTE_API_KEY` already supplied through the host's approved secret mechanism:

```bash
python tools/fast_load_test.py --base-url "$FORTE_PUBLIC_ORIGIN" \
  --video /private/owned-input.mov --parallel 5 \
  --deadline-seconds 240 --report /private/load-five.json
python tools/fast_load_test.py --base-url "$FORTE_PUBLIC_ORIGIN" \
  --video /private/owned-input.mov --parallel 6 \
  --deadline-seconds 240 --report /private/load-six.json
```

Repeat `--video` for each real input in the six-long-video stress set. A single input is repeated to fill the selected concurrency; it is not a six-distinct-video result. The deadline begins before client queueing and upload. Reports omit result images and text, validate the complete API payload and record terminal failures. These tests measure latency/schema, not recognition accuracy.

## Completion record

The fresh unchanged-service Brev baseline on 18 October completed all 40 videos with **33 correct (82.5%), 1 false alarm, 6 misses and 0 processing errors**. It used four parallel videos and the existing model process; the earlier rv6 used ten. This measures a failure to reproduce the historical zero-false-alarm result, without establishing its cause. See [the exact aggregate and runtime binding](../benchmarks/quality/accuracy-upgrade-baseline-20261018.json). No specialist weights were trained or selected in this run.

The optional crop LoRA and tracked-object Transformer recipes are separate executable paths:

```bash
python -m training.specialist_lora --config /private/crop-lora.json --preflight
python -m training.specialist_lora --config /private/crop-lora.json
python -m training.specialist_temporal --config /private/tracked-temporal.json --preflight
python -m training.specialist_temporal --config /private/tracked-temporal.json
python -m training.specialists.track --run /private/run/run.json --metrics /private/eval/metrics.json --tracking-dir /private/mlflow
```

See [crop LoRA](../training/specialist-lora.md) and [tracked features and event training](../training/specialist-temporal.md) for their reviewed-data contracts. The MLflow command exports detector-run metadata and verified scalar metrics to a local SQLite store. The new tracked head is not selected in the fast service: it still needs a validated feature/visibility producer, actual training and a complete-video gate.

The optional [GPU worker image](../deploy/Dockerfile.specialists) and [Compose overlay](../deploy/compose.specialists.yaml) provide RF-DETR dependencies, GPU access and read-only artifact/config mounts. The ordinary API/worker image remains CPU-only. Build and measure the optional image before enabling it; no successful build or GPU operation is claimed here. RTMPose's MMCV dependencies need a separately compatible environment.

This document does not assert that any GPU training, new-recording collection, accuracy improvement, load result or deployment has completed. Fill the release record from artifacts, not target numbers:

| Gate | Required evidence |
| --- | --- |
| Fresh baseline | Exact image/code/model/data hashes and all per-video results |
| New verified data | Manifest, review provenance, source hashes and locked splits |
| Specialist training | Successful run manifest and checkpoint hash |
| Object holdout | Per-class counts, precision/recall and camera split report |
| Complete-video acceptance | Out-of-fold or fixed-holdout predictions, no added clean-clip false alarms, type/timecode/evidence checks |
| Operational acceptance | Five- and six-upload public API results including queue/deadline outcomes |
| Release | Type allowlist, license/notice manifest, pinned artifacts, deployment health and rollback reference |

If a gate cannot run because its recordings, labels or GPU are missing, record it as pending. Retain the accepted baseline and the disabled status of unverified types.
