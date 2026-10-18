# Accuracy upgrade: methods and measured benchmarks — 18 October 2026

The deployed analyzer scores **34/40 development videos, zero false alarms and six misses**. The fresh current six-upload public load check passed at **142.568 s maximum / 92.076 s median**; the under-30-second target is unmet. Both public fits and frozen sealed attempts completed, but quality targets failed: **RF ID recall 13/16 (81.25%) plus six banknote false positives; compact crop primary macro accuracy 94.53125%, below 95%**. Neither public-object experiment establishes improved bank-video accuracy or enables additional violations.

The [final machine benchmark](../../benchmarks/quality/accuracy-final-benchmark-20261018.json)
contains exact metrics and source-receipt hashes; 19 aggregate-validation checks
passed. [Final source/stager checks](../../benchmarks/quality/final-source-delivery-checks-20261018.json)
passed 15 tests with one Windows symlink skip. Software execution, benchmark
integrity and model-quality qualification are separate outcomes.

## What is actually running or implemented

| Method | Actual state | Purpose and evidence |
| --- | --- | --- |
| Local Qwen3-VL-30B-A3B, BF16, vLLM on one A100 80 GB | **Deployed** | Visual scene questions; model process/image/start binding unchanged during this work. Recorded server limit is 48 sequences and evaluator/client concurrency is 64; no new server-tuning speedup is claimed. [Accepted comparison](../../benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json) |
| Single PyAV visual-stream decode, rotation/source timestamps, retained original pixels | **Deployed** | Baseline retention is 2 fps, at most 600 frames / 2 GiB; invalid timestamps, truncation or resource limits fail rather than silently shortening the clip. No audio decoding. [Decoder](../../src/forte/fast/video.py) |
| Per-video layout, full-scene and enlarged laptop/desk/seat questions | **Deployed** | Locate fixed regions once; extract at most about 100 scene states at up to 1 fps. Four optional crop questions supplement the full-scene question. Ordinary crop resize does not recover missing detail. [Analyzer](../../src/forte/fast/analyzer.py) |
| Schema-constrained JSON, temperature 0 / seed 0, finite-value validation | **Deployed** | Short structured responses and bounded request retries; request failures remain technical failures rather than clean verdicts. [Client](../../src/forte/fast/vlm.py) |
| Deterministic temporal scene rules and door-region pixel comparison | **Deployed** | Rules use observed scene states, persistence and event anchors; duplicate candidates merge in a six-second window. Door comparison requires a usable initial reference. These are not a trained temporal transformer. [Analyzer](../../src/forte/fast/analyzer.py) |
| Candidate corroboration and separate VLM verification | **Deployed accepted change** | `require_corroboration=true`, `use_verify=true`; verification samples up to ten frames around a four-second window on each side. The same VLM is reused, so agreement is not statistically independent proof. `use_holistic=false`. [Frozen profile](../../benchmarks/quality/accuracy-upgrade-verified-config-20261018.json) |
| Original-frame evidence and Russian norm-linked description | **Deployed** | One to three JPEGs near the event, source timestamps, catalogued norm and human-review wording. A visual finding does not establish invisible account/document facts. [Result builder](../../src/forte/fast/service.py) |
| PostgreSQL leases, queue admission, readiness/deadlines | **Deployed** | At most four simultaneous jobs were observed in the accepted load check; remaining jobs queue. End-to-end schema, JPEG and time-bound checks passed. [Rollout](../../benchmarks/quality/accuracy-upgrade-rollout-20261018.json) |
| RF-DETR + ByteTrack native-frame observations, checkpoint/class/gate hashes | **Implemented; disabled in accepted profile** | Optional streaming processes every native frame while retaining dense 5 fps samples for verification. Code availability does not imply qualified weights or active detector-first serving. [Specialists](../../src/forte/fast/specialists.py) |
| Optional local RTMPose hand backend | **Implemented; unqualified/inactive** | Explicit local config/weight hashes and a separate hand gate; 2-D proximity does not prove physical contact or roles. No bank hand/contact training result exists. [Specialists](../../src/forte/fast/specialists.py) |
| Specialist temporal candidate checks and abstention audit | **Implemented; inactive without applicable gates/providers** | Persistent desk objects and a key-before/after-departure path require calibrated positive observations and VLM verification. Omission/partial-handover paths require explicit source-bound facts and complete coverage; absence of a detection is insufficient. [Temporal checks](../../src/forte/fast/temporal_events.py) |
| Tracked-feature temporal transformer | **Implemented; untrained/inactive** | Half-second count/proximity/seat/optional-embedding channels and observation masks. The production fast analyzer does not invoke it; a synthetic CPU optimizer smoke is not banking-event training. [Temporal head](../../src/forte/fast/tracked_head.py) |
| RF-DETR Small/512 public-data fine-tuning | **Trained; staged on host, serving disabled** | Frozen encoder, new two-class heads, BF16, physical batch 1 / accumulation 8, balanced replacement sampling and train-only small-object zoom-out. Completed 2,752 updates; selected on public development only. [Model card](public-model-cards-20261018.md) |
| Qwen3-VL-2B crop QLoRA | **Trained; staged on host, serving disabled** | NF4 language layers, frozen visual/base weights, Q/K/V/O language-attention LoRA, rank 4 / alpha 8, 1,605,632 trainable parameters, AdamW LR `1e-4`, batch 1 / accumulation 4, 80 fixed updates. 320 balanced draws reuse 160 note / 36 ID crops; no independent support is added. Known-box two-class question with original/64/32/16-pixel stress conditions. [Frozen receipt](../../infra/public-crop2b-fit-frozen-20261018.json) |
| Provenance locks, duplicate quarantine, failure-inclusive evaluators, GPU guards | **Implemented and exercised** | Source/data/config/run/checkpoint hashes, frozen test cohort, immutable run sidecars, serialized GPU use, memory/deadline guards and local preservation. The exact source used for fitting is retained separately from later operator helpers. [Ledger](public-training-results-20261018.md) |

The accepted backend/worker image is `8c8b9255…`. The frontend overlay recorded at 23:50 UTC is `088c4875…`; all ten actually imported analyzer modules match that backend, and protected worker/Qwen bindings are unchanged. [Source/binding proof](../../infra/public-training-production-bindings-v2-20261018.json). All six protected types remain disabled: client ID omission, authenticity-check omission, recount omission, key left in safe, departure during service and irregular cash handover. Public image metrics do not qualify these procedural types.

Both trained artifacts are staged in a read-only versioned host release:
**48 hash-verified files / 135,469,180 bytes**, with configs, source/result pins
and failed quality gates. `activation=false`, `approved_types=[]`; live
configuration did not change and staging performed no inference. This completes
artifact delivery without active serving qualification.
[Manifest](../../infra/public-trained-candidates-release-20261018.json),
[copy verification](../../infra/public-trained-candidates-release-verification-20261018.json).

## Data and protection against test fitting

The frozen combined export contains **1,370 training / 905 development / 879 sealed images**, after quarantining 1,114 of 4,268 records. It combines CC BY 4.0 KZ-BD currency/coin photographs and CC BY-SA 2.5 MIDV-Holo fictional document scenes with retained Generated Photos attribution. Derived KZ JPEGs strip metadata; no organizer frame, crop, re-encode, background or label enters public fitting or calibration. [Sources, conversions and exact locks](public-training-results-20261018.md).

Paper denominations 500/1,000/2,000 remain in training, 10,000 in development and 20,000 sealed. The predeclared 5,000 training group was wholly quarantined after a prepared-image annotation conflict; no metric informed removal. Only the publisher's original training portion is used within training denominations. Holo designs 1–6 train, 7–8 develop and 9–10 remain sealed, with derivatives inheriting design membership. Checkpoints/configuration/protocol freeze before one registered test exposure per participant; failures consume their attempt. The joint cohort coordinates two different tasks, not a fusion/ensemble accuracy claim.

These checks do not create missing scene independence: the 879 sealed images have only five combined template/components; 16 ID frames cover four physical mock documents/two designs, and 863 non-ID images lack publisher scene metadata. Source/class are confounded, cross-class absence labels are not exhaustive, and upstream pretraining exposure is unknown. Compact VLM crops use author boxes; four degradations reuse the same objects. The whole immutable dataset is mounted for integrity verification; train/development-only usage and sealed access are code/hash/audit enforced, not an OS access boundary. Public success is not CCTV or banking-procedure validation.

The supplied 40 bank clips were used in earlier development and profile selection. They are a regression suite; no leave-one-out, grouped cross-validation, genuine new-angle test or 95% unseen-video result has been executed. A split-planning tool is available, but a split plan is not an evaluation. Augmented-angle copies do not simulate a genuinely unseen camera reliably.

## Bank-video accuracy

| Run on the same 40 supplied clips | Correct | TP / TN | FP / FN | Processing errors | Median nearest-event error; within 2 s |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical rv5/rv6 | 34/40 (85%) | Historical summary | 0 / 6 | Historical record | 1.0 s; 12/15 |
| Fresh unchanged-service baseline | 33/40 (82.5%) | 15 / 18 | 1 / 6 | 0 | 1.0 s; 12/15 |
| Accepted verified profile | **34/40 (85%)** | **15 / 19** | **0 / 6** | **0** | **1.5 s; 11/15** |

The accepted change removed one false alarm without losing a previously correct binary verdict; it restored the historical score and worsened the measured nearest-event timecode metric. Binary precision was 15/15 and recall 15/21 (71.43%). Labels lack event-type identities, so this is whole-video verdict regression, not per-type accuracy. It was a direct analyzer run, not a new 40-video public API test. [Baseline](../../benchmarks/quality/accuracy-upgrade-baseline-20261018.json), [matched comparison](../../benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json).

## Processing-time benchmarks

| Measurement | Work/load | Measured result | Scope |
| --- | --- | --- | --- |
| Fresh baseline, direct analyzer | 40 clips, four evaluator slots | Median **31.539 s**, maximum **68.421 s** per analysis | Excludes evaluator queue; not public upload latency |
| Accepted verified profile, direct analyzer | Same 40 clips/four slots | Median **31.320 s**, maximum **97.710 s**; total run **397.943 s** | Verification fixed one FP; no processing-speed improvement established |
| Previous unchanged-service public HTTPS | Five / six simultaneous uploads | Maximum **141.524 / 167.543 s**, 5/5 and 6/6 valid | Separate historical live load measurement |
| Accepted isolated API/database/worker stage | Six longest clips | **6/6**, maximum **137.990 s** | Internal Docker DNS, excludes public ingress |
| Accepted public HTTPS release | Six longest simultaneous uploads | **6/6 first attempt**, maximum **161.068 s**, final queue 0/0 | Client on GPU host; schema/JPEG/time bounds valid, at most four leases |
| Fresh post-training current public API | Six simultaneous uploads, exactly one group | **6/6**, zero failures/retries; maximum **142.568 s**, median **92.076 s**, makespan **142.589 s** | Warm GPU-host→public HTTPS; 19.261 s preparation excluded; no new accuracy, cold/desktop latency or causal trained-model speedup |
| RF selected-checkpoint public development | 905 images; native / fixed prediction-scoring passes | **26.217 / 14.962 s**, **34.52 / 60.49 images/s** | Includes phase I/O/preprocessing/setup/scoring; neither GPU-kernel nor whole-video throughput |
| Compact crop adapter, public development | 32 known-box crops reused across four conditions | **3.89–4.12 crops/s** including preprocessing/generation; fit **112.542 s** | Single-crop microbatches; not finding objects or processing full videos |

Sources: [direct comparison](../../benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json), [previous public load](../../benchmarks/quality/accuracy-upgrade-public-load-20261018.json), [stage](../../benchmarks/quality/accuracy-upgrade-stage-20261018.json), [accepted rollout](../../benchmarks/quality/accuracy-upgrade-rollout-20261018.json), [fresh current load](../../benchmarks/quality/final-public-six-load-20261018T005948Z.json), [RF development receipt](../../infra/public-rf-small512-selected-dev-20261018.json). The under-30-second six-upload goal remains unmet. VLM calls across the 40-clip comparison were 5,729 baseline versus 5,745 verified; the proposed tenfold reduction has not occurred. No public-candidate whole-video latency gain has been measured.

## Public trained-model measurements

| Frozen public task | Development result | Sealed result |
| --- | --- | --- |
| RF scene localization, native COCO mAP50:95 | **0.929119**, selected checkpoint; last fitting epoch was 0.900783 | **0.905428**, 879 images / one attempt |
| RF banknotes, confidence/IoU 0.5/0.5 | 287 TP / 0 FP / 4 FN; **98.625% recall**, 100% precision | **278 TP / 6 FP / 3 FN; 97.887% precision / 98.932% recall** |
| RF ID cards, same fixed protocol | 16 TP / 0 FP / 0 FN; **100% recall/precision on 16 frames** | **13 TP / 0 FP / 3 FN; 81.25% recall, below 95% target** |
| RF declared coin negatives | **0/598 false-positive images** | **1/582 (0.172%) false-positive images** |
| Compact crop VLM | **92.1875% base → 96.09375% adapter** equal-class/equal-condition macro | **94.53125% adapter-only primary macro, below 95%; 48 original crops** |

Zero observed publisher-relative FP is not a measured zero bank-alarm rate. Native AP50 can be 1 while the fixed-confidence protocol still misses four banknotes; the protocols answer different questions. Object recall is not whole-image or whole-video verdict accuracy. Exact checkpoint/run/config/protocol identities, source terms, model limits and later results belong to the [model cards](public-model-cards-20261018.md) and [result ledger](public-training-results-20261018.md).

The RF sealed result has observed FP/FN and failed the planned ID recall target;
aggregate recall would hide this weakness. Its **56.808 s** result elapsed and
**93.425 s** host elapsed span multiple phases and cannot be converted to
single-pass detector FPS. The final guard passed; the single test exposure
remains consumed, with no threshold/checkpoint retuning. [RF final receipt](../../infra/public-rf-small512-final-20261018.json),
[pre-exposure two-member cohort](../../infra/public-training-cohort-20261018.json).

Compact development uses 32 unique positive crops, 16 per class, in four
correlated views: base→adapter correct counts are original 32→32, 64-pixel 31→31,
32-pixel 28→31 and 16-pixel 27→29, each out of 32. At 16 pixels, ID accuracy
worsened **14/16→13/16**, while notes improved 13/16→16/16. Thus the macro gain
does not establish improvement in every subclass. No room-negative, absent-object
or procedural task is measured. [Completed compact fit](../../infra/public-crop2b-fit-frozen-20261018.json).

The compact sealed adapter used 32 note/16 ID originals in four correlated
conditions: original **48/48**, 64-pixel **48/48**, 32-pixel **43/48** and 16-pixel
**45/48**. At 32 pixels ID accuracy was **12/16 (75%)**, versus 14/16 at 16 pixels.
The predeclared primary macro was **94.53125%**; pooled view decisions 184/192
(95.8333%) are a different, non-primary statistic. No sealed base comparison was
performed. Preprocessing/generation measured **3.62–4.18 crops/s**; execution
guard passed in 116.193 s. These results do not supply object localization or
false-alarm performance. [Compact final receipt](../../infra/public-crop2b-final-20261018.json).

## Attempts and operational warnings retained

| Event | Recorded outcome |
| --- | --- |
| RF synthetic Nano/Small memory probes | Real one-step optimizer feasibility only; not public-data recognition benchmarks. First Nano finalizer permission failure and successful operator repeat retained. [Runtime](public-training-runtime-20261018.md) |
| RF actual fit | Native exit 0; 16 completed epochs / 2,752 updates; changed trainable and unchanged frozen hashes. **Strict host `passed=false`** because an external API frontend restart changed its binding. [Frozen receipt](../../infra/public-rf-small512-fit-frozen-20261018.json) |
| Independent RF post-fit verification and development pass | Protected worker/Qwen/database/ingress unchanged, imported analyzer bytes match; development guard **passed**. This does not overwrite the failed fit postcheck. [Post-fit](../../infra/public-rf-small512-postfit-bindings-20261018.json), [development](../../infra/public-rf-small512-selected-dev-20261018.json) |
| RF local preservation | Initial Windows-to-WSL path failure occurred before payload transfer; corrected copy matches 127,421,695-byte selected checkpoint hash, remote originals retained. [Backup receipt](../../infra/public-rf-small512-preservation-20261018.json) |
| Compact GPU smoke attempts 1 and 2 | Failed before any optimizer update; retained, no test exposure. Diagnostic isolated Triton shared-object loading. [First](../../infra/public-crop2b-smoke-failed-20261018.json), [second](../../infra/public-crop2b-smoke-v2-failed-20261018.json) |
| Compact kernel-cache remedy and third smoke | CPU comparison proved `noexec` mount failure; a bounded executable cache fixed it. Generation plus one finite optimizer update passed, adapter changed/frozen hash unchanged. [Diagnosis](../../infra/public-tmpfs-exec-cpu-20261018.json), [smoke](../../infra/public-crop2b-smoke-v3-20261018.json) |
| Compact full fit | Completed 80 updates; changed adapter/frozen hash unchanged; full guard **passed**, production unchanged and no test exposure. [Frozen receipt](../../infra/public-crop2b-fit-frozen-20261018.json) |
| Compact CPU cold reload | **Passed** exact saved-adapter binding: 224 tensor keys/shapes/F32 values, all CPU-only/frozen, no CUDA/dataset/inference. Earlier entrypoint and pre-load suffix-check helper failures retained; no fitted artifact changed. This does not prove NF4 GPU inference equivalence. [Reload receipt](../../infra/public-crop2b-cold-load-20261018.json) |

All actual GPU phases serialize through one lock beside the accepted model, with memory guards and the unchanged **11:01:25 Asia/Qyzylorda** account stop. Software-contract checks are separate from model-quality evidence: the historical source-bound core gate passed **297 tests / one Windows symlink skip / zero failures**, before later diagnostic-only helper changes. Separate targeted diagnostic, host/receiver and preservation tests are not summed into that count. [Core receipt](../../benchmarks/quality/public-training-software-checks-20261018.json).

## Considered methods that are not an executed improvement

| Method | Decision / actual state |
| --- | --- |
| Detector-first serving, GPU crops, TensorRT/ONNX export | Candidate architecture/tooling only; trained public candidates have no full-video promotion. No 3 ms/frame, tenfold VLM reduction or whole-video speedup established. |
| SAM 3 pre-labelling; CVAT/Label Studio recording workflow | Researched/available workflow; actual public run uses publisher annotations and deterministic conversions, not SAM-generated labels or reviewed own footage. |
| Large Qwen3-VL-30B LoRA or Qwen3.6-27B migration | Not fitted/deployed; actual experimental adapter uses compact Qwen3-VL-2B. |
| Trained temporal/SigLIP2 event producer and bank hand-contact models | Temporal transformer code exists, but no active SigLIP feature producer or trained procedural checkpoint/measurement. Bank contact models remain unqualified. |
| MLflow experiment export | Native external logging disabled; actual evidence is local immutable/hash-bound receipts. Public-fit MLflow export remains optional. |
| Additional public cup/hand/action sources or TAPNext tracking | Primary-source/license research only; not incorporated into the frozen two-class fit. [Dataset review](public-dataset-research-20261018.md), [action review](public-action-dataset-research-20261018.md) |
| FP8 on H100, model-server sequence tuning, second-account migration | No H100/FP8 or migration execution; live A100 BF16 model/settings preserved. Bootstrap/configuration support does not establish a measurement. |
| Super-resolution as evidence; noncommercial model/data assets | Not adopted. Generated detail cannot establish source evidence; exact code/weight/data licenses are tracked separately. [License record](../submission/licenses.md) |

No 40/40, 95% unseen performance, perfect new-angle performance or guaranteed 100% result is established. The remaining six bank misses require task-specific visibility and labelled sequence evidence; public banknote/document images alone cannot supply key absence, identification timing, detector contact, recount coverage or complete cash handover.

The final [source/operator bundle](../../dist/forte-release-source-20261018-final-trained.zip)
has a [manifest](../../dist/forte-release-source-20261018-final-trained.manifest.json)
and [verification record](../../dist/forte-release-source-20261018-final-trained.verification.json). It excludes
media, model weights and private state. Separately preserved exact fitting-source
archives and private local weight backups retain the artifact identities used
by the experiments; a later operator bundle does not replace those histories.
