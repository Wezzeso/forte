# Public-data training: results and acceptance state — 18 October 2026

**Both fits, reload and single frozen sealed attempts are COMPLETE. Quality targets failed: RF ID recall is 13/16 (81.25%), with six banknote false positives; compact crop primary macro accuracy is 94.53125%, below 95%.** The public data, checkpoint/adapter and protocols were frozen before registration; neither candidate is authorized for activation. The strict RF fit host postcheck failed after an independent frontend replacement; native training exited normally, and later execution guards passed. Execution success does not mean quality qualification. The 100% target and faster processing remain goals. [All methods and benchmark tables](accuracy-methods-benchmarks-20261018.md).

The [final machine-readable benchmark](../../benchmarks/quality/accuracy-final-benchmark-20261018.json)
binds the source receipts and separates exposed bank regression, public-object
tests, current API load and disabled staging. Its aggregate checks passed 19
tests. [Final source/stager checks](../../benchmarks/quality/final-source-delivery-checks-20261018.json)
passed 15 tests with one Windows symlink skip; these source-bound software
results are separate from trained quality and are not summed with older gates.

## Evidence available now

| Source | Verified acquisition and usable supervision | Limits |
| --- | --- | --- |
| **KZ-BD v1** | Author [Mendeley record](https://data.mendeley.com/datasets/dny3dgvvw8/1), CC BY 4.0. Received 6,187,937,805-byte archive SHA-256 `daae1a4eb371a7b6ff672e0d7c83edbe27f85ff02a584bb9ea42ab3856eec392`, matching the publisher. Inspection found 4,200 images: 1,800 paper-currency and 2,400 coin images; published train/validation counts 2,940/1,260. | Class IDs do not reliably encode the 14 denomination folders. TXT files can label nested currency/features; use an unambiguous outer box and quarantine ambiguous/disjoint/invalid cases. Original coordinates are normalized center-XYWH. No verified camera/session or physical-banknote groups. |
| **MIDV-Holo** | Author [repository/license](https://github.com/SmartEngines/midv-holo), CC BY-SA 2.5, artificial Utopia documents with Generated Photos attribution. Acquired templates/markup/metadata and 68 scene JPEGs with exact author document quadrangles. Scene split declared before payloads: **36 train / 16 development / 16 sealed**, across all ten ID designs and 17 document/clip groups. | Phone-captured document localization differs from overhead CCTV. Passport pages are a separate category. Full-canvas transformed template labels are synthetic; scene quadrangles are author annotations. Neither source supplies bank procedural labels. |

Private evidence files:

- `training/private/public_datasets/kz_bd_v1/dataset-inspection-receipt.json`, SHA-256 `621ad5c99c89dd2df5c2f8751e85b5340c46d0c4239f1a6e1a8b8a56cd7f1559`.
- `training/private/public_datasets/midv_holo/acquisition-manifest.json`, plus retained license/citation/Generated Photos attribution and structural inventories.
- `training/private/public_datasets/midv_holo/scene_subset/acquisition-manifest.json`, SHA-256 `ffabf76b503d49632bb0d68b4770836350cd443a60c3d8d01ee78a329d2a81a0`; every selected JPEG/annotation pair is individually hashed. The whole 14 GB image archive was not downloaded or hashed.

KZ inspection found 91 exact duplicate groups involving 182 images, with none crossing the published split or denomination folders. This does not establish absence of near-duplicates. Twenty-four files had ambiguous outer boxes (19 paper, five coin); five invalid annotation rows occurred in three coin files. The stricter preparation/merge quarantine below is broader than this initial acquisition screen.

## Frozen prepared export

The combined export at `training/private/public_datasets/combined-banknote-id-v1` passed full portable verification and the fit-bundle/augmentation preflight. Its `dataset-lock.json` SHA-256 is `eef3f1ad5fabb9cb720e42522265a6946a02196af01221af6ebf0522bf498fb8`, and dataset SHA-256 is `2439037ef673dbb5ab47dc0fab15575e687e4da8243d84d29261e6e44ac745ec`. The portable closure contains **11,630 files / 1,253,289,866 bytes**, retaining CC BY 4.0, CC BY-SA 2.5 and Generated Photos attribution.

| Prepared records | Train | Development | Sealed test |
| --- | ---: | ---: | ---: |
| Total included images | **1,370** | **905** | **879** |
| Banknote positives | 616 | 291 | 281 |
| ID-card positives | 36 | 16 | 16 |
| Coin negatives | 718 | 598 | 582 |

Of 4,268 source records, **1,114 are quarantined**. During merging, an exact prepared-image duplicate had conflicting target annotations. The existing group-closure policy therefore quarantined the **entire 5000-tenge group**, adding 198 otherwise included training images (quarantine 916 → 1,114). The retained training banknotes cover **500/1,000/2,000**, not all four originally planned denominations. Source 5000-tenge records remain in the audit and were not fitted. No model metrics informed this exclusion or relaxed the preregistered policy.

These counts are annotated images/boxes, not independent physical-note, scene or transaction support. Holo's 68 frames retain their design grouping; adjacent frames and repeated views do not become new independent identities. Each publisher annotated its own target class; merging does not establish exhaustive two-class absence labels in every image or banking-scene validity.

The separate [runtime record](public-training-runtime-20261018.md) distinguishes synthetic memory probes/CPU staging from the completed public-data fits and single frozen joint evaluations below.

The detector's exact 61-file fitting source is separately preserved in [the frozen source archive](../../dist/forte-public-training-frozen-source-20261018.tar.gz), with [manifest](../../dist/forte-public-training-frozen-source-20261018.source-manifest.json) and [member verification](../../dist/forte-public-training-frozen-source-20261018.verification.json). This source-only artifact contains no private media or model weights and preserves the actual fit closure independently of later operator-tool updates.

## Completed detector fit

The fixed RF-DETR Small/512 process exited **0** after **16 completed epochs** (last zero-based epoch index 15), **2,752 optimizer updates** and **22,016 physical batches**. Replacement sampling reused the 1,370 available training images; those batches are not independent data support. Fit wall time was **2,106.128 s / 35.10 min**, including recurring development validation; whole-run elapsed was 2,145.755 s. Trainable-parameter hashes changed while the frozen-parameter hash remained unchanged, and recorded losses were finite. [Completed fit/frozen receipt](../../infra/public-rf-small512-fit-frozen-20261018.json).

| Native development measurement | mAP at IoU 0.50:0.95 |
| --- | ---: |
| Newly initialized target head on pretrained backbone, before updates | 0.062256 |
| First completed epoch, index 0 | 0.883368 |
| Best development checkpoint selected during fitting | **0.929119** |
| Last completed epoch, index 15 | 0.900783 |
| Separate selected-checkpoint rehearsal | **0.929119** |

The initial and first-epoch values retain their [baseline](../../infra/public-rf-small512-baseline-dev-20261018.json) and [progress](../../infra/public-rf-small512-first-optimizer-20261018.json) receipts. The best checkpoint is `checkpoint_best_total.pth`, **127,421,695 bytes**, SHA-256 `f01c7b9c897f6eee5a519a72d9dbe5d9f7344ebec4bf9e0187dcd44b683f848e`; immutable run SHA-256 is `662def0d466404ad9efa1e828ddcda372915cf9487733add0c9c881562307d2b`. These native development values do not establish held-out or bank-video accuracy. The run records **zero test evaluations**.

The checkpoint also has a [verified local backup](../../infra/public-rf-small512-preservation-20261018.json) under `training/private/public_runs/20261018/rf-small512-v1-preserved`, matching the same byte count/hash; remote originals remain retained. An initial Windows-to-WSL path failure occurred before payload transfer and is preserved in the receipt alongside the corrected successful copy. No source images or datasets are in that weight backup.

The guarded [selected-checkpoint development rehearsal](../../infra/public-rf-small512-selected-dev-20261018.json) completed on **905 development images**, using the same frozen checkpoint/run/config/protocol. At fixed confidence **0.5** and same-class IoU **0.5**, it recorded:

| Public development object | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Banknote | 287 | 0 | 4 | 100% | 98.625% |
| ID card | 16 | 0 | 0 | 100% | 100% |

Declared coin-negative false-positive images were **0/598**. These are annotation-relative public-object measurements; 16 ID frames are limited support, and zero observed false positives does not establish a zero bank-alarm rate. The common scorer's internal split name remains `sealed_test`, but the wrapper identifies the actual source as `validation`, records `development_only`, zero test evaluations and no sealed pixel access. No checkpoint selection or threshold changed during this rehearsal.

Native evaluation took **26.217 s / 34.52 images/s**; the separate fixed prediction/scoring pass took **14.962 s / 60.49 images/s**. These component timings include phase I/O, preprocessing, scoring and setup; they are not GPU-kernel throughput and do not include video decoding, tracking, VLM verification or queueing. Peak allocator usage was **310.52 MiB allocated / 342 MiB reserved**, versus **876 MiB** maximum sampled host GPU usage. The development host guard passed, with unchanged production bindings, readiness true and queue 0/0. This separate pass does not erase the fit guard failure described below.

The strict host report is **`passed=false`**, with `production_postcheck_failed`: the independently updated API frontend changed its container binding during fitting. There was no native training failure, and immutable inputs stayed unchanged. The separate [post-fit verification](../../infra/public-rf-small512-postfit-bindings-20261018.json) confirmed all imported analyzer files match the accepted backend, protected worker/Qwen/database/ingress bindings unchanged, readiness true, queue 0/0 and 7,181 MiB GPU memory free. This evidence permits auditing the completed artifact; it does not rewrite the failed strict guard as a pass.

The launch configuration file is pinned as `4287852e…`; the runner's saved effective configuration is `194c888c1c957c39942d4d59a03e206c3dd367332cf7e64d59cee5f97ca14e16`. They parse identically; only serialization differs. Development, cohort and final artifact records must use the saved effective identity while retaining the launch pin. [Equivalence receipt](../../infra/public-rf-small512-config-equivalence-20261018.json).

**Metadata handling:** GPS EXIF tags are present in 2,564 KZ source images; coordinates were not read/exported. The paper's de-identification statement is an author claim, not independently verified privacy clearance. Derived training JPEGs are metadata-stripped and the prepared export passed verification; source archives retain their separate provenance. No real ID/customer records are part of this experiment; Holo document identities are author-declared artificial. One public KZ image, `10000_1.jpg`, was viewed to validate outer-box interpretation. Holo faces were not visually reviewed, and no organizer/private footage was opened.

## Compact crop-VLM attempts

The first guarded [Qwen3-VL-2B GPU memory smoke](../../infra/public-crop2b-smoke-failed-20261018.json) **failed with exit 2 and zero optimizer steps**. It produced no completed adapter or optimizer proof. The bounded log records a bitsandbytes CUDA-12.8 fallback warning under Torch CUDA 12.9, but the original runner suppressed the underlying exception; that warning alone does not establish the cause. The failed attempt remains retained. Production bindings stayed unchanged, readiness passed, queue was 0/0 and test exposure was zero.

A subsequent GPU-hidden [CPU/meta compatibility diagnostic](../../infra/public-crop-cpu-compatibility-20261018.json) passed import/module-path/shared-library checks without loading weights or initializing CUDA. A [second diagnostic GPU smoke](../../infra/public-crop2b-smoke-v2-failed-20261018.json) also failed before updates, now exposing the `cuda_utils` shared-object load path in Triton. Both failed attempts remain retained, with no sealed exposure.

The [CPU mount comparison](../../infra/public-tmpfs-exec-cpu-20261018.json) then reproduced the failure: the guard's `/tmp` was mounted `noexec`, blocking a freshly compiled shared object from loading; an explicit executable tmpfs loaded the same probe successfully. This identified an execution-policy fault, not a measured model-quality failure. The bounded kernel-cache mount repair did not change the model, data, image, recipe or memory caps.

The repaired [third guarded GPU smoke](../../infra/public-crop2b-smoke-v3-20261018.json) **passed**: generation completed, followed by one finite optimizer update with nonzero gradient, changed adapter hash and unchanged frozen-parameter hash. Its run SHA-256 is `6022b4f5975fc94cb7e4cc1315df28542941f5fe644ad3ce3793efb7a7f3f1f6`. Peak allocator allocation/reservation was **3,636.42 / 3,970 MiB**; the host sampled **4,538 MiB** maximum trainer memory and **2,638 MiB** minimum global free memory. Production/immutable inputs stayed unchanged, readiness passed and queue was 0/0. Test exposure remained zero. This smoke alone is a one-step feasibility proof; the separate full fit and accuracy results follow below.

The [full compact fit](../../infra/public-crop2b-fit-frozen-20261018.json) then **completed 80 optimizer steps** with finite recorded losses/nonzero gradients, changed adapter weights and unchanged frozen weights. Fit wall time was **112.542 s**; command elapsed including loading/development was **197.716 s**, and host supervision elapsed **210.733 s**. Configuration SHA-256 `d029457071b32e13fb71a0563fdbd1e40f124dfbeeea6a03ccbd960e34625f64` binds the successful smoke; actual diagnostic runner source is `43a9153a…`, within manifest `779e2f56…`. Immutable run SHA-256 is `aaec63508133bf227170be43e10f426d6804cd09c804fdc58b4e913e33e0ffcd`; saved `adapter_model.safetensors` SHA-256 is `a3733ccf3c888224c2887dd24900e45d2462e9481299ed71cec82518cfc6c7ad`.

| Known-box public development condition | Base correct / 32 | Adapter correct / 32 | Base → adapter macro accuracy |
| --- | ---: | ---: | --- |
| Original | 32 | 32 | 100% → 100% |
| 64-pixel short side | 31 | 31 | 96.875% → 96.875% |
| 32-pixel short side | 28 | 31 | 87.5% → 96.875% |
| 16-pixel short side | 27 | 29 | 84.375% → 90.625% |
| Equal-class/equal-condition primary score | Correlated views | Correlated views | **92.1875% → 96.09375%** |

These are the same 32 unique source crops (16/class) reused in four conditions, not 128 independent objects. At 16 pixels the ID subclass worsened **14/16 → 13/16**, while notes improved 13/16 → 16/16; the macro gain does not mean every subclass improved. Adapter preprocessing/generation measured **3.89–4.12 crops/s** by condition, not full-video speed. The full fit guard passed with unchanged inputs/production, readiness true, queue 0/0; host maximum trainer memory was **4,558 MiB**, minimum global free memory **2,618 MiB**, allocator peak **3,636.42 MiB allocated / 3,990 MiB reserved**. No test exposure occurred.

The [saved-adapter CPU cold reload](../../infra/public-crop2b-cold-load-20261018.json) then passed in **30.468 s**: the official BF16 CPU base and saved PEFT adapter loaded, and all **224 adapter tensor keys/shapes/F32 values / 1,605,632 parameters** matched exactly. Parameters/buffers stayed CPU-only and frozen, CUDA was uninitialized, and input hashes were unchanged. There was no processor, dataset, inference or held-out access. It proves saved-artifact binding, not equivalence to NF4 GPU inference or additional accuracy. The initial operator entrypoint failure and a pre-load helper check's PEFT-suffix serialization mismatch remain retained alongside the successful helper-only repair; fitted source/config/adapter stayed frozen. Sealed evaluation began only after both candidates froze, as recorded below.

The exact compact fitting closure is separately preserved in [its source-only archive](../../dist/forte-compact-training-frozen-source-20261018.tar.gz), [manifest](../../dist/forte-compact-training-frozen-source-20261018.source-manifest.json) and [member verification](../../dist/forte-compact-training-frozen-source-20261018.verification.json). It is the 61-file `repo-v3` closure; the earlier RF archive is `repo-v2` and is not the compact fit's final source identity.

The [verified local compact backup](../../infra/public-crop2b-preservation-20261018.json)
retains the 6,455,336-byte saved adapter and its config under
`training/private/public_runs/20261018/crop2b-fit-v1-preserved`; hashes match the
frozen run, remote originals remain retained and no source images are included.

## Registered sealed evaluation

The [two-member cohort](../../infra/public-training-cohort-20261018.json) was
registered at **00:49:59 UTC**, binding both frozen model/run/config/protocol
identities and the same sealed inventory before either exposure. It permits
one attempt per participant, including failures, without reset or test tuning.

The [single RF sealed attempt](../../infra/public-rf-small512-final-20261018.json)
completed **879 images**, with native mAP50:95 **0.905428**. Fixed confidence/IoU
0.5/0.5 gave:

| Sealed public object | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Banknote | 278 | 6 | 3 | 97.887% | 98.932% |
| ID card | 13 | 0 | 3 | 100% | **81.25% — below the 95% target** |

Declared coin-negative false-positive images were **1/582 (0.172%)**. Thus the
public detector has observed false positives and misses; high aggregate/native
metrics do not remove the failed fixed-threshold ID recall target. The ID
denominator is only 16 correlated frames, and this public-object test does not
satisfy own-camera/full-video promotion even if a numerical target had passed.
No threshold/checkpoint tuning or second test is permitted from these results.

Whole result elapsed was **56.808 s**, covering setup/native/fixed/scoring work;
host supervision was **93.425 s**. Neither is a single-pass inference-FPS
measurement. The final host guard passed with unchanged production/inputs,
readiness true, queue 0/0 and 876 MiB maximum sampled evaluation GPU memory.
Attempt count is **one**, `serving_eligible=false`, `approved_types=[]`.

The [single compact sealed attempt](../../infra/public-crop2b-final-20261018.json)
also completed. It evaluated the adapter only: **48 original known-box crops**,
32 notes and 16 IDs, reused in four correlated conditions. There is no sealed
base-model comparison and therefore no measured held-out before/after gain.

| Sealed crop condition | Notes correct / 32 | IDs correct / 16 | Pooled correct / 48 | Equal-class accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original | 32 | 16 | 48 | 100% |
| 64-pixel short side | 32 | 16 | 48 | 100% |
| 32-pixel short side | 31 | **12** | 43 | **85.9375%** |
| 16-pixel short side | 31 | 14 | 45 | 92.1875% |

The predeclared equal-class/equal-condition primary score is **94.53125%**,
below 95%. Pooled view accuracy is 184/192 (95.8333%); unequal class support
makes that a different statistic, not the primary result. The 192 views are
not independent objects. ID performance at 32 pixels is **12/16 (75%)**. Zero
invalid outputs were observed, but forced two-class positive crops supply no
absent-object or bank-alarm rate.

Per-condition preprocessing/generation took **13.249 / 11.492 / 11.758 / 11.625 s**
for original/64/32/16 conditions, **3.62–4.18 crops/s**. The host guard passed
in **116.193 s**, with unchanged production/inputs, queue 0/0, readiness true
and 2,704 MiB maximum sampled evaluation memory. Attempt count is one; no test
reuse/tuning or promotion is allowed. Both final guards establish successful
execution, not a passing recognition target.

## Host release and preservation

The [candidate release manifest](../../infra/public-trained-candidates-release-20261018.json)
and [copy verification](../../infra/public-trained-candidates-release-verification-20261018.json)
record **48 files / 135,469,180 bytes** copied and hash-verified into the read-only
host directory `/home/ubuntu/forte/deploy/releases/public-specialists-20261018-0ab06fd0`.
Weights/configs/source/result/attempt identities remain bound together. State is
**`staged_disabled`**, `activation=false`, `serving_eligible=false`,
`promotion_eligible=false`, `approved_types=[]`; both failed quality gates are
explicit. Staging performed no model/test inference and changed no live
configuration. Local model backups and [final sidecar preservation](../../infra/public-training-final-preservation-20261018.json)
retain the work separately from the unchanged account stop.

## No-overfit protocol

1. Pin source archives, received bytes, annotation conversions, attribution, class mappings and every derivative's parent hash. Use only authorized public-training partitions; organizer/jury recordings and their crops, backgrounds, re-encodes or labels do not enter fitting/calibration.
2. Follow the preregistered KZ whole-denomination policy: paper **500/1,000/2,000/5,000** assigned to training, **10,000** to development and **20,000** to the sealed test; coins **1/2/5/10** to training, **20/50** to development and **100/200** to the sealed test. Use only the author's published training partition within training denominations; exclude their original published validation images. Whole development/test denominations follow the declared policy independently of the author's random split. This assignment was fixed without metric feedback; strict conflict quarantine subsequently removed all 5000-tenge records as reported above. Do not invent scene IDs. Holo designs 1–6 are training, 7–8 development and 9–10 sealed; every derivative of a design inherits that role.
3. Select fit settings/checkpoints using training and public development data only. **Freeze both candidates' weights, configuration, thresholds, importer and split hashes before the joint sealed test.** Record the freeze receipt before executing that test. A failed sealed test stays a failure; it does not permit another tuning pass against the same test.
4. The 40 supplied bank clips were already used for development. They remain a regression suite, not an untouched generalization test. Do not choose public weights, prompts or thresholds by repeatedly fixing those clips. Any later frozen comparison must preserve failed/unprocessed items in its denominator.

The host mounts the whole immutable combined dataset, including sealed files, for integrity verification. Fit code uses training/development views with `run_test=false`; the sealed view is materialized only after its recorded claim. Separation is enforced by code, hashes and the exposure audit, **not by operating-system denial of access to sealed bytes**.

## Scope of any final public result

- **Source and class are confounded:** ID cards come from MIDV-Holo's artificial documents, while notes/coin negatives come from KZ-BD. Models can learn publisher, design, capture or background cues. The splits do not establish performance when both classes appear together in an unfamiliar bank camera.
- Published pretraining inventories do not establish whether the initial detector/VLM encountered these public sources. The local split and derivative checks cannot prove absence of upstream pretraining exposure.
- The **879 sealed images comprise five combined template/components**, not 879 independent scenes. Their 16 ID frames cover four physical mock documents across two designs; the 863 non-ID images lack publisher scene/capture metadata. Exact content/component separation does not create missing scene independence.
- Detector false positives are measured against the publishers' annotations and declared label scope. They are **not bank false alarms** or proof of exhaustive absence of the other class in every photograph.
- Native detector scene evaluation uses 512-pixel input and the public images' original object scales. The 12–96-pixel zoom-out augmentation is training-only; it does not create a held-out CCTV tiny-object test.
- Compact VLM evaluation uses **known author boxes** for a two-class crop question and the normal processor resize. It does not measure finding the object, deciding there is no object, procedural compliance or an open-ended new violation. Original/64/32/16-short-side conditions reuse the same source objects; four views do not quadruple independent support, and a 16-pixel crop result does not establish recall for a 4×6-pixel key.

Any final metric table must retain these limits, identify the selected frozen artifact and separate development, sealed object/crop results, and bank regression.
The joint cohort coordinates one registered exposure for two frozen participants
with different tasks; it is not a trained fusion pipeline or an ensemble-accuracy measurement.

## Execution/result ledger

| Gate | State | Evidence required to change state |
| --- | --- | --- |
| Public source acquisition | **COMPLETE** for the sources above | Existing source/inspection receipts; no model-quality claim |
| Prepared data and privacy/split checks | **COMPLETE** | Frozen combined lock, exact counts, metadata removal, conflict/duplicate quarantine and portable verification above |
| Actual public-data fits | **COMPLETE** | RF completed 2,752 updates; compact completed 80; parameter-change/frozen-weight proofs and earlier failures retained |
| Public development selection | **COMPLETE** | RF frozen rehearsal and compact base/adapter development results above; zero sealed-test exposure |
| Both candidates frozen | **COMPLETE; cold reload verified** | Both checkpoint/adapter/config/protocol hashes recorded before registration; saved adapter binding passed CPU reload |
| Joint-test cohort | **REGISTERED before exposure** | [Authorization](../../infra/public-training-cohort-authorization-20261018.json) and [cohort](../../infra/public-training-cohort-20261018.json) bind both frozen participants and inventory |
| Joint sealed public test | **COMPLETE; quality targets FAILED** | One attempt each; RF ID recall 81.25% / six FP, compact primary macro 94.53125%; no tuning/reset |
| Bank-video candidate regression | **NOT RUN in this report** | A frozen full-video comparison, typed/time evidence where labels exist, no added clean-clip alarms; public object metrics alone are insufficient |
| Candidate host release | **STAGED, DISABLED; verified** | 48 hash-verified read-only files, failed quality gates retained; no inference/live change |
| New-model activation | **NOT ELIGIBLE** | Own-camera/task coverage and existing full-video/policy/operational promotion gates; explicit accepted release record |

Both completed fits, development measurements, CPU reload and single sealed results are recorded above. No new bank-video accuracy result for these public candidates was run; task/domain promotion remains unmet. Import/preflight, execution success and public-object scores do not complete those gates.

The actual recipe uses local hash-bound receipts with native external MLflow logging disabled. The existing tracker targets the own-recording schema; exporting public-fit receipts to MLflow remains optional integration work, not an executed experiment log. The historical coherent core software gate passed **297 tests, with one Windows symlink-creation skip and zero failures, in 65.27 s**. Its [hash-bound receipt](../../benchmarks/quality/public-training-software-checks-20261018.json) covers the recorded public import/splits, augmentation, fitting/evaluation contracts, crop/cohort and release source versions; it predates the later diagnostic-only crop helper changes. It replaces earlier overlapping core counts. Targeted operator/diagnostic checks and separate Linux host/receiver checks are not added to this total; software checks do not establish trained recognition quality.

## Bank accuracy and speed remain separate measurements

The accepted analyzer's previous direct run scored **34/40 (85%), zero false alarms, six misses and zero processing errors**, restoring the historical rv5/rv6 score. Median nearest-event timecode error was **1.5 s**, compared with **1.0 s** in the fresh unchanged-service baseline; detections within 2 s were 11 rather than 12. This was binary development regression, not unseen accuracy or a new 40-video public API test. [Recorded comparison](../../benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json).

The accepted release's historical six-upload maximum was **161.068 s**. The [fresh single post-training public load check](../../benchmarks/quality/final-public-six-load-20261018T005948Z.json) passed **6/6 accepted/done/schema-JPEG-timeline checks**, **zero failures/retries**, maximum **142.568 s**, median **92.076 s**, mean **89.328 s** and makespan **142.589 s**. The 240-second gate passed; the **30-second target failed**. Preparation of 19.261 s was excluded from upload timing. This warm GPU-host→public-HTTPS check is not desktop/cold latency, new accuracy or proof that inactive trained models accelerated serving. [Historical accepted report](../../benchmarks/quality/accuracy-upgrade-rollout-20261018.json).

The frontend-only API overlay remains `088c4875…`, worker `8c8b9255…` and the Qwen process/image/start binding are preserved. [Post-load verification](../../infra/public-training-final-postload-bindings-20261018.json) matched all ten imported analyzer files, readiness true and queue 0/0. The [effective profile](../../infra/public-training-final-profile-binding-20261018.json) retains corroboration/verification and disabled specialists; the [11:01:25 budget stop](../../infra/public-training-final-budget-guard-20261018.json) is unchanged. These operational facts do not qualify a public-data candidate.

Banknote/document pictures do not establish ID presentation before cash, a key's absence after departure, physical hand/device contact, full recount coverage, note counts or an incomplete handover. Missing detections are not verified omissions. Those types remain gated even if public localization metrics are strong. The roughly 7 GB spare GPU allocation and the existing **11:01:25 Asia/Qyzylorda** stop constrain the experiment; memory fit, training completion and speed must be measured rather than promised.

Artifact provenance and limits: [public model cards](public-model-cards-20261018.md). Source selection and alternatives: [public dataset research](public-dataset-research-20261018.md), [action-data research](public-action-dataset-research-20261018.md), [specialist workflow](../../training/specialists/README.md).
