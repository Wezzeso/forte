# Public specialist model cards — 18 October 2026

These cards describe two reusable **public-object research candidates**. Both fits, development evaluations and the predeclared one-attempt joint sealed test completed. The detector's fixed-threshold ID recall was **81.25%**; the crop adapter's primary sealed score was **94.53125%**. Their frozen artifacts are ineligible for production activation under the current gates. `serving_eligible=false`, `promotion_eligible=false`, and `approved_types=[]` apply to both candidates.

The public experiment measures banknote/document appearance. It has no banking-procedure supervision, no key/contact/stack labels, and no evidence of improved bank-video verdict accuracy. Source and class are confounded: banknotes/coin backgrounds come from KZ-BD; ID cards come from MIDV-Holo. The models can learn publisher, design, camera or background cues. The upstream models' pretraining corpora have not been audited for overlap with either public dataset; pretrained contamination is **unknown**. Detector/VLM agreement is not established as statistically independent evidence; these candidates share public data and correlated limitations.

## Shared data and evaluation identity

| Field | Frozen value |
| --- | --- |
| Dataset | Combined KZ-BD v1 banknotes/coin backgrounds and MIDV-Holo fictional ID-card scenes |
| Dataset-lock schema | `forte.specialists.public-dataset-lock.v1` |
| Dataset-lock SHA-256 | `eef3f1ad5fabb9cb720e42522265a6946a02196af01221af6ebf0522bf498fb8` |
| Dataset digest | `2439037ef673dbb5ab47dc0fab15575e687e4da8243d84d29261e6e44ac745ec` |
| Included images | Train **1,370**; development **905**; sealed **879** |
| Banknote-positive images/boxes | Train 616; development 291; sealed 281 |
| ID-card-positive images/boxes | Train 36; development 16; sealed 16 |
| Coin-background images | Train 718; development 598; sealed 582 |
| Quarantine | 1,114 of 4,268 source records |
| Portable export | 11,630 files; 1,253,289,866 bytes; individual byte hashes and nested source locks retained |
| Runtime class order | `0: single_banknote`, `1: id_card`; COCO export category IDs `1,2` |
| Organizer data | No organizer/jury frame, crop, label or background enters this public fitting/selection pipeline |

Paper denominations 500/1,000/2,000 remain in training, 10,000 in development and 20,000 sealed. The predeclared 5,000-tenge training group was wholly quarantined after exact prepared-image duplicates had conflicting annotations; no model metric informed this exclusion. Only the publisher's original training portion of training denominations was eligible for fitting. Coin denominations are grouped by role. Holo designs 1–6 train, 7–8 develop and 9–10 remain sealed; physical documents, clips and derivatives inherit their design's role. Exact/near-duplicate and group conflicts are quarantined under the frozen policy.

Image counts do not establish independent scenes or transactions. The 879 sealed images cover **five combined template/components**. The 16 sealed ID frames cover four physical mock documents across two designs; KZ-BD supplies no reliable physical-banknote/session/camera groups. Source labels do not establish exhaustive cross-class absence in every photograph. Derived KZ JPEGs strip EXIF/GPS metadata; original source evidence stays in the audit.

The methods perform different tasks: RF-DETR searches complete images, including coin backgrounds; the crop VLM receives a known positive object box. Their detection AP/recall and crop-classification accuracy have different denominators. Input populations and preprocessing likewise differ, so the timing tables are not a head-to-head bank-video accuracy or latency benchmark.

Both candidates must finish training/development selection and freeze checkpoint, run, config, protocol and relevant source hashes before the authorized two-member cohort can consume the sealed inventory. The central exposure registry allows one attempt per frozen participant, including failed attempts. Selected runs remain immutable; evaluation results and attempt markers are separate sidecars. The shared inventory claim is not a bank deployment approval. [Experiment ledger](public-training-results-20261018.md), [public-data protocol](accuracy-public-training-protocol-20261018.md), [cohort helper](../../training/specialists/public_cohort.py).

The [actual joint cohort](../../infra/public-training-cohort-20261018.json) was registered at **2026-10-18 00:49:59 UTC**, before evaluation, with SHA-256 `0ab06fd0e9c17c259b8464fc43d7466f8b55cc2e2bfd8182df5ab73942a0a437` and sealed-inventory SHA-256 `6fd01ecf63dceb738a069626bdebf59e286565e73c0c38b67a9715097b474a29`. Its members are exactly `detector` and `crop_vlm`; interpretation is one joint comparison, not independent replications. Fitting runs retain their historical zero-test counters after testing; final attempt/result sidecars determine actual exposure and must accompany them.

The historical **RF actual-fit source, repo-v2**, is preserved as [the frozen source archive](../../dist/forte-public-training-frozen-source-20261018.tar.gz), SHA-256 `c1287008592a539d384568a8ddfb257c6c9ffb10ca4a29a806c57e1ed17c665d`. Its [61-file source manifest](../../dist/forte-public-training-frozen-source-20261018.source-manifest.json), SHA-256 `37632be61eda7a8b81ac63617c89d06b484343e90bc7fb3aebd7ac1d04b1cf1a`, passed [member-by-member verification](../../dist/forte-public-training-frozen-source-20261018.verification.json). The timed RF development rehearsal uses a separately pinned helper overlay. **Compact fitting used repo-v3**, preserved in [its own frozen archive](../../dist/forte-compact-training-frozen-source-20261018.tar.gz), SHA-256 `fcc6822e69b70555c194850f483d47e60fc04b6abc5c6e50039e43485a8416d3`, with [61-file manifest](../../dist/forte-compact-training-frozen-source-20261018.source-manifest.json) SHA-256 `779e2f56f8ab575737b4113577bdb5ed9fe293f05cce73bc1d2cc3894120e137` and [verification proof](../../dist/forte-compact-training-frozen-source-20261018.verification.json). These source-only archives contain no media or weights. They are distinct source closures; a broader release must preserve each historical fitting identity.

## RF-DETR public localizer

**Intended artifact:** a two-class object localizer/warm start for research on visible banknotes and ID cards. It accepts an RGB image and returns boxes, confidence scores and class IDs. A visible ID card does not establish identification; a visible note does not establish a stack, cash amount, authenticity check, full handover or recount.

| Field | Value / acceptance state |
| --- | --- |
| Architecture | `RFDETRSmall`, RF-DETR **1.11.2**, input resolution **512** |
| Initialization | Official pretrained Small checkpoint; pretrained localization/backbone retained, new random two-class target heads; no COCO semantic row is relabelled as a banknote |
| Initial checkpoint SHA-256 | `d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33` |
| Encoder | Frozen; unchanged frozen-parameter hash required in the completed run |
| Fitting | BF16, physical/evaluation batch 1, accumulation 8, learning rate `1e-4`, encoder LR 0, at most 30 epochs; development early stopping and best-checkpoint selection |
| Other recipe fields | Fixed scale; no EMA, compilation, CUDA graphs or encoder gradient checkpointing |
| Training sampler | Equal expected mass for banknote, ID-card and verified coin-background groups; replacement draws add no independent support |
| Training-only augmentation | Positive-draw zoom-out probability 0.8; longest target box uniform 12–96 px at the 512-px reference on a neutral canvas; exact box/area transform; original smaller objects are not upscaled |
| Augmentation policy SHA-256 | `40bbf0c628dac2cbb002464e57d2fef3152356a80fb75f458e9782bacc6966be` |
| Launch config-file SHA-256 | `4287852e1bf43d1c9865e2fb86416602436d3926dd6aaf9c93383865cb67e45e` |
| Saved effective `output/config.json` SHA-256 | `194c888c1c957c39942d4d59a03e206c3dd367332cf7e64d59cee5f97ca14e16`; this is the immutable run/development/cohort/final configuration pin |
| Fitting/final runner SHA-256 | `87f33ae226ddd32ddb32b85691ba8e86a3f686fe67190cdb15b93175f72f87c6` |
| Separate timed-development helper SHA-256 | `e3a0fe00df318aaccf047205c2b1df931714d37a32658989d6a7ea54d3f3136c`; timings wrap the existing two development passes without extra model calls |
| Runtime image SHA-256 | `2a7af0c822817bda42d8eabba5085b1756ffd051d37f2c3aa3ec138a166462ba` |
| Allocator budget | 5,120 MiB; serialized shared-GPU execution and external deadline monitoring |
| Selected checkpoint | `checkpoint_best_total.pth`, **127,421,695 bytes** |
| Selected checkpoint SHA-256 | `f01c7b9c897f6eee5a519a72d9dbe5d9f7344ebec4bf9e0187dcd44b683f848e` |
| Immutable selected-run SHA-256 | `662def0d466404ad9efa1e828ddcda372915cf9487733add0c9c881562307d2b` |
| Evaluation-protocol SHA-256 | `9d407311f31fb7ef6aa46fda98a4479c13823fd7674866b14f278fd3c036880d` |
| Frozen fitting-run state | `trained_unselected_pending_test`; historical zero-test counter; exposure is recorded in separate final sidecars |
| Joint-cohort identity | `0ab06fd0e9c17c259b8464fc43d7466f8b55cc2e2bfd8182df5ab73942a0a437`; detector's **one** final attempt completed |
| Release-artifact license | **PENDING release-manifest determination**; upstream model/code and training-source terms listed below |
| Deployment eligibility | **False**; public-data-only artifact; no approved violation types |

The launch and saved effective configurations have exactly equal parsed values; the writer's JSON serialization changes their byte hashes. They are separate file identities, not different recipes. [Configuration-equivalence receipt](../../infra/public-rf-small512-config-equivalence-20261018.json). The baseline is the pretrained backbone/localizer with an untrained target head, evaluated on development only. It is not a zero-shot COCO banknote detector. The completed fit must show finite losses, nonzero supervised gradients, actual optimizer updates, changed trainable weights, unchanged frozen parameters and the selected checkpoint hash. A memory smoke or interim epoch metric is not a completed fit result.

The [completed fit/freeze receipt](../../infra/public-rf-small512-fit-frozen-20261018.json) records **2,752 optimizer updates**, **22,016 physical batches**, **16 epochs** (final index 15), native/container exit code **0**, and unchanged immutable inputs. Fit wall time was **2,106.128 s**; the full runner elapsed time was **2,145.755 s**. Peak PyTorch allocation/reservation was **681.675 / 790 MiB**; the host observed at most **1,360 MiB** for the trainer and at least **5,816 MiB** globally free. Recorded losses were finite. Trainable-parameter SHA-256 changed from `220e18d9f871aea162b39d634d2824c33a654af602ff2171accb1ab68558682c` to `62458f6df7c9b1abb1327238bd7fa822f7569906ba3d9d12cd3684075a860e06`; frozen-parameter SHA-256 remained `aef0b941ab16545029539a2cac9f6644a0d71679e8bdba60226f4320b22d5708`.

The native training callback selected development AP50:95 **0.929118991**; the final epoch's AP50:95 was **0.900782585**. These are different states. The independent selected-checkpoint API rehearsal reproduced the best-checkpoint value; neither value is a sealed-test result. The selected checkpoint is [locally preserved](../../training/private/public_runs/20261018/rf-small512-v1-preserved/checkpoint_best_total.pth), with a separate [preservation receipt](../../infra/public-rf-small512-preservation-20261018.json).

The strict host guard retained `passed=false` because an external frontend API restart changed its protected container binding. Native fitting still exited successfully, and its immutable inputs were unchanged. The separate [postfit binding proof](../../infra/public-rf-small512-postfit-bindings-20261018.json) confirms accepted analyzer file hashes, unchanged worker/model/database/ingress bindings, readiness and an empty queue. Retain both receipts; this distinction is not a model-quality or deployment approval.

The native final protocol measures COCO bbox AP/AR with at most 500 detections per image. A separate frozen-confidence scorer uses confidence **0.5**, IoU **0.5**, same-class one-to-one matching in descending confidence order; duplicates count as false positives and unmatched labels count as misses. It reports per-class TP/FP/FN, precision/recall, total false positives and false-positive image rate on declared empty coin backgrounds. Missing, failed, duplicate or invalid result rows cannot silently disappear. The exact documented no-object output ID `2` is excluded and counted; other unknown/invalid IDs fail evaluation. [Fitting/final runner](../../training/specialists/public_train.py), [fixed scorer](../../training/specialists/public_metrics.py).

| Selected-checkpoint measurement | Development | Sealed public test |
| --- | --- | --- |
| Native COCO AP50:95 / AP50 / AR | **0.929119 / 1.000000 / 0.955842** | **0.905428 / 0.999162 / 0.932952** |
| Banknote TP / FP / FN, precision / recall at 0.5/0.5 | **287 / 0 / 4; 100% / 98.6254%** | **278 / 6 / 3; 97.8873% / 98.9324%** |
| ID-card TP / FP / FN, precision / recall at 0.5/0.5 | **16 / 0 / 0; 100% / 100%** | **13 / 0 / 3; 100% / 81.25%** |
| Coin-negative FP images / denominator / FPR | **0 / 598 / 0%** | **1 / 582 / 0.171821%** |
| Completed / failed image coverage | **905 / 0**; no API-shape errors; zero excluded background proposals | **879 / 0**; six total false positives; zero excluded background proposals |
| Native-evaluation / fixed-predict wall time and images/s | **26.2173 s / 34.5192 images/s**; **14.9622 s / 60.4860 images/s** | Separate phase throughput unavailable; **56.8079 s** whole attempt, **93.425 s** host guard |
| Actual GPU placement / peak memory | CUDA:0 parameters/buffers verified after evaluation and first predict; **310.519 MiB allocated / 342 MiB reserved**, host peak **876 MiB** | CUDA evaluation; **310.521 MiB allocated / 342 MiB reserved**, host peak **876 MiB** |

The [selected-development receipt](../../infra/public-rf-small512-selected-dev-20261018.json) binds result SHA-256 `8a6b5718c1f67a94d0bd0be1716f141b2a01f9b0b7cd928f4d33bac455f88053`, recorded zero sealed evaluations at that development phase, and has a passing host guard with unchanged production bindings. Interim native development progress belongs in the [results ledger](public-training-results-20261018.md). Keep final-epoch metrics separate from metrics of the actual selected checkpoint. The final runner's total elapsed time includes setup, native evaluation, fixed prediction and scoring; dividing 879 by that total is not single-pass detector throughput. Its GPU peaks likewise cover the complete attempt. Report these object measurements separately from whole-video API latency or bank verdict precision/recall.

The [one-attempt sealed result](../../infra/public-rf-small512-final-20261018.json) binds result SHA-256 `26d369c730456abd96ec0673ed0a8350cdf572835bc032a19f630ff5ccbe7af1` and registry claim `6754b13f4d8e6e1f4d6a7d629586964277732aaf35ec8bb72f6dac88a649ce79`. Its host guard passed with unchanged inputs/production. **At the frozen threshold, ID recall is 81.25%, below the planned ≥95% target, and six false positives miss the zero-FP target.** Native AP is a different metric and does not override these failures. The sealed attempt is consumed; these results do not authorize threshold tuning and another test on the same inventory. The artifact remains a public-data warm start with no approved banking violation types.

For reuse, verify the selected artifact hash and retain its class map, RF-DETR version, resolution and source notices. Load the local selected checkpoint through `RFDETR.from_checkpoint(..., device="cuda:0")`. Prediction IDs are already zero-based; never subtract one. Preserve only the two foreground classes, account for the documented no-object sentinel, and retain all other invalid-output failures. Camera-specific confidence calibration requires new independent domain data. No temporal/procedural claim follows from an absent box.

## Qwen3-VL-2B public crop adapter

**Intended artifact:** a compact two-class classifier for a **known, publisher-boxed crop**. The adapter has no object-search, absent-object, banking-procedure, OCR, identity or face-matching target. Its exact question is: `Identify the visible object. Answer exactly one label: single_banknote or id_card.`

| Field | Value / acceptance state |
| --- | --- |
| Official base model | `Qwen/Qwen3-VL-2B-Instruct`; `Qwen3VLForConditionalGeneration` |
| Base revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Base `model.safetensors` SHA-256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| Quantization / compute | Language linear layers use NF4 with double quantization and BF16 compute; vision tower and embeddings stay frozen BF16; SDPA |
| Trainable weights | Language attention Q/K/V/O LoRA only; no vision/backbone update; frozen tensor hashes must remain unchanged |
| Completed recipe | **80 optimizer steps**, LR `1e-4`, LoRA rank 4 / alpha 8, batch 1, accumulation 4, gradient checkpointing; 1,605,632 trainable adapter parameters |
| Processor/text limits | Minimum image pixels 1,024; maximum 65,536; maximum sequence length 256 |
| Crop labels | Exact publisher object class and tight enclosing box; no procedural, absent-object or local-human-review labels invented |
| Crop selection | Seeded metadata-only ordering; caps train 256/class, development 16/class, sealed 32/class; actual available ID support remains smaller than the train/test caps |
| Training support / draw plan | 256 note / 36 ID crops eligible under the cap; 160 draws per class, covering 160 note / 36 ID crops with repetition; optimizer proof and support counts are recorded separately |
| Runtime pins | Torch `2.13.0+cu129`, Transformers `5.17.0`, PEFT `0.21.2`, Accelerate `1.15.0`, bitsandbytes `0.50.2` |
| Predeclared recipe-file SHA-256 | `1f63fac83b0641f3d4f9284d296e10349c8ccad99a8922924ebfa04ba1f42cd1` |
| Actual fit-config SHA-256 | `d029457071b32e13fb71a0563fdbd1e40f124dfbeeea6a03ccbd960e34625f64`; binds successful smoke run `6022b4f5975fc94cb7e4cc1315df28542941f5fe644ad3ce3793efb7a7f3f1f6` |
| Actual crop-runner SHA-256 | `43a9153abd54d49fef7720a33ed0b3bedbbe65b5fc137c9c6bfdf6f9cf6b93a6` |
| Snapshot-manifest SHA-256 | `e7ed200b59c15fc3abb9935b7f765e2258e7e79c9b85a3e642c107c5ad85fee5` |
| Runtime image SHA-256 | `2be4db9629837c6a3c0e252d2a4b45b54bd0a0d25d41e04274dbff0dd2755712` |
| Saved adapter file | [adapter_model.safetensors](../../training/private/public_runs/20261018/crop2b-fit-v1-preserved/adapter/adapter_model.safetensors), **6,455,336 bytes** |
| Adapter `adapter_model.safetensors` SHA-256 | `a3733ccf3c888224c2887dd24900e45d2462e9481299ed71cec82518cfc6c7ad` |
| Adapter config / README SHA-256 | `0e103b394246ce25605074baf140ee4423ad3aa615fa8c10f477f7d2c9c8d750` / `05f1fe7129dcdad43c614fe061d385c4818ef1da793c11dd3ca7718b5a25e92a` |
| Immutable selected-run SHA-256 | `aaec63508133bf227170be43e10f426d6804cd09c804fdc58b4e913e33e0ffcd` |
| Evaluation-protocol SHA-256 | `b5be8845c6bd69e3033d953c80e3270a1d08064d9ceaf4f10b245e895a693b12` |
| Training state / joint cohort | `trained_unselected`; no sealed use for training/selection; cohort `0ab06fd0e9c17c259b8464fc43d7466f8b55cc2e2bfd8182df5ab73942a0a437`; crop participant's **one** final attempt completed |
| Fresh adapter cold-load proof | **Passed CPU BF16-base binding**: all 224 adapter tensors/1,605,632 parameters match exactly; the separate final evaluation exercised fresh NF4/GPU generation, without a BF16/NF4 equivalence claim |
| Release-artifact license | **PENDING release-manifest determination**; upstream/source terms below |
| Deployment eligibility | **False**; no approved banking violation types |

The frozen training/evaluation curriculum uses the original crop and Lanczos downsampling to short sides **64, 32 and 16 pixels**. The degradation step never enlarges a smaller original crop. The normal Qwen processor then resizes within its prescribed pixel/grid limits; this is a resolution stress experiment, not detail recovery. All four views are correlated derivatives of the same source objects and do not quadruple independent support.

Greedy generation accepts only an exact class name after whitespace stripping; malformed/unknown answers count as errors. Each condition reports pooled and per-class accuracy. The primary score averages the two class accuracies equally, then averages the four conditions equally. The base-versus-adapter comparison uses development only; the joint sealed test evaluates the single frozen adapter. Coin backgrounds and unboxed scenes are outside this crop classifier's label space. Positive crop accuracy supplies no clean-room or bank-video false-alarm rate. [Actual crop runner](../../training/public_crop_lora.py), [crop protocol and loading recipe](../../training/public-crop-lora.md).

The [completed compact fit receipt](../../infra/public-crop2b-fit-frozen-20261018.json) records **80** finite optimizer-step proofs, changed adapter tensors and unchanged frozen tensor hash `2d4a86b4865b6e080cd87cc64ebc25a78101e0ba693b286309d8dc184a4ce182`. Adapter tensor hash changed from `6faff90345e6d7df8e167854b481081edcf8551f280a83f810fe80aa66963252` to `536d4e637deaf367491c0c6a137aba77adc2ecf4313066839057f9ab495c4850`; these tensor hashes are distinct from the saved safetensors file hash above. Fit wall time was **112.542 s**, complete command time **197.716 s**, peak allocation/reservation **3,636.423 / 3,990 MiB**, host-observed peak **4,558 MiB**. The host guard passed with unchanged immutable inputs/production and recorded zero sealed evaluations at fitting completion. The two failed initial smoke attempts and qualified third smoke are retained in the [results ledger](public-training-results-20261018.md); a smoke is not counted as the completed 80-step fit.

The separate [CPU cold-load evidence](../../infra/public-crop2b-cold-load-20261018.json) binds successful receipt SHA-256 `b082af40f037825c7a94f0d52bbca0364694edac1c5092ff6c154531c5f8cac3`. A fresh official BF16 CPU base plus the saved adapter reproduced all 224 tensor keys, shapes, F32 dtypes and values exactly in **30.468 s**, with unchanged inputs and CUDA uninitialized. No dataset, processor, inference or sealed examples were used. This validates adapter binding/restoration; it does not establish NF4/GPU numerical equivalence or accuracy. Prior operator/helper compatibility failures remain preserved separately; the fitted source and artifacts were unchanged.

The [compact preservation receipt](../../infra/public-crop2b-preservation-20261018.json) verifies the local adapter weights/config and seven sanitized metadata receipts; remote originals remain retained. It includes no source images or datasets. Preservation and CPU restoration supply artifact integrity evidence, while the one-attempt GPU evaluation supplies the actual final task measurement.

| Frozen adapter measurement | Development base | Development adapter | Sealed adapter |
| --- | --- | --- | --- |
| Original crop: note / ID / macro accuracy | **100% / 100% / 100%** | **100% / 100% / 100%** | **100% / 100% / 100%** |
| 64-px short side: note / ID / macro accuracy | **93.75% / 100% / 96.875%** | **93.75% / 100% / 96.875%** | **100% / 100% / 100%** |
| 32-px short side: note / ID / macro accuracy | **81.25% / 93.75% / 87.5%** | **100% / 93.75% / 96.875%** | **96.875% / 75% / 85.9375%** |
| 16-px short side: note / ID / macro accuracy | **81.25% / 87.5% / 84.375%** | **100% / 81.25% / 90.625%** | **96.875% / 87.5% / 92.1875%** |
| Equal-class/equal-condition primary score | **92.1875%** | **96.09375%** | **94.53125%** |
| Pooled correct / predictions across four conditions | **118 / 128 (92.1875%)** | **123 / 128 (96.09375%)** | **184 / 192 (95.8333%)** |
| Unique crops / source groups; invalid outputs | **32 / 32; 0** | **32 / 32; 0** | **48 / 48; 0** |
| Crop preprocessing + generation throughput, original / 64 / 32 / 16 | **3.979 / 5.148 / 5.441 / 5.381 crops/s** | **4.119 / 3.945 / 3.922 / 3.888 crops/s** | **3.623 / 4.177 / 4.082 / 4.129 crops/s** |

Each development condition uses **16 note + 16 ID crops**. The four conditions produce 128 correlated predictions, not 128 independent objects. The aggregate development gain coexists with a **16-px ID regression from 14/16 to 13/16**. Preserve that class-specific failure rather than interpreting the higher macro score as improvement for every small object. Timing includes each single-crop preprocessing/generation pass; answer lengths and errors differ, so these figures do not isolate adapter compute overhead or establish whole-video speed. Development baseline/adapter report hashes are `5d24709b966656091ffc114e710b649ea43d53d3cd8e3ff28e1684fc27495342` / `ff2b3ff3e4c2f8252ce5a2ded02652107bd74504597fcf57039f5beb538287f3`.

The [one-attempt compact sealed result](../../infra/public-crop2b-final-20261018.json) binds result SHA-256 `d2b71ebb677909913ddaa7433a742983925230f77e84fd3d087879cfe08f0b2a` and registry claim `4487d1bd3867bfb19028ce29ca73a90d2e5f324b18db06c9455324cb62b7eb32`. Each condition uses the same **32 note + 16 ID crops**: correct counts are **48/48, 48/48, 43/48, 45/48** for original/64/32/16. There are 48 unique originals and 192 correlated predictions, with no invalid outputs. **The predeclared primary score is 94.53125%, below 95%.** The higher pooled 95.8333% gives the larger banknote class twice the ID weight and must not replace the frozen equal-class score. No base-model held-out comparison was run, so this result establishes no held-out adaptation gain. The host guard passed in **116.193 s**, peak observed GPU use **2,704 MiB**, with unchanged immutable inputs/production. Condition timings are preprocessing-plus-generation passes; the full guard also includes model loading and checks. This consumes the crop participant's only attempt and supplies no bank-video false-alarm or latency claim.

Reuse requires the exact base revision/snapshot, processor, prompt, crop construction, quantization settings and PEFT adapter inventory from the completed run. Keep the visual encoder frozen and distinguish the native crop from each stress condition. A different crop finder, novel document design, background, occlusion or bank camera changes the task and needs separate validation.

## Licenses, provenance and release contents

| Component / data | Verified upstream terms and retained evidence |
| --- | --- |
| RF-DETR 1.11.2 Small code/model family | [Apache-2.0 source license](https://github.com/roboflow/rf-detr/blob/1.11.2/LICENSE); [official weight provenance](../../infra/public-training-runtime-20261018/model-weights-provenance.json) retains the Small source URL and hash. Other RF-DETR families are outside this card. |
| Qwen3-VL-2B-Instruct | [Pinned official model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md) declares Apache-2.0; [local snapshot inventory](../../infra/public-training-runtime-20261018/qwen2b-snapshot.json) verifies all files. |
| KZ-BD v1 | [Author Mendeley deposit](https://data.mendeley.com/datasets/dny3dgvvw8/1), CC BY 4.0; source archive SHA-256 `daae1a4eb371a7b6ff672e0d7c83edbe27f85ff02a584bb9ea42ab3856eec392`; outer-box derivation and quarantine retained. |
| MIDV-Holo | [Author repository](https://github.com/SmartEngines/midv-holo), CC BY-SA 2.5; retained author/Generated Photos attribution; license-file SHA-256 `3b6bc8eb8a4a6bba41619051803991d9de949f7f722eee30bc0954c7b38aed7d`. Source identities are author-declared fictional. |

The release package should contain the selected checkpoint or adapter, exact file hashes, config, class map, immutable run and evaluation protocol, dataset/source-lock identifiers, metrics with coverage/failure counts, attempt/cohort references, notices and this model card. Raw source media are not required in a weights-only package; its receipts must accurately state which source bytes are retained separately. Any chosen license for the resulting checkpoint/adapter must be explicit in that release manifest, together with applicable attribution/source terms.

Canonical provenance and updates: [dataset research](public-dataset-research-20261018.md), [runtime record](public-training-runtime-20261018.md), [measured-results ledger](public-training-results-20261018.md), [public-pretraining workflow](../../training/specialists/PUBLIC-PRETRAINING.md), [domain promotion gates](../../training/specialists/README.md). The recorded hashes and metrics identify these exact frozen attempts. Further tuning requires a fresh independent evaluation inventory; preserve these outcomes and all scope, contamination, grouping and deployment limits. Only release-manifest license determination remains pending in these cards, separately from completed experimental measurement.
