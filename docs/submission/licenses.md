# Component license register

Status: metadata/provenance inventory in progress; this is not a declaration that every distribution artifact has cleared redistribution review. Preserve upstream license/notice files in the release image. Source and bundled native libraries can have separate obligations.

## Application dependencies

Direct dependency metadata was read from public PyPI JSON on 17 October 2026. Full installed CPU dependency metadata is recorded in `installed-dependencies.json` with version and license-file fields.

| Component | License metadata | Release action |
|---|---|---|
| FastAPI, Pydantic, SQLAlchemy, jsonschema, Hatchling, pytest | MIT | Preserve upstream notices |
| Uvicorn, PyAV | BSD-3-Clause | Preserve copyright, conditions, and disclaimer |
| python-multipart, pytest-asyncio | Apache-2.0 | Preserve license and applicable notices |
| Pillow | MIT-CMU | Preserve license notices, including bundled libraries |
| HTTPX | BSD-3-Clause (upstream license) | Preserve upstream license |
| openpyxl | MIT (upstream license) | Development/audit only; excluded by production `--no-dev` |
| psycopg | LGPL-3.0-only | Preserve LGPL terms and source/relinking obligations applicable to distribution |
| psycopg-binary | LGPL-3.0-only plus bundled native components | Inventory bundled libpq/OpenSSL notices and required terms in the selected artifact |
| FFmpeg / FFprobe / PyAV bundled FFmpeg | Exact build/configuration dependent | Inspect the deployed binary build flags and licenses; GPL-enabled wheels/binaries need corresponding redistribution compliance. Do not infer LGPL-only from the PyAV wrapper license. |

Primary package metadata: [PyPI](https://pypi.org/). Upstream native references: [FFmpeg licensing](https://ffmpeg.org/legal.html), [Psycopg 3 license](https://github.com/psycopg/psycopg/blob/master/LICENSE.txt), [PyAV binary-wheel documentation](https://pyav.basswood.io/docs/stable/overview/installation.html).

Actual deployed CPU build provenance is recorded in `infra/cpu-license-provenance-20261017.json`: FFmpeg package `7:7.1.5-0+deb13u1`, its build/license banner, FFprobe, OpenSSL, psycopg-binary's libpq metadata and license-file hashes. This FFmpeg build includes `--enable-gpl` and excludes `--enable-nonfree`; upstream describes the GPL effect of enabled optional components in its [license notes](https://ffmpeg.org/legal.html). The report identifies image `sha256:97fd2dd3a5dafd63309fab12868925a8c6c39b363703ae0a5809ad6ff28afdd9`. It inventories observed metadata rather than clearing redistribution of the image or every bundled native library. The source ZIP excludes native binaries and weights.

## Model and GPU environment

### Local model evaluation register, 18 October 2026

Official model-card/API metadata was checked for the session's local-only evaluation; Qwen3.8 and
vLLM also provide the linked license files. Gemma4, Qwen3-VL-32B-FP8 and Qwen3-VL-30B-A3B have no
standalone LICENSE/NOTICE files in their checked repository listings. The revisions below identify upstream repositories; they do **not** assert that the
running production weight bytes or downloaded test snapshots match them without a separate
artifact receipt. Preserve the Apache license and applicable notices. No cloud inference API is
used by this evaluation.

| Component | Verified upstream license | Upstream revision / scope |
|---|---|---|
| [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | Apache-2.0 ([pinned model card](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/README.md), [Google Apache terms](https://ai.google.dev/gemma/apache_2)) | `842da3794eaa0b77d5f08bae87a17459d91ff475`; local candidate, not gated; architecture `Gemma4ForConditionalGeneration` |
| [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/LICENSE) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`; local candidate, not gated; architecture `Qwen3_5ForConditionalGeneration` |
| [Qwen/Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) | Apache-2.0 ([pinned model-card metadata](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8/blob/4bf2c2f39c37c0fede78bede4056e1f18cdf8109/README.md)) | `4bf2c2f39c37c0fede78bede4056e1f18cdf8109`; local evaluation checkpoint, not selected in production |
| [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | Apache-2.0 ([pinned model-card metadata](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c/README.md)) | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`; upstream revision checked; existing production model remains selected, weight-byte digest unrecorded |
| [vLLM v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) | [Apache-2.0](https://github.com/vllm-project/vllm/blob/v0.31.0/LICENSE) | Tag commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`; server code; CUDA image/native dependencies retain their own terms |

Source metadata: official Hugging Face model APIs for
[Gemma](https://huggingface.co/api/models/google/gemma-4-31B-it),
[Qwen3.8](https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8),
[Qwen3-VL-32B](https://huggingface.co/api/models/Qwen/Qwen3-VL-32B-Instruct-FP8) and
[Qwen3-VL-30B](https://huggingface.co/api/models/Qwen/Qwen3-VL-30B-A3B-Instruct).
`Qwen3.8-Flash-Next` is excluded from this session: its custom Qwen Community License 1.0 is
outside the approved Apache/MIT/BSD model choices. Listing a license or candidate is not a
quality result or approval to deploy it.

A07's `models/manifest.json` describes the historical [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) reference, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, reported Apache-2.0. Its 15 downloaded model/processor artifacts (17,545,914,364 bytes) passed verification and an independent offline recheck; `model-artifact-verification.json` records that manifest digest. The recorded 18 October baseline and isolated accuracy-upgrade stage instead use [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct), whose upstream card declares Apache-2.0. The fresh baseline binds its running image/process but does not hash its weight bytes; the historical 8B verification is not 30B artifact proof.

The actual running model image's **269 installed distributions** are recorded in `gpu-dependency-licenses.json`, collected through read-only standard-library package metadata on Python 3.12.3. This is separate from the 39-package CPU inventory. It records version, license fields and license-file metadata; legacy license text is capped at 300 characters, file bodies were not opened, and missing fields remain unknown. Native OS/CUDA/driver/image notices and redistribution obligations still require a separate inventory. Metadata is not clearance.

The optional training environment (`torch`, `torchvision`, `transformers`, `peft`, `accelerate`) has its own pinned image/lock and must preserve inherited and added-package notices. Actual W1 training is documented separately in `qa/integration/actual-w1-aggregate.json`; this license inventory itself is not training or quality proof.

The optional SigLIP2 image encoders have separate pinned manifests under `configs/serving/encoders/`. They record Apache-2.0 metadata and exact README/license artifact hashes. Both four-artifact snapshots were downloaded and independently verified (eight artifacts / 3,008,466,624 bytes), as recorded in `staging-validation.json`. Preserve their upstream notices in the private runtime. The source ZIP contains manifests and generic code, not encoder weights. Staging and feature extraction do not imply production selection or redistribution clearance. W1 trained only the custom temporal head with the encoder frozen.

## Accuracy-upgrade qualification, 18 October 2026

The [isolated stage record](../integration/accuracy-upgrade-stage-20261018.md), [sanitized stage report](../../benchmarks/quality/accuracy-upgrade-stage-20261018.json) and [candidate comparison](../../benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json) bind the qualified source and inference profile to these identities:

| Artifact | Recorded identity |
|---|---|
| Isolated stage | `forte-accuracy-v2-20261018-223127` |
| Existing CPU API/worker base | `sha256:30691bd4230396c011f292beab7094bcdf160e7845c417e85bf8d11e8dcbdf32` |
| Candidate source archive, 13 files | `dcb5f16574df668aaaf6f6011cb0f13f70f0056a0c20027faa8777584dc21ce3` |
| Reviewed `fast_config.json` bytes | `993d2f22867576be053d7a05663043077937d5cd749d67709dcab0d52016789c` |
| Effective config, sorted compact JSON | `113323095c2031430f0aff58f3aecc4fe28907dfbe84dfb0eea1d9545a7a7b29` |
| Worker audit config, sorted JSON with spaces | `0bc38acdb31659d0ee525b3c702d2516317e49420b0318237536c2fbccc3a8e2` |
| Existing `q30` model image | `sha256:046137dc151b7741b52fbb438de7d606fa74694969c7bb4ab6642d9408b9516e` |
| Existing `vlm30` model container | `d8dd1da4a4a811ccac2dc3f8b0ff34aeafe433dd94dd5926d3bdc7b6929e0691`, started `2026-10-17T19:14:49.608401273Z` |

The two semantic config hashes differ only in serialization whitespace. This profile enables `require_corroboration` and `use_verify`, keeps `specialists.enabled=false`, and retains the existing six disabled violation types. The stage reused the existing `q30` model process without creating, restarting or reconfiguring its container. No new trained detector, hand-pose checkpoint, tracked-object Transformer head or LoRA adapter was selected by this qualification. The worker probe found the specialist runtime absent and neither `torch` nor `rfdetr` imported. The source archive includes optional specialist code; inclusion of code does not select its runtime or weights.

The recorded 34/40 development-set result and passing six-request internal stage concern the source/profile above. They provide no weight-byte digest or additional license clearance. The historical installed CPU/GPU inventories remain separate records; they do not establish that optional RF-DETR, RTMPose, temporal-training or crop-LoRA environments were installed or used in this stage. This section records qualification evidence, not the status of a subsequent production rollout. Native runtime notices and exact model-weight redistribution review remain pending.

## Accepted accuracy-upgrade rollout, 18 October 2026

The [final rollout report](../../benchmarks/quality/accuracy-upgrade-rollout-20261018.json) records accepted live API/worker overlay image `sha256:8c8b92550c5dfc6702efbe266773ea4bfd86639b9f5edd8e86e6a0e3e45b6a3d`, built over the CPU base above from the same qualified source archive and reviewed profile. Imported module hashes and effective config were checked in both API and worker before and after the public acceptance run. The existing `q30` model image, container, start time and serving command remained unchanged. No new model weights or adapters were deployed; specialists and the six disabled types remain unselected.

The six longest supplied development clips completed through the existing public HTTPS API on their first attempt: **6/6 passed**, with a maximum request time of **161.068215 seconds**, below the 240-second gate. This verifies public transport, queue, worker, response schema and evidence handling for those six requests; it is not a 40-video API accuracy evaluation. Neither the overlay identity nor this acceptance run clears inherited native-runtime redistribution obligations or supplies a missing model-weight digest. The historical inventories and pending legal reviews above still apply.

## Completed public-data candidates

The later public fits use official RF-DETR Small weights/code (Apache-2.0) and
Qwen3-VL-2B-Instruct (Apache-2.0), with exact revisions, received hashes and
runtime notices in the [model cards](../integration/public-model-cards-20261018.md).
Their data are author-deposited [KZ-BD v1](https://data.mendeley.com/datasets/dny3dgvvw8/1)
(CC BY 4.0) and [MIDV-Holo](https://github.com/SmartEngines/midv-holo)
(CC BY-SA 2.5 with retained Generated Photos attribution). Derived data retain
source terms, attribution and modification notices; code/initial-model licenses
do not silently relicense source data or settle the resulting adapter/checkpoint
release license. That artifact-license determination remains explicit/pending
where the cards say so.

Both frozen sealed candidates failed quality targets and are [staged disabled](../../infra/public-trained-candidates-release-verification-20261018.json).
Staging neither authorizes serving nor adds redistribution/license approval.
The [fresh six-upload operational check](../../benchmarks/quality/final-public-six-load-20261018T005948Z.json)
passed on the unchanged accepted analyzer; it is separate from public-model
accuracy and rights review.

## Assets and original work

The optional accuracy-upgrade tooling pins RF-DETR Small 1.11.2 (Apache-2.0) and `supervision` 0.29.1 (MIT). It excludes RF-DETR Plus checkpoints with a different license. Optional MMPose/RTMPose code is Apache-2.0; exact hand/detector weights need their own provenance and dataset-term review. The optional tracked-object Transformer and crop-LoRA recipes require separate reviewed data, checkpoint/adapter provenance and release acceptance; they remain unselected in the qualified profile above. SAM3 remains unselected under Meta's custom license. ONNX Runtime is MIT; the full TensorRT runtime has NVIDIA SDK terms and is not introduced by this qualification. Primary references and artifact-specific limits are recorded in [the accuracy upgrade dependency register](../accuracy-upgrade.md#dependency-decisions). No root project `LICENSE` was identified during that audit; no PyTorch or other dependency license is claimed to license Forte's original source automatically.

A19 owns source provenance, logo/font rights, and local asset inventory. Official public availability alone does not grant redistribution rights. The release must use permitted packaged assets or the documented text treatment/fallback.

The project builds case-specific contracts, durable queueing, visual event/rule logic, evidence assembly, API, review UI, and evaluation tooling. No license is invented for user-provided source, organizer media, or the bank's identity assets. Organizer videos and labels are excluded from public distribution.

## Public-object adapter dependencies, 18 October 2026

The cancelled Gemma preparation and the current Qwen30 adaptation attempt use the pinned components and public-source
attributions in [the training license register](../integration/gemma4-training-licenses.md).
Transformers5.17.0, PEFT0.21.2 and Accelerate1.15.0 are Apache-2.0; bitsandbytes0.50.2 is MIT;
PyTorch2.13.0/torchvision0.28.0 use BSD-style licenses. The train dataset is public KZ-BD (CC BY4.0)
and MIDV-Holo (CC BY-SA2.5 with preserved Generated Photos attribution). Adapter artifact licensing
remains explicit/pending; the base model license does not relicense source data or an adapter.
No benchmark video labels are used to train or choose this adapter. No production deployment is
implied by recording dependencies. Preserve all notices in the private training bundle/runtime.

The user redirected the experiment to **Qwen3-VL-30B-A3B-Instruct BF16 LoRA**, with frozen base
and vision weights. The base model is Apache-2.0 at the revision in the model table above; both
test-host downloads have separate weight receipts:
[main H100](../../benchmarks/quality/q30-main-download-20261018.json),
[second H100](../../benchmarks/quality/q30-training-download-20261018.json).
Each records all 13 shards / 62,141,618,016 bytes verified against upstream LFS at
`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`. These test-host receipts do not attest production's
existing weight bytes. The [Qwen30 runtime receipt](../../benchmarks/quality/qwen30-training-runtime-20261018.json)
reuses the pinned training packages and confirms native `Qwen3VLMoeForConditionalGeneration`
support on CPU; package presence, including bitsandbytes, does not imply NF4 training.
The current attempt keeps the base/MoE weights in BF16 and trains only the desk adapter's supported
cash/document boolean targets. Gemma training was cancelled before GPU optimizer steps.
The data attribution and pending adapter-distribution review above still apply to this Qwen adapter;
no training outcome, bank-video accuracy gain or production clearance follows from this license entry.

The separately recorded [100-step training completion](../../benchmarks/quality/qwen30-training-summary-20261018.json)
created an internal adapter while preserving frozen base hashes. Completion does not assign the
adapter a distribution license or change the source-data obligations above. Quality qualification
and production selection remain separate and pending.
