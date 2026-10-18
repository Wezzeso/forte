# Gemma 4 public-object training license record

Verified 18 October 2026. Scope: an internal, local Gemma 4 31B adapter experiment using the existing public banknote/ID-card training split. This record is a dependency and source-data license check, not a claim that an adapter has trained, improved bank-video accuracy, or qualified for production.

**Decision:** the named upstream code, weights and public-data copyright licenses permit the proposed internal experiment, including commercial use. Keep the source attribution/provenance with the training bundle. The resulting adapter's distribution license remains unassigned pending an artifact-specific decision; neither a public release nor production selection follows from this check. No additional copyright permission from the dataset authors is required for the licensed internal reuse described here. [CC guidance on AI reuse](https://creativecommons.org/faq/#what-are-the-limits-on-how-cc-licensed-works-can-be-used-in-the-development-of-new-technologies-such-as-training-of-artificial-intelligence-software), [CC guidance on private reuse and ShareAlike](https://creativecommons.org/faq/).

## Code and base weights

The proposed foundation is `vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb`, as recorded in [the existing runtime lock](../../training/runtime-lock.json). The runtime owner supplied the package choices below; upstream license files were checked at the listed tags. Actual installation versions, image identity and received wheel/model hashes belong in the run receipt.

| Component | Proposed revision/version | Verified upstream terms |
| --- | --- | --- |
| Gemma 4 31B-it weights and processor | `google/gemma-4-31B-it`, `842da3794eaa0b77d5f08bae87a17459d91ff475` | Apache-2.0 in the [pinned official model card](https://huggingface.co/google/gemma-4-31B-it/raw/842da3794eaa0b77d5f08bae87a17459d91ff475/README.md), linking to [Google's Apache terms](https://ai.google.dev/gemma/apache_2). This is Gemma **4**; do not substitute older Gemma license terms. |
| Transformers | `5.17.0` | [Apache-2.0](https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/LICENSE) |
| PEFT | `0.21.2` | [Apache-2.0](https://raw.githubusercontent.com/huggingface/peft/v0.21.2/LICENSE) |
| Accelerate | `1.15.0` | [Apache-2.0](https://raw.githubusercontent.com/huggingface/accelerate/v1.15.0/LICENSE) |
| bitsandbytes | `0.50.2` | [MIT](https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.50.2/LICENSE); proposed Linux wheel SHA-256 `55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283`, reported by runtime owner, to be verified upon receipt |
| PyTorch | `2.13.0+cu129` | [BSD-style upstream license and retained historical copyrights](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/LICENSE); wheel/native dependencies retain their own notices |
| torchvision | `0.28.0+cu129` | [BSD-3-Clause](https://raw.githubusercontent.com/pytorch/vision/v0.28.0/LICENSE) |
| vLLM | `0.31.0` | [Apache-2.0](https://github.com/vllm-project/vllm/blob/v0.31.0/LICENSE); inherited image/CUDA dependencies are separate |

Apache distributions retain the license, relevant notices, and modification notices. MIT distributions retain their copyright and permission notice. BSD distributions retain copyright, conditions and disclaimer in the required source/binary materials; the torchvision license also excludes endorsement. The training image's CUDA/driver, bundled native components and other inherited dependencies are not relicensed by these package licenses. Existing inherited-runtime inventory and unresolved redistribution review remain in [the main register](../submission/licenses.md).

## Frozen public inputs

The data owner selected publisher-annotated object crops from the existing combined dataset: 540 banknote / 36 ID-card training crops and 64 banknote / 16 ID-card validation crops, subject to the actual export receipt. This record does not authorize opening a sealed split or recycling the earlier sealed results as fresh independent evidence.

Combined source: `training/private/public_datasets/combined-banknote-id-v1/dataset-lock.json`, SHA-256 `eef3f1ad5fabb9cb720e42522265a6946a02196af01221af6ebf0522bf498fb8`; recorded dataset digest `2439037ef673dbb5ab47dc0fab15575e687e4da8243d84d29261e6e44ac745ec`. The lock's source license metadata and attribution were inspected as text; no image was opened.

| Source | Publisher grant and retained provenance |
| --- | --- |
| KZ-BD v1 | The [author Mendeley deposit](https://data.mendeley.com/datasets/dny3dgvvw8/1) states **CC BY 4.0**, DOI `10.17632/dny3dgvvw8.1`, authors Ualikhan Sadyk and Akgul Bozshina. Source lock SHA-256 `f78bf03f51347d98081302430fe8552b9ac0c89ce014b6fbcb31a4f4d751d24b`; source archive SHA-256 `daae1a4eb371a7b6ff672e0d7c83edbe27f85ff02a584bb9ea42ab3856eec392`. |
| MIDV-Holo | The [author repository](https://github.com/SmartEngines/midv-holo/blob/main/README.md) states **CC BY-SA 2.5** and fictional Utopia document data. Source lock SHA-256 `60a25b8a8c34847be8bf7634bb0845dc8020ef76044ef0ea9fc142a96f91fc9f`; retained license-file SHA-256 `3b6bc8eb8a4a6bba41619051803991d9de949f7f722eee30bc0954c7b38aed7d`. Only selected scene members were acquired; the whole `images.tar` digest is unknown. |

For KZ-BD, retain author/title/DOI, license link, existing source notices and the actual changes: annotation selection/quarantine, cropping, resizing, metadata removal and training-specific augmentation. Do not imply author endorsement. Commercial copying and adaptation are allowed under the [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en), with the applicable attribution and modification requirements when sharing.

For MIDV-Holo, retain the authors, title, DOI `10.1007/978-3-031-41682-8_30`, source link, CC BY-SA 2.5 license and modification details. The upstream authors expressly encourage **Generated Photos** attribution for their generated faces; retain `Images of faces: Generated Photos (https://generated.photos/)` in the bundle. This is reliance on the dataset publisher's grant for the included fictional documents; it does not grant rights to download unrelated Generated Photos assets. No OCR, identity or face-recognition target is introduced.

Commercial reproduction/adaptation is permitted by the [CC BY-SA 2.5 legal code](https://creativecommons.org/licenses/by-sa/2.5/legalcode.en). If modified MIDV-Holo images/crops are shared or publicly performed, follow its attribution and ShareAlike terms, including an allowed same-or-later license and no incompatible additional restrictions. Keep per-source licenses in a mixed collection. The fact that the base weights are Apache-2.0 does not change those image terms.

## Adapter handling

Bundle the existing `ATTRIBUTION.md`, source-lock identifiers, retained source-license/readme evidence, exact transformation recipe, runtime notices, base revision, adapter file hashes and run configuration with the internal artifact. The data owner confirmed that source attribution and provenance will accompany the export. Keep organizer videos and labels out of this public-object training dataset; their benchmark use is separate.

Do not assert that every trained parameter is automatically a copyright adaptation of every input, or that ShareAlike can never affect weights. CC distinguishes protected adaptations and public sharing from private reuse; its [AI training guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) explains how attribution/ShareAlike apply when their conditions are triggered. Before sharing a trained adapter or merged checkpoint, determine and record the actual artifact license and applicable source obligations. Until then retain it as an internal experiment, preserving all source attribution. This is compatible with proceeding with the authorized local training attempt now.

Historical public-data provenance and failed candidates remain documented in [the public model cards](public-model-cards-20261018.md); this experiment does not overwrite those measurements.
