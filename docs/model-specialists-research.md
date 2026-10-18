# Local models for bank-video observations

Research frozen on 17 October 2026. This is a comparison proposal, not a model
selection or deployment. No candidate weights were downloaded, no gated terms
were accepted, and no private videos were used. The current Qwen teacher, W1
experiment, rental budget, and stop guard remain unchanged. Exact public file
identities, sizes, source links, and explicit unknowns are recorded in
[`specialist-candidates.json`](../configs/serving/specialist-candidates.json).

The first practical comparisons should be **Cosmos-Embed1 anomaly embeddings**
and **a small VideoMAE V2 temporal encoder** against the existing SigLIP feature
baseline. **Holmes-VAU-2B** is a useful separate explanation comparator. None has
demonstrated the bank's cash-handling rules, legal compliance, or perfect accuracy.

| Candidate | Useful output | Published training scope | Main constraint |
|---|---|---|---|
| Cosmos-Embed1-448p-anomaly-detection | Video/text embeddings and anomaly similarity scores | 24 generic anomaly categories plus normal chunks | No generated explanations; NVIDIA Open Model License |
| VideoMAE V2 distilled ViT-B K710 | Compact temporal action features | General Kinetics actions | Needs bank-specific heads and interval supervision |
| Holmes-VAU-2B + ATS | Surveillance anomaly descriptions and temporal focus | HIVAU-70k on UCF-Crime/XD-Violence | Older custom runtime; sampler may miss subtle bank actions |
| CamVLM-8B | Surveillance reasoning and viewpoint actions | CCTV-Anomaly and CamTrack-53K | Gated contact-sharing; prerecorded video cannot gain unseen evidence |
| V-JEPA 2.1 B/L | Dense temporal features | Self-supervised video; downstream action/interaction tasks | Requires trained heads; official CDN hashes not yet frozen |

## First comparison: Cosmos-Embed1 anomaly variant

The [official anomaly model card](https://huggingface.co/nvidia/Cosmos-Embed1-448p-anomaly-detection)
describes a roughly 1B-parameter EVA/Q-Former embedder, optimized for 8 frames at
448 pixels and producing a normalized 768-dimensional feature. Its anomaly
fine-tune uses 5-second Vad-Reasoning chunks. Categories include stealing and
robbery, alongside traffic, fire, weather and other anomalies; they do not encode
cashier procedure rules. The card reports **46.44% top-1 hit rate and 38.94% macro
F1 on its generic anomaly test**, which are not bank accuracy measurements.

Pin `8f4a7e904f07ccb7848a7cf2bdcb71b0db3ba3d3`: ten weight shards total
**4,792,128,248 bytes**, with **905,619 bytes** of other repository assets.
Stored tensors include FP32; BF16 inference on A100/Ampere is publisher-tested.
Use an isolated reviewed local PyTorch/Transformers or separately validated ONNX
runtime. The custom source must be pinned and reviewed before execution. The
card's example points to the base `448p` ID; a comparison must load the exact
**anomaly-detection** checkpoint instead. [NVIDIA integration documentation](https://docs.nvidia.com/vss/latest/models/cosmos-embed1.html)
describes its embedding role. Governing terms are the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
with additional Apache/MIT components; this is not an Apache-only checkpoint.

## First comparison: VideoMAE V2 temporal features

The [official model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/docs/MODEL_ZOO.md)
releases a distilled ViT-B trained on Kinetics-710 with 16-frame input. The
[official Kinetics-400 label list](https://github.com/google-deepmind/kinetics-i3d/blob/master/data/label_map.txt)
includes **counting money**. This is a useful transfer lead, not a claim that the
model counts notes, tracks denominations, or verifies a bank procedure.

Use `OpenGVLab/VideoMAE2` revision
`706cc172d65ebd4dedbee3f9c0183a93df9fa125`, artifact
`distill/vit_b_k710_dl_from_giant.pth`: **173,574,417 bytes**, SHA256
`8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec`.
The [official weight repository](https://huggingface.co/OpenGVLab/VideoMAE2)
declares Apache-2.0; implementation code is MIT. Use trusted-source safe loading
or conversion for the pickle-format checkpoint, in an isolated local runtime.
Its small weight footprint makes this a practical action-feature experiment;
bank latency, activation memory and quality remain unmeasured. Published
multi-view action scores do not describe a single-window bank deployment.

## Explanation comparator: Holmes-VAU-2B

[Holmes-VAU](https://github.com/pipixin321/HolmesVAU/tree/fdef61ec58e8cfa747ff5f40d7e89e9bd6f9e90d)
is the newer surveillance-specific successor to Holmes-VAD. It combines an
anomaly-focused temporal sampler with hierarchical clip/event/video instruction
tuning. The publisher's example selects 12 frames and claims less than 9GB on one
GPU; this does not establish our latency or safe co-residency with Qwen's cache.

The [MIT-declared checkpoint](https://huggingface.co/ppxin321/HolmesVAU-2B)
at `4ddd79211b9f144faa8fed08afc070f8b7c0db06` has **4,411,571,040 bytes** of
weights and **1,632,884 bytes** of auxiliary assets. The separately required ATS
checkpoint is **25,985,433 bytes (25.985 MB, not GB)** in the pinned code repository.
Model weights plus ATS total **4,437,556,473 bytes**; including Hugging Face
auxiliary files totals **4,439,189,357 bytes**, excluding repository source.
The ATS SHA256 has not been computed; its immutable Git blob identity is recorded.

The reference stack pins PyTorch 2.1.2 and Transformers 4.37.2 and uses custom
InternVL/InternLM2 code. It should not replace dependencies in the current
worker. Preserve ordinary coverage sampling when assessing ATS: a generic crime
sampler may discard a brief cash-transfer or counting action.

## Gated and optional comparisons

[CamVLM-8B](https://huggingface.co/xiaozhang79/CamVLM-8B) is surveillance-tuned
Qwen3-VL-8B. Public metadata pins `6aaed8fb0de20bd728ef64796833f2bd9b37ae42`,
**17,534,339,488 weight bytes**. Its weight card declares Apache-2.0, but access
requires accepting contact-sharing conditions; no acceptance occurred. The
[official repository](https://github.com/xiaozhang79/CamVLM) licenses code as MIT
and CCTV-Anomaly annotations as research/noncommercial CC-BY-NC-4.0. Weight and
dataset rights must be kept distinct. Its evaluation scripts use an external
LLM judge; private evaluation must use local/manual scoring. Camera-control gains
cannot recover missing pixels from prerecorded CCTV.

[V-JEPA 2.1](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)
offers 80M/300M B/L encoders for dense temporal features. Official checkpoint HEAD
sizes are **1,664,223,428 / 5,151,198,524 bytes**, including checkpoint contents
beyond a bare inference parameter estimate. The project uses MIT plus named
Apache utility files. CDN multipart ETags are not SHA256; actual content hashes
must be frozen before use. These encoders require bank-trained task heads and do
not generate explanations. [InternVideo-Next](https://github.com/OpenGVLab/InternVideo/tree/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo-Next)
remains optional further feature-transfer research, without a selected checkpoint.

VadCLIP is a possible cheap temporal proposal baseline; its released heads learn
UCF/XD crime and violence labels. VAD-LLaMA has an unresolved declared-license
chain. The industrial/medical `zqhang/AnomalyCLIP` is not a surveillance action
model; the distinct `lucazanella/AnomalyCLIP` video method needs separate license
review. Official repository links and pinned commits are in the JSON inventory.

## General teachers and a fair bank test

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) remains a general offline
teacher candidate: revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Apache-2.0,
**55,563,006,776 weight bytes**. Qwen3.6-35B-A3B and Qwen3-VL-32B require about
71.90GB and 66.71GB of weights respectively; 3B active parameters do not make all
35B experts disappear from memory. These footprints exclude activations, cache
and runtime buffers. They require separate short-context, low-concurrency fit
tests on the A100. [vLLM 0.31's architecture table](https://docs.vllm.ai/en/v0.31.0/models/supported_models/)
is not proof of exact-checkpoint compatibility. The existing driver565/cu129
deployment should be preserved; a newer CUDA image is not automatically usable.

[Cosmos-Reason2](https://github.com/nvidia-cosmos/cosmos-reason2) has real 2B/8B/32B
physical/embodied reasoning variants. Their NVIDIA license, gated access where
applicable, weight identities and sizes are recorded separately. They are not
bank-compliance detectors. No alternate teacher has been downloaded or benchmarked.

Compare matched source windows and hand/cash regions while retaining full-scene
context. Reuse the frozen 28/6/6 train/validation/final partition and train only on
the training partition. This is a machine-experiment split; human-lock flags are
false and human-reviewed fact/event gold is unavailable. Current supplied-label
diagnostics support coarse binary/time comparisons only. Report end-to-end
latency and peak VRAM separately; per-fact precision/recall, source-bound temporal
overlap and unclear/abstention calibration remain unverified without suitable
gold labels. The current comparison can proceed with those limits, without
requiring manual work from the user. Generic anomaly similarity, a fluent
explanation, and a larger model do not establish a legal violation or 100% accuracy.
