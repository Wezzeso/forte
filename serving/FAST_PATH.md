# Fast discriminative video path: research recommendation

Candidate experiment, 2026-10-17. Both public encoders were staged and fully hash
verified twice; isolated synthetic CUDA inference passed. The running Qwen
service was unchanged. The target is a measured reviewed
set with complete results under ten seconds from upload start. Perfect general
accuracy, sub-frame timing accuracy and arbitrary-network latency cannot be
guaranteed by an encoder selection.

## Two candidates

Root selected **Google SigLIP2 B/16 at 256 pixels** as the detail-first candidate,
with **B/32 at 256** as the speed comparator. Both official repositories declare Apache-2.0
and permit the intended vision-feature transfer use. Immutable model/processor
revisions, published weight SHA256 values and file sizes are captured in
`configs/serving/fast-encoder-candidates.json`. These are published metadata,
not downloaded-file verification. [B/32 model card](https://huggingface.co/google/siglip2-base-patch32-256),
[B/16 model card](https://huggingface.co/google/siglip2-base-patch16-256).

`serving.fast_encoder` supplies an isolated optional runtime and a generated-
only benchmark CLI. CPU guard/provenance tests and the first actual model/encoder
checks passed; no bank-task or full-video latency claim follows. Importing the module
requires no Torch, Transformers, NumPy, Pillow or API-core import.

| Property | B/32-256 | B/16-256 |
|---|---:|---:|
| Spatial patch tokens | 64 | 256 |
| Pooled feature width | 768 | 768 |
| Approximate vision parameters, including pooling | 94.6 M | 92.9 M |
| Estimated BF16 vision weights | 180 MiB | 177 MiB |
| Dominant matrix work per image, multiply-add counted as two FLOPs | 11.3 GFLOP | 46.2 GFLOP |
| Public generic ImageNet zero-shot result | 74.0% | 79.1% |

Parameter/work estimates are derived from the public architecture; they exclude
some normalization, activation and pooling operations. Published generic scores
are useful for a speed/detail comparison, not bank-fact accuracy. The official
release identifies these as fixed-resolution image encoders, and supplies a
separate NaFlex architecture. Use existing `transformers==5.17.0` support for
`SiglipVisionModel` (`model_type=siglip`), not the NaFlex class, with the observed
compatible Torch/CUDA stack, BF16 and SDPA. Avoid text-tower inference. The Hub
weight file contains both towers, around 1.5 GB in FP32; loading the full model
temporarily would require more memory than the retained vision tower.
[Official release and results](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md),
[versioned Transformers API](https://huggingface.co/docs/transformers/v5.17.0/model_doc/siglip).

OpenCLIP's B/32 DataComp checkpoint is another usable MIT-weight option; its
immutable revision/hash are recorded in the candidate metadata. It yields a
512-wide pooled feature and adds a different runtime/normalization contract.
V-JEPA2/2.1 is a commercially usable MIT video-native alternative with published
frozen-probe support. However, its joint temporal/spatial tokens, larger clips
and overlapping windows make it a less simple first latency baseline; its action
benchmarks do not supply the 19 bank fact heads. MobileCLIP2 is excluded: although
its code is MIT, the official model terms permit non-commercial research only.
[OpenCLIP checkpoint](https://huggingface.co/laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K),
[V-JEPA official models](https://github.com/facebookresearch/vjepa2),
[MobileCLIP model terms](https://github.com/apple-aiml-research/ml-mobileclip/blob/main/LICENSE_MODELS).

## A100 budget, without a fabricated benchmark

A05 counted a longest source of 142.533 seconds and a maximum 4273 native frames.
With a six-second vision allowance, a dense full-scene pass needs **712 frames/s**
including conversion, copies and encoding. At batch 64, approximately 67 batches
must average below 90 ms each. An extra ROI view on every frame doubles that
requirement. At four/eight fps there are about 570/1140 images, but those sampling
rates can miss brief events and cannot establish native-frame onset precision.

Estimated dominant encoder work for 4273 views is roughly 48 TFLOP for B/32 or
197 TFLOP for B/16. NVIDIA publishes a dense BF16 tensor peak of 312 TFLOP/s for
the A100 80 GB PCIe; dividing work by that peak is a theoretical compute floor,
not expected wall time. Small matrix efficiency, decoder work, image conversion,
memory copies, queues and ROI selection can dominate. No A100 encoder FPS or
peak results are recorded below; this calculation itself remains an estimate.
[A100 specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf).

For batch 64, BF16 input occupies 24 MiB. The intermediate MLP tensor is about
24 MiB for B/32 or 96 MiB for B/16, before additional buffers. A provisional
single-encoder working-memory allowance of 1–4 GiB is plausible with inference
mode and SDPA, but must be measured; it is not an allocation guarantee alongside
the VLM. Keep two bounded CPU/GPU batch buffers instead of materializing hundreds
of megabytes of pixels. Scene-only FP16 pooled features for 4273 frames need
about 6.3 MiB. Test batches 32, 64 and 128 only after an authorized quiet window.

The end-to-end stopwatch starts with the first upload byte. Ten seconds includes
upload, queue, validation, decode, feature extraction, heads, boundary checks,
evidence encoding and final transfer. The six-second vision allowance is a
planning scenario. Upload alone imposes a bytes/link-rate lower bound. Measure
all stages and the longest uncached content; encoder-only FPS is insufficient.

## Feature-to-fact architecture

Decode source frames once to RGB uint8 with exact integer PTS and rational time
base. Produce bounded `[B,3,256,256]` batches, initially using the official resize,
rescale and normalization. Preserve every global view. Add selected counter,
hand or object ROIs as additional views from original pixels; retain each source
box and never substitute ROI-only coverage. A 256-pixel scene can erase small
cash details before the head sees them, especially with 32-pixel patches. ROI
recall and preprocessing policy need reviewed validation; letterboxing or a new
tensor resize backend changes the processor identity.

Cache private pooled features `[T,V,768]` plus source hash, frame IDs, exact PTS,
view boxes, validity masks, immutable encoder/processor identity and feature
normalization. Retaining only pooled features is the small baseline; patch
features should be an explicitly budgeted option if localization demands them.
Use a small shared projection and a bidirectional temporal convolutional or
similarly bounded temporal head, with multiscale context. Emit per-frame
`[absent,present,unclear]` logits for each of 19 facts, frame-support logits and
boundary scores. A09 owns head design/training; A08 owns calibration and strict
conversion to current canonical facts/candidates. Deterministic rule grounding
and original evidence extraction remain downstream.

Train only from explicit reviewed frame/interval fact targets. Video-level event
labels are not 19 fact labels; unreviewed labels stay masked rather than absent.
The small supplied set does not establish sufficient training data. The current
VLM can propose offline teacher targets for human adjudication, but its outputs
are not gold truth and heldout media cannot enter fitting or threshold selection.
Never feed filenames, source labels, audit verdicts or train/test membership into
inference. Group duplicates/source scenes in splits; evaluate precision, recall,
unclear rate and boundary error per fact and event, with all failures counted.

Refine candidate boundaries on adjacent original PTS frames, retaining last
verified opposite/first supported state brackets. A05 observed approximately
33.333 ms or 66.667 ms frame gaps. Rendering timestamps in milliseconds records
source time; it does not create one-millisecond visual certainty. Interpolating
an onset between unobserved frames cannot prove exact onset. Keep wider
uncertainty brackets and human review where evidence is insufficient.

The VLM remains a selectable slow fallback and teacher. If a clip needs that
fallback, the existing autoregressive timing means the ten-second goal may be
missed; report that outcome rather than silently drop context or force a
negative. Promote the discriminative path only after reviewed heldout quality,
source-frame boundary validation, numerical preprocessing parity and actual
uncached first-upload-byte-to-final-response measurements pass together.

## Optional runtime and later synthetic command

A18 released its measurement window. A07 staged and fully hashed both snapshots,
eight files totaling 3,008,466,624 bytes, with a second independent offline check.
See `configs/serving/encoders/staging-validation.json`. The runtime repeats
full artifact verification before importing ML libraries or creating a device.
It uses only concrete `SiglipVisionModel`/`SiglipImageProcessor` classes and local
files, requires every vision weight to load, and retains no text tower. It checks
the pinned package versions, effective processor, output shape and finite FP16
CPU features. `EncodedBatch` excludes feature values from its representation.

Transformers 5.17's `SiglipImageProcessor` uses the Torchvision backend. The
default recipe explicitly binds that CPU backend and package versions, official
256-square resize with configured bilinear enum 2, rescale/normalization and
unmodified BF16 vision pooling. A distinct `decoder_rgb_256` mode requires a
caller-attested decoder transform SHA256, exact pre-resized geometry and
`do_resize=False`. `encode_frame_batch(batch)` accepts A05's byte-buffer batches
without importing the decoder and rejects transform mismatch. A05's libswscale
resize is not claimed pixel-identical to the official processor. A09's
`build_feature_spec` additionally binds decoder/sampling/crop/view recipes.
[Versioned processor source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/siglip/image_processing_siglip.py).

The following command was executed for both variants in separate optional GPU
processes with offline networking and read-only snapshots. Run as the snapshot
owner UID/GID 1000:1000: with capabilities dropped, UID0 cannot bypass the
owner-only directories. The existing Qwen process remained warm and unchanged.

```text
python3 -m serving.fast_encoder \
  --snapshot /encoders/siglip2-base-patch16-256 \
  --manifest configs/serving/encoders/siglip2-base-patch16-256.manifest.json \
  --manifest-sha256 cd51863090e8c411a1227d8039e44118c44488243e3d615d73907106568877e7 \
  --batch-size 32 --frames 4273 --warmup-batches 2 --trials 1
```

The CLI generates all RGB pixels itself and accepts no media path. It reports
actual load/hash time, CPU preprocessing, synchronized H2D/forward/D2H duration,
CUDA-event forward duration, wall throughput and peak allocated/reserved device
memory. Warmup and loading are separate from the measured iteration. Default
inputs are generated 256-square images; `--input-width`/`--input-height` can
measure official resizing from a larger generated geometry. This is an encoder
benchmark and omits upload, video decode, head, evidence and network. It makes no
event-accuracy or ten-second end-to-end claim. A failed/OOM run remains a failure;
the utility does not silently shrink batches or drop frames.

The first 4273-image batch32 trial measured 7.639 seconds for B16 and 6.504 seconds
for B32, including about 2.56 seconds of generating synthetic pixels. B16's CPU
preprocessing took 2.591 seconds and H2D/forward/D2H 2.172 seconds; B32 took 2.596
and 1.029 seconds. CUDA-event forward totals were 1.861/0.719 seconds. Peak encoder
allocations were 349/258 MiB. Full hashing and cold library/model loading took
about 12 seconds separately. Raw numerical proofs and the actual synthetic
Torch/safetensors write/load/corruption/partial-stream test are under
`configs/serving/encoders/`, summarized in `runtime-validation-summary.json`.

Final optional runtime hardening then enabled deterministic algorithms,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, a fixed seed, TF32 off and source-code hash
bindings. The historical timings retain their original settings; the final
deterministic recipe needs its own smoke and timing before a new claim. A09's
cache identity consumes the actual executed recipe rather than an old digest.

`serving.feature_stream.iter_encoded_batches` overlaps a bounded A05 decode
producer with one encoder consumer. It retains at most two lookahead batches
plus the active batch, preserves canonical source IDs/PTS/transforms and requires
native continuity, EOF and tail completion. It exposes the metadata-only
`EncodedSourceBatch` protocol consumed by the private cache writer. Premature
exit/deadline cancels and drains the reader; no partial stream can publish a
complete cache. Optional ROI views must supplement a full global view. The first
cache writer accepts global V1 only; a second view must not be silently flattened
into extra source frames.
