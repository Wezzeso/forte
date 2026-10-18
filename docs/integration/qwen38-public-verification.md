# Qwen3.8-27B public candidate verification

Verified 17 October 2026. This receipt establishes public identity, downloaded metadata bytes and source-level compatibility leads. It does not establish a model load, a trained adapter, working A100 kernels, private-video accuracy, latency or production selection. No private data or inference endpoint was used for this work.

## Frozen candidate and artifacts

The official repository [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) resolves at the handoff revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The revision was retained; no model substitution or new revision was selected. The [revision-specific public API](https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0?blobs=true) returns that exact SHA and the complete file inventory.

Local artifacts:

- `configs/serving/qwen38/public-candidate.manifest.json`: expected identity of every public file, candidate architecture and explicit verification limits. SHA-256 `9be85d7b1a57be0f59e707b9a8872b964c506a515f7670958d2d5d60de197225`.
- `configs/serving/qwen38/public-api-snapshot.json`: raw revision-specific API response with repository blob identities and expected LFS SHA-256 for every weight shard.
- `configs/serving/qwen38/public-metadata.inventory.json`: local byte counts and measured SHA-256 for all fourteen downloaded nonweight files.
- `configs/serving/qwen38/public-metadata.validation.json`: passing byte-size/SHA-256 and public Git-blob/LFS-identity verification for all fourteen files, visual token IDs, processor geometry, template equality and index completeness. This is a static public-metadata check; it does not load a processor or weights.
- `configs/serving/qwen38/public-snapshot/`: the fourteen original public metadata/license/tokenizer files, downloaded at the frozen revision.
- `configs/serving/qwen38/upstream-source/inventory.json`: public source URLs, resolved commits and measured source hashes; source inspection is separate from installed-runtime verification.

The public snapshot has 32 files. Eighteen safetensor shards total **55,563,006,776 bytes** (approximately 51.75 GiB); fourteen remaining files total **23,108,087 bytes**. The complete expected inventory is **55,586,114,863 bytes**. The weight index references exactly those eighteen shards and 1,199 tensor names. Its `metadata.total_size` is **55,562,855,904 bytes**, which counts tensor payload rather than physical shard bytes; these two byte totals must not be interchanged. Local shard download/hash verification belongs to the staging receipt, not this receipt.

The downloaded [LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE) is Apache-2.0 (SHA-256 `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`). Preserve its license and notices when distributing the model or derivatives. Public model licensing does not establish rights in private training videos or third-party application assets.

## Architecture and processor

The actual [pinned configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json) declares:

| Property | Verified value |
| --- | --- |
| `model_type` | `qwen3_5` |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Language layers | 64, hidden size 5,120, intermediate size 17,408 |
| Hybrid layout | 48 `linear_attention`; 16 `full_attention` at zero-based indices 3, 7, …, 63 |
| Full attention | 24 query heads, 4 KV heads, head dimension 256 |
| Linear attention | 16 key heads, 48 value heads, key/value head dimension 128 |
| Native context | 262,144 positions |
| Source weight dtype | BF16; no source quantization configuration |
| Vision | 27 layers; patch size 16; temporal patch size 2; spatial merge size 2 |
| Image/video token | 248056 / 248057 |
| Vision start/end token | 248053 / 248054 |

The `transformers_version` value `5.8.0.dev0` records the config writer. It is not proof that any specific Transformers wheel loads, trains or serves this model correctly.

The [image preprocessor](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/preprocessor_config.json) and [video preprocessor](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/video_preprocessor_config.json) both declare **`Qwen3VLProcessor`**. The image processor is `Qwen2VLImageProcessorFast`; the video processor is `Qwen3VLVideoProcessor`. The [tokenizer configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json) declares `Qwen2Tokenizer`. Reusing the processor family is compatible with a different model architecture; it does not justify retaining the old Qwen3-VL model class or visual token IDs.

The pinned [tensor index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json) contains these distinct language module families:

- Linear layers: `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`.
- Full attention layers: `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Every language layer: `mlp.gate_proj`, `up_proj`, `down_proj`.

These names provide public structural evidence for target selection. A training receipt must still enumerate actual `named_modules()` and adapted tensors in the installed model. The index also contains vision and MTP tensors; neither should be accidentally adapted by a language-only recipe.

## Template and generation controls

The standalone [chat template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja) is byte-identical to `tokenizer_config.json`'s embedded template after UTF-8 encoding: SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

The template defaults to thinking enabled, `reasoning_effort="xhigh"` and preserved historical thinking. Supported reasoning values are `xhigh`, `medium`, `low`. A bounded observation candidate should explicitly pass `enable_thinking=False` and `preserve_thinking=False` and verify rendered prompt tokens in its actual processor. Disabling thinking still inserts a closed empty think block in the assistant generation prefix; blindly stripping an assumed Qwen3-VL prefix will produce the wrong completion boundary.

The [generation configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/generation_config.json) enables sampling with temperature 1.0, top-k 20 and top-p 0.95. Task-specific greedy decoding is a deliberate override, which must be bound in the candidate receipt. It needs complete-output and repeatability checks; a fixed seed alone does not establish deterministic end-to-end results.

## Public training and kernel leads

The [ms-swift Qwen3.8 guide pinned at `80e55df909c83d198ad1091a6cebceb5428a46ff`](https://github.com/modelscope/ms-swift/blob/80e55df909c83d198ad1091a6cebceb5428a46ff/docs/source_en/BestPractices/Qwen3_8-Best-Practice.md) specifies a dedicated `qwen3_8` template, Transformers ≥5.9 and `qwen_vl_utils` ≥0.0.14. It recommends Python 3.12 for the FLA path and lists FLA, causal-conv1d and attention acceleration dependencies. These are a framework recipe, not project runtime attestations. No Swift or heavy dependency installation was performed here.

The [Transformers v5.17.0 implementation pinned at `856157a2f3e9594954310df18fdccc31ffddebe9`](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/qwen3_5/modeling_qwen3_5.py) defines `Qwen3_5GatedDeltaNet` and actual linear projections matching the pinned tensor index. Its convolution and gated-delta functions use `use_kernel_func_from_hub_with_fallback` for `causal_conv1d` and `fla`, plus kernelization decorators. Record effective callable/kernel selection and actual performance in the candidate image; a package import or a class-name match is insufficient. The [official model documentation](https://huggingface.co/docs/transformers/v5.17.0/model_doc/qwen3_5) warns that fallback paths can be slower and use more memory.

The [bitsandbytes hardware guide](https://huggingface.co/docs/bitsandbytes/installation) lists NF4/FP4 support from compute capability 6.0 and SM80 in its supported Linux wheel targets. That makes A100 NF4 a plausible candidate, not a measured working combination of this project's driver, CUDA, wheel, vision inputs, GDN kernels and adapter reload. Pin and test the actual wheel; preserve the existing driver/cu129 stack until another image is independently verified.

## vLLM source findings

The [official Qwen3.8 vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) explicitly limits its demonstrated serving coverage to text and presents Blackwell quantization examples. Its quoted speeds and memory sizes must not be used as A100 video measurements.

The [vLLM v0.31.0 implementation pinned at `db9527a46873454610df6dbedf79a36d6bf1a7f6`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/qwen3_5.py) has LoRA support and explicit multimodal packed mappings for both hybrid projection groups: `in_proj_qkv` plus `in_proj_z` map to `in_proj_qkvz`; `in_proj_b` plus `in_proj_a` map to `in_proj_ba`. This is positive source-level evidence. The installed image must be checked for the same implementation, then actual video and base-plus-adapter reload tested before runtime selection.

**vLLM 0.31 uses an out-of-tree bitsandbytes plugin.** The [v0.31 documentation](https://docs.vllm.ai/en/v0.31.0/features/quantization/bnb/) requires `vllm-bnb-plugin`. The [pinned official plugin metadata at `d4914a9e3c5927b33b7ef941385a4fda2187e2b9`](https://github.com/vllm-project/vllm-bnb-plugin/blob/d4914a9e3c5927b33b7ef941385a4fda2187e2b9/pyproject.toml) declares version 0.0.3, depends on `bitsandbytes>=0.48.1`, and registers `vllm_bnb_plugin:register` through `vllm.general_plugins`. Installing bitsandbytes alone does not establish that vLLM's quantization registry contains this backend. The old in-tree `bitsandbytes.py` URL returned 404; this is recorded in the source inventory. Use a separately tested, hash-pinned plugin/image if choosing that path.

## Processor time and remaining gates

The [v5.17.0 Qwen3-VL processor implementation](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/qwen3_vl/processing_qwen3_vl.py) derives video prompt times from `frames_indices / fps` and averages the frames in each temporal pair. When fps is missing, it warns and uses 24 fps. Native PTS and source time-base semantics therefore need a separately verified bridge, especially for variable frame rate and nonzero source starts. The downloaded processor files alone do not prove this bridge exists.

Required gates remain actual staged-shard hashes, installed image/module identities, synthetic processor prefix/mask/time checks, actual A100 kernel and quantization selection, tiny and full-model visual forward/backward, two real optimizer steps, adapter changes/frozen-base proof, exact save/reload, training-only supervision/exposure gates, quality selection, full API validation, offline operation and upload-to-result timing. Preserve the baseline manifest and rollback while these are unresolved.

This receipt reports no GPU execution, private source viewing, training step, deployment, quality score or latency result.
