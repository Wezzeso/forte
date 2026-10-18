# Local VLM compatibility preflight — 18 October 2026

This record verifies upstream metadata and documents a synthetic compatibility
diagnostic. It contains no new development-set accuracy measurement, model
promotion, production change, or observation of supplied media. No accuracy
parameter was tuned. The synthetic geometry check has one fixed diagnostic
tolerance, 65 normalized coordinate units, chosen before GPU testing.

## Verified upstream identities

The official Hugging Face metadata APIs returned these exact revisions on
18 October 2026. Both repositories are public, ungated, and tagged Apache-2.0.

| Repository | Revision | Architecture |
|---|---|---|
| `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | `Gemma4ForConditionalGeneration` |
| `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | `Qwen3_5ForConditionalGeneration` |

Sources: [Google repository metadata](https://huggingface.co/api/models/google/gemma-4-31B-it),
[Qwen repository metadata](https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8).
Gemma's pinned config records BF16 weights; Qwen's config records fine-grained
FP8 with 128×128 blocks. These are model metadata facts, not measurements of
memory use or accuracy on Forte data.
[Gemma config](https://huggingface.co/google/gemma-4-31B-it/resolve/842da3794eaa0b77d5f08bae87a17459d91ff475/config.json),
[Qwen config](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json).

Gemma's repository does not contain a separate LICENSE file in its API file
inventory. Its model card links Google's complete Apache 2.0 terms.
Qwen3.8's repository contains a LICENSE file.
[Google Apache 2.0 terms](https://ai.google.dev/gemma/apache_2),
[Qwen pinned LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/LICENSE).
The central register is `docs/submission/licenses.md`; it must be updated before
the checkpoints are used.

## vLLM support and runtime configuration

The pinned v0.31.0 supported-model list includes both architectures, so this
preflight does not justify changing to a newer runtime. The tag resolves to
`db9527a46873454610df6dbedf79a36d6bf1a7f6`. vLLM's license is Apache-2.0.
[Supported models](https://docs.vllm.ai/en/v0.31.0/models/supported_models/),
[tag API](https://api.github.com/repos/vllm-project/vllm/git/ref/tags/v0.31.0),
[pinned runtime license](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/LICENSE).

v0.31.0 was published on 5 October 2026 at 06:44, 13 calendar days before
this preflight. The handoff explicitly permits this release; its two-week
condition applies to a fallback if this release lacks model support. The exact
CUDA 12.9 image is `vllm/vllm-openai:v0.31.0-cu129`; the plain v0.31.0 image
uses CUDA 13.0. Record the pulled image digest in the actual run record.
[Release and image table](https://github.com/vllm-project/vllm/releases/tag/v0.31.0).

Start Gemma BF16 on the separate rented H100 with the following conservative
resource choices. They are proposed starting values, not a proven throughput
configuration. Use the existing driver workaround from the handoff, uninstalling
`torchcodec` inside the cu129 container before serving when required.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

vllm serve /ephemeral/models/gemma4 \
  --served-model-name gemma4 \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":32,"video":0}' \
  --mm-processor-cache-gb 4 \
  --mm-processor-kwargs '{"max_soft_tokens":560}' \
  --seed 0 \
  --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --no-enable-log-requests \
  --no-enable-log-outputs
```

The CLI documents the logging switches above. The `gemma4` parser and constrained
JSON output are documented in the official recipe. Keep both request and output
logging disabled, since requests contain image data.
[Pinned CLI](https://docs.vllm.ai/en/v0.31.0/cli/serve/),
[Gemma recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html).

Gemma's card says its non-thinking 31B mode can still produce an empty thought
channel. Therefore disabling thinking alone is insufficient evidence of strict
JSON compatibility; verify the parser removes the channel. The card recommends
images before associated question text and lists image budgets of 70, 140, 280,
560, and 1120 tokens. The model default is 280. Before real-data runs, the
source-only engineering recommendation was changed to 560 for the first Gemma
candidate, with the rationale below. Keep the selected value identical across
both independent runs and record it in server provenance. This is one declared
processor setting and zero label-tuned parameters.
[Official Gemma card](https://huggingface.co/google/gemma-4-31B-it).

For Qwen3.8, use the same local/offline arrangement with its local weight
directory, `--served-model-name q38`, and `--reasoning-parser qwen3`.
The official family recipe explicitly combines that parser with
`--default-chat-template-kwargs '{"enable_thinking":false}'` to disable
reasoning. Qwen3.8's card also documents this chat-template option.
[Qwen family recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html#configuration-tips),
[Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).

Do not add online quantization to the already quantized Qwen checkpoint.
If Gemma BF16 does not leave enough memory, v0.31.0 documents online conversion
as `--quantization fp8_per_tensor`; its release changed the older `fp8`
shorthand. Any quantized Gemma result is a separate model candidate.
[Pinned online quantization](https://docs.vllm.ai/en/v0.31.0/features/quantization/online/).

Offline environment switches prevent supported libraries' network activity but
do not constitute a network firewall. A fully isolated arrangement is a VLM
container with `--network none`, and smoke/evaluator containers sharing its
network namespace with `--network container:<vlm-container>`, reaching the
same `127.0.0.1:8000`. Bind the local weight directory read-only.
[Hugging Face offline environment option](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline),
[vLLM environment options](https://docs.vllm.ai/en/v0.31.0/configuration/env_vars/).

## Synthetic diagnostic

`tools/model_schema_smoke.py` accepts no video/image path. It creates all pixels
with Pillow, sends requests only to HTTP loopback, ignores proxy environment
variables, refuses redirects, and never prints server response text or exception
messages. It writes only validated enums, booleans, numeric values, boxes, null,
timings, token counts, and diagnostic status. Free-form schema fields are
replaced with null after validation; the report counts the dropped fields.

```bash
python tools/model_schema_smoke.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model gemma4 \
  --out /runs/gemma4-schema-smoke.json
```

Default requests exercise the server's thinking defaults. An additional run with
`--disable-thinking` can diagnose a server-template configuration problem;
it does not make a failing default run pass.

The 15 cases cover every current schema in `src/forte/fast/prompts.py`, two-image
door comparison, the production concise verifier path, a known red rectangle,
and inputs containing 14, 18, and 32 synthetic images. Each image in capacity
cases has distinct pixels, including across cases, to prevent encoder/cache
deduplication. These cases record prompt tokens to reveal context-pressure failures. Rectangle coordinates are
deliberately different in source pixels and normalized coordinates; accepting
pixel coordinates cannot pass this calibration. Layout boxes must be ordered
and within 0–1000. No coordinate adapter is introduced by the tool.

The concise-verifier case sends the exact full `VERIFY_SCHEMA`, production stop
string, and closes the truncated object with the same helper as the worker.
It then validates against `VERIFY_DECISION_SCHEMA`, requiring `visible`,
`confidence`, and `t_sec`. If the server emits `observation_ru` before any of
those decision fields, the stop interrupts generation and the case fails.
The case also requires an actually truncated response without `observation_ru`;
a complete object from a server that ignores stop sequences fails.
Thus a schema-valid full response alone is insufficient. Do not disable this
failure or silently reorder a model response; any adapter needs its own default
off flag, tests, staging, and repeated accuracy qualification.

Local checks completed: CLI/import, registration of all 15 cases, normalized
versus pixel rectangle distinction, free-text stripping, strict rejection of
Markdown fences and non-finite JSON, rejection of nonempty reasoning and length
termination, HTTP error sanitization, and concise-stop success/failure fixtures.
Mocked transport responses are synthetic. No GPU inference result is claimed in
this document; actual smoke files and full evaluations must be recorded separately.

## Visual-token choice before real-data runs

The exact server option `--mm-processor-kwargs '{"max_soft_tokens":560}'` is
documented by the official Gemma recipe and read as a top-level processor option
by pinned v0.31.0 code. The physical principle is to retain a finer spatial
representation of small objects by reducing compression into pooled vision
tokens. This is an architecture-based choice made before reading any new accuracy
result, not a score sweep.
[Official processor option](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html#dynamic-vision-resolution),
[pinned processor implementation](https://docs.vllm.ai/en/v0.31.0/api/vllm/model_executor/models/gemma4_mm/).

Using that implementation's resize formula, patch size 16, and 3×3 pooling gives
the following calculated values. These are code-derived estimates, not measured
tokens from an executed model request.

| Input size | Budget | Processor size | Pooled tokens/image | Tokens for 32 images, including image delimiters |
|---|---:|---|---:|---:|
| 464×848 | 280 | 576×1056 | 264 | 8,512 |
| 464×848 | 560 | 816×1488 | 527 | 16,928 |
| 640×960 synthetic smoke | 280 | 624×960 | 260 | 8,384 |
| 640×960 synthetic smoke | 560 | 912×1344 | 532 | 17,088 |
| 640×960 synthetic smoke | 1120 | 1296×1920 | 1080 | 34,624 |

For native 464×848, default 280 does not downsample the source raster. The
relevant difference is the pooled representation: one token spans approximately
38.6 native pixels on each axis at 280 versus 27.3 at 560. More tokens can
preserve fine distinctions in the encoder representation, but cannot create
detail missing from the source. An accuracy improvement is still unproven.
The conservative 560 budget bound is 32×562=17,984 image-side tokens, leaving
space in 32,768 for prompts and outputs. Budget 1120 cannot fit the 32-image
smoke before adding any question text. Distinct-image encoder memory must still
pass on the actual H100.

## Optional safe-lock perception diagnostic

`tools/safe_lock_probe.py` is a separate diagnostic tool; it changes no analyzer
flag, rule, disabled type, evidence path, or production file. It reads cached
per-video layout, schedules the same evenly spaced local windows over every
video, localizes the lock within a native safe crop, and asks two differently
worded enum-only questions over native lock crops plus original context. It
accepts no labels and emits no violation/candidate/fire decision.

```bash
python tools/safe_lock_probe.py --run /runs/gemma4a.json --videos /videos \
  --model gemma4 --out /runs/gemma4-safe-lock.json
```

Default sampling is six windows per video, five frames per three-second window,
with original frames retained at 2 fps. Lock crops expand the locally predicted
lock box by 2× with a minimum native side of 24 pixels; they are never resized or
enhanced. No median/fused image or evidence file is written. These are six
declared sampling/crop choices, none tuned to labels or benchmark outcomes.
Window placement uses only the video's observed temporal range; identifiers are
used solely to load and associate results with files.

There are up to three calls per window: one localization and two classifications.
For 40 videos, this is 240 localization calls and at most 480 classification
calls. Unresolved/occluded locks abstain. HTTP, schema, decoding, malformed cached
layout, and cached processing errors remain failures, with exception types only.
Cached correctness and labels are ignored. Raw outputs never leave the local
model client; all reported model fields have closed enums or numeric boxes.

Synthetic checks cover native crop offsets, bounded/finite geometry, paired
structured output, unresolved abstention, sanitized technical failure, rejection
of extra free text, and preservation of cached processing failure. These checks
do not establish whether real keys can be resolved at the source resolution.

## General proposals requiring approval and fresh qualification

The current omission probes sample a transaction into 16 frames and accept any
located configured region; they do not measure temporal visibility coverage or
prove that the complete relevant transaction was captured. A stronger model alone
cannot make sparse sampling evidence of an absent step.

Only if stronger perception is measured, consider explicit transaction-boundary
and per-frame visibility enums, dense sequential coverage of the handover/device
region, and abstention whenever a complete opportunity to observe the step is
missing. A procedural step observed in any window must suppress an omission
claim across that complete transaction. Coverage must be measured over time,
with the original brief's 90% requirement; counting 90% of a sparse sample is
not 90% temporal coverage. This is a proposal, not implemented decision logic.

For a future key-left check, the general visual principle is a resolved key
remaining in the resolved lock after hand withdrawal and subsequent separation
of the cashier from the safe. The present tool reports only the needed visual
primitives. Any temporal persistence threshold must come from all negatives,
and enabling the type still requires zero fires on every required negative and
augmented copy. No threshold or rule has been selected here.
