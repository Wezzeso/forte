# Private local model adapter

`LocalModelProvider` sends job-owned JPEG images to the private vLLM service.
It has no external AI provider, label lookup, inference-time download, or mock
production fallback. `models/manifest.json` pins the model and processor to
`Qwen/Qwen3-VL-8B-Instruct` revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, with upstream hashes for all four
weight shards and locally computed hashes for the public tokenizer/config files.
The public model card declares Apache-2.0. These checks establish artifact
provenance; event accuracy and capacity require actual evaluation.

```python
from pathlib import Path
import time
from forte.model import LocalModelConfig, LocalModelProvider

async with LocalModelProvider(LocalModelConfig(
    media_root=Path("/data/work"),
    base_url="http://model:8000/v1",
    model="Qwen3-VL-8B-Instruct",
)) as provider:
    # ClipRequest frame paths must be inside /data/work/{clip.job_id}/.
    response = await provider.infer(
        clip, prompt=bundle.prompt, response_schema=bundle.response_schema,
        deadline=time.monotonic() + 60,
        cancel_event=cancel_event,  # asyncio.Event or threading.Event
    )
    # Keep response.data private; send it to A08's semantic validator.
```

Production inference requires `/health`, `/readyz`, and `/v1/models`. The private
launcher attests verified weight hashes, exact revision, vLLM version, completed
synthetic prewarm, and a live process. Readiness is cached for two seconds and
invalidated after an HTTP failure. The explicit `require_readiness=False` option
exists for synthetic transport tests; deployment uses the default `True`.
`check_available()` checks only the served alias; `check_ready()` checks full
launcher readiness.

Only private IP literals, localhost, and configured exact Docker service names
are accepted. HTTP proxies and redirects are disabled. Local file resolution
rejects cross-job paths, remote URLs, network shares, and alternate data streams.
JPEG decoding verifies dimensions and optional source checksums; metadata is
removed through RGB JPEG re-encoding. Every supplied source pixel is retained;
the server applies its pinned processor budget. Evidence originals remain with
the evidence owner.

The default request budget is 16 frames, 10 MiB per image, 64 MiB per clip, 4096
generated tokens, and at most two HTTP attempts. Only connection establishment
or connection-pool failures before admission can retry. HTTP responses and
ambiguous read/write/protocol failures never replay a POST. The caller's absolute monotonic deadline
and cancellation signal bound preprocessing, readiness, backoff, and HTTP work.
Invalid JSON, duplicate keys, nonfinite numbers, tool calls, refusals, truncated
generation, unknown model identity, and schema mismatches fail explicitly.
The A08 validator subsequently verifies requested facts, taxonomy, frame IDs,
and event timing. No exception includes raw model output or server error bodies.
Cancellation is idempotent; response close runs in a retained shielded task with
a separate one-second drain grace. Tasks outliving that grace remain retained
until settled. `pending_cleanup_count` reports local drain state only; actual
backend abort requires engine-gauge evidence. Responses request identity
encoding and reject compressed bodies so the byte budget applies before decode.

Metrics include actual input dimensions, encoded bytes, preprocessing and HTTP
timings, attempt counts, and token usage when returned. The standard response
does not expose actual processor grid sizes or visual-token counts; those are
`None`. Input pixels are never reported as processed pixels. Model scores are
not invented.
When enabled on vLLM, `server_metrics` preserves only actual numeric TTFT,
decode duration, queue duration, inter-token latency (milliseconds), and output
throughput (tokens/second) from its documented per-request response. Missing or
unavailable values remain null. TTFT is not reported as pure GPU prefill time.

The snapshot preflight is offline and uses the Python standard library only:

```text
PYTHONPATH=src python -m forte.model verify --snapshot /models/Qwen3-VL-8B-Instruct --manifest models/manifest.json
```

It hashes every declared artifact and exits nonzero for unavailable/modified
weights. A successful verifier does not claim a successful inference run.
The build-time `forte.model.refresh_metadata` command only fetches public model
metadata and configuration files; it is never imported by inference. Full model
weights must be downloaded into the team environment before offline startup.
The explicit provisioning command downloads and verifies the pinned public
artifacts directly on that host, with at most four streams and resume support:

```text
PYTHONPATH=src python -m forte.model stage --snapshot /models/Qwen3-VL-8B-Instruct --manifest models/manifest.json --workers 4 --timeout 1800
```

It needs outbound access during staging. The serving launcher subsequently uses
the verifier with offline model-loading flags; production inference does not
import or invoke the downloader.

After A15 launches and prewarms the real private server, exercise the production
adapter and A08's generated schema/semantic validator with synthetic media:

```text
PYTHONPATH=src python -m forte.model smoke --base-url http://127.0.0.1:8000/v1 --media-root /tmp/forte-model-smoke
```

This command generates its own three-frame JPEG sequence and cannot accept supplied footage. It
prints validation counts and timings. A pass establishes a real local request;
it does not establish event-detection accuracy on the datathon data.
The launcher may call `synthetic_smoke(require_readiness=False)` after offline
artifact verification and shape warmup, before publishing `/readyz`, to avoid a
circular readiness dependency. The public smoke CLI keeps readiness enabled.

Implementation sources verified on 17 October 2026:

- [Qwen model card and license](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- [Pinned Hugging Face metadata](https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct/revision/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b?blobs=true).
- [vLLM structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/): `structured_outputs: {"json": schema}`; deprecated `guided_json` is not used.
- [vLLM multimodal inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/): inline JPEG data URIs.
- [vLLM 0.31.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0).

Synthetic checks: `PYTHONPATH=src python -m unittest discover -s tests -p test_model_adapter.py -q`.
These tests prove transport/parsing/privacy boundaries, not model quality.

The separate Qwen3.8 candidate uses explicit `expected_runtime_identity` and
`chat_template_kwargs` tuples in `LocalModelConfig`. Load the expected pins with
`serving.qwen38_identity.load_expected_runtime_identity()` from its reviewed
candidate launch configuration. All model, processor, tokenizer, template,
adapter tree/configuration, training run, quantization, generation, image,
source and complete runtime configuration hashes must match the actual private `/readyz` response. The service
must also attest a loaded model, completed prewarm and a positive adapter tensor
count. `transformers_peft` candidates use this separate contract without a
vLLM version field. Empty candidate identity tuples retain baseline checks.

Candidate generation is greedy with the pinned seed and token bound. Both
thinking controls must be explicitly false; their canonical generation hash
is checked before requests. A response with reasoning content despite disabled
thinking fails. `synthetic_smoke()` accepts the candidate alias, revision, seed,
token bound, controls and expected identity; its defaults still exercise the
baseline. Strict JSON also rejects numeric exponent overflow such as `1e999`.
Only assistant messages with no function/tool calls are observations.

The public API opts into this candidate readiness contract only when
`FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG` identifies its complete runtime config.
Missing or mismatched candidate pins fail readiness without falling back to
the baseline profile. Optional `FORTE_MODEL_API_KEY` authenticates readiness
requests and is excluded from configuration representations.
