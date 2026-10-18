# Explicit untuned Qwen3.8 comparison runtime

The untuned arm is a separate base-only diagnostic. It loads the pinned real
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, with the same verified processor,
template, NF4/BF16 choice, pixel/context budgets and generation implementation as
the trained candidate. It never creates a dummy adapter or zero-step training
receipt. Adapter/config/training hashes are explicitly null. Production candidate
verification, `LocalModelConfig` candidate backends, API readiness and production
worker selection remain unchanged.

No GPU model was loaded or called to implement this route. CPU tests exercise
invented scalar objects, generated JPEGs and mock HTTP transports only.

## Bind and launch privately

Start from `configs/serving/qwen38/untuned.runtime.example.json`. It intentionally
fails until the actual image identity, source archive, installed package versions
and shared profile digest are bound. Keep the resulting `*.bound.json` outside
public packages and image build contexts. Do not supply `adapter_path` or
`training_run`: the diagnostic rejects them, including a zero-step run.

After freezing the real source/archive and image/package pins, compute
`serving.qwen38_comparison_profile.runtime_profile_sha256(config)` and set the
config's `runtime_profile_sha256`. This digest covers the exact base/processor,
tokenizer/template, quantization, generation, limits, prewarm shapes, device,
package versions, runtime image and source archive. It excludes only backend/arm
names, served alias/listen address and adapter/training identity. The tuned arm
must use the same profile; verify it before either arm's development results are
compared:

```sh
python -m serving.qwen38_untuned --config /private/qwen38-untuned.bound.json --compare-trained-profile /private/qwen38-tuned.bound.json
python -m serving.qwen38_untuned --config /private/qwen38-untuned.bound.json --verify-only
python -m serving.qwen38_untuned --config /private/qwen38-untuned.bound.json --state /run/forte/qwen38-untuned-ready.json
python -m serving.qwen38_untuned --config /private/qwen38-untuned.bound.json --check-ready
```

Run the service in the authorized exclusive A100 window in the same immutable
candidate image/source release, with verified local read-only base/source/config
mounts and an isolated namespace with no default route. Use the existing private
network or execute the diagnostic caller in the same network namespace. There is
no public port requirement. Namespace and package/image assertions are checked;
an independent provider/no-egress/image inspection remains separate evidence.
The working baseline must first drain and hand off the GPU under root's approved
operator plan. This module performs no handoff, provisioning or budget extension.

`--verify-only` hashes the full base snapshot and source artifact without CUDA or
inference. A launch then actually loads the base using A06's shared
`load_processor_and_base`, verifies A100 SM80/CUDA 12.9/BF16, effective template
and tokens, loaded hybrid architecture and actual NF4 state where selected. It
freezes inference parameters, rejects loaded LoRA tensors, runs generated-image
shape warmup and the existing real decision-schema/visual sanity smoke. Only then
does its distinct `/readyz` advertise ready. It reports `trained:false`,
`adapter_loaded_tensor_count:0`, `diagnostic_only:true`, and
`backend:"transformers_base_diagnostic"`. Kernel-path performance and model
quality are not established by load/warmup.

## Use the unchanged analysis pipeline through guarded injection

`serving.qwen38_untuned_provider.UntunedModelProvider` subclasses the existing
local provider. All media ownership/budget, inline JPEG, prompt, strict JSON,
response-schema, disabled-thinking, deadline/cancellation and output checks are
inherited. Its separate readiness contract checks every expected
base/processor/tokenizer/template/quantization/generation/image/source/config and
shared-profile pin, actual load receipt, completed warmup/schema smoke and zero
adapter tensors. It requires explicit null adapter/training identities and the
exact pinned revision. Readiness is checked before every inference request.

The default baseline provider and trained candidate provider reject this
diagnostic readiness. No API setting enables it. For an already authorized
direct private comparison runner, inject it explicitly:

```python
import threading
import time
from serving.qwen38_untuned_provider import make_untuned_pipeline

# frozen_settings is the SAME WorkerSettings recipe used for the other arms.
# owned_job is the already accepted, content-bound local source copy, with
# job_id, attempt, owned_path and content_hash. No filename/answer lookup occurs.
pipeline = make_untuned_pipeline(
    frozen_settings,
    "/private/qwen38-untuned.bound.json",
    base_url="http://127.0.0.1:8000/v1",
    api_key=private_model_api_key,
)
try:
    result = await pipeline.analyze(
        owned_job,
        stage=private_stage_callback,
        deadline=time.monotonic() + frozen_settings.job_timeout_seconds,
        cancel_event=threading.Event(),
    )
finally:
    await pipeline.aclose()
```

The helper changes only the injected model endpoint/alias and candidate selector,
preserving the caller's decode, sampling, scheduler, prompt, deterministic rules,
screening/refinement and evidence settings. It adds explicit diagnostic expected
identity to the private audit; actual provider readiness is separately attached
to inference metrics. Bind the full runtime config and shared profile in the
comparison recipe and retain actual server receipts with every terminal. A
source/archive hash binds the new helper code as well as the reused pipeline.

For the tuned arm use A06's strict trained provider path. For the baseline use the
unchanged baseline provider. Do not accept a failure in any arm as a clean
negative or replace repeat zero with a successful retry. Preserve reserved
attempt IDs and export every failure to `evaluation.qwen38_comparison` with the
correct `expected_model_identity`; actual model identity remains unknown when a
failure precedes verified execution.

Direct `AnalysisPipeline` measurements start after source acceptance and
therefore do not establish upload-start, transfer or usable-UI latency. Keep
those fields unknown unless A12's actual client clock measured them. They are
useful quality comparisons and fresh decode/evidence diagnostics, not a fabricated
end-to-end <10-second result. Load/warmup clocks also remain separate.

## Development population and tests

A09's private development role export is
`benchmarks/private/qwen38/development-roles-20261017-a09.json`, digest
`00eb66893d023fea0dd99533970afa76a13f6eb1d0ffc103670038c30c7ce7fb`.
Its 34 records contain 28 TRAIN and six reused VALIDATION contents, no final-check
content or answers. The comparison's six validation contents have three positive
and three negative supplied binary labels; four supplied coarse points remain
approximate diagnostics. Refresh exposure/accepted-job coverage immediately
before real attempts; an older successful audit is not a fresh authorization.
No final-check labels or media are needed by this route.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qwen38_untuned.py tests/test_qwen38_comparison.py -q
```
