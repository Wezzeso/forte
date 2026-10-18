# Specialist crop QA LoRA

`python -m training.specialist_lora` is a separate, offline training entry point for
`Qwen/Qwen3-VL-30B-A3B-Instruct`. It supports the model's actual
`qwen3_vl_moe` / `Qwen3VLMoeForConditionalGeneration` architecture. The existing
`training.train` baseline loader remains separate. Qwen3.6, Qwen3.8, text-only
Qwen models, quantized snapshots, and arbitrary architecture substitutions are
unsupported here. No quality improvement or completed GPU run is claimed by the
recipe or its synthetic guard tests.

The model and processor load through Transformers `AutoModelForImageTextToText`
and `AutoProcessor` with `local_files_only=True`, `trust_remote_code=False` and
offline environment flags. There are no downloads, hosted teacher calls, audio
inputs, external experiment logging, or automatic production selection. Use an
already provisioned, exclusive GPU training window. BF16 holds the full model in
GPU memory: 30B-A3B describes active expert parameters, not the memory footprint
of a 3B model. The example's 72 GiB free-memory floor is an allocation guard, not
a promise that every 80 GB GPU configuration fits. Reduce visual tokens or
sequence length if the measured training peak requires it; truncated answers are
never silently accepted.

## Prepare and review inputs

First prepare the reviewed source-frame dataset through
`python -m training.specialists.prepare`, as described in
[specialists/README.md](specialists/README.md). Preserve its dataset-lock hash.
Organizer clips and the held-out camera remain evaluation-only. Crops use the
locked source frame directly; they cannot point at a separate image or URL.

Create a private crop manifest with the following shape. Replace every example
value with real reviewed data. Include at least one QA row from each of the
`train` and `validation` partitions; the example below shows one row only.

```json
{
  "schema_version": "forte.crop-lora.data.v1",
  "frozen": true,
  "reviewed_by": "human-reviewer",
  "reviewed_at": "2026-10-18T07:00:00+05:00",
  "dataset_lock_sha256": "<exact dataset-lock.json SHA-256>",
  "records": [{
    "example_id": "own-take-001-id-qa",
    "source_sample_id": "<sample_id from dataset-lock records>",
    "source_recording_sha256": "<same exact source recording hash>",
    "scenario_group": "<same scenario group>",
    "camera_id": "<same camera ID>",
    "split": "train",
    "origin": "own",
    "crop_xyxy": [80, 160, 250, 350],
    "question": "Что держит человек: удостоверение личности или банкноты?",
    "answer": "Удостоверение личности.",
    "human_reviewed": true,
    "reviewer": "human-reviewer",
    "reviewed_at": "2026-10-18T07:00:00+05:00"
  }]
}
```

`source_sample_id` links to the detector dataset's `sample_id`. The recording
hash, group, camera, origin and split must match that source exactly. Every
source must retain verified exhaustive human labels and provenance. Only own or
reviewed synthetic training/validation sources are accepted. Never relabel a
holdout or organizer source to make it trainable. Validation questions do not
contribute gradients. Their loss is reported, not used as a quality gate.

Create an operator-reviewed snapshot inventory **outside** the model directory:

```json
{
  "schema_version": "forte.crop-lora.snapshot.v1",
  "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
  "revision": "<exact 40-character upstream commit>",
  "processor_revision": "<same exact commit>",
  "license": "apache-2.0",
  "license_verified": true,
  "reviewed_by": "human-model-reviewer",
  "reviewed_at": "2026-10-18T07:00:00+05:00",
  "files": {
    "config.json": "<SHA-256>",
    "tokenizer_config.json": "<SHA-256>",
    "tokenizer.json": "<SHA-256>",
    "preprocessor_config.json": "<SHA-256>",
    "chat_template.json": "<SHA-256>",
    "model.safetensors.index.json": "<SHA-256>",
    "model-00001-of-00013.safetensors": "<SHA-256 of this shard>"
  }
}
```

Inventory **every** immediate model file, including all shards, vocabulary,
generation/video processor configuration and license/readme files if present.
The shortened example omits shards and is deliberately incomplete. The loader
rejects extra unverified files, missing shards, architecture mismatches and
changed bytes. HF-cache blob symlinks are supported. This verifies locally
reviewed identity/revision declarations and file hashes; it is not an upstream
signature or a license review performed by the code.

Copy [../configs/training/specialist-lora.example.json](../configs/training/specialist-lora.example.json)
to a private config, set the paths and exact hashes, and keep the pinned runtime
versions. Paths in a command-line config resolve relative to that config file.
No packages are installed by the runner.

## Execute

```bash
python -m training.specialist_lora --config /private/crop-lora.json --preflight
python -m training.specialist_lora --config /private/crop-lora.json
```

Preflight verifies model/data bytes, QA review, source lineage and parameters
without importing torch, Transformers or PEFT, allocating a GPU, or creating a
run. It may read the entire model snapshot to verify hashes. Errors return a
sanitized JSON code and exit status 2. The placeholder example fails closed.

The training command uses one crop per microbatch and accumulation as configured.
It inventories all 48 language layers' attention `q_proj`, `k_proj`, `v_proj` and
`o_proj` modules, then adapts only those exact module paths. Expert MLPs, routers,
vision layers, embeddings and the language output head stay frozen. The
collator renders prompt and complete chat with the same pixels, checks the exact
token prefix after visual expansion, and masks prompt/visual/padding positions
from the completion loss. Empty, misaligned and oversized answers fail.
The prompt and complete conversation must have identical processed visual
tensors. Runtime checks verify the spatial pixel cap, patch grid, patch-vector
width and expanded image-token count, including resize rounding.

The runner records finite CUDA losses, nonzero finite gradients, optimizer-step
counts, and the number of adapter tensors that actually changed. It saves an
adapter, its processor, hashes and `step-proof.json`, and rechecks all pinned
inputs. A failed or interrupted run never receives `trained_unselected` status.
These receipts substantiate fitting only. They do not prove 95% accuracy or
compatibility with the serving pipeline.

`merge_adapter=true` explicitly enables a safe PEFT merge into an additional
local full-model snapshot. It requires 1.2 times the base weight bytes free on
disk and a 2 GiB CUDA margin, and may still fail under hardware constraints.
Merge leaves the candidate unselected. Evaluate crop probes (including ID vs.
banknotes), scene regressions, held-out camera recordings, grouped organizer
evaluation and API load separately before any reviewed serving change. A merged
snapshot is an export for subsequent vLLM compatibility/quality testing; this
runner never changes production configuration.

Architecture and target names were checked against the primary
[Qwen model configuration](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/main/config.json),
[Transformers Qwen3-VL-MoE documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl_moe)
and [Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py).
The local immutable revision and pinned installed runtime remain authoritative
for each actual run.
