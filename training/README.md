# Local LoRA adaptation

This optional path creates a PEFT candidate for A10's validation comparison. The untuned baseline stays selectable. No LoRA training has been executed and no adapter has been selected: reviewed training targets, an A20-locked scenario split, and a reserved training GPU window are still required. A07 has verified the actual 17.55 GB model snapshot, and A15 has established local serving/CUDA; mount that verified snapshot read-only into training. Supplied workbook timestamps alone do not establish the fact/event labels required by this reviewed-gold LoRA track.

The separate [automated W1 temporal-head workflow](temporal/README.md) has executed a real GPU optimizer run: 336 AdamW steps over 28 recordings, organizer binary labels, 15 coarse positive point bags and provenance-bound sparse silver. Its candidate scored 2/6 on the frozen validation set versus the untrained reference's 3/6, so it was not selected. The serving model is unchanged. Private verified artifacts are under `training/runs/W1`; no fact/event accuracy or upload-to-result latency target has been established.

Run from the repository root. Keep training isolated from serving benchmarks. The official [Torch cu129 index](https://download.pytorch.org/whl/cu129/torch/) and [torchvision cu129 index](https://download.pytorch.org/whl/cu129/torchvision/) do not publish `2.14.1+cu129` / `0.29.1+cu129` for Linux/Python 3.12. Their newest compatible entries are `2.13.0+cu129` / `0.28.0+cu129`, matching the actual A100 runtime. The default CUDA 13 image failed on this host's driver 565.57.01; the cu129 variant passed BF16. The dated index proof and selected wheel URLs are in `cu129-index-verification.json`; refresh it with `python -m training.verify_index` without downloading wheels or using a GPU.

Use the isolated training Docker image for E1. It inherits the exact immutable, verified vLLM 0.31.0-cu129 producer image, preserving Torch 2.13.0+cu129, torchvision 0.28.0+cu129, Transformers **5.17.0**, and Pillow **12.3.0**. It adds only PEFT 0.21.2 and accelerate 1.15.0 to a separate venv, using `requirements-addons.lock` SHA256 hashes and `--no-deps`. The producer image digest locks inherited dependencies. This creates a separate image and never upgrades the running serving container. No DeepSpeed, external tracking service, online model API, or custom CUDA build is required.

The root `training` extra now selects the same explicit cu129 Torch wheels and Transformers 5.17.0, while the CPU application's Pillow remains 13.0.0. That root environment can run CPU guards and prepare inputs; it does **not** match the isolated E1 processor lock. `training.runtime_check` rejects package drift before real training. Keep the runtime lock unchanged unless the resulting training/serving compatibility has been explicitly revalidated.

Build and run CPU-only checks without allocating the GPU:

```sh
docker build --file training/Dockerfile --tag forte-training:cu129 .
docker run --rm --network none --entrypoint python forte-training:cu129 -m unittest training.test_training training.test_alignment -v
docker run --rm --network none --entrypoint python forte-training:cu129 -m training.runtime_check
```

The image is prepared, not yet built/tested. Its Dockerfile-specific ignore file permits only source/public manifests and excludes private data, environments and weights. Builds check package metadata without initializing CUDA. Build outside serving benchmark windows to avoid CPU/disk interference. After A15/A18 release the GPU or a separate training allocation is granted, verify its actual CUDA operation:

```sh
docker run --rm --gpus device=0 --network none --entrypoint python forte-training:cu129 -m training.runtime_check --gpu
```

Mount the approved private training directory and model snapshot read-only, plus an empty output parent read-write. Set paths in the private config to `/models/baseline`, `/private/...`, and `/runs/E1`; the image's model manifest and runtime lock stay under `/app`. Then validate inputs before launching training:

```sh
docker run --rm --network none \
  --mount type=bind,src=/absolute/private-training,dst=/private,readonly \
  --mount type=bind,src=/absolute/verified-model-snapshot,dst=/models/baseline,readonly \
  --mount type=bind,src=/absolute/training-runs,dst=/runs \
  forte-training:cu129 --config /private/e1.local.json --preflight

docker run --rm --gpus device=0 --network none \
  --mount type=bind,src=/absolute/private-training,dst=/private,readonly \
  --mount type=bind,src=/absolute/verified-model-snapshot,dst=/models/baseline,readonly \
  --mount type=bind,src=/absolute/training-runs,dst=/runs \
  forte-training:cu129 --config /private/e1.local.json
```

For CPU development and synthetic processor checks in an explicitly prepared local environment:

```sh
uv sync --extra training --python 3.12
uv run --no-sync python -m unittest training.test_training training.test_alignment -v
uv run --no-sync python -m training.check_processor --processor-dir /private/models/Qwen3-VL-8B-Instruct --media-format image_sequence
uv run --no-sync python -m training.check_processor --processor-dir /private/models/Qwen3-VL-8B-Instruct --media-format video
uv run --no-sync python -m training.check_gradient --processor-dir /private/models/Qwen3-VL-8B-Instruct
uv run --no-sync python -m training.train --config training/e1.local.json --preflight
```

Copy `e1.example.json` to a local configuration and set paths before running. The model and processor must be the same local snapshot, pinned to revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. A07's `verify_local_snapshot` verifies all 15 declared artifacts by size/SHA256. Runtime loading uses `local_files_only=True`, `trust_remote_code=False`, HF offline mode, and SDPA. Pre-stage dependencies and weights separately before disabling egress.

E1 uses the same system prompt, alternating frame-ID/timestamp text and images, JPEG sanitation, and pixel budget as A07's image-sequence serving path. Training requires A08's **current** prompt/config versions; the compact-JSON prompt is `forte.visual-observation.v1.0.1` at this handoff. Build each approved prompt with the live `build_prompt`, never an old literal. Compact frame IDs remain disabled in the current decision default. The configured `decision_config` file is hashed. If its `compact_frame_ids` flag is selected later, the matching training formatter described below is applied automatically. Native video formatting is supported by the synthetic processor check, with explicit metadata derived from source PTS in a millisecond timebase. Native video training is gated off while serving uses image sequences. Successful synthetic formatting is not model-quality evidence.

The collator processes the identical prompt twice to verify its token prefix after multimodal expansion. Only assistant completion tokens receive loss; prompt and visual tokens are masked. Padding/packing and silent truncation are disabled. A window exceeding the token budget fails; review and reduce its approved frame budget explicitly. One window forms a microbatch, with gradient accumulation. All base parameters and the vision encoder remain frozen. Only text self-attention q/v LoRA parameters train, and the loop checks finite nonzero gradients and absence of vision gradients before saving.

## Required private inputs

`annotations` points to A03's human-reviewed JSONL export. `source_manifest` points to A02's recording JSONL, and `scenario_groups` to the reviewed scenario-group file. `split_manifest` must carry `state: locked`, `locked: true`, `training_authorized: true`, `scenario_groups_reviewed: true`, `approved_by`, `locked_at`, and actual file hashes. `records_sha256` uses A10's canonical digest: compact sorted-key UTF-8 JSON, records sorted by sample/recording ID. All train/validation/final-check groups and exact-content identities must remain separate.

The entry point calls A02's importable `scripts.split_gate.validate_locked_training_inputs` with latest exposure-chain and fresh complete accepted-job inventory coverage. Configure `exposure_registry`, `exposure_inventory_status`, and an **independently A20-approved** `expected_assignments_sha256`; never populate that pin by trusting a replacement split file. The sample intentionally leaves the pin null. The helper checks the approved exposure-snapshot prefix/content set, excludes exposed scenarios from final-check, and restricts requested training contents/recordings to the fixed approved train assignment. A moved private inventory file can use `source_inventory_path_override`; its exact saved checksum must still match. Exposure is never training authority. Missing/stale inventory or an altered assignment fails closed.

`train_manifest` is a separately approved JSON object, stored under ignored `training/private/`:

```json
{
  "schema_version": "forte.training.v1",
  "split_manifest_sha256": "SHA256_OF_LOCKED_SPLIT_FILE",
  "annotations_sha256": "SHA256_OF_REVIEWED_JSONL_FILE",
  "approved_by": "LOCAL_REVIEWER_REFERENCE",
  "records": []
}
```

An empty record list intentionally fails. Each approved window contains:

- `sample_id`: unique opaque window ID; `source_recording_id`: original A02 recording ID.
- `content_sha256`, `group_id`, `split: train`, `data_origin: development`; derived windows inherit their source assignment.
- `annotation_record_sha256`: canonical hash of its exact A03 review record; `review_status: human_reviewed`, `reviewer_id`, and `reviewed_at` attest review of the window's training target.
- `start_sec`, `end_sec`, `negative_verified`, and 2–16 ordered frames with `path`, `pts_sec`, `frame_id`, `sha256`. Every frame ID/PTS must occur in its source review. Paths resolve inside `media_root`; frame checksums must match.
- `prompt`, `prompt_version`, `config_version` from A08's `build_prompt`; `requested_fact_ids`, `reviewed_event_ids`; `target` in A08's exact raw `facts`/`candidates` decision schema. Use generic definitions and opaque frame IDs. Keep source names, binary labels, and content hashes out of prompts.

## Optional compact frame IDs

Keep original frame IDs in reviewed annotations and the approved canonical training manifest. The default formatting is unchanged. To match an explicitly selected compact-ID serving configuration, set `compact_frame_ids: true` in the training run's `decision_config`. Each canonical window must additionally contain `clip_request` (the complete frozen A01 `ClipRequest`, including actual generic fact descriptions and source frame metadata) and `event_definitions` (the generic reviewed taxonomy definitions used for its prompt, including an empty object for facts-only windows). Its stored canonical prompt must match A08's current `build_prompt` with the same limits and `compact_frame_ids: false`.

`training.alignment.provider_record` makes a private copy, calls A08's `compact_frame_request` exactly once, and regenerates the model prompt on the aliased request. It translates frame IDs in input frames, fact supports, and candidate start/end/support fields; fact IDs, event IDs and notes stay unchanged. The alias target is validated and restored through A08's `restore_frame_ids`, then compared with the original canonical facts/candidates. Missing or mismatched frozen request/prompt data fails. The canonical manifest and review records remain unchanged. Run metadata records the selected flag and an aggregate alignment hash.

An optional CPU-only export makes these private model inputs locally reviewable before allocating a GPU:

```sh
python -m training.export_provider --config training/e1.local.json --output training/private/provider-inputs.json
```

This command first requires the same approved training records, human reviews and locked split checks. Its output must be a new file inside the canonical training manifest's private directory. It references the immutable canonical manifest and record hashes; it never replaces that manifest or prints prompts/targets. The training entry point always reads canonical records and applies the selected mapping once; do not use the provider export as a new canonical training manifest.

## Optional concise observations

The same private formatter forwards `DecisionConfig.concise_observations` with its default `false` unchanged. When explicitly selected, it rebuilds A08's distinct concise prompt/schema and sets only optional fact `notes` to null in the copied provider target. Original local notes remain in the canonical reviewed manifest. Every fact ID, state, supporting frame reference, candidate, event type and timestamp is retained. Concise formatting can combine with compact frame IDs; the canonical restoration check ignores only the deliberately omitted optional fact-note prose.

Candidate visibility limitations are preserved exactly if already compliant with A08's limit: at most one note of at most 160 characters. Longer or multiple notes fail with `concise_training_visibility_requires_review`; the formatter does not truncate or discard a limitation. A human must review a compliant target before export/training. Both mode flags and the mode-specific prompt hash/version are recorded. Frozen `clip_request` and generic `event_definitions` are required for either optional formatting mode.

E1 remains a **full 19-fact observation** experiment. `expected_fact_count` is fixed at 19 in its configuration and checked before provider export/training. The optional four-fact screening route is not a trained target, and this pipeline does not derive reduced targets from it. Human-reviewed target and locked-split requirements apply in every mode.

Targets require explicit local human review. This pipeline never manufactures them from timestamps, normalizes provisional local-teacher output into ground truth, or marks unannotated intervals negative. A negative window must fit within an A03 interval with `negative_verified`, `context_complete`, and clear visibility. Positive windows need an overlapping reviewed event of the target type. Source records with unresolved items fail conservatively. Jury, final-check, validation, unreviewed, and unknown-origin examples are rejected.

## Artifacts and selection

Each run uses a fresh output directory. `run.json` stores config/dataset/split/source/scenario/model-manifest hashes, exact package versions, seed, model/processor revisions, GPU identity, parameter counts, gradient verification, adapter file checksums, and status. `config.json` and aggregate `metrics.jsonl` remain local. No per-video predictions or annotations are printed. Failures print a sanitized code.

A completed run has `status: candidate_unselected`, `selection: unselected`, and `serving_load_verified: false`. The adapter uses ordinary `adapter_config.json` and `adapter_model.safetensors`; it is re-read with PEFT before handoff. A15 must load it against the pinned baseline, run a real private request, and report whether serving load works. A10 compares baseline/candidate on the same validation input and prompt conditions. Training loss never selects a checkpoint. Before any E2, supply a hashed A10 validation analysis with `justifies_e2: true`, `baseline_comparison_completed: true`, `selection_split: validation`. Stop new training work at the plan's hour-11 gate.

The serving handoff is an explicit selected adapter path plus `run.json` and A10/A20's selection record. vLLM may load the selected adapter via `--enable-lora --lora-modules selected=/private/path/adapter`; the baseline model alias remains available. Neither file existence nor a synthetic check proves a real adapter loaded in serving.

## Verification status

Thirty-four synthetic CPU checks passed: split/review/masking/runtime guards, full-19-fact versus screening scope, and compact/concise alignment checks covering unchanged defaults, exact states/references/candidates, source immutability, authoritative mapping/restoration, prompt versions, and refusal to drop visibility limitations. The example preflight correctly returned `required_input_missing`.

Actual Transformers 5.20.1 processor checks also passed using 11 SHA-verified public processor files (11,574,852 bytes) from the pinned model revision, with no model-weight download: image sequence 407 input / 26 completion tokens / 4 visual segments; native video 77 input / 26 completion tokens / 1 segment. These are synthetic fixtures, never private recordings.

`check_gradient` passed a real forward/backward, one optimizer step, and PEFT save/load/forward roundtrip on a tiny **randomly initialized** Qwen3-VL with synthetic frames: 448 trainable LoRA parameters, frozen vision encoder. It remaps tokenizer IDs into a 128-token synthetic vocabulary to stay small. This verifies the current Transformers/PEFT APIs and adapter gradients; it provides no evidence about the actual 8B model's accuracy, memory, or CUDA performance. These ML smoke checks ran in an isolated ignored environment with Python 3.14, torch 2.12.0 (CPU), torchvision 0.27.0, Transformers 5.20.1, PEFT 0.21.2, and accelerate 1.15.0. The requested Linux/CUDA pins remain untested.

After the serving compatibility handoff, the actual image-sequence processor and tiny random-model gradient/save/load checks also passed with Transformers **5.17.0** and Pillow **12.3.0** on the same local CPU Torch 2.12.0 test environment. This establishes those APIs work together; it is not a CUDA/training-image build claim.

Actual 8B training, validation comparison, and serving adapter load remain blocked by human-reviewed targets, a locked scenario split, and reserved training GPU access. The isolated training image still needs build/metadata/CUDA checks. Baseline retention is the current recommendation.

Primary references: [Qwen training guide](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md), [Transformers Qwen3-VL](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl), [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora). Current package metadata: [Transformers](https://pypi.org/project/transformers/5.20.1/), [PEFT](https://pypi.org/project/peft/0.21.2/), [PyTorch](https://pypi.org/project/torch/2.14.1/). Upstream availability does not establish runtime compatibility on the eventual GPU.
