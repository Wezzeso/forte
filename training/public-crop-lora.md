# Public crop QLoRA experiment

`python -m training.public_crop_lora` trains a separate compact object-crop candidate. It never changes the serving model or enables banking violation types. The only targets are `single_banknote` and `id_card`, derived from publisher object boxes in the frozen combined Kazakhstan banknotes and MIDV-Holo exports. No human review, participant consent, procedure, counting, safe-lock or absent-object labels are invented.

## Pinned model and runtime

The official [Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) declares Apache-2.0. The pinned [configuration](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/config.json) has 28 dense language layers and the `Qwen3VLForConditionalGeneration` architecture. This compact model is a feasible experiment for the memory remaining beside the live 30B model; its memory fit and quality require measurement.

The exact commit is `89644892e4d85e24eaac8bacfd4f463576704203`. All 12 files total 4,266,648,961 bytes; `model.safetensors` is 4,255,140,312 bytes and its official LFS SHA-256 is `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`. Snapshot inventory independently hashes every local file and rejects unknown files, alternate weights, remote code and a missing Apache declaration. Downloads are a separate CPU-only operation controlled by deployment; the runner never downloads.

The existing isolated runtime is pinned to torch 2.13.0+cu129, Transformers 5.17.0, PEFT 0.21.2, Accelerate 1.15.0 and bitsandbytes 0.50.2. [Transformers documents bitsandbytes NF4](https://huggingface.co/docs/transformers/quantization/bitsandbytes) and [PEFT documents quantized adapter training](https://huggingface.co/docs/peft/developer_guides/quantization). The recipe uses NF4 double quantization for language linear layers, BF16 computation, frozen BF16 vision/embeddings, SDPA, rank-4 adapters on language attention Q/K/V/O only, batch size one, four accumulated examples and gradient checkpointing. It avoids the preparation helper's blanket FP32 conversion of the large frozen vision tower. Exact finite gradients, adapter changes and unchanged frozen tensor hashes must be measured.

## Data and frozen evaluation protocol

The public import and merge verifiers must pass before the exact combined lock hash is authorized. This runner authenticates that lock, checks all source/template/scene/content group boundaries, and reads only the selected train/development images during fitting. Sealed images may be omitted from the training mount. Quarantine records, coins without banknote labels and unknown classes never become positive crop examples. Publisher licenses and attributions remain in the imported dataset and must accompany derivative datasets.

The supplied 40 Forte clips do not enter this pipeline. The published denomination/template partitions stay intact; there is no random frame split. A fixed seed orders crop metadata before model results exist. The default sample limits are 256 train, 16 development and 32 sealed crops per class. Training draws alternate classes and cycle each class's available support with deterministic reshuffling: the 80-step, four-example accumulation recipe draws 160 examples per class. Repeated ID crops do not become additional independent examples; receipts report unique crop support and unique drawn crops separately. These are compute limits, not a claim of independent sample counts.

The fixed training curriculum cycles original pixels and Lanczos downsampling to 64, 32 and 16 pixels on the crop's short side. Both classes see every resolution equally. The normal model processor subsequently resizes each image between 1,024 and 65,536 pixels. A minimum equal to the maximum can cause Qwen's aspect-ratio rounding to exceed the cap; the lower minimum avoids that while strict pixel/grid checks remain active. Smaller original crops are never enlarged by the degradation step. No generated detail or super-resolution is used. The same four predeclared conditions are reported separately on development and final evaluation. Derived stress views are correlated views of the same held-out examples, not four independent datasets. This stress test does not reproduce every camera, occlusion, lighting or compression property of banking footage.

Baseline and adapter predictions use greedy generation. Only the two exact output labels are accepted; unknown or malformed output counts as an error. Each condition reports per-class and pooled accuracy. The primary score averages the two class accuracies equally, then averages the four conditions equally, so limited ID support cannot make the larger banknote class dominate. The fixed 80-step recipe uses no test feedback and saves an unselected adapter even if it does not improve development accuracy. Positive two-class crops cannot establish false-alarm rates on clean rooms or banking video verdict accuracy.

## Resource boundaries and commands

Deployment owns the shared GPU lock. RF-DETR work runs first; never run both trainers concurrently. Wrap the entire process in the host's external timeout and retain the existing budget stop. The CLI also starts an absolute-deadline watchdog, including during model loading, generation and native forward calls. Internal per-step checks supplement those watchdogs.

The default PyTorch allocator cap is 5,120 MiB, with at least 1,024 MiB additional global free memory required at initialization. CUDA context and third-party allocations are not fully bounded by the PyTorch allocator; deployment also monitors the process externally. A failed allocation or deadline is a failed experiment, never evidence of model quality. Maximum fitting time is 60 minutes with a separate 15-minute evaluation reserve in the example.

```bash
python -m training.public_crop_lora --inventory-model /path/to/flat/snapshot --manifest-output /path/to/qwen2b-snapshot.json
python -m training.public_crop_lora --config /path/to/memory-smoke.json --preflight
# Deployment applies its shared flock and external timeout to both CUDA commands.
python -m training.public_crop_lora --config /path/to/memory-smoke.json --memory-smoke
python -m training.public_crop_lora --config /path/to/full-fit.json
```

The full-fit config uses a fresh output directory plus `memory_smoke_run` and `expected_memory_smoke_run_sha256` from the successful smoke. It must match the dataset, snapshot, runtime, source hashes, accumulation and memory-sensitive recipe. A smoke exercises greedy generation on both training classes and one real optimizer step with accumulated curriculum crops, checks finite loss/gradients, proves adapter change and frozen-weight stability, and saves a `memory_smoke_passed` receipt. It is not a completed training run or accuracy result, and it cannot prove every possible sample's memory use.

Final evaluation is a separate command and cannot influence fitting:

```bash
python -m training.public_crop_lora --config /path/to/full-fit.json --cohort-descriptor-run-sha256 EXACT_COMPLETED_RUN_SHA256
# Root registers both frozen models using the shared cohort helper, then:
python -m training.public_crop_lora --config /path/to/full-fit.json --final-eval-run-sha256 EXACT_COMPLETED_RUN_SHA256 --cohort-receipt /path/to/shared/cohort.json --expected-cohort-sha256 EXACT_COHORT_SHA256
```

The shared dataset exposure registry blocks repeated evaluation across output directories, including retries after a failed attempt. RF-DETR and compact VLM enter a predeclared joint cohort: both checkpoint, config, source, model asset and evaluation-protocol hashes are frozen before any sealed-test metric is read. Root authorizes the central registry. The registry permits one attempt per fixed participant and its receipts remain immutable; completion metrics are saved separately. The crop sealed report evaluates only the selected adapter. Base-versus-adapter comparisons stay on development. Without a cohort argument, the strict single-exposure fallback prevents a second model from consuming the same inventory.

Receipts distinguish `preflight_only`, `memory_smoke_passed`, `trained_unselected` and failed runs. They include data/model/config hashes, real optimizer steps, before/after parameter hashes, loss and gradient records, fit duration, development stress results, adapter bytes and peak allocator usage. Timed development inference reports crop preprocessing plus generation throughput and CUDA-synchronized generation duration for each condition. These single-crop measurements are separate from whole-video API latency. No artifact is automatically selected for production.
