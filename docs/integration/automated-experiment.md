# Automated W1 experiment

The user authorized autonomous supervision and actual model training without manual annotation as a prerequisite. W1 is a separate weak-supervision experiment: organizer workbook labels supervise a video-level multiple-instance objective, and eligible local-model windows supply sparse fact/support masks. Neither source is human-reviewed event ground truth.

A20 independently checked the frozen split, canonical assignments, source/scenario files, workbook source and each role-label manifest. There are 42 source records representing 40 unique contents: **28 train, 6 validation and 6 final-check**. Content aliases stay together. Grouping is machine-provisional; physical scenario independence has not been established. Known exposed scenarios are excluded from final-check. The original human-reviewed gate remains a separate track and does not block W1.

The base configuration is pinned to SHA-256 `f3da125a472425f937468c607790334fb5e8d4c66c357c70641e3e676871fafa`. The private independent receipt is `qa/integration/private/automated-experiment-pin.json`; its public aggregate is `qa/integration/automated-pin-aggregate.json`. Private records, per-content hashes, role labels and caches stay outside the source bundle.

Recheck the metadata gate immediately before execution. A passing preparation check does not make an old exposure snapshot current:

```powershell
.venv\Scripts\python.exe scripts/check_automated_experiment.py --expected-config-sha256 f3da125a472425f937468c607790334fb5e8d4c66c357c70641e3e676871fafa --role train
.venv\Scripts\python.exe scripts/check_automated_experiment.py --expected-config-sha256 f3da125a472425f937468c607790334fb5e8d4c66c357c70641e3e676871fafa --role validation
```

Root released computation on the existing rental, with no additional spending, serialized through encoder staging, CUDA/feature checks, then optimization/checkpoint/validation. The existing stop remains unchanged. This release does not authorize replacing the active Qwen service or selecting the experimental head.

The consumed machine preparation has six summaries, 59 scoped windows with exact prompt/teacher provenance and 255 per-second bins. Unsupported bins remain abstentions. The worklist covers 34 train/validation contents and excludes the six final-check contents. This export is not event-correctness proof.

A17 actually extracted frozen-encoder features from **34 sources / 35,186 native frames in 84.214 seconds**. A complete cache retains every native source frame, raw PTS/timebase identity, original tail, encoder/decoder/crop recipe hashes and immutable tensor/manifest hashes. Binary labels are recording-level bags; a positive video label is never copied onto every chunk or fact. Missing teacher prompt/revision/support provenance masks that silver loss while preserving independent binary MIL training.

The shared private relocation bundle preserves the consumed teacher and original configuration bytes. It includes separate train/validation label and approximate class-agnostic anchor manifests, and excludes the separate final-check role-label/anchor manifests and raw all-record anchor capture. The byte-identical frozen split retains organizer metadata for all roles; that metadata is not an input to training loss or calibration. Registry/coverage freshness is renewed in a new immutable snapshot. `qa/integration/w1-private-metadata-aggregate.json` contains only safe counts and artifact digests.

The frozen active axis has 19 sorted facts and three enabled event types, independently checked against legal catalog/taxonomy hashes. A synthetic CPU integration preserved 97 native frames through an FP16 cache, performed two optimizer steps and reloaded a bound safetensors checkpoint with identical outputs. It uses synthetic features and CPU Torch 2.12, not the actual CUDA stack (`qa/integration/weak-cpu-integration.json`).

Actual W1 trained a temporal head on frozen features: **336 AdamW steps** on 28 recording bags / 28,124 native-frame features. The binary head changed 128 weights and one bias; saved checkpoint/tensor hashes were independently reverified. Numeric parameter changes establish optimizer updates, not that any branch learned useful behavior: shared-trunk updates or AdamW decay can move parameters even without a supervised target. Actual target coverage and masked losses determine supervision. Training took 5.587 seconds using precomputed features; that is not service latency. This is not a Qwen or encoder fine-tune.

Six supplied-label validation contents all completed, but the trained head matched **2/6 labels versus 3/6** for its untrained comparator. W1 was not selected and the active Qwen service was unchanged. A20 binds the owner-verified receipt in `qa/integration/actual-w1-aggregate.json`; independent evaluation reconciliation is separate. Validation uses its organizer-label manifest after checkpoint/threshold settings are frozen. Final-check cannot fit or calibrate the model.

Consumed weak supervision was limited: 28 teacher windows eligible, 31 fully masked, 28 window-fact labels and 408 positive support labels. No frame-fact, negative support or event-boundary labels were available. Missing supervision was not fabricated. Four approximate class-agnostic validation anchor points had 0/4 matches within the predeclared +/-1-second tolerance; those points do not establish onset/duration/legal ground truth.

Repeat checks used two checkpoints x six validation contents x three independent GPU head forwards (36 forwards), with the same precomputed native features. Observed forwards matched bitwise, maximum delta zero. There were no fresh source decoding/encoder passes and no upload-to-result measurement. These checks establish neither universal determinism nor the under-10-second service target.

Machine-based validation supports a scoped experiment, not universal accuracy, human-reviewed legal correctness or millisecond action-boundary accuracy. The default legal catalog and optional experimental catalog have separate vocabularies/hashes. Experimental branches are not injected into W1's frozen 19-fact axis without a new binding and matching temporal/context support.


A separate CPU-only post-run diagnosis used the frozen 28 training bags without additional optimization, threshold tuning or validation/final-label access. Training-set binary matches improved from 13/28 to 23/28, while six-content validation remained 2/6; this is an in-sample fit diagnostic, not improved generalization. The coarse-point BCE worsened, and all performance/selection caveats remain (`qa/integration/actual-w1-train-fit-aggregate.json`).
