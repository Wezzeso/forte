# Isolated public-data training runtime, 18 October 2026

The accepted analyzer/worker and live Qwen model remain running while the
frontend-only API overlay retains the same imported analyzer files, as recorded
in [the production bindings recorded at 23:50 UTC](../../infra/public-training-production-bindings-v2-20261018.json).
This trainer is a separate image and has no production data, credentials,
database, ingress, or model-cache mounts. A successful synthetic memory probe
establishes runtime feasibility only; it does not establish detector accuracy or
authorize promotion into serving.

The image is `forte-specialist-training:rfdetr1112-cu129`, immutable ID
`sha256:2a7af0c822817bda42d8eabba5085b1756ffd051d37f2c3aa3ec138a166462ba`.
It derives from the existing cached `forte-features:cu129-w1`; the exact base ID
is in `infra/public-banknotes-runtime-20261018.json`. Its Dockerfile is
`deploy/Dockerfile.specialist-train`. Building used public package egress with
GPU devices hidden, and verified no CUDA initialization or NVIDIA device nodes.
The image defaults to GPU hidden and offline Hugging Face operation.

The pinned RF-DETR wheel is version 1.11.2, SHA256
`2f23589567123b59302f1cfa9d14d23630f9f583ee6f3468c5891e33fb103b2a`.
Torch 2.13.0+cu129, torchvision 0.28.0+cu129, Transformers 5.17.0 and PEFT 0.21.2
are preserved from the cached foundation. Native Lightning 2.6.6 imports and a
complete `pip check` pass. The isolated image removes unused inherited vLLM,
LMCache and FastAPI Cloud CLI distributions, and aligns the inherited telemetry
family; this does not modify their original images or any running container.
Resolver, freeze and version receipts are in
`infra/public-training-runtime-20261018/` with hashes in the runtime report.

Official Apache-2.0 Nano and Small COCO checkpoints are downloaded separately,
checked against the exact package asset registry's MD5 values, and pinned by
SHA256. Provenance includes their official GCS URLs, bytes, package version and
license source. Remote files are under
`/data/forte-ml/public/rfdetr-training-runtime-20261018/weights/`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `rf-detr-nano.pth` | 366287238 | `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d` |
| `rf-detr-small.pth` | 386045550 | `d81979a9213a2109345158ce9232668df4c1ae52e9b8db3f2ec0a8cbad959b33` |

`training/rfdetr_memory_smoke.py` uses only generated tensors, two boxes and the
final foreground classes `single_banknote` / `id_card`. It validates local
checkpoint digests and the exact upstream API, constructs the model on CPU,
freezes the encoder, disables gradient checkpointing/compilation/EMA, then uses
native Lightning for one bf16 forward, genuine detection loss, backward pass and
AdamW step. It checks finite nonzero head gradients, optimizer moments and a
changed classifier. No real dataset, accuracy measurement or checkpoint output
is involved. Python socket connections and weight redownloads fail closed;
the container network is also disabled.

The host launcher holds
`/data/claude/training/public-20261018/gpu-training.lock` across the entire
container lifetime. Any later detector fit or crop-VLM training must use this
same lock and run sequentially. Admission requires at least 6656 MiB globally
free GPU memory and an idle API queue. The initial allocator cap is 5120 MiB,
with an independent host kill guard below 1536 MiB globally free, above 5632 MiB
observed trainer memory, or after 240 seconds. PyTorch's cap applies to its
caching allocator; CUDA context/library allocations are outside that cap and
are covered by the host measurements. Only the isolated trainer is killed on
failure. Production readiness and exact container/image/start bindings are
checked before and after. The scheduled 11:01:25 +05:00 host stop is unchanged.

The Nano probe passed: one genuine bf16 optimizer step took 15.97 seconds
(18.40 seconds including host supervision), with 455.19 MiB peak PyTorch
allocation and 556 MiB peak reservation. The host observed 950 MiB maximum
trainer GPU memory and at least 6226 MiB globally free during the run; after exit
free memory returned to 7181 MiB. The live queue was empty before and after,
all five ready checks passed, and exact production container/image/start bindings matched.
`infra/public-rfdetr-nano-memory-smoke-20261018.json` contains the complete
receipt. The first identical synthetic probe also passed but its host finalizer
could not read a root-owned mode-600 output file. The recorded repeat uses the
host user's UID/GID, preserving safe atomic report permissions and host access.
Both probe containers are stopped and retained for audit.

This is a minimum viable runtime measurement with two generated boxes. Real
data, augmentation, more targets and evaluation can use additional memory, so
the allocator, host guards and shared lock remain mandatory during the fixed
public-data recipe. Small at 512 pixels also passed one synthetic step in
16.07 seconds (18.41 seconds including host supervision), using 568.02 MiB
allocated / 684 MiB reserved and 1078 MiB maximum observed trainer GPU memory.
Its minimum globally free memory was 6098 MiB, returning to 7181 MiB after exit,
with unchanged production bindings and successful readiness checks. Its receipt
is `infra/public-rfdetr-small-memory-smoke-20261018.json`. These elapsed times
include imports and initialization, so they do not measure steady throughput.
Subsequent fixed public-data training authorization and its held-out
accuracy are recorded separately; neither pretrained checkpoint is enabled in
the live service by this runtime preparation.

The optional crop-VLM model was also staged without GPU initialization:
`Qwen/Qwen3-VL-2B-Instruct` at immutable revision
`89644892e4d85e24eaac8bacfd4f463576704203`. The flat 12-file snapshot is under
`/data/claude/training/public-20261018/qwen3-vl-2b-instruct-89644892` and totals
4,266,648,961 bytes. The model SHA256 is
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`;
the pinned README declares Apache-2.0. The crop runner's own inventory command
passed and produced `/data/claude/training/public-20261018/qwen2b-snapshot.json`,
SHA256 `e7ed200b59c15fc3abb9935b7f765e2258e7e79c9b85a3e642c107c5ad85fee5`.
The public receipt is `infra/public-qwen2b-prestage-20261018.json`.
This download starts neither inference nor fitting. Its later GPU probe and
training must acquire the same shared lock and run after detector GPU work.

The compact runner has a separate clean image,
`sha256:2be4db9629837c6a3c0e252d2a4b45b54bd0a0d25d41e04274dbff0dd2755712`,
derived from the checked RF-DETR image. Its base filesystem layers are verified
against that exact parent. It adds only `bitsandbytes==0.50.2` using the official
Linux wheel SHA256
`55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283`,
with dependency upgrades disabled and a complete successful `pip check`.
Qwen, PEFT and bitsandbytes import successfully with GPU devices hidden and no
CUDA initialization. The image, resolver and freeze receipts are recorded in
`infra/public-crop-training-runtime-20261018.json` and the runtime receipt folder.
Its additional dependency is MIT licensed according to the exact
[PyPI release](https://pypi.org/project/bitsandbytes/0.50.2/).

The actual CPU crop-processor smoke caught a minimum-area rounding bug before
any compact GPU launch. A 2:1 crop with both minimum and maximum area set to
65,536 became a 73,728-pixel grid. The compact wrapper now uses minimum area
1024 while retaining the strict 65,536-pixel ceiling. All four declared image
conditions (original / 64 / 32 / 16 short-side pixels) pass token-grid,
completion-mask and CPU-only assertions. The setup failure, diagnosis and
successful receipt are retained separately in `infra/public-qwen2b-processor-*`.

The subsequent [compact CPU preflight](../../infra/public-crop-cpu-preflight-20261018.json)
also passed with GPU devices hidden and CUDA uninitialized, using smoke-config
SHA-256 `5b43751c330b63bebca7b849f0ff0182b239c9130a184d719056c3c2eb852472`.
It performed no public-data fitting or sealed inference; production bindings
matched before and after. A successful GPU smoke and complete crop fit still
need separate receipts.

The first actual [compact GPU smoke](../../infra/public-crop2b-smoke-failed-20261018.json)
then failed with exit 2 after 34.857 s of host supervision and **zero optimizer
steps**. The host sampled 2,664 MiB maximum trainer memory and at least 4,512 MiB
globally free; production bindings stayed unchanged, readiness passed and queue
was 0/0. The stopped failed attempt is retained, with no test exposure or
completed adapter. A bitsandbytes CUDA-12.8 fallback warning under CUDA 12.9 is
recorded, but the original exception was suppressed, so the cause is not proven.

The later [GPU-hidden compatibility diagnostic](../../infra/public-crop-cpu-compatibility-20261018.json)
passed CPU/meta module-path and shared-library resolution checks with no weight
mount or CUDA initialization. This diagnostic is not a successful GPU model
load or optimizer proof. The [second diagnostic GPU smoke](../../infra/public-crop2b-smoke-v2-failed-20261018.json)
also failed before any optimizer step, now recording the failing Triton
`cuda_utils` shared-object loader. Both attempts remain retained with zero test
exposure.

The [CPU tmpfs execution comparison](../../infra/public-tmpfs-exec-cpu-20261018.json)
reproduced the loader failure under the guard's `/tmp` `noexec` mount; the same
compiled probe loaded successfully with an explicitly executable tmpfs. The
diagnosed execution-policy repair is separate from model/data/recipe selection.
The runtime change permits executable code only in a bounded kernel-cache tmpfs;
other memory, serialization, network and production checks remain in force.

The repaired [third compact GPU smoke](../../infra/public-crop2b-smoke-v3-20261018.json)
**passed** generation and one finite optimizer update. It recorded nonzero
gradient, changed adapter weights and unchanged frozen-parameter hash; the
run SHA-256 is
`6022b4f5975fc94cb7e4cc1315df28542941f5fe644ad3ce3793efb7a7f3f1f6`.
Peak allocator allocation/reservation was 3,636.42/3,970 MiB; the host sampled
4,538 MiB maximum trainer memory and at least 2,638 MiB globally free. Its full
guard passed with unchanged production/inputs, readiness true and queue 0/0.
Zero test exposure was recorded. This proves one-step GPU feasibility, not
completed full training, selected-adapter quality or bank accuracy; the full
80-step fit subsequently completed as recorded below. The
two earlier failures and their diagnostics remain linked above.

The [actual compact fit launch](../../infra/public-crop2b-fit-launch-20261018.json)
binds 80 predeclared optimizer steps to configuration
`d029457071b32e13fb71a0563fdbd1e40f124dfbeeea6a03ccbd960e34625f64`
and diagnostic source manifest
`779e2f56f8ab575737b4113577bdb5ed9fe293f05cce73bc1d2cc3894120e137`.
It completed under the same shared lock. The
[frozen compact receipt](../../infra/public-crop2b-fit-frozen-20261018.json)
records 80 finite optimizer updates, nonzero gradients, changed adapter weights
and unchanged frozen weights. Fit wall time was **112.542 s**, complete command
elapsed **197.716 s**, and host elapsed **210.733 s**. The full guard passed with
unchanged production/inputs, readiness true, queue 0/0 and no test exposure.
Host sampled peak trainer memory was **4,558 MiB** and minimum global free memory
**2,618 MiB**; allocator peak was **3,636.42 MiB allocated / 3,990 MiB reserved**.

The saved adapter file SHA-256 is
`a3733ccf3c888224c2887dd24900e45d2462e9481299ed71cec82518cfc6c7ad`,
run SHA-256
`aaec63508133bf227170be43e10f426d6804cd09c804fdc58b4e913e33e0ffcd`.
Development macro accuracy increased 92.1875% to 96.09375% over 32 known-box
positive crops reused in four resolution conditions. At 16 pixels, ID accuracy
fell 14/16 to 13/16; no per-class or bank-video perfection follows. Adapter
preprocessing/generation measured 3.89–4.12 crops/s. Sealed evaluation stayed
deferred until both artifacts froze; the later final results are below.

The [CPU cold-load receipt](../../infra/public-crop2b-cold-load-20261018.json)
records successful loading of the official unquantized BF16 CPU base and saved
PEFT adapter in 30.468 s. All 224 adapter keys/shapes/F32 values and 1,605,632
parameters matched; all parameters/buffers were CPU-only/frozen, CUDA remained
uninitialized and inputs were unchanged. No dataset/processor/inference or test
was accessed. This proves artifact binding, not NF4 GPU numerical equivalence.
Earlier operator-entrypoint and pre-load PEFT suffix-check failures remain in
the same receipt; the helper repair did not modify the fit/config/adapter.

The exact compact 61-file `repo-v3` fitting source has its own
[source archive](../../dist/forte-compact-training-frozen-source-20261018.tar.gz),
[manifest](../../dist/forte-compact-training-frozen-source-20261018.source-manifest.json)
and [member verification](../../dist/forte-compact-training-frozen-source-20261018.verification.json).
Its archive SHA-256 is
`fcc6822e69b70555c194850f483d47e60fc04b6abc5c6e50039e43485a8416d3`;
the separate historical RF `repo-v2` archive is not the compact source identity.

Actual train/development jobs use `tools/run_public_training_host.py`, not the
one-step probe launcher. It performs a GPU-hidden preflight, verifies the entire
frozen source closure and model/config pins, and supervises the whole container
lifetime with memory guards and an absolute cutoff no later than 05:45 UTC.
An exact run parent and the root-authorized test registry are its only writable
host mounts. The dataset, source, config and model inputs keep identical absolute
paths inside the container and on the host. Streamed public-data transfers use
`tools/receive_public_training_bundle.py`: every expected file must match the
separately pinned dataset lock; links, directory headers, extra files and unsafe
paths are rejected. These tools passed 25 and 14 Linux contract tests,
respectively. No test evaluation is part of the fit phase.

## Actual public-data fit: complete

The full public-data transfer passed independent receiver verification:
**11,630 files / 1,253,289,866 bytes**, bound to combined lock
`eef3f1ad5fabb9cb720e42522265a6946a02196af01221af6ebf0522bf498fb8`.
The sender reported a tar-option ordering warning; the receiver separately
accepted every expected payload by hash. [Transfer receipt](../../infra/public-training-dataset-transfer-20261018.json).

The actual RF-DETR Small/512 source is the immutable `repo-v2` closure:
manifest SHA-256 `37632be61eda7a8b81ac63617c89d06b484343e90bc7fb3aebd7ac1d04b1cf1a`,
archive SHA-256 `c1287008592a539d384568a8ddfb257c6c9ffb10ca4a29a806c57e1ed17c665d`,
and launched configuration file SHA-256
`4287852e1bf43d1c9865e2fb86416602436d3926dd6aaf9c93383865cb67e45e`.
[Source/config transfer](../../infra/public-training-source-v2-transfer-20261018.json).
This training closure is separate from production analyzer source.
The runner's saved effective `output/config.json` has SHA-256
`194c888c1c957c39942d4d59a03e206c3dd367332cf7e64d59cee5f97ca14e16`;
the launched and saved files parse to the same configuration. The byte difference
is serialization, not a recipe change. Development/cohort/final records use the
saved effective identity while retaining the original launch-file pin.
[Config-equivalence receipt](../../infra/public-rf-small512-config-equivalence-20261018.json).
The exact 61-file fit closure is preserved in the
[frozen source archive](../../dist/forte-public-training-frozen-source-20261018.tar.gz),
with its [source manifest](../../dist/forte-public-training-frozen-source-20261018.source-manifest.json)
and [member-by-member verification](../../dist/forte-public-training-frozen-source-20261018.verification.json).
It contains source only, without private media or model weights; later operator
tools can be packaged separately without changing the fitted source identity.

The native fit **completed with exit 0**: 16 epochs (last zero-based index 15),
**2,752 optimizer updates / 22,016 physical batches** and finite recorded losses.
Fit wall time was **2,106.128 s**, including repeated development validation;
whole-run elapsed was 2,145.755 s. Trainable-parameter hashes changed and the
frozen-parameter hash stayed unchanged. The best development checkpoint is
**127,421,695 bytes**, SHA-256
`f01c7b9c897f6eee5a519a72d9dbe5d9f7344ebec4bf9e0187dcd44b683f848e`,
and run SHA-256 is
`662def0d466404ad9efa1e828ddcda372915cf9487733add0c9c881562307d2b`.
[Completed fit/frozen receipt](../../infra/public-rf-small512-fit-frozen-20261018.json).
Its [verified local preservation](../../infra/public-rf-small512-preservation-20261018.json)
retains the same selected checkpoint and remote originals; an initial transport
path failure before payload transfer remains recorded alongside the successful
corrected copy.

Native best-development mAP at IoU 0.50:0.95 was **0.929119**; the last completed
epoch measured **0.900783**. The initial newly initialized target head measured
0.062256 and epoch index 0 measured 0.883368. These fitting values are distinct
from the selected-checkpoint rehearsal and fixed-threshold metrics below, and are not
sealed or bank-video results. The run records zero test evaluations.
[Baseline development](../../infra/public-rf-small512-baseline-dev-20261018.json),
[first-epoch progress](../../infra/public-rf-small512-first-optimizer-20261018.json).

Peak PyTorch allocation/reservation was **681.67 / 790 MiB**. The host sampled
**1,360 MiB maximum trainer GPU memory** and **5,816 MiB minimum global free
memory**. These cover fitting and recurring validation, not steady inference
throughput. Both fits, CPU adapter reload and the later single frozen sealed
attempts completed; execution guards do not establish quality qualification.

The separate [selected-checkpoint development rehearsal](../../infra/public-rf-small512-selected-dev-20261018.json)
completed all 905 development images with a full **passing** host guard,
unchanged production bindings, readiness true, queue 0/0 and zero test exposure.
Native mAP50:95 was 0.929119. At fixed confidence/IoU 0.5/0.5, banknotes scored
287 TP / 0 FP / 4 FN, ID cards 16 TP / 0 FP / 0 FN, and declared coin-background
false-positive images were 0/598. These are public-annotation metrics rather than
bank verdicts.

The native phase took 26.217 s (34.52 images/s); fixed prediction/scoring took
14.962 s (60.49 images/s). Timings include their phase I/O, preprocessing,
scoring and setup; they are not GPU-kernel or whole-video throughput. Peak
allocator allocation/reservation was 310.52/342 MiB; the host sampled 876 MiB
maximum evaluation GPU memory. These development measurements use the same
frozen checkpoint and saved effective config without changing selection or
thresholds. They do not consume the joint sealed attempt.

The strict host report is **`passed=false`** solely for
`production_postcheck_failed`: an independent frontend replacement changed the
API container binding during the detector fit. Native training exited normally
and immutable inputs stayed unchanged. A separate
[post-fit source/binding verification](../../infra/public-rf-small512-postfit-bindings-20261018.json)
confirmed all imported analyzer files match the accepted backend, protected
worker/Qwen/database/ingress bindings unchanged, readiness true, queue 0/0 and
7,181 MiB GPU memory free. This does not convert the strict guard failure into a
pass. Root authorized continuing from the valid artifact with that failure
retained. Fitting was not repeated; the later registered sealed attempt is a
separate sidecar result rather than a rewritten training run.

The immutable dataset mount includes sealed files for closure verification.
Fit code uses only train/development views with `run_test=false`; a sealed view
is materialized only after the registered claim. This is code-enforced use with
hash/exposure auditing, not OS-denied access to sealed bytes. Both model/protocol
artifacts must freeze before joint cohort registration. No test evaluation is
recorded in the completed run, and no public-only model is activated.

Native development metrics are not held-out or bank-video accuracy. Class and
publisher are confounded, the 879 sealed images have only five combined
template/components, and the crop-VLM's known-box two-class views and their four
degradations reuse the same originals. Detailed support and final-result limits
are retained in [the result ledger](public-training-results-20261018.md).

## Single frozen sealed attempts: complete, quality targets failed

The [cohort](../../infra/public-training-cohort-20261018.json) froze both
participants before either exposure. The [RF final](../../infra/public-rf-small512-final-20261018.json)
completed 879 images, native mAP50:95 0.905428. At confidence/IoU 0.5/0.5, ID
recall was **13/16 (81.25%)**, below the 95% target; banknotes had **six FP**,
278 TP and three FN. Declared negative-image FP was 1/582. Result elapsed was
56.808 s across setup/two passes/scoring, not single-pass inference time;
host guard passed in 93.425 s, sampling 876 MiB peak evaluation memory.

The [compact final](../../infra/public-crop2b-final-20261018.json) evaluated the
saved adapter only on 48 original known-box crops (32 notes / 16 IDs), each
reused in original/64/32/16 conditions. Primary equal-class/equal-condition
macro accuracy was **94.53125%**, below 95%; at 32 pixels IDs scored 12/16 (75%).
Pooled 184/192 view decisions (95.8333%) are a different non-primary statistic,
not 192 independent objects. There was no sealed base-model comparison.
The host guard passed in 116.193 s, sampling 2,704 MiB peak memory.

Both participants consumed exactly one attempt, with frozen identities,
unchanged production/inputs and no reset/test tuning. They remain ineligible
for serving/violation activation; quality failures and bank-domain data gaps
remain separate from successful execution. The ledger and
[methods/benchmark tables](accuracy-methods-benchmarks-20261018.md) retain full
counts, source/class confounding and limited independent support.

## Current live path after training

The [fresh single six-upload public check](../../benchmarks/quality/final-public-six-load-20261018T005948Z.json)
passed 6/6 with zero failures/retries, maximum 142.568 s and median 92.076 s.
It was warm, from the GPU host through public HTTPS; 19.261 s preparation was
excluded. The 240-second operational gate passed and the 30-second target failed.
This is not desktop/cold latency, new bank accuracy or an improvement caused by
the inactive public models.

[Post-load bindings](../../infra/public-training-final-postload-bindings-20261018.json)
confirm accepted analyzer bytes, unchanged protected worker/Qwen/database/ingress,
readiness true and queue 0/0. The [profile](../../infra/public-training-final-profile-binding-20261018.json)
keeps specialists disabled, and the [account stop](../../infra/public-training-final-budget-guard-20261018.json)
remains 11:01:25 Asia/Qyzylorda.

The [versioned candidate release](../../infra/public-trained-candidates-release-20261018.json)
is copied to `/home/ubuntu/forte/deploy/releases/public-specialists-20261018-0ab06fd0`:
[48 files / 135,469,180 bytes verified](../../infra/public-trained-candidates-release-verification-20261018.json),
read-only, with both quality failures explicit. `activation=false`,
`serving_eligible=false`, `promotion_eligible=false`, `approved_types=[]`;
staging performed no model/test inference and changed no live configuration.
Local weight backups and [final sidecars](../../infra/public-training-final-preservation-20261018.json)
are preserved independently of the host stop.
