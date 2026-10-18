# Actual native-clock staging for TRAIN coverage expansion

Completed 17 October 2026 at 14:05:08 UTC / 19:05:08 Asia/Qyzylorda under root's explicit CPU preparation authority. This stage makes no model calls, training updates, quality claims or release decision.

The bounded preparation decoded the 17 previously uncovered TRAIN contents selected by A05's frozen two-window plan. It produced complete original native-PTS ledgers, freshly probed rational source clocks and exact original-frame JPEG identities/hashes for all 34 requested windows. Sources were mounted read-only; file stats and content hashes were checked before and after preparation.

| Measured preparation result | Value |
| --- | ---: |
| TRAIN contents | 17 |
| Selected windows | 34 |
| Native source frames decoded for metadata | 19,070 |
| Original JPEGs encoded in private output | 416 |
| Per-window frame uses | 445 |
| Preparation runtime | 55.319 s |
| Model calls | 0 |
| GPU device nodes present | False |
| Original content hashes unchanged | True |

The separate CPU container used the existing API CPU image `5f635274bd9dd7d8ca0bde5a430332607972031c976b11ec91c63ab1d22277cf`, two CPU cores, a 4-GiB memory bound, no network, no GPU devices, read-only root filesystem and read-only source mount. Its only persistent writes were the new private preparation bundle/output. The container exited and was removed. Existing serving/GPU processes were not changed by this operation.

Runtime versions: Python 3.12.15, PyAV 19.0.1, Pillow 13.0.0, Pydantic 2.14.1. The private receipt also binds the actual loaded video module source hashes.

## Exact bindings

| Artifact | SHA-256 |
| --- | --- |
| A05 teacher-plan file | `1ac7d103fda745867774c6d2c9d23919b25ead5f70aaf5170cbfd41d547867ab` |
| Teacher-plan semantic identity | `276311c6ccfc1a98b56b2d2dc2b2dbdc2690bfc57e271a4c9f8fb52c034daa29` |
| Original attempt-plan file | `deefe47a52e9758a04b0bc5ea50311d42a22554be7e712936ab821e3c592888f` |
| Resolved native attempt plan | `f265c55af42da9d92495da7b3af2116ed38e65d9194bf9869e2c10e960106521` |
| Actual private preparation receipt | `78c33c0fc6ad60a5bedabbff5b7af615932e280d37a46d10240e0b1c52b32c52` |
| Prepared output tree | `7f477278603ce9e8f4ec0e176e8e0e841b78127e828731746771896c36205568` |
| Metadata-only transfer archive | `5fbffce5333178c8e01b64f6bae3d8ce5b12fd37d4c8797ab7a34ba58f5d241e` |

Private remote output: `/data/forte-ml/qwen38-coverage-a05/native-stage/run`. Original JPEG pixels remain there. Only JSON/JSONL metadata was copied to `training/private/qwen38-coverage-a05/native-stage-remote/`; archive identity and every transferred metadata file were verified against the remote tree.

`resolved-attempt-plan.json` has schema `forte.native-resolved-teacher-attempt-plan.v1`. Each window binds its content/partition/index, actual source time base and integer origin, complete native-ledger hash, context/focal intervals, resolved request timestamps, exact native anchor, and selected frame identities with original JPEG hashes. `sources.json` binds the probed assets and the full native JSONL ledgers. No source time was inferred from nominal FPS.

Manifest durations were rounded, so this preparation follows the real teacher runner: it recomputes the selected windows from the freshly probed exact source duration while preserving the original frozen plan hash. The resolved file retains both the prior planned timestamps and actual request timestamps. A later provider must compare its freshly decoded source-frame packets against this resolved ledger before any POST.

The staged JPEG hash identifies the decoder's original-resolution JPEG. The ordinary model adapter verifies that original hash and then recompresses the image for its data URI, so an outbound JPEG hash must not be mislabeled as the original evidence hash.

New helper: `training/prepare_native_ledgers.py`. Generated-source tests: `tests/test_native_ledger_preparation.py`, **2 passed**. Tests cover complete nonzero-origin/B-frame ledgers, exact selected JPEG hashes, unchanged originals and rejection of changed plan pins before any output/source mutation.

A09 still owns the fresh role/exposure checks; root still owns the actual model/GPU execution release. These 34 windows do not establish full-clip visual coverage, independent ground truth, complete negative supervision, actual fine-tuning or acceptance quality.
