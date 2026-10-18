# Adapter topology and separate source lineage

The completed fit records **496 qualified language projection targets**.
PEFT 0.21.2 intentionally minimizes the saved selector to **12 leaf suffixes**.
Direct set equality between those two representations rejected the valid
checkpoint. The original run and adapter configuration are preserved.

The new CPU-only `serving/qwen38_adapter_contract.py` proves equivalence through
the complete topology rather than accepting suffix names alone:

- Exactly 64 language layers; every fourth layer uses full attention, giving
  48 linear-attention and 16 full-attention layers.
- Exactly 496 qualified targets: 240 linear-attention, 64 full-attention and
  192 MLP projections, with pinned input/output dimensions.
- The inventory's exact layer schedule, target records and canonical digest.
- Rank 8, alpha 16, and no auxiliary selectors, rank patterns, bias, DoRA,
  RSLoRA, QALoRA, token adaptation or replicated layers.
- Exactly 992 saved A/B tensor keys, matching shapes, supported floating-point
  storage widths, contiguous nonoverlapping ranges and exact file size.

The safetensors check reads only the header and file size. It imports no Torch,
PEFT or model classes and does not inspect tensor values. Startup's separate
actual model/PEFT load verification remains required.

`deploy/qwen38_bind_runtime.py` and `serving/qwen38_identity.py` use the same
helper. These corrections belong to a **new serving release**. They do not
modify or relabel the original `91263…` source archive, fit receipt or adapter.

The binder now accepts the original fit archive explicitly:

```text
--source-archive <new-serving-archive> --source-sha256 <new-serving-digest>
--training-source-archive <original-91263-archive> --training-source-sha256 <original-digest>
```

`serving/qwen38_source_lineage.py` hashes both actual archives and compares their
member bytes. The original digest must equal the immutable fit's recorded
`code_archive_sha256`. Training, rule and taxonomy files remain identical.
Reviewed validator/binder/helpers, generated serving tests and the explicit
quality-equivalence proof module may differ. Arbitrary serving, evaluation,
grammar or prompt changes fail this lineage gate.

The necessary later timeout correction has one explicit common-source exception:
the candidate branch in `_provider_configuration` forwards its validated
`limits.request_timeout_seconds` into `LocalModelConfig.request_timeout_sec`.
The baseline keeps its 60-second default. The complete pipeline AST must match
the original after removing exactly this added keyword, and removing its single
isolated source line must recover every original byte. Other statements,
keywords, comments or whitespace changes fail. The receipt reports byte-identical
and audited-exception counts separately; quality comparison reports **87 of 88
common files byte-identical plus one audited timeout difference**, rather than
claiming identical caller timeout conditions. The untuned provider already used
the configured budget; the tuned caller previously inherited 60 seconds, below
an observed 69.507-second generated full-schema request. No quality improvement
is inferred from this transport correction.

The trained runtime config retains a distinct `training_source_archive` and
recomputed lineage receipt. Its ordinary `source_archive` continues to identify
the new serving release. API/worker expected identity still reads only config
and returns the existing 15 fields; full `config_sha256` binds the additional
provenance. GPU startup additionally needs the original archive mounted read-only
at `/release/training-source.tar`. The untuned diagnostic config excludes fit
source and lineage fields and keeps its adapter/training identities null.

Generated CPU tests cover missing targets, false layer schedules, vision or
extra tensor keys, wrong ranks/shapes/storage, falsey PEFT selector bypasses,
unchanged and distinct source archives, protected-file changes and archive-pin
relabeling. A separate A02 header-only audit verified the actual completed fit
with this helper, preserving all original artifact hashes. That audit is
artifact evidence, not model-quality or latency evidence.
