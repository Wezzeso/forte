# Gemma 4 trainer: independent implementation review

18 October 2026. Reviewed `training/gemma4_lora_train.py` at source SHA256
`27ff0f63fdda1091d4f6d585d5460352417562c204e803edd41b6e7b0d9a3a3c`.
This is a code and synthetic-metadata review. It does not certify a GPU fit,
checkpoint reload, public-development improvement, bank-video score or release.

## Loss and trainable weights

The collator expands the prompt and complete conversation through the same
native processor, checks the token prefix and image-tensor equality, and masks
the entire prompt and padding. Visual tokens in the answer are rejected. It
requires nonempty completion supervision and refuses sequence truncation.

For a completion containing C tokens, the custom loss requests the final C+1
logits and discards the last. This correctly pairs prediction positions
start-1 through end-1 with the C answer labels. Gemma's forward applies its
normal final-logit softcap before returning these logits. The pinned
[Transformers implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/gemma4/modeling_gemma4.py)
supports this suffix selection.

The exact Gemma 4 31B configuration yields 60 query projections and 50 value
projections. Its ten full-attention layers share key and value projections, so
their value projection is absent. The inventory accounts for that and excludes
the vision tower. These conclusions follow from the pinned
[base configuration](https://huggingface.co/google/gemma-4-31B-it/blob/842da3794eaa0b77d5f08bae87a17459d91ff475/config.json)
and Transformers source; the real runtime must still pass the inventory gate.

All base parameters are frozen before PEFT attachment. The post-attachment gate
permits only language-layer query/value LoRA tensors. It rejects quantization
of the vision tower and puts that tower in evaluation mode. The before/after
digest covers all frozen parameter storage bytes, shapes, names and dtypes.
It does not cover every auxiliary bitsandbytes quantization-state tensor.

## Optimizer, checkpoint and provenance safeguards

Finite loss, finite nonzero gradients, finite clipped gradient norm and finite
updated adapters are required. A completed fit additionally needs at least one
committed optimizer update and changed adapter bytes. The final partial
accumulation is discarded when a deadline or declared compute budget expires.
The deadline is checked again immediately before optimizer commit.

Checkpoints are written to a partial directory and atomically renamed only
after a complete, hashed inventory exists. They retain adapter weights,
optimizer state, Torch CPU/CUDA RNG state, step metrics and cumulative optimizer
loop time. A final checkpoint is retained even when the final update falls
between periodic checkpoint intervals. Resume binds the recipe, dataset,
trainer source, base-weight manifest and processor/model metadata hashes.
Existing final adapters are preserved before resumed output is saved.

The final adapter and processor file inventories are recorded only after saving.
Consumers must verify a permitted terminal run state and those file hashes;
the presence of an adapter directory alone is insufficient.

## Data boundary and offline scope

The fit accepts a pinned public image manifest, closed object targets and one
shared generic question. It does not load the supplied bank-video label file,
video files, free-text model descriptions or inference reports. Source lineage
and content hashes are checked across train/validation; the dataset importer
also verifies its complete frozen public lock before selecting rows. Validation
rows are excluded from the optimizer loop.

Model and processor loaders use local paths, `local_files_only=True` and
`trust_remote_code=False`; Hugging Face offline and telemetry controls are set.
The training container's network isolation remains an operator responsibility.
This code review alone does not establish operating-system egress prevention.

## Verification performed

- Python compilation passed.
- The trainer's seven stdlib synthetic checks passed, covering closed targets,
  split leakage, balanced draws, completion masks and projection selection.
- Five independent synthetic checkpoint cases passed: valid metadata accepted;
  absent receipt, modified optimizer bytes, uninventoried extra file and
  zero-update state rejected.

These checks opened zero image pixels, bank videos, supplied labels or model
completions. Real GPU training, saved-adapter reload and downstream regressions
remain separate measured gates.
