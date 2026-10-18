# Original profile: exploratory provided-label diagnostic

This directory preserves the original six-content proposed-train measurement before staged/concise configuration trials. `snapshot-manifest.json` records byte hashes of the original performance artifacts and private input/terminal snapshots; private records remain under `evaluation/private/original-all19/`. No supplied frames, captions or event narratives are in this aggregate diagnostic.

Against the organizer's **provided binary labels**, all six distinct contents joined successfully: three positive and three negative. One of six completed with a validated public payload (16.7%); five failed. Completed-only confusion is TP 0, FP 0, FN 1, TN 0 (denominator 1). Unprocessed cases remain separate: two provided positives and three provided negatives. Failure-inclusive positive recall is 0/3 and correct rate is 0/6. The cutoff-aware diagnostic has the same counts.

These are exploratory development observations on duration-selected proposed-train data. They are **not human-reviewed heldout accuracy**, do not select a checkpoint and authorize no training or split lock. Event timestamps, evidence support and legal grounding remain unreviewed.

`gpu-original-profile-proof.json` is the deployed-profile binding supplied by A18: immutable deployed CPU image plus runtime override evidence. Effective runtime settings must not be inferred from a post-run local default configuration hash. No retrospectively verified live rule-catalog digest is claimed. Future configuration trials must use separate filenames and retain this original evidence.

Reproduce aggregation without any API, GPU, model or media operation:

```powershell
python -m evaluation.exploratory --selection evaluation/private/original-all19/a18-inputs.json --terminals evaluation/private/original-all19/a18-terminal-verdicts.json --output benchmarks/quality/original-all19/provided-label-diagnostic.json
```
