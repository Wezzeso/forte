# Later genuine serving archives with the frozen quality family

The already executed baseline keeps its original `bca671...` archive, plan,
terminals and family unchanged. A later training or serving implementation can
have a different genuine archive. `qwen38_quality_equivalence` verifies both
actual archive byte hashes and every archived member. It rejects any change to
the frozen common pipeline, prompt/model adapter, sampling, decisions, rules,
taxonomy, base manifests or evaluator/gate dependencies. Every other changed
path retains both hashes and an explicit disclosure. Source-only equivalence
does not establish model quality or permit additional calls.

The narrowly authorized exception is candidate HTTP timeout forwarding in
`src/forte/pipeline.py`. Use `--allow-candidate-timeout-forwarding` only for that
release. The pure lineage helper must prove that removing one isolated
`request_timeout_sec=config["limits"]["request_timeout_seconds"]` keyword line
restores every original byte and the whole original AST. Any other code,
comment, whitespace, default-timeout or protected-file change is rejected.
Such a receipt explicitly reports **87 of 88 common files byte-identical, plus
one audited delta**; it never claims all 88 match. Both public source versions
and the re-executable proof remain in the private receipt. Historical strict
88-file receipts remain unchanged and valid.

After the new immutable archive exists, prepare a new private proof:

```sh
python -m evaluation.qwen38_quality_equivalence prove --family family.frozen.json --evaluator-archive qwen38-ml-final-20261017T144038Z.tar.gz --actual-archive actual-new-source.tar.gz --disclosure "Actual training mask and native structured decoding source changes; common evaluator files unchanged" --output source-equivalence.json
```

Bind each future arm with its genuine config, fresh roles, actual source/image/
base pins, and actual adapter for the tuned arm. The proof has schema
`forte.qwen38-quality-source-equivalence.v1`; `receipt_sha256` covers the complete
receipt excluding that field. No future adapter placeholder is accepted.

```sh
python -m evaluation.qwen38_quality_equivalence bind-arm --family family.frozen.json --spec actual-arm-spec.json --equivalence source-equivalence.json --runtime-config actual-runtime.json --role-config fresh-role-config.json --output actual-arm-plan.json
```

The new plan separates `evaluator_source_archive_sha256` from
`serving_source_archive_sha256`. Its `runtime_bindings.source_archive_sha256`
is the genuine serving archive, checked against real readiness by the unchanged
`evaluation.qwen38_pipeline execute` command. The original strict runtime gate
is used. Source/slot `case_id` aliases preserve the existing preflight format.
Each arm still requires a new exact root release and exposure reservation for
six reused validation sources and 208 maximum generation POSTs. This module
never starts a service, reserves calls, reads video, or makes model calls.

Both 27B runtime profiles must match exactly, including native grammar policy,
pixels, quantization, frame/context limits, generation, packages, image and
actual serving source. The baseline's distinct architecture/processor remains
an explicit comparison condition. Keep the common worker recipe and population
unchanged. Compare original artifacts without rewriting them:

```sh
python -m evaluation.qwen38_quality_equivalence compare --family family.frozen.json --arm-plan original-baseline-plan.json --arm-plan actual-untuned-plan.json --arm-plan actual-tuned-plan.json --terminals original-baseline-terminals.json --terminals actual-untuned-terminals.json --terminals actual-tuned-terminals.json --labels provided_binary_labels.validation.jsonl --metadata comparison-metadata.json --output three-arm-report.json
```

The final report discloses effective client timeouts and that caller/source
variant. It does not make a pure model-only causal comparison with the historical
baseline. Baseline failed/unpublished cases retain their failed status and do
not become measured negative predictions or evidence of zero false positives.

The report retains original plan/terminal hashes, real runtime bindings, both
complete equivalence receipts, failure-inclusive six-source denominators and
all three actual identities. Metadata verification flags have the same meaning
as `QWEN38-OPERATIONS.md` and the original family comparison. Provided workbook
binary labels remain development diagnostics; no event/fact/legal gold or
held-out accuracy is invented. FINAL_CHECK remains unused.
