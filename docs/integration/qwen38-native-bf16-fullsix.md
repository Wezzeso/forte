# Native BF16 full-six diagnostic

This is a separate base-versus-FIT diagnostic, frozen before any native private
outputs. It is not an expansion or retry of the HF/NF4 pilots. Those protocols,
control failures, source outcomes, evidence and exposure records remain intact.

Frozen metadata:

- Protocol: `benchmarks/private/qwen38/native-bf16-fullsix-a12/protocol.frozen.json`.
- Raw SHA-256: `2b5e22c90040084f694eab9d0dab675c9efdede8c200cdbf985a461160ef9ea3`.
- Semantic SHA-256: `f7566a354e514f29cbb75803d1ba11fe1d2b2ff30b5b7caa29823a23e3585477`.
- Original six-source population protocol: `850fdb4837d549845104fee7984d44588196965396f4a0e034e72e3c8fcdfd9c`.
- Authoritative root selection policy raw SHA-256: `7dd16f74490bb1d4076d67b6bf99336d976839e32344ff4fec941a71cbf16944`.

`benchmarks.performance.qwen38_native_fullsix` provides metadata creation,
`validate_protocol` and `case_budgets`. It reads no videos or provided labels,
constructs no runtime, and issues no requests. Runtime identity is explicitly
pending. No empty or guessed runtime hashes are inserted.

Both `qwen38_untuned` (native BF16 base) and `qwen38_tuned` (native BF16 FIT)
must receive all six unchanged validation contents, reference pins, nineteen
facts, prompt, rules, evidence, generation controls and source geometry:
8-second windows, 4-second stride, 2 fps, sixteen frames and at most eight
refinements. These are reused development sources, not independent human gold.

| Execution order: original ordinal | Initial windows | Logical-call ceiling | POST-slot ceiling | Diagnostic seconds |
|---|---:|---:|---:|---:|
| 5 | 4 | 12 | 24 | 960 |
| 1 | 18 | 26 | 52 | 1,980 |
| 2 | 5 | 13 | 26 | 1,080 |
| 3 | 5 | 13 | 26 | 1,080 |
| 4 | 5 | 13 | 26 | 1,080 |
| 6 | 19 | 27 | 54 | 2,040 |

Each arm has **six source attempts, 208 reserved POST slots and 8,220 seconds**.
The protocol's `provider_max_attempts=2` is the conservative reservation envelope
used to derive 208, not a claim about a loaded native provider. The existing
native provider's actual one-attempt policy stays one. Actual plans must record
that actual setting and keep it within the envelope; they must not add retries
or requests to consume unused reservations. Every unused, failed or cancelled
slot is reconciled without an exposure refund.

Release all six sources at once per arm, in the fixed order **5, 1, 2, 3, 4, 6**.
An individual source timeout, invalid model semantics, incomplete coverage,
evidence/publication failure, decode error or request failure stays failed and
execution continues to the next source. There is no pilot advancement gate,
source retry, label-agreement gate, source substitution or threshold tuning.
Never turn failure into a clean no-violation result.

Only global authorization expiry/invalidity, the hard rental stop, the full-arm
wall/POST bound, a changed actual runtime/config/profile identity, or confirmed
global runtime process failure may halt the arm. A single failed schema or
semantic result does not establish a global runtime failure. If a global halt
prevents later cases, keep them explicitly unprocessed in the six-case
denominator. The runtime boundary must support this policy before execution;
the protocol alone does not prove that integration.

Root froze a separate admission addendum before any native private output:
`benchmarks/private/qwen38/native-bf16-fullsix-a12/runtime-admission-addendum.frozen.json`.
Its raw SHA-256 is
`ade38909b0dbfec6987ecd6a7e69fd291bef1f7e592e4565dd8d5c9e2a754670`;
its semantic SHA-256 is
`86fb6845a066b0744c11e25a7515cc2495e433685a14a139dbdce979289b714e`.
The original `2b5e22…` protocol bytes and selection rule remain unchanged.

The addendum clarifies a further global invalid-runtime gate: **closed admission
or unknown server drain requires a stop**, even while the process remains alive.
Record `runtime_admission_closed_drain_unknown`; do not claim that the process
died or that an uncertain in-flight request drained. Completed source-local
failures continue only with positive evidence that the same runtime is live,
identity-matched, drained and admission-healthy. A fully finished invalid
semantic response with that healthy gate remains a source-local failure.

There is no automatic admission restoration, restart, model retry or source
retry. Root must stop, verify actual drain and live runtime identity, and issue
a new explicit release before any restoration or further execution. Any new
release must account for all prior source attempts and reserved slots; the
addendum itself grants no reusable IDs or new execution scope. Keep the current
failure and every later unprocessed case in the six-case denominator.

The new `qwen38_native_admission_addendum` helper validates this policy separately
and binds the addendum's raw and semantic digests into the later actual native
plan and root release. Its pure Boolean classification helper is not a runtime
readiness observation. The runner must obtain genuine A10 native gate evidence.

Freeze the selection criterion before native outcomes and apply it without
tuning:

1. Higher failure-inclusive `(TP+TN)/6`.
2. Higher `TP/3`, retaining every provided-positive source in the denominator.
3. Higher defined precision `TP/(TP+FP)`; undefined remains null and ranks below
   a defined value for this tie-break.
4. Higher complete, evidence-valid case count.
5. Lower median genuine pipeline latency only when both arms have all six valid,
   complete cases and six valid timings each; otherwise omit this tie-break.
6. Native BF16 base wins an exact remaining tie.

A case counts correct only with a valid published prediction, actual source and
model binding, full temporal coverage and genuine native evidence receipts.
Failures, missing rows and invalid results count incorrect. Provided binary
agreement does not establish reviewed event, fact-support, boundary or legal
accuracy, and selection grants no automatic promotion.

Later actual binding must use the isolated native `vllm_bf16_native` identity,
provider and profile contracts. HF/NF4 readiness is not interchangeable with
native readiness. Base and FIT must share a genuinely matching native BF16
profile; the actual adapter identity is separately bound. Actual native GPU6
and HTTP gates, source/image/base/config proofs, exact root full-six release,
fresh A09 role preflight and exact 208-slot reservation are all prerequisites.
No native private model request is authorized by this protocol.

A09 requires `now + 8220 seconds <= release expiry <= fixed rental stop`.
Pilot-style one-hour expiry is insufficient. The role snapshot must be fresh
at admission (at most 300 seconds); freshness does not enlarge the execution
expiry. No release clock or reservation has been started by this preparation.

Record actual elapsed times and every observed 285-second production-deadline
miss. A longer successful diagnostic does not meet that production target.
Direct pipeline time makes no upload-to-result or usable-UI latency claim.
The six FINAL_CHECK sources remain untouched and unauthorized.

To reproduce metadata into a new output path:

```powershell
.venv/Scripts/python.exe -m benchmarks.performance.qwen38_native_fullsix --parent-protocol benchmarks/private/qwen38/accuracy-diagnostic-a12/protocol.frozen.json --selection-policy benchmarks/private/qwen38/accuracy-diagnostic-a09/selection-policy.ROOT-CORRECTED-v2.FROZEN-before-verdicts.a09.json --output benchmarks/private/qwen38/native-bf16-fullsix-a12/protocol.new-metadata-only.json
```

The original frozen file is never overwritten. A later timestamp changes the
new file's digest and does not replace the root-frozen protocol.
