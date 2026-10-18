# Latency feasibility and bounded accuracy diagnostic

Do not spend all six validation sources on the current speed assumptions.
First separate the frozen 285-second speed benchmark from a newly versioned,
explicitly longer accuracy diagnostic. No plan, runtime, source or deadline has
been changed by this analysis, and no model request was submitted.

The verified generated-only receipt has SHA-256
`02ea4e17e89ec37aaf2485c260d284d42e81b1912bffeed624ca93611581cbfa`:
one old-91263 untuned request, sixteen generated frames, nineteen validated
facts, 5,848 prompt tokens and 561 generated tokens. Its full probe wall clock
was **69.55404964 s**, provider elapsed time **69.507114998 s**, HTTP time
**69.464927564 s**, preprocessing **0.033613341 s**, and readiness
**0.007123574 s**. Server stage metrics are absent. These numbers do not isolate
prefill, recurrent decoding, grammar masking or pure token throughput. They
measure no real-video truth, legal correctness or clip latency.

A06 confirms that the full nineteen-fact schema exactly matched the precompiled
public schema. Its grammar lookup was already a cache hit. The separately
observed startup public compilation took 6.416 seconds; it does not explain the
69-second request. Small startup warmups and a 183-second startup do not
establish steady full-task performance.

## Frozen geometry and conditional arithmetic

All six sources have complete 8-second windows with an end-aligned tail, 4-second
stride and sixteen frames per window. Concurrency is one, screening is disabled,
and up to eight additional refinement requests may be necessary.

The following numbers assume every initial window costs exactly the single
generated probe's 69.55404964 seconds. This is a sensitivity calculation, not a
measured video result, a bound on future calls, or deterministic impossibility.

| Original case ordinal | Duration | Initial windows | Conditional initial model time | Maximum average/window with 15 s reserved from 285 s |
|---|---:|---:|---:|---:|
| 1 | 72.7 s | 18 | 1,251.973 s | 15.000 s |
| 2 | 20.866667 s | 5 | 347.770 s | 54.000 s |
| 3 | 23.8 s | 5 | 347.770 s | 54.000 s |
| 4 | 22.6 s | 5 | 347.770 s | 54.000 s |
| 5 | 17.3 s | 4 | 278.216 s | 67.500 s |
| 6 | 78.733021 s | 19 | 1,321.527 s | 14.211 s |

There are **56 initial requests**. Under that assumption, five cases exceed
285 seconds on initial inference alone. The four-window case leaves only
6.784 seconds for decoding, rules, evidence, publication and any refinement.
With all eight refinements, the longest case needs average calls of at most
10 seconds when reserving fifteen seconds for other work. These thresholds
remain necessary budget arithmetic; sufficient quality is a separate question.

Client timeouts also differ by path. The successful generated untuned probe used
the launch config's **240-second** limit; the baseline evaluation provider
explicitly uses its 285-second job limit. The pre-fix tuned pipeline provider
defaulted to **60 seconds**. Root's A10 fix forwards the actual candidate config
limit, which future acceptance binding must verify. Fixing that mismatch lets
a long request finish; it does not make nineteen serial requests fit 285 seconds.
Read timeouts have ambiguous execution outcomes and are not automatically
retried. No timeout should become a clean no-violation result.

Reproduce the verified numeric analysis with:

```powershell
.venv/Scripts/python.exe -m benchmarks.performance.qwen38_latency_feasibility --model-plan benchmarks/private/qwen38/baseline-first-quality/baseline.model-plan.ready2-a09.json --reference-seconds 69.554049639999 --provider-timeout-seconds 240 --reference-receipt training/private/qwen38-serving-91263-baaa/full19generated-untuned-actual/full19generated.gate.receipt.json --reference-provenance "Verified generated-only old91263 receipt; untuned client240; no real-video latency or quality measurement" --output benchmarks/performance/qwen38-latency-feasibility-new-observation.json
```

The executable reads only plan geometry and the numeric generated receipt.
Existing output files are never replaced. The current verified aggregate is
`benchmarks/performance/qwen38-latency-feasibility-verified-a12.json`.
The earlier `qwen38-latency-feasibility-a12.json` is provisional and superseded;
its companion `.superseded.json` records the correction from supplied rounded
timing labels and the scope of the pre-fix tuned 60-second timeout. Its historical
bytes remain preserved.

## Root-approved separate experiment

Root approved **accuracy-diagnostic-v1** as a separate named protocol, with the same
six fixed validation contents, reference-label pins, source geometry, all
nineteen facts, prompt, four rules, semantics, generation controls and original
evidence. Preserve the historical speed family unchanged. Record the actual
new source/image/model/adapter/provider identities and source-equivalence
receipt. New diagnostic attempt IDs and exact A09 releases are required.

Declare execution order **5, 1, 2, 3, 4, 6** before any new output, identically
for each compared arm. This chooses the first case by duration metadata alone.
The full six-case expected denominator remains declared even if the pilot stops.

**Phase 1:** release only case 5, the shortest 17.3-second source. Give it a hard
**960-second** diagnostic cap, eight refinements, and at most **24 actual model
POST slots**: `(4 initial + 8 refinements) × 2 transport attempts`. Make one
fresh source attempt, without an automatic source retry. Stop on timeout,
invalid structure, incomplete coverage, unresolved publication, missing
provenance/evidence, or operational failure. Preserve that terminal outcome and
leave the other five explicitly unattempted.

Advance only if the pipeline completes with actual model identity, full
requested-fact contracts, complete source coverage and valid original evidence.
Do not advance or stop based on whether its verdict agrees with a provided
binary label. A successful pilot is technical completion evidence, not an
accuracy score or a reason to choose favorable cases.

**Phase 2, only after root reviews phase 1:** release the remaining fixed cases.
Preserve the first case's actual result; do not rerun it to obtain a better
outcome. Predeclare these per-case diagnostic caps for every arm:

| Original case ordinal | Logical request cap | Actual POST slot cap | Diagnostic wall cap |
|---|---:|---:|---:|
| 1 | 26 | 52 | 1,980 s |
| 2 | 13 | 26 | 1,080 s |
| 3 | 13 | 26 | 1,080 s |
| 4 | 13 | 26 | 1,080 s |
| 5, already attempted | 12 | 24 | 960 s |
| 6 | 27 | 54 | 2,040 s |

Caps total **8,220 seconds (137 minutes) per arm** and **208 actual model POST
slots**, including unused/failed transport slots. Each wall cap is the next
whole minute above `(initial windows + 8) × 69.55404964 + 120 seconds`.
The 120 seconds is an explicit planning allowance, not observed I/O. Longer
outputs or refinements may still exceed these caps; they must remain failures.
Apply the existing rental stop and whichever stricter exact root release applies.

A13 needs a separate operational diagnostic/subset runner and separate report;
do not alter the frozen worker's 285-second recipe while still calling it the
same speed benchmark. Report real elapsed times, actual POSTs, failed and
unattempted cases and finite provided-label agreement. Any actual duration above
285 seconds is a measured speed miss for that case. This diagnostic cannot
establish a general 285-second SLO or ten-second upload-to-UI result. Reviewed
event/type/legal accuracy still requires independent human gold.
The concrete metadata schema and commands are in
`docs/integration/qwen38-accuracy-diagnostic.md`. This approval prepares the
experiment; model execution still needs its exact fresh release and reservation.

## Performance work alongside the diagnostic

Keep A08's faster-serving research separate and bounded. Lowering a token cap
does not accelerate an output that already stops at 561 tokens; a 512-token cap
would truncate this observed valid response. Dropping frames, facts or windows
would change the task. Increasing timeouts does not fix throughput.

If root chooses a new optimized engine or precision profile, first run at most
two generated, full-sixteen-frame/nineteen-fact requests under its actual bound
provider and strict schema, with a warm immutable grammar and fresh matchers.
Record full endpoint/provider timings, actual usage, peak memory, failure and
real model/source/image identity. Stop after the first failure or unconfirmed
in-flight request. This is a separately released public diagnostic, not a new
private validation attempt or automatic model selection. To justify trying all
nineteen initial windows within 285 seconds, the observed per-call budget must
approach 14.211 seconds with the stated fifteen-second reserve; a single short
warmup or cached answer cannot establish that.
