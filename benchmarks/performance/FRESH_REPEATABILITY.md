The optional `fresh_head_repeatability` harness measures three independent
decode → offline encoder → frozen temporal head runs on one permitted train or
validation source. One persistent encoder and head are loaded once. Each attempt
freshly decodes the original bytes, computes fresh features, calls the same head,
and writes a new private feature artifact.
No completed feature artifact or cached HTTP response is reused.

The current semantic scope is the trained weak binary score at the frozen zero
logit threshold. Zero is this experiment's fixed binary diagnostic threshold;
it is not a calibrated deployment threshold. Physical facts, rule applicability, event boundaries, legal
completion and JPEG evidence are unavailable in this binary-only path. Their
absence is not treated as a successful equality check. Head-only repeats on an
existing feature tensor, owned by A09, are a separate measurement.

Prepare and execute only after the real checkpoint exists. The operator must
coordinate the actual GPU slot with root/A17; this document grants no slot.
Preparation opens source bytes only for hash/size verification and consumes the
existing private, split-bound feature-input manifest. It never selects on labels.
The source index must resolve to train or validation; final_check is rejected.

1. In the final feature runtime, capture metadata with
   `python -m benchmarks.performance.fresh_head_repeatability runtime --output PRIVATE/runtime.json`.
   This queries package versions and GPU/driver identity without inference.
2. Use `prepare --help` to supply the prepared input manifest, one explicit
   source index, checkpoint directory, offline encoder snapshot and manifest,
   taxonomy, rules catalog, runtime JSON, and deployment proof from actual Docker
   inspection. Supply a new private plan path and a new private output root.
   The plan freezes three attempt IDs and all source/artifact/code/runtime pins.
   A deployment proof is an externally captured image/build observation; the
   process cannot independently inspect its own Docker image ID. Preserve that
   proof together with the matching container launch record.
3. Copy the private plan to the central exposure-registry host. Run
   `reserve --plan PRIVATE/plan.json --receipt PRIVATE/reservation.json` there,
   using the central manifest/registry defaults or their explicit paths. This
   appends all three attempts without replacing prior exposure history. Copy
   the resulting private receipt to the GPU host. No model executes in reserve.
4. After the explicit checkpoint/GPU-window release, run
   `execute --plan PRIVATE/plan.json --receipt PRIVATE/reservation.json --receipt-sha256 RECEIPT_DIGEST --report SAFE_REPORT.json --gpu-window-released`.
   The release flag records the operator's already obtained slot authorization.
   The receipt must match the exact frozen source and all three planned IDs.
   An existing attempt root or report fails closed; a retry needs a new plan.

Every successful attempt retains private feature safetensors, raw output
safetensors, canonical source identity, decoded RGB hashes and binary semantics.
Comparison preserves exact original integer PTS, rational time base, source
origin, ordered frame IDs, view geometry, uncertainty and rule references when
present. Only enumerated job/request identities, paths and timing fields are
removed by the generic semantic canonicalizer. Human decisions are separate.

The safe report contains the full three-attempt denominator, missing/failure
counts, observed hardware/packages and budgets, pairwise equality/difference
counts, feature/logit maximum absolute differences and stage wall timings.
Bitwise equality and numerical closeness are reported separately. No source
hash, label, score, binary prediction, caption or private path enters that report.
The report does not establish universal determinism, accuracy or a ten-second
public-service result. Its one-source sample cannot estimate a latency tail.

`decode_active_sec` measures time inside decoder iteration, including process
startup and waiting for returned batches. `encoder_active_sec` sums actual
encoder calls. Those stages overlap. `fresh_feature_pipeline_sec` includes the
stream, tensor concatenation and source-identity validation. `head_forward_sec`
includes input transfer and synchronized forward completion. Private feature
and output serialization occurs after the timed binary output and is recorded
as `artifact_write_sec`. Cold model loading is reported separately, with actual
load-call and process counts. These are local monotonic durations;
they are not upload-to-complete-legal-JSON timings.

Generated-only verification:
`python -m pytest qa/system/test_repeatability.py tests/test_observability.py qa/system/test_reliability.py -q`.
The changed-pin test retains all three failures before CUDA/media/model access.
A generated orchestration test checks that the three fresh calls share one
runtime/model pair and that cold loading is recorded once. These tests do not
establish an actual GPU run.
