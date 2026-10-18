# Rebind acceptance after actual runtime selection

Use this continuation for the future selected worker. The earlier `b4c2ba…`
six-fixture bundle is historical. It supplies the two unchanged TRAIN files,
but its old trace bindings and API-image assumptions must not be reused.

The new six-file operational overlay is
`benchmarks/private/qwen38/acceptance-rebind.ops.timeout-v2-a12.tar.gz`:
23,705 bytes, SHA-256
`3c6b293bb049c5065e114c2581eec180602764060df5ce279ddfd745ae539ebf`.
It contains only the acceptance client/binders and host launchers. It has no
`serving`, `forte`, or `evaluation` package, no fixtures, key, release or selected
runtime. Preserve its companion member manifest.
The earlier `0e2f0d…` overlay remains historical. Version 2 additionally verifies
the actual provider's `request_timeout_sec` against the bound config limit and
permits only A13's independently audited candidate timeout forwarding source
delta. Actual worker and model archive pipeline bytes must match each other;
all other common prompt, sampling and rule pins remain exact.

This preparation is CPU-only. No runtime has been selected or rebound by A12,
and no new upload, reservation, model call, or service change was made.

## Readiness findings

- The historical API image and worker image differ. Future helpers must use
  the genuine new worker CPU image and its actual source/config mounts.
- A19's future CPU derivative replaces the installed `forte` package and public
  source roots, removes duplicate source directories, and inventories actual
  imported module files. A12 does not substitute the API image for this proof.
- A06's new 27B config must bind native HF XGrammar 0.2.7, the exact structured
  decoding policy, actual model image/source/base/config hashes, and actual
  selected adapter. Null template pins remain invalid.
- A13's new source-equivalence receipt must bind the genuine later model
  source archive to the unchanged frozen quality family. A rewritten family or
  substituted historical source hash is rejected.
- Actual `LocalModelConfig.request_timeout_sec` must equal the selected runtime
  config's `limits.request_timeout_seconds`, currently 240 seconds. The context
  records that effective provider value independently of the worker job limit.
- The actual sampling file and frozen-family pin currently both equal
  `aace82575cd12222ade587a413a2e330f81d277bee7038e11378d9cc25459a9a`.
  Stable clip identity is already in that frozen shared source. No mismatch is
  inferred from the date of an older operational bundle.
- One active job, one clip, one in-flight request, screening disabled, 8/4-second
  windows, 2 fps, 16 frames, eight refinements and two transport attempts remain
  required. The two fixtures still require six source jobs and 126 maximum model
  POST slots. Failed or timed-out quality outcomes remain failures.

## Prepare from the genuine worker context

Extract the new overlay into a fresh protected operational root. Copy the
unchanged `acceptance-six-a12` population, separate class reference, and two
hashed input files there. Do not copy the old `evaluation`/`serving` package
trees into this root. The new CPU image must supply A13's actual trace,
exporter, family/equivalence tools and central gate dependencies.

After root selects the actual runtime, write a private JSON argument array for
`qwen38_acceptance_rebind prepare`, using actual existing files and receipts.
No arguments file is generated now because the final image, archive,
equivalence receipt and selected runtime are not yet available.

The array must contain these options:

- `--population`: the copied private `population.json` under `/ops`.
- `--runtime-config`: the actual worker's mounted candidate config path.
- `--family`: the unchanged frozen quality family file under `/ops`.
- `--quality-equivalence`: A13's genuine receipt after hashing the later archive.
- `--worker-archive` and `--worker-archive-sha256`: A19's actual worker source.
- `--model-archive` and `--model-archive-sha256`: the actual selected 27B source.
- `--trace-archive` and `--trace-archive-sha256`: actual archive containing the
  imported A13 trace/export/evaluation modules. It may be the same new archive
  when those actual bytes are included there.
- `--arm`: the actual selected 27B arm.
- `--output`: a new private root under `/ops/benchmarks/private`.

Do not supply `--cpu-image-sha256`. The host launcher obtains that exact value
from the selected worker container's Docker image metadata and supplies it to
the offline helper. It passes only allowlisted non-secret worker settings and
read-only source/config mounts, disables networking, and attaches no GPU.

Run from the fresh operational root on the existing host:

```bash
python3 -m deploy.qwen38_acceptance_selected prepare \
  --ops-root "$actual_operational_root" \
  --worker-container "$actual_selected_worker_container" \
  --arguments-file "$actual_prepare_arguments_file"
```

The helper compares actual module origin files against worker/trace archive
members, validates the real candidate config and native grammar, verifies
A13's equivalence receipt and common source/rule/taxonomy pins, and matches
the full actual worker/generation recipe to the frozen family. It never
constructs a provider, pipeline, worker, model or HTTP client. A mismatch creates
no bound execution plan. Success writes a new acceptance envelope, context
inventory, six fresh source attempt IDs and 126 fresh model slots. Historical
attempt IDs are never reused.

These receipts establish configured source equivalence. Loaded GPU weights,
readiness, actual stage calls and result quality remain to be verified in the
released execution.

## Trace binding and exact releases

A09 next reserves the new model plan under a new exact root model release.
Prepare a second private JSON array containing `--rebound-root`,
`--model-release`, and `--model-reservation`, all referring to actual files
inside the helper's context, then run:

```bash
python3 -m deploy.qwen38_acceptance_selected trace \
  --ops-root "$actual_operational_root" \
  --worker-container "$actual_selected_worker_container" \
  --arguments-file "$actual_trace_arguments_file"
```

This repeats actual context validation, checks the exact model-plan release and
reservation against genuine model source/image/base pins, and writes
`bound/trace-binding.json` using the actual imported pipeline and wrapper
hashes. It does not activate the worker. Root mounts the binding at the same
path in A13's traced worker and supplies the actual CPU image assertion.

After readiness and root's service handoff, reserve the new six-source plan
with its separate exact release. Preserve the additional source release fields
documented in `qwen38-acceptance-operational.md`: actual trace-binding hash,
budget-file hash, and maximum 126 generation HTTP attempts.

Only root's released execution invokes:

```bash
python3 -m deploy.qwen38_acceptance_selected execute \
  --ops-root "$actual_operational_root" \
  --worker-container "$actual_selected_worker_container" \
  --bound-root "$actual_bound_root_relative_to_ops" \
  --run-root "$new_run_root_relative_to_ops" \
  --release "$actual_source_release_relative_to_ops" \
  --reservation "$actual_source_reservation_relative_to_ops"
```

The selected worker supplies the genuine CPU execution image and existing
private key/data mounts. The launcher requires that worker to be running A13's
trace entrypoint with the exact binding. It changes no service or admission
setting. Uncertain, failed and rejected attempts stay in the denominator.

The offline `report` action uses the same worker/container and bound/run root
arguments. It reads the shared data volume read-only, joins genuine fresh
witnesses and immutable result bytes, and exports native/evidence receipts.
Neither old audits nor a scalar call budget can replace these witnesses.

## UI evidence boundary

The HTTP batch measures upload start through the complete result's final byte,
including all JPEG evidence. It cannot establish displayed readiness. The live
frontend's `data-e2e-state="complete"` and `data-e2e-seconds` are produced only
after a genuine upload in that document, complete evidence decode, source video
readiness and rendering. A restored job reports that clock as unavailable.

If root allocates the six source attempts to a controlled browser runner, those
same six uploads can additionally capture the frontend's successful frozen
timer, failed/incomplete states, viewport and actual image/video readiness.
That runner must use the identical new source IDs/releases and feed its actual
canonical response bytes and UI-origin times into the private terminals. Do
not also run the six-upload HTTP batch or infer a UI time from its results.
No browser upload runner has been executed or bound by this preparation;
usable-result timing remains unverified until a genuine released browser run.
