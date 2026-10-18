# Separate accuracy diagnostic v1

Root approved this operational diagnostic after the generated full-task request
took 69.554 seconds wall time. It preserves the six frozen validation contents,
provided reference pins, 8/4-second geometry, sixteen frames, nineteen facts,
prompt, rules, generation, semantics and genuine evidence. It has its own recipe
digest and attempt IDs. The original 285-second family and source archive remain
historical and unchanged. This diagnostic grants no speed, UI or quality claim.

`benchmarks.performance.qwen38_accuracy_diagnostic` prepares metadata only. It
never probes or decodes media, reads labels, constructs a provider, submits a
request or grants a release. A13's separate operational runner imports its
strict `validate_diagnostic_plan`; the original six-case executor is unchanged.
The flattened metadata contract imports only the Python standard library on the
host. Protocol creation additionally invokes the frozen family validator; actual
CPU execution independently invokes the frozen source-equivalence/AST verifier.

Freeze the shared protocol before either new arm executes:

```powershell
.venv/Scripts/python.exe -m benchmarks.performance.qwen38_accuracy_diagnostic protocol --family benchmarks/private/qwen38/baseline-first-quality/family.frozen.json --historical-arm-plan benchmarks/private/qwen38/baseline-first-quality/baseline.model-plan.ready2-a09.json --output benchmarks/private/qwen38/accuracy-diagnostic-a12/protocol.frozen.json
```

The protocol contains metadata for all six sources, exact original family and
recipe digests, reference-label and anchor digests, and the fixed order
**5, 1, 2, 3, 4, 6**. No new runtime identity is inserted into that protocol.

| Original ordinal | Initial windows | Maximum logical calls | Maximum actual POSTs | Diagnostic seconds |
|---|---:|---:|---:|---:|
| 5, pilot | 4 | 12 | 24 | 960 |
| 1 | 18 | 26 | 52 | 1,980 |
| 2 | 5 | 13 | 26 | 1,080 |
| 3 | 5 | 13 | 26 | 1,080 |
| 4 | 5 | 13 | 26 | 1,080 |
| 6 | 19 | 27 | 54 | 2,040 |

Total bounds are **208 actual POST slots and 8,220 seconds per arm**, including
the pilot. Remaining sources receive **184 slots and 7,260 seconds**; these are
not additional full-arm budgets. Each source has one attempt, without source
retry. Transport retries consume exact POST slots before the request, including
failed or cancelled requests. Unused slots are settled without exposure refunds.

After the actual warmup/full-19 gate, bind a new pilot plan from A13's genuine
current arm plan. `--actual-arm-plan` supplies actual model/adapter, source,
image, base manifest, config, source-equivalence and runtime-profile pins. No
runtime plan is produced here while those actual artifacts remain pending.

```bash
python -m benchmarks.performance.qwen38_accuracy_diagnostic bind \
  --protocol "$protocol_file" --actual-arm-plan "$actual_bound_arm_plan" \
  --role-config "$actual_fresh_role_config" --stage pilot --output "$new_pilot_plan"
```

The standard `forte.qwen38-attempt-plan.v1` has `kind=accuracy_diagnostic` and
`phase=evaluation`. `source_attempts` and `protocol_cases` contain all six;
`cases` contains only ordinal 5; `attempts` contains only its 24 POST slots.
`released_source_attempts` contains its one source ID. Its exact
`maximum_execution_seconds` is 960. Root releases and A09 reservations must bind
the actual raw file digest and those exact 24 attempts, with the existing expiry,
source, config, model, image and hard rental-stop guards. Root must separately
authorize each arm. Plan preparation is not permission to call the model.
Root will run the untuned and trained pilot independently. An untuned pilot
failure does not waive or prohibit the separately released trained pilot.

`worker` retains the exact historical 285-second profile as immutable parent
metadata. Only the new diagnostic runner explicitly replaces
`job_timeout_seconds` with each case's `diagnostic_deadline_seconds`. This is
recorded in the new recipe's timeout policy. Provider timeout remains the actual
configured 240 seconds for the candidate; no hidden per-request extension is
allowed. Effective stop time is also bounded by the exact release and rental stop.

The runner initializes and preserves six source terminals. A failed, rejected,
timed-out or incomplete pilot remains failed; the other five remain explicitly
not attempted. Advancement requires complete coverage, strict publication,
native source completeness, actual evidence joins, actual model identity and a
fresh decode/preprocess/model/evidence witness. The runner derives the
`technical_completion` record from the actual receipt and original evidence.
Provided verdict agreement is never read to decide progression.

Only after root reviews a technically successful pilot and authorizes remaining
sources may a new remaining plan be prepared:

```bash
python -m benchmarks.performance.qwen38_accuracy_diagnostic bind \
  --protocol "$protocol_file" --actual-arm-plan "$same_actual_bound_arm_plan" \
  --role-config "$actual_fresh_remaining_role_config" \
  --stage remaining --retained-pilot-plan "$immutable_pilot_plan" \
  --retained-source-terminals "$immutable_pilot_source_terminals" \
  --output "$new_remaining_plan"
```

The plan binds absolute paths and raw SHA-256 digests for both retained files.
The strict metadata gate requires exactly six original terminals, one verified
pilot success and five unreleased sources. The operational runner independently
rereads the original pilot receipt/result/witness, checks their hashes and full
technical criteria, and retains the pilot row byte-for-byte in meaning. It
never reruns ordinal 5. Its 184 newly released slots exclude every pilot slot.
Actual arm/runtime identity must remain the same across these two stages.
This includes the provider's genuine `launch_config_file_sha256` and actual
configured request timeout. Before inference, the CPU runner rechecks the
launch file's bytes and the independently audited source-equivalence proof.

The original pilot role-config hash is retained separately as
`retained_pilot_role_config_sha256`. The remaining release may bind a fresh raw
role config and new arm-plan digest because the central exposure snapshot ages
out after 300 seconds. It must preserve immutable source assignments and role
policy, and pass A09's fresh snapshot, exposure-ancestor and exact new reservation
checks. A fresh role snapshot grants no provider/model/runtime change.
`role_policy` contains the actual config's four immutable pins:
`expected_assignments_sha256`, `expected_source_manifest_sha256`,
`expected_automated_split_sha256` and `expected_scenario_groups_sha256`, plus
validation-only scope, the actual snapshot-age limit (at most 300 seconds), and
`final_check_used=false`. `--role-config` checks its raw bytes against the actual
arm-plan role pin and derives that object; it can be omitted only when the actual
arm plan already carries the genuine object. The runner's
`validate_reserved_role_policy` also matches it to the real reservation's central
`exposure_gate` input checksums, assignments and released-subset role counts.

The new report must preserve failures and six-case denominators, disclose reused
development validation and finite provided binary agreement, and record actual
elapsed times. Direct pipeline timing excludes upload and browser rendering.
A duration above 285 seconds is an observed miss for that case; successful
longer execution does not satisfy the historical speed target. Event, fact,
support, boundary and legal accuracy still require independent human gold.

## One corrected operational control launch

The first untuned control stopped before a source pipeline or model POST because
its AST proof was prepared under Python 3.14 and verified under the actual CPU
image's Python 3.12. Its original packet, failure reconciliation and all 24 spent
slots remain immutable. Root authorized exactly one corrected control launch.

The operational helper `qwen38_accuracy_reissue` requires the six original
artifacts: plan, six source terminals, 24 model-slot terminals, actual failure
reconciliation, A09 settlement and root's explicit one-reissue authorization.
It embeds each artifact's exact UTF-8 bytes, raw digest and parsed metadata.
Every source must show zero model POSTs and no source pipeline execution;
the sole failed pilot must identify the pre-execution proof failure. All old
slots must remain consumed without refunds. Missing proof, prior source/model
execution, a third control attempt or a broader authorization is rejected.

Actual proof and corrected plan generation require Python 3.12 in the verified
EAD CPU image. Passing `control_reissue=proof` to `bind_plan` creates a distinct
control-2 namespace for all six source IDs and the new pilot's 24 slot IDs.
The protocol, recipe, population, source geometry and actual runtime remain
unchanged. Only the independently reverified AST receipt metadata may differ
from the old control packet. This metadata comparison never changes or
normalizes a model result.

A corrected release must explicitly bind control attempt 2, the original spent
settlement and the complete reissue proof digest. The controller first validates
the actual new packet inside the actual CPU image, then atomically claims the
original failed plan's digest as the one-use reissue key. Different authorization
copies or refreshed role configs cannot grant another corrected launch.

The extra spent control budget is disclosed: **232 reserved slots**, consisting
of the old spent 24 plus the diagnostic's 208, with **at most 208 actual model
POSTs** because the original 24 have verified zero POSTs. There is still at most
one actual source pipeline attempt per case. Any later remaining plan must
carry the same control-2 proof and preserve the corrected pilot result; it cannot
rerun the pilot or hide the original control failure.
