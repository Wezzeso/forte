# Group-safe candidate splits

`candidate_split.json` is a deterministic proposal, not training authority.
The audit assigns exact-content duplicates together, uses a fixed seed, and
stratifies content/scenario groups by the coarse workbook state. A group
with conflicting or unknown labels is assigned to the unknown stratum.
Largest-remainder allocation targets train/validation/final_check ratios of
0.70/0.15/0.15. Group integrity takes precedence over exact file counts.
`final_check` is the untouched final-check split.

The supplied files currently yield 40 unique contents, with 28/6/6 unique
contents in train/validation/final_check. Raw recording counts differ because both
copies of a byte-identical recording inherit the same split. The proposal
does not establish independence between recordings of related scenarios.

`scenario_groups.review.json` initially contains exact-content groups with
`review_state: pending`. A local reviewer must review relationships across
recordings, merge related scenarios into one group, and explicitly approve
all groups and the enclosing document. Each recording must appear exactly
once. An existing review file is preserved by repeated audits. After local
review, rerun `audit` to generate a proposal using reviewed scenario unions.
Scenario grouping may change the final stratum counts.

The private UI can capture scenario proposals without changing protected
groups. Prefer the bridge below when using those forms. A02 does not set
human review flags or approval flags on the user's behalf.

A10 and A20 review and lock the proposal before any training. The locked
artifact must have `state: locked`, `locked: true`, `training_authorized:
true`, a nonempty `approved_by`, complete approved scenario groups, and a
correct `assignments_sha256`. Preserve `source_manifest_sha256`,
`source_workbook_sha256`, and `scenario_groups_sha256` to bind approval to
the exact inputs. Approval must be real; editing the flags is not approval.

Integrity verification:

```powershell
python scripts/audit_data.py check-split --split data/splits/locked_split.json --require-locked
```

Each derived frame/window/crop/augmentation JSONL record must contain:

```json
{
  "source_recording_id": "rec_synthetic",
  "content_hash": "source recording full SHA-256",
  "group_id": "source exact-content group",
  "split": "train"
}
```

Validate derived records with `check-split --derived PATH`. The checker
rejects unknown parents, changed source hashes, group changes, split changes,
missing/duplicate assignments, content/scenario groups crossing splits, and
assignment digest changes. `--require-locked` also rejects any candidate or
incomplete scenario review. No derived artifact may choose its own split.

All split files include private annotation metadata and stay out of production
images, source control, public artifacts, and inference prompts. No approved
training manifest is supplied while local scenario/annotation review remains
pending. Preserve final-check data until checkpoint selection; record any use
for iterative tuning as development-data exposure.

## Human proposal bridge

The local UI stores one correction per unique-content review unit:

```json
{
  "kind": "scenario_group_proposal",
  "proposed_group_id": "synthetic-scenario-a",
  "scenario_reviewed": true,
  "reviewer_id": "local-human-reviewer"
}
```

`scripts/build_reviewed_split.py` opens a consistent, read-only SQLite
snapshot from `annotations/private/review.sqlite3`. SQLite JSON projections
select only identities, revisions, human review metadata and scenario
corrections. Event descriptions, media, frames, facts and model proposals are
not selected or interpreted. The bridge requires every unique content and
all its duplicate recording aliases. It rejects missing/unchecked proposals,
empty group IDs, multiple proposals, conflicting identities, unknown contents,
and missing human attribution. A saved `review_update` audit must attest the
exact proposal under its original human author. A later notes editor may be a
different human; that does not erase the original author.

Check readiness, then build only after real confirmations exist:

```powershell
python scripts/build_reviewed_split.py readiness
python scripts/build_reviewed_split.py build
```

Incomplete review exits with code 2, reports aggregate error counts, and
writes no candidate. A successful build creates separate private
`scenario_groups.human_proposal.json` and `reviewed_candidate_split.json`.
It preserves the original candidate, protected grouping, review database and
all lock files. Complete scenario confirmation remains a proposal pending A20
approval. Full annotation review is tracked separately. Neither success nor a
file's existence authorizes training or locks a split.

Validate a generated proposal with its corresponding scenario file:

```powershell
python scripts/audit_data.py check-split --split data/splits/reviewed_candidate_split.json --scenario-groups data/splits/scenario_groups.human_proposal.json
```

## Model exposure and final-check independence

The append-only private registry is
`benchmarks/private/model-exposure.jsonl`. It stores technical IDs and full
content hashes only. Entries form a SHA-256 chain, appends use an OS file
lock and fsync, and repeated registration of the same trial/phase/content is
idempotent. New trials append new events while preserving all earlier bytes.
All six original selected inputs were conservatively registered, including
an input whose upload/model exposure was uncertain. Failed or uncertain
attempts remain excluded. Generated synthetic transport content outside the
known corpus cannot silently become a dataset identity.

A18 calls `register_selection` from `scripts/exposure_registry.py` immediately
before actual benchmark submission. Metadata registration errors stop the
attempt. It records later paired trials as well. The helper opens only the
selection metadata and manifest; it does not inspect media or labels.

Manual uploads also matter. A private repeatable-read service snapshot of
**all accepted job content hashes** must be refreshed and reconciled:

```powershell
python scripts/exposure_registry.py register-accepted-inventory
python scripts/exposure_registry.py status
```

The input is `benchmarks/private/service-accepted-content-hashes.json` with
schema `forte.service-content-exposure-inventory.v1`, complete-snapshot scope
`all_accepted_jobs_since_service_start`, capture time, accepted/unique counts,
and duplicate-preserving `records` containing only `content_hash`. The first
legacy unique `content_hashes` snapshot is accepted explicitly as well. Only
matches to the known corpus are registered.
Unknown external uploads remain outside the training dataset. The helper
writes registry-bound `benchmarks/private/exposure-inventory-status.json`.
It never prints source identities, result contents or private paths.

The bridge verifies that inventory's byte digest, registry ancestry, known
content coverage and snapshot age. The default age limit is 300 seconds, a
team freshness gate configurable by
`--max-exposure-snapshot-age-sec`. It does not establish that uploads accepted
after the snapshot are covered. Refresh the accepted-job snapshot and rerun
the bridge immediately before A20 considers a lock while the API is live.

Every scenario containing any registered content is excluded from
`final_check`, including its unexposed recordings and duplicate aliases.
Assignment is deterministic and stratified among eligible groups. If enough
unexposed groups do not remain for the proposed final-check group allocation,
the bridge fails readiness and writes no candidate. It never silently treats
exposed footage as unseen or grants training authority to compensate.
`check-split` reads the latest registry and rejects exposed final-check
scenarios; a candidate's registry tip must remain an ancestor of the current
append-only log. Later exposure of a final-check scenario invalidates that
split. An accepted-job snapshot remains an explicit temporal boundary, not a
promise about future manual uploads.

## Importable training guard

Training/evaluation code can import:

```python
from scripts.split_gate import validate_locked_training_inputs

report = validate_locked_training_inputs(
    approved_split,
    source_records,
    reviewed_scenario_groups,
    expected_assignments_sha256=independently_pinned_approved_digest,
    registry_path=private_exposure_registry,
    coverage_status_path=fresh_private_coverage_status,
    requested_content_hashes=requested_training_content_hashes,
)
if report["ok"] is not True:
    raise ValueError("training_metadata_gate_failed")
```

The pinned digest must come from the independently approved run configuration.
Copying it from a replacement split would defeat the protection against
promoting validation/final-check content into training. The helper verifies
the approved lock and scenario integrity, current exposure hash chain,
approved registry snapshot ancestry and prefix content-set digest, fresh
accepted-job inventory, final-check exposure exclusions, and requested
recording/content membership in approved training only. It returns aggregate
counts and errors, never rows. No helper infers training permission from
exposure or validates target facts from a binary label. Reviewed-target and
fact schemas remain separate mandatory checks in training code.

The keyword `source_inventory_path_override` permits byte-identical relocation
of a private accepted-job inventory to a private GPU mount. The original
status SHA-256 must still match the relocated bytes; no provenance or lock
is rewritten. `requested_recording_ids` can supplement or replace content IDs.
The helper is importable as a namespace module and also supports direct script
execution of the underlying audit/bridge. Copy all four source modules when
preparing an isolated training environment: `audit_data.py`,
`build_reviewed_split.py`, `exposure_registry.py`, and `split_gate.py`.

Exposure event schema, kept only in the private JSONL log:

| Field | Meaning |
| --- | --- |
| `schema_version` | `forte.model-exposure.v1` |
| `event_id` | Deterministic trial/phase/purpose/content identity |
| `trial_id`, `phase`, `purpose` | Technical attempt provenance; no event narrative |
| `content_hash`, `recording_ids` | Full SHA-256 and every exact-content alias |
| `recorded_at_utc` | Actual registration time |
| `exclusion_state` | `exclude_from_final_check` |
| `previous_event_sha256`, `event_sha256` | Append-only chain binding |

`scripts.exposure_registry.read_registry(path, source_records)` verifies the
chain and source identities and returns private `events`, `content_hashes`,
`event_count`, `tip_sha256`, and `content_set_sha256`. Callers must keep these
values private; only aggregate status is safe to print.

Synthetic checks:

```powershell
python -m unittest discover -s scripts -p test_reviewed_split.py
```

## Separate autonomous first-pass experiment

`scripts/build_automated_split.py` provides an independent machine-supervised
path authorized by the user. It does not require human annotation and does
not relax the human-verified lock/target gates above. Inputs are private
`forte.automatic-group-proposals.v1` opaque membership proposals from a team
local model, local visual features, or exact-content metadata. They must bind
the source manifest/configuration and cover every recording. Organizer
answers cannot appear in the grouping input. Exact duplicates bridge machine
components conservatively. All groups retain `machine_provisional` origin,
`human_reviewed: false`, and unvalidated scenario independence.

The builder reads original `open_labels.xlsx` again locally for independent
binary targets. It exports one record per unique content into separate
`data/manifests/provided_binary_labels.train.jsonl`, `.validation.jsonl`, and
`.final_check.jsonl` files. Only the training file belongs in training input.
Validation labels stay with evaluation. A binary organizer label means a
video-level MIL target only; it is never expanded into event/fact, boundary,
evidence or grounding truth. Teacher proposals and their masks/provenance
remain separate from those independent binary labels.

The default private outputs are `automated_provisional_split.json` and
`scenario_groups.automated_provisional.json`. Existing files are never
overwritten. The candidate remains unlocked with strict `training_authorized:
false` and `scenario_review_complete: false`. A separate explicit trusted
experiment authorization can create `automated_experiment.run-config.json`,
which freezes the assignment, workbook-source and each role's label-manifest
SHA-256. Its `validation_assignments_frozen` means machine experiment
assignments are pinned; it never means a human split lock exists.

The root-authorized local preparation command passes explicit metadata
release and experiment authorization references. Optional
`--allow-stale-exposure-snapshot-for-preparation` permits only stale-time
coverage during local metadata preparation and marks
`pre_run_coverage_refresh_required`. It does not forgive incomplete inventory,
unknown hashes, broken chains, or exposed final-check groups. Actual execution
still requires a fresh accepted-job snapshot and an explicit GPU window
release. No model/network request is made by the builder. Fresh exposure of a
frozen final-check group invalidates the experiment; it must not be silently
resplit after observing validation answers.

`scripts.split_gate.validate_automated_experiment_inputs` is a separate weak
gate. It requires the trusted run configuration's explicit authorization,
provisional-group allowance, frozen assignment pin, workbook-source pin,
role-specific provided-label manifest pins (`expected_provided_label_manifests_sha256`
under `experiment_authorization`), and `allowed_target:
video_binary_mil`. It also checks current exposure ancestry/coverage,
train/validation scenario disjointness, and requested role membership. Use
`label_split="validation"` only in evaluation; its default is training.
Returned proof binds the assignment, independent labels, current registry tip,
coverage snapshot and grouping provenance. It explicitly does not certify
human review, scenario independence, event truth or accuracy. Copy
`build_automated_split.py` as the fifth helper module for isolated weak
experiments.

Recheck the frozen machine run configuration using its independently supplied
SHA-256, without opening media or contacting the service:

```powershell
python scripts/check_automated_experiment.py --expected-config-sha256 TRUSTED_CONFIG_SHA256 --role train --output data/splits/automated_gate.train.json
python scripts/check_automated_experiment.py --expected-config-sha256 TRUSTED_CONFIG_SHA256 --role validation --output data/splits/automated_gate.validation.json
```

This wrapper additionally verifies the exact config, split, scenario, source
manifest and role-specific label file bytes. The bound proof remains false
if accepted-job exposure coverage becomes stale. Refresh and ingest the
accepted-job snapshot only after the orchestration window allows network
activity, then rerun the gate immediately before execution. Stale coverage
does not make human annotation a prerequisite for this machine experiment.

If all groups are exposed, final-check receives no exposed content and its
independence remains unverified; no pristine final-check claim is possible.
If provisional grouping cannot support distinct train and validation groups,
the builder fails rather than inventing an independent evaluation set.
Neither machine agreement nor provided-label validation guarantees 100%.

Run both independent tracks' synthetic tests:

```powershell
python -m unittest discover -s scripts -p "test_*split.py"
```

## Direct feature and teacher exposure

Model work outside the public API also appends to the same central inventory.
Call `scripts.exposure_registry.register_content_manifest` locally before
the first feature or teacher call. It accepts a private JSON `records` list
or JSONL containing content SHA-256, optional recording aliases, and split;
verifies the independently pinned frozen assignments and source manifest;
and rejects the entire request if any input is outside permitted roles.
`feature_extraction_attempt` may explicitly permit train and validation for a
frozen feature encoder. `bounded_teacher_attempt` permits training only.
Neither path permits final-check. The helper opens no media and makes no model
calls. Repeating a trial/purpose/phase/content identity is idempotent.

The CLI is `scripts/exposure_registry.py register-content-manifest` with
`--input`, `--split`, `--expected-assignments-sha256`, `--trial-id`, `--purpose`,
and repeated `--allowed-split` arguments. Defaults permit training only.
All accepted-job snapshots add the known matches to this inventory; they
never replace or erase direct feature/teacher exposures. Exposure of all
train/validation contents therefore preserves the untouched final-check
contents and the frozen assignment. Generate a new proof against the latest
registry/coverage snapshot without editing the frozen run configuration.

## User-provided coarse anchors

`scripts/ingest_coarse_anchors.py` reconciles a separately captured private
user filename/binary/second table against the original workbook, audited
Unicode joins, duplicate content groups, source durations and frozen split.
The capture records its message provenance and explicitly distinguishes its
own byte digest from unavailable original chat-message bytes. The supplied
42 rows reconcile into 40 contents. Distinct coarse points attached to a
byte-identical source are retained for local uncertainty-aware use.

Separate private outputs `data/manifests/user_coarse_anchors.{train,validation,final_check}.jsonl`
contain `content_sha256`, aliases, frozen role/group, `provided_video_label`,
`coarse_anchor_sec` (a list, empty for supplied negatives), original
recording-to-point metadata, source/provenance digests and scope restrictions.
`data/splits/user_coarse_anchors.pins.json` freezes each role's exact bytes and
the unchanged assignment. Raw capture and final-check anchor values must not
enter the training bundle. Source hashes alone can bind their provenance.

These are class-agnostic approximate points, not starts/ends, exact event
timing, type/fact/evidence labels or legal truth. `uncertainty_sec` is unknown
and the timeline outside an anchor remains unknown. Training anchors can
focus training context and weak eventness bags. Validation anchors are only
for diagnostics after predictions, never inference conditioning or threshold
fitting. Final-check anchors remain held out. Original frame PTS and observed
facts must still be determined independently inside the team-local service.

Run all local synthetic metadata checks:

```powershell
python -m unittest discover -s scripts -p "test_*.py"
```
