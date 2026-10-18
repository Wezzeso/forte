# Automatic first pass

This is the executable machine supervision path. It does not require human
annotation and does not turn machine output into independently verified truth.
All input/output records and source media remain private on the execution host.
Console output is aggregate counts and static error codes only.
After freezing a consumed experiment, pass `--export-track w2` for further
teacher improvements. This writes `annotations/private/automatic/w2/` and leaves
the consumed default export bytes intact. Only simple lowercase track names are
accepted; no arbitrary filesystem paths are allowed.
For a W2-only teacher export, also pass `--proposal-index` with the exact private
`forte.proposal-artifact-handoff-index.v1` terminal index. Every selected audit
SHA must already be imported and current; the exporter rejects missing,
superseded, duplicate or nonterminal receipts. It excludes unrelated earlier
proposals and binds each record to the index file SHA. This selection requires a
separate export track, so it cannot replace the default consumed artifact.

From the repository root, after a terminal artifact handoff and an explicit
measurement-window release:

```powershell
.venv\Scripts\python.exe -m annotations.cli fetch-proposal --handoff <private-handoff.json>
.venv\Scripts\python.exe -m annotations.cli auto-first-pass --manifest data/manifests/recordings.jsonl --split-manifest data/splits/automated_provisional_split.json --publish-workbench
```

The fetch command downloads only the handed-off audit and its referenced JPEGs,
checks hashes, rejects unsafe archive/path mappings, binds the original input by
SHA-256, and imports an unreviewed local-model proposal. It does not infer or call
a model. Existing human work and optimistic revisions remain protected. See
[FETCH_REVIEW.md](FETCH_REVIEW.md) for the exact terminal handoff contract.

`auto-first-pass` reads these already imported outputs, validates original
source-frame identity with A05 `validate_frame_ref` (content hash, raw source PTS,
time base and origin), rechecks local JPEG hashes, and writes:

| Private path under `annotations/private/automatic/` | Schema | Consumer |
| --- | --- | --- |
| `machine_annotations.jsonl` | `forte.machine-annotations.v1` | A09 weak teacher targets and local workbench |
| `automatic_group_proposals.json` | `forte.automatic-group-proposals.v1` | A02 label-blind provisional grouping |
| `autonomous_annotation_worklist.json` | `forte.autonomous-annotation-worklist.v1` | Train-only teacher runner and feature extraction |

The JSON Schema is [machine.schema.json](machine.schema.json). Python interfaces
are `summarize_record`, `build_group_proposals`, `build_annotation_worklist`, and
`automatic_first_pass` in `annotations.machine`.

Each machine record includes its content identity, actual source frame references,
scope-bound windows, observations, targets, masks, candidate events, limitations,
and provenance. `human_reviewed` and `training_authorized` are always false.
Independent A02 experiment authorization can enable a weak candidate without
changing those fields. Organizer binary labels are never read by this module,
never supplied to the teacher prompt, and never expanded into factual states.
The newest imported attempt supplies the active summary, including partial or
failed output; older longer attempts remain available separately in the store.

Window `observed_facts` retain model observations. `target.facts` masks unsupported
observations as `unclear`; `fact_supervision_mask` states which facts may contribute
to weak supervision. Missing facts are not filled. Absence needs a supporting
frame and complete visible procedure context; count absence also needs all
counting paths visible. Candidate target endpoints must match unambiguous actual
source PTS. Rule matches express required-state consistency only; temporal order
and exceptions are not independently verified by this export.

Every window carries its own `provenance.prompt_sha256`, copied from the exact
original audit window's `prompt_hash` or `prompt_sha256`. Different windows can
have different prompts. Missing legacy hashes remain null. Model identity,
revision, artifact manifest, proposal hash and inference location are preserved
from actual imported provenance. Training must mask/reject teacher windows whose
required provenance is unavailable; independent organizer-label MIL remains a
separate path.

`explanation_ru` supplies readable local WHAT/WHY from catalog phrases and strictly
supported machine observations. It never consumes free model notes, a filename,
or an organizer label. Rule candidates list missing required facts and retain
unresolved temporal/exception checks. When the teacher supplies no supported
event type, the explanation explicitly abstains instead of inventing one.
Optional teacher focal metadata is copied into `inspection_focus`; factual states
still describe the full requested context with cited source-frame support.

An independent direct anchor question may additionally produce
`windows[].frame_fact_targets`. This path accepts only the eight explicitly
localizable physical atom IDs in `ANCHOR_PHYSICAL_FACT_IDS`, exact original anchor
identity, direct-question provenance, and cited anchor support. It does not derive
frame labels from full-window facts or from supporting-frame membership. The
actual `question_definition` and `question_fact_id` are preserved; compact
canonical JSON SHA-256 verifies `provenance.focal_question_sha256`. The actual
prompt must match the enclosing window. The question must name the request-local
frame alias and exact source timestamp, with the same declared canonical anchor.
Missing questions/prompts or unsupported decisive answers are masked or omitted.
Temporal and procedure/context facts are rejected from this per-frame branch.

An actual explicit anchor answer `unclear` remains an eligible `unclear` frame
class even with no decisive frame citation. It additionally produces a separate
`support_reviews` receipt with `supports_state:false`, `teacher_state:unclear`,
`frame_id == focal_frame_id`, and both `reference_kind` and
`negative_reference_kind` equal to `focal_frame_explicit_unclear`. This means no
decisive support for either present or absent at that exact anchor. It is never
an absent fact and does not label surrounding frames. Missing/unasked answers do
not produce this receipt. Both rows carry the same original question and prompt
provenance. If full-context support contradicts the focal receipt, the receipt
is retained with `supervision_mask:false` and
`mask_reason:direct_anchor_window_support_conflict`; W2 training masks disputed
support directions/citations while retaining the independently answered frame
state. Original observations remain unchanged.

Calibration comparison digests (`teacher_record_sha256`, `teacher_window_sha256`,
`teacher_target_sha256`) must be computed over the actual exported JSON objects
using compact canonical JSON SHA-256. The legacy `summary_sha256` uses a different
serialization and is not a substitute for those joined artifact digests.

Optional `--provided-anchor-manifest` arguments may point to A02's train and
validation `forte.user-coarse-anchor.v1` files. The join verifies source aliases,
assignment digest, duration and split. Train rows attach `provided_focus`, while
validation rows attach `provided_evaluation_focus` for diagnostics after
prediction. Only train focus enters a worklist. Final-check manifests are rejected.
Provided points remain class-agnostic approximate sampling priors with unknown
boundaries/uncertainty; they never become event types, rule truth or factual states.
Binary labels are not copied into the worklist, machine facts or explanations.

`per_second` is a coverage index across the source duration. Each bin names actual
frame IDs/PTS and overlapping model windows. Its states are
`unknown_no_sampled_frame`, `source_frames_available_unanalyzed`,
`partial_machine_observation`, or `machine_observed`. A sampled observation does
not establish continuous visibility throughout a second. Unobserved or
event-free bins abstain; failed/incomplete attempts do not become negatives.
Numeric millisecond timestamps do not claim one-millisecond localization accuracy.

The worklist pins `split_file_sha256` and `assignments_sha256`. Records include
`sample_id`, `content_sha256`, `source_path` (local original input), `duration_sec`,
`split`, `scenario_group_id`, and `pending_second_bins`. Only train and validation
contents are included; final-check/test contents are excluded. No remote media
path is inferred from a filename. The execution host must map/probe original
sources by content hash. The teacher runner uses train by default and isolates
validation. Creating a worklist dispatches no requests and authorizes no training.

Machine grouping covers every alias and preserves exact duplicate identities.
Observed event/fact signatures plus geometry/duration are provisional heuristic
groups. Unmodeled sources retain exact-content-only groups. Scenario independence
is explicitly unvalidated; the export does not lock a production split.

`--publish-workbench` writes a separate `machine_first_pass` field with revision
fencing. Human verdicts, events, notes, negative windows, split locks and training
permission remain intact. The browser reads static UI files on refresh; neither
server needs a restart for these display updates. Save any existing draft before
refreshing. Human review/export remains an optional, stricter independent path.

Synthetic checks:

```powershell
.venv\Scripts\python.exe -m pytest annotations -q
node annotations/ui/test-forms.mjs
```

Tests use generated source bytes/JPEGs and mocked browser endpoints; they do not
inspect supplied recordings, call a GPU, or create private screenshots.
