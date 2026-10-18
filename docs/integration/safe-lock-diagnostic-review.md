# Safe-lock diagnostic review — 18 October 2026

`tools/safe_lock_probe.py` is suitable for an isolated perception diagnostic. It
does not establish a `KEY_LEFT_IN_SAFE` finding or justify enabling that type.
No real safe-lock benchmark was present under `benchmarks/quality/` at this
review. There is no measured recall, false-positive rate, or improvement toward
37/40 from this review.

Reviewed tool SHA-256:
`d6d1641243bfb48011140511fb9d7c79b38ed24d8c3d2ce5a1abace43dd0c825`.
Also reviewed `src/forte/fast/video.py`, the image encoding helpers in `vlm.py`,
the existing specialist key rule in `temporal_events.py`, and the relevant
catalog/local normative text. No analyzer, host, bridge queue, or evidence file
was changed. No real image or model free-text output was inspected.

## What the tool preserves

- The decoder selects only the visual stream. Crops come from retained source
  frames at native dimensions, with no crop resizing, generated detail,
  super-resolution, or temporal fusion.
- Safe geometry comes from the source run's layout. Numeric boxes must be finite,
  bounded and ordered before cropping; crop expansion stays inside the frame.
  Names are used to load and associate results, never as perception features.
- Every video receives the same uniformly spaced window schedule. The tool has
  no label input, no type-enable path, and no candidate or verdict output.
- Its HTTP client accepts HTTP loopback endpoints, disables environment proxies
  and redirects, and sends content to the local VLM only.
- Localization and classification use closed enum/numeric schemas. Extra fields,
  free text, incomplete generation and reasoning fields are rejected. The frame
  index sequence must match the requested sequence exactly.
- Exceptions retain only their type. Decode, cached-source, HTTP and schema
  failures remain `failed`; unresolved localization becomes `abstained`.

## Limitations that matter before a rule is considered

1. **Native dimensions are not pixel-exact model transport.** `image_part()`
   calls `jpeg_data_url()` with its default JPEG quality of 88. There is no
   geometric resizing, but JPEG recompression changes pixel values and can
   discard detail in a key only a few pixels wide. Describe this as native-size
   JPEG crops, rather than a lossless pixel-preservation guarantee. The tool
   does not write evidence; original-frame evidence requirements are unchanged.
   A lossless diagnostic transport, if introduced, must be a separately recorded
   candidate and evaluated uniformly rather than changed for selected videos.
2. **Localization is per window, not per frame.** The lock is localized using
   the middle frame and that same native rectangle is used for every frame in
   its window. A moving safe door can move the lock out of that rectangle. Such
   a frame is unresolved, not proof that the key was removed. Before/after
   comparisons need independently resolved, associated lock geometry or an
   observed stable lock.
3. **Enum validity does not imply semantic consistency.** The schema permits
   `key_state=inserted` together with `lock_visibility=occluded`, `too_small` or
   `not_visible`. A future scorer must explicitly require `resolved` in each
   supporting answer. The agreement count includes shared `unclear` and shared
   `empty`; it is not a count of confirmed inserted keys.
4. **Window context does not establish a departure.** `cashier_near_safe` refers
   only to the middle full frame, while `hand_at_lock` is per cropped frame.
   This cannot establish that the cashier removed their hand, walked away, and
   left the same key behind. A stationary visible key during active safe use
   is not enough. Existing specialist rules already require explicit cashier
   departure and resolved lock observations before and after it; the diagnostic
   supplies neither the complete departure timeline nor tracked key facts.
5. **Completion is not resolution.** A video row can be `done` even when every
   window is `abstained`. Report resolved windows, abstained windows and failed
   windows separately. Zero resolved positives across unresolved windows is not
   a tested zero-false-alarm operating point. Sparse windows can also miss a
   brief relevant event; they do not establish full temporal coverage.
6. **Call counters are successful-response counters.** HTTP attempts that fail
   before JSON is read do not increment `calls`. Use host/server request metrics
   if diagnosing retry load, billing exposure or total request attempts.

## General visual principle for a future candidate

A key must be visibly inserted in the same resolved safe lock before and after
an observed cashier departure, after hand contact has ended. Unclear pixels,
occlusion, lost lock association, moving geometry, or missing departure support
must abstain for this check. Two prompt phrasings must agree on the actual
inserted key and resolved visibility; agreement about uncertainty is not
positive support. This principle contains no clip identifier, labelled time,
fixed camera geometry or threshold selected to recover a known miss.

The diagnostic should first answer whether that perception is possible on the
source resolution. If it is, any candidate still needs the original brief's
non-target/augmentation gate, independent repeated evaluations, original
evidence validation, and staging/release checks. Numerical thresholds must come
from the prescribed non-target distributions or grouped folds. No threshold
has been fitted or recommended here, and no new type should be enabled from
these diagnostic enums alone.

## Normative scope

The local source `rules/sources/nbk_231_current.txt`, paragraph 80 at line 411,
states that the procedure for storage, issuance and transfer of keys and other
items is defined by internal bank documents that prevent unauthorized use.
It does not state a universal categorical rule that every visibly inserted
safe key is itself a violation. Any eventual finding therefore remains a
visible sign for human review, grounded in the actual local clause and with
the internal-policy limitation stated. The current catalog's shortened text
must not be presented as a verbatim quotation of paragraph 80.

## Independent synthetic checks

Twelve in-memory checks passed on Windows Python 3.14, using the actual tool's
AST-extracted functions and actual image encoder function. AST extraction
avoided importing the unavailable local PyAV package. Inputs were generated
synthetic pixels and fake structured client replies; there were zero real
video inputs, zero network calls, and zero model calls.

The checks covered full-frame crop coordinates; edge-clipped crop expansion;
rejection of reversed, infinite, boolean and NaN geometry; rejection of external,
HTTPS, credential-bearing and query-bearing endpoints; cached failure
preservation; explicit documentation that contradictory enums pass schema;
rejection of an extra text field; sanitized technical failure; unresolved
abstention; paired-answer completion; JPEG dimension preservation; and JPEG's
lack of pixel-exact preservation. The first harness invocation failed because
the harness omitted its `time` import; after correcting that harness-only
setup error, all twelve checks passed. These checks are not a full integration
test of decoding or a real vLLM server.

## When the structured result arrives

Summarize all videos uniformly: number of windows `done`/`abstained`/`failed`,
resolved frame counts in both phrasings, paired `inserted` plus `resolved`
counts, key-state disagreement counts, hand-contact enums, native crop sizes,
timings and token totals. Record the tool digest and source-run digest. Keep
perception support distinct from whole-video verdicts. A failed or missing
diagnostic is not a negative finding. Labels may enter the prescribed scorer
or grouped evaluation process only, never the diagnostic schedule, visual
questions, crop positions or branch decisions.

## Contextual successor prepared before inference

`tools/safe_key_sequence_probe.py` and
`tmp/agent-bridge/payload/q30_safe_key_sequence.remote.sh` address the isolated
crop diagnostic's main limitations without enabling a rule. This is a distinct
experiment, not a change to the original diagnostic or the frozen analyzer.

The sequence tool projects only cached processing status, cashier-role states,
timestamps and safe layout. It caches numeric native safe-ROI activity from
uniformly decoded source frames, combines activity-end and cached role-departure
proposals, and selects up to two uniformly spaced proposals for every video.
Six before/during/after samples carry original full-frame context, native safe
crops and separately localized native lock crops. All transport is lossless PNG;
there is no resize, enhancement, fusion or saved image/evidence file. Proposal
overflow remains explicitly unassessed, and missing key detail abstains.

Paired closed responses must agree on the same physical lock and a distinguishable
same key, actual contact, release while the cashier is still at the safe, a
subsequent role-based departure, and two resolved inserted-key snapshots at
least 0.5 seconds apart afterward. Unknown or contradictory intervening observed
lock states cannot bridge positive endpoints. Support includes explicit time
bounds and remains support at sampled times only: continuous key visibility and
the exclusion of unseen removal/reinsertion between samples are always false.
This does not prove a policy violation.

All 40 source rows survive into the outcome denominator. The host-only scorer
opens labels after the CPU inference container has exited; it reports all 19
negatives, failure/abstention/overflow counts, support on failed videos and an
explicit negative-assessment-complete flag. Zero observed negative support with
19 failed or unresolved negatives never qualifies anything. Type enablement and
accuracy claims remain false even for a fully resolved diagnostic.

The wrapper binds main test instance `plp5me7vz`, its exact Q30 model-container ID,
the frozen baseline source manifest/profile, and the exact cached baseline run
digest. A nonroot host UID/GID owns the private stage and writable output. The
CPU container drops all capabilities, uses a read-only root, and shares the
already verified offline internal model namespace. Only frozen `src`, the
separate diagnostic tool/input, videos, and private output are bind-mounted;
there is no label mount. Cleanup binds the immutable newly created CPU container
ID through a private marker, so an existing same-name run is never discovered
and stopped by name. The evaluator budget is 240 seconds; the outer timeout and
cleanup ceiling is 298 seconds. No model stop/reload or frozen-file mutation is
performed. Dispatch remains the parent agent's responsibility.

The principle and 17 fixed resource/sampling controls were declared in
`docs/integration/agent-log.md` before real inference. Zero controls were fitted
to bank labels. Nineteen local synthetic checks passed, including lossless
PNG pixel equality, contact/release/departure ordering, same-key uncertainty,
intervening occlusion, schema text rejection, all-40 failure retention and an
expired-deadline run with no model calls. An initial overflow fixture used
proposals within the declared merge separation; spacing that synthetic fixture
farther apart corrected the test without changing the proposal policy. Bash
syntax and Python compilation also passed. Independent read-only review found
no blocking issue and passed ten additional synthetic checks. No real images,
labels, model calls, host or bridge queue actions were used by this review.

Prepared immutable digests:

- Tool: `aff11846f9c503b351c7b1055e3635b5757ab387e796a4de209acb250bc060ce`.
- Payload: `01700ab7e87588e02df5bf12f801cd2612a9a5e6b9503d5584b2deec503a10b3`.

Real inference, contextual negative scoring and any measured accuracy gain are
still pending. Temporal median fusion was deliberately deferred; native PNG and
the ordered interaction are the prepared experiment.
