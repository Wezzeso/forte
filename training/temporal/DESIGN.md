# Frozen encoder and temporal fact head

This path targets the requested **100% accuracy on the genuinely reviewed set** and **under 10 seconds end to end from upload start** on supplied clips shorter than 143 seconds. Neither target has been achieved. The separate automated W1 track has now executed native-frame encoding, actual head fitting and held-out provided-binary validation; its 2/6 candidate result failed selection against the 3/6 untrained reference. Physical fact/event calibration and production activation remain unavailable. Perfect finite-set results would not establish perfect performance on unseen recordings. The reviewed-gold design below remains separate from automated W1.

The development corpus contains 42 recordings/40 unique contents. Weak binary/timestamp workbook labels do not provide the 19 factual states, visibility, event boundaries or support negatives needed for this head. A complete human review, scenario-group lock, exposure audit and sufficient validation coverage precede any feature extraction/training. A local VLM may propose records privately; provisional proposals are never ground truth without human adjudication.

## Interfaces and owners

- A15 binds exact weights, processor revision, license/source and artifact hashes. Root selected Google SigLIP2 base patch16-256 as the detail-first candidate, with patch32-256 as its speed comparator. Both use `SiglipVisionModel`/`SiglipImageProcessor`, pooled 768-dimensional features without L2 normalization. The immutable B16 manifest hash is `cd51863090e8c411a1227d8039e44118c44488243e3d615d73907106568877e7`, B32 is `e2bcad4f9ca992c71d2628e6f402d9120f8a50e1f6b3e594d4961844814ee0f9`; shared processor artifact hash is `d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff`. A15's `configs/serving/encoders/` manifests and `serving/FAST_PATH.md` remain authoritative for revisions, Apache-2.0 provenance and all artifacts. This directory downloads neither model. Public staging waits for A18's active benchmarks to drain.
- A05/A06 provide source-PTS-correct RGB feature batches, explicit coverage, global and optional ROI views, original-frame boxes and a refinement path. Feature coverage never implies adequate visual/procedural visibility. No JPEG is required for feature extraction; original evidence remains separately retrievable.
- A09 owns private feature-cache provenance, supervision and the small head. `targets.py` preserves unknown masks. `head.py`, `train_weak.py`, `checkpoint.py` and `evaluate_weak.py` have executed in the pinned GPU runtime; the immutable W1 result remains unselected. The encoder stays frozen and the reviewed-human E1 gate remains unchanged.
- A08 owns the canonical `forte.temporal-fact-head.v1` contract in `src/forte/decision/head_contract.py`: `HeadIdentity`, `TemporalHeadOutput`, source frame bindings, explicit proposal brackets and reviewed calibration readiness. Training imports its `FULL_FACT_COUNT` and `STATE_ORDER`; tensor output keys/axes match it. `FeatureSequence`/`SparseTargets` here describe private caches/supervision only, not a competing prediction schema. A11 owns grounded temporal candidates and rules. Window presence alone cannot create a localized event. A10 owns evaluation/selection; A20 owns the release decision.

## Frozen feature cache

Each private sequence uses features `[T,V,D]`, view masks `[T,V]`, view role (`global`/`roi`) and normalized original-frame `xyxy` boxes. Every requested valid frame retains at least one global view. All scene/ROI embeddings use the same selected encoder, feature kind and normalization policy. The head receives only feature tensors, actual timestamps, view geometry and validity masks. Content hashes, recording names, binary labels, scenario IDs, split IDs and teacher outputs are metadata and cannot be head inputs.

`FeatureSequence` binds canonical frame IDs to integer `source_pts`, rational `time_base_num/time_base_den`, integer `origin_pts`, normalized `pts_sec` and the next native source PTS time. It verifies the rational mapping and strictly ordered source time and recomputes every source ID through A05's pure `forte.video.batches.source_frame_id`; no JPEG path/FrameRef is fabricated. It carries content/group/split plus encoder-manifest, decoder, sampling and crop-config hashes. The complete cache artifact additionally needs its checksum, embedding dtype/shape, feature kind/normalization, stream/rotation/origin metadata, exact requested versus delivered frames, dropped/missing targets and inspected intervals. For the final native frame, a source-end bound must remain distinct from an observed next-frame timestamp. A cache key must bind every preprocessing choice; a stale cache is invalid, not approximately equivalent.

`feature_spec.build_feature_spec` keeps A15's complete encoder/pixel recipe verbatim and hashes it together with decoder, sampling, crop and global-view policies. Its digest binds A08's `HeadIdentity.feature_spec_sha256`; changing normalization, preprocessing or view policy invalidates the old calibration identity. The encoder-recipe hash alone is insufficient for that full binding.

A05's metadata audit found source intervals of about 33.333 or 66.667 ms, despite a much finer source clock. A sparse sampling grid is coarser still. Displaying milliseconds does not establish millisecond onset accuracy. Boundary outputs retain source frame brackets and native-frame intervals; dense local refinement must be measured and its remaining uncertainty reported.

## Reviewed targets and masks

`build_targets` accepts an approved canonical full-19-fact window and the matching frozen-feature frame IDs. It is a low-level supervision builder, not training authorization. The calling pipeline must first verify the underlying human annotation record/file hashes, locked scenario assignments and current exposure ledger. Its outputs are:

| Tensor | Labels | Unknown policy |
|---|---|---|
| Window states `[19]` | absent=0, present=1, explicitly reviewed unclear=2 | Unreviewed/unsupported labels use -100 |
| Frame states `[T,19]` | Only separately human-reviewed frame/fact states | All unmarked pairs use -100 |
| Frame support `[T,19]` | Explicit cited support=1; separately reviewed non-support=0 | Unmarked frames are not negatives |
| Event boundaries `[T,E,2]` | Explicit reviewed candidate start/end frame IDs; separately reviewed non-boundaries | No interval filling or nearest-frame invented labels |

Window `unclear` and frame `unclear` are different: the former can reflect missing procedure/context, while the latter can reflect the actual frame's observation quality. A window-level unclear label never fills all frame labels. Absent window labels require explicit human-reviewed complete context and adequate visibility; otherwise their training loss is masked. Present support does not label every neighboring frame. A normal/negative video label never generates absent facts. A missing candidate never makes all event/frame positions negative. Reviewed negative windows are useful sources for explicit factual and support adjudication, not a label-expansion shortcut.

Sparse positive-only support or boundary labels cannot identify a reliable discriminator. Aggregate coverage counts expose missing negative support, per-state/type coverage and missing boundaries. Training/calibration/selection must remain blocked when the reviewed negative and difficult-normal evidence required for a meaningful comparison is absent. Repeated same-type occurrences keep separate reviewed brackets. Neither bracket pairing nor suppression thresholds are invented by this prototype.

## Small temporal model and future training

The head projects frozen per-view features to 128 dimensions, includes view role/geometry and actual elapsed/gap timestamps, fuses valid views, applies a small masked temporal-convolution stack and pools the complete supplied window. It emits full-19 window and per-frame 3-state logits, per-frame support logits and optional per-event start/end logits. Features are detached, so the encoder receives no gradients. Padding and missing-view masks remain explicit. The sparse loss handles fully masked terms without turning them into negatives or producing a NaN. Parameter/compute scale is modest relative to the encoder, but no runtime or quality measurement follows from that description.

An eventual first experiment should train only this head on precomputed features, with a fixed seed, small bounded epoch/step budget and no broad architecture search. Group-based sampling prevents long recordings/correlated windows from overwhelming independent scenarios. All crops/windows share their source assignment. Each run must bind encoder/processor/cache/sampling hashes, 19 fact IDs, event IDs, review/target/split/exposure hashes, random seed, exact packages, head weights/checksum and local logs. Candidate artifacts remain unselected until A10 compares them against the baseline under identical reviewed inputs.

`experiment.example.json` records this proposed T1 binding; it is explicitly untrained/unselected and is not an executable training claim. Any future artifact must carry A08's exact `HeadIdentity` and its independently verified calibration manifest. Prototype raw tensors do not create those attestations.

Separate trusted validation must set any per-fact threshold/margin, support threshold, uncertainty policy and event-proposal decoder. Calibrated confidence is not assumed from logits. A08 must return unclear when calibration, visibility, procedure context, source support or a valid localization bracket is missing. A11 still requires applicable verified rules. A head cannot infer an exact event or legal conclusion solely from a pooled positive logit.

## Exposure and release gates

Before any real temporal training, call A02's shared `scripts.split_gate.validate_locked_training_inputs` and reject unless `ok is True`. It combines the append-only `benchmarks/private/model-exposure.jsonl`, fresh complete accepted-job inventory coverage, latest exposed-hash scenario audit, approved exposure-prefix/content-set binding, independent expected-assignment checksum and approved train-only requested IDs. The existing LoRA entry point is wired to the same gate. Exposed contents exclude their entire reviewed scenario from final-check. This exposure record is not training permission; only the reviewed A20 lock authorizes train groups. Validation/final groups never become train data, and every derived cache/frame/crop inherits its original assignment. A final-check set is consumed once after model selection, with exposure role/provenance recorded honestly.

The 10-second gate starts when the public upload begins and ends only when the complete valid result, including required JPEG evidence, reaches the client. Benchmark upload bytes/network, decode, encoder shapes/views, temporal head, refinement, evidence, queue and full response transfer separately, plus total. Report cold/warm and cache hits separately. A whole-content cache or source-label lookup cannot establish uncached inference speed. Adaptive frame/ROI budgets require a miss/quality comparison and complete tail coverage. A deadline miss, failed refinement, missing evidence or uncalibrated head cannot be reported as a clean result.

Release requires actual reviewed-set accuracy with all failures counted, no grounding/evidence regression, honest uncertainty and group coverage, actual uncached under-10-second end-to-end measurements on every claimed supplied clip, and verified baseline fallback. Current status: design/prototype only; human-reviewed targets and a locked/exposure-safe split remain gates.
# Executable W1 addendum

The reviewed-gold requirements below apply to the original T1/E1 track. The
separate automated W1 track now has executable cache, optimizer/checkpoint and
validation commands in [README.md](README.md). W1 uses independently pinned
organizer binary labels, optional exact-provenance silver targets, and separately
pinned user-provided coarse temporal points. It does not require human review or
pretend that machine targets passed the human-gold gate. All actual GPU work is
serialized after the explicit A18/root release and fresh exposure check.

