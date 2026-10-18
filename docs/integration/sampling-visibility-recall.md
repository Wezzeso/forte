# Sampling visibility and missed-positive diagnosis

17 October 2026, Asia/Qyzylorda. Local generated-video analysis only: no private pixels, captions, raw observations, model calls or GPU changes. Existing sampling, prompt and pipeline files were not modified by this audit.

## Executable visibility counterexample

```powershell
$env:PYTHONPATH='src;.'
.venv\Scripts\python.exe -m qa.sampling_visibility_probe --output qa/integration/sampling-visibility-synthetic.json
.venv\Scripts\python.exe -m pytest tests/test_sampling_visibility_probe.py tests/test_sampling.py -q
```

Validation: **29 passed in 0.73 seconds**. The probe creates and deletes its own generated H.264 source and JPEGs. The retained JSON contains only synthetic timings/counts, with `model_calls=0` and unchanged source-byte verification.

The generated eight-second, 25-fps clip has 200 native frames. A plainly visible synthetic red marker exists at 0.20, 0.24, 0.28 and 0.32 seconds: 160 ms of source visibility. This is a colored-marker oracle, not a bank-rule annotation.

| Profile | Requests | Frame uses | Unique source frames | Largest selected source gap | Visible marker frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current 8-s windows / 4-s stride / nominal 2 fps | 1 | 16 | 16 | 0.56 s | 0 |
| Diagnostic 4-s windows / 2-s stride / nominal 4 fps | 3 | 48 | 47 | 0.28 s | 1 |

The current profile selects 0.00, 0.52, 1.08 seconds and subsequent points, skipping every marker frame. If every request succeeds, its interval-coverage metadata is still complete. Motion analysis of those sampled frames returns zero cues because none contains the action. This proves that complete sparse request coverage does not establish continuous visual coverage. It does not establish which real user uploads contain brief missed actions, nor their true violation labels.

The nominal sample rate is approximate: `_timestamps` requests `ceil(span * fps)` end-inclusive samples. An eight-second window therefore has 15 intervals between 16 targets, with a 0.5333-s requested gap, rather than exactly 0.5 s. Nearest native-frame selection can increase an individual actual gap; the generated 25-fps fixture reaches 0.56 s.

## Refinement can lose temporal density

Current refinement uses `fps = min(4, max_frames_per_window / span)` with a 16-frame cap. Longer procedural context is therefore not necessarily a denser observation:

| Context span | Nominal requested fps | Requested frames | Largest target gap |
| --- | ---: | ---: | ---: |
| 2 s | 4 | 8 | 0.286 s |
| 4 s | 4 | 16 | 0.267 s |
| 8 s | 2 | 16 | 0.533 s |
| 16 s | 1 | 16 | 1.067 s |
| 32 s | 0.5 | 16 | 2.133 s |

Expanded context can support procedure interpretation while missing a brief physical action. Preserve the parent context and add separate dense action brackets instead of assuming one widened, fixed-frame request serves both purposes.

## Pixel and source-clock findings

`LocalModelProvider._prepare_media` preserves each JPEG's width and height, strips metadata, and re-encodes at JPEG quality 95. It contains no 512-pixel resize. Its `512 * 1024` setting is the maximum model-output byte count. The checked local baseline serving configuration declares a processor maximum of 1,048,576 pixels. These source observations do not prove the effective live processor grid or token allocation.

The standard adapter records source `input_dimensions` separately from `processor_output_dimensions`; actual processed dimensions and visual-token counts are unavailable when the server does not report them. Do not interpret source dimensions or combined prompt-token usage as a measured processor output grid.

The prompt and image blocks carry each frame's actual normalized source PTS and the same frame ID. Compact aliases preserve order and restore exact original identities before event assembly. No frame-index/FPS substitution or clip-relative/global-clock mismatch was found in the inspected path. Correct timestamps cannot recover pixels that were never sampled.

## Bounded quality-fix proposal

1. Address the screening shortcut first. An absent screen fact across sparse samples does not itself prove that a brief action never occurred or that a full procedure was visible. Require supported context/visibility gates before treating every rule branch as refuted. Otherwise route the clip to full observation or retain insufficient analysis. A14 owns the catalogue/screening counterexample; this probe does not change those semantics.
2. For an explicitly declared diagnostic, disable staged screening while keeping the same source sampling and pixel policy. This isolates the shortcut from resolution/rate changes. Compare every attempted file and failure; this is an evaluation proposal, not an authorization for new model calls.
3. Separately compare the bounded 4-s/2-s-stride/nominal-4-fps profile above, retaining the 16-image request cap. It recovers this marker but triples calls/frame uses for the eight-second case and can reduce procedural context. It cannot promise recall of every sub-280-ms action or meet the ten-second target without measurement.
4. A subsequent implementation can compute inexpensive native-frame thumbnail changes during the existing forward decode, retaining source PTS brackets and selecting additional original frames around changes. Computing motion only on already sparse JPEGs cannot recover this counterexample. Motion proposals must supplement baseline coverage; static or low-motion violations must not become automatic negatives. Bound extra brackets, check resulting maximum gaps, and preserve incomplete/error outcomes when the budget is exceeded.
5. If a strict maximum sampling gap is required, use an endpoint-aware sample count (`ceil(span / max_gap) + 1`) and reconcile it with the image/window budget. Silently adding a seventeenth frame to the existing 16-frame serving envelope or dropping an endpoint would break a separate contract.

The first two controls address false clean conclusions. The latter controls address temporal visibility and require a separate recall/runtime comparison. None establishes 100% correctness, millisecond semantic onset accuracy, or a less-than-ten-second upload-to-render result.
