# Qwen3.8 source-clock and original-evidence audit

Recorded 17 October 2026, Asia/Qyzylorda. This is a local CPU component audit for handoff phases D/F. It does not establish Qwen3.8 model quality, semantic onset accuracy, GPU readiness or upload-to-render latency. No private recording, external inference service or remote GPU was used.

## Correctness change

The established JPEG decoder now verifies that the decoded video stream has the probed stream index, rational time base and presentation origin. It rejects frames marked corrupt by FFmpeg before their pixels can be encoded as evidence. The native RGB decoder already had equivalent metadata/corruption checks; the change closes this gap in the ordinary screening and exact-JPEG paths. A mismatch raises `VideoDecodeError`; it cannot become a clean verdict.

Changed code: `src/forte/video/decode.py`. Added regressions: `tests/test_video_source_clock_guards.py`. No pipeline, source video, model, rule or deployment configuration was changed by this audit.

## Clock and provenance contracts checked

- `probe_video` obtains integer `start_pts` and `duration_ts` when available, retaining the reduced stream time base and the content SHA-256. It prefers video-stream duration rather than an audio duration.
- Native RGB identities retain integer source PTS, rational time-base numerator/denominator, integer origin, normalized seconds, the next native PTS or a distinct source-end bound, job/content identity, dimensions and applied orthogonal rotation. Native selection does not invent timestamps from nominal FPS.
- Full-source native iteration checks the original start, strict PTS ordering, delayed B-frame flush, actual tail and declared frame count. A sparse rate selection is distinct from complete native coverage.
- Evidence seek accepts only selected identities from a complete native ledger over the same owned source. It verifies the source hash, stream units/origin and file stat, groups targets by preceding original keyframe, then verifies every decoded native PTS/identity between the earliest/latest target in each group.
- Selected original decoded frames go directly to the original-resolution JPEG writer. Feature resizing, letterboxing and optional model crops are not used as replacement evidence. Orthogonal display rotation preserves source pixels; JPEG encoding itself is lossy. “Original JPEG evidence” means a JPEG encoded from the original decoded source frame, not an original camera JPEG bitstream.
- Returned evidence identities, directory ownership and JPEG SHA-256 are checked again. Failed seek operations remove their own private partial-output directory. A missing intended source frame fails: this path has no nearest-frame substitute or automatic full-forward fallback.

## Local validation

Executed using the existing `.venv` runtime, PyAV and FFprobe; generated fixture pixels only:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_source_clock_guards.py tests/test_video_decode.py tests/test_video_batches.py tests/test_video_evidence_seek.py tests/test_evidence.py -q
```

Result: **77 passed in 13.85 seconds**. The duration is test-suite runtime, not product latency.

Additional cross-component validation:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_integration.py tests/test_pipeline_review.py -q
```

Result: **25 passed in 17.93 seconds**, with one existing Starlette/httpx deprecation warning. These checks use generated videos and explicit synthetic observation providers to exercise API/worker/evidence integration; they make no learned-model accuracy claim.

The five new cases cover two mismatched asset/stream origins or indexes, an FFmpeg corrupt-frame flag, a deliberately skipped exact seek target, and three fresh native-decode/evidence passes. The repeated case uses a generated nonzero-origin, rotated H.264 video with B frames; each pass probes and decodes afresh with a source hash check and separately encodes its selected original JPEGs. All three passes produce equal native clock ledgers and evidence identities/times/JPEG hashes after excluding the intentionally fresh job ID. The input content hash remains unchanged. Existing cases cover VFR, B-frame seeks, source mutation, exact JPEG equality with forward decoding, bounds, subprocess deadlines/cancellation and failed-output cleanup.

The repeated codec test is component reproducibility only. It does not count as the required three fresh uploads through a warmed selected Qwen3.8 model, and it makes no claim about model-result repeatability.

## Decode sharing and remaining work

The ordinary pipeline already deduplicates timestamps across initial overlapping windows and decodes those samples in one forward pass. Refinement rounds and the default final evidence extractor can still require additional original-source passes. Share verified within-job source identities/packets or original JPEGs at integration time; retain fresh decode/preprocessing for declared repeatability trials. Do not cache outputs by a known file hash as a substitute for inference.

The optional native path already retains the completed source ledger and reuses materialized, checked original JPEGs during final assembly. It currently requests evidence in batches of up to three; each batch revalidates/reopens the source. This protects bounded operations but may repeat hashing or keyframe preroll across batches. Any broader per-job grouped-seek optimization needs an explicit event evidence bound, immutable ownership binding, cancellation and metrics. This audit does not change that pipeline contract.

Boundary RGB refinement can retry from the original presentation origin when a keyframe seek skips the required bracket. Its exact native PTS-set check still prevents a wrong-frame result, but the extra decode work must be included in runtime measurement. Original JPEG evidence seek has no such fallback.

Qwen3.8 processor/tokenizer/template integration still needs its own receipt binding model-visible frames back to these identities and documenting any processor resampling, timestamp rounding or frame omission. This source-layer check cannot establish that the model consumes all intended frames.

Native tick precision, frame quantization, reference-label tolerance and semantic onset uncertainty are separate quantities. Exact source-frame retrieval and millisecond display formatting do not establish one-millisecond event localization. Actual new-upload quality, all-attempt denominators, full warm-model repeatability and the less-than-ten-second timer remain to be measured by the selected-profile acceptance run.
