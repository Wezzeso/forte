# Stable semantic identities across fresh upload jobs

17 October 2026. Generated-video and synthetic observation tests only; no private media or model calls.

`make_clip_requests` previously constructed each clip ID from the upload job ID plus the window index. The same source bytes uploaded under two new jobs therefore produced different `CandidateEvent.clip_id` values even when their source frames, physical observations and bounds were identical. This could change internal candidate projections used for repeatability comparison.

Inspection found that the real model prompt and HTTP adapter already omit ownership-only `job_id`, `clip_id` and private media paths. Their image blocks use source frame identities (or deterministic request-local aliases) and actual normalized source PTS. The old UUID-bearing clip ID was a downstream projection issue; this audit does not claim it changed the current real-model prompt.

`src/forte/video/sampling.py` now derives a versioned semantic clip ID from the request purpose, actual interval bounds and ordered source-frame identity/timestamp/geometry metadata. Canonical source-frame IDs bind immutable content SHA-256, native PTS and source time-base units. Ownership job IDs, JPEG paths and orchestration window indices are excluded. Coverage/refinement purposes remain distinct, and changing content, bounds or source observations changes the semantic identity.

The request's separate `job_id`, frame ownership and private JPEG paths remain unchanged. `ClipRequest` still checks frame/job consistency; the model adapter still rejects paths outside that job's media directory. A stable semantic ID does not authorize cross-job media access. Public API response shapes and original evidence source clocks are unchanged.

Validation:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_semantic_clip_identity.py tests/test_sampling.py tests/test_pipeline_sampling_review.py tests/test_model_adapter.py training/test_teacher_sampling.py tests/test_native_ledger_preparation.py -q
```

**102 tests and 27 subtests passed in 7.86 seconds.** New generated-source tests prove that two fresh job IDs for the same original content produce identical semantic clip IDs, actual prompts/schema hashes, image-data packets and source-bound candidate projections. They also prove that changing source bytes or bounds/purpose changes the ID, reindexing orchestration does not, and a forged cross-job JPEG path fails. No repeatability normalization was broadened and no semantic fields were stripped to suppress variance.

The earlier actual CPU native-ledger stage remains bound to its recorded image/module hashes. Its selected native frame IDs, source clocks, original JPEG hashes and anchors are unchanged by this clip-ID policy update. A05's pre-POST packet validation binds those physical inputs, rather than pretending an old runtime used this newer policy.

This fix removes an avoidable job-identity difference. Actual fresh-upload model verdict/type/timing/evidence repeatability and GPU/kernel behavior still require their declared measurements.
