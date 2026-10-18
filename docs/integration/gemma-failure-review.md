# Gemma failure-semantics review — 18 October 2026

Scope: read-only review of the local Fast analyzer, VLM transport, worker, evaluator,
synthetic preflight and new-model bridge payloads. No supplied images, model scene
descriptions, private reports or host logs were opened; no host commands or queue jobs
were run. Only this document was changed.

## Conclusion

Ordinary request, JSON and schema failures in the service do propagate to a failed job.
The catches returning empty states or `visible=false` do not by themselves convert those
exceptions into a clean verdict: `_CheckedClient` records each exception and `analyze`
rejects the video before returning. Verification errors also have an explicit rejection.

There are compatibility gaps in the runtime transport and geometry validation. They are
known limitations, not evidence that any completed real-data Gemma run suffered these
errors. The synthetic Gemma preflight documented in `agent-log.md` passed its stricter
plain-JSON, reasoning, finish-reason, concise-stop and coordinate checks. Do not invalidate
measured runs without observing an actual failure, and do not claim runtime enforcement
that the current transport does not provide.

## Findings requiring consideration before promotion

1. **Response finish reason and reasoning are not enforced at runtime.**
   `src/forte/fast/vlm.py:166–171` takes `message.content` without checking
   `finish_reason`, `reasoning` or `reasoning_content`.
   `parse_json` at line 49 accepts fenced or embedded JSON, including JSON following a
   thinking prefix. Consequently schema-valid JSON can be accepted after a `length`
   finish. In concise mode, `close_truncated_object` can repair a response that actually
   exhausted its token budget after a complete decision field. That must be distinguished
   from the intentional verifier stop: `finish_reason=stop` after `VERIFY_STOP` with all
   decision fields present is expected and valid. A future default-off strict transport
   flag should reject genuine `length` finishes, nonempty reasoning and non-JSON prefixes
   using static errors, while preserving the intentional concise stop.

2. **Layout geometry has no runtime convention check.**
   `prompts.py:51–58` requires numeric arrays but no range, order or coordinate-scale
   constraints. `_layout_one` at `analyzer.py:154–166` silently discards out-of-range or
   reversed boxes; a box in a different 0–1 convention passes the 0–1000 bounds check and
   yields a tiny misplaced crop. All-null boxes are also schema-valid. Missing/occluded
   objects can legitimately be null, so blindly failing every null is inappropriate.
   The passing synthetic normalized-box case supports the current Gemma convention;
   it does not prove every runtime layout is correct. Any coordinate adapter requires a
   declared flag and observed model convention, as the handoff requires.

3. **Offline evaluation is not the five-minute API test.**
   `tools/fast_eval.py:213–227` does not place a whole-video deadline around `run_one`.
   The current `model_eval_twice.remote.sh` invokes its accuracy gate without
   `--max-wall-s`. Individual HTTP requests have timeouts and retries, but semaphore
   waits plus many requests can make a video much slower than five minutes. A passing
   accuracy gate therefore cannot replace the required six parallel longest-clip API
   check with all jobs completing within 285 seconds.

## Synthetic confirmation

A local `httpx.MockTransport` diagnostic used no media and made zero network requests.
It called the existing `VLMClient.chat_json` with a two-value enum schema. Results:

| Synthetic response | Current result |
|---|---|
| Valid decision JSON, `finish_reason=length` | Accepted |
| Valid JSON plus nonempty reasoning field | Accepted |
| Thinking prefix followed by valid JSON | Accepted |
| Complete field plus comma, `truncated=true`, `finish_reason=length` | Repaired and accepted |
| JSON ending inside a field value | Rejected with `ValueError` |

These confirm code paths only. They are not model-quality measurements, supplied-video
failures, evidence of leakage, or accuracy results. No test or core implementation file
was added or edited in this review.

## Failure handling that is already correct

- `_CheckedClient` (`analyzer.py:17–34`) records request, parsing and schema exceptions
  per video even when downstream helpers catch them. At lines 729–730 any recorded
  failure raises `fast_model_observation_failed`; no `Analysis` is returned.
- `verify` includes an exception-class-only error marker; `analyze` rejects it at
  lines 715–716 instead of treating it as a negative observation.
- Nonfinite generated numbers are rejected by `vlm.py:34–46` before schema validation.
- `FastPipeline.analyze` (`service.py:171–193`) bounds native decode and model analysis
  by the remaining job deadline; decode receives cancellation and deadline controls.
  Result schema validation precedes publication.
- `Worker.process_claimed` (`worker.py:145–152`) records exceptions as `failed` using
  static error codes. `safe_error_code` does not publish exception messages. The outer
  worker deadline also covers assembly and publication.
- Malformed envelope types not caught by the VLM retry clause still escape to
  `_CheckedClient` and fail the job. They do not become clean results merely because
  they are not retried.
- `tools/model_schema_smoke.py:213–241` independently rejects reasoning, thought
  markup, actual length finishes, bad geometry and a nonhonored concise stop, and
  outputs error class names only. Its synthetic-only nature is explicit.

## Logging and export boundaries

- `fast_eval.py:77–79` writes full events/candidates/verifications to its private host
  report, including description fields. Its line 221 stores `repr(exc)`. Keep these
  files private on the user's rented host; do not read or transfer them to the agent.
  Unknown exception text can include generated content or library payloads.
- `collect_models.remote.sh` projects reports through explicit enum, numeric, boolean,
  box and hash allowlists. It extracts only known exception class names, recomputes
  processed/failed totals, and exports a failed verdict as null. This is the correct
  agent-visible export path.
- `model_diagnostics.remote.sh` redirects library stdout/stderr and exports static
  status/error classes; its probes and teacher responses are schema-checked enums.
  The older standalone `teacher_windows.py` lacks that protective terminal wrapper;
  unexpected library tracebacks should not be copied to agent-visible logs.
- The BF16 `gemma4_serve.remote.sh:17` still prints filtered raw startup-log lines.
  Startup currently precedes supplied-video inference, which limits exposure, but a
  filter by exception keyword is not a general privacy sanitizer. The newer FP8
  payload keeps startup logs private and emits allowlisted error categories/classes.
- Both Gemma serving payloads disable request/output logging and set offline model
  flags. The FP8 payload also asserts an internal Docker network with no default route.
  No service behavior, serving option, threshold, label or accuracy parameter changed
  during this review.
