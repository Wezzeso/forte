# ForteBank service requirements and traceability

Status: requirements extracted; this document does not certify an implemented or deployed service. Reviewed on 17 October 2026, Asia/Qyzylorda.

## Sources and authority

- **O1: supplied organizer brief**, [Кейс_ForteBank.docx.pdf](../Кейс_ForteBank.docx.pdf), seven pages, sections 1-10 and FAQ A-G. SHA-256: `376d6fc5970f406a29211cc00ccd732d837eea900ca79f5a6dfa016f00aadf8f`. The complete PDF text was extracted locally with `pdftotext -layout`; no private videos, annotations, or model predictions were needed. Organizer requirements below cite its section or FAQ item.
- **P1: execution plan**, [FORTE_20_AGENT_EXECUTION_PLAN.md](../FORTE_20_AGENT_EXECUTION_PLAN.md), especially sections 2, 4, 5, and A01. P1 supplies implementation choices and additional safeguards; these are marked **team policy**, not attributed to the organizer.
- **C1: canonical code contracts**, [schemas.py](../src/forte/shared/schemas.py) and [contracts](../contracts/). Code and fixtures are maintained by A01. This requirements document is a traceability guide; consumers must import canonical contracts rather than reimplement their own enums.

Only the supplied PDF was verified as organizer evidence. No event deadline date, hidden scoring formula, live endpoint, current legal clause, model license, dataset accuracy, or performance result is established by this extraction. Legal research and dependency licenses need their own verified records.

## Mandatory user interface and functional result

| ID | Requirement | Organizer source | Owner / verification evidence |
|---|---|---|---|
| UI-01 | Deliver working frontend and backend. A user can upload a new video and receive a result without manually running code. | Sections 2, 4.1, 5 | A13, A16, A20 / end-to-end upload |
| UI-02 | Display analysis status while processing. | Section 4.2 | A16 / processing view |
| UI-03 | Show exactly the two completed outcomes: no violation found, or signs of possible violations found. Explanations remain support for human review. | Sections 4.3, 5, 7 | A01, A16 / positive and negative UI checks |
| UI-04 | Support several possible violations in one video; show a type, timestamp, and description for each event. | Sections 4.4, 5 | A11, A16 / multiple-event example |
| UI-05 | Show one or more original evidence frames from the relevant fragment so a human can check the conclusion. The API constrains this to 1-3 JPEG frames per event. | Sections 4.5, 5, 10 | A05, A12, A16 / frame and timeline check |
| UI-06 | Explain concisely what is visible and why it may indicate a violation. Avoid unsupported assumptions. | Sections 4.6, 5, 7 | A08, A11, A16 / reviewed result |
| UI-07 | Identify the relevant rule and normative source, including document title and clause/section when available; retain the FAQ qualification below. | Sections 4.7, 6; FAQ 31 | A04, A11, A16 / rule-source trace |
| UI-08 | Make the working UI publicly reachable at a URL. The jury uploads remotely. | Sections 8, 10; FAQ 13, 23 | A17, A20 / outside-network smoke check |

The interface language is the team's choice (FAQ 26). Russian is P1's default. User accounts, history, report export, and batch upload are not required (FAQ 27); this does not remove the required API-key authentication.

## Exact jury API

The exact public routes and spellings are mandatory. A deployed origin is not yet established by this document.

| Method and exact route | Request | Required result |
|---|---|---|
| `POST /api/v1/analyze` | `X-API-Key` with a valid team-issued key; `multipart/form-data`; `.mov` in field `file` | JSON containing `job_id` |
| `GET /api/v1/jobs/{job_id}` | `X-API-Key` with a valid team-issued key | JSON containing the job status and the result when complete |

| ID | Requirement | Organizer source | Owner / verification evidence |
|---|---|---|---|
| API-01 | Expose both routes publicly in addition to the UI. | Sections 4.8, 10 | A13, A17 / external route smoke check |
| API-02 | The team creates the API key and supplies it to the jury. Require `X-API-Key` and reject requests without a correct key on both routes. | Sections 8, 10 | A13, A20 / missing/wrong/valid-key checks |
| API-03 | Accept `.mov` uploads in multipart field `file` and return a `job_id`. Approximately 29 MB is the described file size, not a published hard upper bound. | Section 10; FAQ 1, 25 | A13 / MOV upload check |
| API-04 | Implement asynchronous upload and polling. Public statuses are exactly `processing`, `done`, and `failed`. | Section 10 | A01, A13, A14 / state fixtures |
| API-05 | Return completed JSON with `job_id`, `status`, `verdict`, and `violations`; each event has the exact fields below. | Section 10 | A01, A13 / fixture and response validation |
| API-06 | Return `verdict` as exactly `violation` or `no_violation`. `violations` is empty for no violation and contains detected events otherwise. | Sections 4, 5, 10 | A01, A11 / list-verdict invariants |
| API-07 | Provide a result within five minutes. An unavailable service or a missing result at five minutes makes the video unprocessed. | Section 10; FAQ 14 | A14, A15, A18, A20 / recorded end-to-end timings |

Required completed-result structure (the text values here describe fields; this is not a valid JPEG fixture):

```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Human-readable event type",
      "timecode_sec": 44.2,
      "description": "Visible observation, possible rule implication, and verified document/section reference.",
      "evidence_frames": [
        {"timestamp_sec": 44.2, "image_base64": "<raw JPEG base64>"}
      ]
    }
  ]
}
```

| ID | Output field or invariant | Source and validation boundary |
|---|---|---|
| OUT-01 | `type`: understandable event type; the team defines the taxonomy. | Sections 3, 4, 10; FAQ 2, 8. No closed organizer taxonomy is provided. |
| OUT-02 | `timecode_sec`: event time in seconds from the beginning of the original video. | Section 10. Nonnegative numeric validation is structural; duration bounds and semantic correspondence need the video pipeline. |
| OUT-03 | `description`: concise visible action and why it may be a violation, with the relevant verified source. | Sections 4.6-4.7, 5, 6, 7, 10. Human/local review checks grounding; JSON validation cannot prove it. |
| OUT-04 | `evidence_frames`: 1-3 JPEG images per event, base64 encoded. | Section 10. Decode base64 and JPEG bytes; references/URLs do not satisfy this field. |
| OUT-05 | Each evidence entry has `timestamp_sec` and `image_base64`; its declared time matches the actual source frame. | Organizer JSON example and localization requirement, section 10. The last property needs source-frame identity. |
| OUT-06 | Support more than one event. | Section 4.4. No organizer maximum event count is published. |
| OUT-07 | A valid done result has a verdict; `violation` requires at least one event and `no_violation` requires none. | Organizer result meaning, section 10; P1 makes the cross-field invariants explicit. |
| OUT-08 | Only publish success after evidence and the full public result validate. Do not turn decode failure, timeout, or missing evidence into `no_violation`. | Team policy from P1 sections 3-4; preserves the distinction between analysis failure and a negative finding. |

The organizer does not prescribe a POST success HTTP status, complete error body, unknown-job HTTP status, upload limit, job-ID format, timestamp matching tolerance, or processing-stage detail. Team choices in P1 are HTTP 202 after durable acceptance, HTTP 404 for unknown jobs, a configurable initial upload limit of at least 256 MiB, a minimal failed body, and mapping queued/claimed/decoding/screening/refining/assembling to public `processing`. Do not present those choices as organizer rules. Keep the public result schema stable; richer progress and diagnostics belong in internal/UI records.

## Locality, evidence, data, and component use

| ID | Requirement | Organizer source | Owner / verification evidence |
|---|---|---|---|
| LOC-01 | Execute every model inside the team's deployed service. Team resources supply the compute; the organizer supplies no GPUs or cloud credits. | Section 7; FAQ 21-22 | A07, A15, A17 / runtime topology |
| LOC-02 | After downloading weights, process video without internet access so deployment can move inside a bank's internal environment. | Sections 2, 7 | A07, A15, A17, A20 / egress-disabled run with local artifacts |
| LOC-03 | Do not call cloud AI services/APIs while processing, including cloud LLMs, multimodal models, and recognition services. Never send them video, frames, or descriptions. | Section 7; FAQ 21, 35 | All agents / network and implementation review |
| LOC-04 | Analyze visuals only; do not use audio. | FAQ 1, 28 | A05, A07 / visual-only decode and model path |
| LOC-05 | Do not recognize or identify faces. | FAQ 29 | A07, A08 / model prompt and feature review |
| DATA-01 | Do not train on, publish, or transfer jury test videos to third parties. | Section 10 | A09, A17, A20 / release/runtime data separation |
| DATA-02 | Use supplied data only within the hackathon. The organizer's FAQ does not specify a concrete retention period or transfer mechanism. | FAQ 6 | A02, A17, A20 / data-use records |
| LAW-01 | Research public Kazakhstan normative sources relevant to cash operations, collection, cash premises/security, and relevant banking procedures. Show the relationship among source, proposed event, and visible fragment. | Section 6; FAQ 30 | A04, A11 / verified rule catalog |
| LAW-02 | Do not claim a violation when the required facts are unavailable on video, such as transaction correctness or system reconciliation. No production banking systems, internal logs, or infrastructure are supplied. | Section 7 | A04, A08, A11 / observable fact definitions |
| LAW-03 | Treat results as assistance for human verification, not final legal or disciplinary conclusions. | Sections 1, 7 | A08, A11, A16 / possible-sign wording |
| LIC-01 | Prewritten code, templates, and pretrained models are allowed when open source and licensed for commercial use. Record actual component/model licenses before release. | FAQ 24 | A07, A17, A20 / model manifest and dependency license inventory |
| LIC-02 | Do not submit a ready-made solution or use prohibited cloud solutions by API; the FAQ identifies these as disqualification grounds. | FAQ 35 | A20 / original case-specific implementation record |
| LIC-03 | Account for the organizer's stated right to use the solution commercially. This document records the brief and does not add licensing terms. | FAQ 34 | A20 / submission license review |

P1 additionally requires that cloud-backed coding-agent outputs expose no supplied frames, per-video captions, observed-action descriptions, raw predictions, or private annotations; use aggregate diagnostics and opaque IDs. Keep raw videos/labels/weights read-only, exclude labels from production inference, and never use filenames or label lookup as a prediction path. These are team safeguards implementing the organizer's locality and data boundaries.

## Submission and event logistics

| ID | Required submission item | Organizer source | Owner |
|---|---|---|---|
| SUB-01 | Working public UI address, API address, and jury API key. Do not commit a real secret to documentation or fixtures. | Sections 8, 10; secret handling is team policy | A13, A17, A20 |
| SUB-02 | Startup and usage instructions. | Section 8 | A17, A20 |
| SUB-03 | Brief architecture, selected approach, and models. | Section 8 | A07, A15, A20 |
| SUB-04 | Discovered normative sources and the method for matching rules to video. | Section 8 | A04, A11, A20 |
| SUB-05 | Supported event types with short descriptions. | Section 8 | A03, A11, A20 |
| SUB-06 | Limitations and known error cases. | Section 8 | A10, A18, A20 |

FAQ 12 describes 5+ final test videos; this is not a published simultaneous-load requirement. Evaluation is automatic against JSON (FAQ 18); the first criterion is a whole-video verdict (FAQ 15). Another camera angle appears in final checking (FAQ 4). Shortlisted teams have five minutes to pitch and two minutes for questions, slides are optional, and ties are settled by pitching (FAQ 19-20). The deadline is stated only as **12:00** (FAQ 33); the calendar date and timezone remain unverified and must not be invented.

## Ambiguities and conservative implementation decisions

| Issue | Organizer wording | Team decision / unresolved boundary |
|---|---|---|
| Uncertainty versus two outcomes | Sections 4, 5, 10 mandate two outcomes; FAQ 15 discusses unjustified use of insufficient data. | Public `verdict` stays binary. Preserve visibility/uncertainty internally and disclose limitations in the UI. Technical failures remain `failed`. Organizer clarification is still needed for any different public shape. |
| Positive event plus unclear fragment | FAQ 11 refers back to the mandatory scenario without a third outcome. | One supported event makes the whole-video result `violation`; do not suppress it because an unrelated interval is unclear. |
| Exact source clauses | Section 4.7 asks for document and clause/section; FAQ 31 accepts references without exact clauses and other banks' public documents. | Seek an exact applicable current RK source. Disclose missing specificity. Do not represent another bank's policy as ForteBank policy or as universally binding law. FAQ 30's RK jurisdiction remains relevant. |
| Internal ForteBank rules | Full internal regulations are not supplied; FAQ 32 says dataset events do not depend on internal rules without an open norm. | Do not invent internal ForteBank rules or unobservable system facts. |
| Pretrained versus ready-made | FAQ 24 permits open-source commercially usable components; FAQ 35 prohibits ready-made solutions. | Document original case-specific pipeline, rule matching, evaluation, and UI. The organizer does not define a precise ready-made threshold. |
| Five-minute cutoff | No exact clock origin or response-transfer boundary is given. | Measure upload start/completion, durable acceptance, result publication, and full response receipt. Budget conservatively from upload start. P1's 240-second safety objective is a team target. |
| Omission events | FAQ 10 leaves timestamp selection to the team. | Define the required action, observed opportunity/interval, and defensible anchor in each relevant rule; no hidden matching tolerance is assumed. |
| Error bodies and extra fields | Only processing statuses and completed example are prescribed. | Preserve required public fields and enums; keep detailed diagnostics internal/UI unless A01 explicitly versions compatibility. |
| No matching/scoring formula | No numerical tolerance or FP/FN tradeoff is published. FAQ 16 asks to minimize false alarms and avoid misses. | Report measured errors and distributions; do not claim a score formula, guaranteed accuracy, or a latency bonus. |

## Traceability checklist

Unchecked items are required release evidence, not reported failures or completed implementation. Populate results with actual commands/reports in A20's integration records.

- [ ] UI-01 through UI-08: new upload, progress, binary outcome, multiple events, evidence, sources, and external URL verified.
- [ ] API-01 through API-06: both exact routes tested externally; missing/wrong keys rejected; valid key works; multipart `file` accepts MOV.
- [ ] OUT-01 through OUT-07: positive, negative, processing, failed, and multiple-event synthetic fixtures validate; illegal enums and inconsistent lists fail.
- [ ] OUT-02, OUT-04, OUT-05: JPEG bytes decode; every time is within the original timeline; declared evidence time maps to the original frame. Private local review verifies relevance.
- [ ] OUT-08: corrupt input, worker failure, missing evidence, and deadline exhaustion produce honest failure, never a clean negative result.
- [ ] API-07: measured upload-to-result and transfer timing meet the 300-second cutoff on development cases; P1 safety target and concurrency tests are labeled separately.
- [ ] LOC-01 through LOC-05: local inference succeeds with egress disabled; no cloud AI, audio, or face-identification path exists.
- [ ] DATA-01 through DATA-02: jury data excluded from training/publication/third-party transfer; private data and labels absent from production configuration.
- [ ] LAW-01 through LAW-03: enabled rules have source provenance, applicability, observable requirements, and human-review wording; unresolved rules stay disabled or explicitly provisional.
- [ ] LIC-01 through LIC-03: exact versions/revisions and commercially usable licenses recorded; original case-specific work and submission rights reviewed.
- [ ] SUB-01 through SUB-06: final URLs/key handoff, startup/use instructions, architecture/models, sources/matching, taxonomy, limitations, and known errors prepared.
- [ ] Deadline calendar date and timezone confirmed through organizer evidence; no unverified date treated as a release fact.
- [ ] [Rubric evidence](rubric.md) covers all seven criteria; planned targets and unmeasured work are clearly marked.
