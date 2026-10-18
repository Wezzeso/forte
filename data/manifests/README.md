# Local dataset identity audit

Run from the repository root:

```powershell
python scripts/audit_data.py self-test
python scripts/audit_data.py audit
python scripts/audit_data.py check-split
python scripts/audit_data.py check-split --require-locked
```

The final command deliberately rejects an unapproved candidate. Audit uses
read-only workbook access, SHA-256, file metadata, and `ffprobe` video-stream
metadata. It never decodes or prints private video content, audio, row values,
captions, predictions, or evidence. The input workbook and every source video
are hashed again after the audit to confirm their bytes did not change.

`openpyxl` and an executable `ffprobe` are required. `--ffprobe` accepts an
explicit executable. Core recording fields are validated against
`forte.shared.schemas.RecordingRecord` when the repository package is present.
Audit fails on shared-schema validation errors.

## Private generated files

| File | Purpose |
| --- | --- |
| `recordings.jsonl` | One record per supplied file; stable path-derived recording ID and full content hash; original Unicode relative path; first video-stream metadata; local workbook row links; review flags |
| `workbook_rows.jsonl` | Local raw row values, parsed timestamp/label, explicit row-to-recording join result |
| `review_queue.jsonl` | One review task per unique content group; duplicate/timestamp conflicts; local review remains pending |
| `audit_summary.json` | Aggregate counts, hash reconciliation, decoder metadata availability, source preservation, and elapsed time |

Per-recording identifiers remain stable while their relative paths remain
unchanged; content hashes and exact-content group IDs remain stable across
renames. The path ID is deliberately different from the content ID because
duplicate files need distinct workbook links. Never join a recording solely
on a shortened hash or on a visually similar filename.

Filename joins first try NFC-normalized original filenames, then NFC stems,
then an explicit leading Latin `X`/Cyrillic `Х` alias. Every accepted strategy
is recorded. Ambiguous candidates are rejected and preserved for review.
Normalization never silently equates these two letters. A join to a unique
NFC stem is allowed when workbook names omit the extension.

Timestamps support numeric seconds, `MM:SS`/`HH:MM:SS`, spreadsheet time values,
and the supplied Russian minute/second unit forms. Parsed workbook labels
and timestamps are annotation suggestions requiring local review. A label
timestamp does not establish that every other interval is negative.

The current local audit reconciles all 42 rows/files, reports 40 unique
contents and two byte-identical pairs, and flags one timestamp disagreement.
No video was visually reviewed by this audit.

## Production and source-control exclusion

These are development data artifacts. Exclude the following paths from the
production image, production bind mounts, package data, release archives,
public static files, Git, and model prompts:

```text
videos/
open_labels.xlsx
data/manifests/*.jsonl
data/manifests/audit_summary.json
data/splits/*.json
annotations/private/
```

The production service must accept only job-owned uploads and permitted model
and rule assets. It must not read organizer labels, these filenames, dataset
splits, annotation targets, or precomputed organizer answers to decide a
verdict. Training/evaluation tooling may access the private development
mounts only after split approval. Upload storage is a separate runtime path.
Public annotation schema/taxonomy code and generic task definitions may be
packaged; only private annotation records and targets are excluded.

All CLI output is aggregate JSON. Keep private files local; do not paste
their rows or screenshots into cloud-backed assistants. Review the media
through the local review UI or an approved local model inside the service.
