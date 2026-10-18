# Forte video reviewer

Russian reviewer UI for the Forte API, using same origin by default. Production code contains no sample verdicts, simulated progress, external AI calls, CDN assets, or bundled API key.

## Run

Use Node.js 22.12 or newer (verified locally with Node 24.12.0). From this directory:

```powershell
npm ci
npm run dev
```

Vite proxies `/api` and `/readyz` to the local API at `http://127.0.0.1:8000`. The UI opens at `http://127.0.0.1:5173`. Enter an operator-provided key through **Ключ доступа**, select a MOV file, then choose **Начать проверку**. The key stays in tab-scoped `sessionStorage`, is sent using `X-API-Key`, and is never added to the build or URL. Closing the tab ends the browser session; **Отключить доступ** removes the stored credential.

```powershell
npm run build
```

Build output is `web/dist`. The API serves it at the root when present at startup. The build imports the sibling `brand/theme.css` and its font, so container builders must copy `brand/` alongside `web/` before building. Open Sans is local and its OFL license is copied to `dist/OpenSans-OFL.txt`. No frontend environment variable should contain a secret.

## Optional operator API origin

Before mounting, the browser fetches same-origin `/runtime-config.json` with `Cache-Control: no-store` request behavior and a 1.5-second timeout. An operator may serve this nonsecret JSON with `Content-Type: application/json` and `Cache-Control: no-store`:

```json
{"apiBaseUrl":"https://trusted-api.example"}
```

The URL must be an absolute HTTPS root origin, without credentials, path, query or fragment. An optional trailing slash is accepted. The operator must configure the API's CORS allowlist for the exact frontend origin and approve `X-API-Key`, `Content-Type` and `Range`. Every API route, authenticated original-video fetch and readiness request uses this origin. Missing, unavailable or invalid configuration falls back to same origin. Query strings, browser storage, form inputs and global variables cannot override the API destination. Keys remain separate and are never placed in this JSON. API requests omit cookies and refuse redirects.

Successful processing polls run every 500ms for the first 10 seconds, then every 1.5 seconds; network failures retain bounded retry backoff. This reduces result-display delay and does not assert an inference or end-to-end completion time.

## Endpoint contract

- `POST /api/v1/analyze`: multipart `file`; returns durable `job_id`.
- `GET /api/v1/jobs/{job_id}`: authoritative jury response; polled sequentially with bounded retry backoff.
- `GET /api/ui/jobs/{job_id}`: optional stage, elapsed/completion timestamps, visibility limitations and generic failure code. Absence does not replace or alter jury results.
- `GET /api/ui/jobs/{job_id}/video`: authenticated original media, fetched as a local blob URL when a browser session resumes a completed task.
- `GET /readyz`: actual readiness, checked without credentials.

Events are displayed in source-time order, preserving their metadata association. Selecting a timestamp seeks the original video; JPEG evidence can be enlarged. Unsupported MOV codecs produce a playback explanation and a local-original download action. Detection remains the backend's responsibility. The UI never turns network failures, processing errors, or an invalid response into a clean verdict. Optional structured source metadata is displayed when supplied; otherwise the complete canonical jury description retains the source reference.

The review workspace starts with portrait proportions and adapts to the source video's decoded dimensions. Portrait playback stays alongside events and evidence on desktop, and stacks above them on mobile. The player and evidence use containment to preserve the full image. Selected uploads include an inline preview. Interface icons use `@phosphor-icons/react`; original SVG illustrations in `src/Illustration.tsx` provide decorative artwork without icon tiles.

## Human review of model findings

Positive model findings remain provisional. The selected event has a separate **Решение человека** panel backed by authenticated review endpoints. The reviewer enters a claimed name (maximum 120 characters), optional comment (maximum 4000), and explicitly confirms or rejects that finding. The UI states that the name is not a verified identity. There is no manual training-annotation prerequisite.

`GET /api/ui/jobs/{id}/reviews` supplies immutable result/evidence SHA-256 bindings and canonical `violation_index` values. `POST .../reviews/{index}` saves `confirm` or `reject` with `expected_revision`; initial `pending` state has revision zero. Chronological presentation preserves the original event index. Version conflicts refresh the saved state while retaining the entered name and per-event comment; resubmission remains an explicit reviewer action. **История решений** loads append-only revisions with `result_sha256`, cursor and a 50-record limit. Failed reads or saves never imply a saved human decision.

Human decisions do not rewrite the model verdict, its canonical JSON, evidence or completion time. **JSON модели** downloads the original authenticated jury endpoint response as a blob without reserializing it. The displayed analysis time stops at model completion; human waiting/review is separate. Review routes use the same operator API origin and `X-API-Key` as the other authenticated routes. Deployment of this workflow is coordinated by the release owners.

## PDF reports and exact event links

Completed checks have **Скачать PDF** and **Поделиться ссылкой** actions. PDF
export supports either the selected event or the entire check, using the local
UpMax typeface and site colours. Reports include original JPEG evidence with
preserved proportions, source times, observed facts, model descriptions,
available rule links, limitations and saved human decisions. Draft comments
are excluded. Text remains searchable; event and source links are clickable.

Each export fetches the authoritative result again and checks the selected
event for changes. Saved human decisions are included only when their result
SHA-256 matches the exact downloaded result bytes. Optional metadata/review
failures are stated in the document; failure to fetch the authoritative result
blocks export with a retryable error. Export runs locally in the browser with
lazy-loaded jsPDF and embedded local fonts, without an external PDF service.

Links use `?job=<id>&event=<canonical-index>&t=<source-seconds>`. The event index
is the original API index, independent of chronological presentation. Opening
a link overrides an unrelated saved task and restores the selected event and
video position after media loads. Recipients connect their own authorised key
using the existing access dialog. URLs contain no key, footage or configurable
API destination. Clipboard failures leave a selected, read-only link for manual
copying. Links require the original task to remain available on the server and
the website to be reachable by the recipient.

Synthetic coverage lives in `qa/report-sharing.spec.ts`; generated samples stay
under `qa/reports/` and are never bundled in the production UI.

## Verification

`npm run build` performs strict TypeScript checks and the production build. Dependencies and transitive hashes are pinned in `package-lock.json`; current stable npm tags were checked on 17 October 2026.

```powershell
npm run test:ui
```

Playwright uses installed Chrome by default; set `PLAYWRIGHT_CHANNEL` if another supported test channel is needed. The harness starts or reuses Vite. Its mocked routes and clearly synthetic video/JPEG live exclusively under `qa/`; they are not imported by the application and are not shipped in `dist`. Tests exercise upload and authentication, key-field focus, multi-event navigation and time seeking, evidence enlargement, honest final/failure/processing states, network loss, format errors, restored jobs, and a 360px layout. Screenshots contain only synthetic material. Never point this screenshot workflow at supplied private footage or actual jury evidence.

The local fixture is an FFmpeg-generated test pattern, not a recorded bank operation. Recreate it if needed:

```powershell
ffmpeg -f lavfi -i 'testsrc2=size=960x540:rate=10:duration=10' -an -c:v libx264 -preset veryfast -crf 25 -pix_fmt yuv420p -movflags +faststart qa/fixtures/review-synthetic.mov
```

Brand provenance and permitted assets are maintained by A19 in `brand/` and `docs/forte-brand-research.md`. Semantic status colors and the reviewer layout are team design choices. The visible product is explicitly identified as a datathon prototype.
