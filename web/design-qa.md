# FORTE Panopticon design QA

final result: passed

## Animated magnifying glass — 2026-10-18

- Added an analyzing illustration with a Phosphor magnifying glass sweeping over a portrait film strip, plus softly animated geometric sparkles. It uses the existing magenta, paper and ink palette without a boxed icon treatment. Reduced-motion preferences show a static illustration. Animation is decorative and independent of processing progress.
- Build and existing processing-state regression passed. Browser checks confirmed different lens positions, reduced-motion behavior, mobile containment and no browser errors. Repeated those checks against the deployed frontend using synthetic, intercepted API responses; no upload or model request was made.
- Published and verified: all 26 public files match the release, API bootstrap and current backend settings are preserved, and readiness passes. Receipt: `../infra/frontend-analysis-animation-deployment-20261018.json`. Start, sweep and mobile screenshots are in `../dist/analysis-animation-release-20261018/`.

## Portrait video layout, Phosphor icons and illustrations — 2026-10-18

- The review workspace defaults to 9:16 and reads decoded source dimensions to select portrait or landscape proportions. Desktop portrait playback occupies a narrower, sticky left column; events, evidence and decisions share the right column. Below 900px the content stacks. Player height adapts to the viewport, including 1280 × 720 desktops. Videos and evidence retain the complete image with `object-fit: contain`.
- Selected uploads show a media preview. Evidence enlargement uses a narrower dialog for portrait sources, and frame buttons retain explicit accessible names when compact captions are hidden.
- Replaced Lucide with pinned `@phosphor-icons/react` 2.1.10 throughout the app and human review panel. Close and back-to-top controls have transparent backgrounds. Original magenta, paper and ink SVG compositions replace isolated icon tiles in upload, instructions, processing, results, empty state and access dialog.
- Build passed and the full UI suite passed, 25/25. Added synthetic portrait MOV/JPEG fixtures and a regression covering 1440 × 1000, 1280 × 720, 900 × 900 and 360 × 800, native dimensions, containment, overflow, source seeking and evidence enlargement. Existing landscape, authentication, error-state and human-review checks also passed.
- Inspected `qa/screenshots/upload-desktop.png`, `portrait-1440.png` and `portrait-360.png`. These captures contain only synthetic footage. Published and verified on `https://hackathon.wezeso.me/`: 26/26 public files and unchanged API bootstrap match the frozen release. Desktop/mobile browser checks report ready service, matching illustrations, no overflow and no browser errors. The portrait regression also passes against the deployed frontend using browser-only synthetic fixtures. Live worker, database, ingress, model and current backend configuration remain unchanged. Receipt: `../infra/frontend-portrait-deployment-20261018.json`.

## Restore oversized, half-visible footer — 2026-10-18

- User reported the fully visible, reduced WEZESO signature in results. Cause: `.workspace-state` overrides limited it to 760px and made its image static/unclipped. Removed those desktop/mobile exceptions. Landing, processing and results now share the same full-width signature and hidden overflow.
- Adjusted the shared signature to display approximately the upper half of the actual letterforms while retaining the compact 16–32px gap. Frame is viewport-width, image 108% of it with deliberate side clipping; aspect 13 and vertical positioning account for the source image's black padding. The logo asset itself is unchanged.
- Actual rendered results evidence from the existing synthetic upload/result test: `qa/design/footer-crop-results.png`; landing 1440 × 900: `qa/design/footer-crop-landing.png`; mobile 390 × 844: `qa/design/footer-crop-mobile.png`. All were opened and inspected. Desktop frame/image widths are 1425/1539px; mobile 375/405px. The bottom half is cropped and no horizontal document overflow is introduced.
- Build passed; existing upload/result regression passed (1/1, 5.4s), including the full-width results-footer assertion and actual result/evidence interactions. No new test was added for this presentation-only fix. No P0/P1/P2 findings remain.
- Published and verified: 26/26 public files and application HTML match the frozen release, unchanged API bootstrap, service ready, no browser errors/warnings. Production frame/image widths are 1425/1539px with absolute positioning and hidden overflow. Screenshot: `../dist/footer-crop-release-20261018/production-footer.png`; receipt: `../infra/frontend-footer-crop-deployment-20261018.json`.

## Hero typography and team portraits — 2026-10-18

- Rechecked the live Zimran hero and measured its actual H1: UpMax 700, 110px/132px, normal letter spacing; inline person slot 140px desktop/66px mobile. Source captures: `qa/design/hero-source-desktop.png` and `qa/design/hero-source-mobile.png`. User's more specific white-stencil reference is preserved at `qa/design/portrait-shape-reference.png`.
- Replaced the oversized product-name/mascot heading with `Видим суть. Решаете вы.` and the concise caption `Видеоаналитика FORTE Panopticon.` Brand title, real logo, magenta palette, source zigzag, CTA and no-eyebrow constraint remain.
- Used built-in Imagegen to improve both supplied portraits on white with soft neutral studio lighting, then create identity-preserving transparent cutouts over the reference's white angular stencil. The initial square-card treatment was superseded by the user's clarification. Final 1254px artwork is preserved in `../assets/team/`; website `public/team/hero-hoodie.webp` and `hero-blazer.webp` are lossless encodings with exact decoded RGBA pixels verified. Prompts and provenance: `../assets/team/SOURCES.md`. Both people remain visible inline rather than rotating, so the supplied faces can be seen together.
- Final desktop rendering uses the source's 110px/132px type and normal tracking with 140px cutouts. Mobile font and cutout dimensions adapt to the Russian wording: 42.9px text and ~60px cutouts at 390px. Initial tight word bounds were fixed with responsive sizing; final text stays inside practical margins at 360px and 390px, with no document overflow. Main CTA bottom is 621.8px in a 1280 × 720 viewport and 590.8px in 390 × 844.
- Opened the source and final implementation together in `qa/design/comparison-portrait-hero-desktop.png` and `comparison-portrait-hero-mobile.png`. Desktop source/render captures are 1425 × 990 from a 1440 × 1000 CSS viewport; normalized to matching 900 × 625 panels. Mobile panels are normalized to 390 × 844. Focused rendered portrait/stencil comparison: `qa/design/comparison-portrait-shape.png` against the user's exact crop. Source violet/copy/persons intentionally become FORTE magenta, short Russian product copy and the supplied people. Geometry, typography, spacing, white motif, photo lighting, alpha integration and CTA treatment pass the requested adaptation. No P0/P1/P2 findings remain.
- Final evidence: `qa/design/portrait-hero-desktop.png`, `portrait-hero-mobile.png`, and `portrait-hero-short-desktop.png`. Narrow layout was also inspected at 360 × 780. The full-resolution generated stencils retain slight perimeter texture; it does not present an actionable mismatch at their 140px/mobile display sizes (P3 only).
- Final build passed. Full 24-test UI suite passed in 36.1s after updating existing landing-heading assertions. After the final stencil and responsive changes, the existing upload/result and 360px invalid-file cases passed again (2/2, 8.0s). No test-only app behavior was introduced.
- Production verification passed: 26/26 public frontend files match the frozen release, exact application HTML and unchanged API bootstrap verified. Both new portrait WebPs load on desktop and mobile; CTA fits, no overflow, service ready, no console errors/warnings. Evidence: `../dist/hero-release-20261018/production-hero.png` and `production-hero-mobile.png`. Receipt: `../infra/frontend-hero-deployment-20261018.json`.

## Footer gap correction — 2026-10-18

- User identified excessive empty space between copyright and the WEZESO signature in a 2476 × 804 crop. Reference preserved at `qa/design/footer-gap-reference.png`.
- Removed the image's black top padding from the visible composition through proportional CSS positioning; reduced signature aspect from 4.58 to 8.5 and its top margin to 16–32px. Image, letter scale and intentional bottom crop remain; no raster editing or regenerated asset was needed.
- Verified at 2476 × 1000 and 390 × 844 CSS viewports: `qa/design/footer-compact-wide.png` and `qa/design/footer-compact-mobile.png`. Focused before/after region comparison opened together at equal width in `qa/design/comparison-footer-gap.png`; final composition is intentionally shorter. The effective gap from copyright box to first bright letter is approximately 41px wide and 17px mobile, reduced from hundreds of pixels at wide widths. No overlap or horizontal overflow. All three social link destinations remain correct.
- Final build passed, and the focused desktop/mobile comparison has no P0/P1/P2 findings. Only the two signature layout rules changed after the social release.
- Published and verified on `https://hackathon.wezeso.me/`: all 24 frontend asset hashes and application HTML match the release, API bootstrap is unchanged, exact three social destinations are present, no horizontal overflow, service ready, and no console errors/warnings. Production screenshot: `../dist/compact-release-20261018/production-footer.png`; receipt: `../infra/frontend-social-deployment-20261018.json`.

## Footer social links — 2026-10-18

- Matched the supplied `Follow us for updates` reference with square light buttons, small northeast arrows, and muted copyright text. Added GitHub, LinkedIn and Instagram. Kept existing FORTE magenta, page gutters, real logo, back-to-top control and large WEZESO signature. Copyright uses WEZESO; the unrelated source address is omitted.
- Destinations verified from rendered DOM: `https://github.com/Wezzeso/forte`, `https://www.linkedin.com/in/timur-ryabov-9886763a4/`, and `https://www.instagram.com/wezeso/` (the user's `wezeso inst` handle). All open a new tab with `noopener noreferrer`. Visible text supplies accessible link names; arrow icons are hidden from assistive technology.
- Reference preserved at `qa/design/social-reference.png`. Final focused comparison: `qa/design/comparison-social.png`, source and rendered region normalized to 564 × 245. Whole desktop/mobile evidence: `qa/design/social-desktop.png`, `qa/design/social-mobile.png`, and `qa/design/social-reference-size.png`. Final combined reference comparison opened and inspected: typography, heading/copyright spacing, white rectangular controls and arrow alignment match the requested treatment. Buttons use 44px touch targets, compared with the source's 36px height; established footer gutters remain.
- Initial mobile layout wrapped Instagram by less than one pixel; reduced horizontal button padding and confirmed all three fit one row at 390px. Narrower widths can wrap naturally. No horizontal document overflow at 390px or 1280px. Keyboard focus on Instagram remains clearly visible. Final `npm run build` passed; no new interaction logic was introduced and no additional regression tests were needed. No P0/P1/P2 findings remain.

## WEZESO footer — 2026-10-18

Requested outcome: adapt the supplied Zimran footer image to `WEZESO`, retaining FORTE's #A31551 background. The existing logo, footer information and back-to-top control remain above the oversized signature.

- Reference: user attachment preserved at `qa/design/footer-reference.png` (2536 × 553). Rendered evidence: `qa/design/footer-desktop.png` (1280 × 720 CSS; 1265 × 712 saved pixels) and `qa/design/footer-mobile.png` (390 × 844 CSS). Focused, equal-width source/implementation comparison: `qa/design/comparison-footer.png`.
- Initial generated transparency assets had visible distressed edges and splatter, a P2 image-fidelity mismatch. Replaced with a clean imagegen asset: six readable, white, sharply angular forward-slanted letters on black matte, 2171 × 724, blended over the brand surface using screen. The generated image is used unaltered; the browser controls scaling and the intentional bottom crop. No placeholder lettering or synthetic vector drawing is used.
- Final comparison opened and inspected alongside the supplied reference. White letter tops begin at a similar depth within the full-width signature. The lower letters and outer edges are intentionally cropped, with the same oversized composition. The requested text and established magenta replace source Zimran/violet. Different letterforms follow the requested uppercase name; no extra copy or decorative elements were introduced.
- Desktop footer is 422.08px high with a 276.19px signature; mobile footer is 244.38px with an 81.88px signature. Logo, footer information and back-to-top control remain legible and unobstructed. No horizontal document overflow at either viewport; image loaded at its native 2171 × 724 size. Back-to-top retains its accessible name; wordmark has `WEZESO` alt text.
- Required fidelity surfaces checked: source-inspired typography and angular slant, spacing/crop, #A31551 and white contrast, clean image edges and matte blending, responsive controls, existing UpMax text and real FORTE logo. No actionable P0/P1/P2 findings remain.
- Final `npm run build`: passed. Existing 24/24 interaction tests passed for the preceding favicon/title/menu polish; this final change is isolated to footer markup, layout and an image and received focused browser verification.
- Production deployment passed: all 24 frontend files match the frozen release over public HTTPS; exact application HTML/title, favicon MIME types, manifest and runtime API origin verified. Browser shows the finished footer and `Сервис готов`; no console errors/warnings. Screenshot: `../dist/footer-release-20261018/production-footer.png`. Receipt: `../infra/frontend-polish-deployment-20261018.json`. Worker, model, database, ingress and API runtime configuration retain their verified pre-release state.

## Favicon, titles and usability polish — 2026-10-18

Requested outcome: add favicon/title and improve the existing experience while retaining the established magenta palette, real logo, UpMax, source zigzag, mascot and no-eyebrow constraint.

- Brand favicon is the original logo's exact two `f` subpaths, uniformly transformed without redrawing. SVG, ICO 16/32/48, Apple 180px and PNG 192/512 are available. Native-size previews: `../tmp/favicon-review/small-sizes.png`; large export also visually checked. Manifest icons use purpose `any`; no unsupported maskability assumption.
- Idle title: `FORTE Panopticon — Видеоаналитика`; processing, result and failure titles reflect actual job state. Metadata includes canonical URL, description, Open Graph title/description, theme and home-screen branding. The real font is preloaded.
- Hero fits desktop viewport: in 1280 × 720 CSS viewport, main CTA bottom is 607.8px and remains fully visible. Source composition, two-line title and source stripe animation remain intact.
- Tablet header switches to the mobile menu through 1023px. Before/after focused comparison at 900 × 720 CSS: `qa/design/comparison-polish-header.png`. Both underlying screenshots are 885 × 708; no density correction required. The before capture is mid-scroll, so only its complete fixed header is used as comparison evidence.
- Upload now uses a recognizable upload icon and a single-line mobile heading; removed redundant section number. At 390px, heading height 28.797px matches one 28.8px line. No horizontal overflow.
- Menu locks background focus, supports Tab/Shift+Tab and Escape, focuses the chosen section, closes on desktop resize, and scrolls on short screens. Access dialog returns focus to the visible menu toggle after its original menu trigger unmounts.
- Latest implementation screenshots: `qa/design/polish-desktop.png` (1280 × 720 CSS), `qa/design/polish-tablet.png` (900 × 720 CSS), `qa/design/polish-upload-mobile.png` (390 × 844 CSS), and `qa/design/polish-menu-short.png` (390 × 390 CSS). Saved images opened and inspected. Captures exclude browser scrollbar/chrome as in prior comparisons.
- Required fidelity surfaces: UpMax fonts/weights preserved; hero and navigation spacing improved at the measured breakpoints; #A31551 and semantic colors preserved; original vector logo geometry remains exact; no eyebrows or extra visible marketing copy added. Upload icon is from the existing Lucide library. No actionable P0/P1/P2 findings remain.
- Final `npm run build`: passed. Full regression suite: 24/24 passed. New cases cover tablet focus wrapping/Escape, resize recovery and short-screen scrolling/access. The final modal-focus fallback additionally passed its targeted test. Idle/processing/failure/result document-title assertions passed.
- One in-app browser mouse dispatch timed out; subsequent keyboard activation/Escape worked and was captured. This was a browser-control failure; independent full UI regression also passed the interaction.

## Refinement — 2026-10-18

User requested removal of eyebrow labels, source-matched lines, and the uploaded logo.

- Removed every `.eyebrow` element from the hero, how-it-works section, and results. Browser DOM count: zero.
- Copied `../logo/Group 1.svg` byte-for-byte to `public/brand/forte-panopticon.svg`; matching SHA256 verified. Native white logo is used in header and footer with its original aspect ratio. Mobile menu uses a black CSS filter for contrast on the light surface; the file itself is unchanged.
- Replaced the custom rotated stripe animation with the source's 15-second ease-in-out cycle, translating from -35% to -25% and back. Desktop: 200vw, scale 1.05. Mobile: 400vw, scale 1.5. Centered at 50%/50%, exactly as the source. Reduced motion remains supported.
- Verified settled source mobile stripe dimensions against implementation: width 1560px, height 257.672px, scale 1.5, top 422px in the 390 × 844 viewport. Captures now wait for the source's resize transition to settle.
- Latest evidence: `qa/design/forte-refined-desktop.png` (1425 × 990), `qa/design/forte-refined-mobile.png` (375 × 812), and `qa/design/forte-refined-menu.png` (390 × 844). Combined visual comparisons: `qa/design/comparison-refined-desktop.png` and `qa/design/comparison-refined-mobile.png`; latest settled mobile source is `qa/design/zimran-lines-mobile.png`.
- Required surfaces checked: original UpMax typography unchanged; source stripe geometry/spacing/motion matched; requested magenta palette retained; uploaded vector logo is sharp and proportional; eyebrow copy removed. No P0/P1/P2 issues remain. Logo contrast and mobile open/close behavior verified; no horizontal overflow at 390px.
- `npm run build`: passed after the refinements. The previous full 21-test regression pass remains documented below; it was not rerun for this presentation-only update.

## Source and scope

Source: https://www.zimran.io/, captured 2026-10-17 in the in-app browser.
Visual truth: `qa/design/zimran-desktop.png`, `qa/design/zimran-mobile.png`, `qa/design/zimran-mobile-menu.png`.
Implementation: http://127.0.0.1:5173/.
Rendered evidence: `qa/design/forte-desktop.png`, `qa/design/forte-mobile.png`, `qa/design/forte-mobile-menu.png`; upload/review states also in `qa/screenshots/`.

Intentional content changes: FORTE Panopticon branding, #A31551 instead of violet, short Russian copy, and working video analysis controls. Zimran's company/team/careers marketing content is replaced by the existing FORTE upload and review experience. The source's own mascot replaces staff portraits. This is a faithful visual adaptation of the reference, not an identical reproduction of its content or every marketing section.

## Comparison evidence

- Desktop: 1440 × 1000 CSS viewport, device scale factor 1. Both source and implementation captures are 1425 × 990 pixels after the browser capture excludes scrollbar/chrome. Identical capture dimensions; no density correction needed. Combined comparison: `qa/design/comparison-desktop.png`.
- Mobile: 390 × 844 CSS viewport, device scale factor 1. Both closed-menu captures are 375 × 812 pixels. Combined comparison normalized both to 390 × 844: `qa/design/comparison-mobile.png`.
- Mobile menu: source 375 × 812, implementation 390 × 844; normalized to 390 × 844 for `qa/design/comparison-menu.png`. No finding is based on capture scaling.
- Focused header/controls comparison: `qa/design/comparison-navigation.png`.
- States: desktop and mobile landing, mobile menu, upload, access dialog, invalid file, processing, completed positive/negative, failure, evidence modal, human review and revision conflict.

The first source tab had a stale 130% emulation scale; discarded those desktop/mobile captures and replaced them using a clean tab with confirmed device scale factor 1 before comparison.

## Required fidelity surfaces

- Typography: actual locally bundled UpMax regular/medium/bold source fonts. Desktop hero 110px/132px and mobile hero 48px/57.6px match the reference. Navigation 20px/32px; mobile menu 32px/48px. Shorter title and copy intentionally use fewer lines.
- Layout: matching full-color hero, fixed header, centered two-line desktop display title, patterned CTA, full-width black ticker, and light mobile menu. Upload is placed below the hero; the existing results workspace remains usable. No horizontal overflow at 390px or the 360px regression viewport.
- Colors: #A31551 primary and #861044 pattern, with source #F2F2F2 surfaces. No remaining violet assets in the rendered FORTE UI. Service/error colors remain semantic.
- Assets: original source SVG paths and fonts copied locally. Only source purple fill/stroke colors were changed. No remote asset dependencies or fabricated decorative artwork.
- Copy: FORTE Panopticon page title and h1; one short subtitle; compact upload instructions. Codec/upload limits remain available under the requirements disclosure. Operational errors, evidence, and review information remain intact.

## Findings and fixes

No remaining actionable P0/P1/P2 findings within the requested adaptation.

Before final captures, removed inherited mobile three-column steps, restored intended heading line breaks, and centered mobile header items. Final mobile and menu comparisons show the corrected layout. Mobile navigation closes on selection/Escape, focuses its first link, locks background scrolling, and makes the main content inert while open.

Expected differences: brand/title, language, fewer navigation items, shorter text, mascot substitution, and the FORTE upload/results sections. Exact source marketing-page content is deliberately not carried over.

## Verification

- `npm run build`: passed.
- `npm run test:ui`: 21/21 passed; covers upload/auth request shape, invalid files, mobile behavior, source seeking, evidence focus trap, review accessibility and conflicts, runtime origin handling, immutable event bindings, and exact JSON downloads.
- In-app browser: tested hero anchor, desktop access dialog, dialog close/focus restoration, mobile menu, mobile upload anchor, and requirements disclosure. Console warning/error list empty on preview inspection.
- Live analysis backend was unavailable in this local design preview; readiness remains honestly unconfirmed. Existing regression flows use isolated synthetic fixtures and do not claim a live model run.

## Implementation checklist

- [x] Real source capture and locally copied assets
- [x] FORTE title and #A31551
- [x] Minimal visible copy
- [x] Desktop/mobile combined comparisons and focused navigation comparison
- [x] Existing upload/evidence/review flow preserved
- [x] Build and 21 regression checks

## Follow-up polish

None required for this design pass.

## Video review layout correction — 18 October 2026

The review page now opens at the top of the result with the video and event list directly below a compact verdict. Removed the repeated event-description previews and result slogan. Job identifiers and the detailed timing/session panel remain available in the closed-by-default **Технические сведения** disclosure below the workspace.

The review cards use the existing UpMax font, FORTE colors, 16px panel corners and 5px controls. Desktop video height adapts to short laptop viewports without cropping the source. Mobile video retains its 16:9 viewing area and evidence thumbnails use two columns. Evidence pixels, model findings, visibility limits and human-decision behavior are unchanged.

The footer is a sibling of the width-constrained main content and spans the viewport. In the review/processing workspace its WEZESO image uses its full intrinsic aspect ratio instead of the landing page's decorative crop. Footer links are inert while the mobile menu is open.

Visual evidence from the current in-app-browser run is in `qa/design/video-check/`: desktop and mobile results, laptop result, and desktop/mobile footers. The previews use only the isolated synthetic fixture server in `qa/preview-server.mjs`; no preview mocks or session seeding enter `src/` or `dist/`.

Verification: production build passed; all 24 UI regression checks passed. Layout assertions cover the closed diagnostic disclosure, returning to the result top after upload, the full video above the fold, a 1280×720 laptop viewport, a full-width 1920px footer, and 360px overflow with diagnostics open and closed. The in-app browser also verified 390px rendering, event selection seeking to 7.2 seconds, and a complete uncropped footer wordmark. Live model/backend accuracy was outside this presentation change and was not evaluated.
