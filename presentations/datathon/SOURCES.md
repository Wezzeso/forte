# FORTE Panopticon — Datathon 2026

15 slides: 13 for the five-minute pitch, then 2 appendix slides. Russian copy follows the product and case language. Sources and speaking notes are included in the PowerPoint notes for every slide.

## Evidence

- Task: `Кейс_ForteBank.docx.pdf` (Datathon 26, ForteBank case and FAQ).
- Development evaluation: `benchmarks/quality/accuracy-upgrade-baseline-20261018.json` and `benchmarks/quality/accuracy-upgrade-candidate-comparison-20261018.json`. Same 40 development videos. Baseline 33/40; confirmation profile 34/40; TP15, TN19, FP0, FN6; processing failures0. Not a held-out evaluation. Timing is nearest-event matching on15 detected positive clips, median1.5s and11 within2s, versus1.0s and12 for baseline. Event-type accuracy is unmeasured.
- Public API check: `benchmarks/quality/final-public-six-load-20261018T005948Z.json`. Six first attempts, six successful, zero retries. Warm GPU-host client through public HTTPS. Median92.076s, maximum142.568s. Not desktop/cold latency or an accuracy test.
- Current implementation and limits: `README.md`, `docs/submission/pitch.md`, `docs/integration/accuracy-upgrade-implementation-20261018.md`, `src/forte/fast/catalog.py`, `src/forte/fast/policy.py`.
- Normative references are the team's catalog mappings, not an independent legal conclusion: [НБРК №231](https://adilet.zan.kz/rus/docs/V1900019680), [НБРК №14](https://adilet.zan.kz/rus/docs/V2000020075), [АРРФР №53](https://adilet.zan.kz/rus/docs/V2600038336). The first two links returned the official site's JavaScript shell during checking; the third could not be fetched. The deck does not assert fresh legal verification of the acts.

## Images and brand

- Original FORTE logo: `logo/Group 1.svg`, identical to `web/public/brand/forte-panopticon.svg`, proportions preserved.
- Original Zimran visual reference: [zimran.io](https://www.zimran.io/). Background vectors, curiosity artwork and UpMax web fonts are from the existing project's `web/public/zimran/`; provenance in that directory's `SOURCES.md`. Palette is the user's FORTE magenta #A31551, burgundy #861044, off-white #F2F2F2 and black.
- Original team cutouts: `assets/team/hero-hoodie.png`, `assets/team/hero-blazer.png`. These are previously edited user-provided portraits. Edit provenance in `assets/team/SOURCES.md`. No invented names or roles.
- Production UI: `dist/hero-release-20261018/production-hero.png`, [product](https://hackathon.wezeso.me/).
- Results UI: `web/qa/screenshots/positive-multiple-desktop.png`. This is an explicitly labeled synthetic fixture, not bank footage or accuracy evidence.
- WEZESO wordmark: `web/public/brand/wezeso-wordmark.png`. Deliberately oversized with only its upper portion visible on the closing slide.
- No private bank videos, evidence frames, credentials or API keys were included.

## New illustration

Saved asset: `C:/Users/Wezeso/Desktop/Forte/presentations/datathon/assets/cashier-illustration.png`.

Generated with the built-in `image_gen` tool. This is conceptual artwork, not an actual bank photograph or dataset frame.

Final prompt:

> Use case: productivity-visual. Asset: one editorial illustration for a Datathon presentation about visual review of bank cashier operations. Wide 3:2 composition, large clean white background #F2F2F2. On the right 70% draw an isometric bank cashier workstation: minimalist desktop, chair, small cash bundles, open laptop, compact safe, overhead security camera. Contemporary flat cut-paper / bold angular vector-like illustration with Zimran-style jagged lightning cuts, playful yet polished. Strict palette magenta #A31551, dark burgundy #861044, black and off-white. Strong simple black outlines, no gradients, no purple, no blue, no generic 3D glass. Left 30% empty white space. No people, no text, no logo, no watermarks. This is a conceptual illustration, not actual bank footage.

## Files

- PowerPoint contains native text, native charts with embedded data workbooks, source notes, links and UpMax font objects. The source font's preview/print embedding flags are retained.
- PDF is a faithful rendered presentation copy with clickable demo and GitHub links.
- Build and validation files stay in `build/`.
- Font transport format follows the [Embedded OpenType specification](https://www.w3.org/submissions/EOT/); glyphs and embedding rights were preserved.
