# Forte: source ledger

Research date: **17 October 2026, Asia/Qyzylorda**. Retrieval metadata, response hashes and seven verified CSS observations are in `reference-captures/public-source-observations.json`. Only public brand materials were accessed; no supplied video, label, frame or description was read.

| ID | Primary source | Evidence and scope | Status / limitation |
|---|---|---|---|
| B01 | [Forte Brand book](https://forte.kz/en/logo) | Official `forte.kz` hosting; numbered sections 01–05 identify master brand, color values, typography and downloads. `/ru/logo` resolves to this English page. | **Official rule** for explicitly published identity guidance. No published revision number/date was found; do not claim compliance with an unpublished full manual. |
| B02 | [Forte rebrand announcement, 5 February 2025](https://forte.kz/ru/news/press-reliz) | Official dated announcement corroborates current sunrise identity and the distinction between Forte and its sub-brands. | **Official statement**; context, not a source of numeric tokens. Historical 2015 design portfolios are not current authority. |
| B03 | [Forte font CSS](https://forte.kz/_next/static/chunks/1aluifwpkd4wx.css) | Linked by B01 at retrieval. `@font-face` uses Open Sans; linked CSS is 5,052 bytes. | **Observed official styling**; versioned path may change. HTTP Last-Modified is not a brandbook revision date. |
| B04 | [Official Open Sans download](https://forte.kz/docs/open-sans.zip) | Linked by B01 section 04. Contains unmodified variable font and `OFL.txt`. The packaged font and license have recorded SHA-256 hashes. | **Permitted local asset** under bundled OFL-1.1. Preserve license/copyright notice. No runtime request to Forte or a font CDN is needed. |
| B05 | [Official logo download](https://forte.kz/docs/forte-logos.zip) | Linked by B01 section 05. Outer archive inventories sub-brand `.7z` packages; no top-level license was present. | **Unresolved** redistribution terms; nested packages were not extracted. No official logo is packaged, and no substitute logo was drawn. Plain project text is the fallback. Public availability is not a blanket trademark license. |
| B06 | [Official Russian homepage](https://forte.kz/ru) | Current master site and navigation distinguish retail, business and Premier. | **Observed official styling/context**; the reviewer UI uses the master Forte identity, not a Business/Premier product identity. |
| A01 | [W3C text contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html) | Team target: 4.5:1 for ordinary text, 3:1 for large text. | **Team accessibility target**, not a Forte rule and not a claim of full WCAG conformance. |
| A02 | [W3C non-text contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html) | Team target: visible control/focus information at least 3:1 against adjacent colors. | **Team accessibility target**; requires validation in the rendered interface. |

## Evidence boundaries

An authentic **official hosted web brandbook was found**. It supplies current master-brand values and type guidance. A separately versioned downloadable complete brand manual was not authenticated. Numeric logo clear-space and minimum-size dimensions remain unresolved. No bank approval, affiliation or endorsement follows from this research.

`tokens.json` distinguishes `official_rule`, `observed_official_styling`, `team_design_choice` and `unresolved`. A semantic status color, panel size or interface composition is never elevated to an official rule simply because it resembles the website. CSS measurements are source observations, not computed browser measurements.

## Reproduction

From the repository root, run `python brand/refresh_public_sources.py` to refresh public source hashes and the allowlisted font/license. The script reads no project media. It does not save the full HTML, CSS, logo archive, analytics scripts or campaign imagery. Run `python brand/build_tokens.py` and `python brand/verify_brand.py --write-report` for a deterministic token/CSS build and offline static checks. A changed upstream source requires review before updating the frozen release tokens.
