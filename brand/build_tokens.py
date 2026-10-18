"""Generate provenance-tagged design tokens and a dependency-free CSS handoff."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OBSERVED = "2026-10-17T14:55:34+05:00"
OFFICIAL = "https://forte.kz/en/logo"


def main() -> None:
    tokens = {}

    def add(name, value, role, status="team_design_choice", section="Analyst interface v1", source="docs/forte-ui-guidelines.md", restriction="Team choice for this prototype; not a Forte mandate.", css=None):
        tokens[name] = {"value": value, "role": role, "status": status,
                        "source_url": source if source.startswith("https:") else None,
                        "source_local": None if source.startswith("https:") else source,
                        "source_section": section, "observed_at": OBSERVED,
                        "restrictions": restriction, "css_variable": css}

    palette = [
        ("brand.primary", "#A31551", "Forte Magenta", "--forte-magenta"),
        ("brand.dark", "#4A0221", "Dark Magenta", "--forte-dark"),
        ("brand.coral", "#FC9385", "Coral Sunshine", "--forte-coral"),
        ("brand.sky", "#61CFFF", "Sky Blue", "--forte-sky"),
    ]
    for name, value, section, css in palette:
        add(name, value, "Master-brand palette color", "official_rule", "03 Sub-brands / Forte / " + section,
            OFFICIAL, "Published master-brand color. Choosing UI placements is a team decision; it does not imply endorsement.", css)
    add("font.family.brand", "Open Sans", "Brand typeface", "official_rule", "04 Typography", OFFICIAL,
        "Local unmodified font is OFL-1.1; retain brand/assets/fonts/OFL.txt.", "--forte-font-family")
    add("font.weights.brand", [300, 400, 600, 700, 800], "Published weights", "official_rule", "04 Typography", OFFICIAL,
        "Body text uses 400; 600/700 create hierarchy. Avoid thin weights for evidence captions.")
    add("logo.clear_space", None, "Numeric clear-space requirement", "unresolved", "01 Logotype color", OFFICIAL,
        "A clear-space instruction exists, but no numeric dimension was authenticated. No logo is packaged.")
    add("logo.minimum_size", None, "Numeric minimum logo size", "unresolved", "01 Logotype color", OFFICIAL,
        "No authenticated numeric requirement. Plain project text is used instead of an official logo.")

    observed = [
        ("color.text", "#1D1D1D", "Primary text", "body / color", "--forte-text"),
        ("color.surface", "#FFFFFF", "Panel background", "body / background-color:#FFF", "--forte-surface"),
        ("color.link", "#185AC5", "Link color", "a / color", "--forte-link"),
        ("color.border.observed", "#D5D5D5", "Decorative divider", ".css-72hos7 / border", "--forte-border"),
        ("radius.control", "12px", "Rounded UI controls", ".css-72hos7 / border-radius", "--forte-radius-control"),
        ("font.body", "1rem", "Body size", "body / font-size", "--forte-body-size"),
        ("font.body_line_height", 1.5, "Body line height", "body / line-height", "--forte-body-line-height"),
    ]
    for name, value, role, section, css in observed:
        add(name, value, role, "observed_official_styling", "Inline CSS: " + section, OFFICIAL,
            "Measured from public page source on observation date; generated selectors may change. Not a universal rule.", css)
    choices = [
        ("color.page", "#F6F5F7", "Workspace background", "--forte-page"),
        ("color.muted", "#5E5660", "Secondary readable text", "--forte-muted"),
        ("color.control_border", "#827883", "Visible control boundary", "--forte-control-border"),
        ("color.focus", "#A31551", "Keyboard focus on white", "--forte-focus"),
        ("color.normal.text", "#146C43", "No detected signs status", "--forte-normal"),
        ("color.normal.surface", "#EAF5EE", "No detected signs background", "--forte-normal-bg"),
        ("color.possible.text", "#7C4800", "Possible signs status", "--forte-possible"),
        ("color.possible.surface", "#FFF3DA", "Possible signs background", "--forte-possible-bg"),
        ("color.failed.text", "#A12B2B", "Processing failure", "--forte-failed"),
        ("color.failed.surface", "#FDECEC", "Processing failure background", "--forte-failed-bg"),
        ("color.processing.text", "#185AC5", "Active processing status", "--forte-processing"),
        ("color.processing.surface", "#EEF4FF", "Active processing background", "--forte-processing-bg"),
        ("color.visibility.text", "#5E5660", "Visibility limitation", "--forte-visibility"),
        ("color.visibility.surface", "#F1EEF2", "Visibility limitation background", "--forte-visibility-bg"),
        ("radius.panel", "20px", "Review panel radius", "--forte-radius-panel"),
        ("layout.maximum_width", "1440px", "Maximum workspace width", "--forte-max-width"),
        ("layout.mobile_breakpoint", "900px", "Stack review panels below this width", "--forte-stack-at"),
        ("layout.minimum_target", "44px", "Minimum primary control target", "--forte-target-size"),
        ("font.h1", "clamp(1.75rem, 3vw, 2.5rem)", "Page heading", "--forte-h1-size"),
        ("font.h2", "1.25rem", "Section heading", "--forte-h2-size"),
        ("font.caption", "0.875rem", "Evidence metadata", "--forte-caption-size"),
        ("motion.duration", "160ms", "Short control transition; respect reduced motion", "--forte-motion-duration"),
    ]
    for name, value, role, css in choices:
        add(name, value, role, css=css)
    add("spacing.scale", [4, 8, 12, 16, 24, 32, 48], "Spacing units in CSS px")
    add("evidence.object_fit", "contain", "Preserve full source frame", css="--forte-evidence-fit")
    add("evidence.aspect_ratio", "source", "Use intrinsic dimensions; no forced square crop")
    payload = {"schema_version": "1.0", "identity": "Forte master brand, 2025 rebrand", "observed_at": OBSERVED,
               "version_note": "Current official hosted web brandbook; no revision/date identifier was published on the inspected page.",
               "status_vocabulary": ["official_rule", "observed_official_styling", "team_design_choice", "unresolved"], "tokens": tokens}
    (ROOT / "tokens.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    css = ["/* Generated from brand/tokens.json by build_tokens.py. Provenance lives in JSON. */",
           "@font-face {", '  font-family: "Open Sans";',
           '  src: url("./assets/fonts/OpenSans-Variable.ttf") format("truetype");',
           "  font-style: normal;", "  font-weight: 300 800;", "  font-display: swap;", "}", ":root {"]
    for token in tokens.values():
        if token["css_variable"]:
            value = json.dumps(token["value"]) if token["role"] == "Brand typeface" else str(token["value"])
            css.append(f"  {token['css_variable']}: {value};")
    css.extend(["}", "/* Consumer: font-family: var(--forte-font-family), system-ui, sans-serif; */", ""])
    (ROOT / "theme.css").write_text("\n".join(css), encoding="utf-8")
    print(json.dumps({"tokens": len(tokens), "css_variables": sum(bool(t["css_variable"]) for t in tokens.values())}))


if __name__ == "__main__":
    main()
