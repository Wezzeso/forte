"""Verify packaged font/license, provenance completeness, glyphs and chosen contrast.

This is static asset QA, not a claim that the running application is accessible.
No external packages or network are needed. No private media is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def glyph_available(font: bytes, codepoint: int) -> bool:
    """Read TrueType cmap formats 4 and 12 without loading executable font code."""
    u16 = lambda position: struct.unpack_from(">H", font, position)[0]
    u32 = lambda position: struct.unpack_from(">I", font, position)[0]
    cmap = None
    for table_index in range(u16(4)):
        record = 12 + table_index * 16
        if font[record:record + 4] == b"cmap":
            cmap = u32(record + 8)
            break
    if cmap is None:
        return False
    for index in range(u16(cmap + 2)):
        record = cmap + 4 + index * 8
        platform, encoding, offset = struct.unpack_from(">HHI", font, record)
        if platform not in (0, 3):
            continue
        subtable = cmap + offset
        fmt = u16(subtable)
        if fmt == 12:
            for group_index in range(u32(subtable + 12)):
                start, end, glyph = struct.unpack_from(">III", font, subtable + 16 + group_index * 12)
                if start <= codepoint <= end:
                    return glyph + codepoint - start != 0
        if fmt == 4 and codepoint <= 0xFFFF:
            count = u16(subtable + 6) // 2
            ends = subtable + 14
            starts = ends + count * 2 + 2
            deltas = starts + count * 2
            offsets = deltas + count * 2
            for segment in range(count):
                if u16(starts + segment * 2) <= codepoint <= u16(ends + segment * 2):
                    delta = u16(deltas + segment * 2)
                    range_offset = u16(offsets + segment * 2)
                    if range_offset == 0:
                        return (codepoint + delta) % 65536 != 0
                    address = offsets + segment * 2 + range_offset + (codepoint - u16(starts + segment * 2)) * 2
                    glyph = u16(address)
                    return glyph != 0 and (glyph + delta) % 65536 != 0
    return False


def luminance(color: str) -> float:
    values = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    values = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in values]
    return sum(coefficient * channel for coefficient, channel in zip((0.2126, 0.7152, 0.0722), values))


def contrast(a: str, b: str) -> float:
    low, high = sorted((luminance(a), luminance(b)))
    return (high + 0.05) / (low + 0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets/manifest.json").read_text(encoding="utf-8"))
    tokens = json.loads((ROOT / "tokens.json").read_text(encoding="utf-8"))["tokens"]
    errors = []
    checked_assets = []
    for asset in manifest["assets"]:
        path = (ROOT / asset["path"]).resolve()
        if not path.is_relative_to(ROOT):
            errors.append("Asset path leaves brand directory")
            continue
        ok = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        checked_assets.append({"path": asset["path"], "hash_matches": ok})
        if not ok:
            errors.append("Asset hash mismatch: " + asset["path"])
    required = {"role", "value", "status", "source_url", "source_local", "source_section", "observed_at", "restrictions"}
    allowed_statuses = {"official_rule", "observed_official_styling", "team_design_choice", "unresolved"}
    for name, token in tokens.items():
        if not required.issubset(token) or token["status"] not in allowed_statuses:
            errors.append("Incomplete provenance: " + name)
        if token["status"] == "unresolved" and token["value"] is not None:
            errors.append("Unresolved token must not silently receive a value: " + name)
        if token["status"] in {"official_rule", "observed_official_styling"} and not token["source_url"].startswith("https://forte.kz/"):
            errors.append("Unexpected official token source: " + name)
    font = (ROOT / "assets/fonts/OpenSans-Variable.ttf").read_bytes()
    required_text = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяӘҒҚҢӨҰҮҺІәғқңөұүһі0123456789№"
    missing = [character for character in required_text if not glyph_available(font, ord(character))]
    if missing:
        errors.append("Required Russian/Kazakh glyphs missing")
    pairs = [
        ("Primary button", "#FFFFFF", "#A31551", 4.5),
        ("Main text", "#1D1D1D", "#FFFFFF", 4.5),
        ("Muted text", "#5E5660", "#FFFFFF", 4.5),
        ("Link", "#185AC5", "#FFFFFF", 4.5),
        ("Normal status", "#146C43", "#EAF5EE", 4.5),
        ("Possible signs", "#7C4800", "#FFF3DA", 4.5),
        ("Failed status", "#A12B2B", "#FDECEC", 4.5),
        ("Processing status", "#185AC5", "#EEF4FF", 4.5),
        ("Visibility status", "#5E5660", "#F1EEF2", 4.5),
        ("Focus outline on white", "#A31551", "#FFFFFF", 3),
        ("Control outline", "#827883", "#FFFFFF", 3),
        ("Dark label on coral", "#4A0221", "#FC9385", 4.5),
    ]
    results = []
    for label, foreground, background, minimum in pairs:
        ratio = contrast(foreground, background)
        passed = ratio >= minimum
        results.append({"pair": label, "foreground": foreground, "background": background, "ratio": round(ratio, 3), "minimum": minimum, "passes": passed})
        if not passed:
            errors.append("Contrast below target: " + label)
    observations = json.loads((ROOT / "reference-captures/public-source-observations.json").read_text(encoding="utf-8"))
    source_css_matches = all(item["matched_in_response"] for item in observations["observed_styles"])
    if not source_css_matches:
        errors.append("Observed source CSS declaration mismatch")
    css = (ROOT / "theme.css").read_text(encoding="utf-8")
    if "https:" in css or "http:" in css:
        errors.append("Unexpected runtime network dependency in theme CSS")
    report = {"passed": not errors, "scope": "Static brand assets and proposed token pairs only; live UI QA is separate.",
              "assets": checked_assets, "tokens_checked": len(tokens), "glyphs_checked": len(required_text),
              "glyphs_missing": missing, "source_css_matches": source_css_matches, "contrast_pairs": results, "errors": errors}
    if args.write_report:
        (ROOT / "reference-captures/brand-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
