"""Refresh public brand evidence and the licensed, unmodified local font.

Uses only the standard library. It never reads project media or labels.
Only explicitly allowlisted font/license entries are extracted from the ZIP.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
BASE = "https://forte.kz"


def fetch(url: str) -> tuple[bytes, dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "Forte-Datathon-Brand-Research/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
        return data, {
            "requested_url": url,
            "final_url": response.url,
            "status": response.status,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "content_type": response.headers.get("Content-Type"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def main() -> None:
    records = []
    html_bytes, record = fetch(BASE + "/en/logo")
    records.append(record)
    html = html_bytes.decode("utf-8")
    css_path = re.search(r'href="([^"]+\.css)"', html).group(1)
    css_bytes, record = fetch(BASE + css_path)
    records.append(record)
    styles = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", html, re.S))
    observed_styles = []
    for selector, prop, value in (
        ("body", "color", "#1D1D1D"),
        ("body", "background-color", "#FFF"),
        ("body", "font-size", "1rem"),
        ("body", "line-height", "1.5"),
        ("a", "color", "#185AC5"),
        (".css-72hos7", "border-radius", "12px"),
        (".css-72hos7", "border", "1px solid #D5D5D5"),
    ):
        matches = re.findall(re.escape(selector) + r"\{([^}]+)\}", styles)
        declaration = f"{prop}:{value};"
        observed_styles.append({
            "selector": selector,
            "property": prop,
            "value": value,
            "matched_in_response": any(declaration in match for match in matches),
            "source_url": BASE + "/en/logo",
            "status": "observed_official_styling",
            "restriction": "An observed declaration, not a universal brand requirement; generated selectors may change.",
        })
    font_bytes, record = fetch(BASE + "/docs/open-sans.zip")
    records.append(record)
    allowed = {
        "OFL.txt": ROOT / "assets/fonts/OFL.txt",
        "OpenSans-VariableFont_wdth,wght.ttf": ROOT / "assets/fonts/OpenSans-Variable.ttf",
    }
    assets = []
    with zipfile.ZipFile(io.BytesIO(font_bytes)) as archive:
        license_text = archive.read("OFL.txt").decode("utf-8")
        if "SIL OPEN FONT LICENSE Version 1.1" not in license_text:
            raise RuntimeError("Expected OFL license missing; no font was written.")
        for entry_name, destination in allowed.items():
            payload = archive.read(entry_name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            assets.append({
                "path": destination.relative_to(ROOT).as_posix(),
                "archive_entry": entry_name,
                "source_url": BASE + "/docs/open-sans.zip",
                "source_page": BASE + "/en/logo",
                "source_section": "04 Typography",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "license": "OFL-1.1",
                "status": "permitted_local_asset",
                "modified": False,
                "restrictions": ["Keep OFL.txt and its copyright notice with redistributed font copies.", "Do not sell the font on its own."],
            })
    logo_bytes, record = fetch(BASE + "/docs/forte-logos.zip")
    records.append(record)
    with zipfile.ZipFile(io.BytesIO(logo_bytes)) as archive:
        logo_inventory = [{"entry": entry.filename, "bytes": entry.file_size} for entry in archive.infolist()]
    observed_at = datetime.now(ZoneInfo("Asia/Qyzylorda")).isoformat(timespec="seconds")
    capture = {
        "observed_at": observed_at,
        "timezone": "Asia/Qyzylorda",
        "method": "HTTPS source fetch; no browser screenshot, private media, or labels were read.",
        "sources": records,
        "observed_styles": observed_styles,
        "font_css_source_url": BASE + css_path,
        "font_css_uses_open_sans": "font-family:Open Sans" in css_bytes.decode("utf-8"),
        "logo_archive_inventory_only": logo_inventory,
        "logo_decision": "No logo extracted: redistribution terms not established by the public outer archive; use a plain text project label.",
    }
    target = ROOT / "reference-captures/public-source-observations.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(capture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "assets/manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "observed_at": observed_at, "assets": assets,
        "not_packaged": [{"type": "Forte logos", "source_url": BASE + "/docs/forte-logos.zip", "status": "unresolved", "reason": "No redistribution license established; no logo or lookalike created."}],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(records), "assets": len(assets), "observed_styles_verified": all(item["matched_in_response"] for item in observed_styles)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
