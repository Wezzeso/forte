"""Inspect official wheel lists without downloading wheels or touching a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.values.extend(value for key, value in attrs if key == "href")


def inspect(package: str, requested: str) -> dict:
    source = f"https://download.pytorch.org/whl/cu129/{package}/"
    raw = urllib.request.urlopen(source, timeout=30).read()
    parser = Links()
    parser.feed(raw.decode("utf-8"))
    matches = []
    for href in parser.values:
        name = urllib.parse.unquote(href.rsplit("/", 1)[-1]).split("#", 1)[0]
        pattern = rf"{package}-(\d+\.\d+\.\d+\+cu129)-cp312-cp312-manylinux_2_28_x86_64\.whl"
        match = re.fullmatch(pattern, name)
        if match:
            matches.append((tuple(int(part) for part in match[1].split("+", 1)[0].split(".")),
                            match[1], urllib.parse.urljoin(source, href)))
    matches.sort()
    if not matches:
        raise ValueError("official_cp312_cu129_wheels_missing")
    return {"package": package, "source": source, "source_html_sha256": hashlib.sha256(raw).hexdigest(),
            "target": "cp312-manylinux_2_28_x86_64", "requested": requested,
            "requested_available": any(version == requested for _, version, _ in matches),
            "newest_available": matches[-1][1], "selected_wheel_url": matches[-1][2],
            "available_versions": [version for _, version, _ in matches]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("training/cu129-index-verification.json"))
    args = parser.parse_args()
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "wheels_downloaded": False,
              "gpu_used": False, "indexes": [inspect("torch", "2.14.1+cu129"), inspect("torchvision", "0.29.1+cu129")]}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"official_indexes_checked": 2, "requested_latest_cu129_available":
                      all(row["requested_available"] for row in result["indexes"]),
                      "compatible_versions": {row["package"]: row["newest_available"] for row in result["indexes"]},
                      "wheels_downloaded": False, "gpu_used": False}))


if __name__ == "__main__":
    main()
