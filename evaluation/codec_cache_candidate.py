"""Isolated, opt-in JPEG memoization experiment; never wired into the worker.

Run on reviewed public images, never organizer/test footage:
  python -m evaluation.codec_cache_candidate --image public/example.jpg \
    --source-url https://publisher.example/dataset --license CC-BY-4.0 \
    --repeats 8 --rounds 5 --report codec-report.json

This measures CPU encoding only. It establishes neither model decision
equivalence nor end-to-end video latency. Every logical image part is retained.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402
from PIL import __version__ as pillow_version  # noqa: E402
from forte.fast.vlm import image_part, jpeg_data_url  # noqa: E402


class PixelJPEGCache:
    """Bounded per-video RGB cache; raw pixels make stale identity impossible.

    Cache only complete immutable encoded strings, never mutable part objects.
    The source decoder yields RGB. Reject other modes rather than confusing
    palette/metadata changes with identical pixels. This class is intended for
    one event-loop owner; it does not alter request scheduling or make awaits.
    """

    def __init__(self, max_bytes: int = 32 * 1024 * 1024, max_entries: int = 64):
        if type(max_bytes) is not int or max_bytes < 0 or type(max_entries) is not int or max_entries < 1:
            raise ValueError("cache limits must be nonnegative bytes and positive entries")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._cache: OrderedDict[tuple, tuple[str, int]] = OrderedDict()
        self.cached_bytes = 0
        self.lookups = self.hits = self.encodes = 0

    def clear(self) -> None:
        self._cache.clear()
        self.cached_bytes = 0
        self.lookups = self.hits = self.encodes = 0

    def url(self, image: Image.Image, *, quality: int = 88, max_side: int | None = None) -> str:
        if image.mode != "RGB":
            raise ValueError("candidate cache accepts decoded RGB images only")
        if type(quality) is not int or not 1 <= quality <= 100:
            raise ValueError("quality must be an integer between 1 and 100")
        if max_side is not None and (type(max_side) is not int or max_side < 1):
            raise ValueError("max_side must be positive or None")
        pixels = image.tobytes()
        key = (image.size, quality, max_side, pixels)
        self.lookups += 1
        if key in self._cache:
            self.hits += 1
            encoded, _ = self._cache[key]
            self._cache.move_to_end(key)
            return encoded
        encoded = jpeg_data_url(image, quality=quality, max_side=max_side)
        self.encodes += 1
        # Raw key bytes are included; caching must not secretly retain an
        # unbounded second copy of every original frame.
        size = len(pixels) + len(encoded.encode("ascii"))
        if size <= self.max_bytes:
            while self._cache and (self.cached_bytes + size > self.max_bytes or len(self._cache) >= self.max_entries):
                _, (_, previous_size) = self._cache.popitem(last=False)
                self.cached_bytes -= previous_size
            self._cache[key] = (encoded, size)
            self.cached_bytes += size
        return encoded

    def part(self, image: Image.Image, **kwargs) -> dict:
        return {"type": "image_url", "image_url": {"url": self.url(image, **kwargs)}}

    def stats(self) -> dict:
        return {"lookups": self.lookups, "hits": self.hits, "encodes": self.encodes,
                "entries": len(self._cache), "cached_bytes": self.cached_bytes,
                "max_bytes": self.max_bytes}


def _part_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def benchmark(images: list[Image.Image], *, repeats: int = 8, rounds: int = 5,
              quality: int = 88, max_side: int | None = 768,
              max_bytes: int = 32 * 1024 * 1024,
              workload: list[tuple[Image.Image, int, int | None]] | None = None) -> dict:
    """Same ordered workload each round, including cold fills and unique inputs."""
    if not images or type(repeats) is not int or repeats < 1 or type(rounds) is not int or rounds < 1:
        raise ValueError("images, repeats and rounds must be positive")
    requests = workload if workload is not None else [(image, quality, max_side) for _ in range(repeats) for image in images]
    if not requests:
        raise ValueError("workload must include at least one logical image part")
    baseline_times, candidate_times, candidate_stats = [], [], []
    expected_digest = None
    for round_index in range(rounds):
        # Alternate measurement order to reduce systematic warmup bias.
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        digests = {}
        for name in order:
            cache = PixelJPEGCache(max_bytes=max_bytes)
            digest = sha256()
            started = time.perf_counter()
            for image, request_quality, request_max_side in requests:
                part = image_part(image, quality=request_quality, max_side=request_max_side) if name == "baseline" else cache.part(image, quality=request_quality, max_side=request_max_side)
                raw = _part_bytes(part)
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
            elapsed = time.perf_counter() - started
            digests[name] = digest.hexdigest()
            if name == "baseline":
                baseline_times.append(elapsed)
            else:
                candidate_times.append(elapsed)
                candidate_stats.append(cache.stats())
        if digests["baseline"] != digests["candidate"]:
            raise RuntimeError("candidate_request_bytes_differ")
        if expected_digest is not None and expected_digest != digests["baseline"]:
            raise RuntimeError("reference_codec_bytes_not_reproducible")
        expected_digest = digests["baseline"]
    baseline = statistics.median(baseline_times)
    candidate = statistics.median(candidate_times)
    return {
        "schema": "forte.jpeg-cache-candidate.v1", "scope": "cpu_codec_only; no_model_or_video_accuracy_claim",
        "logical_image_parts_per_round": len(requests), "source_images": len(images),
        "repeats": repeats, "rounds": rounds, "quality": quality, "max_side": max_side,
        "same_order_and_count": True, "request_parts_byte_identical": True,
        "ordered_request_parts_sha256": expected_digest,
        "baseline_seconds": baseline_times, "candidate_seconds": candidate_times,
        "baseline_median_seconds": baseline, "candidate_median_seconds": candidate,
        "codec_speedup": baseline / candidate if candidate else None, "candidate_stats": candidate_stats,
        "runtime_integrated": False, "model_calls": 0,
        "runtime": {"python": platform.python_version(), "pillow": pillow_version,
                    "platform": platform.system(), "processor": platform.processor()},
        "limitations": ["Repeated-image synthetic workload; real overlap frequency is not established.",
                        "No promise of faster unique images, model inference, or upload-to-result latency.",
                        "Cache budget bounds stored pixel/JPEG payload; active images and Python object overhead use additional RAM.",
                        "Live scheduling, retries and deadlines can change even when request bytes are identical."]}


def benchmark_analysis_geometry(images: list[Image.Image], *, rounds: int = 5,
                                max_bytes: int = 32 * 1024 * 1024) -> dict:
    """Mimic exact encoder/crop recipes on public stills, never scene labels.

    Every public image becomes a distinct synthetic 464x848 frame. Four state
    crops use the production expansion factors, and overlapping verification
    windows repeat original frame/crop parts. Synthetic crop boxes describe
    geometry only: they do not assert these stills depict desks, people or keys.
    """
    from forte.fast.analyzer import crop

    frames = [image.resize((464, 848), Image.Resampling.BICUBIC) for image in images]
    crops = []
    workload = []
    boxes = [[50, 50, 450, 350], [0, 350, 1000, 850], [0, 100, 400, 950], [550, 100, 1000, 950]]
    try:
        for frame in frames:
            workload.append((frame, 88, 896))
            for box, expansion in zip(boxes, (1.4, 1.15, 1.8, 1.8)):
                region = crop(frame, box, expand=expansion)
                crops.append(region)
                workload.append((region, 88, 768))
        # Windows overlap by two public stills, with no omitted logical parts.
        for start in range(0, len(frames), 2):
            for frame in frames[max(0, start - 2):min(len(frames), start + 4)]:
                workload.append((frame, 88, 768))
            frame = frames[start]
            for box in boxes:
                region = crop(frame, box, expand=1.6)
                crops.append(region)
                workload.append((region, 88, 640))
        report = benchmark(frames, repeats=1, rounds=rounds, max_bytes=max_bytes, workload=workload)
        report["workload_geometry"] = "public_stills_resized_to_464x848; production_crop_expansions; synthetic_overlapping_verification_windows"
        report["real_video_hit_ratio_claim"] = False
        return report
    finally:
        for image in (*frames, *crops):
            image.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--source-url", required=True, help="Reviewed public publisher/dataset URL")
    parser.add_argument("--license", required=True, help="Reviewed license identifier; declaration is not a legal verification")
    parser.add_argument("--public-receipt", type=Path,
                        help="Source/license/path/SHA receipt required for public samples stored under private roots")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--max-cache-mib", type=int, default=32)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    source = urlsplit(args.source_url)
    if source.scheme != "https" or not source.hostname or source.username or source.password:
        parser.error("--source-url must be a public HTTPS publisher URL without embedded credentials")
    if not args.license.strip() or args.max_cache_mib < 0:
        parser.error("a license declaration and nonnegative cache size are required")
    receipt = None
    if args.public_receipt:
        receipt = json.loads(args.public_receipt.read_text(encoding="utf-8"))
        if (not isinstance(receipt, dict) or receipt.get("source") != args.source_url
                or receipt.get("license") != args.license or not isinstance(receipt.get("files"), dict)):
            parser.error("public receipt source/license/files do not match this benchmark")
    images, inputs = [], []
    try:
        for path in args.image:
            declared = None
            if receipt:
                declared = next((row for row in receipt["files"].values() if isinstance(row, dict)
                                 and isinstance(row.get("path"), str)
                                 and Path(row["path"]).resolve() == path.resolve()), None)
                if declared is None:
                    parser.error("image path is absent from the explicit public receipt")
            elif any(part.lower() in {"private", "videos", "evidence", "frames"} for part in path.parts):
                parser.error("public samples in private storage require an explicit public receipt")
            raw = path.read_bytes()
            if declared and (sha256(raw).hexdigest() != declared.get("sha256") or len(raw) != declared.get("bytes")):
                parser.error("public image bytes differ from the source receipt")
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
            inputs.append({"sha256": sha256(raw).hexdigest(), "bytes": len(raw)})
        report = benchmark(images, repeats=args.repeats, rounds=args.rounds,
                           quality=args.quality, max_side=args.max_side,
                           max_bytes=args.max_cache_mib * 1024 * 1024)
        report["unique_input_probe"] = benchmark(images, repeats=1, rounds=args.rounds,
                                                 quality=args.quality, max_side=args.max_side,
                                                 max_bytes=args.max_cache_mib * 1024 * 1024)
        report["analysis_geometry_probe"] = benchmark_analysis_geometry(images, rounds=args.rounds,
                                                                         max_bytes=args.max_cache_mib * 1024 * 1024)
    finally:
        for image in images:
            image.close()
    report["public_input_provenance"] = {"source_url": args.source_url, "declared_license": args.license,
                                          "inputs": inputs, "license_verified_by_tool": False}
    if args.public_receipt:
        report["public_input_provenance"]["receipt_sha256"] = sha256(args.public_receipt.read_bytes()).hexdigest()
    report["candidate_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    report["reference_source_sha256"] = sha256((Path(__file__).resolve().parents[1] / "src/forte/fast/vlm.py").read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("scope", "codec_speedup", "request_parts_byte_identical", "logical_image_parts_per_round", "runtime_integrated")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
