"""Synthetic CPU-only candidate tests; no organizer footage/model requests."""

from __future__ import annotations

from hashlib import sha256
import json
import pytest
from PIL import Image

from evaluation.codec_cache_candidate import PixelJPEGCache, benchmark, benchmark_analysis_geometry, main
from forte.fast.vlm import image_part


@pytest.mark.parametrize("size,max_side,quality", [((464, 848), 768, 88), ((64, 96), 896, 88),
                                                    ((120, 100), None, 90), ((43, 51), 16, 31)])
def test_cached_parts_are_exact_reference_bytes_and_fresh_dicts(size, max_side, quality):
    image = Image.new("RGB", size, "navy")
    cache = PixelJPEGCache()
    expected = image_part(image, max_side=max_side, quality=quality)
    first = cache.part(image, max_side=max_side, quality=quality)
    assert first == expected
    first["image_url"]["url"] = "caller mutation"
    second = cache.part(image, max_side=max_side, quality=quality)
    assert second == expected and first is not second
    assert cache.stats()["encodes"] == cache.stats()["hits"] == 1


def test_content_key_detects_changed_pixels_and_reuses_equal_pixels_from_other_image():
    image = Image.new("RGB", (20, 30), "black")
    cache = PixelJPEGCache()
    original = cache.part(image)
    same = image.copy()
    assert cache.part(same) == original
    assert cache.hits == 1
    image.paste("white", (0, 0, 20, 30))
    assert cache.part(image) == image_part(image) != original
    assert cache.encodes == 2


def test_encoder_options_are_part_of_identity_and_cache_is_per_instance():
    image = Image.new("RGB", (60, 80), "red")
    first = PixelJPEGCache()
    for quality, max_side in ((88, None), (90, None), (88, 768), (88, 32)):
        assert first.part(image, quality=quality, max_side=max_side) == image_part(image, quality=quality, max_side=max_side)
    assert first.encodes == 4 and first.hits == 0
    second = PixelJPEGCache()
    second.part(image)
    assert second.hits == 0 and second.encodes == 1
    first.clear()
    assert first.stats() == {"lookups": 0, "hits": 0, "encodes": 0, "entries": 0, "cached_bytes": 0, "max_bytes": 32 * 1024 * 1024}


def test_lru_evicts_oldest_and_counts_pixel_key_memory():
    images = [Image.new("RGB", (20, 30), color) for color in ("red", "green", "blue")]
    cache = PixelJPEGCache(max_bytes=100_000, max_entries=2)
    cache.part(images[0])
    cache.part(images[1])
    cache.part(images[0])  # Keep red, evict green next.
    cache.part(images[2])
    assert cache.stats()["entries"] == 2
    assert cache.cached_bytes > 2 * 20 * 30 * 3
    cache.part(images[1])
    assert cache.encodes == 4
    assert cache.cached_bytes <= cache.max_bytes


def test_zero_budget_and_oversized_image_never_cache():
    image = Image.new("RGB", (20, 30), "red")
    for budget in (0, 10):
        cache = PixelJPEGCache(max_bytes=budget)
        assert cache.part(image) == cache.part(image) == image_part(image)
        assert cache.stats()["entries"] == cache.cached_bytes == cache.hits == 0
        assert cache.encodes == 2


@pytest.mark.parametrize("kwargs", [{"max_bytes": -1}, {"max_bytes": True}, {"max_entries": 0}])
def test_invalid_limits_fail(kwargs):
    with pytest.raises(ValueError):
        PixelJPEGCache(**kwargs)


@pytest.mark.parametrize("kwargs", [{"quality": 0}, {"quality": True}, {"max_side": 0}, {"max_side": False}])
def test_invalid_encoder_options_fail(kwargs):
    with pytest.raises(ValueError):
        PixelJPEGCache().part(Image.new("RGB", (5, 5)), **kwargs)


def test_palette_mode_requires_explicit_rgb_conversion():
    with pytest.raises(ValueError, match="RGB"):
        PixelJPEGCache().part(Image.new("P", (5, 5)))


def test_codec_benchmark_proves_all_ordered_parts_equal_without_claiming_model_speed():
    images = [Image.new("RGB", (40, 60), color) for color in ("red", "blue")]
    report = benchmark(images, repeats=3, rounds=2)
    assert report["logical_image_parts_per_round"] == 6
    assert report["request_parts_byte_identical"] and report["same_order_and_count"]
    assert report["model_calls"] == 0 and report["runtime_integrated"] is False
    assert all(row["encodes"] == 2 and row["hits"] == 4 for row in report["candidate_stats"])
    assert report["codec_speedup"] > 0  # No speed target asserted on synthetic tiny images.


def test_unique_workload_reports_cache_overhead_honestly():
    images = [Image.new("RGB", (40, 60), color) for color in ("red", "blue")]
    report = benchmark(images, repeats=1, rounds=2)
    assert all(row["hits"] == 0 and row["encodes"] == 2 for row in report["candidate_stats"])
    assert report["request_parts_byte_identical"]


def test_production_geometry_probe_preserves_every_ordered_request_part():
    images = [Image.new("RGB", (40, 60), color) for color in ("red", "blue", "green", "navy")]
    report = benchmark_analysis_geometry(images, rounds=1)
    # 4 frames x5 state parts, plus windows of4+4 frames and4 crops each.
    assert report["logical_image_parts_per_round"] == 36
    assert report["request_parts_byte_identical"]
    assert report["real_video_hit_ratio_claim"] is False
    assert report["candidate_stats"][0]["hits"] > 0


def test_public_sample_in_private_storage_requires_exact_receipt(tmp_path):
    path = tmp_path / "private/public/sample.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 20), "red").save(path)
    output = tmp_path / "codec.json"
    args = ["--image", str(path), "--source-url", "https://publisher.example/dataset", "--license", "CC-BY-4.0",
            "--rounds", "1", "--repeats", "2", "--report", str(output)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2 and not output.exists()
    receipt = tmp_path / "receipt.json"
    raw = path.read_bytes()
    record = {"source": "https://publisher.example/dataset", "license": "CC-BY-4.0",
              "files": {"sample.jpg": {"path": str(path), "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}}}
    receipt.write_text(json.dumps(record), encoding="utf-8")
    assert main([*args, "--public-receipt", str(receipt)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["public_input_provenance"]["receipt_sha256"] == sha256(receipt.read_bytes()).hexdigest()
    assert str(path) not in output.read_text(encoding="utf-8")
    record["files"]["sample.jpg"]["sha256"] = "changed"
    receipt.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main([*args, "--public-receipt", str(receipt)])
    assert error.value.code == 2
