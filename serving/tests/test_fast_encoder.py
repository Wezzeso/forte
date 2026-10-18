"""CPU-only safety/provenance checks; these never instantiate an ML model."""

import array
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from serving.fast_encoder import (
    EncodedBatch, FastEncoder, FastEncoderConfig, FastEncoderError,
    _expected_rgb_bytes, canonical_sha256, encoder_recipe, validate_rgb_buffer,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/serving/encoders/siglip2-base-patch16-256.manifest.json"
MANIFEST_SHA = "cd51863090e8c411a1227d8039e44118c44488243e3d615d73907106568877e7"


class FastEncoderCpuTests(unittest.TestCase):
    def config(self, **changes):
        return FastEncoderConfig(Path("not-downloaded"), MANIFEST, MANIFEST_SHA, **changes)

    def test_import_and_help_work_without_site_packages_or_ml_imports(self):
        result = subprocess.run([sys.executable, "-S", "-c",
            "import json,sys; import serving.fast_encoder; print(json.dumps([x for x in "
            "('torch','transformers','numpy','PIL','forte.pipeline') if x in sys.modules]))"],
            cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])
        help_result = subprocess.run([sys.executable, "-S", "-m", "serving.fast_encoder", "--help"],
            cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertIn("--snapshot", help_result.stdout)

    def test_batch_geometry_and_byte_ceiling_checked_before_allocation(self):
        cfg = self.config(batch_size=2, max_input_bytes=100)
        self.assertEqual(_expected_rgb_bytes((2, 2, 3, 3), cfg), 36)
        for shape in ((3, 2, 3, 3), (0, 2, 3, 3), (2, 2, 3, 4), (True, 2, 3, 3),
                      (2, 8193, 1, 3), (2, 10, 10, 3)):
            with self.subTest(shape=shape), self.assertRaises(FastEncoderError):
                _expected_rgb_bytes(shape, cfg)

    def test_wrong_size_nonbyte_and_noncontiguous_buffers_rejected(self):
        cfg = self.config(batch_size=1)
        self.assertEqual(validate_rgb_buffer(bytes(18), (1, 2, 3, 3), cfg), (1, 2, 3, 3))
        bad = (bytes(17), bytearray(18), memoryview(bytes(36))[::2], memoryview(array.array('H', [0]*9)))
        for pixels in bad:
            with self.subTest(type=type(pixels).__name__), self.assertRaises(FastEncoderError):
                validate_rgb_buffer(pixels, (1, 2, 3, 3), cfg)

    def test_decoder_preprocessing_requires_identity_and_exact_target_geometry(self):
        with self.assertRaisesRegex(FastEncoderError, "transform_identity_required"):
            self.config(preprocessing_mode="decoder_rgb_256")
        cfg = self.config(preprocessing_mode="decoder_rgb_256", input_transform_sha256="a"*64)
        self.assertEqual(_expected_rgb_bytes((1, 256, 256, 3), cfg), 256*256*3)
        with self.assertRaisesRegex(FastEncoderError, "geometry_mismatch"):
            _expected_rgb_bytes((1, 255, 256, 3), cfg)
        with self.assertRaises(FastEncoderError):
            self.config(input_transform_sha256="a"*64)

    def test_manifest_failure_precedes_any_optional_runtime_import(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {"torch", "transformers", "numpy", "PIL"}:
                self.fail("ML import occurred before pinned artifact verification")
            return original(name, *args, **kwargs)
        with TemporaryDirectory() as directory:
            cfg = replace(self.config(), manifest=Path(directory)/"missing.json")
            with patch("builtins.__import__", side_effect=guarded):
                from forte.model.encoder_staging import EncoderManifestError
                with self.assertRaises(EncoderManifestError):
                    FastEncoder(cfg).load()

    def test_unloaded_encoder_does_not_initialize_runtime_from_pixels(self):
        runtime = FastEncoder(self.config())
        with self.assertRaisesRegex(FastEncoderError, "not_loaded"):
            runtime.encode_rgb_bytes(bytes(3), (1, 1, 1, 3))
        runtime.close()

    def test_decoder_batch_must_match_its_declared_transform_before_loading(self):
        from types import SimpleNamespace
        cfg = self.config(preprocessing_mode="decoder_rgb_256", input_transform_sha256="a"*64)
        runtime = FastEncoder(cfg)
        batch = SimpleNamespace(pixels=bytes(256*256*3), shape=(1,256,256,3), transform_sha256="b"*64)
        with self.assertRaisesRegex(FastEncoderError, "transform_identity_mismatch"):
            runtime.encode_frame_batch(batch)
        batch.transform_sha256 = "a"*64
        with self.assertRaisesRegex(FastEncoderError, "not_loaded"):
            runtime.encode_frame_batch(batch)

    def test_feature_values_are_absent_from_repr(self):
        class PrivateFeatures:
            def __repr__(self):
                raise AssertionError("Private features must never be formatted")
        batch = EncodedBatch(PrivateFeatures(), (1, 768), {"elapsed_sec": .1}, "a"*64)
        rendered = repr(batch)
        self.assertIn("768", rendered)
        self.assertNotIn("features=", rendered)

    def test_recipe_tracks_preprocessing_and_keeps_batch_size_out_of_feature_identity(self):
        manifest = json.loads(MANIFEST.read_text())
        versions = {"torch":"test","torchvision":"test","transformers":"test","pillow":"test","numpy":"test"}
        cfg = self.config()
        recipe = encoder_recipe(cfg, manifest, versions)
        same = encoder_recipe(replace(cfg, batch_size=64), manifest, versions)
        self.assertEqual(canonical_sha256(recipe), canonical_sha256(same))
        changed = encoder_recipe(replace(cfg, preprocessing_mode="decoder_rgb_256",
                                 input_transform_sha256="a"*64), manifest, versions)
        self.assertNotEqual(canonical_sha256(recipe), canonical_sha256(changed))
        self.assertEqual(recipe["feature_kind"], "pooler_output")
        self.assertFalse(recipe["l2_normalization"])
        self.assertEqual(recipe["processor_backend"], "torchvision_cpu")
        self.assertTrue(recipe["deterministic_algorithms"])
        self.assertFalse(recipe["allow_tf32"])
        self.assertEqual(recipe["cublas_workspace_config"], ":4096:8")
        self.assertEqual(set(recipe["implementation_sha256"]), {"fast_encoder.py", "feature_stream.py"})

    def test_configuration_cannot_request_cpu_or_unbounded_work(self):
        for changes in ({"device":"cpu"}, {"device":"cuda"}, {"batch_size":129},
                        {"batch_size":True}, {"manifest_sha256":"main"},
                        {"max_input_bytes":2**40}):
            with self.subTest(changes=changes), self.assertRaises(FastEncoderError):
                FastEncoderConfig(Path("x"), MANIFEST, **({"manifest_sha256":MANIFEST_SHA} | changes))


if __name__ == "__main__":
    unittest.main()
