"""CPU-free guard tests for the generated processor audit's own boundaries."""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "contracts/audit_grounding_processor.py"
SPEC = importlib.util.spec_from_file_location("grounding_processor_audit_test", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ProcessorAuditGuards(unittest.TestCase):
    def asset_fixture(self, root):
        assets = root / "assets"
        assets.mkdir()
        rows = []
        for index in range(12):
            raw = ("public generated config " + str(index)).encode()
            name = str(index) + ".json"
            (assets / name).write_bytes(raw)
            rows.append({"path": name, "kind": "processor_or_config", "size_bytes": len(raw),
                         "sha256": hashlib.sha256(raw).hexdigest()})
        rows.append({"path": "model.safetensors", "kind": "weights", "size_bytes": 1000000,
                     "sha256": "0" * 64})
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"revision": AUDIT.REVISION, "processor_revision": AUDIT.REVISION,
                                       "files": rows}), encoding="utf-8")
        return assets, manifest

    def test_exact_nonweight_set_hashes_without_weight_access(self):
        with tempfile.TemporaryDirectory() as name:
            assets, manifest = self.asset_fixture(Path(name))
            with patch.object(AUDIT, "MANIFEST_SHA", AUDIT.sha(manifest)):
                result = AUDIT.verify_assets(assets, manifest)
            self.assertEqual(len(result), 12)
            self.assertFalse((assets / "model.safetensors").exists())

    def test_weight_file_presence_is_rejected_before_asset_reads(self):
        with tempfile.TemporaryDirectory() as name:
            assets, manifest = self.asset_fixture(Path(name))
            (assets / "model.safetensors").write_bytes(b"never read")
            with patch.object(AUDIT, "MANIFEST_SHA", AUDIT.sha(manifest)):
                with self.assertRaisesRegex(RuntimeError, "audit_asset_set_invalid"):
                    AUDIT.verify_assets(assets, manifest)

    def test_public_asset_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            assets, manifest = self.asset_fixture(Path(name))
            (assets / "0.json").write_bytes(b"x" * (assets / "0.json").stat().st_size)
            with patch.object(AUDIT, "MANIFEST_SHA", AUDIT.sha(manifest)):
                with self.assertRaisesRegex(RuntimeError, "audit_asset_digest_invalid"):
                    AUDIT.verify_assets(assets, manifest)

    def test_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            assets, manifest = self.asset_fixture(Path(name))
            with self.assertRaisesRegex(RuntimeError, "audit_manifest_pin_invalid"):
                AUDIT.verify_assets(assets, manifest)

    def test_odd_tail_pair_is_true_last_timestamp(self):
        self.assertEqual(AUDIT.independent_pair_means((0, 17, 44, 91, 150), (0, 2, 4), Fraction(1, 100)),
                         (.22, 1.5))

    def test_cases_cover_nonzero_vfr_rational_and_odd_conditions(self):
        cases = AUDIT.generated_cases()
        self.assertTrue(any(origin != 0 for _, _, origin, _, _ in cases))
        self.assertTrue(any(base.numerator != 1 for _, base, _, _, _ in cases))
        self.assertTrue(any(len(ordinals) % 2 for _, _, _, _, ordinals in cases))
        self.assertTrue(any(len({second-first for first, second in zip(offsets, offsets[1:])}) > 1
                            for _, _, _, offsets, _ in cases))


if __name__ == "__main__":
    unittest.main()
