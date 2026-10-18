"""Operational binder validation with synthetic artifacts; never imports torch."""

from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest

from deploy.qwen38_bind_runtime import (BindError, CORE_FILES, _write_private_new,
                                        PACKAGES, bind_quality_family, canonical, read_json, sha256, verify_source_tree)
from serving.tests.qwen38_adapter_fixtures import adapter_config_fixture, inventory_fixture, write_adapter_fixture


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.archive = self.root / "source.tar.gz"
        self.files = {name: b"# synthetic source fixture\n" for name in CORE_FILES}
        for name, content in self.files.items():
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        self.pack()

    def pack(self):
        with tarfile.open(self.archive, "w:gz") as package:
            for name, content in sorted(self.files.items()):
                entry = tarfile.TarInfo(name)
                entry.size = len(content)
                package.addfile(entry, BytesIO(content))

    def test_all_archived_core_bytes_verified_without_model_claim(self):
        proof = verify_source_tree(self.source, self.archive, sha256(self.archive))
        self.assertEqual(proof["archive_files_verified"], len(CORE_FILES))
        self.assertTrue(proof["source_verified_before_import"])
        self.assertEqual(set(proof["executed_core_files"]), CORE_FILES)
        self.assertNotIn("model_loaded", proof)

    def test_changed_source_and_unarchived_python_fail(self):
        target = self.source / next(iter(CORE_FILES))
        target.write_bytes(b"tampered")
        with self.assertRaisesRegex(BindError, "executed_source_bytes_mismatch"):
            verify_source_tree(self.source, self.archive, sha256(self.archive))
        target.write_bytes(self.files[target.relative_to(self.source).as_posix()])
        (self.source / "serving/unreviewed.py").write_bytes(b"unreviewed")
        with self.assertRaisesRegex(BindError, "unarchived_executable_source"):
            verify_source_tree(self.source, self.archive, sha256(self.archive))

    def test_unsafe_member_and_archive_digest_fail(self):
        with self.assertRaisesRegex(BindError, "source_archive_mismatch"):
            verify_source_tree(self.source, self.archive, "0" * 64)
        self.files["../escape.py"] = b"untrusted"
        self.pack()
        with self.assertRaisesRegex(BindError, "unsafe_artifact_path"):
            verify_source_tree(self.source, self.archive, sha256(self.archive))

    def test_outputs_are_private_and_never_overwritten(self):
        path = self.root / "private/bound.json"
        _write_private_new(path, {"model_loaded": False})
        self.assertFalse(read_json(path)["model_loaded"])
        with self.assertRaisesRegex(BindError, "output_already_exists"):
            _write_private_new(path, {"model_loaded": True})

    def test_nonfinite_and_duplicate_input_rejected(self):
        path = self.root / "bad.json"
        for content in ('{"x":1e999}', '{"x":1,"x":2}', '{"x":NaN}'):
            path.write_text(content)
            with self.assertRaises(BindError):
                read_json(path)

    def test_frozen_family_exact_profile_and_prompt_bytes_are_bound(self):
        # All values are synthetic; no source media or model access.
        code = self.root / "family-code"
        for name in ("src/forte/pipeline.py", "rules/catalog.json", "annotations/taxonomy.v1.json"):
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(name.encode())
        config = {"limits": {"max_images": 16, "max_context_tokens": 32768, "max_pixels_per_image": 262144},
                  "generation": {"max_new_tokens": 4096, "seed": 20261017, "do_sample": False,
                                 "temperature": 0, "num_beams": 1},
                  "source_archive": {"sha256": "a" * 64}}
        recipe = {"worker_settings": {"window_seconds": 8, "stride_seconds": 4, "sample_fps": 2,
                    "max_frames_per_window": 16, "staged_screening": False, "max_concurrent_clips": 1,
                    "max_concurrent_jobs": 1, "max_inflight_per_job": 1, "max_refinement_passes": 8,
                    "job_timeout_seconds": 285}, "context_tokens": 32768, "generation": config["generation"],
                  "rule_selection": {"enabled_rules": 4, "optional_expanded_rules_selected": False, "visible_fact_axes": 19},
                  "prompt_implementation_pins": {"src/forte/pipeline.py": sha256(code / "src/forte/pipeline.py")},
                  "rules_sha256": sha256(code / "rules/catalog.json"),
                  "taxonomy_sha256": sha256(code / "annotations/taxonomy.v1.json"),
                  "source_archive_sha256": "b" * 64}
        family = self.root / "family.json"
        family.write_text(json.dumps({"schema_version": "forte.qwen38-quality-family.v1", "recipe": recipe,
                                      "recipe_sha256": canonical(recipe)}))
        proof = bind_quality_family(family, sha256(family), config, code)
        self.assertEqual(proof["family_source_archive_sha256"], "b" * 64)
        self.assertEqual(proof["actual_serving_source_archive_sha256"], "a" * 64)
        self.assertTrue(proof["same_family_prompt_and_pipeline_bytes_verified"])
        self.assertTrue(proof["decoder_constrained_json"])
        config["limits"]["max_images"] = 8
        with self.assertRaisesRegex(BindError, "runtime_profile_mismatch"):
            bind_quality_family(family, sha256(family), config, code)

    def test_bind_completed_fixture_against_immutable_real_source_in_fresh_process(self):
        archive = Path("tmp/qwen38-ml-phase2-20261017T132453Z.tar.gz").resolve()
        if not archive.exists():
            self.skipTest("optional immutable source archive is absent")
        frozen = self.root / "frozen"
        frozen.mkdir()
        # Public source archive only. Verify each safe regular-file path before
        # fixture extraction; no private data/model or external inference.
        with tarfile.open(archive) as package:
            for member in package:
                self.assertTrue(member.isfile())
                self.assertFalse(Path(member.name).is_absolute())
                self.assertNotIn("..", Path(member.name).parts)
                target = frozen / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(package.extractfile(member).read())
        # Historical df11 is immutable and predates native grammar. Build an
        # explicitly synthetic archive fixture containing the current public
        # validator code; this does not rewrite any release artifact.
        for name in ("serving/qwen38_grammar.py", "serving/qwen38_identity.py", "serving/qwen38_service.py",
                     "serving/qwen38_adapter_contract.py", "serving/qwen38_source_lineage.py",
                     "serving/qwen38_untuned.py", "serving/qwen38_comparison_profile.py"):
            (frozen / name).write_bytes(Path(name).read_bytes())
        archive = self.root / "synthetic-current-validator-source.tar.gz"
        with tarfile.open(archive, "w:gz") as package:
            for target in sorted(frozen.rglob("*")):
                if target.is_file():
                    package.add(target, arcname=target.relative_to(frozen).as_posix())
        run_root = self.root / "run"
        adapter = run_root / "adapter"
        adapter.mkdir(parents=True)
        revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        (adapter / "adapter_config.json").write_text(json.dumps(adapter_config_fixture()))
        write_adapter_fixture(adapter / "adapter_model.safetensors")
        checksums = {path.name: sha256(path) for path in adapter.iterdir()}
        manifest = frozen / "configs/serving/qwen38/public-candidate.manifest.json"
        run = {"schema_version": "forte.qwen38.silver-run.v1", "status": "candidate_unselected",
               "gpu_executed": True, "dataset_origin": "local_teacher_silver", "human_verified": False,
               "optimizer_steps": 2, "seed": 20261017, "code_archive_sha256": sha256(archive),
               "runtime_image_sha256": "a" * 64, "processor_limits": {"max_pixels_per_image": 262144},
               "base_manifest_sha256": sha256(manifest), "base_revision": revision, "processor_revision": revision,
               "changed_adapter_tensors": 1, "changed_adapter_groups": {"self_attn": True, "linear_attn": True, "mlp": True},
               "finite_loss_verified": True, "gradients_verified": True, "frozen_tensors_unchanged": True,
               "peft_load_verified": True, "reload_forward_verified": True,
               "adapter_checksums": checksums, "adapter_tree_sha256": canonical(checksums),
               "quantization": {"format": "nf4", "compute_dtype": "bfloat16", "double_quant": True},
               "adapted_module_inventory": inventory_fixture()}
        (run_root / "run.json").write_text(json.dumps(run))
        packages = self.root / "packages.json"
        package_map = {name: "1.2.3" for name in PACKAGES}
        package_map["xgrammar"] = "0.2.7"
        packages.write_text(json.dumps(package_map))
        output = self.root / "output"
        result = subprocess.run([sys.executable, "-B", str(Path("deploy/qwen38_bind_runtime.py").resolve()),
            "--code-root", str(frozen), "--source-archive", str(archive), "--source-sha256", sha256(archive),
            "--run-root", str(run_root), "--image-sha256", "a" * 64,
            "--packages-json", str(packages), "--output-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        receipt = read_json(output / "runtime-binding.receipt.json")
        tuned, untuned = read_json(output / "candidate.runtime.bound.json"), read_json(output / "untuned.runtime.bound.json")
        self.assertEqual(tuned["generation"], untuned["generation"])
        self.assertNotIn("training_source_archive", untuned)
        self.assertNotIn("training_source_lineage", untuned)
        self.assertEqual(tuned["limits"]["max_pixels_per_image"], 262144)
        self.assertFalse(receipt["model_loaded"])
        self.assertFalse(receipt["gpu_initialized"])
        self.assertTrue(receipt["source_verification"]["source_verified_before_import"])
        self.assertEqual(receipt["adapter_topology"]["serialized_tensor_count"], 992)
        # The fit keeps its original source digest while a new serving release
        # contains the reviewed validator-only change. Neither run nor adapter
        # bytes may be rewritten to manufacture equal archive identities.
        run_sha = sha256(run_root / "run.json")
        old_archive = archive
        new_archive = self.root / "synthetic-reviewed-serving-source.tar.gz"
        identity_path = frozen / "serving/qwen38_identity.py"
        identity_path.write_bytes(identity_path.read_bytes() + b"\n# generated reviewed lineage fixture\n")
        with tarfile.open(new_archive, "w:gz") as package:
            for target in sorted(frozen.rglob("*")):
                if target.is_file():
                    package.add(target, arcname=target.relative_to(frozen).as_posix())
        new_output = self.root / "new-serving-output"
        command = [sys.executable, "-B", str(Path("deploy/qwen38_bind_runtime.py").resolve()),
            "--code-root", str(frozen), "--source-archive", str(new_archive), "--source-sha256", sha256(new_archive),
            "--training-source-archive", str(old_archive), "--training-source-sha256", sha256(old_archive),
            "--run-root", str(run_root), "--image-sha256", "a"*64, "--packages-json", str(packages),
            "--output-dir", str(new_output)]
        distinct = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(distinct.returncode, 0, distinct.stdout)
        lineage = read_json(new_output / "runtime-binding.receipt.json")["training_source_lineage"]
        self.assertEqual(lineage["training_source_archive_sha256"], sha256(old_archive))
        self.assertEqual(lineage["runtime_source_archive_sha256"], sha256(new_archive))
        self.assertFalse(lineage["same_archive"])
        self.assertEqual(sha256(run_root / "run.json"), run_sha)


if __name__ == "__main__":
    unittest.main()
