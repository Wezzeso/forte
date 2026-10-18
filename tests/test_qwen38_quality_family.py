"""Metadata-only baseline-first family checks, no real media or model calls."""
from copy import deepcopy
import unittest

from evaluation.io import digest
from evaluation.qwen38_comparison import ARMS
from evaluation.qwen38_quality_family import arm_plan, compare_family
from evaluation.weak_supervision import provided_label_digest
from test_qwen38_operational_evaluation import worker_recipe


def fixtures():
    durations = [72.7, 20.866667, 23.8, 22.6, 17.3, 78.733021]
    records = [{"sample_id": f"synthetic_{i}", "content_sha256": str(i+1)*64, "split": "validation"} for i in range(6)]
    labels = [{"content_sha256": row["content_sha256"], "split": "validation", "provided_video_label": i<3,
               "provided_label_origin": "organizer_workbook", "target_scope": "video_binary_mil_only",
               "teacher_used_as_binary_ground_truth": False} for i,row in enumerate(records)]
    recipe = {"worker_settings": worker_recipe(), "provider_max_attempts": 2,
        "generation": {"max_new_tokens": 4096, "seed": 20261017}, "source_archive_sha256": "a"*64,
        "rules_sha256": "b"*64, "prompt_source_sha256": "c"*64, "context_tokens": 32768}
    family = {"schema_version": "forte.qwen38-quality-family.v1", "records": records,
        "attempts": [{**row, "arm": arm, "repeat_index": 0, "attempt_id": f"{arm}-{i}"}
                     for arm in ARMS for i,row in enumerate(records)],
        "base_models": {arm: {"model_id": "synthetic8" if arm=="baseline_8b" else "synthetic27", "model_revision": "1"*40} for arm in ARMS},
        "recipe": recipe, "recipe_sha256": digest(recipe), "provided_labels_sha256": provided_label_digest(labels)}
    spec = {"arm": "baseline_8b", "expected_model_identity": {**family["base_models"]["baseline_8b"], "adapter_sha256": None},
        "provider": {}, "runtime_bindings": {"source_archive_sha256": "a"*64},
        "sources": [{**row, "owned_path": f"/private/validation/{row['content_sha256']}.mov", "source_bytes": 100,
                     "duration_sec": durations[i]} for i,row in enumerate(records)]}
    return family, spec, labels


class QualityFamilyTests(unittest.TestCase):
    def test_baseline_first_needs_no_future_adapter_and_freezes_208_slots(self):
        family, spec, labels = fixtures()
        self.assertNotIn("adapter_sha256", family["base_models"]["qwen38_tuned"])
        plan = arm_plan(family, spec, role_config_sha256="d"*64)
        self.assertEqual(len(plan["source_attempts"]), 6)
        self.assertEqual(len(plan["attempts"]), 208)
        self.assertEqual(sum(case["coarse_windows"] for case in plan["cases"]), 56)
        self.assertIsNone(plan["expected_model_identity"]["adapter_sha256"])
        self.assertEqual(plan["comparison_plan_sha256"], digest(family))

    def test_tuned_arm_requires_actual_adapter_before_its_execution_plan(self):
        family, spec, _ = fixtures()
        spec["arm"] = "qwen38_tuned"
        spec["expected_model_identity"] = {**family["base_models"]["qwen38_tuned"], "adapter_sha256": None}
        with self.assertRaises(ValueError):
            arm_plan(family, spec, role_config_sha256="d"*64)
        spec["expected_model_identity"]["adapter_sha256"] = "f"*64
        self.assertEqual(arm_plan(family, spec, role_config_sha256="d"*64)["expected_model_identity"]["adapter_sha256"], "f"*64)

    def test_shared_source_recipe_or_heldout_population_changes_are_rejected(self):
        family, spec, _ = fixtures()
        spec["runtime_bindings"]["source_archive_sha256"] = "f"*64
        with self.assertRaises(ValueError):
            arm_plan(family, spec, role_config_sha256="d"*64)
        family, spec, _ = fixtures()
        family["records"][0]["split"] = "final_check"
        with self.assertRaises(ValueError):
            arm_plan(family, spec, role_config_sha256="d"*64)

    def test_original_family_results_compare_without_rewriting_provenance(self):
        family, spec, labels = fixtures()
        plans, terminals = [], []
        for arm in ARMS:
            current = deepcopy(spec)
            current["arm"] = arm
            current["expected_model_identity"] = {**family["base_models"][arm], "adapter_sha256": "e"*64 if arm=="qwen38_tuned" else None}
            plan = arm_plan(family,current,role_config_sha256="d"*64)
            plans.append(plan)
            for i,attempt in enumerate(plan["source_attempts"]):
                terminals.append({**attempt,"comparison_plan_sha256":digest(family),"recipe_sha256":family["recipe_sha256"],
                    "expected_model_identity":plan["expected_model_identity"],"status":"timeout","public_contract_validated":False})
        original = digest(terminals)
        metadata = {"inference_kind":"synthetic","final_check_used":False}
        report = compare_family(family,plans,terminals,labels,metadata=metadata)
        self.assertEqual(digest(terminals),original)
        self.assertEqual(report["status"],"SYNTHETIC_ONLY")
        for arm in ARMS:
            self.assertEqual(report["arms"][arm]["completion"]["failed_or_unprocessed"],6)
            self.assertEqual(report["arms"][arm]["provided_binary"]["unprocessed_positive"],3)
            self.assertEqual(report["arms"][arm]["provided_binary"]["unprocessed_negative"],3)
        terminals[0]["comparison_plan_sha256"] = "f"*64
        with self.assertRaises(ValueError):
            compare_family(family,plans,terminals,labels,metadata=metadata)


if __name__=="__main__":
    unittest.main()
