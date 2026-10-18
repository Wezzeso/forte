"""Generated archive-only proofs; no private sources, model, CUDA or network."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from evaluation.io import digest
from evaluation.qwen38_quality_family import arm_plan
from evaluation.qwen38_quality_equivalence import bind_arm, compare, prove, validate_equivalence,comparable_runtime_profiles
from serving.qwen38_comparison_profile import runtime_profile_sha256
from test_qwen38_quality_family import fixtures
from test_qwen38_untuned import config_fixture


def archive(path, files):
    with tarfile.open(path, "w") as target:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            target.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.old, self.new = root/"old.tar", root/"new.tar"
        self.files = {"src/forte/pipeline.py": b"same pipeline", "rules/catalog.json": b"same rules",
            "annotations/taxonomy.v1.json": b"same taxonomy", "models/manifest.json": b"same manifest",
            "training/runner.py": b"old training"}
        self.old_sha = archive(self.old, self.files)
        self.new_files = {**self.files, "training/runner.py": b"new training", "serving/new_grammar.py": b"new grammar"}
        self.new_sha = archive(self.new, self.new_files)
        self.family, self.spec, self.labels = fixtures()
        recipe = self.family["recipe"]
        recipe.update(source_archive_sha256=self.old_sha,
            rules_sha256=hashlib.sha256(self.files["rules/catalog.json"]).hexdigest(),
            taxonomy_sha256=hashlib.sha256(self.files["annotations/taxonomy.v1.json"]).hexdigest(),
            prompt_implementation_pins={"src/forte/pipeline.py": hashlib.sha256(self.files["src/forte/pipeline.py"]).hexdigest()})
        recipe["prompt_source_sha256"] = digest(recipe["prompt_implementation_pins"])
        self.family["recipe_sha256"] = digest(recipe)
        self.config = config_fixture()
        for arm in ("qwen38_untuned", "qwen38_tuned"):
            self.family["base_models"][arm] = {key:self.config[key] for key in ("model_id", "model_revision")}
        self.spec["runtime_bindings"]["source_archive_sha256"] = self.old_sha
        self.receipt = prove(self.family,self.old,self.new,disclosure="Training mask and model serving grammar changed; common evaluator bytes unchanged.")

    def future(self, arm):
        spec, config = deepcopy(self.spec), deepcopy(self.config)
        spec.update(arm=arm, expected_model_identity={**self.family["base_models"][arm],
            "adapter_sha256":"f"*64 if arm=="qwen38_tuned" else None})
        spec["runtime_bindings"].update(source_archive_sha256=self.new_sha,
            base_manifest_sha256="a"*64, runtime_image_sha256="e"*64)
        config["source_archive"]["sha256"] = self.new_sha
        config["adapter_sha256"] = spec["expected_model_identity"]["adapter_sha256"]
        config["runtime_profile_sha256"] = runtime_profile_sha256(config)
        return bind_arm(self.family,spec,self.receipt,config,role_config_sha256="d"*64)

    def test_training_and_disclosed_serving_changes_keep_both_real_source_pins(self):
        before = digest(self.family)
        plan = self.future("qwen38_untuned")
        self.assertEqual(plan["runtime_bindings"]["source_archive_sha256"],self.new_sha)
        self.assertEqual(plan["evaluator_source_archive_sha256"],self.old_sha)
        self.assertEqual(plan["comparison_plan_sha256"],before)
        self.assertEqual(digest(self.family),before)
        self.assertEqual(len(plan["attempts"]),208)
        self.assertTrue(all(row["case_id"]==row["source_attempt_id"] for row in plan["attempts"]))
        self.assertEqual({row["scope"] for row in self.receipt["source_changes"]},{"training","model_serving"})

    def test_changed_common_prompt_pipeline_rule_taxonomy_or_manifest_fails(self):
        for name in ("src/forte/pipeline.py","rules/catalog.json","annotations/taxonomy.v1.json","models/manifest.json"):
            with self.subTest(name=name):
                archive(self.new,{**self.new_files,name:b"changed"})
                with self.assertRaises(ValueError):
                    prove(self.family,self.old,self.new,disclosure="changed source")

    def test_missing_disclosure_or_tampered_proof_or_wrong_actual_source_fails(self):
        with self.assertRaises(ValueError):
            prove(self.family,self.old,self.new,disclosure="")
        changed=deepcopy(self.receipt)
        changed["source_changes"]=[]
        with self.assertRaises(ValueError):validate_equivalence(self.family,changed,self.new_sha)
        with self.assertRaises(ValueError):validate_equivalence(self.family,self.receipt,"0"*64)

    def test_original_baseline_plan_and_failure_terminals_are_never_rewritten(self):
        plans=[arm_plan(self.family,self.spec,role_config_sha256="d"*64),self.future("qwen38_untuned"),self.future("qwen38_tuned")]
        terminals=[{**row,"comparison_plan_sha256":digest(self.family),"recipe_sha256":self.family["recipe_sha256"],
            "expected_model_identity":plan["expected_model_identity"],"status":"timeout","public_contract_validated":False}
            for plan in plans for row in plan["source_attempts"]]
        before=digest([self.family,plans,terminals])
        report=compare(self.family,plans,terminals,self.labels,metadata={"inference_kind":"synthetic","final_check_used":False})
        self.assertEqual(report["status"],"SYNTHETIC_ONLY")
        self.assertEqual(digest([self.family,plans,terminals]),before)
        self.assertEqual(report["arms"]["qwen38_tuned"]["actual_runtime_bindings"]["source_archive_sha256"],self.new_sha)
        self.assertEqual(report["arms"]["baseline_8b"]["actual_runtime_bindings"]["source_archive_sha256"],self.old_sha)
        self.assertTrue(all(row["completion"]["failed_or_unprocessed"]==6 for row in report["arms"].values()))
        plans[2]["comparison_runtime_profile"]["limits"]["max_pixels_per_image"]=262144
        plans[2]["comparison_runtime_profile_sha256"]=digest(plans[2]["comparison_runtime_profile"])
        with self.assertRaises(ValueError):compare(self.family,plans,terminals,self.labels,metadata={})

    def timeout_archive_pair(self):
        old=b'''def _provider_configuration(settings):
    values = {"request_timeout_sec": 60}
    if getattr(settings, "model_candidate_launch_config_path", None) is not None:
        config = {}
        values.update(
            seed=20261017,
        )
    return values
'''
        new=old.replace(b'            seed=20261017,\n',
            b'            seed=20261017,\n            request_timeout_sec=config["limits"]["request_timeout_seconds"],\n')
        self.files['src/forte/pipeline.py']=old
        self.old_sha=archive(self.old,self.files)
        self.new_files['src/forte/pipeline.py']=new
        self.new_sha=archive(self.new,self.new_files)
        recipe=self.family['recipe']
        recipe['source_archive_sha256']=self.old_sha
        recipe['prompt_implementation_pins']['src/forte/pipeline.py']=hashlib.sha256(old).hexdigest()
        recipe['prompt_source_sha256']=digest(recipe['prompt_implementation_pins'])
        self.family['recipe_sha256']=digest(recipe)
        self.spec['runtime_bindings']['source_archive_sha256']=self.old_sha
        return old,new

    def test_opt_in_exact_timeout_delta_has_explicit_counts_and_preserves_family(self):
        self.timeout_archive_pair()
        frozen=digest(self.family)
        with self.assertRaises(ValueError):prove(self.family,self.old,self.new,disclosure='timeout fix')
        self.receipt=prove(self.family,self.old,self.new,disclosure='Audited candidate caller timeout variant.',
            allow_candidate_timeout_forwarding=True)
        self.assertEqual(self.receipt['protected_common_file_count'],4)
        self.assertEqual(self.receipt['byte_identical_common_file_count'],3)
        self.assertEqual(self.receipt['audited_common_source_delta_count'],1)
        self.assertIs(self.receipt['all_common_files_byte_identical'],False)
        self.assertTrue(self.receipt['audited_common_source_deltas'][0]['proof']['baseline_default_unchanged'])
        plan=self.future('qwen38_tuned')
        self.assertEqual(plan['effective_client_request_timeout_seconds'],240)
        self.assertEqual(len(plan['audited_common_source_deltas']),1)
        self.assertEqual(digest(self.family),frozen)
        for field in ('protected_common_file_count','byte_identical_common_file_count',
            'byte_identical_common_file_sha256','audited_common_source_delta_count','all_common_files_byte_identical'):
            self.receipt.pop(field)
        self.receipt['receipt_sha256']=digest({key:value for key,value in self.receipt.items() if key!='receipt_sha256'})
        with self.assertRaises(ValueError):validate_equivalence(self.family,self.receipt,self.new_sha)

    def test_timeout_opt_in_never_allows_other_ast_bytes_or_new_protected_files(self):
        _,new=self.timeout_archive_pair()
        for replacement in (new.replace(b'"request_timeout_sec": 60',b'"request_timeout_sec": 240'),
            new+b'# extra comment\n',new.replace(b'config["limits"]["request_timeout_seconds"]',b'240')):
            archive(self.new,{**self.new_files,'src/forte/pipeline.py':replacement})
            with self.assertRaises(ValueError):prove(self.family,self.old,self.new,disclosure='timeout fix',
                allow_candidate_timeout_forwarding=True)
        archive(self.new,{**self.new_files,'src/forte/added.py':b'new protected module'})
        with self.assertRaises(ValueError):prove(self.family,self.old,self.new,disclosure='timeout fix',
            allow_candidate_timeout_forwarding=True)

    def inference_archive_pair(self):
        self.timeout_archive_pair()
        old=b'class CandidateRuntime:\n    def load(self):\n        self.model.set_adapter("candidate")\n        self.model.eval()\n'
        new=old.replace(b'set_adapter("candidate")',b'set_adapter("candidate", inference_mode=True)')
        self.new_files['serving/qwen38_service.py']=old
        previous=Path(self.tmp.name)/'predecessor.tar';previous_sha=archive(previous,self.new_files)
        predecessor=prove(self.family,self.old,previous,disclosure='Existing timeout variant',allow_candidate_timeout_forwarding=True)
        self.new_files['serving/qwen38_service.py']=new;self.new_sha=archive(self.new,self.new_files)
        receipt=prove(self.family,self.old,self.new,disclosure='Paired inference-only PEFT activation.',
            allow_candidate_timeout_forwarding=True,serving_predecessor_archive=previous,
            serving_predecessor_sha256=previous_sha,training_source_archive=self.old,training_source_sha256=self.old_sha)
        return previous,previous_sha,predecessor,receipt,new

    def test_paired_serving_activation_keeps_common_counts_and_genuine_profiles(self):
        _,previous_sha,predecessor,receipt,_=self.inference_archive_pair()
        self.assertEqual((receipt['byte_identical_common_file_count'],receipt['audited_common_source_delta_count']),(3,1))
        self.assertEqual(receipt['audited_serving_source_delta_count'],1)
        self.assertTrue(receipt['audited_serving_source_deltas'][0]['proof']['serving_only'])
        steps=receipt['training_serving_archive_chain']['steps']
        self.assertEqual([s['changed_protected_common_path_count'] for s in steps],[1,0,1])
        self.assertEqual(steps[1]['changed_members'][0]['path'],'serving/qwen38_service.py')
        self.receipt=receipt;newer=self.future('qwen38_tuned')
        older=deepcopy(newer);older['comparison_runtime_profile']['source_archive_sha256']=previous_sha
        older['runtime_bindings']['source_archive_sha256']=previous_sha;older['source_equivalence']=predecessor
        before=digest([older,newer]);result=comparable_runtime_profiles(older,newer)
        self.assertFalse(result['exact_profile_identity']);self.assertTrue(result['audited_serving_control_variant'])
        self.assertEqual(before,digest([older,newer]))
        for key in ('generation','limits','quantization'):
            with self.subTest(profile=key):
                changed=deepcopy(newer);changed['comparison_runtime_profile'][key]['unauthorized']=True
                with self.assertRaises(ValueError):comparable_runtime_profiles(older,changed)
        changed=deepcopy(newer);changed['source_equivalence'].pop('audited_serving_source_deltas')
        with self.assertRaises(ValueError):comparable_runtime_profiles(older,changed)
        with patch('evaluation.qwen38_quality_equivalence.REVIEWED_INFERENCE_SERVICE_SHA256',
            receipt['actual_serving_archive_files']['serving/qwen38_service.py']):
            changed=deepcopy(receipt);changed.pop('audited_serving_source_deltas');changed.pop('audited_serving_source_delta_count')
            changed['receipt_sha256']=digest({k:v for k,v in changed.items() if k!='receipt_sha256'})
            with self.assertRaises(ValueError):validate_equivalence(self.family,changed,self.new_sha)

    def test_serving_proof_rejects_other_bytes_tampering_or_wrong_predecessor(self):
        previous,previous_sha,_,receipt,new=self.inference_archive_pair()
        for mutated in (new.replace(b'inference_mode=True',b'inference_mode=False'),new+b'# other bytes\n'):
            archive(self.new,{**self.new_files,'serving/qwen38_service.py':mutated})
            with self.assertRaises(ValueError):prove(self.family,self.old,self.new,disclosure='fix',
                allow_candidate_timeout_forwarding=True,serving_predecessor_archive=previous,
                serving_predecessor_sha256=previous_sha)
        changed=deepcopy(receipt);changed['audited_serving_source_deltas'][0]['proof']['verified']=False
        changed['receipt_sha256']=digest({k:v for k,v in changed.items() if k!='receipt_sha256'})
        with self.assertRaises(ValueError):validate_equivalence(self.family,changed,self.new_sha)
        archive(self.new,{**self.new_files,'serving/extra_generation.py':b'not reviewed'})
        with self.assertRaises(ValueError):prove(self.family,self.old,self.new,disclosure='fix',
            allow_candidate_timeout_forwarding=True,serving_predecessor_archive=previous,
            serving_predecessor_sha256=previous_sha,training_source_archive=self.old,training_source_sha256=self.old_sha)


if __name__=="__main__":unittest.main()
