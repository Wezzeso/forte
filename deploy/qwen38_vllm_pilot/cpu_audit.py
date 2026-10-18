"""Public generated tensors/tokenizer only. Never instantiate model or engine."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from .request import prepare, native_grammar_cpu_audit
from .adapter_probe import compare_group, expected_groups


def main():
    assert not list(Path('/dev').glob('nvidia*'))
    import torch
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    import vllm.platforms as platforms
    from vllm.platforms.cpu import CpuPlatform
    platforms.current_platform = CpuPlatform()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.lora.utils import parse_fine_tuned_lora_name
    from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA

    case = prepare('/ops', '/tmp/vllm-generated-prepare')
    tokenizer = AutoTokenizer.from_pretrained('/processor', local_files_only=True, trust_remote_code=False)
    grammar = native_grammar_cpu_audit(tokenizer, case['derived_schema'], case['valid_synthetic'])
    sampling = SamplingParams(temperature=0, seed=20261017, max_tokens=4096, n=1,
        stop_token_ids=[248046,248044], structured_outputs=StructuredOutputsParams(json=case['derived_schema']))
    assert sampling.structured_outputs.json == case['derived_schema']
    rows = json.loads(Path('/ops/all-496-logical-targets.expected.json').read_text())['logical_targets']
    mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper()
    for row in rows:
        for kind in ('a','b'):
            name, is_a = parse_fine_tuned_lora_name(row['serialized_'+kind+'_key'], mapper)
            expected = row['logical_hf_target'].replace('model.language_model.','language_model.model.',1)
            assert name == expected and is_a == (kind == 'a')
    assert len(rows) == 496 and len(expected_groups(rows)) == 304

    # Exercise the installed vLLM GDN expansion, with generated small-input
    # tensors. Physical output widths remain the actual27B widths.
    tests = {}
    def check_group(leaves, widths):
        members, values = [], {}
        for index,(leaf,width) in enumerate(zip(leaves,widths)):
            row = {'logical_hf_target':'model.language_model.layers.0.linear_attn.'+leaf,
                'serialized_a_key':leaf+'.A','serialized_b_key':leaf+'.B',
                'a_shape':[2,3],'b_shape':[width,2]}
            members.append(row)
            values[leaf+'.A'] = torch.arange(6,dtype=torch.float32).reshape(2,3)/17 + index
            values[leaf+'.B'] = torch.arange(width*2,dtype=torch.float32).reshape(width,2)/max(width*3,1)
        la=[values[r['serialized_a_key']].to(torch.bfloat16) for r in members]
        lb=[values[r['serialized_b_key']].to(torch.bfloat16)*2 for r in members]
        physical_widths = [2048,2048,6144,6144] if leaves == ['in_proj_qkv','in_proj_z'] else widths
        if len(physical_widths) != len(lb):
            dummy=SimpleNamespace(n_slices=4,output_sizes=physical_widths)
            expanded_a,expanded_b=MergedColumnParallelLinearWithLoRA.expand_packed_lora(dummy,la,lb)
        else: expanded_a,expanded_b=la,lb
        wrapper=SimpleNamespace(tp_size=1,n_slices=len(physical_widths),output_slices=tuple(physical_widths),
            lora_a_stacked=tuple(a.clone()[None,None] for a in expanded_a),
            lora_b_stacked=tuple(b.clone()[None,None] for b in expanded_b))
        loaded=SimpleNamespace(lora_a=la if len(la)>1 else la[0],
            lora_b=lb if len(lb)>1 else lb[0],scaling=[1]*len(la) if len(la)>1 else 1)
        assert compare_group(torch,wrapper,loaded,members,values.__getitem__,0)[0] == len(leaves)
        wrapper.lora_b_stacked[-1][0,0,0,0] += 1
        try: compare_group(torch,wrapper,loaded,members,values.__getitem__,0)
        except RuntimeError: return True
        raise AssertionError('corrupt_packed_slice_not_rejected')
    tests['actual_gdn_expansion_and_corrupt_z_rejected'] = check_group(['in_proj_qkv','in_proj_z'],[10240,6144])
    tests['full_gated_qkv_and_corruption_rejected'] = check_group(['q_proj','k_proj','v_proj'],[12288,1024,1024])
    tests['ba_order_and_corruption_rejected'] = check_group(['in_proj_b','in_proj_a'],[48,48])
    tests['mlp_gate_up_and_corruption_rejected'] = check_group(['gate_proj','up_proj'],[17408,17408])
    tests['single_row_projection_and_corruption_rejected'] = check_group(['out_proj'],[5120])
    assert all(tests.values())
    environment = dict(os.environ)
    environment.pop('FORTE_ROOT_GPU_RELEASE_ID', None)
    denied = subprocess.run([sys.executable, '-m', 'deploy.qwen38_vllm_pilot.pilot', '--mode','execute',
        '--source-root','/ops','--output','/tmp/vllm-denied-root-grant',
        '--mapping','/ops/all-496-logical-targets.expected.json'],
        capture_output=True, env=environment, timeout=30)
    assert denied.returncode == 2 and json.loads(denied.stdout)['raw_error_suppressed'] is True
    timer_output = Path('/tmp/vllm-generated-hard-timer')
    timer_output.mkdir()
    timer_code = ('from deploy.qwen38_vllm_pilot.pilot import arm_timeout;from pathlib import Path;'
        'import time;arm_timeout(Path("/tmp/vllm-generated-hard-timer"),"cpu-guard",0.05);time.sleep(5)')
    timer = subprocess.run([sys.executable, '-c', timer_code], capture_output=True, timeout=10)
    assert timer.returncode == 124
    assert json.loads((timer_output/'timeout.cpu-guard.json').read_text())['reason_code'] == 'isolated_pilot_stage_timeout'
    assert not torch.cuda.is_initialized() and not list(Path('/dev').glob('nvidia*'))
    print(json.dumps({'schema_version':'forte.qwen38.vllm-instrumentation-cpu-audit.v1',
        'captured_at_utc':datetime.now(timezone.utc).isoformat(), 'status':'passed',
        'native_grammar':grammar,'generated_tensor_tests':tests,
        'actual_rename_mapper_keys_verified':992,'expected_logical_targets':496,'expected_physical_wrappers':304,
        'llm_class_import_only':True,'llm_collective_rpc_available':hasattr(LLM,'collective_rpc'),
        'llm_chat_parameters':list(inspect.signature(LLM.chat).parameters),
        'sampling_params_constructed_without_engine':True,
        'controller_cpu_guards': {'missing_root_release_denied_before_model_import':True,
            'isolated_controller_hard_timeout_exit124_and_receipt':True},
        'request_metadata':case['metadata'],'model_calls':0,'engine_instances':0,
        'cuda_initialized':False,'weights_mounted':False,'private_media_used':False,
        'platform_override':'CPU process only; CUDA engine defaults remain unverified'}))


if __name__ == '__main__':
    main()
