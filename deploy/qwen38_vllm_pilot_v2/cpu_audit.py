"""Actual safe encoder/extension and unchanged public packed-grammar tests."""
from contextlib import redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


def main():
    # Repeat original actual installed grammar/keymap/packed-corruption proof.
    from deploy.qwen38_vllm_pilot.cpu_audit import main as original_audit
    captured=io.StringIO()
    with redirect_stdout(captured): original_audit()
    original=json.loads(captured.getvalue().strip().splitlines()[-1])
    import torch
    from vllm import envs,LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.v1.serial_utils import MsgpackEncoder
    from deploy.qwen38_vllm_pilot import adapter_probe
    from serving.qwen38_vllm_worker import NativeAuditWorkerExtension
    from .pilot import EXTENSION,EXTENSION_SHA
    assert not envs.VLLM_ALLOW_INSECURE_SERIALIZATION
    assert adapter_probe.file_sha('/ops/serving/qwen38_vllm_worker.py')==EXTENSION_SHA
    assert 'worker_extension_cls' in inspect.signature(EngineArgs).parameters
    engine_args=EngineArgs(model='/processor',worker_extension_cls=EXTENSION,
        enable_lora=True,max_lora_rank=8,max_loras=1,lora_dtype='bfloat16')
    assert engine_args.worker_extension_cls==EXTENSION
    encoder=MsgpackEncoder()
    args=('/candidate/adapter','/ops/all-496-logical-targets.expected.json',1)
    encoder.encode((0,1,'add_lora',(LoRARequest('forte-qwen38-fit-bf16-pilot',1,args[0]),)))
    for method,callargs in [('forte_native_adapter_probe',args),('forte_native_runtime_state',())]:
        inspect.signature(LLM.collective_rpc).bind(None,method,timeout=180,args=callargs)
        encoder.encode((0,2,'collective_rpc',(method,180,callargs,None)))
    rejected=False
    try:encoder.encode((0,2,'collective_rpc',(adapter_probe.worker_probe,180,args,None)))
    except TypeError:rejected=True
    assert rejected
    # Exercise the installed hook's dynamic mixin pattern without a GPU worker
    # instance. Delegated strict functions are replaced only in this test.
    class FakeBase:pass
    class FakeWorker(FakeBase):pass
    methods=['forte_native_adapter_probe','forte_native_runtime_state']
    assert not any(hasattr(FakeWorker,n) for n in methods)
    FakeWorker.__bases__=FakeWorker.__bases__+(NativeAuditWorkerExtension,)
    fake=FakeWorker()
    layers=['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]
    fake.model_config=SimpleNamespace(hf_config=SimpleNamespace(text_config=SimpleNamespace(layer_types=layers)))
    seen=[]
    def fake_probe(worker,*received):
        assert worker is fake and received==args
        seen.append('strict_adapter_delegate')
        return {'status':'passed','logical_targets_verified':496,'original_tensors_verified':992}
    def fake_state(worker):
        assert worker is fake
        seen.append('strict_state_delegate')
        return {'base_dtype':'BF16','quantization':None}
    with patch.object(adapter_probe,'worker_probe',fake_probe),patch.object(adapter_probe,'worker_runtime_state',fake_state),\
         patch.object(torch.cuda,'get_device_capability',return_value=(8,0)):
        delegated=fake.forte_native_adapter_probe(*args)
        state=fake.forte_native_runtime_state()
    assert seen==['strict_adapter_delegate','strict_state_delegate']
    assert delegated['logical_targets_verified']==496 and state['language_layer_count']==64
    assert state['linear_attention_layers']==48 and state['full_attention_layers']==16
    # Test new controller's durable failure before any model import/GPU call.
    environment=dict(os.environ);environment.pop('FORTE_ROOT_GPU_RELEASE_ID',None)
    result=subprocess.run([sys.executable,'-m','deploy.qwen38_vllm_pilot_v2.pilot','--mode','execute',
        '--source-root','/ops','--output','/tmp/vllm-v2-no-root-release',
        '--mapping','/ops/all-496-logical-targets.expected.json'],capture_output=True,env=environment,timeout=30)
    assert result.returncode==2
    failure=json.loads(Path('/tmp/vllm-v2-no-root-release/controller.failed.json').read_text())
    assert failure['phase']=='verify_root_release_and_runtime' and failure['error_type']=='RuntimeError'
    assert failure['generation_calls_entered']==failure['generation_calls_returned']==failure['completions_validated']==0
    assert failure['reserved_attempt_files']==failure['validated_call_receipt_files']==0
    assert failure['public_traceback'] and 'raw_error_suppressed' in failure and 'message' not in failure
    # A genuine public TypeError preserves function/line metadata but no text.
    diagnostic_code='''from pathlib import Path
from deploy.qwen38_vllm_pilot_v2.pilot import STATE,phase,durable_failure
directory=Path('/tmp/vllm-v2-safe-typeerror');directory.mkdir()
STATE.update(output=directory,output_owned=True)
phase('rpc_full_adapter_loaded_gpu_gate')
try:raise TypeError('SYNTHETIC_SECRET_MUST_NOT_BE_STORED')
except TypeError as error:durable_failure(error)
'''
    checked=subprocess.run([sys.executable,'-c',diagnostic_code],capture_output=True,timeout=10)
    assert checked.returncode==0
    raw=Path('/tmp/vllm-v2-safe-typeerror/controller.failed.json').read_text()
    typed=json.loads(raw)
    assert typed['error_type']=='TypeError' and typed['public_traceback']
    assert 'SYNTHETIC_SECRET' not in raw
    timer_dir=Path('/tmp/vllm-v2-deadline');timer_dir.mkdir()
    timer_code='''from pathlib import Path
import time
from deploy.qwen38_vllm_pilot_v2.pilot import STATE,phase,arm_timeout
directory=Path('/tmp/vllm-v2-deadline');STATE.update(output=directory,output_owned=True)
phase('engine_initialization');arm_timeout(directory,'cpu-only-check',.05);time.sleep(5)
'''
    timer=subprocess.run([sys.executable,'-c',timer_code],capture_output=True,timeout=10)
    assert timer.returncode==124
    timed=json.loads((timer_dir/'controller.failed.json').read_text())
    assert timed['phase']=='engine_initialization' and timed['error_type']=='TimeoutError'
    assert timed['generation_calls_entered']==0
    assert not torch.cuda.is_initialized() and not list(Path('/dev').glob('nvidia*'))
    original.update(schema_version='forte.qwen38.vllm-instrumentation-v2.cpu-audit.v1',
        worker_extension_class=EXTENSION,worker_extension_sha256=EXTENSION_SHA,
        safe_rpc_string_encoding_passed=True,raw_function_rpc_rejected_as_expected=True,
        actual_engine_args_signature_accepts_worker_extension=True,actual_engine_args_created_without_engine=True,
        dynamic_mixin_adapter_and_state_delegation_passed=True,
        fake_worker_extension_counts={'logical':496,'ab_tensors':992,'language_layers':64,'linear':48,'full':16},
        durable_failure_phase_type_trace_counts_and_redaction_passed=True,
        hard_timeout_durable_failure_exit124_passed=True,
        original_adapter_probe_unchanged_sha256=adapter_probe.file_sha('/ops/deploy/qwen38_vllm_pilot/adapter_probe.py'),
        original_request_unchanged_sha256=adapter_probe.file_sha('/ops/deploy/qwen38_vllm_pilot/request.py'),
        fake_dispatch_is_not_actual_gpu_adapter_proof=True)
    print(json.dumps(original))


if __name__=='__main__':main()
