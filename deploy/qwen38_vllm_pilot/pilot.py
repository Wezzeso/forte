"""Root-authorized, isolated generated-only offline vLLM BF16 pilot.

Default prepare mode creates generated fixture/request files and makes zero
model calls. Execute needs matching explicit root release environment, a new
restricted output directory, and the exact pinned image/source/model/adapter.
It is a separate BF16 variant; it never modifies or starts HF serving.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from .adapter_probe import IMAGE, file_sha, need, worker_probe, worker_runtime_state
from .request import prepare, compact, native_grammar_cpu_audit


def write(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def arm_timeout(output, stage, seconds):
    """Hard deadline only for this future isolated controller/container PID1."""
    def expired():
        try:
            write(output/('timeout.' + stage + '.json'), {'status':'failed',
                'reason_code':'isolated_pilot_stage_timeout','stage':stage,'limit_seconds':seconds})
        finally:
            os._exit(124)
    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    return timer


def metrics_only(metrics):
    if metrics is None:
        return {'available': False, 'prefill_decode_split_measured': False}
    names = ('arrival_time', 'first_scheduled_time', 'first_token_time', 'last_token_time', 'finished_time',
             'scheduler_time', 'model_forward_time', 'model_execute_time', 'time_in_queue')
    values = {n: getattr(metrics, n) for n in names if type(getattr(metrics, n, None)) in (int, float)}
    if all(n in values for n in ('arrival_time', 'first_token_time', 'finished_time')):
        values['ttft_seconds'] = values['first_token_time'] - values['arrival_time']
        values['after_first_token_seconds'] = values['finished_time'] - values['first_token_time']
    values['prefill_decode_split_measured'] = False
    values['note'] = 'TTFT includes scheduler/grammar/prefill overhead; after-first-token time is not isolated kernel time.'
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'execute'), default='prepare')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--model', default='/models/Qwen3.8-27B')
    p.add_argument('--adapter', default='/candidate/adapter')
    p.add_argument('--root-release-id')
    p.add_argument('--calls-per-arm', type=int, choices=(1, 2, 3), default=3)
    args = p.parse_args()
    need(not args.output.exists(), 'new_private_output_required')
    args.output.mkdir(parents=True, mode=0o700)
    sys.path[:0] = [str(args.source_root), str(args.source_root/'src')]
    case = prepare(args.source_root, args.output)
    write(args.output/'request.base.json', case['payload'])
    fit_payload = dict(case['payload'], model='forte-qwen38-fit-bf16-pilot')
    write(args.output/'request.fit.json', fit_payload)
    write(args.output/'schema.original.json', case['original_schema'])
    write(args.output/'schema.canonical19.json', case['derived_schema'])
    write(args.output/'request.metadata.json', case['metadata'])
    if args.mode == 'prepare':
        print(json.dumps({'status': 'prepared', 'maximum_generated_requests': 2*args.calls_per_arm,
            'model_calls': 0, 'gpu_operations': 0, **case['metadata']}))
        return 0
    need(args.root_release_id and re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}', args.root_release_id)
        and os.environ.get('FORTE_ROOT_GPU_RELEASE_ID') == args.root_release_id,
        'explicit_matching_root_gpu_release_required')
    need(os.environ.get('FORTE_RUNTIME_IMAGE_SHA256') == IMAGE, 'exact_runtime_image_binding_required')
    need(version('vllm') == '0.31.0+cu129' and version('torch') == '2.13.0+cu129'
        and version('xgrammar') == '0.2.7', 'runtime_package_pin_mismatch')
    source_manifest = args.source_root/'pilot-source.sha256.json'
    pins = json.loads(source_manifest.read_text())
    for relative, expected in pins.items():
        need(file_sha(args.source_root/relative) == expected, 'pilot_source_pin_mismatch')
    write(args.output/'execution.attempt.json', {'root_release_id': args.root_release_id,
        'runtime_image_sha256': IMAGE, 'source_manifest_sha256': file_sha(source_manifest),
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'max_generated_requests': 2*args.calls_per_arm,
        'variant': 'original_bf16_base__fit_f32_checkpoint_cast_bf16', 'equivalent_to_hf_nf4': False})
    startup_deadline = arm_timeout(args.output, 'startup', 1200)
    public_manifest = args.source_root/'base-public.manifest.json'
    need(file_sha(public_manifest) == '9be85d7b1a57be0f59e707b9a8872b964c506a515f7670958d2d5d60de197225',
        'public_base_manifest_pin_mismatch')
    base = json.loads(public_manifest.read_text())
    for entry in base['files']:
        relative = Path(entry['path'])
        need(not relative.is_absolute() and '..' not in relative.parts, 'public_model_relative_path_invalid')
        path = Path(args.model)/relative
        need(path.stat().st_size == entry['size_bytes'] and file_sha(path) == entry['sha256'],
            'public_model_file_pin_mismatch')
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.lora.request import LoRARequest
    from forte.decision import validate_decisions
    from forte.model.adapter import strict_json_object
    from jsonschema import Draft202012Validator
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    grammar = native_grammar_cpu_audit(tokenizer, case['derived_schema'], case['valid_synthetic'])
    write(args.output/'native-grammar.audit.json', grammar)
    started = time.perf_counter()
    llm = LLM(model=args.model, dtype='bfloat16', quantization=None, max_model_len=32768,
        tensor_parallel_size=1, max_num_seqs=1, max_num_batched_tokens=8192,
        gpu_memory_utilization=.85, limit_mm_per_prompt={'image': 16, 'video': 0},
        mm_processor_kwargs={'max_pixels':262144},
        enable_lora=True, max_lora_rank=8, max_loras=1, lora_dtype='bfloat16',
        generation_config='vllm', override_generation_config={'eos_token_id': [248046,248044], 'pad_token_id':248044},
        structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True},
        enable_prefix_caching=False, disable_log_stats=False, seed=20261017)
    write(args.output/'engine.initialization.json', {'elapsed_seconds': time.perf_counter()-started,
        'autoregressive_requests': 0, 'startup_profiling_forwards_not_counted_as_requests': True})
    fit = LoRARequest('forte-qwen38-fit-bf16-pilot', 1, args.adapter)
    llm.llm_engine.add_lora(fit)
    proof = llm.collective_rpc(worker_probe, timeout=180,
        args=(args.adapter, str(args.mapping), 1))
    need(len(proof) == 1 and proof[0]['status'] == 'passed', 'complete_worker_adapter_proof_required')
    write(args.output/'adapter.loaded-proof.json', proof)
    write(args.output/'runtime.before.json', llm.collective_rpc(worker_runtime_state, timeout=30))
    startup_deadline.cancel()
    validator = Draft202012Validator(case['original_schema'])
    calls = []
    for repeat in range(args.calls_per_arm):
        for arm, request in (('base', None), ('fit', fit)):
            need(len(calls) < 6, 'six_generated_request_cap_exceeded')
            index = len(calls)
            write(args.output/f'call-{index}.attempt.json', {'arm': arm, 'repeat': repeat,
                'request_sha256': file_sha(args.output/f'request.{arm}.json'),
                'max_tokens':4096,'seed':20261017,'retry':False})
            params = SamplingParams(temperature=0, seed=20261017, max_tokens=4096, n=1,
                stop_token_ids=[248046,248044], structured_outputs=StructuredOutputsParams(json=case['derived_schema']))
            request_deadline = arm_timeout(args.output, 'call-' + str(index), 240)
            started = time.perf_counter()
            output = llm.chat(case['payload']['messages'], sampling_params=params,
                lora_request=request, use_tqdm=False, chat_template_kwargs=case['payload']['chat_template_kwargs'],
                mm_processor_kwargs={'max_pixels':262144})
            elapsed = time.perf_counter()-started
            request_deadline.cancel()
            need(len(output)==1 and len(output[0].outputs)==1, 'one_generated_completion_required')
            result, completion = output[0], output[0].outputs[0]
            actual_id = getattr(getattr(result,'lora_request',None),'lora_int_id',None)
            need(actual_id == (1 if arm=='fit' else None), 'request_adapter_activation_identity_mismatch')
            data = strict_json_object(completion.text)
            validator.validate(data)
            batch = validate_decisions(data, case['clip'], reviewed_event_ids=case['event_ids'],
                config=case['decision_config'], finish_reason=completion.finish_reason)
            need(len(batch.facts)==19 and [f['fact_id'] for f in data['facts']] == sorted(f['fact_id'] for f in data['facts']),
                'nineteen_canonical_facts_required')
            record = {'arm':arm,'repeat':repeat,'elapsed_seconds':elapsed,'prompt_tokens':len(result.prompt_token_ids),
                'completion_tokens':len(completion.token_ids),'finish_reason':completion.finish_reason,
                'actual_lora_int_id':actual_id,'strict_original_schema_and_semantics':True,
                'output_sha256':hashlib.sha256(completion.text.encode()).hexdigest(),
                'metrics':metrics_only(result.metrics),'num_cached_tokens':result.num_cached_tokens}
            calls.append(record)
            write(args.output/f'call-{index}.receipt.json',record)
    write(args.output/'runtime.after.json',llm.collective_rpc(worker_runtime_state, timeout=30))
    write(args.output/'pilot.receipt.json',{'status':'passed','generated_requests':len(calls),'calls':calls,
        'private_quality_measured':False,'prefill_decode_kernel_time_measured':False,
        'arithmetic_equivalent_to_hf_nf4':False,'root_release_id':args.root_release_id})
    print(json.dumps({'status':'passed','generated_requests':len(calls),'private_media_used':False}), flush=True)
    # All receipt files are closed. Ending PID1 releases only this isolated
    # pilot's child workers; root remains responsible for restoring serving.
    os._exit(0)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Exception messages can contain generated strings; do not echo them.
        print(json.dumps({'status':'failed','error_type':type(error).__name__,
            'reason_code':'isolated_generated_pilot_failed','raw_error_suppressed':True}), flush=True)
        os._exit(2)
