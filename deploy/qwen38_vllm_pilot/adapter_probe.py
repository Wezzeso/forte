"""Fail-closed proof of every original logical pair and active GPU packed slice.

This module imports no ML package until root executes its worker RPC. Source
checkpoint tensors are immutable; only copies are cast to the declared BF16
variant. No tensor values, media, generated descriptions or raw errors return.
"""
import hashlib
import json
from pathlib import Path

IMAGE = 'baaaafe69f963b32f2fa698166fe9abaf85afb68b91974f59527c6bfd7c95f02'
WEIGHTS = 'dd9ba2303067d843b362e2881ffc456fc05c0b35231108a3b6270fcec4818ed2'
CONFIG = 'edb87b169ced09e6b84c9a2292c9e047f015a0c37b72096b3b64695de95cd012'
MAPPING = '6e9f8c821aab28be6476e46baeeba44e2ca10005b73587fb3933637a4118f359'


def need(value, code):
    if not value:
        raise RuntimeError(code)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def expected_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['expected_runtime_projection_candidate'], []).append(row)
    priority = {'q_proj': 0, 'k_proj': 1, 'v_proj': 2, 'gate_proj': 0, 'up_proj': 1,
                'in_proj_qkv': 0, 'in_proj_z': 1, 'in_proj_b': 0, 'in_proj_a': 1}
    for values in groups.values():
        values.sort(key=lambda r: priority.get(r['logical_hf_target'].rsplit('.', 1)[-1], 0))
    return groups


def physical_parts(row):
    """Original B ranges and actual kernel-buffer slice indices."""
    leaf = row['logical_hf_target'].rsplit('.', 1)[-1]
    if leaf == 'in_proj_qkv': return [(0, 0, 2048), (1, 2048, 4096), (2, 4096, 10240)]
    index = {'in_proj_z': 3, 'in_proj_b': 0, 'in_proj_a': 1,
             'q_proj': 0, 'k_proj': 1, 'v_proj': 2, 'gate_proj': 0, 'up_proj': 1}.get(leaf, 0)
    return [(index, 0, row['b_shape'][0])]


def compare_group(torch, wrapper, loaded, rows, tensors, slot):
    """Check one group; callable with small generated tensors in CPU tests."""
    need(wrapper.tp_size == 1, 'tensor_parallel_variant_not_supported')
    la = loaded.lora_a if isinstance(loaded.lora_a, list) else [loaded.lora_a]
    lb = loaded.lora_b if isinstance(loaded.lora_b, list) else [loaded.lora_b]
    scales = loaded.scaling if isinstance(loaded.scaling, list) else [loaded.scaling]
    need(len(la) == len(lb) == len(rows) and all(x == 1 for x in scales), 'loaded_group_shape_or_scale_mismatch')
    expected_slices = []
    seen = set()
    for i, row in enumerate(rows):
        original_a = tensors(row['serialized_a_key'])
        original_b = tensors(row['serialized_b_key'])
        need(original_a.dtype == original_b.dtype == torch.float32, 'serialized_adapter_dtype_mismatch')
        need(list(original_a.shape) == row['a_shape'] and list(original_b.shape) == row['b_shape'], 'serialized_adapter_shape_mismatch')
        a = original_a.to(dtype=torch.bfloat16, device='cpu')
        b = original_b.to(dtype=torch.bfloat16, device='cpu')
        b.mul_(2)  # Installed vLLM first casts checkpoint, then folds alpha/rank.
        need(la[i] is not None and lb[i] is not None
            and la[i].dtype == lb[i].dtype == torch.bfloat16
            and torch.equal(la[i].detach().cpu(), a) and torch.equal(lb[i].detach().cpu(), b), 'loaded_logical_tensor_mismatch')
        for j, start, end in physical_parts(row):
            need(j not in seen, 'duplicate_physical_slice')
            seen.add(j)
            need(j < len(wrapper.lora_a_stacked) and j < len(wrapper.lora_b_stacked), 'missing_physical_slice')
            actual_a = wrapper.lora_a_stacked[j][slot, 0]
            actual_b = wrapper.lora_b_stacked[j][slot, 0]
            need(actual_a.dtype == actual_b.dtype == torch.bfloat16, 'active_gpu_adapter_dtype_mismatch')
            need(tuple(actual_a.shape) == tuple(a.shape) and tuple(actual_b.shape) == tuple(b[start:end].shape), 'active_gpu_slice_shape_mismatch')
            need(torch.equal(actual_a.detach().cpu(), a)
                and torch.equal(actual_b.detach().cpu(), b[start:end]), 'active_gpu_packed_tensor_mismatch')
            expected_slices.append((j, end - start))
    expected_slices.sort()
    need(seen == set(range(wrapper.n_slices))
        and len(wrapper.lora_a_stacked) == len(wrapper.lora_b_stacked) == wrapper.n_slices
        and tuple(wrapper.output_slices) == tuple(size for _, size in expected_slices), 'physical_slice_coverage_mismatch')
    return len(rows), len(seen)


def worker_probe(worker, adapter_path, mapping_path, adapter_id=1):
    """Root-only future collective_rpc. Loads/activates, never generates."""
    import torch
    from safetensors import safe_open
    from vllm.lora.request import LoRARequest
    from serving.qwen38_adapter_contract import expected_targets
    from importlib.metadata import version
    need(version('vllm') == '0.31.0+cu129', 'vllm_version_mismatch')
    need(torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0), 'pilot_requires_a100_sm80')
    mapping = json.loads(Path(mapping_path).read_text())
    need(file_sha(mapping_path) == MAPPING, 'expected_mapping_file_hash_mismatch')
    rows = mapping['logical_targets']
    expected = expected_targets()
    need(len(rows) == 496 and {r['logical_hf_target'] for r in rows} == set(expected), 'logical_inventory_mismatch')
    for row in rows:
        incoming, outgoing = expected[row['logical_hf_target']]
        need(row['a_shape'] == [8, incoming] and row['b_shape'] == [outgoing, 8], 'expected_shape_mismatch')
    directory = Path(adapter_path)
    need(file_sha(directory/'adapter_model.safetensors') == WEIGHTS
        and file_sha(directory/'adapter_config.json') == CONFIG, 'original_adapter_hash_mismatch')
    req = LoRARequest('forte-qwen38-fit-bf16-pilot', adapter_id, str(directory))
    worker.add_lora(req)
    manager = worker.model_runner.lora_manager._adapter_manager
    need(manager.lora_config.max_loras == 1 and manager.lora_config.max_lora_rank == 8
        and manager.lora_config.lora_dtype == torch.bfloat16, 'lora_runtime_configuration_mismatch')
    manager.activate_adapter(adapter_id)
    loaded = manager.get_adapter(adapter_id)
    need(loaded is not None and loaded.rank == 8, 'loaded_adapter_missing')
    slot = manager.lora_index_to_id.index(adapter_id)
    groups = expected_groups(rows)
    need(len(groups) == 304 and set(loaded.loras) == set(groups), 'loaded_physical_inventory_mismatch')
    keys = {r[k] for r in rows for k in ('serialized_a_key', 'serialized_b_key')}
    logical = physical = 0
    wrapper_classes = {}
    with safe_open(directory/'adapter_model.safetensors', framework='pt', device='cpu') as source:
        need(set(source.keys()) == keys and len(keys) == 992, 'serialized_tensor_set_mismatch')
        for name, members in sorted(groups.items()):
            need(name in manager.modules, 'expected_runtime_wrapper_missing')
            wrapper = manager.modules[name]
            need(all(t.device.type == 'cuda' for t in (*wrapper.lora_a_stacked, *wrapper.lora_b_stacked)),
                'active_adapter_buffers_not_on_cuda')
            # The exact wrapper must be the module used by the loaded model.
            need(manager.model.get_submodule(name) is wrapper, 'runtime_forward_wrapper_identity_mismatch')
            count, slices = compare_group(torch, wrapper, loaded.loras[name], members, source.get_tensor, slot)
            logical += count
            physical += slices
            cls = type(wrapper).__name__
            wrapper_classes[cls] = wrapper_classes.get(cls, 0) + 1
    torch.cuda.synchronize()
    need(logical == 496 and adapter_id in manager._active_adapters, 'adapter_complete_activation_failed')
    # No non-target adapters may be loaded under a different path/id.
    need(set(manager.list_adapters()) == {adapter_id}, 'unexpected_adapter_registered')
    need(file_sha(directory/'adapter_model.safetensors') == WEIGHTS, 'adapter_input_changed')
    return {'status': 'passed', 'logical_targets_verified': logical, 'original_tensors_verified': logical * 2,
        'physical_wrappers_verified': len(groups), 'physical_output_slices_verified': physical,
        'wrapper_classes': wrapper_classes, 'adapter_id': adapter_id, 'active_slot': slot,
        'checkpoint_dtype': 'F32', 'execution_dtype': 'BF16', 'b_scale_folded': 2,
        'loaded_cpu_and_active_gpu_exact_equality': True, 'no_partial_adapter_allowed': True,
        'no_model_forward_by_probe': True, 'arithmetic_equivalent_to_hf_nf4': False}


def worker_runtime_state(worker):
    """Aggregate loaded configuration/peak memory only, no forward/model text."""
    import torch
    need(worker.vllm_config.quant_config is None
        and worker.model_config.quantization is None
        and worker.model_config.dtype == torch.bfloat16, 'bf16_unquantized_base_required')
    model = worker.model_runner.model
    need(type(model).__name__ == 'Qwen3_5ForConditionalGeneration', 'actual_model_class_mismatch')
    gdn = [m for m in model.modules() if type(m).__name__ == 'QwenGatedDeltaNetAttention']
    need(len(gdn) == 48, 'actual_gdn_layer_count_mismatch')
    from collections import Counter
    return {'model_class': type(model).__name__, 'base_dtype': 'BF16', 'quantization': None,
        'gdn_layers': len(gdn),
        'gdn_prefill_configurations': dict(Counter(m.gdn_prefill_backend for m in gdn)),
        'gdn_decode_configurations': dict(Counter(m.gdn_decode_kernel for m in gdn)),
        'gdn_fused_decode_configured': sum(m.enable_fused_gdn_decode for m in gdn),
        'kernel_execution_profiled': False,
        'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'active_adapter_ids': list(worker.model_runner.lora_manager._adapter_manager._active_adapters)}
