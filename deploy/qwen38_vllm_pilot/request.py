"""Exact public full19 generated fixture and canonical native JSON request."""
import hashlib
import json
from pathlib import Path
import time

ORIGINAL_SHA = 'a514e2238190cbc50967f74d984143cc467e2c2a32fc9ac0a01e3363abe96f1d'
DERIVED_SHA = '2bbeee8123a306837c0c713933d905709f8b544670702662e45843ac551b357e'
RULES_SHA = '3ba1c79c00f39a40addf4e1d2299c5bbade26c7949aeee10c382b9322fdde1dd'


def need(value, code):
    if not value:
        raise RuntimeError(code)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def prepare(source_root, output):
    """No HTTP/model calls. The only pixels are freshly drawn abstract shapes."""
    from PIL import Image, ImageDraw
    from forte.decision import build_prompt
    from forte.decision.structured import DecisionConfig
    from forte.model.adapter import LocalModelConfig, _prepare_media
    from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef
    from serving.qwen38_grammar import checked_schema, trusted_fact_order_schema

    source_root, output = Path(source_root), Path(output)
    rules = source_root / 'rules/catalog.json'
    need(hashlib.sha256(rules.read_bytes()).hexdigest() == RULES_SHA, 'public_rules_pin_mismatch')
    catalog = json.loads(rules.read_text())
    enabled = [r for r in catalog['rules'] if r['enabled'] is True]
    facts = sorted({f for r in enabled for f in catalog['bindings'][r['rule_id']]['required_fact_states']})
    events = sorted({catalog['bindings'][r['rule_id']]['event_type_id'] for r in enabled})
    need((len(enabled), len(facts), len(events)) == (4, 19, 3), 'public_catalog_geometry_mismatch')
    owned = output / 'full19_generated'
    owned.mkdir(parents=True, exist_ok=False)
    frames = []
    for index in range(16):
        path = owned / f'f{index:02d}.jpg'
        image = Image.new('RGB', (448, 448), (225, 225, 225))
        draw = ImageDraw.Draw(image)
        draw.rectangle((50 + index * 2, 100, 260, 300), fill=(180, 30, 90))
        draw.ellipse((270, 100, 380, 210), fill=(30, 100, 180))
        draw.text((15, 15), 'GENERATED STRUCTURE TEST ONLY', fill=(0, 0, 0))
        image.save(path, 'JPEG', quality=95)
        image.close()
        frames.append(FrameRef(frame_id=f'f{index:02d}', pts_sec=index * .5, path=path,
            width=448, height=448, job_id='full19_generated',
            jpeg_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    clip = ClipRequest(job_id='full19_generated', clip_id='full19_generated_clip', frame_refs=frames,
        fact_definitions=[FactDefinition(fact_id=fid, description=catalog['facts'][fid]['question_ru'] + ' '
            + catalog['facts'][fid]['definition_ru']) for fid in facts])
    decision = DecisionConfig(compact_frame_ids=True, concise_observations=True)
    bundle = build_prompt(clip, reviewed_event_ids=events,
        event_definitions={event: catalog['event_types'][event] for event in events}, config=decision)
    original = bundle.response_schema
    need(hashlib.sha256(checked_schema(original).encode()).hexdigest() == ORIGINAL_SHA, 'original_schema_pin_mismatch')
    derived, count = trusted_fact_order_schema(original)
    need(count == 19 and digest(derived) == DERIVED_SHA, 'derived_schema_pin_mismatch')
    # Use the frozen provider's authoritative cleaning/encoding/metadata path.
    blocks, dimensions, media_bytes = _prepare_media(clip, LocalModelConfig(media_root=output,
        max_frames=16, max_tokens=4096), time.monotonic() + 120, None)
    payload = {'model': 'forte-qwen38-base-bf16-pilot',
        'messages': [{'role': 'system', 'content': bundle.prompt}, {'role': 'user', 'content': blocks}],
        'structured_outputs': {'json': derived}, 'temperature': 0.0, 'seed': 20261017,
        'max_tokens': 4096, 'stream': False, 'n': 1,
        'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': False}}
    valid = {'facts': [{'fact_id': fid, 'state': 'unclear', 'supporting_frame_ids': []} for fid in facts],
             'candidates': []}
    return {'payload': payload, 'original_schema': original, 'derived_schema': derived,
        'clip': clip, 'decision_config': decision, 'event_ids': events, 'valid_synthetic': valid,
        'metadata': {'original_schema_sha256': ORIGINAL_SHA, 'derived_schema_sha256': DERIVED_SHA,
            'prompt_sha256': bundle.prompt_hash, 'messages_sha256': digest(payload['messages']),
            'encoded_media_bytes': media_bytes, 'image_dimensions': dimensions,
            'frames': 16, 'facts': 19, 'rules': 4, 'event_types': 3,
            'private_media_used': False, 'generated_structure_only': True}}


def native_grammar_cpu_audit(tokenizer, schema, valid):
    """Run actual installed vLLM backend with metadata only, no engine/model."""
    from types import SimpleNamespace
    from vllm.config import StructuredOutputsConfig
    from vllm.v1.structured_output.backend_types import StructuredOutputOptions
    from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
    from copy import deepcopy
    import time

    config = SimpleNamespace(structured_outputs_config=StructuredOutputsConfig(
        backend='xgrammar', disable_any_whitespace=True), speculative_config=None)
    backend = XgrammarBackend(vllm_config=config, tokenizer=tokenizer, vocab_size=248320)
    started = time.perf_counter()
    def grammar():
        return backend.compile_grammar(StructuredOutputOptions.JSON, compact(schema), {248046, 248044})
    first = grammar()
    elapsed = time.perf_counter() - started
    # vLLM's native JSON backend exposes no separators option. With whitespace
    # disabled, installed XGrammar uses json.dumps' default comma/colon spaces,
    # whereas the HF wrapper explicitly pins compact separators. Test the
    # actual native contract, retaining strict semantic postvalidation.
    def native_text(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    def accepts(value):
        probe = grammar()
        text = value if isinstance(value, str) else native_text(value)
        return bool(probe.matcher.accept_string(text) and probe.matcher.is_completed())
    tests = {'all_unclear_allowed': accepts(valid)}
    for state in ('present', 'absent', 'unclear'):
        changed = deepcopy(valid)
        for fact in changed['facts']:
            fact['state'], fact['supporting_frame_ids'] = state, ['f00']
        tests['all_' + state + '_allowed'] = accepts(changed)
    for name in ('duplicate', 'missing', 'unknown', 'extra', 'noncanonical'):
        changed = deepcopy(valid)
        if name == 'duplicate': changed['facts'][1] = changed['facts'][0]
        elif name == 'missing': changed['facts'].pop()
        elif name == 'unknown': changed['facts'][0]['fact_id'] = 'UNKNOWN_FACT'
        elif name == 'extra': changed['injected'] = True
        else: changed['facts'][0], changed['facts'][1] = changed['facts'][1], changed['facts'][0]
        tests[name + '_rejected'] = not accepts(changed)
    tests['prose_rejected'] = not accepts(native_text(valid) + ' prose')
    tests['duplicate_json_key_rejected'] = not accepts(native_text(valid).replace(
        '"candidates": []', '"candidates": [], "candidates": []'))
    tests['fresh_matcher'] = first.matcher is not grammar().matcher
    need(all(tests.values()), 'native_vllm_grammar_contract_failed:' + ','.join(k for k,v in tests.items() if not v))
    need(first.ctx.memory_size_bytes <= 64 * 1024 * 1024, 'native_vllm_grammar_memory_exceeded')
    need(tokenizer.convert_tokens_to_ids('<|im_end|>') == 248046
        and tokenizer.convert_tokens_to_ids('<|endoftext|>') == 248044, 'tokenizer_eos_mismatch')
    ids = tokenizer.encode(native_text(valid), add_special_tokens=False)
    token_probe = grammar()
    need(token_probe.accept_tokens('generated-cpu-audit', ids), 'native_vllm_token_sequence_failed')
    need(token_probe.matcher.is_completed(), 'native_vllm_token_completion_failed')
    stops = {}
    for stop in (248046, 248044):
        probe = grammar()
        need(not probe.matcher.accept_token(stop), 'native_vllm_early_eos_accepted')
        need(probe.accept_tokens('generated-cpu-audit', ids), 'native_vllm_token_sequence_failed')
        stops[str(stop)] = probe.matcher.accept_token(stop) and probe.matcher.is_terminated()
    need(all(stops.values()), 'native_vllm_completed_eos_failed')
    return {'backend': 'vllm.v1.structured_output.backend_xgrammar.XgrammarBackend',
        'compile_seconds': elapsed, 'compiled_memory_bytes': first.ctx.memory_size_bytes,
        'vocab_size': first.vocab_size, 'stop_tokens': stops, 'tests': tests,
        'schema_sha256': digest(schema), 'model_calls': 0,
        'native_json_separators': [', ', ': '],
        'hf_json_separators': [',', ':'],
        'note': 'Actual CPU backend/grammar/tokenizer proof; CUDA engine scheduling is not covered.'}
