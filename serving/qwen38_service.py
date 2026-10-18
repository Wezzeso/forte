"""Private Transformers/PEFT candidate service; imports GPU libraries lazily.

There is no base-only fallback. The verified adapter is actually reloaded into
the model, inspected, and used for generated-image warmup before readiness.
Requests accept only bounded JPEG data, and malformed/truncated generations
remain failures. This simple serialized backend has no latency claim.
"""

import asyncio
import base64
import binascii
from datetime import datetime, timezone
import hmac
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
import uuid

from serving.qwen38_identity import CandidateError, canonical_sha256, require
from serving.security import DataOnlyMiddleware


def parse_request(document: dict, config: dict) -> tuple[list[dict], list, dict, int]:
    """Validate the request and decode JPEGs locally, without model imports."""
    from PIL import Image
    import jsonschema

    require(isinstance(document, dict) and set(document).issubset({
        "model", "messages", "temperature", "max_tokens", "seed", "stream", "n",
        "structured_outputs", "chat_template_kwargs"}), "candidate_request_fields_invalid")
    generation, limits = config["generation"], config["limits"]
    require(document.get("model") == config["served_model_name"], "candidate_request_model_mismatch")
    require(document.get("temperature", 0) == 0 and document.get("seed", 20261017) == 20261017
            and document.get("stream", False) is False and document.get("n", 1) == 1,
            "candidate_request_generation_override")
    template_kwargs = document.get("chat_template_kwargs", {
        "enable_thinking": False, "preserve_thinking": False})
    require(template_kwargs == {"enable_thinking": False, "preserve_thinking": False},
            "candidate_request_template_override")
    maximum = document.get("max_tokens", generation["max_new_tokens"])
    require(type(maximum) is int and 0 < maximum <= generation["max_new_tokens"],
            "candidate_request_token_limit_invalid")
    structured = document.get("structured_outputs", {})
    require(isinstance(structured, dict) and set(structured) == {"json"}
            and isinstance(structured["json"], dict), "candidate_json_schema_required")
    schema = structured["json"]
    def local_schema(value):
        if isinstance(value, dict):
            require(all(not (key in {"$ref", "$dynamicRef"} and
                                 (not isinstance(item, str) or not item.startswith("#")))
                        for key, item in value.items()), "candidate_remote_schema_reference_forbidden")
            for item in value.values():
                local_schema(item)
        elif isinstance(value, list):
            for item in value:
                local_schema(item)
    local_schema(schema)
    jsonschema.Draft202012Validator.check_schema(schema)
    source = document.get("messages")
    require(isinstance(source, list) and 0 < len(source) <= 32, "candidate_messages_invalid")
    messages, images = [], []
    try:
        for row in source:
            require(isinstance(row, dict) and set(row) == {"role", "content"}
                    and row["role"] in {"system", "user", "assistant"}, "candidate_message_invalid")
            content = row["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            require(isinstance(content, list), "candidate_content_invalid")
            converted = []
            for block in content:
                require(isinstance(block, dict), "candidate_content_block_invalid")
                if block.get("type") == "text":
                    require(set(block) == {"type", "text"} and isinstance(block["text"], str),
                            "candidate_text_invalid")
                    converted.append(block)
                else:
                    require(set(block) == {"type", "image_url"} and block.get("type") == "image_url"
                            and set(block["image_url"]) == {"url"}, "candidate_jpeg_block_invalid")
                    address = block["image_url"]["url"]
                    prefix = "data:image/jpeg;base64,"
                    require(isinstance(address, str) and address.startswith(prefix), "candidate_jpeg_data_required")
                    raw = base64.b64decode(address[len(prefix):], validate=True)
                    require(0 < len(raw) <= 10 * 1024 * 1024, "candidate_jpeg_bytes_exceeded")
                    with Image.open(BytesIO(raw)) as decoded:
                        require(decoded.format == "JPEG"
                                and 0 < decoded.width * decoded.height <= limits["max_source_pixels_per_image"],
                                "candidate_jpeg_pixels_exceeded")
                        decoded.load()
                        image = decoded.convert("RGB")
                    images.append(image)
                    require(len(images) <= limits["max_images"], "candidate_image_count_exceeded")
                    converted.append({"type": "image"})
            messages.append({"role": row["role"], "content": converted})
        require(bool(images), "candidate_visual_input_required")
        return messages, images, schema, maximum
    except Exception:
        for image in images:
            image.close()
        raise


def load_processor_and_base(config: dict):
    """Shared exact processor/base loader for candidate and untuned diagnostics.

    This initializes CUDA. Callers must first verify base artifacts and obtain
    their own exclusive GPU release; this function does not imply selection.
    """
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, Qwen3VLProcessor

    require(torch.cuda.is_available(), "candidate_cuda_unavailable")
    require(tuple(torch.cuda.get_device_capability(0)) == (8, 0), "candidate_a100_sm80_required")
    torch.manual_seed(config["generation"]["seed"])
    torch.cuda.manual_seed_all(config["generation"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    processor = Qwen3VLProcessor.from_pretrained(
        config["model_path"], local_files_only=True, trust_remote_code=False)
    expected_template = (Path(config["model_path"]) / "chat_template.jinja").read_text(encoding="utf-8")
    require(processor.chat_template == expected_template, "candidate_effective_template_mismatch")
    for token, expected_id in (("<|image_pad|>", 248056), ("<|video_pad|>", 248057),
                               ("<|vision_start|>", 248053), ("<|vision_end|>", 248054)):
        require(processor.tokenizer.convert_tokens_to_ids(token) == expected_id,
                "candidate_effective_visual_token_mismatch")
    require(torch.cuda.is_bf16_supported() and torch.version.cuda == "12.9", "candidate_cuda_stack_mismatch")
    options = {"local_files_only": True, "trust_remote_code": False, "attn_implementation": "sdpa",
               "dtype": torch.bfloat16, "device_map": {"": config["device"]}}
    if config["quantization"]["format"] == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(config["model_path"], **options)
    layer_types = getattr(getattr(base.config, "text_config", None), "layer_types", [])
    require(base.config.model_type == "qwen3_5" and len(layer_types) == 64
            and layer_types.count("linear_attention") == 48 and layer_types.count("full_attention") == 16,
            "candidate_loaded_hybrid_architecture_mismatch")
    quantized_count = 0
    if config["quantization"]["format"] == "nf4":
        import bitsandbytes as bnb
        require(getattr(base, "is_loaded_in_4bit", False) is True, "candidate_nf4_not_loaded")
        quantized = [module for module in base.modules() if isinstance(module, bnb.nn.Linear4bit)]
        require(bool(quantized) and all(getattr(module.weight, "quant_type", None) == "nf4"
                and getattr(module.weight, "quant_state", None) is not None
                and module.compute_dtype == torch.bfloat16 for module in quantized),
                "candidate_nf4_runtime_state_mismatch")
        quantized_count = len(quantized)
    else:
        require(getattr(base, "is_loaded_in_4bit", False) is not True
                and getattr(base, "is_loaded_in_8bit", False) is not True,
                "candidate_unexpected_quantized_base")
    base._forte_candidate_load_receipt = {
        "quantization": config["quantization"], "nf4_module_count": quantized_count,
        "actual_nf4_state_verified": config["quantization"]["format"] == "nf4",
        "model_type": base.config.model_type, "attn_implementation": "sdpa",
        "cuda_version": torch.version.cuda, "gpu_compute_capability": [8, 0],
        "hybrid_kernel_path_verified": False, "model_gpu_loaded": True}
    return processor, base


class CandidateRuntime:
    def __init__(self, config: dict, identity: dict):
        self.config, self.identity = config, identity
        self.ready = False
        self.model = self.processor = None
        self.busy = threading.Lock()
        self.adapter_tensor_count = 0
        self.runtime_load = None
        self.grammar = None

    def load(self) -> None:
        import torch
        from peft import PeftModel, load_peft_weights

        self.processor, base = load_processor_and_base(self.config)
        self.runtime_load = base._forte_candidate_load_receipt
        self.model = PeftModel.from_pretrained(base, self.config["adapter_path"], adapter_name="candidate",
                                              is_trainable=False, local_files_only=True)
        self.model.set_adapter("candidate", inference_mode=True)
        self.model.eval()
        require(self.model.active_adapter == "candidate", "candidate_adapter_inactive")
        loaded = {name: tensor for name, tensor in self.model.named_parameters()
                  if ".lora_A.candidate." in name or ".lora_B.candidate." in name}
        expected = load_peft_weights(self.config["adapter_path"], device="cpu", local_files_only=True)
        require(bool(loaded) and len(loaded) == len(expected), "candidate_adapter_tensor_inventory_mismatch")
        # PEFT adds its adapter name to serialized module keys. Compare every
        # actual loaded tensor, not an alias or a successful constructor call.
        for name, tensor in loaded.items():
            key = name.replace(".candidate.", ".")
            require(key in expected and torch.equal(tensor.detach().cpu(), expected[key].to(tensor.dtype)),
                    "candidate_loaded_adapter_tensor_mismatch")
        require(all(not parameter.requires_grad for parameter in self.model.parameters()),
                "candidate_inference_parameters_trainable")
        self.adapter_tensor_count = len(loaded)
        from serving.qwen38_grammar import NativeGrammar
        self.grammar = NativeGrammar(self.processor.tokenizer, self.model, self.config["structured_decoding"])
        self.grammar.precompile_public_schema(max_images=self.config["limits"]["max_images"])

    def generate(self, document: dict, *, stop: threading.Event | None = None) -> dict:
        import torch
        import jsonschema
        from transformers import StoppingCriteria, StoppingCriteriaList

        require(self.busy.acquire(blocking=False), "candidate_busy")
        images = []
        started = time.monotonic()
        stopped = stop or threading.Event()
        deadline = started + self.config["limits"]["request_timeout_seconds"]

        class StopAtDeadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stopped.is_set() or time.monotonic() >= deadline

        try:
            messages, images, schema, maximum = parse_request(document, self.config)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False, preserve_thinking=False)
            maximum_pixels = self.config["limits"]["max_pixels_per_image"]
            inputs = self.processor(text=[text], images=images, padding=False, truncation=False,
                                    return_tensors="pt", images_kwargs={"size": {
                                        "longest_edge": maximum_pixels,
                                        "shortest_edge": min(maximum_pixels,
                                            self.processor.image_processor.size["shortest_edge"]),
                                    }})
            inputs.pop("token_type_ids", None)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            require(prompt_tokens + maximum <= self.config["limits"]["max_context_tokens"],
                    "candidate_context_would_truncate")
            require("pixel_values" in inputs and "image_grid_thw" in inputs
                    and int(inputs["image_grid_thw"].shape[0]) == len(images),
                    "candidate_processor_image_grid_mismatch")
            grid = inputs["image_grid_thw"]
            image_processor = self.processor.image_processor
            require(grid.ndim == 2 and grid.shape[1] == 3 and bool((grid > 0).all().item())
                    and bool((grid[:, 1:] % image_processor.merge_size == 0).all().item())
                    and bool((grid[:, 1] * grid[:, 2] * image_processor.patch_size ** 2
                              <= maximum_pixels).all().item())
                    and inputs["pixel_values"].ndim == 2
                    and inputs["pixel_values"].shape[0] == int(grid.prod(dim=-1).sum().item()),
                    "candidate_processor_grid_or_pixel_budget_invalid")
            inputs = inputs.to(self.config["device"])
            # Compiler/matcher failures never fall back to unconstrained text.
            require(self.grammar is not None, "candidate_grammar_not_loaded")
            grammar_processor = self.grammar.new_processor(schema)
            with torch.inference_mode():
                output = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=maximum,
                    use_cache=True, stopping_criteria=StoppingCriteriaList([StopAtDeadline()]),
                    logits_processor=[grammar_processor], eos_token_id=[248046, 248044], pad_token_id=248044)
            require(not stopped.is_set() and time.monotonic() < deadline, "candidate_generation_cancelled_or_timed_out")
            generated = output[0, prompt_tokens:]
            end = self.model.generation_config.eos_token_id
            end_ids = set(end if isinstance(end, list) else [end])
            require(len(generated) > 0 and (len(generated) < maximum or int(generated[-1]) in end_ids),
                    "candidate_generation_truncated")
            content = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
            # No JSON repair: trailing prose, nonfinite values and duplicate
            # fields must not be converted into a successful model verdict.
            from serving.qwen38_identity import _pairs, _finite_float
            require(len(content.encode()) <= 512 * 1024, "candidate_output_bytes_exceeded")
            parsed = json.loads(content, object_pairs_hook=_pairs,
                                parse_float=_finite_float,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            jsonschema.Draft202012Validator(schema).validate(parsed)
            completion_tokens = int(len(generated))
            return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
                    "created": int(time.time()), "model": self.config["served_model_name"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens}}
        except torch.cuda.OutOfMemoryError:
            self.ready = False
            raise CandidateError("candidate_gpu_oom_requires_recovery") from None
        finally:
            for image in images:
                image.close()
            self.busy.release()

    def readiness(self) -> dict:
        return {**self.identity, "ready": self.ready, "model_loaded": self.model is not None,
                "adapter_loaded_tensor_count": self.adapter_tensor_count,
                "runtime_load": self.runtime_load,
                "structured_decoding": self.grammar.receipt() if self.grammar is not None else None,
                "prewarm_completed": self.ready, "quality_or_latency_measured": False,
                "production_selected": False}


def create_app(runtime: CandidateRuntime):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    class CandidateDataOnlyMiddleware(DataOnlyMiddleware):
        async def __call__(self, scope, receive, send):
            # Candidate readiness has its own identity contract. Baseline's
            # fixed vLLM revision/readiness middleware remains unchanged.
            if scope.get("path") == "/readyz":
                return await self.app(scope, receive, send)
            return await super().__call__(scope, receive, send)

    app.add_middleware(CandidateDataOnlyMiddleware,
                       max_body_bytes=runtime.config["limits"]["max_body_bytes"])

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        key = os.environ.get("VLLM_API_KEY")
        if key and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + key):
            return JSONResponse({"error": "candidate_auth_required"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return JSONResponse({"loaded": runtime.model is not None}, status_code=200 if runtime.model is not None else 503)

    @app.get("/readyz")
    async def ready():
        return JSONResponse(runtime.readiness(), status_code=200 if runtime.ready else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": runtime.config["served_model_name"], "object": "model",
                 "owned_by": "forte-private", "parent": runtime.config["model_id"]}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        if not runtime.ready and (request.client is None or request.client.host not in {"127.0.0.1", "::1"}):
            return JSONResponse({"error": "candidate_not_warm"}, status_code=503)
        stop = threading.Event()
        async def watch_disconnect():
            while not stop.is_set():
                if await request.is_disconnected():
                    stop.set()
                    return
                await asyncio.sleep(.1)
        watcher = None
        try:
            document = await request.json()
            watcher = asyncio.create_task(watch_disconnect())
            result = await run_in_threadpool(runtime.generate, document, stop=stop)
            return result
        except CandidateError as error:
            code = str(error)
            return JSONResponse({"error": code}, status_code=429 if code == "candidate_busy" else 502)
        except Exception:
            return JSONResponse({"error": "candidate_inference_failed"}, status_code=502)
        finally:
            stop.set()
            if watcher:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)

    return app

