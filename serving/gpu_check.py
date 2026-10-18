"""Real CUDA preflight for the pinned image; no media or weight access."""

from datetime import datetime, timezone
import json


def main() -> int:
    record = {"checked_at": datetime.now(timezone.utc).isoformat(), "cuda_verified": False}
    try:
        # vLLM initializes its documented CUDA forward-compatibility library
        # handling before torch is imported, when enabled in the environment.
        import vllm
        import torch

        record.update(vllm_version=vllm.__version__, torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), device_count=torch.cuda.device_count())
        if not torch.cuda.is_available():
            print(json.dumps(record))
            return 1
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            # A real small operation plus synchronization, not metadata alone.
            left = torch.ones((32, 32), dtype=torch.bfloat16, device=f"cuda:{index}")
            output = left @ left
            torch.cuda.synchronize(index)
            passed = bool(torch.all(output == 32).item())
            devices.append({"index": index, "name": properties.name,
                            "total_memory_bytes": properties.total_memory,
                            "compute_capability": [properties.major, properties.minor],
                            "bf16_available": torch.cuda.is_bf16_supported(),
                            "matrix_operation_verified": passed})
            del left, output
        record.update(devices=devices, cuda_verified=all(d["matrix_operation_verified"] for d in devices))
        print(json.dumps(record))
        return 0 if record["cuda_verified"] else 1
    except Exception as error:
        record["error_type"] = type(error).__name__
        print(json.dumps(record))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
