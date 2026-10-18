"""Build-time public metadata refresh; never imported by production inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def main() -> None:
    metadata_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    with urlopen(metadata_url, timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION or metadata["cardData"]["license"] != "apache-2.0":
        raise RuntimeError("upstream_identity_or_license_mismatch")
    files = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name != "README.md" and not name.endswith((".json", ".txt", ".jinja", ".safetensors")):
            continue
        if "lfs" in item and item["lfs"]:
            size, digest, kind = item["lfs"]["size"], item["lfs"]["sha256"], "weight"
        else:
            with urlopen(f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", timeout=60) as response:
                raw = response.read(64 * 1024 * 1024)
            size, digest, kind = len(raw), hashlib.sha256(raw).hexdigest(), "processor_or_config"
        files.append({"path": name, "size_bytes": size, "sha256": digest, "kind": kind})
    manifest = {
        "schema_version": "1.0.0", "model_id": MODEL_ID, "revision": REVISION,
        "processor_id": MODEL_ID, "processor_revision": REVISION,
        "license": "apache-2.0", "license_verified": True,
        "license_source_url": f"https://huggingface.co/{MODEL_ID}/blob/{REVISION}/README.md",
        "metadata_source_url": metadata_url, "metadata_verified_at": "2026-10-17",
        "commercial_use_permitted": True, "served_model_name": "Qwen3-VL-8B-Instruct",
        "selected_adapter": None, "adapter_sha256": None, "dtype": "bfloat16", "files": files,
        "serving": {"backend": "vllm", "version": "0.31.0", "precision": "bfloat16",
                    "max_model_len": 32768, "max_images": 16, "max_pixels_per_image": 1048576,
                    "trust_remote_code": False, "offline_required": True, "external_ai_allowed": False},
        "validation": {"metadata_verified": True, "local_weights_verified": False,
                       "local_inference_verified": False,
                       "baseline_status": "not_run_requires_local_verified_weights_and_gpu",
                       "note": "Upstream checksums/license verified through public metadata; no local inference claimed."},
        "sources": ["https://docs.vllm.ai/en/stable/features/structured_outputs/",
                    "https://docs.vllm.ai/en/stable/features/multimodal_inputs/",
                    "https://github.com/vllm-project/vllm/releases/tag/v0.31.0"],
    }
    target = Path("models/manifest.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"revision": REVISION, "license": "apache-2.0", "files": len(files),
                      "local_inference_verified": False}))


if __name__ == "__main__":
    main()
