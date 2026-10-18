"""Private local-only inference. Offline snapshot verification uses stdlib only."""

from importlib import import_module

__all__ = [
    "InferenceMetrics", "LocalModelConfig", "LocalModelProvider", "ModelCancelled",
    "ModelDeadlineExceeded", "ModelError", "ModelResponse", "ModelUnavailable",
    "InvalidModelOutput", "UnsafeMedia", "ManifestError", "load_manifest",
    "verify_local_snapshot",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = "manifest" if name in {"ManifestError", "load_manifest", "verify_local_snapshot"} else "adapter"
    return getattr(import_module(f"{__name__}.{module}"), name)
