"""Real synthetic optimizer smoke, with no media/model-weight downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-after-release", action="store_true")
    args = parser.parse_args()
    try:
        from training.temporal.train_weak import configure_determinism, fit
        from training.runtime_check import verify_runtime
        from training.temporal.head import FrozenFeatureWeakMILHead
        from training.test_temporal import fixtures
        policy = configure_determinism()
        runtime = verify_runtime(Path("training/runtime-lock.json"), gpu=args.gpu_after_release)
        import torch
        torch.manual_seed(20261017)
        sequence, _, _ = fixtures()
        features = torch.randn(3, 2, 768)
        model = FrozenFeatureWeakMILHead(768, hidden_dim=16, event_count=3)
        metrics = fit(model, [(sequence, features, {}, 1, [], ((1, 2),))],
                      device="cuda:0" if args.gpu_after_release else "cpu", epochs=2, max_steps=2,
                      learning_rate=.001, weight_decay=.01, teacher_weight=.1, seed=20261017)
        print(json.dumps({"status": "passed", "synthetic_only": True, "gpu_used": args.gpu_after_release,
                          "optimizer_steps": metrics["optimizer_steps"], "runtime": runtime,
                          "deterministic_policy": policy, "private_media_or_labels_read": False}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "private_values_printed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
