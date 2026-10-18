"""Check a LOCAL processor with synthetic images/video; never access private media."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .data import SilverCompletionCollator, VideoCompletionCollator, video_metadata
from .guards import TrainingGateError, require


def check(processor_dir: Path, media_format: str, model_type: str = "qwen3_vl", silver: bool = False) -> dict:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    from PIL import Image
    from transformers import AutoProcessor

    require(processor_dir.is_dir(), "local_processor_missing")
    processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True, trust_remote_code=False)
    require(processor.__class__.__name__ == "Qwen3VLProcessor", "processor_architecture_mismatch")
    with tempfile.TemporaryDirectory(prefix="forte-synthetic-") as temporary:
        root = Path(temporary)
        frames = []
        for index, pts in enumerate((0.125, 0.625, 1.375, 2.125)):
            path = root / f"synthetic-{index}.jpg"
            Image.new("RGB", (96, 64), (index * 50, 80, 140)).save(path)
            frames.append({"path": path.name, "frame_id": f"synthetic-{index}", "pts_sec": pts})
        row = {"frames": frames, "prompt": "Return the requested synthetic fact as JSON.",
               "target": {"facts": [{"fact_id": "synthetic", "state": "unclear", "supporting_frame_ids": []}],
                          "candidates": []}}
        collator = VideoCompletionCollator
        if silver:
            collator = SilverCompletionCollator
            row.update(split="train", annotation_origin="local_model", human_reviewed=False,
                       target_scope="exact_anchor_frame", focal_frame_id=frames[0]["frame_id"],
                       fact_supervision_mask={"CASH_VISIBLE": True, "CASH_COUNT_VISIBLE": False},
                       target={"facts": [
                           {"fact_id": "CASH_VISIBLE", "state": "present", "supporting_frame_ids": [frames[0]["frame_id"]]},
                           {"fact_id": "CASH_COUNT_VISIBLE", "state": "unclear", "supporting_frame_ids": []}],
                           "candidates": []})
        tensors = collator(processor, root, max_length=8192, media_format=media_format, model_type=model_type)([row])
        labels = tensors["labels"][0]
        count = int((labels != -100).sum().item())
        require(0 < count < labels.numel(), "synthetic_mask_check_failed")
        decoded_labels = processor.tokenizer.decode(labels[labels != -100])
        if silver:
            require("CASH_VISIBLE" in decoded_labels and "present" in decoded_labels
                    and all(value not in decoded_labels for value in ("CASH_COUNT_VISIBLE", "unclear", "candidates")),
                    "synthetic_silver_mask_check_failed")
        grid_name = "image_grid_thw" if media_format == "image_sequence" else "video_grid_thw"
        clock = {"source_pts_sec": [frame["pts_sec"] for frame in frames],
                 "source_pts_text_ledger": media_format == "image_sequence" or model_type == "qwen3_5" or silver}
        if media_format == "video":
            metadata = video_metadata(frames)
            midpoints = processor._calculate_timestamps(metadata["frames_indices"].copy(), metadata["fps"],
                                                        processor.video_processor.temporal_patch_size)
            decoded_input = processor.tokenizer.decode(tensors["input_ids"][0])
            require(all(f"<{value:.1f} seconds>" in decoded_input for value in midpoints),
                    "synthetic_video_timestamp_bridge_failed")
            clock.update(processor_temporal_patch_midpoints_sec=midpoints, processor_timestamp_display_decimals=1,
                         processor_video_timestamps_are_individual_source_pts=False)
        return {"status": "passed", "fixture": "synthetic_only", "media_format": media_format,
                "model_type": model_type, "processor_class": type(processor).__name__,
                "template_options": {"enable_thinking": False, "preserve_thinking": False}
                                    if model_type == "qwen3_5" else {},
                "mask_policy": "supported_physical_silver_fact_spans" if silver else "human_reviewed_completion_only",
                "clock": clock,
                "input_tokens": labels.numel(), "completion_tokens": count,
                "visual_segments": tensors[grid_name].shape[0], "model_forward_executed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-dir", required=True, type=Path)
    parser.add_argument("--media-format", choices=("image_sequence", "video"), default="image_sequence")
    parser.add_argument("--model-type", choices=("qwen3_vl", "qwen3_5"), default="qwen3_vl")
    parser.add_argument("--silver", action="store_true", help="Verify separate synthetic partial-SILVER masks")
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.processor_dir, args.media_format, args.model_type, args.silver)))
        return 0
    except TrainingGateError as exc:
        print(json.dumps({"status": "blocked", "reason_code": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason_code": "local_processor_check_failed",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
