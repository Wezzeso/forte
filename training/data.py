"""Qwen video collation with completion-only loss and explicit PTS metadata."""
from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

from .guards import local_file, require

def _active_positions(ids: list[int], mask: list[int] | None) -> list[int]:
    if mask is None:
        return list(range(len(ids)))
    require(len(mask) == len(ids) and all(type(value) is int and value in {0, 1} for value in mask),
            "attention_mask_invalid")
    active = [index for index, value in enumerate(mask) if value]
    require(bool(active) and active == list(range(active[0], active[-1] + 1)), "attention_mask_noncontiguous")
    return active


def completion_labels(input_ids: list[int], prompt_ids: list[int], visual_ids: set[int], *,
                      attention_mask: list[int] | None = None,
                      prompt_attention_mask: list[int] | None = None) -> list[int]:
    """Fail closed when template or visual expansion changes the prompt prefix."""
    positions = _active_positions(input_ids, attention_mask)
    prefix = [prompt_ids[index] for index in _active_positions(prompt_ids, prompt_attention_mask)]
    active_ids = [input_ids[index] for index in positions]
    require(len(active_ids) > len(prefix), "completion_empty")
    require(active_ids[:len(prefix)] == prefix, "processor_prefix_alignment_failed")
    labels = [-100] * len(input_ids)
    completion_positions = positions[len(prefix):]
    require(not ({input_ids[index] for index in completion_positions} & visual_ids),
            "completion_contains_visual_token")
    for index in completion_positions:
        labels[index] = input_ids[index]
    require(any(label != -100 for label in labels), "completion_all_masked")
    return labels


SILVER_PHYSICAL_FACT_IDS = frozenset({
    "CASH_VISIBLE", "CASH_ON_WORK_SURFACE", "CASH_HANDLER_PRESENT", "CASH_HANDLER_LEAVES_WORKSTATION",
    "CASH_HANDOVER_VISIBLE", "CASH_COUNT_VISIBLE", "CASH_PLACED_IN_CLOTHING_POCKET",
    "CASH_PLACED_IN_PERSONAL_CONTAINER",
})


def silver_target_text(row: dict[str, Any]) -> tuple[str, list[tuple[int, int]]]:
    """Retain unknowns in context; mark only supported, exact-frame physical facts for loss.

    This is a collation check, not a replacement for the separate source/provenance gate.
    No event, legal interpretation, empty-candidate assertion or boundary is a loss target.
    """
    require(row.get("split") == "train" and row.get("annotation_origin") == "local_model"
            and row.get("human_reviewed") is False, "silver_collator_origin_invalid")
    require(row.get("target_scope") == "exact_anchor_frame", "silver_target_scope_invalid")
    frames = {frame["frame_id"] for frame in row["frames"]}
    focal = row.get("focal_frame_id")
    require(focal in frames, "silver_focal_frame_missing")
    target, mask = row.get("target"), row.get("fact_supervision_mask")
    require(isinstance(target, dict) and set(target) == {"facts", "candidates"}
            and target["candidates"] == [] and isinstance(target["facts"], list), "silver_target_root_invalid")
    require(isinstance(mask, dict) and all(type(value) is bool for value in mask.values()),
            "silver_fact_mask_invalid")
    text, spans, seen = '{"facts":[', [], set()
    for fact in target["facts"]:
        require(isinstance(fact, dict) and set(fact) == {"fact_id", "state", "supporting_frame_ids"},
                "silver_fact_fields_invalid")
        fact_id, state, support = fact["fact_id"], fact["state"], fact["supporting_frame_ids"]
        require(fact_id in SILVER_PHYSICAL_FACT_IDS and fact_id not in seen, "silver_fact_id_invalid")
        require(state in {"present", "absent", "unclear"} and isinstance(support, list)
                and all(isinstance(item, str) for item in support) and set(support) <= frames,
                "silver_fact_support_invalid")
        require(fact_id in mask, "silver_fact_mask_mismatch")
        if mask[fact_id]:
            require(state in {"present", "absent"} and focal in support, "silver_supervised_fact_unsupported")
        if seen:
            text += ","
        start = len(text)
        text += json.dumps(fact, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if mask[fact_id]:
            spans.append((start, len(text)))
        seen.add(fact_id)
    require(set(mask) == seen, "silver_fact_mask_mismatch")
    require(bool(spans), "silver_completion_all_masked")
    return text + '],"candidates":[]}', spans


def masked_completion_labels(input_ids: list[int], prompt_ids: list[int], visual_ids: set[int], *,
                             tokenizer: Any, prompt_text: str, full_text: str, target_text: str,
                             target_spans: list[tuple[int, int]], attention_mask: list[int] | None = None,
                             prompt_attention_mask: list[int] | None = None,
                             return_span_positions: bool = False) -> list[int] | tuple[list[int], list[list[int]]]:
    """Join text offsets to the processor's expanded visual sequence by exact token suffix.

    Offsets from multimodal processors may refer to expanded image placeholders. Instead,
    tokenize the rendered template separately and prove its answer suffix equals the actual
    model tokens. Tokens crossing an enabled/disabled span boundary receive no loss.
    """
    labels = completion_labels(input_ids, prompt_ids, visual_ids, attention_mask=attention_mask,
                               prompt_attention_mask=prompt_attention_mask)
    require(full_text.startswith(prompt_text) and full_text[len(prompt_text):].startswith(target_text),
            "masked_target_template_alignment_failed")
    require(bool(getattr(tokenizer, "is_fast", False)), "masked_target_requires_fast_tokenizer")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    raw_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    raw_prefix = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_positions = [index for index, value in enumerate(labels) if value != -100]
    require(len(raw_ids) == len(offsets) and raw_ids[:len(raw_prefix)] == raw_prefix
            and raw_ids[len(raw_prefix):] == [input_ids[index] for index in completion_positions],
            "masked_target_token_alignment_failed")
    absolute_spans = [(len(prompt_text) + start, len(prompt_text) + end) for start, end in target_spans]
    require(bool(absolute_spans) and all(len(prompt_text) <= start < end <= len(prompt_text) + len(target_text)
                                       for start, end in absolute_spans)
            and all(left[1] <= right[0] for left, right in zip(absolute_spans, absolute_spans[1:])),
            "masked_target_spans_invalid")
    result, coverage = [-100] * len(input_ids), [0] * len(absolute_spans)
    span_positions: list[list[int]] = [[] for _ in absolute_spans]
    for position, (start, end) in zip(completion_positions, offsets[len(raw_prefix):]):
        require(type(start) is int and type(end) is int and 0 <= start <= end <= len(full_text),
                "masked_target_offsets_invalid")
        for index, (low, high) in enumerate(absolute_spans):
            if low <= start < end <= high:
                result[position] = input_ids[position]
                coverage[index] += 1
                span_positions[index].append(position)
                break
    require(all(coverage), "masked_target_span_unrepresented")
    return (result, span_positions) if return_span_positions else result


def _validate_frame_pts(frames: list[dict[str, Any]]) -> None:
    require(bool(frames) and all(type(frame.get("pts_sec")) in {int, float}
                                and math.isfinite(frame["pts_sec"]) and frame["pts_sec"] >= 0 for frame in frames),
            "frame_pts_invalid")


def video_metadata(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Use a millisecond timebase from source PTS, never index/nominal source FPS."""
    _validate_frame_pts(frames)
    ticks = [round(frame["pts_sec"] * 1000) for frame in frames]
    require(all(a < b for a, b in zip(ticks, ticks[1:])), "frame_pts_precision_collision")
    return {"fps": 1000.0, "frames_indices": ticks, "total_num_frames": ticks[-1] + 1,
            "duration": (ticks[-1] + 1) / 1000.0}


class VideoCompletionCollator:
    """One reviewed window per microbatch avoids ambiguous multimodal padding/packing."""

    def __init__(self, processor: Any, media_root: Path, max_length: int = 4096,
                 media_format: str = "image_sequence", max_pixels: int = 1048576, *, model_type: str = "qwen3_vl"):
        self.processor, self.media_root, self.max_length = processor, media_root, max_length
        require(media_format in {"image_sequence", "video"}, "media_format_invalid")
        require(model_type in {"qwen3_vl", "qwen3_5"}, "collator_model_type_invalid")
        require(type(max_length) is int and max_length > 0 and type(max_pixels) is int and max_pixels > 0,
                "collator_budget_invalid")
        self.model_type = model_type
        self.media_format, self.max_pixels = media_format, max_pixels

    def target_text(self, row: dict[str, Any]) -> tuple[str, list[tuple[int, int]] | None]:
        return json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"), allow_nan=False), None

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self._collate(rows, collect_fact_spans=False)[0]

    def _collate(self, rows: list[dict[str, Any]], *, collect_fact_spans: bool
                 ) -> tuple[dict[str, Any], list[list[int]] | None]:
        import numpy as np
        import torch
        from PIL import Image

        require(len(rows) == 1, "microbatch_must_be_one")
        row = rows[0]
        # Validate source time even for image sequences; timestamps become model input text.
        _validate_frame_pts(row["frames"])
        images, blocks = [], []
        for frame in row["frames"]:
            with Image.open(local_file(frame["path"], self.media_root)) as image:
                # Match A07's local JPEG sanitation and quality; remove source metadata.
                clean = BytesIO()
                image.convert("RGB").save(clean, format="JPEG", quality=95)
                clean.seek(0)
                with Image.open(clean) as decoded:
                    images.append(np.asarray(decoded.convert("RGB")).copy())
            blocks.extend([{"type": "text", "text": json.dumps({"frame_id": frame["frame_id"],
                           "timestamp_sec": frame["pts_sec"]}, separators=(",", ":"))}, {"type": "image"}])
        if self.media_format == "video":
            require(len({image.shape for image in images}) == 1, "video_frame_shapes_differ")
            blocks = [{"type": "video"}]
            if self.model_type == "qwen3_5" or isinstance(self, SilverCompletionCollator):
                # Qwen3VL's native video placeholders show rounded temporal-patch
                # midpoints (one decimal), not each original source frame's PTS.
                ledger = [{"frame_id": frame["frame_id"], "timestamp_sec": frame["pts_sec"]}
                          for frame in row["frames"]]
                blocks.insert(0, {"type": "text", "text": json.dumps(
                    {"source_frames_in_video_order": ledger}, separators=(",", ":"), allow_nan=False)})
        # E1 roles, frame-ID text, timestamps and pixels mirror A07's serving adapter.
        messages = [{"role": "system", "content": row["prompt"]}, {"role": "user", "content": blocks}]
        template_options = ({"enable_thinking": False, "preserve_thinking": False}
                            if self.model_type == "qwen3_5" else {})
        target_text, target_spans = self.target_text(row)
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                         **template_options)
        full_text = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": target_text}],
            tokenize=False, add_generation_prompt=False, **template_options,
        )
        # Process prompt and full conversation with identical pixels, timestamps, and token policy.
        # The second processor pass costs CPU work but makes completion masking auditable.
        options = {"return_tensors": "pt", "padding": False, "truncation": False}
        if self.media_format == "video":
            temporal_patch_size = self.processor.video_processor.temporal_patch_size
            padded_frames = math.ceil(len(images) / temporal_patch_size) * temporal_patch_size
            options.update(videos=[np.stack(images)], video_metadata=[video_metadata(row["frames"])],
                           do_sample_frames=False, cap_pixels_per_frame=False,
                           videos_kwargs={"size": {
                               "longest_edge": self.max_pixels * padded_frames,
                               "shortest_edge": min(self.processor.video_processor.size["shortest_edge"],
                                                    self.max_pixels * padded_frames),
                           }})
        else:
            options.update(images=images, images_kwargs={"size": {
                "longest_edge": self.max_pixels,
                "shortest_edge": min(self.max_pixels, self.processor.image_processor.size["shortest_edge"]),
            }})
        prompt = self.processor(text=[prompt_text], **options)
        full = self.processor(text=[full_text], **options)
        full.pop("token_type_ids", None)
        prompt_ids = prompt["input_ids"][0].tolist()
        input_ids = full["input_ids"][0].tolist()
        require(len(input_ids) <= self.max_length, "window_exceeds_token_budget")
        visual_ids = {self.processor.image_token_id, self.processor.video_token_id}
        visual_ids.update(self.processor.tokenizer.convert_tokens_to_ids(token)
                          for token in ("<|vision_start|>", "<|vision_end|>"))
        require(all(type(token) is int and token >= 0 for token in visual_ids), "visual_token_ids_invalid")
        masks = {"attention_mask": full["attention_mask"][0].tolist() if "attention_mask" in full else None,
                 "prompt_attention_mask": prompt["attention_mask"][0].tolist() if "attention_mask" in prompt else None}
        span_positions = None
        if target_spans is None:
            require(not collect_fact_spans, "fact_spans_require_partial_silver_targets")
            labels = completion_labels(input_ids, prompt_ids, visual_ids, **masks)
            if self.model_type == "qwen3_5":
                raw = self.processor.tokenizer(full_text, add_special_tokens=False)["input_ids"]
                raw_prefix = self.processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                require(raw[:len(raw_prefix)] == raw_prefix and raw[len(raw_prefix):]
                        == [input_ids[index] for index, value in enumerate(labels) if value != -100],
                        "completion_token_alignment_failed")
        else:
            aligned = masked_completion_labels(input_ids, prompt_ids, visual_ids, tokenizer=self.processor.tokenizer,
                prompt_text=prompt_text, full_text=full_text, target_text=target_text, target_spans=target_spans,
                return_span_positions=collect_fact_spans, **masks)
            if collect_fact_spans:
                labels, span_positions = aligned
            else:
                labels = aligned
        required_tensors = ({"pixel_values_videos", "video_grid_thw"} if self.media_format == "video"
                            else {"pixel_values", "image_grid_thw"})
        require(required_tensors <= full.keys(), "visual_processor_missing_tensors")
        require(required_tensors <= prompt.keys() and all(torch.equal(full[name], prompt[name])
                                                        for name in required_tensors),
                "visual_processor_prompt_full_mismatch")
        grid = full["video_grid_thw" if self.media_format == "video" else "image_grid_thw"]
        require(grid.ndim == 2 and grid.shape[1] == 3 and grid.dtype in {torch.int32, torch.int64}
                and bool((grid > 0).all().item()),
                "visual_processor_grid_invalid")
        require(grid.shape[0] == (1 if self.media_format == "video" else len(images)),
                "visual_processor_grid_count_mismatch")
        visual_processor = self.processor.video_processor if self.media_format == "video" else self.processor.image_processor
        merge_size = visual_processor.merge_size
        require(bool((grid[:, 1] * grid[:, 2] * visual_processor.patch_size ** 2 <= self.max_pixels).all().item()),
                "visual_processor_pixel_budget_exceeded")
        require(bool((grid[:, 1:] % merge_size == 0).all().item()), "visual_processor_grid_merge_invalid")
        patches = int(grid.prod(dim=-1).sum().item())
        pixels = full["pixel_values_videos" if self.media_format == "video" else "pixel_values"]
        require(pixels.ndim == 2 and pixels.shape[0] == patches, "visual_processor_pixel_grid_mismatch")
        modality_id = self.processor.video_token_id if self.media_format == "video" else self.processor.image_token_id
        require(input_ids.count(modality_id) == patches // merge_size ** 2, "visual_processor_token_grid_mismatch")
        full["labels"] = torch.tensor([labels], dtype=torch.long)
        # Processor-only metadata is not a model.forward argument.
        full.pop("video_metadata", None)
        return dict(full), span_positions


class SilverCompletionCollator(VideoCompletionCollator):
    """Separate machine-SILVER route; the human-reviewed E1 contract stays intact."""

    def target_text(self, row: dict[str, Any]) -> tuple[str, list[tuple[int, int]]]:
        return silver_target_text(row)

    def collate_with_fact_spans(self, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[int]]]:
        """Opt in to positions for the separate balanced fact objective.

        Positions index labels before the causal shift. They come from the same
        verified template/tokenizer/processor join as the default labels, in the
        serializer's fact order. Every supervised token belongs to exactly one
        enabled physical fact. Metadata stays outside model.forward arguments.
        """
        batch, positions = self._collate(rows, collect_fact_spans=True)
        row = rows[0]
        fact_ids = [fact["fact_id"] for fact in row["target"]["facts"]
                    if row["fact_supervision_mask"][fact["fact_id"]]]
        require(positions is not None and len(positions) == len(fact_ids), "silver_fact_token_span_count_mismatch")
        by_fact = dict(zip(fact_ids, positions))
        flattened = [position for group in by_fact.values() for position in group]
        labels = batch["labels"][0].tolist()
        require(len(flattened) == len(set(flattened)) and sorted(flattened)
                == [index for index, value in enumerate(labels) if value != -100],
                "silver_fact_token_span_coverage_mismatch")
        return batch, by_fact
