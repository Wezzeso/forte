"""Synthetic supervision checks: no model weights or private recordings are opened."""
from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from training.data import (completion_labels, masked_completion_labels, silver_target_text,
                           video_metadata, SilverCompletionCollator, VideoCompletionCollator)
from training.guards import TrainingGateError


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) + 1000 for char in text],
                "offset_mapping": [(index, index + 1) for index in range(len(text))]}


def row_fixture():
    return {"split": "train", "annotation_origin": "local_model", "human_reviewed": False,
            "target_scope": "exact_anchor_frame", "focal_frame_id": "f1",
            "frames": [{"frame_id": "f1", "pts_sec": .125}, {"frame_id": "f2", "pts_sec": 1.375}],
            "target": {"facts": [
                {"fact_id": "CASH_VISIBLE", "state": "present", "supporting_frame_ids": ["f1"]},
                {"fact_id": "CASH_COUNT_VISIBLE", "state": "unclear", "supporting_frame_ids": []},
                {"fact_id": "CASH_HANDOVER_VISIBLE", "state": "absent", "supporting_frame_ids": ["f1"]}],
                "candidates": []},
            "fact_supervision_mask": {"CASH_VISIBLE": True, "CASH_COUNT_VISIBLE": False,
                                      "CASH_HANDOVER_VISIBLE": True}}


class CompletionMaskTests(unittest.TestCase):
    def test_padding_is_masked_on_both_sides(self):
        self.assertEqual(completion_labels([0, 1, 9, 2, 3, 0], [1, 9, 0], {9},
                         attention_mask=[0, 1, 1, 1, 1, 0], prompt_attention_mask=[1, 1, 0]),
                         [-100, -100, -100, 2, 3, -100])

    def test_pad_token_equal_to_eos_does_not_mask_real_eos(self):
        self.assertEqual(completion_labels([1, 2, 0, 0], [1], {9}, attention_mask=[1, 1, 1, 0]),
                         [-100, 2, 0, -100])

    def test_prompt_visual_tokens_have_zero_loss(self):
        self.assertEqual(completion_labels([1, 8, 9, 9, 10, 2], [1, 8, 9, 9, 10], {8, 9, 10}),
                         [-100] * 5 + [2])

    def test_interleaved_attention_mask_rejected(self):
        with self.assertRaisesRegex(TrainingGateError, "attention_mask_noncontiguous"):
            completion_labels([1, 0, 2], [1], set(), attention_mask=[1, 0, 1])

    def test_truncated_completion_rejected(self):
        with self.assertRaisesRegex(TrainingGateError, "completion_empty"):
            completion_labels([1, 9], [1, 9], {9})

    def test_invalid_pts_rejected_before_processing(self):
        for frames in ([], [{"pts_sec": -1}], [{"pts_sec": float("nan")}], [{"pts_sec": True}]):
            with self.subTest(frames=frames), self.assertRaisesRegex(TrainingGateError, "frame_pts_invalid"):
                video_metadata(frames)

    def test_source_pts_precision_collision_is_rejected(self):
        with self.assertRaisesRegex(TrainingGateError, "frame_pts_precision_collision"):
            video_metadata([{"pts_sec": .10001}, {"pts_sec": .10002}])


class SilverMaskTests(unittest.TestCase):
    def test_unknowns_remain_unclear_with_no_loss_and_no_candidate_loss(self):
        row = row_fixture()
        target, spans = silver_target_text(row)
        self.assertEqual(json.loads(target), row["target"])
        # Expanded image tokens differ from raw tokenizer tokens in the prompt only.
        prefix, prompt_text = [7, 9, 9, 9, 8], "p"
        tail = CharacterTokenizer()(target + "!")["input_ids"]
        labels = masked_completion_labels(prefix + tail + [0], prefix, {9}, tokenizer=CharacterTokenizer(),
            prompt_text=prompt_text, full_text=prompt_text + target + "!", target_text=target,
            target_spans=spans, attention_mask=[1] * (len(prefix) + len(tail)) + [0])
        self.assertEqual(labels[:len(prefix)], [-100] * len(prefix))
        supervised = "".join(chr(value - 1000) for value in labels if value != -100)
        self.assertIn('"state":"present"', supervised)
        self.assertIn('"state":"absent"', supervised)
        self.assertNotIn("unclear", supervised)
        self.assertNotIn("CASH_COUNT_VISIBLE", supervised)
        self.assertNotIn("candidates", supervised)
        self.assertEqual(labels[-2:], [-100, -100])

    def test_all_unknowns_rejected_instead_of_zero_loss_fit(self):
        row = row_fixture()
        row["fact_supervision_mask"] = dict.fromkeys(row["fact_supervision_mask"], False)
        with self.assertRaisesRegex(TrainingGateError, "silver_completion_all_masked"):
            silver_target_text(row)

    def test_mask_cannot_promote_unclear_to_label(self):
        row = row_fixture()
        row["fact_supervision_mask"]["CASH_COUNT_VISIBLE"] = True
        with self.assertRaisesRegex(TrainingGateError, "silver_supervised_fact_unsupported"):
            silver_target_text(row)

    def test_support_must_include_exact_focal_frame(self):
        row = row_fixture()
        row["target"]["facts"][0]["supporting_frame_ids"] = ["f2"]
        with self.assertRaisesRegex(TrainingGateError, "silver_supervised_fact_unsupported"):
            silver_target_text(row)

    def test_physical_fact_allowlist_rejects_procedure_context(self):
        row = row_fixture()
        row["target"]["facts"][0]["fact_id"] = "PROCEDURE_CONTEXT_COMPLETE"
        with self.assertRaisesRegex(TrainingGateError, "silver_fact_id_invalid"):
            silver_target_text(row)

    def test_event_or_boundary_targets_cannot_enter_completion(self):
        row = row_fixture()
        row["target"]["candidates"] = [{"start_sec": 0, "end_sec": 1}]
        with self.assertRaisesRegex(TrainingGateError, "silver_target_root_invalid"):
            silver_target_text(row)

    def test_no_missing_or_extra_masks(self):
        for mutation in ("missing", "extra"):
            row = row_fixture()
            if mutation == "missing":
                row["fact_supervision_mask"].pop("CASH_VISIBLE")
            else:
                row["fact_supervision_mask"]["invented"] = False
            with self.subTest(mutation=mutation), self.assertRaisesRegex(TrainingGateError, "silver_fact_mask_mismatch"):
                silver_target_text(row)

    def test_validation_final_check_and_human_promotion_rejected(self):
        for key, value in (("split", "validation"), ("split", "final_check"), ("human_reviewed", True)):
            row = row_fixture()
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(TrainingGateError, "silver_collator_origin_invalid"):
                silver_target_text(row)

    def test_processor_suffix_mismatch_fails_closed(self):
        target, spans = silver_target_text(row_fixture())
        tail = CharacterTokenizer()(target + "!")["input_ids"]
        tail[-1] += 1
        with self.assertRaisesRegex(TrainingGateError, "masked_target_token_alignment_failed"):
            masked_completion_labels([7] + tail, [7], set(), tokenizer=CharacterTokenizer(),
                prompt_text="p", full_text="p" + target + "!", target_text=target, target_spans=spans)

    def test_template_thinking_mismatch_fails_closed(self):
        target, spans = silver_target_text(row_fixture())
        tail = CharacterTokenizer()(target + "!")["input_ids"]
        with self.assertRaisesRegex(TrainingGateError, "masked_target_template_alignment_failed"):
            masked_completion_labels([7] + tail, [7], set(), tokenizer=CharacterTokenizer(),
                prompt_text="p<think>", full_text="p<think></think>" + target + "!",
                target_text=target, target_spans=spans)

    def test_missing_answer_prefix_cannot_look_like_an_aligned_suffix(self):
        target, spans = silver_target_text(row_fixture())
        tail = CharacterTokenizer()(target + "!")["input_ids"][5:]
        with self.assertRaisesRegex(TrainingGateError, "masked_target_token_alignment_failed"):
            masked_completion_labels([7] + tail, [7], set(), tokenizer=CharacterTokenizer(),
                prompt_text="p", full_text="p" + target + "!", target_text=target, target_spans=spans)

    def test_optional_positions_use_same_verified_join_with_padding_and_visual_expansion(self):
        target, spans = silver_target_text(row_fixture())
        prefix = [0, 7, 9, 9, 8]
        tail = CharacterTokenizer()(target + "!")["input_ids"]
        labels, positions = masked_completion_labels(prefix + tail + [0], prefix, {9},
            tokenizer=CharacterTokenizer(), prompt_text="p", full_text="p" + target + "!", target_text=target,
            target_spans=spans, attention_mask=[0] + [1] * (len(prefix) - 1 + len(tail)) + [0],
            prompt_attention_mask=[0, 1, 1, 1, 1], return_span_positions=True)
        self.assertEqual(len(positions), 2)
        flattened = [index for group in positions for index in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(sorted(flattened), [index for index, value in enumerate(labels) if value != -100])
        self.assertTrue(all(index >= len(prefix) for index in flattened))
        for group, span in zip(positions, spans):
            self.assertEqual("".join(chr(labels[index] - 1000) for index in group), target[slice(*span)])

    def test_overlapping_or_reordered_fact_spans_fail_closed(self):
        target, spans = silver_target_text(row_fixture())
        tail = CharacterTokenizer()(target + "!")["input_ids"]
        for invalid in ([spans[1], spans[0]], [spans[0], spans[0]]):
            with self.subTest(spans=invalid), self.assertRaisesRegex(TrainingGateError, "masked_target_spans_invalid"):
                masked_completion_labels([7] + tail, [7], set(), tokenizer=CharacterTokenizer(),
                    prompt_text="p", full_text="p" + target + "!", target_text=target,
                    target_spans=invalid, return_span_positions=True)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "optional local processor runtime is unavailable")
class RealProcessorMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        from transformers import AutoProcessor
        from PIL import Image
        snapshot = Path(__file__).resolve().parents[1] / "configs/serving/qwen38/public-snapshot"
        if not (snapshot / "tokenizer.json").is_file():
            raise unittest.SkipTest("public processor-only snapshot is not staged")
        cls.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        cls.temporary = tempfile.TemporaryDirectory(prefix="forte-synthetic-mask-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.media_root = Path(cls.temporary.name)
        cls.row = row_fixture()
        cls.row["prompt"] = "Return the physical facts at focal frame f1 as JSON."
        for index, frame in enumerate(cls.row["frames"]):
            frame["path"] = f"generated-{index}.jpg"
            Image.new("RGB", (96, 64), (index * 50, 80, 140)).save(cls.media_root / frame["path"])

    def test_real_image_and_video_templates_mask_prompt_visuals_unknowns_and_boundaries(self):
        for media_format in ("image_sequence", "video"):
            with self.subTest(media_format=media_format):
                tensors = SilverCompletionCollator(self.processor, self.media_root, model_type="qwen3_5",
                                                    media_format=media_format)([self.row])
                labels = tensors["labels"][0]
                text = self.processor.tokenizer.decode(labels[labels != -100])
                self.assertIn('"state":"present"', text)
                self.assertIn('"state":"absent"', text)
                for masked in ("unclear", "CASH_COUNT_VISIBLE", "candidates", "think", "timestamp_sec"):
                    self.assertNotIn(masked, text)
                for visual in (self.processor.image_token_id, self.processor.video_token_id,
                               self.processor.vision_start_token_id, self.processor.vision_end_token_id):
                    self.assertFalse(bool((labels == visual).any()))

    def test_actual_default_thinking_template_cannot_be_used_accidentally(self):
        with self.assertRaisesRegex(TrainingGateError, "processor_prefix_alignment_failed"):
            VideoCompletionCollator(self.processor, self.media_root)([self.row])

    def test_no_truncation_to_meet_token_budget(self):
        with self.assertRaisesRegex(TrainingGateError, "window_exceeds_token_budget"):
            SilverCompletionCollator(self.processor, self.media_root, model_type="qwen3_5", max_length=1)([self.row])

    def test_small_pixel_budget_is_respected_below_processor_default_minimum(self):
        for media_format in ("image_sequence", "video"):
            with self.subTest(media_format=media_format):
                tensors = SilverCompletionCollator(self.processor, self.media_root, model_type="qwen3_5",
                    max_pixels=1024, media_format=media_format)([self.row])
                grid = tensors["image_grid_thw" if media_format == "image_sequence" else "video_grid_thw"]
                self.assertTrue(bool((grid[:, 1] * grid[:, 2] * 16 ** 2 <= 1024).all()))

    def test_opt_in_positions_preserve_default_batch_and_fact_mass_for_long_source_ids(self):
        import torch
        from training.silver_balance import fact_token_loss_weights
        counts = {}
        weights_by_fact = {"CASH_VISIBLE": .2, "CASH_HANDOVER_VISIBLE": .3, "CASH_COUNT_VISIBLE": 0.0}
        for media_format in ("image_sequence", "video"):
            for long_ids in (False, True):
                with self.subTest(media_format=media_format, long_ids=long_ids):
                    row = deepcopy(self.row)
                    if long_ids:
                        mapping = {frame["frame_id"]: str(index + 1) * 64 for index, frame in enumerate(row["frames"])}
                        row["focal_frame_id"] = mapping[row["focal_frame_id"]]
                        row["prompt"] = "Return physical facts at the first source frame as JSON."
                        for frame in row["frames"]:
                            frame["frame_id"] = mapping[frame["frame_id"]]
                        for fact in row["target"]["facts"]:
                            fact["supporting_frame_ids"] = [mapping[identity] for identity in fact["supporting_frame_ids"]]
                    collator = SilverCompletionCollator(self.processor, self.media_root,
                        model_type="qwen3_5", media_format=media_format)
                    default = collator([row])
                    batch, by_fact = collator.collate_with_fact_spans([row])
                    self.assertEqual(set(default), set(batch))
                    for key in batch:
                        self.assertTrue(torch.equal(default[key], batch[key]), key)
                    self.assertEqual(set(by_fact), {"CASH_VISIBLE", "CASH_HANDOVER_VISIBLE"})
                    labels = batch["labels"][0].tolist()
                    weights = fact_token_loss_weights(labels, by_fact, weights_by_fact)
                    for fact, positions in by_fact.items():
                        self.assertAlmostEqual(sum(weights[index] for index in positions), weights_by_fact[fact])
                        text = self.processor.tokenizer.decode([labels[index] for index in positions])
                        self.assertIn(fact, text)
                        self.assertNotIn("unclear", text)
                        self.assertNotIn("candidates", text)
                    self.assertAlmostEqual(sum(weights), .5)
                    self.assertTrue(all(weight == 0 for label, weight in zip(labels, weights) if label == -100))
                    counts[(media_format, long_ids)] = len(by_fact["CASH_VISIBLE"])
        for media_format in ("image_sequence", "video"):
            self.assertGreater(counts[(media_format, True)], counts[(media_format, False)])


if __name__ == "__main__":
    unittest.main()
