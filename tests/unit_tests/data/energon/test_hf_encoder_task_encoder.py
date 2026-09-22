# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import unittest
from unittest.mock import MagicMock

import torch
from megatron.energon import SkipSample

from megatron.bridge.data.energon.hf_encoder_task_encoder import (
    HFEncoderTaskBatch,
    HFEncoderTaskSample,
    HFEncoderVLMTaskEncoder,
)
from megatron.bridge.data.energon.task_encoder_utils import IGNORE_INDEX, ChatMLSample
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


def _make_processor(
    pad_token_id=0,
    eos_token_id=1,
    input_ids=None,
    pixel_values=None,
    apply_chat_template_return="Hello assistant",
    encode_return=None,
):
    """Build a mock HF processor + tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = pad_token_id
    tokenizer.eos_token_id = eos_token_id
    tokenizer.apply_chat_template.return_value = apply_chat_template_return

    if encode_return is None:
        encode_return = [12, 13]
    tokenizer.encode.return_value = encode_return

    processor = MagicMock()
    processor.tokenizer = tokenizer

    if input_ids is None:
        input_ids = torch.tensor([[10, 11, 12, 13]])

    proc_output = {"input_ids": input_ids}
    if pixel_values is not None:
        proc_output["pixel_values"] = pixel_values
    processor.return_value = proc_output

    return processor


def _make_chatml_sample(conversation, imgs=None, videos=None, key="k1"):
    """Create a ChatMLSample with the correct base-class fields."""
    return ChatMLSample(
        __key__=key,
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__={},
        imgs=imgs,
        videos=videos,
        conversation=conversation,
    )


class TestHFEncoderTaskSample(unittest.TestCase):
    def test_fields(self):
        s = HFEncoderTaskSample(
            __key__="k1",
            __subflavors__={},
            input_ids=torch.tensor([1, 2, 3]),
            labels=torch.tensor([2, 3, -100]),
            loss_mask=torch.tensor([1.0, 1.0, 0.0]),
            visual_tensors={"pixel_values": torch.randn(1, 3, 4, 4)},
        )
        self.assertEqual(s.__key__, "k1")
        self.assertEqual(s.input_ids.shape, (3,))
        self.assertIn("pixel_values", s.visual_tensors)


class TestHFEncoderVLMTaskEncoderEncodeSample(unittest.TestCase):
    def test_text_only(self):
        processor = _make_processor(
            input_ids=torch.tensor([[10, 11, 12, 13]]),
            encode_return=[12, 13],
        )
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128)

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            ),
        )

        encoded = encoder.encode_sample(sample)
        self.assertIsInstance(encoded, HFEncoderTaskSample)
        self.assertEqual(encoded.input_ids.shape[0], 4)
        self.assertEqual(encoded.labels.shape[0], 4)
        self.assertEqual(encoded.loss_mask.shape[0], 4)
        self.assertEqual(len(encoded.visual_tensors), 0)

    def test_with_images(self):
        pv = torch.randn(1, 3, 224, 224)
        processor = _make_processor(
            input_ids=torch.tensor([[10, 11, 12, 13]]),
            pixel_values=pv,
            encode_return=[12, 13],
        )
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128, visual_keys=("pixel_values",))

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Describe <image>"},
                    {"role": "assistant", "content": "A photo"},
                ]
            ),
            imgs=[torch.rand(3, 4, 4)],
        )

        encoded = encoder.encode_sample(sample)
        self.assertIn("pixel_values", encoded.visual_tensors)
        self.assertEqual(encoded.visual_tensors["pixel_values"].shape, pv.shape)

    def test_truncation(self):
        long_ids = torch.tensor([list(range(200))])
        processor = _make_processor(input_ids=long_ids, encode_return=[150, 151])
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=50)

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "long prompt"},
                    {"role": "assistant", "content": "answer"},
                ]
            ),
        )
        encoded = encoder.encode_sample(sample)
        self.assertEqual(encoded.input_ids.shape[0], 50)

    def test_truncation_slices_pixel_values_by_patch_count_not_image_count(self):
        """Native-res ``pixel_values`` has one row per PATCH (sum of h_i*w_i across images), not
        one row per image like ``image_grid_thw`` -- truncation must slice it by patch count, not
        by the generic ``shape[0] == num_images`` check that correctly handles ``image_grid_thw``.

        3 images, patch grids (h,w) = (2,2)=4, (3,3)=9, (4,4)=16 patches -> pixel_values has
        4+9+16=29 rows, deliberately != num_images=3, mirroring the real MoonViT/Qwen3-VL-style
        processor output this bug was found against (molmo2_table, see the investigation this
        test codifies). ``seq_length`` is set to truncate exactly after the first two images'
        token blocks, dropping the third entirely.
        """
        image_token_id = 99
        # [prefix(3)] [img0 block(3)] [mid(2)] [img1 block(4)] [mid(2)] [img2 block(5)] [suffix(2)]
        ids = (
            [1, 2, 3] + [image_token_id] * 3 + [4, 5] + [image_token_id] * 4 + [6, 7] + [image_token_id] * 5 + [20, 21]
        )
        input_ids = torch.tensor([ids])  # length 21; img2 block spans positions [14, 19)

        pixel_values = torch.randn(29, 3, 14, 14)  # 4 + 9 + 16 patches, all 3 images
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 4, 4]])

        processor = MagicMock()
        processor.tokenizer = MagicMock(pad_token_id=0, eos_token_id=1)
        processor.tokenizer.apply_chat_template.return_value = "prompt"
        processor.tokenizer.encode.return_value = [20, 21]
        processor.image_token_id = image_token_id
        processor.return_value = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

        # seq_length=14 keeps positions [0,14) -- through the end of "mid", i.e. exactly the two
        # complete image blocks (ending at 6 and 12) -- and excludes img2's block (starts at 14).
        encoder = HFEncoderVLMTaskEncoder(
            processor=processor, seq_length=14, visual_keys=("pixel_values", "image_grid_thw")
        )
        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "<image><image><image> describe"},
                    {"role": "assistant", "content": "answer"},
                ]
            ),
            imgs=[torch.rand(3, 4, 4) for _ in range(3)],
        )

        encoded = encoder.encode_sample(sample)

        grid = encoded.visual_tensors["image_grid_thw"]
        pv = encoded.visual_tensors["pixel_values"]
        self.assertEqual(tuple(grid.shape), (2, 3), "image_grid_thw should keep the 2 complete images")
        expected_patches = int(grid[:, 1:].prod(dim=1).sum())  # 2*2 + 3*3 = 13
        self.assertEqual(expected_patches, 13)
        self.assertEqual(
            pv.shape[0],
            expected_patches,
            "pixel_values must be sliced to the surviving images' PATCH count, not num_images",
        )

    def test_truncates_by_default(self):
        """skip_on_truncation defaults to False -- other HF-encoder VLMs (Gemma3-VL,
        Ministral3, GLM-4.5V) keep today's truncate-and-warn behavior unchanged."""
        image_token_id = 99
        input_ids = torch.tensor([[1, 2] + [image_token_id] * 3 + [3, 4, 5, 6, 7]])
        processor = _make_processor(input_ids=input_ids)
        processor.image_token_id = image_token_id
        processor.video_token_id = None
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=6, visual_keys=("pixel_values",))

        sample = _make_chatml_sample(
            conversation=json.dumps([{"role": "user", "content": "<image>"}, {"role": "assistant", "content": "ok"}]),
            imgs=[torch.rand(3, 4, 4)],
        )
        encoded = encoder.encode_sample(sample)
        self.assertEqual(tuple(encoded.input_ids.shape), (6,))

    def test_skips_on_truncation_when_enabled(self):
        """skip_on_truncation=True raises SkipSample for ANY sample needing truncation,
        not just ones where vision tokens alone exceed seq_length."""
        image_token_id = 99
        input_ids = torch.tensor([[1, 2] + [image_token_id] * 3 + [3, 4, 5, 6, 7]])
        processor = _make_processor(input_ids=input_ids)
        processor.image_token_id = image_token_id
        processor.video_token_id = None
        encoder = HFEncoderVLMTaskEncoder(
            processor=processor, seq_length=6, visual_keys=("pixel_values",), skip_on_truncation=True
        )

        sample = _make_chatml_sample(
            conversation=json.dumps([{"role": "user", "content": "<image>"}, {"role": "assistant", "content": "ok"}]),
            imgs=[torch.rand(3, 4, 4)],
        )
        with self.assertRaises(SkipSample):
            encoder.encode_sample(sample)

    def test_no_skip_when_sample_already_fits(self):
        """skip_on_truncation=True must not raise for a sample that already fits seq_length
        (no truncation needed)."""
        input_ids = torch.tensor([[10, 11, 12, 13]])
        processor = _make_processor(input_ids=input_ids)
        processor.video_token_id = None
        encoder = HFEncoderVLMTaskEncoder(
            processor=processor, seq_length=128, visual_keys=("pixel_values",), skip_on_truncation=True
        )
        sample = _make_chatml_sample(
            conversation=json.dumps([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "ok"}]),
        )
        encoded = encoder.encode_sample(sample)
        self.assertEqual(tuple(encoded.input_ids.shape), (4,))

    # ------------------------------------------------------------------
    # min_answer_tokens_after_trim: trim the FINAL answer instead of skipping
    # ------------------------------------------------------------------
    # Layout used below (15 tokens): question [1, 2, 3] at 0..2, answer 20..29 at 3..12,
    # closing template [90, 91] at 13..14. Answer span = (3, 13).
    QUESTION, ANSWER, TEMPLATE = [1, 2, 3], list(range(20, 30)), [90, 91]

    def _trim_encoder(self, seq_length, min_kept, answer_tokens=None, input_ids=None):
        ids = input_ids if input_ids is not None else self.QUESTION + self.ANSWER + self.TEMPLATE
        processor = _make_processor(
            input_ids=torch.tensor([ids]),
            encode_return=self.ANSWER if answer_tokens is None else answer_tokens,
        )
        processor.video_token_id = None
        return HFEncoderVLMTaskEncoder(
            processor=processor,
            seq_length=seq_length,
            visual_keys=("pixel_values",),
            skip_on_truncation=True,
            min_answer_tokens_after_trim=min_kept,
        )

    @staticmethod
    def _qa_sample():
        return _make_chatml_sample(
            conversation=json.dumps([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
        )

    def test_trims_when_cut_falls_inside_final_answer(self):
        encoded = self._trim_encoder(seq_length=12, min_kept=5).encode_sample(self._qa_sample())

        self.assertEqual(encoded.input_ids.tolist(), self.QUESTION + self.ANSWER[:9])
        # The closing template is gone, so the model is never taught to stop mid-answer ...
        self.assertNotIn(90, encoded.input_ids.tolist())
        # ... and the last kept position predicts the TRUE next answer token (the first one cut).
        self.assertEqual(encoded.labels[-1].item(), self.ANSWER[9])
        self.assertEqual(encoded.loss_mask[-1].item(), 1.0)
        # The question is untouched and unsupervised.
        self.assertTrue((encoded.loss_mask[:2] == 0).all())

    def test_skips_when_too_few_answer_tokens_survive(self):
        # Cutting at 12 keeps 9 answer tokens, below the required 10.
        with self.assertRaises(SkipSample):
            self._trim_encoder(seq_length=12, min_kept=10).encode_sample(self._qa_sample())

    def test_skips_when_cut_reaches_into_the_question(self):
        # Cutting at 2 would remove question tokens: never trimmed, whatever the threshold.
        with self.assertRaises(SkipSample):
            self._trim_encoder(seq_length=2, min_kept=1).encode_sample(self._qa_sample())

    def test_skips_when_final_answer_not_located(self):
        # The answer's tokens do not occur in input_ids, so there is no span to trim within.
        with self.assertRaises(SkipSample):
            self._trim_encoder(seq_length=12, min_kept=1, answer_tokens=[77, 78]).encode_sample(self._qa_sample())

    def test_cut_in_closing_template_keeps_whole_answer(self):
        # Overflow of one token lands on the template, so the full answer survives.
        encoded = self._trim_encoder(seq_length=14, min_kept=10).encode_sample(self._qa_sample())
        self.assertEqual(encoded.input_ids.tolist()[3:13], self.ANSWER)

    def test_multi_turn_uses_the_final_answer(self):
        # [1] q1, [20, 21] answer 1, [2] q2, [30..39] final answer, [90] template.
        final = list(range(30, 40))
        ids = [1, 20, 21, 2] + final + [90]
        processor = _make_processor(input_ids=torch.tensor([ids]))
        processor.tokenizer.encode.side_effect = [[20, 21], final]
        processor.video_token_id = None
        conversation = json.dumps(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        )

        def encoder(min_kept):
            processor.tokenizer.encode.side_effect = [[20, 21], final]
            return HFEncoderVLMTaskEncoder(
                processor=processor,
                seq_length=12,
                visual_keys=("pixel_values",),
                skip_on_truncation=True,
                min_answer_tokens_after_trim=min_kept,
            )

        # Cutting at 12 keeps 8 tokens of the FINAL answer (positions 4..11).
        encoded = encoder(min_kept=8).encode_sample(_make_chatml_sample(conversation=conversation))
        self.assertEqual(encoded.input_ids.tolist(), [1, 20, 21, 2] + final[:8])
        with self.assertRaises(SkipSample):
            encoder(min_kept=9).encode_sample(_make_chatml_sample(conversation=conversation))

    def test_final_answer_span_is_last_supervised_run(self):
        import numpy as np

        span = HFEncoderVLMTaskEncoder._final_answer_span
        # Two answers: [2, 4) and [6, 9); the final one is the last run.
        self.assertEqual(span(np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=np.float32)), (6, 9))
        # Answer running to the very end of the sequence.
        self.assertEqual(span(np.array([0, 1, 1, 1], dtype=np.float32)), (1, 4))
        # Nothing supervised -> no span, so the trim rule skips.
        self.assertEqual(span(np.zeros(5, dtype=np.float32)), (-1, -1))

    def test_trim_threshold_requires_skip_on_truncation(self):
        with self.assertRaises(ValueError):
            HFEncoderVLMTaskEncoder(processor=_make_processor(), seq_length=8, min_answer_tokens_after_trim=4)

    def test_trim_threshold_must_be_positive(self):
        with self.assertRaises(ValueError):
            HFEncoderVLMTaskEncoder(
                processor=_make_processor(), seq_length=8, skip_on_truncation=True, min_answer_tokens_after_trim=0
            )

    def test_loss_mask_only_on_assistant(self):
        # Tokens: [10, 11, 12, 13, 14]
        # Assistant answer tokens: [13, 14]  (at positions 3,4)
        processor = _make_processor(
            input_ids=torch.tensor([[10, 11, 12, 13, 14]]),
            encode_return=[13, 14],
        )
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128)

        sample = _make_chatml_sample(
            conversation=json.dumps(
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A B"},
                ]
            ),
        )
        encoded = encoder.encode_sample(sample)
        self.assertTrue(encoded.loss_mask.sum() > 0, "loss_mask should have nonzero entries for assistant tokens")
        # The user input region (beginning) should have zero loss
        self.assertEqual(encoded.loss_mask[0].item(), 0.0)


class TestHFEncoderVLMTaskEncoderBatch(unittest.TestCase):
    def setUp(self):
        self.processor = _make_processor()
        self.encoder = HFEncoderVLMTaskEncoder(processor=self.processor, seq_length=128)

    def test_padding(self):
        s1 = HFEncoderTaskSample(
            __key__="k1",
            __subflavors__={},
            input_ids=torch.tensor([1, 2, 3, 4, 5]),
            labels=torch.tensor([2, 3, 4, 5, IGNORE_INDEX]),
            loss_mask=torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
            visual_tensors={},
        )
        s2 = HFEncoderTaskSample(
            __key__="k2",
            __subflavors__={},
            input_ids=torch.tensor([1, 2, 3]),
            labels=torch.tensor([2, 3, IGNORE_INDEX]),
            loss_mask=torch.tensor([0.0, 1.0, 0.0]),
            visual_tensors={},
        )

        batch = self.encoder.batch([s1, s2])
        self.assertIsInstance(batch, HFEncoderTaskBatch)
        self.assertEqual(batch.input_ids.shape, (2, 5))
        self.assertEqual(batch.labels.shape, (2, 5))
        self.assertEqual(batch.loss_mask.shape, (2, 5))
        self.assertIsNotNone(batch.attention_mask)
        self.assertEqual(batch.position_ids.shape, (2, 5))

    def test_visual_tensor_aggregation(self):
        pv1 = torch.randn(1, 3, 4, 4)
        pv2 = torch.randn(2, 3, 4, 4)
        s1 = HFEncoderTaskSample(
            __key__="k1",
            __subflavors__={},
            input_ids=torch.tensor([1, 2]),
            labels=torch.tensor([2, IGNORE_INDEX]),
            loss_mask=torch.tensor([1.0, 0.0]),
            visual_tensors={"pixel_values": pv1},
        )
        s2 = HFEncoderTaskSample(
            __key__="k2",
            __subflavors__={},
            input_ids=torch.tensor([3, 4]),
            labels=torch.tensor([4, IGNORE_INDEX]),
            loss_mask=torch.tensor([1.0, 0.0]),
            visual_tensors={"pixel_values": pv2},
        )
        batch = self.encoder.batch([s1, s2])
        self.assertIn("pixel_values", batch.visual_tensors)
        self.assertEqual(batch.visual_tensors["pixel_values"].shape[0], 3)  # 1 + 2


class TestHFEncoderVLMTaskEncoderEncodeBatch(unittest.TestCase):
    def test_encode_batch(self):
        processor = _make_processor()
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128)

        pv = torch.randn(2, 3, 4, 4)
        batch = HFEncoderTaskBatch(
            __keys__=["k1", "k2"],
            __subflavors__=[{}, {}],
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            labels=torch.tensor([[2, -100], [4, -100]]),
            loss_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            attention_mask=torch.randn(2, 1, 2, 2),
            position_ids=torch.tensor([[0, 1], [0, 1]]),
            visual_tensors={"pixel_values": pv},
        )

        result = encoder.encode_batch(batch)
        self.assertIsInstance(result, dict)
        self.assertIn("visual_inputs", result)
        self.assertIsInstance(result["visual_inputs"], GenericVisualInputs)
        self.assertNotIn("visual_tensors", result)
        self.assertNotIn("__subflavors__", result)
        self.assertIn("input_ids", result)

    def test_encode_batch_no_visuals(self):
        processor = _make_processor()
        encoder = HFEncoderVLMTaskEncoder(processor=processor, seq_length=128)

        batch = HFEncoderTaskBatch(
            __keys__=["k1"],
            __subflavors__=[{}],
            input_ids=torch.tensor([[1, 2]]),
            labels=torch.tensor([[2, -100]]),
            loss_mask=torch.tensor([[1.0, 0.0]]),
            attention_mask=torch.randn(1, 1, 2, 2),
            position_ids=torch.tensor([[0, 1]]),
            visual_tensors={},
        )

        result = encoder.encode_batch(batch)
        self.assertIn("visual_inputs", result)
        vi = result["visual_inputs"]
        self.assertIsInstance(vi, GenericVisualInputs)
        self.assertIsNone(vi.pixel_values)


class TestGenericVisualInputsCompat(unittest.TestCase):
    """Test GenericVisualInputs is compatible with vlm_step.py patterns."""

    def test_as_model_kwargs(self):
        vi = GenericVisualInputs(pixel_values=torch.randn(1, 3, 4, 4))
        kwargs = vi.as_model_kwargs()
        self.assertIn("pixel_values", kwargs)
        self.assertNotIn("image_grid_thw", kwargs)

    def test_normalized_for_model(self):
        vi = GenericVisualInputs(
            pixel_values=torch.randn(1, 3, 4, 4),
            image_sizes=torch.tensor([[4, 4]]),
        )
        result = vi.normalized_for_model()
        self.assertIn("pixel_values", result)
        self.assertIn("image_sizes", result)

    def test_dict_iteration(self):
        """vlm_step.py iterates __dict__ and calls .cuda() on non-None values."""
        vi = GenericVisualInputs(
            pixel_values=torch.randn(1, 3, 4, 4),
            image_grid_thw=None,
        )
        non_none = {k: v for k, v in vi.__dict__.items() if v is not None}
        self.assertIn("pixel_values", non_none)
        self.assertNotIn("image_grid_thw", non_none)


if __name__ == "__main__":
    unittest.main()
