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
from PIL import Image

from megatron.bridge.data.energon.euro_vl_task_encoder import EuroVLTaskEncoder
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample


def _make_processor(input_ids=None, encode_return=None, apply_chat_template_return="Hi there"):
    """Build a mock EuroVLProcessor sufficient to drive encode_sample."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    processor = MagicMock()
    processor.tokenizer = tokenizer
    processor.apply_chat_template.return_value = apply_chat_template_return
    processor.image_token_id = 99
    processor.video_token_id = 98

    if input_ids is None:
        input_ids = torch.tensor([[10, 11, 12, 13]])
    if encode_return is None:
        encode_return = [12, 13]
    tokenizer.encode.return_value = encode_return

    processor.return_value = {"input_ids": input_ids}
    return processor


class TestEuroVLTaskEncoderCook(unittest.TestCase):
    """``_cook`` must accept text-only crude samples (e.g. ``text/*/euroblocks``), which
    carry no ``jpg``/``img{i}``/``mp4`` part at all -- only a ``json`` conversation.
    """

    def _encoder(self):
        return EuroVLTaskEncoder(processor=_make_processor())

    def test_text_only_sample_does_not_raise(self):
        conversation = json.dumps(
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        )
        crude_sample = {
            "__key__": "text_sample_1",
            "__restore_key__": (),
            "__subflavor__": None,
            "__subflavors__": {},
            "json": conversation,
        }
        cooked = self._encoder()._cook(crude_sample)
        self.assertIsNone(cooked.imgs)
        self.assertIsNone(cooked.videos)
        self.assertEqual(cooked.conversation, conversation)

    def test_text_only_sample_with_bytes_conversation(self):
        conversation = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        crude_sample = {
            "__key__": "text_sample_2",
            "__restore_key__": (),
            "__subflavor__": None,
            "__subflavors__": {},
            "json": json.dumps(conversation).encode("utf-8"),
        }
        cooked = self._encoder()._cook(crude_sample)
        self.assertIsNone(cooked.imgs)
        self.assertIsNone(cooked.videos)
        self.assertEqual(json.loads(cooked.conversation), conversation)

    def test_sample_with_no_conversation_still_raises(self):
        """A sample missing the conversation entirely is still a real error (unrelated to
        the text-only fix), not a silently-accepted empty sample."""
        crude_sample = {"__key__": "broken_sample"}
        with self.assertRaises(KeyError):
            self._encoder()._cook(crude_sample)

    def test_text_only_sample_encodes_end_to_end(self):
        """A text-only crude sample should cook and encode without touching any vision path."""
        conversation = json.dumps(
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        )
        crude_sample = {
            "__key__": "text_sample_3",
            "__restore_key__": (),
            "__subflavor__": None,
            "__subflavors__": {},
            "json": conversation,
        }
        encoder = self._encoder()
        cooked = encoder._cook(crude_sample)
        encoded = encoder.encode_sample(cooked)
        self.assertEqual(encoded.visual_tensors, {})
        self.assertEqual(tuple(encoded.input_ids.shape), (4,))


class TestEuroVLTaskEncoderMaxNumImages(unittest.TestCase):
    """``_cook`` must reject samples with more images than ``max_num_images`` before
    decoding any of them (cheap pre-filter; see ``EuroVLTaskEncoder.__init__``)."""

    def test_too_many_images_raises_skip_sample(self):
        conversation = json.dumps(
            [{"role": "user", "content": "<image>" * 17}, {"role": "assistant", "content": "ok"}]
        )
        crude_sample = {
            "__key__": "too_many_images",
            "__restore_key__": (),
            "__subflavor__": None,
            "__subflavors__": {},
            "json": conversation,
        }
        for i in range(17):
            crude_sample[f"img{i}.jpg"] = Image.new("RGB", (4, 4))

        encoder = EuroVLTaskEncoder(processor=_make_processor(), max_num_images=16)
        with self.assertRaises(SkipSample):
            encoder._cook(crude_sample)

    def test_within_max_num_images_does_not_raise(self):
        conversation = json.dumps(
            [{"role": "user", "content": "<image>" * 16}, {"role": "assistant", "content": "ok"}]
        )
        crude_sample = {
            "__key__": "exactly_max_images",
            "__restore_key__": (),
            "__subflavor__": None,
            "__subflavors__": {},
            "json": conversation,
        }
        for i in range(16):
            crude_sample[f"img{i}.jpg"] = Image.new("RGB", (4, 4))

        encoder = EuroVLTaskEncoder(processor=_make_processor(), max_num_images=16)
        cooked = encoder._cook(crude_sample)
        self.assertEqual(len(cooked.imgs), 16)


class TestEuroVLTaskEncoderSkipOnTruncation(unittest.TestCase):
    """``encode_sample`` must skip (not truncate) ANY sample whose natural length exceeds
    seq_length -- truncating risks cutting off the answer itself. EuroVLTaskEncoder opts into
    this via skip_on_truncation=True (passed to the shared base encoder).
    """

    def _make_chatml_sample(self, conversation, imgs):
        return ChatMLSample(
            __key__="overflow_sample",
            __restore_key__=(),
            __subflavor__=None,
            __subflavors__={},
            imgs=imgs,
            videos=None,
            conversation=conversation,
        )

    def test_vision_heavy_overflow_raises_skip_sample(self):
        image_token_id = 99
        # seq_length=10; 15 image-placeholder tokens alone already exceed it.
        input_ids = torch.tensor([[1, 2] + [image_token_id] * 15 + [3, 4]])
        processor = _make_processor(input_ids=input_ids)
        encoder = EuroVLTaskEncoder(processor=processor, seq_length=10)

        conversation = json.dumps([{"role": "user", "content": "<image>"}, {"role": "assistant", "content": "ok"}])
        sample = self._make_chatml_sample(conversation, imgs=[torch.rand(3, 4, 4)])
        with self.assertRaises(SkipSample):
            encoder.encode_sample(sample)

    def test_text_heavy_overflow_also_raises_skip_sample(self):
        """Even when vision tokens are a small fraction of the overflow, any sample needing
        truncation is skipped now -- not just vision-dominated ones (truncating text risks
        cutting off the answer, e.g. a reasoning chain that concludes at the very end)."""
        image_token_id = 99
        # seq_length=10; only 3 image-placeholder tokens, but the full sequence (13 tokens)
        # still exceeds seq_length -- would previously truncate, now skips instead.
        input_ids = torch.tensor([[1, 2] + [image_token_id] * 3 + list(range(20, 30))])
        processor = _make_processor(input_ids=input_ids)
        encoder = EuroVLTaskEncoder(processor=processor, seq_length=10)

        conversation = json.dumps([{"role": "user", "content": "<image>"}, {"role": "assistant", "content": "ok"}])
        sample = self._make_chatml_sample(conversation, imgs=[torch.rand(3, 4, 4)])
        with self.assertRaises(SkipSample):
            encoder.encode_sample(sample)

    def test_sample_that_already_fits_is_not_skipped(self):
        image_token_id = 99
        # seq_length=10; total length (6) fits without truncation -- must not be skipped.
        input_ids = torch.tensor([[1, 2] + [image_token_id] * 3 + [4]])
        processor = _make_processor(input_ids=input_ids)
        encoder = EuroVLTaskEncoder(processor=processor, seq_length=10)

        conversation = json.dumps([{"role": "user", "content": "<image>"}, {"role": "assistant", "content": "ok"}])
        sample = self._make_chatml_sample(conversation, imgs=[torch.rand(3, 4, 4)])
        encoded = encoder.encode_sample(sample)
        self.assertEqual(tuple(encoded.input_ids.shape), (6,))


if __name__ == "__main__":
    unittest.main()
