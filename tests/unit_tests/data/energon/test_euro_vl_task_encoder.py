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


# ---------------------------------------------------------------------------
# Message-tree samples
# ---------------------------------------------------------------------------

_IM_START, _IM_END, _NEWLINE, _VIDEO_TOKEN = 5, 6, 7, 98
_VIDEO_TOKENS_PER_VIDEO = 4


class _WordTokenizer:
    """Deterministic word-level tokenizer: specials fixed, words get ids from 100 on."""

    pad_token_id = 0
    eos_token_id = 1
    added_tokens_decoder = {}

    def __init__(self):
        self.vocab = {"<|im_start|>": _IM_START, "<|im_end|>": _IM_END, "\n": _NEWLINE, "<video>": _VIDEO_TOKEN}

    def _ids(self, text):
        import re

        ids = []
        for piece in re.split(r"(<\|im_start\|>|<\|im_end\|>|<video>|\n| )", text):
            if piece in ("", " "):
                continue
            if piece not in self.vocab:
                self.vocab[piece] = 100 + len(self.vocab)
            ids.append(self.vocab[piece])
        return ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._ids(text)}

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class _FakeVideoProcessor:
    """ChatML template + joint call that expands each ``<video>`` into a fixed block of tokens."""

    video_token_id = _VIDEO_TOKEN
    image_token_id = 99

    def __init__(self):
        self.tokenizer = _WordTokenizer()
        self.calls = 0

    def apply_chat_template(self, conversation, tokenize=False):
        out = ""
        for turn in conversation:
            content = turn["content"]
            if isinstance(content, list):
                content = " ".join("<video>" if c["type"] == "video" else c["text"] for c in content)
            out += f"<|im_start|> {turn['role']}\n{content}<|im_end|>\n"
        return out

    def __call__(self, text, videos=None, return_tensors="pt", **kwargs):
        self.calls += 1
        ids = []
        for tok in self.tokenizer._ids(text):
            ids.extend([_VIDEO_TOKEN] * _VIDEO_TOKENS_PER_VIDEO if tok == _VIDEO_TOKEN else [tok])
        n_videos = len(videos) if videos else 0
        return {
            "input_ids": torch.tensor([ids]),
            "pixel_values_videos": torch.ones(n_videos * _VIDEO_TOKENS_PER_VIDEO, 3),
            "video_grid_thw": torch.tensor([[1, 2, 2]] * n_videos),
        }


def _tree_sample(branches, shared=None, key="clip_g0", videos="default"):
    tree = {
        "message_tree": True,
        "shared": shared if shared is not None else [{"role": "user", "content": "<video>"}],
        "branches": branches,
        "meta": {},
    }
    return ChatMLSample(
        __key__=key,
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__={},
        imgs=None,
        videos=[[torch.rand(3, 4, 4), torch.rand(3, 4, 4)]] if videos == "default" else videos,
        conversation=json.dumps(tree),
    )


def _qa(question, answer):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


class TestEuroVLMessageTree(unittest.TestCase):
    """Message-tree samples: one shared media prefix, several independent QA branches."""

    BRANCHES = [_qa("what color", "red car"), _qa("how many", "three"), _qa("where is it", "a big road here")]

    def _encoder(self, seq_length=256, sqrt=False, root_subsegments=False):
        processor = _FakeVideoProcessor()
        encoder = EuroVLTaskEncoder(
            processor=processor,
            seq_length=seq_length,
            sqrt_loss_weighting=sqrt,
            root_subsegments=root_subsegments,
        )
        return encoder, processor

    def _supervised_text(self, processor, encoded, positions):
        inv = {v: k for k, v in processor.tokenizer.vocab.items()}
        # Supervision on label position i targets input token i + 1.
        return " ".join(inv[int(encoded.input_ids[i + 1])] for i in positions)

    def test_flat_json_list_is_not_a_tree(self):
        self.assertIsNone(EuroVLTaskEncoder._parse_message_tree(json.dumps(self.BRANCHES[0])))
        self.assertIsNone(EuroVLTaskEncoder._parse_message_tree(json.dumps({"branches": []})))
        self.assertIsNotNone(EuroVLTaskEncoder._parse_message_tree(json.dumps({"message_tree": True})))

    def test_video_encoded_once_and_tokens_appear_once(self):
        encoder, processor = self._encoder()
        encoded = encoder.encode_sample(_tree_sample(self.BRANCHES))
        self.assertEqual(processor.calls, 1)
        self.assertEqual(int((encoded.input_ids == _VIDEO_TOKEN).sum()), _VIDEO_TOKENS_PER_VIDEO)
        self.assertEqual(tuple(encoded.visual_tensors["pixel_values_videos"].shape), (_VIDEO_TOKENS_PER_VIDEO, 3))

    def test_subsegment_ids_mark_prefix_and_branches(self):
        encoder, _ = self._encoder()
        encoded = encoder.encode_sample(_tree_sample(self.BRANCHES))
        ids = encoded.subsegment_ids
        self.assertEqual(ids.dtype, torch.int32)
        self.assertEqual(ids.shape, encoded.input_ids.shape)
        im_starts = (encoded.input_ids == _IM_START).nonzero(as_tuple=True)[0].tolist()
        # Shared turn (turn 0) is attend-all, including the video; branch b starts at turn 1 + 2b.
        self.assertTrue(bool((ids[: im_starts[1]] == 10000).all()))
        for b in range(3):
            end = im_starts[3 + 2 * b] if b < 2 else len(ids)
            self.assertTrue(bool((ids[im_starts[1 + 2 * b] : end] == b).all()), f"branch {b}")

    def test_shared_user_turn_is_not_relabelled_system(self):
        """The odd-length turn list must not go through cook_chatml_sample's system heuristic."""
        encoder, processor = self._encoder()
        encoder.encode_sample(_tree_sample(self.BRANCHES))
        self.assertNotIn("system", processor.tokenizer.vocab)

    def test_loss_only_on_answers(self):
        encoder, processor = self._encoder()
        encoded = encoder.encode_sample(_tree_sample(self.BRANCHES))
        positions = (encoded.loss_mask > 0).nonzero(as_tuple=True)[0].tolist()
        self.assertEqual(self._supervised_text(processor, encoded, positions), "red car three a big road here")
        self.assertTrue(bool((encoded.labels[encoded.loss_mask == 0] == -100).all()))
        # Default: no /sqrt(B) -- a tree weighs exactly what the same branches would weigh flat.
        self.assertTrue(bool((encoded.loss_mask[encoded.loss_mask > 0] == 1.0).all()))

    def test_sqrt_weighting_is_per_branch_and_flat_equivalent_by_default(self):
        """Each branch gets 1/sqrt(N_b), exactly as it would as a standalone flat sample."""
        encoder, _ = self._encoder(sqrt=True)
        encoded = encoder.encode_sample(_tree_sample(self.BRANCHES))
        for b, n_tokens in enumerate((2, 1, 4)):
            weights = encoded.loss_mask[(encoded.subsegment_ids == b) & (encoded.loss_mask > 0)]
            self.assertEqual(len(weights), n_tokens)
            torch.testing.assert_close(weights, torch.full((n_tokens,), 1 / n_tokens**0.5))

    def test_root_subsegments_divides_by_sqrt_num_branches(self):
        encoder, _ = self._encoder(sqrt=True, root_subsegments=True)
        encoded = encoder.encode_sample(_tree_sample(self.BRANCHES))
        for b, n_tokens in enumerate((2, 1, 4)):
            weights = encoded.loss_mask[(encoded.subsegment_ids == b) & (encoded.loss_mask > 0)]
            torch.testing.assert_close(weights, torch.full((n_tokens,), 1 / (n_tokens**0.5 * 3**0.5)))

    def test_over_length_keeps_whole_branches(self):
        encoder, _ = self._encoder()
        full = encoder.encode_sample(_tree_sample(self.BRANCHES))
        im_starts = (full.input_ids == _IM_START).nonzero(as_tuple=True)[0].tolist()
        branch2_start = im_starts[5]
        encoder, _ = self._encoder(seq_length=branch2_start + 3)  # third branch does not fit
        encoded = encoder.encode_sample(_tree_sample(self.BRANCHES))
        self.assertEqual(len(encoded.input_ids), branch2_start)
        torch.testing.assert_close(encoded.input_ids, full.input_ids[:branch2_start])
        self.assertEqual(set(encoded.subsegment_ids.tolist()), {10000, 0, 1})
        self.assertEqual(int(encoded.labels[-1]), -100)
        self.assertEqual(float(encoded.loss_mask[-1]), 0.0)
        self.assertTrue(bool((encoded.loss_mask[encoded.loss_mask > 0] == 1.0).all()))

    def test_root_subsegments_counts_kept_branches_only(self):
        encoder, _ = self._encoder(root_subsegments=True)
        full = encoder.encode_sample(_tree_sample(self.BRANCHES))
        im_starts = (full.input_ids == _IM_START).nonzero(as_tuple=True)[0].tolist()
        encoder, _ = self._encoder(seq_length=im_starts[5] + 3, root_subsegments=True)
        encoded = encoder.encode_sample(_tree_sample(self.BRANCHES))
        self.assertEqual(set(encoded.subsegment_ids.tolist()), {10000, 0, 1})
        self.assertTrue(bool((encoded.loss_mask[encoded.loss_mask > 0] == 1 / 2**0.5).all()))

    def test_skip_when_no_branch_fits(self):
        encoder, _ = self._encoder(seq_length=8)
        with self.assertRaises(SkipSample):
            encoder.encode_sample(_tree_sample(self.BRANCHES))

    def test_media_placeholder_in_branch_raises(self):
        encoder, _ = self._encoder()
        with self.assertRaises(ValueError):
            encoder.encode_sample(_tree_sample([_qa("<video> what", "red")]))

    def test_turn_count_mismatch_raises(self):
        encoder, _ = self._encoder()
        # An extra <|im_start|> inside a question breaks the one-per-turn assumption.
        with self.assertRaisesRegex(ValueError, "im_start"):
            encoder.encode_sample(_tree_sample([_qa("what <|im_start|> color", "red")]))

    def test_branch_without_supervised_tokens_raises(self):
        encoder, _ = self._encoder()
        with self.assertRaisesRegex(ValueError, "branch 0"):
            encoder.encode_sample(_tree_sample([_qa("what", "")]))

    def test_packing_carries_subsegment_ids(self):
        encoder, _ = self._encoder(seq_length=256)
        tree = encoder.encode_sample(_tree_sample(self.BRANCHES))
        flat = encoder.encode_sample(
            ChatMLSample(
                __key__="flat",
                __restore_key__=(),
                __subflavor__=None,
                __subflavors__={},
                imgs=None,
                videos=None,
                conversation=json.dumps(_qa("hello", "hi there")),
            )
        )
        self.assertIsNone(getattr(flat, "subsegment_ids", None))

        packed = encoder.pack_selected_samples([flat, tree])
        n_flat = len(flat.input_ids)
        self.assertTrue(bool((packed.subsegment_ids[:n_flat] == 10000).all()))
        torch.testing.assert_close(packed.subsegment_ids[n_flat:], tree.subsegment_ids)

        out = encoder.encode_batch(encoder.batch([packed]))
        self.assertEqual(tuple(out["subsegment_ids"].shape), (1, 256))
        self.assertTrue(bool((out["subsegment_ids"][0, len(packed.input_ids) :] == 10000).all()))

        flat_only = encoder.encode_batch(encoder.batch([encoder.pack_selected_samples([flat])]))
        self.assertIsNone(flat_only["subsegment_ids"])
