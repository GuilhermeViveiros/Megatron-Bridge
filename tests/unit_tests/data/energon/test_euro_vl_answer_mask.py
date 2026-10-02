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

"""Marker-based assistant-answer masking for EuroVL.

The mask used to be found by searching for the answer's *text* from index 0 and marking its
first occurrence, so an answer repeated earlier in the sequence was supervised inside the
question -- silently on flat samples, and as a "no supervised tokens for branch" error on
message trees. These tests pin the marker-derived behaviour instead.
"""

import numpy as np
import pytest
import torch
from megatron.energon import SkipSample

from megatron.bridge.data.energon.euro_vl_task_encoder import (
    EuroVLTaskEncoder,
    assistant_answer_mask,
    assistant_answer_spans,
    check_media_placeholders,
    count_media_markers,
)


IM_START, IM_END, NL = 3, 4, 271
IMAGE_PAD = 128003
# Content ids; values are arbitrary but distinct.
TOK = {
    "assistant": 10,
    "user": 11,
    "system": 12,
    "2": 20,
    "3": 21,
    "A": 22,
    "cats": 30,
    "dogs": 31,
    "and": 32,
    "yes": 33,
    "text": 34,
}
REV = {v: k for k, v in TOK.items()}
REV.update({IM_START: "<|im_start|>", IM_END: "<|im_end|>", NL: "\n", IMAGE_PAD: "<|vision_pad|>"})


class FakeTokenizer:
    """Just the surface ``assistant_answer_spans``/``assistant_answer_mask`` touch.

    Mirrors how a real tokenizer renders ``<|im_start|>{role}\\n``: specials and the role name
    are separate tokens.
    """

    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": IM_START, "<|im_end|>": IM_END}.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for piece in text.replace("<|im_start|>", "<|im_start|> ").replace("\n", " \n").split(" "):
            if piece == "<|im_start|>":
                ids.append(IM_START)
            elif piece == "<|im_end|>":
                ids.append(IM_END)
            elif piece == "\n":
                ids.append(NL)
            elif piece in TOK:
                ids.append(TOK[piece])
            elif piece:
                raise AssertionError(f"unexpected piece {piece!r}")
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(REV.get(int(i), "?") for i in ids)


def turn(role, content_ids):
    """Token ids for one rendered ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` turn."""
    return [IM_START, TOK[role], NL, *content_ids, IM_END, NL]


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.mark.unit
class TestAssistantAnswerSpans:
    def test_single_turn_span_covers_only_the_answer(self, tok):
        ids = turn("user", [TOK["dogs"]]) + turn("assistant", [TOK["2"]])
        ((start, end),) = assistant_answer_spans(ids, tok)
        assert ids[start:end] == [TOK["2"]]

    def test_user_and_system_turns_are_not_spans(self, tok):
        ids = turn("system", [TOK["text"]]) + turn("user", [TOK["cats"]]) + turn("assistant", [TOK["3"]])
        spans = assistant_answer_spans(ids, tok)
        assert len(spans) == 1
        assert ids[spans[0][0] : spans[0][1]] == [TOK["3"]]

    def test_every_assistant_turn_gets_a_span(self, tok):
        ids = (
            turn("user", [TOK["dogs"]])
            + turn("assistant", [TOK["2"]])
            + turn("user", [TOK["cats"]])
            + turn("assistant", [TOK["3"]])
        )
        spans = assistant_answer_spans(ids, tok)
        assert [ids[s:e] for s, e in spans] == [[TOK["2"]], [TOK["3"]]]

    def test_multi_token_answer(self, tok):
        answer = [TOK["2"], TOK["cats"], TOK["and"], TOK["3"], TOK["dogs"]]
        ids = turn("user", [TOK["text"]]) + turn("assistant", answer)
        ((start, end),) = assistant_answer_spans(ids, tok)
        assert ids[start:end] == answer

    def test_no_assistant_turn_yields_no_spans(self, tok):
        assert assistant_answer_spans(turn("user", [TOK["text"]]), tok) == []

    def test_empty_answer_is_not_a_span(self, tok):
        """``<|im_start|>assistant\\n<|im_end|>`` has no content to supervise."""
        assert assistant_answer_spans(turn("assistant", []), tok) == []

    def test_truncated_header_is_ignored(self, tok):
        """A trailing ``<|im_start|>assistant`` with no newline/content must not crash."""
        ids = turn("user", [TOK["text"]]) + turn("assistant", [TOK["2"]]) + [IM_START, TOK["assistant"]]
        spans = assistant_answer_spans(ids, tok)
        assert [ids[s:e] for s, e in spans] == [[TOK["2"]]]

    def test_missing_markers_raises(self):
        class NoMarkers(FakeTokenizer):
            def convert_tokens_to_ids(self, token):
                return self.unk_token_id

        with pytest.raises(ValueError, match="im_start"):
            assistant_answer_spans([1, 2, 3], NoMarkers())


@pytest.mark.unit
class TestAssistantAnswerMask:
    def test_answer_repeated_in_the_question_is_not_supervised(self, tok):
        """The regression: searching for "2" marked the question's "2"; markers must not.

        Question renders as "... and 2 dogs", so the answer token appears before the answer.
        """
        question = [TOK["and"], TOK["2"], TOK["dogs"]]
        ids = turn("user", question) + turn("assistant", [TOK["2"]])
        answer_at = len(turn("user", question)) + 3  # im_start, role, newline
        mask = assistant_answer_mask(ids, tok, supervise_turn_end=False)
        assert mask[answer_at] == 1.0
        assert np.flatnonzero(mask).tolist() == [answer_at], "supervision landed outside the answer"

    def test_question_quoting_a_previous_answer_is_not_supervised(self, tok):
        """Table-QA shape: turn 2's question repeats turn 1's answer verbatim."""
        ids = (
            turn("user", [TOK["text"]])
            + turn("assistant", [TOK["2"], TOK["cats"]])
            + turn("user", [TOK["2"], TOK["cats"], TOK["dogs"]])
            + turn("assistant", [TOK["2"], TOK["cats"]])
        )
        spans = assistant_answer_spans(ids, tok)
        mask = assistant_answer_mask(ids, tok, supervise_turn_end=False)
        expected = sorted(i for s, e in spans for i in range(s, e))
        assert np.flatnonzero(mask).tolist() == expected
        assert len(spans) == 2

    def test_supervises_turn_end_by_default(self, tok):
        """``<|im_end|>`` is EuroVL's EOS, so the model must learn to emit it."""
        ids = turn("user", [TOK["text"]]) + turn("assistant", [TOK["2"]])
        marked = np.flatnonzero(assistant_answer_mask(ids, tok)).tolist()
        assert [ids[i] for i in marked] == [TOK["2"], IM_END]

    def test_turn_end_can_be_excluded(self, tok):
        ids = turn("user", [TOK["text"]]) + turn("assistant", [TOK["2"]])
        marked = np.flatnonzero(assistant_answer_mask(ids, tok, supervise_turn_end=False)).tolist()
        assert [ids[i] for i in marked] == [TOK["2"]]

    def test_turn_end_survives_the_pad_token_filter(self, tok):
        """``extract_skipped_token_ids`` lists ``<|im_end|>``; supervising EOS must override it."""
        ids = turn("user", [TOK["text"]]) + turn("assistant", [TOK["2"]])
        skipped = torch.tensor([IM_START, IM_END, IMAGE_PAD])
        marked = np.flatnonzero(assistant_answer_mask(ids, tok, skipped)).tolist()
        assert [ids[i] for i in marked] == [TOK["2"], IM_END]
        # ... and stays masked when EOS supervision is off.
        off = np.flatnonzero(assistant_answer_mask(ids, tok, skipped, supervise_turn_end=False)).tolist()
        assert [ids[i] for i in off] == [TOK["2"]]

    def test_skipped_ids_inside_an_answer_are_masked(self, tok):
        ids = turn("user", [TOK["text"]]) + turn("assistant", [TOK["2"], IMAGE_PAD, TOK["3"]])
        marked = np.flatnonzero(assistant_answer_mask(ids, tok, torch.tensor([IMAGE_PAD]))).tolist()
        assert [ids[i] for i in marked] == [TOK["2"], TOK["3"], IM_END]

    def test_mask_shape_and_dtype_match_the_sequence(self, tok):
        ids = turn("user", [TOK["text"]]) + turn("assistant", [TOK["2"]])
        mask = assistant_answer_mask(ids, tok)
        assert mask.shape == (len(ids),)
        assert mask.dtype == np.float32

    def test_accepts_tensor_input(self, tok):
        ids = turn("user", [TOK["text"]]) + turn("assistant", [TOK["2"]])
        from_list = assistant_answer_mask(ids, tok)
        from_tensor = assistant_answer_mask(torch.tensor(ids), tok)
        np.testing.assert_array_equal(from_list, from_tensor)

    def test_no_answer_raises_instead_of_supervising_nothing(self, tok):
        """Silently returning zeros is what hid the original bug."""
        with pytest.raises(ValueError, match="No assistant answer span"):
            assistant_answer_mask(turn("user", [TOK["text"]]), tok)

    def test_label_shift_trains_the_model_to_emit_eos(self, tok):
        """With the encoder's shift, a supervised position predicts the NEXT token."""
        ids = turn("user", [TOK["text"]]) + turn("assistant", [TOK["2"]])
        mask = assistant_answer_mask(ids, tok)
        shifted = np.zeros_like(mask)
        shifted[:-1] = mask[1:]
        targets = [ids[p + 1] for p in np.flatnonzero(shifted)]
        assert targets == [TOK["2"], IM_END]


@pytest.mark.unit
class TestMediaPlaceholderCheck:
    """Attached media must correspond 1:1 with the conversation's <image>/<video> markers.

    Without this, a sample with an image but no ``<image>`` marker encoded with zero image
    tokens and no ``pixel_values``: text-only training on a question about an image.
    """

    @staticmethod
    def _conv(user_text, answer="ok"):
        return [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]

    def test_matching_image_marker_passes(self):
        check_media_placeholders(self._conv("<image>\nWhat is this?"), n_images=1, n_videos=0, key="k")

    def test_matching_video_marker_passes(self):
        check_media_placeholders(self._conv("<video>\nDescribe it."), n_images=0, n_videos=1, key="k")

    def test_text_only_sample_passes(self):
        check_media_placeholders(self._conv("What is 2+2?"), n_images=0, n_videos=0, key="k")

    def test_image_without_marker_skips(self):
        """The reported bug: mminstruct_qa sample with 1 image and no <image> tag."""
        with pytest.raises(SkipSample):
            check_media_placeholders(self._conv("What is this?"), n_images=1, n_videos=0, key="k")

    def test_video_without_marker_skips(self):
        with pytest.raises(SkipSample):
            check_media_placeholders(self._conv("Describe it."), n_images=0, n_videos=1, key="k")

    def test_fewer_markers_than_images_skips(self):
        """Two images, one marker -> the second image is dropped silently."""
        with pytest.raises(SkipSample):
            check_media_placeholders(self._conv("<image>\nCompare these."), n_images=2, n_videos=0, key="k")

    def test_more_markers_than_images_skips(self):
        """A marker with no media stays literal '<image>' text in the prompt."""
        with pytest.raises(SkipSample):
            check_media_placeholders(self._conv("<image><image>\nCompare."), n_images=1, n_videos=0, key="k")

    def test_markers_counted_across_turns(self):
        conv = [
            {"role": "user", "content": "<image>\nFirst?"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "<image>\nSecond?"},
            {"role": "assistant", "content": "b"},
        ]
        check_media_placeholders(conv, n_images=2, n_videos=0, key="k")

    def test_mixed_image_and_video(self):
        conv = self._conv("<image>\n<video>\nCompare the photo and the clip.")
        check_media_placeholders(conv, n_images=1, n_videos=1, key="k")
        with pytest.raises(SkipSample):  # video attached but only the image marker present
            check_media_placeholders(self._conv("<image>\nWhat?"), n_images=1, n_videos=1, key="k")

    def test_counts_structured_content(self):
        """After placeholder structuring, markers are content items rather than inline text."""
        conv = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]},
            {"role": "assistant", "content": "ok"},
        ]
        assert count_media_markers(conv) == {"image": 1, "video": 0}
        check_media_placeholders(conv, n_images=1, n_videos=0, key="k")

    def test_counts_inline_and_structured_text_parts(self):
        conv = [{"role": "user", "content": [{"type": "text", "text": "<video> then <video>"}]}]
        assert count_media_markers(conv) == {"image": 0, "video": 2}


@pytest.mark.unit
class TestVideoDecodeFailureSkips:
    """A clip whose seek fails must become SkipSample, not an arbitrary PyAV exception.

    Clips cut with `-c copy` can have their first keyframe after the decoder's first seek
    target, and PyAV then fails the seek with EPERM (BUGS.md A3 / B8). Energon's normal
    iteration would merely log the traceback, but its RESTORE path re-runs the sample encoder
    with restore_error_handler=reraise_exception, so a bad clip sitting in a restored packing
    buffer would kill every resume from that checkpoint. SkipSample is energon's sanctioned
    control flow and is handled on both paths.
    """

    class _Proc:
        """Minimal processor whose video_processor always fails to decode."""

        class _VP:
            def __init__(self, exc):
                self._exc = exc

            def decode_video_bytes(self, _data):
                raise self._exc

        def __init__(self, exc):
            self.video_processor = self._VP(exc)
            self.tokenizer = FakeTokenizer()

    def _encoder(self, exc):
        enc = EuroVLTaskEncoder.__new__(EuroVLTaskEncoder)  # no __init__: no real processor needed
        enc.processor = self._Proc(exc)
        return enc

    @pytest.mark.parametrize(
        "exc",
        [
            PermissionError(1, "Operation not permitted"),  # what PyAV raises on a failed seek
            ValueError("no frames decoded from video bytes"),
            ValueError("clip exposes no container/stream duration; cannot seek-decode."),
            RuntimeError("some other decoder failure"),
        ],
    )
    def test_decode_failure_becomes_skip_sample(self, exc):
        enc = self._encoder(exc)
        with pytest.raises(SkipSample):
            enc._frames_from_video(b"not-a-real-mp4", key="shard-000024.tar/9K2xEOO7rgg_g0")

    def test_skip_sample_from_the_decoder_is_not_rewrapped(self):
        enc = self._encoder(SkipSample())
        with pytest.raises(SkipSample):
            enc._frames_from_video(b"bytes", key="k")

    def test_warning_names_the_failing_sample(self, caplog):
        enc = self._encoder(PermissionError(1, "Operation not permitted"))
        with caplog.at_level("WARNING"):
            with pytest.raises(SkipSample):
                enc._frames_from_video(b"bytes", key="shard-000002.tar/5VtOqePIlmw_g0")
        assert "shard-000002.tar/5VtOqePIlmw_g0" in caplog.text
        assert "keyframe" in caplog.text

    def test_successful_decode_passes_through(self):
        enc = EuroVLTaskEncoder.__new__(EuroVLTaskEncoder)

        class _OK:
            class _VP:
                def decode_video_bytes(self, _data):
                    return (["frame"], [0.0])

            video_processor = _VP()

        enc.processor = _OK()
        assert enc._frames_from_video(b"bytes", key="k") == (["frame"], [0.0])
