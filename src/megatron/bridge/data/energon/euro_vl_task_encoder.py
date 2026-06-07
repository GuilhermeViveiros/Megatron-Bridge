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

"""Energon task encoder for EuroVL (MoonViT + EuroLLM).

EuroVL data is curated as a **CrudeWebdataset** with a single, category-agnostic
sample structure: each sample is a raw ``{key}.jpg`` plus a ``{key}.json`` that
already holds the full ChatML conversation
(``[{"role": "user", "content": "<image>\\n..."}, {"role": "assistant", ...}]``).
The same layout is used for captioning, VQA, OCR, etc. — only the conversation
content differs — so a single cooker handles every source.

"Crude" means energon does not decode the sample; this encoder registers a
:class:`~megatron.energon.Cooker` that decodes the image and passes the conversation
through into a :class:`ChatMLSample`, then reuses the generic
:class:`HFEncoderVLMTaskEncoder` machinery (which drives ``EuroVLProcessor`` for
joint tokenization + MoonViT preprocessing and emits ``GenericVisualInputs`` with
``pixel_values`` + ``image_grid_thw``).
"""

import io
import json
from typing import Any, Optional

import torch
from megatron.energon import Cooker, basic_sample_keys
from PIL import Image

from megatron.bridge.data.energon.hf_encoder_task_encoder import HFEncoderTaskSample, HFEncoderVLMTaskEncoder
from megatron.bridge.data.energon.task_encoder_utils import IGNORE_INDEX, ChatMLSample, cook_chatml_sample
from megatron.bridge.data.vlm_datasets.collate import create_multiturn_loss_mask_by_search
from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids


# Crude-sample extensions to look for, in priority order.
_IMAGE_EXTS = ("jpg", "jpeg", "png", "image")
_CONVERSATION_EXTS = ("json", "conversation", "txt")


def _crude_get(sample: dict, keys: tuple[str, ...]) -> Optional[Any]:
    """Return the first present key from a crude energon sample dict."""
    for k in keys:
        if k in sample:
            return sample[k]
    return None


class EuroVLTaskEncoder(HFEncoderVLMTaskEncoder):
    """Crude-sample task encoder for EuroVL energon datasets.

    Category-agnostic: every source shares the same crude structure (``jpg`` +
    ``json`` conversation), so one cooker serves captioning, VQA, OCR, etc.

    Args:
        processor: An ``EuroVLProcessor`` (supports ``apply_chat_template`` and
            ``__call__(text=, images=)`` returning ``pixel_values`` + ``image_grid_thw``).
        seq_length: Maximum sequence length (tokens truncated to this).
    """

    def __init__(self, processor, seq_length: int = 4096) -> None:
        # EuroVLProcessor returns pixel_values + image_grid_thw (3D, t=1); capture both
        # so GenericVisualInputs forwards them to EuroVLModel and the FLOP counter sees the grid.
        super().__init__(
            processor=processor,
            seq_length=seq_length,
            visual_keys=("pixel_values", "image_grid_thw"),
        )
        # Register the cooker that decodes a crude sample into a ChatMLSample. A bound
        # method is picklable (the encoder itself is sent to dataloader workers).
        self.cookers = [Cooker(cook=self._cook)]

    def _cook(self, sample: dict) -> ChatMLSample:
        """Decode a crude sample (``jpg`` + ``json`` conversation) into a :class:`ChatMLSample`.

        Energon's webdataset decoder already turns ``jpg`` into a CHW tensor and ``json``
        into a parsed object, so we pass those through: ``HFEncoderVLMTaskEncoder``
        converts the image tensor to PIL (``_images_to_pil``) and ``cook_chatml_sample``
        parses the conversation. Raw-bytes inputs are handled too, in case a different
        energon decode config is used.
        """
        raw_img = _crude_get(sample, _IMAGE_EXTS)
        if raw_img is None:
            raise KeyError(f"crude sample has no image (looked for {_IMAGE_EXTS}); keys={list(sample.keys())}")
        # tensor / PIL -> pass through (converted to PIL downstream); bytes -> decode here.
        if isinstance(raw_img, (bytes, bytearray)):
            raw_img = Image.open(io.BytesIO(raw_img)).convert("RGB")

        raw_conv = _crude_get(sample, _CONVERSATION_EXTS)
        if raw_conv is None:
            raise KeyError(f"crude sample has no conversation (looked for {_CONVERSATION_EXTS}); keys={list(sample.keys())}")
        if isinstance(raw_conv, (bytes, bytearray)):
            raw_conv = raw_conv.decode("utf-8")
        # ChatMLSample.conversation is a JSON string; serialize if energon already parsed it.
        if not isinstance(raw_conv, str):
            raw_conv = json.dumps(raw_conv)

        return ChatMLSample(
            **basic_sample_keys(sample),
            conversation=raw_conv,
            imgs=[raw_img],
        )

    def encode_sample(self, sample: ChatMLSample) -> HFEncoderTaskSample:
        """Encode like the generic HF encoder, but build the loss mask with the
        repo-standard search helper.

        The base ``HFEncoderVLMTaskEncoder`` masks via a naive exact-token search of the
        standalone assistant text, which fails for EuroLLM's SentencePiece tokenizer (a
        response following a newline tokenizes without its leading ``▁``). We reuse
        ``create_multiturn_loss_mask_by_search`` — the same helper every VLM collate uses
        (qwen2_5, glm4v, ministral3, the EuroVL mock path) — which searches the *final*
        ``input_ids`` (robust to ``<image>`` expansion) with newline-context candidates.
        """
        encoded = super().encode_sample(sample)

        conversation = cook_chatml_sample(sample.conversation)
        skipped = extract_skipped_token_ids(self.processor)
        mask = create_multiturn_loss_mask_by_search(
            {"conversation": conversation}, encoded.input_ids, self.processor, skipped
        )
        loss_mask = torch.tensor(mask, dtype=torch.float32)

        # Shift to align the loss with next-token labels (same convention as the base encoder).
        shifted = torch.zeros_like(loss_mask)
        shifted[:-1] = loss_mask[1:]
        labels = encoded.input_ids.clone().to(torch.long)
        labels[:-1] = encoded.input_ids[1:].to(torch.long)
        labels[-1] = IGNORE_INDEX
        labels[shifted == 0] = IGNORE_INDEX

        encoded.loss_mask = shifted
        encoded.labels = labels
        return encoded
