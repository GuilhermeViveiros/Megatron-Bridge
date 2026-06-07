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

EuroVL data is curated as a **CrudeWebdataset**: each sample is a raw ``{key}.jpg``
plus a ``{key}.json`` metadata blob (NeMo-Curator layout). "Crude" means energon
hands the task encoder the undecoded sample, so this encoder implements
``cook_crude_sample`` to turn it into a :class:`ChatMLSample`, then reuses the
generic :class:`HFEncoderVLMTaskEncoder` machinery (which drives ``EuroVLProcessor``
for joint tokenization + MoonViT preprocessing and emits ``GenericVisualInputs``
with ``pixel_values`` + ``image_grid_thw``).

The same encoder serves every crude EuroVL source; per-source text construction is
selected by ``task`` (captioning today; extend for VQA/OCR).
"""

import io
import json
from typing import Any, Optional

from PIL import Image

from megatron.bridge.data.energon.hf_encoder_task_encoder import HFEncoderVLMTaskEncoder
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample


# Extensions we look for in a crude sample, in priority order.
_IMAGE_EXTS = ("jpg", "jpeg", "png", "image")
_META_EXTS = ("json", "caption", "txt")


def _crude_get(sample: Any, keys: tuple[str, ...]) -> Optional[Any]:
    """Return the first present key from a crude energon sample (dict-like)."""
    for k in keys:
        if isinstance(sample, dict):
            if k in sample:
                return sample[k]
        elif hasattr(sample, k):
            return getattr(sample, k)
    return None


def _crude_meta(sample: Any, key: str, default: Any = None) -> Any:
    """Read an energon sample metadata field (``__key__`` etc.) defensively."""
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


class EuroVLTaskEncoder(HFEncoderVLMTaskEncoder):
    """Crude-sample task encoder for EuroVL captioning/VQA datasets.

    Args:
        processor: An ``EuroVLProcessor`` (supports ``apply_chat_template`` and
            ``__call__(text=, images=)`` returning ``pixel_values`` + ``image_grid_thw``).
        seq_length: Maximum sequence length (tokens truncated to this).
        task: Source task type controlling conversation construction. ``"captioning"``
            builds a single user(``<image>`` + prompt) / assistant(caption) turn.
        prompt: Instruction text paired with the image for captioning samples.
    """

    def __init__(
        self,
        processor,
        seq_length: int = 4096,
        task: str = "captioning",
        prompt: str = "Describe this image.",
    ) -> None:
        # EuroVLProcessor returns pixel_values + image_grid_thw (3D, t=1); capture both
        # so GenericVisualInputs forwards them to EuroVLModel and the FLOP counter sees the grid.
        super().__init__(
            processor=processor,
            seq_length=seq_length,
            visual_keys=("pixel_values", "image_grid_thw"),
        )
        if task != "captioning":
            raise ValueError(f"EuroVLTaskEncoder currently supports task='captioning', got {task!r}")
        self.task = task
        self.prompt = prompt

    def cook_crude_sample(self, sample: Any) -> ChatMLSample:
        """Decode a raw crude sample (``jpg`` + ``json``) into a :class:`ChatMLSample`.

        The ``json`` blob holds at least a ``caption`` (NeMo-Curator metadata). The
        result is a single-turn captioning conversation; the image is passed as a PIL
        image (``HFEncoderVLMTaskEncoder`` handles PIL via ``_images_to_pil``).
        """
        raw_img = _crude_get(sample, _IMAGE_EXTS)
        if raw_img is None:
            raise KeyError(f"crude sample has no image (looked for {_IMAGE_EXTS}); key={_crude_meta(sample, '__key__')}")
        image = raw_img if isinstance(raw_img, Image.Image) else Image.open(io.BytesIO(raw_img)).convert("RGB")

        raw_meta = _crude_get(sample, _META_EXTS)
        if isinstance(raw_meta, (bytes, bytearray)):
            raw_meta = raw_meta.decode("utf-8")
        if isinstance(raw_meta, str):
            try:
                meta = json.loads(raw_meta)
            except json.JSONDecodeError:
                meta = {"caption": raw_meta}
        else:
            meta = raw_meta or {}
        caption = meta.get("caption", "") if isinstance(meta, dict) else str(meta)

        conversation = [
            {"from": "human", "value": f"<image>\n{self.prompt}"},
            {"from": "gpt", "value": caption},
        ]

        return ChatMLSample(
            __key__=_crude_meta(sample, "__key__", ""),
            __restore_key__=_crude_meta(sample, "__restore_key__", ()),
            __subflavor__=_crude_meta(sample, "__subflavor__", None),
            __subflavors__=_crude_meta(sample, "__subflavors__", {}) or {},
            conversation=json.dumps(conversation),
            imgs=[image],
        )
