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

"""EuroVL processor — MoonViT image + video processors + EuroLLM tokenizer.

Mirrors the Qwen2.5-VL processor design: a ``ProcessorMixin`` with three
sub-components (``image_processor``, ``video_processor``, ``tokenizer``) and a
``__call__(images, text, videos)`` that routes each modality and expands the
``<image>`` / ``<video>`` placeholders by their grid token counts.

Unlike Qwen, EuroVL's vision encoder (MoonViT) is 2D-only, so the video path
produces a per-frame ``(t, h, w)`` grid with no temporal merge (``t`` = number of
sampled frames). See ``moonvit/video_processing_moonvit.py``.

EuroVL has no single packaged HF processor — MoonViT and the EuroLLM tokenizer
live at separate paths — so ``from_pretrained`` takes both paths explicitly.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import make_flat_list_of_images
from transformers.processing_utils import ProcessorMixin
from transformers.video_utils import make_batched_videos

from megatron.bridge.models.euro_vl.utils import format_timestamp


logger = logging.getLogger(__name__)


class EuroVLProcessor(ProcessorMixin):
    """Combined image + video + text processor for EuroVL."""

    attributes = ["image_processor", "video_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_kwargs = ["chat_template"]

    image_token = "<image>"
    video_token = "<video>"
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        timestamp_format: str = "seconds",
        default_fps: float = 2.0,
        **kwargs,
    ):
        # Assign sub-components directly (custom in-repo processors don't round-trip
        # through ProcessorMixin's Auto-class resolution).
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        # Per-frame timestamp prefix (Qwen3-VL style): "seconds", "hms", or "random".
        # "random" mixes both formats per video so the model learns diverse timecodes.
        self.timestamp_format = timestamp_format
        # Fallback fps when video_metadata (fps + frame indices) is not provided.
        self.default_fps = default_fps
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: str,
        chat_template: Optional[str] = None,
        timestamp_format: str = "seconds",
        default_fps: float = 2.0,
        fps: float = 1.0,
        min_frames: int = 2,
        max_frames: int = 64,
        min_pixels: int = 40_000,
        max_pixels: int = 802_816,
        seq_length: Optional[int] = None,
        budget_fraction: float = 0.85,
    ) -> "EuroVLProcessor":
        """Build the processor from a single EuroVL checkpoint directory.

        The assembled EuroVL HF checkpoint holds the MoonViT image-processor config,
        the tokenizer (vision tokens + chat template), and the model weights together,
        so one path provides everything.

        Args:
            pretrained_model_name_or_path: Path to the EuroVL HF checkpoint directory.
            chat_template:  Optional Jinja chat template string (else taken from tokenizer).
            timestamp_format: per-frame timestamp format — "seconds", "hms", or "random".
            default_fps: fallback fps for TIMESTAMP TEXT when video_metadata doesn't supply one
                (distinct from ``fps`` below, which drives frame SAMPLING).
            fps: Target frames sampled per second of clip duration (frame-count policy).
            min_frames / max_frames: Bounds on the fps-derived frame count.
            min_pixels / max_pixels: Bounds on a single resized frame's pixel count.
            seq_length: If given, sets the video token budget to
                ``budget_fraction * seq_length * factor**2`` (``factor`` = patch*merge = 28),
                matching the model's actual context length. If ``None``, keeps
                ``MoonViTVideoProcessor``'s built-in default (equivalent to seq_length=8192).
            budget_fraction: Fraction of ``seq_length`` the video budget may spend (see above).
        """
        from transformers import AutoTokenizer

        from megatron.bridge.models.euro_vl.moonvit.image_processing_moonvit import MoonViTImageProcessor
        from megatron.bridge.models.euro_vl.moonvit.video_processing_moonvit import MoonViTVideoProcessor

        path = pretrained_model_name_or_path
        # MoonViTImageProcessor resizes with torchvision (not the reference's PIL) — the former
        # VectorizedMoonViTImageProcessor subclass, folded back into the base class once video
        # landed its own batched path. See that class's module docstring.
        image_processor = MoonViTImageProcessor.from_pretrained(path)
        # Load the video processor then set attributes directly — passing them through
        # from_pretrained would trip BaseImageProcessor's unknown-kwarg warning.
        video_processor = MoonViTVideoProcessor.from_pretrained(path)
        video_processor.fps = fps
        video_processor.min_frames = min_frames
        video_processor.max_frames = max_frames
        video_processor.min_pixels = min_pixels
        video_processor.max_pixels = max_pixels
        if seq_length is not None:
            video_processor.total_pixels = int(budget_fraction * seq_length * video_processor._merge_factor**2)
            logger.info(
                "Video token budget: total_pixels=%d (budget_fraction=%.2f * seq_length=%d * merge_factor^2=%d)",
                video_processor.total_pixels, budget_fraction, seq_length, video_processor._merge_factor**2,
            )
        logger.info(
            "Video smart-resize policy: fps=%.2f frames=[%d,%d] pixels=[%d,%d] total_pixels=%d",
            fps, min_frames, max_frames, min_pixels, max_pixels, video_processor.total_pixels,
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)
        return cls(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            timestamp_format=timestamp_format,
            default_fps=default_fps,
        )

    @property
    def _merge_area(self) -> int:
        """Patches merged into one token (merge_h * merge_w), e.g. 2*2 = 4."""
        mk = self.image_processor.merge_kernel_size
        return int(mk[0]) * int(mk[1])

    def _video_timestamps(self, num_frames: int, metadata: Optional[dict]) -> list[float]:
        """Per-frame timestamps (seconds) for the sampled frames of one video.

        Priority: explicit ``metadata["timestamps"]`` (length must be num_frames) ->
        ``metadata["fps"]`` (uniform ``f/fps``) -> ``self.default_fps`` fallback. For
        accurate timecodes the data pipeline should pass per-frame ``timestamps``
        matching exactly the frames it sampled.
        """
        if metadata is not None and metadata.get("timestamps") is not None:
            ts = list(metadata["timestamps"])
            if len(ts) != num_frames:
                raise ValueError(f"timestamps length {len(ts)} != num_frames {num_frames}")
            return [float(x) for x in ts]
        fps = (metadata.get("fps") if metadata is not None else None) or self.default_fps
        return [f / float(fps) for f in range(num_frames)]

    def __call__(
        self,
        images=None,
        text: Optional[Union[str, list[str]]] = None,
        videos=None,
        *,
        padding: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        """Process text with optional images and/or videos.

        Expands each ``<image>`` placeholder to ``(h//mh)*(w//mw)`` tokens and each
        ``<video>`` placeholder to ``Σ_frames (h//mh)*(w//mw)`` tokens, matching the
        number of projected MoonViT tokens. Mirrors Qwen2.5-VL's placeholder
        expansion (which uses ``<|placeholder|>`` as a scratch token to avoid
        re-expanding already-inserted placeholders).

        Args:
            images: PIL Image(s) for image inputs (flat list, one per ``<image>``).
            text:   Prompt string or list of strings.
            videos: List of videos, each a list of frames (PIL Images), one per ``<video>``.
            padding: Pad sequences to a common length.
            return_tensors: Tensor type for the returned BatchFeature.

        Returns:
            BatchFeature with ``input_ids``, ``attention_mask``, and (when present)
            ``pixel_values`` + ``image_grid_thw`` (``[num_images, 3]`` = (1, h, w)) and/or
            ``pixel_values_videos`` + ``video_grid_thw``.
        """
        if text is None:
            raise ValueError("`text` is required.")
        if isinstance(text, str):
            text = [text]
        text = list(text)

        image_inputs: dict = {}
        video_inputs: dict = {}

        if images is not None:
            # apply_chat_template passes nested lists ([[img], ...]); flatten to a
            # flat list of images the way transformers' own processors do.
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, return_tensors="pt")
            image_grid_hws = image_inputs["image_grid_hws"]  # [num_images, 2]
            # Expand each <image> by the per-image projected token count.
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    h, w = int(image_grid_hws[index][0]), int(image_grid_hws[index][1])
                    n = (h // self.image_processor.merge_kernel_size[0]) * (
                        w // self.image_processor.merge_kernel_size[1]
                    )
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * n, 1)
                    index += 1
            for i in range(len(text)):
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            # Normalize to a list of videos (each a list of frames), matching
            # transformers' video batching used by apply_chat_template.
            videos = make_batched_videos(videos)
            video_inputs = self.video_processor.vectorized_preprocess(videos, return_tensors="pt")
            video_grid_thw = video_inputs["video_grid_thw"]  # [num_videos, 3] = (t, h, w)

            # Qwen3-VL-style per-frame timestamped blocks. Each frame becomes:
            #   <{timestamp}><|vision_start|><video>*frame_tokens<|vision_end|>
            # Replace the whole <|vision_start|><video><|vision_end|> block emitted by
            # the chat template (falling back to a bare <video> if not wrapped).
            video_metadata = kwargs.get("video_metadata")
            mh, mw = self.video_processor.merge_kernel_size
            whole_block = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    t, h, w = (int(x) for x in video_grid_thw[index])
                    frame_tokens = (h // mh) * (w // mw)
                    md = video_metadata[index] if video_metadata is not None else None
                    timestamps = self._video_timestamps(t, md)
                    per_frame = "".join(
                        f"<{format_timestamp(ts, self.timestamp_format)}>"
                        + self.vision_start_token
                        + "<|placeholder|>" * frame_tokens
                        + self.vision_end_token
                        for ts in timestamps
                    )
                    if whole_block in text[i]:
                        text[i] = text[i].replace(whole_block, per_frame, 1)
                    else:
                        text[i] = text[i].replace(self.video_token, per_frame, 1)
                    index += 1
            for i in range(len(text)):
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(
            text,
            padding=padding,
            return_tensors=return_tensors,
        )

        data = {**text_inputs}
        if image_inputs:
            data["pixel_values"] = image_inputs["pixel_values"]
            # MoonViT is 2D, but emit the codebase-standard image_grid_thw [num_images, 3]
            # by prepending a unit temporal dim (t=1) to the (h, w) grid. This makes the
            # collate, energon task encoder, shared FLOP counter, and model all read a
            # single name; consumers that need the 2D grid slice ``[:, 1:]``.
            hw = torch.as_tensor(np.asarray(image_inputs["image_grid_hws"]))  # [num_images, 2]
            t = torch.ones((hw.shape[0], 1), dtype=hw.dtype)
            data["image_grid_thw"] = torch.cat([t, hw], dim=-1)  # [num_images, 3] = (1, h, w)
        if video_inputs:
            data["pixel_values_videos"] = video_inputs["pixel_values_videos"]
            data["video_grid_thw"] = torch.as_tensor(np.asarray(video_inputs["video_grid_thw"]))

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """Delegate to the tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Delegate to the tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)


class Qwen3EuroVLProcessor(EuroVLProcessor):
    """EuroVL processor for the Qwen3-backbone variant (:class:`Qwen3EuroVLModel`).

    Identical pipeline (MoonViT image/video processing + ChatML chat template); the only
    difference is the vision placeholder strings: the Qwen3 tokenizer ships its own vision
    tokens built into the base vocab (no vocab extension), so we use them directly —
    ``<|image_pad|>`` (151655) / ``<|video_pad|>`` (151656) instead of EuroVL's appended
    ``<image>`` / ``<video>``. ``<|vision_start|>`` / ``<|vision_end|>`` already match.
    These are the exact ids Qwen3-VL uses, keeping the oracle faithful.
    """

    image_token = "<|image_pad|>"
    video_token = "<|video_pad|>"
