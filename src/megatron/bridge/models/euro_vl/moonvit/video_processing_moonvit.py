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

"""Video processor for MoonViT (per-frame 2D encoding, no temporal merge).

Mirrors the Qwen2.5-VL processor *structure* — a dedicated video processor that
returns ``pixel_values_videos`` + ``video_grid_thw`` — but reuses MoonViT's own
per-frame patchify. MoonViT has no temporal dimension, so each frame is encoded
exactly like an image and the ``t`` axis of ``video_grid_thw`` is simply the
number of sampled frames (NOT a merged temporal patch count). Per-video token
count is therefore ``t * (h // merge_h) * (w // merge_w)``.

PIL cannot decode video; an upstream video decoder (decord / torchvision / av,
invoked by ``apply_chat_template`` -> ``make_batched_videos``) yields numpy/tensor
frames. We then convert each frame as needed.

Two preprocessing paths:

- ``preprocess_videos`` (default): each frame -> PIL -> MoonViT's PIL ``_preprocess``.
  BIT-IDENTICAL to the image path. This keeps the (frozen during projector
  alignment) pretrained encoder in-distribution: MoonViT was pretrained on
  PIL-preprocessed images, and a frozen encoder cannot adapt to a different resize
  backend. Safe but per-frame (Python loop).

- ``vectorized_preprocess``: tensor-native resize/patchify (torchvision bicubic),
  no PIL round-trip. ~5% relative-L2 feature difference vs PIL (measured), faster.
  Adopt only once the encoder is unfrozen / trained with a small LR (e.g. SFT),
  NOT during frozen-vision projector alignment. See Task #12.
"""

import math
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from transformers.image_processing_utils import BatchFeature
from transformers.utils import TensorType

from megatron.bridge.models.euro_vl.moonvit.image_processing_moonvit import MoonViTImageProcessor


def _to_pil(frame) -> Image.Image:
    """Coerce a decoded frame (PIL / numpy [H,W,C] / torch tensor) to a PIL RGB image."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if float(arr.max()) <= 1.0 else arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


class MoonViTVideoProcessor(MoonViTImageProcessor):
    """Process videos as sequences of 2D frames through MoonViT's image pipeline."""

    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, num_frames: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames

    def _sample_frames(self, frames: list) -> list:
        """Uniformly subsample a list of frames down to at most ``num_frames``."""
        n = len(frames)
        if self.num_frames is None or n <= self.num_frames:
            return frames
        idx = np.linspace(0, n - 1, self.num_frames).round().astype(int)
        return [frames[i] for i in idx]

    def _pack(self, pixel_values: list, video_grid_thw: list, return_tensors) -> BatchFeature:
        data = {
            "pixel_values_videos": torch.concat(pixel_values, dim=0),
            "video_grid_thw": np.array(video_grid_thw),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess_videos(
        self,
        videos: list,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """Preprocess videos via MoonViT's PIL path (bit-identical to the image path).

        Args:
            videos: List of videos; each video is a list of frames (PIL/numpy/tensor).
            return_tensors: Optional tensor type for the returned BatchFeature.

        Returns:
            BatchFeature with ``pixel_values_videos`` [total_patches, C, p, p] and
            ``video_grid_thw`` [num_videos, 3], each row ``(t, h, w)`` with ``t`` =
            number of sampled frames and ``(h, w)`` the per-frame patch grid.
        """
        pixel_values, video_grid_thw = [], []
        for frames in videos:
            frames = self._sample_frames(list(frames))
            grids = set()
            for frame in frames:
                patches, (h, w) = self._preprocess(_to_pil(frame))
                pixel_values.append(patches)
                grids.add((h, w))
            # Frames of a single video share resolution, so rescale is deterministic.
            if len(grids) != 1:
                raise ValueError(f"Video frames produced inconsistent grids {grids}; frames must share resolution.")
            h, w = grids.pop()
            video_grid_thw.append((len(frames), h, w))
        return self._pack(pixel_values, video_grid_thw, return_tensors)

    # ------------------------------------------------------------------
    # Future fast path (Task #12) — tensor-native, NOT bit-identical to PIL.
    # ------------------------------------------------------------------
    def _rescale_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Tensor mirror of MoonViTImageProcessor.rescale (torchvision bicubic)."""
        _, h, w = image.shape
        patch_size = self.patch_size
        if (w // patch_size) * (h // patch_size) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((w // patch_size) * (h // patch_size)))
            new_w, new_h = int(w * scale), int(h * scale)
            image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        if self.pad_input:
            _, new_h, new_w = image.shape
            pad_size_h = self.merge_kernel_size[0] * patch_size
            pad_size_w = self.merge_kernel_size[1] * patch_size
            pad_h = (pad_size_h - new_h % pad_size_h) % pad_size_h
            pad_w = (pad_size_w - new_w % pad_size_w) % pad_size_w
            image = TF.pad(image, [0, 0, pad_w, pad_h])  # [left, top, right, bottom]
        else:
            _, new_h, new_w = image.shape
            new_w = new_w - new_w % patch_size
            new_h = new_h - new_h % patch_size
            image = TF.center_crop(image, [new_h, new_w])
        return image

    def vectorized_preprocess(
        self,
        videos: list,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """Tensor-native video preprocessing (torchvision bicubic; NOT PIL-identical).

        Same outputs/shapes as ``preprocess_videos`` but uses torch ops directly on
        the decoded frames — no PIL rescale. ~5% feature difference vs the PIL path.
        Use only when the encoder is unfrozen / trained with a small LR (e.g. SFT),
        never during frozen-vision projector alignment. See Task #12.
        """
        pixel_values, video_grid_thw = [], []
        for frames in videos:
            frames = self._sample_frames(list(frames))
            grids = set()
            for frame in frames:
                t = TF.to_tensor(_to_pil(frame))  # [C, H, W] in [0, 1]
                t = self._rescale_tensor(t)
                t = TF.normalize(t, self.image_mean, self.image_std)
                patches, (h, w) = self.patchify(t)
                pixel_values.append(patches)
                grids.add((h, w))
            if len(grids) != 1:
                raise ValueError(f"Video frames produced inconsistent grids {grids}; frames must share resolution.")
            h, w = grids.pop()
            video_grid_thw.append((len(frames), h, w))
        return self._pack(pixel_values, video_grid_thw, return_tensors)
