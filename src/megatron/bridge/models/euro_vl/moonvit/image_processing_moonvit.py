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
#
# Adapted from moonshotai/MoonViT-SO-400M
# (https://huggingface.co/moonshotai/MoonViT-SO-400M). Vendored in-repo to drop
# the auto_map / trust_remote_code dependency (see configuration_moonvit.py).

"""Image processor class for MoonViT (vendored from KimiVL).

Deviation from the upstream MoonViT reference: ``rescale``/``_preprocess`` use torchvision
(tensor, bicubic + antialias) instead of the reference's PIL bicubic. This was originally added
as a separate ``VectorizedMoonViTImageProcessor`` subclass to unblock video's high frame counts
(a per-frame PIL Python loop does not scale to ~100 frames/clip), then adopted as the only
backend after an isolated PA loss-curve A/B showed the two overlay within noise (final lm loss
2.3111 PIL vs 2.3129 vectorized after 4000 PA iters) and geometry parity matched on both pad
branches. The subclass was folded back into this class in 2026-09 once video landed its own
batched implementation (``MoonViTVideoProcessor.vectorized_preprocess``), which left the
subclass with no remaining purpose: image preprocessing is one image at a time, so it gained no
batching benefit, and keeping two backends meant the same frozen vision encoder saw two
different resize implementations depending on modality. torchvision is now the single path, so
images and video frames are resized identically.
"""

import math
import numpy as np
from PIL import Image
from typing import Optional, Union

import torch
from torchvision.transforms import functional as TF
from transformers.image_utils import ImageInput, make_list_of_images, valid_images
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class MoonViTImageProcessor(BaseImageProcessor):
    model_type = "moonvit"

    def __init__(
        self,
        patch_size: int = 14,
        pad_input: bool = False,
        image_mean: tuple[float, float, float] = OPENAI_DATASET_MEAN,
        image_std: tuple[float, float, float] = OPENAI_DATASET_STD,
        in_token_limit: int = 4096,
        merge_kernel_size: list[int, int] = [2, 2],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_token_limit = in_token_limit
        self.patch_size = patch_size
        self.pad_input = pad_input
        self.image_mean = image_mean
        self.image_std = image_std
        self.merge_kernel_size = merge_kernel_size

    def rescale(self, image: torch.Tensor, merge_kernel_size: list[int, int] = [2, 2]) -> torch.Tensor:
        """Downscale to fit ``in_token_limit`` patches, then align to the patch/merge grid.

        Operates on a ``[C, H, W]`` tensor (torchvision bicubic), not PIL — see the module
        docstring for why this replaced the reference's PIL path.
        """
        _, h, w = image.shape
        patch_size = self.patch_size

        if (w // patch_size) * (h // patch_size) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((w // patch_size) * (h // patch_size)))
            new_w, new_h = int(w * scale), int(h * scale)
            image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        if self.pad_input:
            _, new_h, new_w = image.shape
            pad_size_h = merge_kernel_size[0] * patch_size
            pad_size_w = merge_kernel_size[1] * patch_size

            pad_h = (pad_size_h - new_h % pad_size_h) % pad_size_h
            pad_w = (pad_size_w - new_w % pad_size_w) % pad_size_w

            image = TF.pad(image, [0, 0, pad_w, pad_h])
        else:
            _, new_h, new_w = image.shape
            new_w = new_w - new_w % patch_size
            new_h = new_h - new_h % patch_size
            image = TF.center_crop(image, [new_h, new_w])

        _, h, w = image.shape
        if w // patch_size >= 512 or h // patch_size >= 512:
            raise ValueError("Exceed pos emb")

        return image

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        return TF.to_tensor(image.convert("RGB"))

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return TF.normalize(image, self.image_mean, self.image_std)

    def patchify(self, image: torch.Tensor) -> tuple[torch.Tensor, list[int, int]]:
        patch_size = self.patch_size
        C, H, W = image.shape
        patches = image.reshape(C, H // patch_size, patch_size, W // patch_size, patch_size)
        patches = patches.permute(1, 3, 0, 2, 4)
        patches = patches.contiguous().view(-1, C, patch_size, patch_size)
        grid_hw = (H // patch_size, W // patch_size)
        return patches, grid_hw

    def _preprocess(self, image: ImageInput) -> tuple[torch.Tensor, list[int, int]]:
        """
        Preprocess image and patchify it.

        Tensor-converts FIRST so ``rescale`` never touches PIL (see module docstring).

        Args:
            image (`ImageInput`):
                Image to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.

        Returns:
            patches: torch.Tensor
            grid_hw: list[int, int]
        """
        image = self.to_tensor(image)
        image = self.rescale(image, self.merge_kernel_size)
        image = self.normalize(image)
        patches, grid_hw = self.patchify(image)
        return patches, grid_hw

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        pixel_values, image_grid_hws = [], []
        for image in images:
            patches, image_grid_hw = self._preprocess(image)
            pixel_values.append(patches)
            image_grid_hws.append(image_grid_hw)
        pixel_values = torch.concat(pixel_values, dim=0)
        image_grid_hws = np.array(image_grid_hws)
        data = {"pixel_values": pixel_values, "image_grid_hws": image_grid_hws}

        return BatchFeature(data=data, tensor_type=return_tensors)


