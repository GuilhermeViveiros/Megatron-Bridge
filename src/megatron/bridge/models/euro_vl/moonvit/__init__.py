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

"""Vendored MoonViT vision encoder (from moonshotai/MoonViT-SO-400M).

Copied in-repo to drop the auto_map / trust_remote_code dependency so the config
and model are first-class importable classes.
"""

from megatron.bridge.models.euro_vl.moonvit.configuration_moonvit import MoonViTConfig
from megatron.bridge.models.euro_vl.moonvit.image_processing_moonvit import MoonViTImageProcessor
from megatron.bridge.models.euro_vl.moonvit.modeling_moonvit import MoonVitPretrainedModel
from megatron.bridge.models.euro_vl.moonvit.video_processing_moonvit import MoonViTVideoProcessor


class MoonViTVisionProcessor(MoonViTVideoProcessor, MoonViTImageProcessor):
    """Unified MoonViT vision processor handling both images and videos.

    Combines the image pipeline (:class:`MoonViTImageProcessor`, via ``__call__`` ->
    ``pixel_values`` + ``image_grid_hws``) and the per-frame video pipeline
    (:class:`MoonViTVideoProcessor`, via ``preprocess_videos`` -> ``pixel_values_videos`` +
    ``video_grid_thw``), so a single instance drives both modalities. The
    :attr:`image_processor` / :attr:`video_processor` views (both ``self``) let a single
    instance be handed to consumers that expect the two separate sub-processors.
    """

    @property
    def image_processor(self) -> "MoonViTImageProcessor":
        """This instance viewed as its image processor (it subclasses MoonViTImageProcessor)."""
        return self

    @property
    def video_processor(self) -> "MoonViTVideoProcessor":
        """This instance viewed as its video processor (it subclasses MoonViTVideoProcessor)."""
        return self


__all__ = [
    "MoonViTConfig",
    "MoonViTImageProcessor",
    "MoonViTVideoProcessor",
    "MoonViTVisionProcessor",
    "MoonVitPretrainedModel",
]
