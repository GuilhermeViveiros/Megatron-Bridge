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

"""Standalone HF reference model for EuroVL (MoonViT + EuroLLM).

This is the pure-transformers definition that ``AutoBridge`` converts to Megatron.
It bundles the vendored MoonViT vision tower, a 2-layer MLP projector, and a causal
LM built from ``text_config`` via ``AutoModelForCausalLM`` (``LlamaForCausalLM`` for
EuroLLM, ``Qwen3ForCausalLM`` for the Qwen3EuroVL oracle backbone), injecting projected
image features into the text embedding stream at ``image_token_id`` positions via
``masked_scatter``.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from megatron.bridge.models.euro_vl.configuration_euro_vl import EuroVLConfig
from megatron.bridge.models.euro_vl.moonvit.modeling_moonvit import MoonVitPretrainedModel


class EuroVLMultiModalProjector(nn.Module):
    """Two-layer MLP mapping merged MoonViT patch tokens into the LLM hidden space.

    Input:  [N, projector_input_dim]  (4 * 1152 = 4608 merged MoonViT tokens)
    Output: [N, projector_output_dim] (LLM hidden size, 2048)
    """

    def __init__(self, config: EuroVLConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.projector_input_dim, config.projector_output_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.projector_output_dim, config.projector_output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class EuroVLForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """EuroVL: MoonViT vision tower + MLP projector + EuroLLM (Llama) decoder."""

    config_class = EuroVLConfig
    base_model_prefix = "model"
    _no_split_modules = ["MoonVitEncoderLayer", "LlamaDecoderLayer", "Qwen3DecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: EuroVLConfig) -> None:
        super().__init__(config)
        self.vision_tower = MoonVitPretrainedModel(config.vision_config)
        self.multi_modal_projector = EuroVLMultiModalProjector(config)
        # Build whichever causal LM `text_config` describes: LlamaConfig -> LlamaForCausalLM
        # (EuroLLM, unchanged behavior) or Qwen3Config -> Qwen3ForCausalLM (the Qwen3EuroVL
        # oracle backbone, incl. QK-norm). Keeps one wrapper for all backbones.
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.post_init()

    # ------------------------------------------------------------------
    # Embedding plumbing (delegate to the language model)
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    # ------------------------------------------------------------------
    # Vision feature extraction
    # ------------------------------------------------------------------
    def _encode(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        """Run packed pixel patches through MoonViT + projector.

        MoonViT returns a list of [N_i, merge_area, hidden] tensors; flatten the
        merge dimension, concatenate, then project to LLM hidden space.
        ``grid_hws`` is the 2D per-image/per-frame grid [num, 2].
        """
        vision_dtype = next(self.vision_tower.parameters()).dtype
        features = self.vision_tower(pixel_values.to(vision_dtype), grid_hws)
        features = torch.cat(features, dim=0).flatten(1)
        return self.multi_modal_projector(features)

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Projected tokens for images. ``image_grid_thw`` is [num_images, 3] = (t=1, h, w);
        MoonViT is 2D, so the unit temporal dim is dropped before encoding."""
        return self._encode(pixel_values, image_grid_thw[:, 1:])

    def get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor) -> torch.Tensor:
        """Projected tokens for videos.

        Video frames are encoded as 2D images (MoonViT has no temporal dim), so the
        per-video ``(t, h, w)`` grid is expanded to ``t`` per-frame ``(h, w)`` rows
        before encoding.
        """
        frame_grids = []
        for t, h, w in video_grid_thw.tolist():
            frame_grids.extend([[h, w]] * t)
        frame_grid_hws = torch.tensor(frame_grids, dtype=video_grid_thw.dtype, device=video_grid_thw.device)
        return self._encode(pixel_values_videos, frame_grid_hws)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw).to(inputs_embeds.dtype)
            image_mask = (
                (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if pixel_values_videos is not None and video_grid_thw is not None:
            video_features = self.get_video_features(pixel_values_videos, video_grid_thw).to(inputs_embeds.dtype)
            video_mask = (
                (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        return self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )
