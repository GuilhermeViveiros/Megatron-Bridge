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
# (https://huggingface.co/moonshotai/MoonViT-SO-400M). Vendored in-repo so the
# config is a first-class importable class (no auto_map / trust_remote_code),
# which lets it serialize cleanly into run_config.yaml and be read by the FLOPs
# calculator — instead of landing in the transformers_modules.* namespace.

from transformers.configuration_utils import PretrainedConfig


class MoonViTConfig(PretrainedConfig):
    """Configuration for the MoonViT vision encoder (patch size, depth, hidden dims, merge kernel)."""

    model_type = "moonvit"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        merge_kernel_size: tuple[int, int] = (2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # Positional embedding config
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        # Transformer config
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Patch merger config
        self.merge_kernel_size = merge_kernel_size

    # --- Aliases for the generic FLOPs calculator (flop_utils.vit_flops) -----------
    # vit_flops reads Qwen-style names; expose them as read-only views of MoonViT's
    # own fields so the shared calculator works without any MoonViT-specific branch.
    @property
    def depth(self) -> int:
        """Number of transformer layers (alias of ``num_hidden_layers``)."""
        return self.num_hidden_layers

    @property
    def spatial_merge_size(self) -> int:
        """Spatial merge factor per side (first dim of ``merge_kernel_size``)."""
        return int(self.merge_kernel_size[0])
