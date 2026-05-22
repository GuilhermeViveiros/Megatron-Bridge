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

from dataclasses import dataclass
from typing import Any, Optional

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.euro_vl.modeling_euro_vl import EuroVLModel
from megatron.bridge.models.gpt_provider import GPTModelProvider


# Token ID of the new <image> special token added to EuroLLM's vocabulary.
# EuroLLM vocab_size=128000; <image> is appended at index 128000,
# and vocab_size is padded to 128128 (next multiple of 128).
IMAGE_TOKEN_ID: int = 128000


@dataclass
class EuroVLModelProvider(GPTModelProvider):
    """Model provider for EuroVL (MoonViT + EuroLLM).

    Inherits all standard GPT/LLM fields from GPTModelProvider.
    VLM-specific fields are defined below.
    """

    # VLMs must not scatter embeddings across SP regions because image token
    # embeddings are inserted into the sequence after the embedding lookup.
    scatter_embedding_sequence_parallel: bool = False

    # MoonViT config object (MoonViTConfig).  Set before calling provide().
    vision_config: Optional[Any] = None

    # MoonViT hidden_size=1152, merge_kernel_size=[2,2] → 4*1152=4608 per merged token.
    projector_input_dim: int = 4608
    # EuroLLM hidden_size.
    projector_output_dim: int = 2048

    # Token ID used as image placeholder in the text sequence.
    image_token_id: int = IMAGE_TOKEN_ID

    # Attention strategy for image token positions in the LLM decoder.
    # False (default): pure causal masking — simplest baseline.
    # True: bidirectional attention within each image's token block.
    use_bidirectional_image_attention: bool = False

    # Freeze flags for two-stage training.
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> EuroVLModel:
        """Instantiate the full EuroVL model and apply freeze flags."""
        model = EuroVLModel(
            self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model

    def provide_language_model(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> MCoreGPTModel:
        """Instantiate the EuroLLM decoder only (used by pipeline-parallel stages)."""
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)