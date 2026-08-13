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

"""EuroVL configuration — EuroLLM (Llama) language model + MoonViT vision encoder."""

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from megatron.bridge.models.euro_vl.moonvit.configuration_moonvit import MoonViTConfig


# EuroLLM-1.7B-Instruct architecture (Llama-style). Vocab is extended from the base
# 128000 to 128005 to hold the 5 vision special tokens (128000-128004).
_EUROLLM_TEXT_DEFAULTS = dict(
    vocab_size=128005,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=8,
    max_position_embeddings=32768,  # EuroLLM-1.7B final context window
    rms_norm_eps=1e-5,
    rope_theta=1000000.0,  # EuroLLM-1.7B rope_theta (NOT 10000)
    hidden_act="silu",
    attention_bias=False,
    mlp_bias=False,
    tie_word_embeddings=False,
)


class EuroVLConfig(PretrainedConfig):
    """Configuration for EuroVL (EuroLLM language model + MoonViT vision encoder).

    Args:
        text_config: EuroLLM language-model config (``LlamaConfig``), or a dict to
            build one. Defaults to the EuroLLM-1.7B architecture with the vocab
            extended to 128005 for the vision special tokens.
        vision_config: MoonViT vision-encoder config (``MoonViTConfig``), or a dict.
            Defaults to MoonViT-SO-400M.
        projector_input_dim: Input dim of the MLP projector — MoonViT hidden (1152)
            times the merge-kernel area (2*2=4) = 4608.
        projector_output_dim: Output dim of the projector — must match the LLM
            hidden size (2048).
        image_token_id: Placeholder token id whose embeddings are replaced by
            projected image features.
        vision_start_token_id: Delimiter token id opening a vision block.
        vision_end_token_id: Delimiter token id closing a vision block.
        vision_pad_token_id: Vision sequence padding token id (loss-masked).
        video_token_id: Placeholder token id for video frames.
        tie_word_embeddings: Whether to tie input/output embeddings. EuroLLM ships
            untied, so this defaults to False.
    """

    model_type = "euro_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": MoonViTConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projector_input_dim: int = 4608,
        projector_output_dim: int = 2048,
        image_token_id: int = 128000,
        vision_start_token_id: int = 128001,
        vision_end_token_id: int = 128002,
        vision_pad_token_id: int = 128003,
        video_token_id: int = 128004,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # Vision sub-config (dict -> object, or default MoonViT-SO-400M).
        if isinstance(vision_config, dict):
            vision_config = MoonViTConfig(**vision_config)
        elif vision_config is None:
            vision_config = MoonViTConfig()
        self.vision_config = vision_config

        # Text sub-config (dict -> object, or default EuroLLM-1.7B). Dicts are resolved
        # by their embedded ``model_type`` (standard HF composite-config pattern, cf.
        # LlavaConfig): "llama" -> LlamaConfig (EuroLLM, default), "qwen3" -> Qwen3Config
        # (the Qwen3EuroVL oracle backbone), etc. — so a saved checkpoint reloads with
        # the correct backbone config class.
        if isinstance(text_config, dict):
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            text_model_type = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_model_type](**text_config)
        elif text_config is None:
            text_config = LlamaConfig(**_EUROLLM_TEXT_DEFAULTS)
        self.text_config = text_config

        self.projector_input_dim = projector_input_dim
        self.projector_output_dim = projector_output_dim

        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vision_pad_token_id = vision_pad_token_id
        self.video_token_id = video_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
