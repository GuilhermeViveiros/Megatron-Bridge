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

"""EuroVL bridge — weight mappings for the EuroLLM language model component.

EuroVL is trained from scratch: there is no pre-existing HF VLM checkpoint to
convert from.  This bridge provides:

  1. Llama-style weight mappings under the ``language_model.*`` prefix so that
     a pretrained EuroLLM-1.7B checkpoint can be imported into the LLM component
     of a freshly initialised EuroVL model.
  2. A ``provider_bridge()`` that builds an ``EuroVLModelProvider`` with the
     correct LLM architecture settings derived from an EuroLLM HF config.

Usage — load EuroLLM weights into EuroVL's language model::

    from megatron.bridge.models.euro_vl import EuroVLBridge
    bridge = EuroVLBridge()
    provider = bridge.provider_bridge(eurollm_hf_pretrained)
    # provider.vision_config must be set before calling provide()
"""

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider
from megatron.bridge.models.euro_vl.modeling_euro_vl import EuroVLModel
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(
    source="EuroLLMForCausalLM",  # string — no HF VLM class exists yet
    target=EuroVLModel,
    provider=EuroVLModelProvider,
    model_type="euro_vl",
)
class EuroVLBridge(MegatronModelBridge):
    """Bridge for loading EuroLLM weights into the EuroVL language model component.

    Weight mappings mirror LlamaBridge but with the ``language_model.`` prefix that
    Megatron uses for the LLM sub-module inside a VLM.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> EuroVLModelProvider:
        """Build EuroVLModelProvider from an EuroLLM-1.7B HF config.

        Reads LLM architecture from the HF config and applies Llama-specific
        Megatron settings.  ``vision_config`` is left as ``None`` and must be
        set by the caller before instantiating the model.
        """
        provider_kwargs = self.hf_config_to_provider_kwargs(hf_pretrained.config)
        provider = EuroVLModelProvider(**provider_kwargs)

        # Llama-specific Megatron settings (mirrors LlamaBridge.provider_bridge)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.hidden_dropout = 0.0
        provider.bias_activation_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.bias_dropout_fusion = True
        provider.apply_rope_fusion = True
        provider.rotary_percent = 1.0
        # Extend vocab for <image> token; Megatron pads internally via make_vocab_size_divisible_by
        provider.vocab_size = 128001

        # vision_config must be provided by the caller
        provider.vision_config = None

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Llama weight mappings under the language_model.* prefix used in EuroVL."""
        auto_mappings = {
            "language_model.embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.norm.weight",
            # TE implementation layer norms
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # Local (non-TE) implementation layer norms
            "language_model.decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Attention output and MLP down projections
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }
        mappings = [AutoMapping(megatron_param=m, hf_param=h) for m, h in auto_mappings.items()]
        mappings.extend(
            [
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mappings)
