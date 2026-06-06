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

"""EuroVL bridge — converts the EuroVL HF checkpoint to Megatron.

Standard HF -> Megatron bridge for ``EuroVLForConditionalGeneration`` (assembled
by ``eurollm_bridge.py``). It maps three components:

  - ``language_model.*``         (Megatron GPT decoder)  <-  EuroLLM (Llama) weights
  - ``vision_tower.*``           (vendored MoonViT)       <-  1:1 copy (same module)
  - ``multi_modal_projector.*``  (MLP projector)          <-  1:1 copy (same module)

The vision tower and projector are the *same* nn.Modules on both sides, so they
copy verbatim; only the Llama decoder needs the usual QKV/GatedMLP fusions.
"""

from typing import List

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider
from megatron.bridge.models.euro_vl.modeling_euro_vl import EuroVLModel


@MegatronModelBridge.register_bridge(
    source="EuroVLForConditionalGeneration",
    target=EuroVLModel,
    provider=EuroVLModelProvider,
    model_type="euro_vl",
)
class EuroVLBridge(MegatronModelBridge):
    """Bridge converting an EuroVL HF checkpoint to the Megatron EuroVL model."""

    def provider_bridge(self, hf_pretrained) -> EuroVLModelProvider:
        """Build EuroVLModelProvider from an EuroVL HF config.

        Reads LLM architecture from ``text_config`` and VLM-specific fields
        (vision_config, token ids, tie_word_embeddings) from the top-level config.
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = EuroVLModelProvider(**provider_kwargs)

        # Llama / EuroLLM Megatron settings (mirror LlamaBridge).
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.hidden_dropout = 0.0
        provider.bias_activation_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.bias_dropout_fusion = True
        provider.apply_rope_fusion = True
        provider.rotary_percent = 1.0
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False

        # tie_word_embeddings lives on the TOP-LEVEL config (EuroLLM is untied).
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # Vision + projector + token ids from the top-level config.
        provider.vision_config = hf_config.vision_config
        provider.projector_input_dim = hf_config.projector_input_dim
        provider.projector_output_dim = hf_config.projector_output_dim
        provider.image_token_id = hf_config.image_token_id
        provider.vision_start_token_id = hf_config.vision_start_token_id
        provider.vision_end_token_id = hf_config.vision_end_token_id
        provider.vision_pad_token_id = hf_config.vision_pad_token_id
        provider.video_token_id = hf_config.video_token_id

        return provider

    def build_conversion_tasks(self, hf_pretrained, megatron_model) -> List[WeightConversionTask]:
        # Defensive: drop any unmapped (None) tasks so base iteration doesn't crash.
        tasks = super().build_conversion_tasks(hf_pretrained, megatron_model)
        return [t for t in tasks if t is not None]

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Weight mappings: EuroLLM (Llama) decoder + 1:1 vision tower & projector.

        HF side nests the language model under ``language_model.`` (a
        ``LlamaForCausalLM``), so HF keys are ``language_model.model.*`` /
        ``language_model.lm_head.weight``.
        """
        auto_mappings = {
            "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "language_model.output_layer.weight": "language_model.lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
            # TE implementation layer norms
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            # Local (non-TE) implementation layer norms
            "language_model.decoder.layers.*.input_layernorm.weight": "language_model.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            # Attention output and MLP down projections
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
        }
        mappings = [AutoMapping(megatron_param=m, hf_param=h) for m, h in auto_mappings.items()]
        mappings.extend(
            [
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.model.layers.*.self_attn.q_proj.weight",
                    k="language_model.model.layers.*.self_attn.k_proj.weight",
                    v="language_model.model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="language_model.model.layers.*.mlp.gate_proj.weight",
                    up="language_model.model.layers.*.mlp.up_proj.weight",
                ),
                # Vision tower and projector are identical modules on both sides: copy 1:1.
                AutoMapping(megatron_param="vision_tower.**", hf_param="vision_tower.**"),
                AutoMapping(megatron_param="multi_modal_projector.**", hf_param="multi_modal_projector.**"),
            ]
        )
        return MegatronMappingRegistry(*mappings)
