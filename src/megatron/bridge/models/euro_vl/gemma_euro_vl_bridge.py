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

"""Bridge for the Gemma2-backbone EuroVL experiment.

Converts a ``GemmaEuroVLForConditionalGeneration`` HF checkpoint (Tower-Plus-2B +
MoonViT + projector, built by ``sanity_check/gemma_euro_vl_assembly.py``) to Megatron.

Registered under its own ``model_type="gemma_euro_vl"`` so it never collides with
:class:`EuroVLBridge`, which serves the EuroLLM and Qwen3 backbones under ``euro_vl``.

Weight layout is Gemma2's, so the decoder mappings follow
``megatron.bridge.models.gemma.gemma2_bridge`` (notably the two *extra* norms per layer
that Llama/Qwen3 lack), re-rooted under EuroVL's ``language_model.`` prefix. The vision
tower and projector are the same nn.Modules on both sides and copy 1:1.
"""

from typing import List

from transformers import AutoConfig, AutoModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.euro_vl.gemma_euro_vl_hf import (
    GemmaEuroVLConfig,
    GemmaEuroVLForConditionalGeneration,
)
from megatron.bridge.models.euro_vl.gemma_euro_vl_provider import GemmaEuroVLModelProvider
from megatron.bridge.models.euro_vl.modeling_euro_vl import EuroVLModel


# Let AutoConfig/AutoModel resolve assembled ``gemma_euro_vl`` checkpoints by model_type,
# as required by the AutoBridge / pretrained_checkpoint path.
AutoConfig.register("gemma_euro_vl", GemmaEuroVLConfig, exist_ok=True)
AutoModel.register(GemmaEuroVLConfig, GemmaEuroVLForConditionalGeneration, exist_ok=True)


@MegatronModelBridge.register_bridge(
    source="GemmaEuroVLForConditionalGeneration",
    target=EuroVLModel,
    provider=GemmaEuroVLModelProvider,
    model_type="gemma_euro_vl",
)
class GemmaEuroVLBridge(MegatronModelBridge):
    """Bridge converting a Gemma2-backbone EuroVL HF checkpoint to Megatron."""

    def provider_bridge(self, hf_pretrained) -> GemmaEuroVLModelProvider:
        """Build a :class:`GemmaEuroVLModelProvider` from a ``gemma_euro_vl`` HF config."""
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)

        # Gemma2-specific fields must go in as constructor kwargs, not post-hoc setattr:
        # the provider's __post_init__ derives softmax_scale from query_pre_attn_scalar,
        # and would not see a later assignment.
        provider_kwargs["query_pre_attn_scalar"] = text_config.query_pre_attn_scalar
        provider_kwargs["final_logit_softcapping"] = text_config.final_logit_softcapping

        # HF `sliding_window` counts the query token itself; TE's window_size is the
        # left-context width, hence the -1. skip_freq=2 reproduces Gemma2's alternating
        # pattern (SWA on HF layer_idx 0, 2, 4, ...).
        sliding_window = getattr(text_config, "sliding_window", None)
        if sliding_window:
            provider_kwargs["window_size"] = (sliding_window - 1, 0)
            provider_kwargs["window_attn_skip_freq"] = 2
        else:
            provider_kwargs["window_size"] = None
            provider_kwargs["window_attn_skip_freq"] = None

        provider = GemmaEuroVLModelProvider(**provider_kwargs)

        # Gemma2 decoder settings. NOTE: attn_logit_softcapping is deliberately absent —
        # it lives in the eager Gemma2DotProductAttention, which cannot run with packed
        # sequences. See gemma_euro_vl_provider's module docstring.
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.layernorm_zero_centered_gamma = True
        provider.position_embedding_type = "rope"
        provider.apply_rope_fusion = True
        provider.rotary_percent = 1.0
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.bias_dropout_fusion = True
        # Gemma2's activation is gelu_pytorch_tanh (MCore `fast_gelu`), which the bias+
        # activation fusion does not support (it accepts only gelu / swiglu / quick_geglu,
        # and MCore raises on anything else). Stock Gemma2ModelProvider likewise leaves it off.
        provider.bias_activation_fusion = False

        # tie_word_embeddings lives on the TOP-LEVEL config (Gemma2 ships tied).
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", True)

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
        """Drop unmapped (None) tasks so base iteration doesn't crash."""
        tasks = super().build_conversion_tasks(hf_pretrained, megatron_model)
        return [t for t in tasks if t is not None]

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Gemma2 decoder + 1:1 vision tower & projector.

        Gemma2 carries four norms per layer. ``input_layernorm`` and
        ``pre_feedforward_layernorm`` are folded into the TE column-parallel inputs;
        ``post_attention_layernorm`` and ``post_feedforward_layernorm`` land on the
        ``post_layernorm`` of the row-parallel outputs (``TERowParallelLinearLayerNorm``).

        No ``lm_head`` mapping: Gemma2 ties its output head to the embedding, so there is
        no separate Megatron output-layer parameter to fill. RMSNorm weights map verbatim
        because ``layernorm_zero_centered_gamma=True`` makes TENorm add the 1 at compute
        time, matching how HF stores Gemma2's gammas.
        """
        auto_mappings = {
            "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
            # Pre-attention / pre-MLP norms (fused into the TE input linears).
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.pre_feedforward_layernorm.weight",
            # Gemma2's extra post-attention / post-MLP norms.
            "language_model.decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": "language_model.model.layers.*.post_feedforward_layernorm.weight",
            # Attention output and MLP down projections.
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
                # Vision tower and projector are identical plain nn.Modules on both sides.
                ReplicatedMapping(megatron_param="vision_tower.**", hf_param="vision_tower.**"),
                ReplicatedMapping(megatron_param="multi_modal_projector.**", hf_param="multi_modal_projector.**"),
            ]
        )
        return MegatronMappingRegistry(*mappings)
