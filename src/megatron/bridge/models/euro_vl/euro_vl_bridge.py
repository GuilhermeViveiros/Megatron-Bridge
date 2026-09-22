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

import torch
from transformers import AutoConfig, AutoModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.euro_vl.configuration_euro_vl import EuroVLConfig
from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider
from megatron.bridge.models.euro_vl.modeling_euro_vl import EuroVLModel
from megatron.bridge.models.euro_vl.modeling_euro_vl_hf import EuroVLForConditionalGeneration
from megatron.bridge.models.euro_vl.utils import EUROLLM_PADDED_VOCAB_SIZE


# Register the custom HF classes so AutoConfig/AutoModel can load assembled ``euro_vl``
# checkpoints by model_type — required for the AutoBridge / pretrained_checkpoint path
# (repo pattern: cf. stepfun / bailing bridges). Covers both backbones (EuroLLM & Qwen3).
AutoConfig.register("euro_vl", EuroVLConfig, exist_ok=True)
AutoModel.register(EuroVLConfig, EuroVLForConditionalGeneration, exist_ok=True)


@MegatronModelBridge.register_bridge(
    source="EuroVLForConditionalGeneration",
    target=EuroVLModel,
    provider=EuroVLModelProvider,
    model_type="euro_vl",
)
class EuroVLBridge(MegatronModelBridge):
    """Bridge converting an EuroVL HF checkpoint to the Megatron EuroVL model."""

    def provider_bridge(self, hf_pretrained) -> EuroVLModelProvider:
        """Build the right EuroVL provider from an EuroVL HF config.

        Reads LLM architecture from ``text_config`` and VLM-specific fields
        (vision_config, token ids, tie_word_embeddings) from the top-level config.
        The backbone is dispatched on ``text_config.model_type`` (both backbones share
        the ``euro_vl`` HF model_type, so one bridge serves both):

          - ``"llama"`` (EuroLLM, default): :class:`EuroVLModelProvider`.
          - ``"qwen3"`` (Qwen3EuroVL oracle): :class:`Qwen3EuroVLModelProvider`, plus Qwen3's
            QK-norm (``qk_layernorm=True``).

        Both providers' defaults carry the interleaved-M-RoPE config (``mrope`` position
        embedding, ``mrope_section``, ``apply_rope_fusion=False``) and are never overridden
        here, so an imported checkpoint's run_config builds (and exports) the M-RoPE model.
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        is_qwen3 = getattr(text_config, "model_type", "llama") == "qwen3"

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        if is_qwen3:
            from megatron.bridge.models.euro_vl.euro_vl_provider import Qwen3EuroVLModelProvider

            provider = Qwen3EuroVLModelProvider(**provider_kwargs)
        else:
            provider = EuroVLModelProvider(**provider_kwargs)

        # Megatron settings common to both backbones (Llama-style dense GQA decoders).
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.hidden_dropout = 0.0
        provider.bias_activation_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.bias_dropout_fusion = True
        provider.rotary_percent = 1.0
        provider.add_bias_linear = False
        provider.add_qkv_bias = False

        if is_qwen3:
            # QK-norm per attention head (q_norm/k_norm weights).
            provider.qk_layernorm = True
        else:
            # EuroLLM's 128005 vocab is odd and cannot be split across TP ranks. Pad it to one
            # TP-independent size so an imported checkpoint loads at TP=1/2/4 (see utils.py); the
            # weight hooks below keep the HF side at its true 128005 rows.
            provider.should_pad_vocab = True
            provider.padded_vocab_size = EUROLLM_PADDED_VOCAB_SIZE

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

    # Vocab-row keys that differ in size between HF (real vocab) and Megatron (padded for TP).
    _VOCAB_HF_PARAMS = ("language_model.model.embed_tokens.weight", "language_model.lm_head.weight")

    def _eurollm_padded_vocab_size(self) -> int | None:
        """Padded vocab for the EuroLLM backbone, or None when padding does not apply.

        Returns None for the Qwen3 backbone (vocab 151936 already splits across TP) and whenever
        the HF config is unavailable, in which case the weights pass through untouched.
        """
        hf_config = getattr(self, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None)
        if text_config is None or getattr(text_config, "model_type", "llama") == "qwen3":
            return None
        return EUROLLM_PADDED_VOCAB_SIZE

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        """Zero-pad the embedding / output rows on import so they match the padded Megatron model.

        The Megatron decoder is built with a padded vocabulary (128512) while the HF checkpoint
        keeps the real one (128005), so these two tensors are the only ones whose row count
        differs. Padding here means the generic ``ColumnParallelMapping`` sees matching shapes and
        the shared conversion code needs no EuroVL special case.
        """
        hf_weights = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
        padded = self._eurollm_padded_vocab_size()
        if padded is None or not isinstance(hf_param, str) or hf_param not in self._VOCAB_HF_PARAMS:
            return hf_weights
        rows = hf_weights.shape[0]
        if rows >= padded:
            return hf_weights
        pad = torch.zeros((padded - rows, *hf_weights.shape[1:]), dtype=hf_weights.dtype, device=hf_weights.device)
        return torch.cat([hf_weights, pad], dim=0)

    def maybe_modify_converted_hf_weight(self, task, converted_weights_dict, hf_state_dict):
        """Drop the padding rows on export so the HF checkpoint matches its config's vocab_size.

        Without this an exported checkpoint carries 128512-row embeddings against a config that
        declares 128005, and ``transformers`` refuses to load it.
        """
        converted = super().maybe_modify_converted_hf_weight(task, converted_weights_dict, hf_state_dict)
        if self._eurollm_padded_vocab_size() is None:
            return converted
        vocab_size = self.hf_config.text_config.vocab_size
        return {
            key: (value[:vocab_size] if key in self._VOCAB_HF_PARAMS and value.shape[0] > vocab_size else value)
            for key, value in converted.items()
        }

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

        # Qwen3 backbone (Qwen3EuroVL oracle): per-head QK-norm weights, absent on Llama.
        # self.hf_config is set by the dispatch system before mapping_registry is called.
        hf_config = getattr(self, "hf_config", None)
        text_model_type = getattr(getattr(hf_config, "text_config", None), "model_type", "llama")
        if text_model_type == "qwen3":
            auto_mappings.update(
                {
                    "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "language_model.model.layers.*.self_attn.q_norm.weight",
                    "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "language_model.model.layers.*.self_attn.k_norm.weight",
                }
            )
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
                # Vision tower and projector are identical plain nn.Modules on both sides
                # (not TP-sharded) -> ReplicatedMapping copies 1:1 and replicates across TP.
                # AutoMapping would fail here: it cannot infer a parallelism type for plain
                # nn.Linear/Conv2d weights (e.g. multi_modal_projector.fc1.bias).
                ReplicatedMapping(megatron_param="vision_tower.**", hf_param="vision_tower.**"),
                ReplicatedMapping(megatron_param="multi_modal_projector.**", hf_param="multi_modal_projector.**"),
            ]
        )
        return MegatronMappingRegistry(*mappings)
