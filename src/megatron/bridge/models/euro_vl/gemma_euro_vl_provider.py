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

"""Gemma2 backbone for EuroVL — standalone experiment provider.

Swaps EuroVL's LLM backbone for a Gemma2 decoder (Tower-Plus-2B) while keeping the
MoonViT vision tower and projector unchanged. This exists to isolate *the LLM backbone*
as a variable against the Qwen3 EuroVL oracle: TowerVision uses Tower-Plus-2B, so this
makes our stack an architectural near-replica of theirs, trained in our own framework.

Self-contained on purpose: nothing here is imported by the EuroLLM/Qwen3 EuroVL paths,
so the whole experiment can be deleted without touching them.

Fidelity to stock Gemma2
------------------------
Reused verbatim from :mod:`megatron.bridge.models.gemma.gemma2_provider`: the
``sqrt(hidden_size)`` embedding scaling, the final-logit softcapping output layer, and
the post-attention / post-MLP norm linears that give Gemma2 its four-norms-per-layer
layout.

One deliberate deviation: **attention logit softcapping is dropped.** Gemma2's
``attn_logit_softcapping=50.0`` is implemented in ``Gemma2DotProductAttention``, an
unfused eager SDPA that hard-asserts ``packed_seq_params is None``. EuroVL trains with
energon fill-to-``seq_length`` packing (``cu_seqlens`` -> ``PackedSeqParams``), so that
module cannot be used here at all. We therefore run the standard TE attention, which
keeps packing, and lose only the softcapping. This mirrors upstream HF, where Gemma2's
attention softcapping is eager-only and silently disabled under FlashAttention-2/SDPA —
the path most Gemma2 finetuning actually runs on.

Everything else attention-side is preserved exactly, because MCore's TE attention
supports it natively: ``query_pre_attn_scalar`` via ``softmax_scale``, and Gemma2's
alternating sliding-window pattern via ``window_size`` + ``window_attn_skip_freq``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from megatron.core.activations import fast_gelu
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider
from megatron.bridge.models.euro_vl.modeling_euro_vl import EuroVLModel
from megatron.bridge.models.gemma.gemma2_provider import (
    Gemma2OutputLayer,
    TERowParallelLinearLayerNorm,
)
from megatron.bridge.models.gemma.modules import EmbeddingScalingMixin, extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider


def gemma_euro_vl_layer_spec(config: "GemmaEuroVLModelProvider") -> ModuleSpec:
    """Gemma2 layer layout on TE attention (packing-compatible).

    Identical to :func:`megatron.bridge.models.gemma.gemma2_provider.gemma2_layer_spec`
    except that ``core_attention`` is :class:`TEDotProductAttention` rather than the
    eager ``Gemma2DotProductAttention``. See the module docstring for why.

    ``TERowParallelLinearLayerNorm`` on both ``linear_proj`` and ``linear_fc2`` is what
    supplies Gemma2's ``post_attention_layernorm`` / ``post_feedforward_layernorm``; the
    ``TELayerNormColumnParallelLinear`` inputs supply ``input_layernorm`` /
    ``pre_feedforward_layernorm``.
    """
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinearLayerNorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinearLayerNorm,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


@dataclass
class GemmaEuroVLModelProvider(EuroVLModelProvider):
    """Provider for EuroVL on a Gemma2 (Tower-Plus-2B) decoder.

    Inherits EuroVL's vision/projector/freeze fields from :class:`EuroVLModelProvider`
    and layers Gemma2's architecture defaults on top. Architecture values are filled from
    the HF config by the bridge; the defaults here match Tower-Plus-2B so the provider is
    usable standalone in a recipe.
    """

    # --- Gemma2 architecture (mirrors Gemma2ModelProvider) ---
    normalization: str = "RMSNorm"
    activation_func: Callable = fast_gelu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 10000.0
    rotary_percent: float = 1.0
    # Gemma2 stores RMSNorm gamma centered at zero; TENorm adds the 1 at compute time, so
    # HF weights map across verbatim (no (w-1) rewrite in the bridge).
    layernorm_zero_centered_gamma: bool = True
    # Gemma2 ties input embeddings to the output head.
    share_embeddings_and_output_weights: bool = True
    # Gemma2-2B: hidden 2304 but head_dim 256 (so q_proj is 2048x2304, not square).
    kv_channels: int = 256

    # Gemma2 alternates sliding-window and full-attention layers. is_layer_window_attention
    # uses a 1-indexed layer_number and applies SWA when `layer_number % freq != 0`, so
    # freq=2 puts SWA on layers 1,3,5,... == HF layer_idx 0,2,4,... == Gemma2's
    # `is_sliding = not bool(layer_idx % 2)`. HF `sliding_window=4096` counts the query
    # token itself, hence 4095 tokens of left context.
    window_size: Optional[tuple[int, int]] = (4095, 0)
    window_attn_skip_freq: Optional[int] = 2

    # Gemma2 scales attention logits by 1/sqrt(query_pre_attn_scalar) rather than
    # 1/sqrt(head_dim); for Gemma2-2B these differ (224 vs 256).
    query_pre_attn_scalar: int = 224
    final_logit_softcapping: float = 30.0

    # Tower-Plus-2B hidden size (MoonViT merged-token dim -> Gemma2 hidden).
    projector_output_dim: int = 2304

    transformer_layer_spec: Union[ModuleSpec, Callable[[Any], ModuleSpec]] = gemma_euro_vl_layer_spec

    def __post_init__(self) -> None:
        """Derive the Gemma2 attention scale unless one was set explicitly."""
        super().__post_init__()
        # TE consumes softmax_scale directly. TransformerConfig only auto-derives this
        # under MuP, so setting it here is safe and will not be overwritten.
        if self.softmax_scale is None:
            self.softmax_scale = float(self.query_pre_attn_scalar) ** -0.5

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> EuroVLModel:
        """Instantiate the full EuroVL model (plain, no M-RoPE) and apply freeze flags.

        Explicit override, NOT inherited from :class:`EuroVLModelProvider`: that base class
        builds an M-RoPE ``Qwen3EuroVLModel``, which Gemma2's decoder (built by
        :meth:`provide_language_model` below, on its own non-mrope layer spec) cannot consume.
        """
        model = EuroVLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
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
        """Build the Gemma2 decoder, applying Gemma2's embedding and output-head extensions.

        Calls ``GPTModelProvider.provide`` directly (NOT ``super().provide_language_model``,
        which is :class:`EuroVLModelProvider`'s M-RoPE builder) -- Gemma2 uses its own
        ``transformer_layer_spec`` (``gemma_euro_vl_layer_spec``) and 1D RoPE.
        """
        model = GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        # Gemma2 scales embeddings by sqrt(hidden_size).
        if is_vp_first_stage(
            vp_stage=vp_stage, vp_size=self.virtual_pipeline_model_parallel_size
        ) and is_pp_first_stage(self._pg_collection.pp):
            extend_instance(model.embedding, EmbeddingScalingMixin)

        # Gemma2 softcaps final logits (independent of attention, so retained here).
        if is_vp_last_stage(vp_stage=vp_stage, vp_size=self.virtual_pipeline_model_parallel_size) and is_pp_last_stage(
            self._pg_collection.pp
        ):
            extend_instance(model.output_layer, Gemma2OutputLayer)

        return model
