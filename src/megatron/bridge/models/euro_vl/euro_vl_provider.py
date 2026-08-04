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

from dataclasses import dataclass, field
from typing import Any, List, Optional

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.euro_vl.modeling_euro_vl import EuroVLModel
from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class EuroVLModelProvider(GPTModelProvider):
    """Model provider for EuroVL (MoonViT + EuroLLM).

    Inherits all standard GPT/LLM fields from GPTModelProvider.
    VLM-specific fields are defined below.
    """

    # VLMs must not scatter embeddings across SP regions because image token
    # embeddings are inserted into the sequence after the embedding lookup.
    scatter_embedding_sequence_parallel: bool = False

    # Vendored MoonViT vision-encoder config (MoonViTConfig). In-repo class, so it
    # serializes cleanly into run_config.yaml (no transformers_modules namespace) and
    # is readable by the FLOPs calculator.
    vision_config: Optional[Any] = None

    # MoonViT hidden_size=1152, merge_kernel_size=[2,2] → 4*1152=4608 per merged token.
    projector_input_dim: int = 4608
    # EuroLLM hidden_size.
    projector_output_dim: int = 2048

    # Vision special tokens appended to EuroLLM's vocabulary (base vocab_size=128000).
    # Vocab is padded to the next multiple of 128 → 128128.
    image_token_id: int = 128000        # <image>           — per-image-token placeholder
    vision_start_token_id: int = 128001  # <|vision_start|>  — block start delimiter
    vision_end_token_id: int = 128002    # <|vision_end|>    — block end delimiter
    vision_pad_token_id: int = 128003    # <|vision_pad|>    — vision sequence padding
    video_token_id: int = 128004         # <video>           — per-video-frame placeholder

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


@dataclass
class Qwen3EuroVLModelProvider(EuroVLModelProvider):
    """Provider for :class:`Qwen3EuroVLModel` — MoonViT + projector on a Qwen3-VL M-RoPE LLM.

    Validation / oracle variant (see ``current.md``): swaps EuroVL's stock 1D-RoPE ``GPTModel``
    backbone for Qwen3-VL's proven **interleaved-M-RoPE** ``Qwen3VLGPTModel``. Point the standard
    architecture fields (``num_layers``, ``hidden_size``, ``vocab_size``, ``rotary_base``, …) at
    Qwen3 to mirror Qwen3-VL (or at EuroLLM later — same stack, different config).

    Key mrope config (overridden here): ``position_embedding_type='mrope'``, ``apply_rope_fusion``
    disabled (the fused RoPE kernel cannot do interleaved mrope), and ``mrope_section`` (channel
    split across t/h/w, must sum to ``head_dim // 2``).

    TODO: confirm ``get_transformer_block_with_experimental_attention_variant_spec`` yields a plain
    dense spec for this config; verify weight loading (Qwen3 LLM + MoonViT) and the bridge/recipe.
    """

    # Interleaved M-RoPE channel split across (t, h, w); must sum to head_dim // 2 (Qwen3-VL default).
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    # Interleaved mrope is applied by Qwen3VLMultimodalRotaryEmbedding, NOT the fused kernel.
    position_embedding_type: str = "mrope"
    apply_rope_fusion: bool = False
    # Qwen3-VL text-path config fields the reused Qwen3VLGPTModel/rope/attention read but that
    # are absent from the base GPTModelProvider:
    #   - apply_rotary_pos_emb_in_fp32: Qwen3-VL's LLM default is False (bf16 rope); only its
    #     vision tower uses fp32. Matches Qwen3-VL language behavior.
    #   - deepstack_visual_indexes: layers where Qwen injects multi-level vision features. We
    #     run deepstack-OFF (MoonViT single-point masked_scatter), so no injection layers.
    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=list)

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> "Any":
        """Instantiate the full Qwen3EuroVLModel and apply freeze flags."""
        from megatron.bridge.models.euro_vl.modeling_euro_vl import Qwen3EuroVLModel

        model = Qwen3EuroVLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
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
    ) -> Any:
        """Build Qwen3-VL's interleaved-M-RoPE ``Qwen3VLGPTModel`` as the language backbone.

        Reuses Qwen3-VL's spec + attention (``Qwen3VLSelfAttention``) and mrope GPT wholesale so no
        mrope math is reimplemented; deepstack is left off (default ``None`` in the forward).
        """
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )

        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
        from megatron.bridge.models.qwen_vl.qwen35_vl_provider import _patch_standard_attention_specs

        assert self.mrope_section is not None, "Qwen3EuroVLModelProvider requires mrope_section"

        block_spec = get_transformer_block_with_experimental_attention_variant_spec(self, vp_stage=vp_stage)
        _patch_standard_attention_specs(block_spec, Qwen3VLSelfAttention)

        return Qwen3VLGPTModel(
            config=self,
            transformer_layer_spec=block_spec,
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_length,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            position_embedding_type="mrope",
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            pg_collection=self._pg_collection,
            vp_stage=vp_stage,
        )