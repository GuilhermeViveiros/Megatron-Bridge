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

from megatron.bridge.models.gpt_provider import GPTModelProvider


def _patch_core_attention_specs(block_spec: "Any") -> int:
    """Swap every layer's ``core_attention`` for the branch-isolating variant.

    Walks the same spec shapes as ``qwen35_vl_provider._patch_standard_attention_specs`` (a block
    spec with ``layer_specs``, or a single ``ModuleSpec``, plus a nested MTP layer spec) and only
    touches specs that actually have a ``self_attention.submodules.core_attention``.

    Args:
        block_spec: Transformer block spec (or one layer spec) to patch in place.

    Returns:
        Number of layer specs patched, so callers can assert the flag had an effect.
    """
    from megatron.core.transformer.spec_utils import ModuleSpec

    from megatron.bridge.models.euro_vl.branch_attention import BranchIsolatedDotProductAttention

    if block_spec is None:
        return 0
    if hasattr(block_spec, "layer_specs"):
        return sum(_patch_core_attention_specs(spec) for spec in block_spec.layer_specs)
    if not isinstance(block_spec, ModuleSpec):
        return 0

    submodules = getattr(block_spec, "submodules", None)
    if submodules is None:
        return 0

    patched = 0
    if hasattr(submodules, "mtp_model_layer"):
        patched += _patch_core_attention_specs(submodules.mtp_model_layer)

    #     TransformerLayer
    #  └─ self_attention: SelfAttention
    #       ├─ linear_qkv        (Q/K/V projection)
    #       ├─ core_attention: TEDotProductAttention   ← softmax(QKᵀ)·V, the only part that sees the mask (replace by BranchIsolatedDotProductAttention, built for message-tree protocol)
    #       └─ linear_proj       (output projection)
    attn_spec = getattr(submodules, "self_attention", None)
    attn_submodules = getattr(attn_spec, "submodules", None) if attn_spec is not None else None
    if attn_submodules is not None and hasattr(attn_submodules, "core_attention"):
        attn_submodules.core_attention = BranchIsolatedDotProductAttention
        patched += 1
    return patched


def _check_message_tree_support(provider: "Any") -> None:
    """Reject parallel/runtime settings that would silently break message-tree branch isolation.

    Runs before any spec is built, so misconfigurations fail fast. The branch mask spans the whole
    packed sequence and rides on packed_seq_params: context parallelism shards q/k/v but not the
    mask, and TE-scoped CUDA graphs drop packed_seq_params.

    Args:
        provider: Model provider; only checked when ``message_tree_attention`` is set.

    Raises:
        ValueError: If the flag is combined with context parallelism, CUDA graphs, attention dropout or
            bidirectional image attention.
    """
    if not getattr(provider, "message_tree_attention", False):
        return
    if (getattr(provider, "context_parallel_size", 1) or 1) > 1:
        raise ValueError("message_tree_attention does not support context_parallel_size > 1")
    if getattr(provider, "cuda_graph_impl", "none") not in (None, "none"):
        raise ValueError("message_tree_attention does not support CUDA graphs (cuda_graph_impl != 'none')")
    # The FlexAttention path implements neither, so reject them here instead of at the first tree pack.
    if (getattr(provider, "attention_dropout", 0.0) or 0.0) > 0.0:
        raise ValueError("message_tree_attention requires attention_dropout == 0 (the recipes use 0.0)")
    if getattr(provider, "use_bidirectional_image_attention", False):
        raise ValueError("message_tree_attention does not support use_bidirectional_image_attention")


def _build_mrope_gpt_model(
    provider: "Any",
    pre_process: Optional[bool],
    post_process: Optional[bool],
    vp_stage: Optional[int],
    patch_qk_norm: bool,
) -> "Any":
    """Build Qwen3-VL's interleaved-M-RoPE ``Qwen3VLGPTModel`` as a language backbone.


    Shared by every EuroVL M-RoPE provider (Qwen3, EuroLLM, ...) — ``Qwen3VLGPTModel`` and
    ``Qwen3VLSelfAttention`` are already config-driven (mrope_section, rotary_base,
    qk_layernorm all come from ``provider``), so no per-backbone subclassing of the model
    itself is needed, only different config values and whether QK-norm must be patched in.

    ``patch_qk_norm`` selects the attention variant: True (Qwen3) swaps in
    ``Qwen3VLSelfAttention`` for its q_norm/k_norm weights; False (e.g. EuroLLM, which has
    none) leaves the base dense spec's plain ``SelfAttention`` — verified empirically to
    produce the exact same parameter names either way when ``qk_layernorm=False``.
    """
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )

    assert provider.mrope_section is not None, f"{type(provider).__name__} requires mrope_section"

    _check_message_tree_support(provider)
    block_spec = get_transformer_block_with_experimental_attention_variant_spec(provider, vp_stage=vp_stage)
    if getattr(provider, "message_tree_attention", False):
        from megatron.bridge.utils.common_utils import print_rank_0

        n_patched = _patch_core_attention_specs(block_spec)
        if n_patched == 0:
            # Fail loudly: training on with the flag set but nothing patched would leave branches
            # attending to each other while the run claims they are isolated.
            raise RuntimeError(
                "message_tree_attention=True but no layer spec has a core_attention slot to patch; "
                "branch isolation would silently be off. Did the Megatron spec layout change?"
            )
        print_rank_0(
            f"message_tree_attention=True: core_attention -> BranchIsolatedDotProductAttention "
            f"in {n_patched} layer spec(s)"
        )
    if patch_qk_norm:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
        from megatron.bridge.models.qwen_vl.qwen35_vl_provider import _patch_standard_attention_specs

        _patch_standard_attention_specs(block_spec, Qwen3VLSelfAttention)

    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel

    return Qwen3VLGPTModel(
        config=provider,
        transformer_layer_spec=block_spec,
        vocab_size=provider.vocab_size,
        max_sequence_length=provider.seq_length,
        pre_process=True if pre_process is None else pre_process,
        post_process=True if post_process is None else post_process,
        position_embedding_type="mrope",
        rotary_percent=provider.rotary_percent,
        rotary_base=provider.rotary_base,
        share_embeddings_and_output_weights=provider.share_embeddings_and_output_weights,
        pg_collection=provider._pg_collection,
        vp_stage=vp_stage,
    )


@dataclass
class EuroVLModelProvider(GPTModelProvider):
    """Model provider for EuroVL (MoonViT + EuroLLM), on an interleaved-M-RoPE ``Qwen3VLGPTModel``
    backbone.

    Inherits all standard GPT/LLM fields from GPTModelProvider. VLM-specific fields are defined
    below. ``qk_layernorm=False``: EuroLLM has no QK-norm weights to load (Qwen3VLSelfAttention
    only allocates q/k_layernorm submodules when ``config.qk_layernorm`` is set — see
    ``megatron.core.transformer.attention.SelfAttention.__init__``), which makes
    ``_build_mrope_gpt_model``'s ``patch_qk_norm=False`` path produce the exact same
    15-parameter-name Megatron module a plain 1D ``GPTModel`` would (verified empirically) --
    so an EuroVL checkpoint imported before M-RoPE loads in unmodified; no new HF assembly or
    bridge-dispatch change is needed.

    ``mrope_section=[24, 20, 20]`` sums to 64 = ``head_dim // 2`` (EuroLLM-1.7B's ``head_dim``
    is 128).

    HF export/inference uses ``EuroVLTextForCausalLM`` (``modeling_euro_vl_hf``), the HF
    counterpart of this backbone: ``Qwen3VLTextModel`` with its QK-norm swapped for identity.
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
    image_token_id: int = 128000  # <image>           — per-image-token placeholder
    vision_start_token_id: int = 128001  # <|vision_start|>  — block start delimiter
    vision_end_token_id: int = 128002  # <|vision_end|>    — block end delimiter
    vision_pad_token_id: int = 128003  # <|vision_pad|>    — vision sequence padding
    video_token_id: int = 128004  # <video>           — per-video-frame placeholder

    # Attention strategy for image token positions in the LLM decoder.
    # False (default): pure causal masking — simplest baseline.
    # True: bidirectional attention within each image's token block.
    use_bidirectional_image_attention: bool = False

    # Message-tree branch isolation (docs/models/euro_vl/message-tree-packing.md). When True,
    # every layer's core_attention is swapped for BranchIsolatedDotProductAttention: packs whose
    # samples carry `subsegment_ids` run FlexAttention with the branch mask (each QA branch sees
    # the shared video prefix and itself, never another branch), and every other pack keeps the
    # untouched TE fused kernel. Opt-in: without it, tree samples train as plain multi-turn
    # conversations, i.e. branches can read each other.
    message_tree_attention: bool = False

    # Freeze flags for two-stage training.
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # Interleaved M-RoPE channel split across (t, h, w); must sum to head_dim // 2.
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    # Interleaved mrope is applied by Qwen3VLMultimodalRotaryEmbedding, NOT the fused kernel.
    position_embedding_type: str = "mrope"
    apply_rope_fusion: bool = False
    # Text-path config fields the reused Qwen3VLGPTModel/rope/attention read but that are
    # absent from the base GPTModelProvider:
    #   - apply_rotary_pos_emb_in_fp32: LLM rope stays bf16; only the vision tower uses fp32.
    #   - deepstack_visual_indexes: layers where multi-level vision features get injected. We
    #     run deepstack-OFF (MoonViT single-point masked_scatter), so no injection layers.
    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=list)
    # EuroLLM has no QK-norm weights (see class docstring).
    qk_layernorm: bool = False

    def provide(
        self,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> "Any":
        """Instantiate the full EuroVL+M-RoPE model and apply freeze flags."""
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
    ) -> "Any":
        """Build the M-RoPE ``Qwen3VLGPTModel`` over EuroLLM's architecture fields.

        No QK-norm patch: EuroLLM has none, and with ``qk_layernorm=False`` the base dense
        spec's plain ``SelfAttention`` already matches what ``Qwen3VLSelfAttention`` would
        produce (see class docstring).
        """
        assert not self.qk_layernorm, "EuroLLM has no QK-norm weights; qk_layernorm must be False"
        return _build_mrope_gpt_model(self, pre_process, post_process, vp_stage, patch_qk_norm=False)


@dataclass
class Qwen3EuroVLModelProvider(EuroVLModelProvider):
    """Provider for :class:`Qwen3EuroVLModel` — MoonViT + projector on a Qwen3-1.7B M-RoPE LLM.

    Validation / oracle variant (see ``current.md``): points the standard architecture fields
    (``num_layers``, ``hidden_size``, ``vocab_size``, ``rotary_base``, …) at Qwen3-1.7B, and
    enables QK-norm (``qk_layernorm=True``), which Qwen3 has and the base provider's backbone
    does not.

    Key mrope config here matches the base provider's (``position_embedding_type='mrope'``,
    ``apply_rope_fusion`` disabled, ``mrope_section`` channel split across t/h/w summing to
    ``head_dim // 2``) — only ``qk_layernorm`` and the architecture fields differ.
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
        return _build_mrope_gpt_model(self, pre_process, post_process, vp_stage, patch_qk_norm=True)
