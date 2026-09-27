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

"""Standalone HF reference model for EuroVL (MoonViT + any supported LLM backbone).

This is the pure-transformers definition that ``AutoBridge`` converts to Megatron.
It bundles the vendored MoonViT vision tower, a 2-layer MLP projector, and
:class:`EuroVLTextForCausalLM` -- one interleaved-M-RoPE causal LM used for every
backbone (EuroLLM, Qwen3, ...), mirroring Megatron's backbone-generic
``_build_mrope_gpt_model``. Injects projected image features into the text embedding
stream at ``image_token_id`` positions via ``masked_scatter``.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from megatron.bridge.models.euro_vl.configuration_euro_vl import EuroVLConfig
from megatron.bridge.models.euro_vl.moonvit.modeling_moonvit import MoonVitPretrainedModel


# Supported LLM backbones, keyed by ``text_config.model_type`` -> whether the backbone has
# per-head QK-RMSNorm weights. Mirrors ``patch_qk_norm`` in ``euro_vl_provider`` and the
# bridge's QK-norm mapping dispatch. Adding a backbone = one entry here.
_BACKBONE_HAS_QK_NORM = {
    "qwen3": True,
    "llama": False,
}

# Default ``Qwen3VLTextRotaryEmbedding`` falls back to when ``rope_parameters`` has no
# ``mrope_section`` (true for every exported EuroVL config today).
_DEFAULT_MROPE_SECTION = [24, 20, 20]


class EuroVLTextForCausalLM(nn.Module):
    """Interleaved-M-RoPE causal LM shared by every EuroVL LLM backbone.

    The trunk is transformers' ``Qwen3VLTextModel`` for all backbones -- the only upstream
    text model that consumes 3D ``(t, h, w)`` position ids (it splits them into a 1D slice for
    the causal mask and the 3-channel tensor for the rotary embedding). Apart from QK-norm,
    its attention, MLP, norms and parameter names are identical to ``LlamaModel``, so for
    backbones without QK-norm the ``q_norm``/``k_norm`` modules are swapped for
    ``nn.Identity``, leaving exactly the Llama parameter set and computation. This matches the
    Megatron side, which builds the same ``Qwen3VLGPTModel`` for every backbone.

    Exposes the ``model.*`` / ``lm_head.*`` attribute layout of ``*ForCausalLM`` so the bridge's
    ``language_model.model.*`` key mapping applies unchanged.

    With ``tie_word_embeddings=True`` no separate ``lm_head`` parameter exists: logits use the
    live ``embed_tokens.weight``. A Parameter-sharing tie made in ``__init__`` would not survive
    ``from_pretrained`` (loading replaces the ``embed_tokens.weight`` Parameter, and this plain
    ``nn.Module`` never goes through ``tie_weights()``), and the on-disk tied checkpoint has no
    ``lm_head.weight`` entry anyway.
    """

    def __init__(self, config) -> None:
        super().__init__()
        model_type = getattr(config, "model_type", None)
        if model_type not in _BACKBONE_HAS_QK_NORM:
            raise ValueError(
                f"Unsupported EuroVL text backbone model_type={model_type!r}; "
                f"supported: {sorted(_BACKBONE_HAS_QK_NORM)}"
            )
        if getattr(config, "mlp_bias", False):
            raise ValueError("EuroVLTextForCausalLM requires mlp_bias=False (Qwen3VLTextMLP has no MLP bias)")
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        mrope_section = (getattr(config, "rope_parameters", None) or {}).get("mrope_section", _DEFAULT_MROPE_SECTION)
        if sum(mrope_section) != head_dim // 2:
            raise ValueError(
                f"mrope_section={mrope_section} must sum to head_dim // 2 = {head_dim // 2}; "
                "set rope_parameters['mrope_section'] in text_config to match the Megatron provider"
            )

        self.config = config
        self.model = Qwen3VLTextModel._from_config(config)
        if not _BACKBONE_HAS_QK_NORM[model_type]:
            for layer in self.model.layers:
                attn = layer.self_attn
                assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm"), (
                    "Qwen3VLTextAttention no longer exposes q_norm/k_norm; revisit the QK-norm removal"
                )
                attn.q_norm = nn.Identity()
                attn.k_norm = nn.Identity()
        self.lm_head = (
            None if config.tie_word_embeddings else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return self.lm_head if self.lm_head is not None else self.model.embed_tokens

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )
        if self.lm_head is not None:
            logits = self.lm_head(outputs.last_hidden_state)
        else:
            logits = torch.nn.functional.linear(outputs.last_hidden_state, self.model.embed_tokens.weight)

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


class EuroVLMultiModalProjector(nn.Module):
    """Two-layer MLP mapping merged MoonViT patch tokens into the LLM hidden space.

    Input:  [N, projector_input_dim]  (4 * 1152 = 4608 merged MoonViT tokens)
    Output: [N, projector_output_dim] (LLM hidden size, 2048)
    """

    def __init__(self, config: EuroVLConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.projector_input_dim, config.projector_output_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.projector_output_dim, config.projector_output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class EuroVLForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """EuroVL: MoonViT vision tower + MLP projector + M-RoPE LLM decoder."""

    config_class = EuroVLConfig
    base_model_prefix = "model"
    _no_split_modules = ["MoonVitEncoderLayer", "Qwen3VLTextDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: EuroVLConfig) -> None:
        super().__init__(config)
        self.vision_tower = MoonVitPretrainedModel(config.vision_config)
        self.multi_modal_projector = EuroVLMultiModalProjector(config)
        self.language_model = EuroVLTextForCausalLM(config.text_config)
        # Cached per-batch M-RoPE delta from the last full position-id computation; see
        # `_compute_position_ids`. Mirrors `Qwen2VLModel.rope_deltas`.
        self._rope_deltas = None
        self.post_init()

    # ------------------------------------------------------------------
    # Embedding plumbing (delegate to the language model)
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    # ------------------------------------------------------------------
    # Vision feature extraction
    # ------------------------------------------------------------------
    def _encode(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        """Run packed pixel patches through MoonViT + projector.

        MoonViT returns a list of [N_i, merge_area, hidden] tensors; flatten the
        merge dimension, concatenate, then project to LLM hidden space.
        ``grid_hws`` is the 2D per-image/per-frame grid [num, 2].
        """
        vision_dtype = next(self.vision_tower.parameters()).dtype
        features = self.vision_tower(pixel_values.to(vision_dtype), grid_hws)
        features = torch.cat(features, dim=0).flatten(1)
        return self.multi_modal_projector(features)

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Projected tokens for images. ``image_grid_thw`` is [num_images, 3] = (t=1, h, w);
        MoonViT is 2D, so the unit temporal dim is dropped before encoding."""
        return self._encode(pixel_values, image_grid_thw[:, 1:])

    def get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor) -> torch.Tensor:
        """Projected tokens for videos.

        Video frames are encoded as 2D images (MoonViT has no temporal dim), so the
        per-video ``(t, h, w)`` grid is expanded to ``t`` per-frame ``(h, w)`` rows
        before encoding.
        """
        frame_grids = []
        for t, h, w in video_grid_thw.tolist():
            frame_grids.extend([[h, w]] * t)
        frame_grid_hws = torch.tensor(frame_grids, dtype=video_grid_thw.dtype, device=video_grid_thw.device)
        return self._encode(pixel_values_videos, frame_grid_hws)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw).to(inputs_embeds.dtype)
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if pixel_values_videos is not None and video_grid_thw is not None:
            video_features = self.get_video_features(pixel_values_videos, video_grid_thw).to(inputs_embeds.dtype)
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        if position_ids is None:
            position_ids = self._compute_position_ids(
                input_ids, inputs_embeds, image_grid_thw, video_grid_thw, past_key_values, attention_mask
            )

        return self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

    def _compute_position_ids(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.FloatTensor,
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        past_key_values,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.LongTensor]:
        """3D M-RoPE position ids, reusing the exact same ``get_rope_index`` that Megatron's
        ``Qwen3EuroVLModel.forward`` calls for every backbone (see
        ``megatron.bridge.models.euro_vl.rope``).

        M-RoPE is one unified formulation that already produces correct positions for plain
        text (no vision), so unlike some reference HF VLMs we do NOT special-case "no vision
        present": Megatron's own forward calls ``get_rope_index`` unconditionally regardless
        of whether an item has vision content, and this mirrors that exactly.

        ``get_rope_index`` needs the full sequence to locate the vision block, so it can only
        run at prefill (``past_key_values`` empty). Cached decode steps get only the single new
        token -- there is no vision block left to re-derive positions from -- so we cheaply
        extend instead: cache the per-batch offset between the 3D positions and a flat token
        count at prefill, then add it to a simple running count on every later step. Mirrors
        ``Qwen2VLModel.rope_deltas`` in the installed ``transformers`` version.

        ``attention_mask`` matters for batched (padded) prefill: without it, ``get_rope_index``
        would treat left-padding tokens as real content and compute wrong positions for every
        row. With it, positions are computed only over each row's valid tokens (see
        ``get_rope_index``'s padded-batch path) -- the per-row delta below still works
        unmodified for padded rows, since ``past_length`` (from the shared KV cache) grows by
        the same padded width for every row regardless of its own content length.
        """
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()

        if past_length == 0:
            from megatron.bridge.models.euro_vl.rope import get_rope_index

            merge_cfg = getattr(self.config.vision_config, "merge_kernel_size", 2)
            spatial_merge_size = merge_cfg[0] if isinstance(merge_cfg, (list, tuple)) else merge_cfg
            position_ids = get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                spatial_merge_size=spatial_merge_size,
                image_token_id=self.config.image_token_id,
                video_token_id=self.config.video_token_id,
                vision_start_token_id=self.config.vision_start_token_id,
            )
            self._rope_deltas = (position_ids.amax(dim=(0, 2)) + 1 - input_ids.shape[1]).view(1, -1, 1)
            return position_ids

        if self._rope_deltas is None:
            return None

        batch_size, seq_length = inputs_embeds.shape[:2]
        position_ids = torch.arange(past_length, past_length + seq_length, device=inputs_embeds.device)
        position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).clone()
        return position_ids + self._rope_deltas.to(inputs_embeds.device)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        """Drop vision inputs once past the prefill step (mirrors ``Qwen2VLForConditionalGeneration``).

        The base ``GenerationMixin.prepare_inputs_for_generation`` has no concept of "vision inputs
        only matter on the first step" — it blindly auto-forwards any kwarg it doesn't recognize
        (including ``pixel_values``) into every step's ``model_inputs``. Without this override,
        ``forward`` re-runs the full MoonViT vision tower on every decode step even though the
        image's embeddings are already baked into the cached key/value states after step one.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs
