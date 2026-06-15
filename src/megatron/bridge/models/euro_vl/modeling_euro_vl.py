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

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from megatron.bridge.models.euro_vl.moonvit.modeling_moonvit import MoonVitPretrainedModel
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


class EuroVLProjector(nn.Module):
    """Two-layer MLP projector mapping MoonViT patch tokens to EuroLLM hidden space.

    Input:  [N, projector_input_dim]  — merged patch tokens from MoonViT
    Output: [N, projector_output_dim] — tokens in EuroLLM embedding space
    """

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class EuroVLModel(MegatronModule):
    """EuroVL: MoonViT (HF) + MLP projector + EuroLLM (Megatron-Core GPT).

    Architecture overview:
    - Vision tower:   MoonVitPretrainedModel loaded via AutoModel
    - Projector:      EuroVLProjector  (4608 → 2048)
    - Language model: Megatron GPT decoder (EuroLLM-1.7B)

    MoonViT outputs List[Tensor[N_i, 4*hidden_size]] after its internal patch merger
    (merge_kernel_size=[2,2], hidden_size=1152 → 4608 per merged token).  Each N_i
    depends on the input image resolution, enabling native-resolution encoding.
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self._vision_dtype = config.params_dtype

        if pre_process:
            # Build the vendored MoonViT vision tower from the in-repo config object
            # (no AutoConfig/AutoModel, no trust_remote_code). Force flash-attn varlen
            # (cu_seqlens, O(N) per image) — HF otherwise auto-selects the sdpa path, which
            # materializes a dense [N, N] patch mask (O(N^2)) and OOMs at long packed lengths.
            config.vision_config._attn_implementation = "flash_attention_2"
            self.vision_tower = MoonVitPretrainedModel(config.vision_config).to(config.params_dtype)
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)

            self.multi_modal_projector = EuroVLProjector(
                input_dim=config.projector_input_dim,
                output_dim=config.projector_output_dim,
            )
            hook_hf_module_setattr_for_tp_grad_sync(self.multi_modal_projector)

        self.language_model = config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    def set_input_tensor(self, input_tensor) -> None:
        """Pass pipeline-parallel input tensor to the language model."""
        self.language_model.set_input_tensor(input_tensor)

    def freeze(
        self,
        *,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        """Set requires_grad=False on selected sub-modules."""
        if freeze_language_model and hasattr(self, "language_model"):
            for p in self.language_model.parameters():
                p.requires_grad = False
        if freeze_vision_model and hasattr(self, "vision_tower"):
            for p in self.vision_tower.parameters():
                p.requires_grad = False
        if freeze_vision_projection and hasattr(self, "multi_modal_projector"):
            for p in self.multi_modal_projector.parameters():
                p.requires_grad = False

    def _compute_bidirectional_attention_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Build a causal mask with bidirectional attention within each vision block.

        A vision block spans from ``<|vision_start|>`` to ``<|vision_end|>`` inclusive,
        covering the delimiter tokens and all ``<image>`` placeholders between them.
        All positions within the same vision block attend to each other bidirectionally;
        all other positions follow standard causal masking.

        Only used when config.use_bidirectional_image_attention=True.  Default is pure
        causal (attention_mask=None), which lets Megatron Core handle masking normally.

        @TODO: Verify attention mask sign convention against Megatron Core's expectation
               (True = blocked vs True = allowed) before enabling this code path.
        """
        if not self.pre_process:
            return None
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len), device=input_ids.device))

        # Assign a unique block index to every token inside a vision block
        # (<|vision_start|>, <image>*N, <|vision_end|>).  Tokens outside any
        # vision block get block_idx=0 and are excluded from bidirectional attention.
        vision_start = input_ids == self.config.vision_start_token_id
        vision_end = input_ids == self.config.vision_end_token_id
        # Cumulative count of opened blocks minus closed blocks gives in-block flag.
        opened = torch.cumsum(vision_start, dim=-1)
        closed = torch.cumsum(
            vision_end.roll(1, dims=-1).masked_fill(torch.arange(seq_len, device=input_ids.device) == 0, False), dim=-1
        )
        in_block = (opened - closed) > 0
        # Use opened as block ID (unique per vision block).
        block_idx = opened * in_block  # 0 outside blocks, ≥1 inside

        bidirectional = torch.logical_and(
            block_idx[:, None, :] == block_idx.unsqueeze(-1),
            block_idx.unsqueeze(-1) > 0,
        )
        return ~torch.logical_or(causal_mask, bidirectional.unsqueeze(1))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass combining MoonViT vision encoding with EuroLLM language model.

        Args:
            input_ids: Token ids, shape [B, T].  Must contain image_token_id placeholders.
            pixel_values: Packed pixel patches for MoonViT, shape [total_patches, C, H, W].
            image_grid_thw: Per-image grid dimensions, shape [num_images, 3] (t, height, width
                in patches). MoonViT is 2D-only so t=1; the temporal dim is stripped before the
                vision tower. Kept in the codebase-wide ``image_grid_thw`` form for the shared
                FLOPs counter.
            labels: Shifted token ids for cross-entropy loss.
            loss_mask: Boolean mask selecting positions that contribute to the loss.

        Returns:
            Tuple of (model_output, loss_mask) where loss_mask may be CP-sliced.
        """
        # import pdb; pdb.set_trace()
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [decoder_seq_len, b, h_language]

                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [b, decoder_seq_len, h_language]

            if pixel_values is not None and image_grid_thw is not None:
                pixel_values = pixel_values.to(self._vision_dtype)
                # MoonViT is 2D-only: drop the (unit) temporal dim -> [num_images, 2] (h, w).
                grid_hws = image_grid_thw[:, 1:]
                image_features = self.vision_tower(pixel_values, grid_hws)
                # MoonViT returns List[Tensor[N_i, merge_k*merge_k, hidden]] — flatten merge dim.
                all_image_features = torch.cat(image_features, dim=0).flatten(1)
                projected = self.multi_modal_projector(all_image_features).to(inputs_embeds.dtype)

                special_image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected)

            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

        if self.config.use_bidirectional_image_attention and input_ids is not None:
            attention_mask = self._compute_bidirectional_attention_mask(input_ids)
        else:
            attention_mask = None

        inputs_embeds, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
            inputs_embeds=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            pg_collection=self.config._pg_collection,
        )

        if self.config.sequence_parallel and inputs_embeds is not None:
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )
        return outputs, loss_mask
