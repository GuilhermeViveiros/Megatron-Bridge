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

"""Multimodal RoPE (M-RoPE) position-index construction for EuroVL.

This module implements ``get_rope_index`` — the algorithm that builds the 3D
``(t, h, w)`` position ids consumed by Qwen3-VL-style interleaved M-RoPE. It is
adapted from ``models/qwen_vl/modelling_qwen3_vl/rope.py`` with one important
difference: it supports EuroVL's **THD packing**, where many samples are packed
into a single ``[1, total]`` sequence delimited by ``cu_seqlens``. Each packed
sub-sequence is an independent document whose positions **reset to 0** at its
``cu_seqlens`` boundary.

The interleaved rotary embedding that consumes these positions is added in a
separate step (P2); this module only constructs the position ids so they can be
validated in isolation first.
"""

from typing import Optional

import torch


def _document_positions(
    input_tokens: list[int],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    image_index: int,
    video_index: int,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[torch.Tensor, int, int]:
    """Build ``(t, h, w)`` positions for a single document (one packed sub-sequence).

    Mirrors the per-document inner loop of Qwen3-VL's ``get_rope_index``: text runs
    advance equally on all three axes; each vision block lays its tokens over the
    merged ``(t, h//merge, w//merge)`` grid; the next segment starts at
    ``max(previous positions) + 1`` (so a vision block consumes only its max grid
    extent in position-space, not its token count).

    Args:
        input_tokens: Token ids of this document as a Python list (no padding).
        image_grid_thw: ``[num_images, 3]`` patch grids ``(t, h, w)`` for all images
            in the batch; consumed in order via ``image_index``.
        video_grid_thw: ``[num_videos, 3]`` patch grids for all videos.
        image_index: Running index into ``image_grid_thw`` (images already consumed).
        video_index: Running index into ``video_grid_thw``.
        spatial_merge_size: MoonViT spatial merge factor (2 for EuroVL).
        image_token_id: Placeholder token id for image patches.
        video_token_id: Placeholder token id for video frames.
        vision_start_token_id: Delimiter opening a vision block.

    Returns:
        ``(positions, image_index, video_index)`` where ``positions`` is a
        ``[3, len(input_tokens)]`` long tensor starting at base 0, and the indices
        are advanced past the vision blocks consumed by this document.
    """
    ids = torch.tensor(input_tokens, dtype=torch.long)
    vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
    vision_tokens = ids[vision_start_indices + 1] if vision_start_indices.numel() > 0 else ids[:0]
    image_nums = int((vision_tokens == image_token_id).sum())
    video_nums = int((vision_tokens == video_token_id).sum())

    llm_pos_ids_list: list[torch.Tensor] = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    for _ in range(image_nums + video_nums):
        ed_image = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens and remain_images > 0) else len(input_tokens) + 1
        ed_video = input_tokens.index(video_token_id, st) if (video_token_id in input_tokens and remain_videos > 0) else len(input_tokens) + 1
        if ed_image < ed_video:
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            remain_images -= 1
            ed = ed_image
        else:
            t, h, w = video_grid_thw[video_index]
            video_index += 1
            remain_videos -= 1
            ed = ed_video
        llm_grid_t = int(t)
        llm_grid_h = int(h) // spatial_merge_size
        llm_grid_w = int(w) // spatial_merge_size
        text_len = ed - st

        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    if len(llm_pos_ids_list) == 0:
        return torch.zeros(3, 0, dtype=torch.long), image_index, video_index
    positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    return positions, image_index, video_index


def get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    spatial_merge_size: int = 2,
    image_token_id: int = 128000,
    video_token_id: int = 128004,
    vision_start_token_id: int = 128001,
) -> torch.Tensor:
    """Build 3D M-RoPE position ids ``[3, B, S]`` for EuroVL.

    The mode is selected by **whether the batch is packed** (``cu_seqlens``), which is
    independent of the batch size — ``B == 1`` does NOT imply packing (a single padded
    sample is a plain batch):

      * **THD packed** (``cu_seqlens`` is set ⇒ ``B == 1``, ``[1, total]``): each
        ``cu_seqlens`` segment is an independent document whose ``(t, h, w)`` positions
        reset to base 0. The trailing pad is just another segment. Images are consumed in
        order from ``image_grid_thw`` across segments.
      * **Plain / padded batch** (``cu_seqlens is None``, any ``B``): each row is one
        document. If ``attention_mask`` is given, positions are computed only over each
        row's valid (``mask == 1``) tokens and scattered onto those slots (pad slots stay
        0); without a mask the whole row is treated as valid.

    Args:
        input_ids: Token ids ``[B, S]`` (``B == 1`` only when packed).
        image_grid_thw: ``[num_images, 3]`` patch grids ``(t, h, w)``.
        video_grid_thw: ``[num_videos, 3]`` patch grids.
        cu_seqlens: Cumulative packed sub-sequence boundaries ``[num_subseqs + 1]``.
            Set this for the THD-packed path; leave ``None`` for a plain/padded batch.
        attention_mask: ``[B, S]`` 1/0 validity mask for the plain/padded path. Ignored
            when ``cu_seqlens`` is set (the pad segment carries the padding instead).
        spatial_merge_size: MoonViT spatial merge factor (2 for EuroVL).
        image_token_id: Image placeholder token id.
        video_token_id: Video placeholder token id.
        vision_start_token_id: Vision-block start delimiter token id.

    Returns:
        Long tensor ``[3, B, S]`` of ``(t, h, w)`` positions; pad slots are 0.
    """
    # Qwen3-VL timestamp video: a [t, h, w] video grid row describes t frames, but each frame
    # is its OWN vision block in the token stream (each with a timestamp text prefix). Split
    # every row into t per-frame [1, h, w] rows so each block has llm_grid_t == 1 -> the t-axis
    # index is always 0; temporal ordering is carried by the timestamp text advancing the base.
    # Idempotent on t == 1 rows (images, or already-per-frame video), so it is always safe.
    if video_grid_thw is not None and video_grid_thw.numel() > 0:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    B, S = input_ids.shape
    position_ids = torch.zeros(3, B, S, dtype=torch.long, device=input_ids.device)
    img_idx = vid_idx = 0

    def _doc(token_ids: list[int]) -> torch.Tensor:
        nonlocal img_idx, vid_idx
        pos, img_idx, vid_idx = _document_positions(
            token_ids,
            image_grid_thw,
            video_grid_thw,
            img_idx,
            vid_idx,
            spatial_merge_size=spatial_merge_size,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
        )
        return pos.to(position_ids.device)

    if cu_seqlens is not None:
        assert B == 1, f"THD packing expects batch size 1, got {B}"
        bounds = cu_seqlens.tolist()
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            position_ids[:, 0, a:b] = _doc(input_ids[0, a:b].tolist())
    else:
        for i in range(B):
            if attention_mask is not None:
                valid = attention_mask[i].bool()
                pos = _doc(input_ids[i][valid].tolist())
                position_ids[:, i, valid] = pos
            else:
                position_ids[:, i, :] = _doc(input_ids[i].tolist())

    return position_ids
