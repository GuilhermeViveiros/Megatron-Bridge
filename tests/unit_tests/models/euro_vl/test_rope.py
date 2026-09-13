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

"""Unit tests for EuroVL's ``get_rope_index`` (3D M-RoPE position-id construction), covering the
Qwen2-VL/Qwen3-VL M-RoPE properties: text tokens get identical (t, h, w) positions, image tokens
get a constant temporal id with varying height/width, and packed multi-document sequences reset
position ids to 0 at each document boundary rather than leaking across samples.
"""

import pytest
import torch

from megatron.bridge.models.euro_vl.rope import get_rope_index


IMAGE_TOKEN_ID = 200
VIDEO_TOKEN_ID = 201
VISION_START_TOKEN_ID = 199
TEXT_TOK = 10


@pytest.mark.unit
class TestGetRopeIndex:
    def test_text_only_matches_1d_rope(self):
        """Plain text: t, h, w must all be identical to a standard 1D position sequence."""
        input_ids = torch.tensor([[TEXT_TOK] * 6], dtype=torch.long)
        pos = get_rope_index(
            input_ids,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
        )
        expected = torch.arange(6).view(1, 1, -1).expand(3, 1, -1)
        assert torch.equal(pos, expected)

    def test_image_constant_temporal_varying_spatial(self):
        """A single image's patches: constant t, h/w varying by grid position (both offset by
        the same base so the block's relative (row, col) structure is preserved)."""
        img_placeholder = [IMAGE_TOKEN_ID] * 4  # grid (1,4,4), merge=2 -> 1*2*2 = 4 merged tokens
        tokens = [TEXT_TOK, TEXT_TOK, TEXT_TOK, VISION_START_TOKEN_ID] + img_placeholder + [TEXT_TOK]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

        pos = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
        )
        t, h, w = pos[0, 0], pos[1, 0], pos[2, 0]

        # image occupies local indices [4:8) (after 3 text tokens + 1 vision_start token)
        t_img, h_img, w_img = t[4:8], h[4:8], w[4:8]
        assert (t_img == t_img[0]).all(), "temporal id must be constant across one image's patches"
        assert h_img.tolist() == [h_img[0].item(), h_img[0].item(), h_img[0].item() + 1, h_img[0].item() + 1]
        assert w_img.tolist() == [w_img[0].item(), w_img[0].item() + 1, w_img[0].item(), w_img[0].item() + 1]

    def test_multiple_images_get_distinct_temporal_offsets(self):
        """Two images in the same document: each is internally constant on t, but the two
        constants differ (later image's t is strictly greater), per the "new segment starts at
        max(previous) + 1" rule applied at every modality transition."""
        img_placeholder = [IMAGE_TOKEN_ID] * 4
        tokens = (
            [TEXT_TOK, TEXT_TOK, TEXT_TOK]
            + [VISION_START_TOKEN_ID] + img_placeholder
            + [TEXT_TOK, TEXT_TOK]
            + [VISION_START_TOKEN_ID] + img_placeholder
            + [TEXT_TOK, TEXT_TOK, TEXT_TOK, TEXT_TOK]
        )
        input_ids = torch.tensor([tokens], dtype=torch.long)
        image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)

        pos = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
        )
        t = pos[0, 0]
        t_img1 = t[4:8]
        t_img2 = t[11:15]

        assert (t_img1 == t_img1[0]).all()
        assert (t_img2 == t_img2[0]).all()
        assert t_img2[0].item() > t_img1[0].item(), "second image must get a strictly later temporal offset"

    def test_packed_documents_reset_position_at_boundary(self):
        """THD-packed multi-document sequences: each cu_seqlens segment is an independent
        document whose positions restart at 0, with no leakage from the previous document."""
        img_placeholder = [IMAGE_TOKEN_ID] * 4
        doc_a = (
            [TEXT_TOK, TEXT_TOK, TEXT_TOK]
            + [VISION_START_TOKEN_ID] + img_placeholder
            + [TEXT_TOK, TEXT_TOK, TEXT_TOK, TEXT_TOK]
        )
        doc_b = [TEXT_TOK, TEXT_TOK] + [VISION_START_TOKEN_ID] + img_placeholder + [TEXT_TOK]

        packed = doc_a + doc_b
        input_ids = torch.tensor([packed], dtype=torch.long)
        image_grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
        cu_seqlens = torch.tensor([0, len(doc_a), len(doc_a) + len(doc_b)], dtype=torch.long)

        pos = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            spatial_merge_size=2,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
        )
        doc_b_offset = len(doc_a)
        first_tok_pos = (
            pos[0, 0, doc_b_offset].item(),
            pos[1, 0, doc_b_offset].item(),
            pos[2, 0, doc_b_offset].item(),
        )
        assert first_tok_pos == (0, 0, 0), "second packed document must reset to (0, 0, 0), not continue from the first"
