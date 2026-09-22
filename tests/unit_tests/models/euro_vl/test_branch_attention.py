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

import pytest
import torch

from megatron.bridge.data.energon.euro_vl_task_encoder import ATTEND_ALL_SUBSEGMENT_ID
from megatron.bridge.models.euro_vl.branch_attention import (
    branch_mask_mod,
    build_branch_block_mask,
    has_message_tree,
    sample_ids_from_cu_seqlens,
)


ATTEND_ALL = ATTEND_ALL_SUBSEGMENT_ID


def _pack(tree, flats, seq_len):
    """Build (cu_seqlens, subsegment_ids) for one pack, as EuroVLTaskEncoder.batch emits them."""
    prefix_len, branch_lens = tree
    sub = torch.full((seq_len,), ATTEND_ALL, dtype=torch.int32)
    pos = prefix_len
    for b, blen in enumerate(branch_lens):
        sub[pos : pos + blen] = b
        pos += blen
    seqlens = [prefix_len + sum(branch_lens)] + list(flats)
    if sum(seqlens) < seq_len:
        seqlens.append(seq_len - sum(seqlens))
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32).cumsum(0)
    return cu, sub


def _dense_reference(cu, sub, seq_len):
    """The mask we want, built the slow, obvious way."""
    sample = sample_ids_from_cu_seqlens(cu, seq_len).long()
    s = sub.long()
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    return causal & (sample[:, None] == sample[None, :]) & (s[:, None] <= s[None, :])


def _dense_from_mask_mod(cu, sub, seq_len):
    """Evaluate the compiled rule over every (q, kv) pair."""
    idx = torch.arange(seq_len)
    q_idx, kv_idx = torch.meshgrid(idx, idx, indexing="ij")
    return branch_mask_mod(sub, cu)(None, None, q_idx, kv_idx)


class TestSampleIdsFromCuSeqlens:
    def test_expands_boundaries_to_per_token_ids(self):
        cu = torch.tensor([0, 3, 7, 8], dtype=torch.int32)
        ids = sample_ids_from_cu_seqlens(cu, 8)
        assert ids.tolist() == [0, 0, 0, 1, 1, 1, 1, 2]
        assert ids.dtype == torch.int32

    def test_pad_subsequence_is_its_own_sample(self):
        cu, sub = _pack(tree=(6, [2, 2]), flats=[3], seq_len=16)
        ids = sample_ids_from_cu_seqlens(cu, 16)
        # tree(10) + flat(3) + pad(3)
        assert ids[:10].tolist() == [0] * 10
        assert ids[10:13].tolist() == [1] * 3
        assert ids[13:].tolist() == [2] * 3


class TestHasMessageTree:
    def test_none_and_all_attend_all_are_not_trees(self):
        assert not has_message_tree(None)
        assert not has_message_tree(torch.full((16,), ATTEND_ALL, dtype=torch.int32))

    def test_any_branch_id_makes_it_a_tree(self):
        sub = torch.full((16,), ATTEND_ALL, dtype=torch.int32)
        sub[7] = 0
        assert has_message_tree(sub)


class TestBranchMaskMod:
    """The rule must match the dense reference exactly, per token."""

    def _check(self, tree, flats, seq_len):
        cu, sub = _pack(tree, flats, seq_len)
        assert torch.equal(_dense_from_mask_mod(cu, sub, seq_len), _dense_reference(cu, sub, seq_len))

    def test_matches_dense_reference(self):
        self._check(tree=(20, [8, 6]), flats=[10, 4], seq_len=64)

    def test_branches_cannot_see_each_other_but_see_the_prefix(self):
        seq_len = 48
        cu, sub = _pack(tree=(16, [8, 8]), flats=[], seq_len=seq_len)
        dense = _dense_from_mask_mod(cu, sub, seq_len)
        prefix, b0, b1 = torch.arange(0, 16), torch.arange(16, 24), torch.arange(24, 32)
        assert dense[b1][:, b0].sum() == 0, "branch 1 must not see branch 0"
        assert dense[b1][:, prefix].all(), "branch 1 must see the whole shared prefix"
        assert dense[b0][:, prefix].all(), "branch 0 must see the whole shared prefix"
        # Causality still holds inside a branch.
        assert not dense[b0[0], b0[-1]], "a branch token must not see later tokens"

    def test_no_attention_across_packed_samples(self):
        seq_len = 64
        cu, sub = _pack(tree=(16, [8, 8]), flats=[16], seq_len=seq_len)
        dense = _dense_from_mask_mod(cu, sub, seq_len)
        tree_rows, flat_rows = torch.arange(0, 32), torch.arange(32, 48)
        assert dense[flat_rows][:, tree_rows].sum() == 0
        assert dense[tree_rows][:, flat_rows].sum() == 0

    def test_flat_only_pack_is_plain_block_causal(self):
        """With no branches the rule must equal today's cu_seqlens causal behaviour."""
        seq_len = 32
        sub = torch.full((seq_len,), ATTEND_ALL, dtype=torch.int32)
        cu = torch.tensor([0, 16, 32], dtype=torch.int32)
        assert torch.equal(_dense_from_mask_mod(cu, sub, seq_len), _dense_reference(cu, sub, seq_len))

    def test_unpacked_sequence_is_causal_with_branches(self):
        seq_len = 32
        sub = torch.full((seq_len,), ATTEND_ALL, dtype=torch.int32)
        sub[16:24] = 0
        sub[24:] = 1
        dense = branch_mask_mod(sub, None)(
            None, None, *torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len), indexing="ij")
        )
        assert dense[torch.arange(24, 32)][:, torch.arange(16, 24)].sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention block masks need CUDA")
class TestBuildBranchBlockMask:
    """``BlockMask`` is block-granular; check it keeps the blocks the rule needs."""

    def test_block_mask_covers_every_allowed_pair(self):
        seq_len = 1024
        cu, sub = _pack(tree=(256, [128, 128]), flats=[256], seq_len=seq_len)
        cu, sub = cu.cuda(), sub.cuda()
        blocks = build_branch_block_mask(sub, cu).to_dense()[0, 0].cpu().bool()
        want = _dense_reference(cu.cpu(), sub.cpu(), seq_len)
        block = 128
        for qb in range(seq_len // block):
            for kb in range(seq_len // block):
                needed = want[qb * block : (qb + 1) * block, kb * block : (kb + 1) * block].any()
                assert not needed or blocks[qb, kb], f"block ({qb},{kb}) dropped but needed"

    def test_block_mask_drops_fully_masked_blocks(self):
        """Branch isolation must actually create sparsity, else there is nothing to gain."""
        seq_len = 1024
        cu, sub = _pack(tree=(256, [256, 256]), flats=[], seq_len=seq_len)
        blocks = build_branch_block_mask(sub.cuda(), cu.cuda()).to_dense()[0, 0].cpu().bool()
        # Branch 1 rows (blocks 4-5) over branch 0 columns (blocks 2-3) must be fully dropped.
        assert blocks[4:6, 2:4].sum() == 0


class TestCoreAttentionSpecPatch:
    """The patcher must swap core_attention on every layer spec it is given, and nothing else."""

    def _layer_spec(self):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

        return get_gpt_layer_with_transformer_engine_spec()

    def test_patches_a_single_layer_spec(self):
        from megatron.bridge.models.euro_vl.branch_attention import BranchIsolatedDotProductAttention
        from megatron.bridge.models.euro_vl.euro_vl_provider import _patch_core_attention_specs

        spec = self._layer_spec()
        before = spec.submodules.self_attention.submodules.core_attention
        assert before is not BranchIsolatedDotProductAttention
        assert _patch_core_attention_specs(spec) == 1
        assert spec.submodules.self_attention.submodules.core_attention is BranchIsolatedDotProductAttention

    def test_patches_every_layer_of_a_block_spec(self):
        from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

        from megatron.bridge.models.euro_vl.branch_attention import BranchIsolatedDotProductAttention
        from megatron.bridge.models.euro_vl.euro_vl_provider import _patch_core_attention_specs

        block = TransformerBlockSubmodules(layer_specs=[self._layer_spec() for _ in range(3)])
        assert _patch_core_attention_specs(block) == 3
        for layer in block.layer_specs:
            assert layer.submodules.self_attention.submodules.core_attention is BranchIsolatedDotProductAttention

    def test_none_and_unrelated_specs_are_ignored(self):
        from megatron.core.transformer.spec_utils import ModuleSpec

        from megatron.bridge.models.euro_vl.euro_vl_provider import _patch_core_attention_specs

        assert _patch_core_attention_specs(None) == 0
        assert _patch_core_attention_specs(ModuleSpec(module=torch.nn.Identity)) == 0

    def test_provider_flag_defaults_to_off(self):
        """Opt-in: an unset flag must leave today's TE fused attention in place."""
        from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider

        provider = EuroVLModelProvider(num_layers=2, hidden_size=256, num_attention_heads=4)
        assert provider.message_tree_attention is False


class TestPositionResetFollowsTheFlag:
    """Per-branch position reset and branch isolation must switch on and off together.

    Resetting positions while branches can still attend to each other would give two mutually
    visible tokens the same position, so Qwen3EuroVLModel.forward only forwards the ids to
    get_rope_index when message_tree_attention is on.
    """

    def _forward(self, flag):
        from types import SimpleNamespace
        from unittest import mock

        from megatron.bridge.models.euro_vl import modeling_euro_vl as m

        model = object.__new__(m.Qwen3EuroVLModel)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(
                message_tree_attention=flag,
                vision_config=SimpleNamespace(merge_kernel_size=[2, 2]),
                image_token_id=128000,
                video_token_id=128004,
                vision_start_token_id=128001,
            ),
        )
        ids = torch.full((1, 8), ATTEND_ALL, dtype=torch.int32)
        ids[0, 4:] = 0
        seen = {}

        def fake_rope(*args, **kwargs):
            seen["rope"] = kwargs.get("subsegment_ids")
            return torch.zeros(3, 1, 8, dtype=torch.long)

        def fake_base_forward(self, **kwargs):
            seen["base"] = kwargs.get("subsegment_ids")
            return None

        with (
            mock.patch("megatron.bridge.models.euro_vl.rope.get_rope_index", fake_rope),
            mock.patch.object(m.EuroVLModel, "forward", fake_base_forward),
        ):
            model.forward(
                input_ids=torch.zeros(1, 8, dtype=torch.long),
                packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32)),
                subsegment_ids=ids,
            )
        return seen, ids

    def test_flag_off_keeps_continuous_positions(self):
        seen, _ = self._forward(False)
        assert seen["rope"] is None, "positions must not be reset without branch isolation"

    def test_flag_on_resets_positions_per_branch(self):
        seen, ids = self._forward(True)
        assert seen["rope"] is ids
        assert seen["base"] is ids, "the base forward still needs the ids to build the mask"


class TestFlexLayout:
    """Megatron core-attention layouts <-> FlexAttention [1, heads, seq, head_dim], both ways."""

    @pytest.mark.parametrize("heads", [16, 1])  # 1 head per rank: the case a blanket squeeze(1) broke
    def test_thd_round_trip(self, heads):
        from megatron.bridge.models.euro_vl.branch_attention import _from_flex_layout, _to_flex_layout

        t = torch.randn(256, heads, 8)
        flex = _to_flex_layout(t)
        assert flex.shape == (1, heads, 256, 8)
        assert torch.equal(flex[0].transpose(0, 1), t)
        out = _from_flex_layout(flex, like=t)
        assert out.shape == (256, heads * 8)  # TE's THD output; attention.py reshapes to [t, 1, -1]
        assert torch.equal(out, t.reshape(256, -1))

    @pytest.mark.parametrize("heads", [16, 1])
    def test_sbhd_round_trip(self, heads):
        from megatron.bridge.models.euro_vl.branch_attention import _from_flex_layout, _to_flex_layout

        t = torch.randn(256, 1, heads, 8)
        flex = _to_flex_layout(t)
        assert flex.shape == (1, heads, 256, 8)
        out = _from_flex_layout(flex, like=t)
        assert out.shape == (256, 1, heads * 8)
        assert torch.equal(out, t.reshape(256, 1, -1))

    def test_batch_larger_than_one_is_rejected(self):
        from megatron.bridge.models.euro_vl.branch_attention import _to_flex_layout

        with pytest.raises(ValueError, match="batch 1"):
            _to_flex_layout(torch.randn(64, 2, 4, 8))


class TestBranchMaskHandOff:
    """The mask rides on packed_seq_params, which survives activation recompute."""

    def test_attach_and_get(self):
        from megatron.core.packed_seq_params import PackedSeqParams

        from megatron.bridge.models.euro_vl.branch_attention import (
            BRANCH_MASK_ATTR,
            attach_branch_mask,
            get_branch_mask,
        )

        params = PackedSeqParams(qkv_format="thd")
        assert get_branch_mask(params) is None and get_branch_mask(None) is None
        with pytest.raises(TypeError, match="BlockMask"):
            setattr(params, BRANCH_MASK_ATTR, torch.ones(2))
            get_branch_mask(params)
        if torch.cuda.is_available():
            cu, sub = _pack(tree=(256, [128, 128]), flats=[], seq_len=1024)
            mask = build_branch_block_mask(sub.cuda(), cu.cuda())
            attach_branch_mask(params, mask)
            assert get_branch_mask(params) is mask

    def test_attention_subclasses_transformer_engine(self):
        """A subclass, not a wrapper: checkpoint keys and TE attributes stay identical."""
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        from megatron.bridge.models.euro_vl.branch_attention import BranchIsolatedDotProductAttention

        assert issubclass(BranchIsolatedDotProductAttention, TEDotProductAttention)


class TestMessageTreeSupportGuards:
    """Settings that would silently break isolation must fail at model build."""

    def _provider(self, **kw):
        from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider

        kw.setdefault("attention_dropout", 0.0)  # the EuroVL recipes; Megatron's default is 0.1
        return EuroVLModelProvider(num_layers=2, hidden_size=256, num_attention_heads=4, **kw)

    def test_flag_off_allows_anything(self):
        from megatron.bridge.models.euro_vl.euro_vl_provider import _check_message_tree_support

        _check_message_tree_support(self._provider(context_parallel_size=2))

    def test_context_parallel_is_rejected(self):
        from megatron.bridge.models.euro_vl.euro_vl_provider import _check_message_tree_support

        with pytest.raises(ValueError, match="context_parallel_size"):
            _check_message_tree_support(self._provider(message_tree_attention=True, context_parallel_size=2))

    def test_cuda_graphs_are_rejected(self):
        from megatron.bridge.models.euro_vl.euro_vl_provider import _check_message_tree_support

        provider = self._provider(message_tree_attention=True)
        provider.cuda_graph_impl = "transformer_engine"
        with pytest.raises(ValueError, match="CUDA graphs"):
            _check_message_tree_support(provider)

    def test_default_single_gpu_config_passes(self):
        from megatron.bridge.models.euro_vl.euro_vl_provider import _check_message_tree_support

        _check_message_tree_support(self._provider(message_tree_attention=True))

    def test_attention_dropout_is_rejected_at_build(self):
        from megatron.bridge.models.euro_vl.euro_vl_provider import _check_message_tree_support

        with pytest.raises(ValueError, match="attention_dropout"):
            _check_message_tree_support(self._provider(message_tree_attention=True, attention_dropout=0.1))

    def test_bidirectional_image_attention_is_rejected(self):
        from megatron.bridge.models.euro_vl.euro_vl_provider import _check_message_tree_support

        with pytest.raises(ValueError, match="bidirectional"):
            _check_message_tree_support(
                self._provider(message_tree_attention=True, use_bidirectional_image_attention=True)
            )
