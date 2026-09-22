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

"""Branch-isolated attention for EuroVL message-tree packing.

A message-tree sample encodes one video once as a shared prefix, followed by several
*independent* QA branches (see ``docs/models/euro_vl/message-tree-packing.md``). Plain causal
attention lets branch *b* read branches ``< b``, which teaches the model to answer using other
answers -- something that never happens at inference. The mask that fixes it is

    allowed(q, k) = causal AND same packed sample AND subsegment_ids[q] <= subsegment_ids[k]

with the shared prefix carrying ``ATTEND_ALL_SUBSEGMENT_ID`` so every branch can read it.

Transformer Engine cannot express that rule on a fused kernel: ``attn_mask_type="arbitrary"``
disables both FlashAttention and FusedAttention and falls back to the unfused backend, which
materializes ``[b, h, S, S]`` scores (~2.1 GB per layer at S=8192, ~51 GB over 24 layers).
Benchmarks (``sanity_check/bench_attn_masks.py``, 1 GH200, S=8192, 16 heads / 8 KV groups /
head_dim 128, bf16, fwd+bwd per layer) priced every alternative:

    TE THD causal (no isolation, today)      2.08 ms   <- baseline
    FlexAttention + mask_mod                 2.76 ms   <- this module
    FlexAttention without isolation          2.64 ms   (isolation itself costs 0.12 ms)
    FA2 prefix/branch split + LSE merge      3.69 ms   (and needs custom autograd)
    torch SDPA + bool mask                   6.90 ms
    TE BSHD + cuDNN post_scale_bias          7.90 ms   (and 2.6 GB peak)

So FlexAttention it is: a flash-style tiled kernel compiled from the mask rule, block-sparse so
fully masked tiles are skipped. Packs *without* a tree keep the untouched TE fused path, which is
what most packs are (video data is 10-15% of the mixture).
"""

import logging
from typing import Optional

import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType

# No fallback on purpose: branch isolation depends on FlexAttention
from torch.nn.attention.flex_attention import BlockMask

from megatron.bridge.models.euro_vl.utils import ATTEND_ALL_SUBSEGMENT_ID


logger = logging.getLogger(__name__)

# FlexAttention's block size; the mask is evaluated per tile, so a tile is skipped entirely when
# no (q, k) pair inside it is allowed. 128 is the default and matches the kernel's tiling.
_FLEX_BLOCK_SIZE = 128


def sample_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Expand THD ``cu_seqlens`` boundaries into a per-token packed-sample index.

    Args:
        cu_seqlens: ``[num_subseq + 1]`` cumulative boundaries, as emitted by the task encoder
            (its last entry equals ``seq_len``, the trailing pad included).
        seq_len: Total number of tokens in the packed sequence.

    Returns:
        ``[seq_len]`` int32 tensor whose value is the index of the packed sub-sequence each
        token belongs to.
    """
    cu = cu_seqlens.to(torch.long).flatten()
    lengths = cu[1:] - cu[:-1]
    ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=cu.device, dtype=torch.int32), lengths)
    if ids.numel() < seq_len:  # defensive: cu_seqlens should already cover the padded length
        ids = torch.nn.functional.pad(ids, (0, seq_len - ids.numel()), value=int(ids[-1]) + 1)
    return ids[:seq_len]


def branch_mask_mod(subsegment_ids: torch.Tensor, cu_seqlens: Optional[torch.Tensor]):
    """Build the FlexAttention ``mask_mod`` rule for a packed message-tree batch.

    Kept separate from :func:`build_branch_block_mask` so the rule can be evaluated directly
    (``BlockMask.to_dense()`` is block-granular, so it cannot verify per-token semantics).

    Args:
        subsegment_ids: ``[seq_len]`` or ``[1, seq_len]`` ids; ``ATTEND_ALL_SUBSEGMENT_ID`` for
            shared-prefix tokens, ``b`` for tokens of branch ``b``.
        cu_seqlens: THD boundaries of the pack, or ``None`` for an unpacked single sequence.

    Returns:
        A callable ``(b, h, q_idx, kv_idx) -> bool`` tensor: True where attention is allowed.
    """

    sub = subsegment_ids.flatten().to(torch.long)
    seq_len = int(sub.shape[0])
    if cu_seqlens is not None:
        sample = sample_ids_from_cu_seqlens(cu_seqlens, seq_len).to(torch.long)
    else:
        sample = torch.zeros(seq_len, dtype=torch.long, device=sub.device)

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: D103 - closure compiled by FlexAttention
        return (q_idx >= kv_idx) & (sample[q_idx] == sample[kv_idx]) & (sub[q_idx] <= sub[kv_idx])

    return mask_mod


def build_branch_block_mask(subsegment_ids: torch.Tensor, cu_seqlens: Optional[torch.Tensor]):
    """Compile the message-tree mask into a FlexAttention ``BlockMask``.

    Built once per micro-batch and shared by every layer: the rule depends only on the packing
    metadata, not on the layer's activations.

    Args:
        subsegment_ids: ``[seq_len]`` or ``[1, seq_len]`` subsegment ids.
        cu_seqlens: THD boundaries of the pack, or ``None`` for an unpacked single sequence.

    Returns:
        A ``torch.nn.attention.flex_attention.BlockMask`` covering the whole packed sequence.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    seq_len = int(subsegment_ids.flatten().shape[0])
    return create_block_mask(
        branch_mask_mod(subsegment_ids, cu_seqlens),
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=subsegment_ids.device,
        BLOCK_SIZE=_FLEX_BLOCK_SIZE,
    )


# The BlockMask travels to every layer as an attribute of the micro-batch's PackedSeqParams rather
# than through `attention_mask`: Megatron's activation checkpointing passes `attention_mask` to
# `ctx.save_for_backward(*args)` (tensor_parallel/random.py), which rejects a non-tensor, while
# `packed_seq_params` is captured by closure in both the selective (attention.py) and full
# (transformer_block.py) recompute paths. One PackedSeqParams per micro-batch also keeps the mask
# correct under pipeline schedules that interleave micro-batches.
BRANCH_MASK_ATTR = "branch_block_mask"


def attach_branch_mask(packed_seq_params: PackedSeqParams, block_mask) -> None:
    """Attach the message-tree ``BlockMask`` to this micro-batch's packed-sequence metadata."""
    setattr(packed_seq_params, BRANCH_MASK_ATTR, block_mask)


def get_branch_mask(packed_seq_params: Optional[PackedSeqParams]):
    """Return the ``BlockMask`` attached by :func:`attach_branch_mask`, or ``None``."""
    mask = getattr(packed_seq_params, BRANCH_MASK_ATTR, None) if packed_seq_params is not None else None
    if mask is not None and not _is_block_mask(mask):
        raise TypeError(f"{BRANCH_MASK_ATTR} must be a FlexAttention BlockMask, got {type(mask).__name__}")
    return mask


def _to_flex_layout(t: torch.Tensor) -> torch.Tensor:
    """Megatron core-attention input -> FlexAttention ``[1, heads, seq, head_dim]``.

    THD packing hands core_attention ``[t, heads, head_dim]`` (attention.py squeezes the batch
    dim); the non-packed layout is ``[s, b, heads, head_dim]``. Handle both explicitly -- a
    blanket ``squeeze(1)`` would also drop the head dim when a rank holds a single head (e.g. TP
    equal to the number of KV groups, or MQA).
    """
    if t.dim() == 3:
        return t.transpose(0, 1).unsqueeze(0)
    if t.dim() == 4:
        if t.shape[1] != 1:
            raise ValueError(f"message-tree attention expects packed batches (batch 1), got batch {t.shape[1]}")
        return t[:, 0].transpose(0, 1).unsqueeze(0)
    raise ValueError(f"unexpected core-attention input rank {t.dim()} (shape {tuple(t.shape)})")


def _from_flex_layout(out: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """FlexAttention ``[1, heads, seq, head_dim]`` -> the layout TE returns for *like*'s input.

    THD: ``[t, heads * head_dim]`` (attention.py then reshapes to ``[t, 1, -1]``);
    SBHD: ``[s, 1, heads * head_dim]``.
    """
    out = out.squeeze(0).transpose(0, 1).reshape(out.shape[2], -1)
    return (out if like.dim() == 3 else out.unsqueeze(1)).contiguous()


class BranchIsolatedDotProductAttention(TEDotProductAttention):
    """TE ``core_attention`` that additionally isolates message-tree branches.

    A subclass rather than a wrapper, so the module keeps TE's parameters, buffers and
    ``_extra_state`` under the same state-dict keys (checkpoints load identically whichever way
    ``message_tree_attention`` is set) and keeps every TE attribute (``softmax_scale``, ...).
    When the micro-batch carries no message tree it simply is TE: ``super().forward`` with the
    untouched fused kernel. When a ``BlockMask`` is attached to ``packed_seq_params`` (see
    :func:`attach_branch_mask`), it runs FlexAttention with that mask instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Compiled lazily on first use: compiling at construction would fire for every layer and
        # for runs that never see a message tree.
        self._flex_attention = None

    def _compiled_flex(self):
        """Return the compiled ``flex_attention`` callable, compiling it on first use."""
        if self._flex_attention is None:
            from torch.nn.attention.flex_attention import flex_attention

            self._flex_attention = torch.compile(flex_attention, dynamic=False)
        return self._flex_attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attend with branch isolation when a ``BlockMask`` is attached, else run TE unchanged.

        Args:
            query: ``[t, heads, head_dim]`` (THD packing) or ``[s, 1, heads, head_dim]``.
            key: Same layout as *query*, with ``num_gqa_groups`` heads.
            value: Same shape as *key*.
            attention_mask: Forwarded to TE; unused on the FlexAttention path.
            attn_mask_type: Mask type override from the caller.
            attention_bias: Additive bias; unsupported on the FlexAttention path.
            packed_seq_params: THD metadata; may carry the branch ``BlockMask``.
            **kwargs: Forwarded to TE unchanged.

        Returns:
            Exactly the layout TE returns for the same input, so the rest of the layer is unchanged.
        """
        block_mask = get_branch_mask(packed_seq_params)
        if block_mask is None:  # TE default attention procedure (fused kernel, no tree in this pack)
            return super().forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )
        if attention_bias is not None:
            raise ValueError("attention_bias is not supported together with message-tree masking")
        if self.training and float(getattr(self, "attention_dropout", 0.0) or 0.0) > 0.0:
            raise ValueError("attention dropout is not supported on the message-tree (FlexAttention) path")

        # Custom block mask for message-tree branches, on a flash-style FlexAttention kernel. The scale
        # comes from the same config TE reads; None means 1/sqrt(head_dim) for both kernels.
        q, k, v = (_to_flex_layout(t) for t in (query, key, value))
        out = self._compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=True, scale=self.config.softmax_scale)
        return _from_flex_layout(out, like=query)


def _is_block_mask(obj: object) -> bool:
    """Whether *obj* is a FlexAttention ``BlockMask`` rather than a tensor mask or ``None``."""
    if obj is None or isinstance(obj, torch.Tensor):
        return False
    return isinstance(obj, BlockMask)


def has_message_tree(subsegment_ids: Optional[torch.Tensor]) -> bool:
    """Whether a batch actually contains branches, i.e. anything but the attend-all id."""
    if subsegment_ids is None:
        return False
    return bool((subsegment_ids != ATTEND_ALL_SUBSEGMENT_ID).any())
