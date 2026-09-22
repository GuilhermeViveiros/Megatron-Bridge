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

"""Vocabulary padding for the EuroVL backbones.

EuroLLM's vocab is 128005, which is odd and therefore cannot be split across tensor-parallel
ranks. These tests pin the two halves of the fix: the provider builds the decoder with a padded
vocabulary, and the bridge keeps the HF side at the real 128005 rows in both directions.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.euro_vl.euro_vl_bridge import EuroVLBridge
from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider, _resolve_padded_vocab_size
from megatron.bridge.models.euro_vl.utils import EUROLLM_PADDED_VOCAB_SIZE


EUROLLM_VOCAB = 128005
EMBED = "language_model.model.embed_tokens.weight"
LM_HEAD = "language_model.lm_head.weight"


def _provider(**overrides) -> EuroVLModelProvider:
    """Minimal EuroVL provider carrying the EuroLLM vocab settings."""
    kwargs = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        vocab_size=EUROLLM_VOCAB,
        make_vocab_size_divisible_by=128,
        should_pad_vocab=True,
        padded_vocab_size=EUROLLM_PADDED_VOCAB_SIZE,
        mrope_section=[4, 2, 2],
    )
    kwargs.update(overrides)
    return EuroVLModelProvider(**kwargs)


def _bridge(model_type: str = "llama") -> EuroVLBridge:
    """Bridge with the `hf_config` the dispatch system would have set."""
    bridge = EuroVLBridge()
    bridge.hf_config = SimpleNamespace(text_config=SimpleNamespace(model_type=model_type, vocab_size=EUROLLM_VOCAB))
    return bridge


class TestResolvePaddedVocabSize:
    @pytest.mark.parametrize("tp", [1, 2, 4])
    def test_pinned_size_is_tp_independent(self, tp):
        """The whole point of pinning: one embedding shape that loads at TP=1, 2 and 4."""
        provider = _provider(tensor_model_parallel_size=tp)
        assert _resolve_padded_vocab_size(provider) == EUROLLM_PADDED_VOCAB_SIZE
        assert EUROLLM_PADDED_VOCAB_SIZE % tp == 0

    def test_unpadded_when_flag_off(self):
        assert _resolve_padded_vocab_size(_provider(should_pad_vocab=False)) == EUROLLM_VOCAB

    @pytest.mark.parametrize("tp,expected", [(1, 128128), (2, 128256), (4, 128512)])
    def test_falls_back_to_per_tp_rule(self, tp, expected):
        """Without an explicit size, Megatron's `128 * TP` rule applies -- and differs per TP."""
        provider = _provider(padded_vocab_size=None, tensor_model_parallel_size=tp)
        assert _resolve_padded_vocab_size(provider) == expected

    def test_rejects_size_below_vocab(self):
        with pytest.raises(ValueError, match="smaller than vocab_size"):
            _resolve_padded_vocab_size(_provider(padded_vocab_size=EUROLLM_VOCAB - 1))

    def test_rejects_size_not_divisible_by_tp(self):
        # 128007 is >= the real vocab but odd, so it cannot be split across 2 TP ranks.
        with pytest.raises(ValueError, match="not divisible by"):
            _resolve_padded_vocab_size(_provider(padded_vocab_size=EUROLLM_VOCAB + 2, tensor_model_parallel_size=2))


class TestBridgeVocabHooks:
    @pytest.mark.parametrize("key", [EMBED, LM_HEAD])
    def test_import_pads_vocab_rows_with_zeros(self, key):
        weights = torch.randn(EUROLLM_VOCAB, 8)
        padded = _bridge().maybe_modify_loaded_hf_weight(key, {key: weights})

        assert padded.shape == (EUROLLM_PADDED_VOCAB_SIZE, 8)
        torch.testing.assert_close(padded[:EUROLLM_VOCAB], weights)
        assert torch.count_nonzero(padded[EUROLLM_VOCAB:]) == 0

    def test_import_leaves_other_weights_alone(self):
        key = "language_model.model.layers.0.self_attn.o_proj.weight"
        weights = torch.randn(8, 8)
        torch.testing.assert_close(_bridge().maybe_modify_loaded_hf_weight(key, {key: weights}), weights)

    def test_import_is_noop_for_qwen3_backbone(self):
        """Qwen3's 151936 vocab already splits across TP, so its rows must not change."""
        weights = torch.randn(151936, 8)
        out = _bridge(model_type="qwen3").maybe_modify_loaded_hf_weight(EMBED, {EMBED: weights})
        torch.testing.assert_close(out, weights)

    @pytest.mark.parametrize("key", [EMBED, LM_HEAD])
    def test_export_trims_padding_rows(self, key):
        padded = torch.randn(EUROLLM_PADDED_VOCAB_SIZE, 8)
        out = _bridge().maybe_modify_converted_hf_weight(
            task=None, converted_weights_dict={key: padded}, hf_state_dict={}
        )

        assert out[key].shape == (EUROLLM_VOCAB, 8)
        torch.testing.assert_close(out[key], padded[:EUROLLM_VOCAB])

    def test_export_leaves_other_weights_alone(self):
        key = "language_model.model.layers.0.mlp.down_proj.weight"
        weights = torch.randn(8, 8)
        out = _bridge().maybe_modify_converted_hf_weight(
            task=None, converted_weights_dict={key: weights}, hf_state_dict={}
        )
        torch.testing.assert_close(out[key], weights)

    def test_import_then_export_round_trips_exactly(self):
        """Pad -> trim must return the original tensor; this is what keeps checkpoints reversible."""
        original = torch.randn(EUROLLM_VOCAB, 8)
        bridge = _bridge()
        padded = bridge.maybe_modify_loaded_hf_weight(EMBED, {EMBED: original})
        restored = bridge.maybe_modify_converted_hf_weight(
            task=None, converted_weights_dict={EMBED: padded}, hf_state_dict={}
        )[EMBED]
        torch.testing.assert_close(restored, original)
