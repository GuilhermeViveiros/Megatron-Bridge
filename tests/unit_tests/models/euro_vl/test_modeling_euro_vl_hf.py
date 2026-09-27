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
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, Qwen3Config

from megatron.bridge.models.euro_vl.modeling_euro_vl_hf import EuroVLTextForCausalLM


# head_dim=128 so the default mrope_section [24, 20, 20] (sum 64) is valid, as in the real models.
_TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=128,
    max_position_embeddings=256,
    hidden_act="silu",
    tie_word_embeddings=False,
)


def _llama_config() -> LlamaConfig:
    config = LlamaConfig(**_TINY)
    config._attn_implementation = "eager"
    return config


def _qk_norm_param_names(model: torch.nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if "q_norm" in name or "k_norm" in name]


def test_llama_backbone_has_no_qk_norm():
    model = EuroVLTextForCausalLM(_llama_config())
    assert _qk_norm_param_names(model) == []


def test_qwen3_backbone_keeps_qk_norm():
    model = EuroVLTextForCausalLM(Qwen3Config(**_TINY))
    assert len(_qk_norm_param_names(model)) == 2 * _TINY["num_hidden_layers"]


def test_llama_backbone_matches_llama_for_causal_lm():
    """Same parameter names as LlamaForCausalLM, and identical logits on text-only input:
    with no vision tokens t=h=w, so interleaved M-RoPE collapses to standard 1D RoPE."""
    torch.manual_seed(0)
    config = _llama_config()
    reference = LlamaForCausalLM(config).eval()
    model = EuroVLTextForCausalLM(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)

    input_ids = torch.randint(0, _TINY["vocab_size"], (2, 17))
    with torch.no_grad():
        expected = reference(input_ids=input_ids).logits
        actual = model(input_ids=input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_unsupported_backbone_raises():
    with pytest.raises(ValueError, match="Unsupported EuroVL text backbone"):
        EuroVLTextForCausalLM(MistralConfig(**_TINY))


def test_mrope_section_must_match_head_dim():
    config = _llama_config()
    config.rope_parameters = {**config.rope_parameters, "mrope_section": [10, 10, 10]}
    with pytest.raises(ValueError, match="mrope_section"):
        EuroVLTextForCausalLM(config)
