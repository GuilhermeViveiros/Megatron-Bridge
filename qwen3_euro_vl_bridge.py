#!/usr/bin/env python3
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

"""Assemble a Qwen3EuroVL **HF** checkpoint from Qwen3-1.7B + MoonViT-SO-400M.

The Qwen3-backbone validation/oracle variant of ``eurollm_bridge.py`` (see current.md):
same EuroVL layout (``vision_tower.*`` MoonViT + ``multi_modal_projector.*`` + a causal
LM under ``language_model.*``), but the LLM is Qwen3 — which brings Qwen3-VL's proven
interleaved M-RoPE stack on the Megatron side.

Differences from the EuroLLM assembly:
  1. **No vocab extension.** Qwen3's tokenizer ships the vision tokens built in
     (<|vision_start|>=151652, <|vision_end|>=151653, <|vision_pad|>=151654,
     <|image_pad|>=151655, <|video_pad|>=151656) — the exact ids Qwen3-VL uses.
  2. **Tied embeddings.** Qwen3-1.7B ties lm_head to embed_tokens; its safetensors
     contain only ``model.embed_tokens.weight``, so ``lm_head.weight`` is expected to
     be "missing" on load (the tie makes it share storage).
  3. **Chat template** inserts ``<|image_pad|>`` / ``<|video_pad|>`` (Qwen3EuroVLProcessor
     uses those strings; <|vision_start|>/<|vision_end|> already match EuroVL's).

Usage (in container)::

    uv run --no-sync python qwen3_euro_vl_bridge.py \\
        --qwen3-path /path/to/Qwen3-1.7B \\
        --moonvit-path /path/to/moonshotai-MoonViT-SO-400M \\
        --output-path /path/to/qwen3_euro_vl_2b_hf
"""

import argparse
import glob

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from megatron.bridge.models.euro_vl.configuration_euro_vl import EuroVLConfig
from megatron.bridge.models.euro_vl.modeling_euro_vl_hf import EuroVLForConditionalGeneration
from megatron.bridge.models.euro_vl.moonvit import MoonViTConfig, MoonViTImageProcessor


# Qwen3's built-in vision special tokens (same ids Qwen3-VL uses). Nothing is added
# to the tokenizer — these are asserted to already exist.
_VISION_TOKENS = {
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
}

# Same ChatML template as eurollm_bridge.py, with Qwen3's vision placeholder strings.
_CHAT_TEMPLATE = "\n".join(
    [
        "{%- if messages[0].role == 'system' %}",
        "    {{- '<|im_start|>system\\n' }}",
        "    {%- if messages[0].content is string %}",
        "        {{- messages[0].content }}",
        "    {%- else %}",
        "        {%- for c in messages[0].content %}",
        "            {%- if 'text' in c %}{{- c.text }}{%- endif %}",
        "        {%- endfor %}",
        "    {%- endif %}",
        "    {{- '<|im_end|>\\n' }}",
        "{%- endif %}",
        "{%- for message in messages %}",
        "    {%- if message.role != 'system' %}",
        "        {{- '<|im_start|>' + message.role + '\\n' }}",
        "        {%- if message.content is string %}",
        "            {{- message.content }}",
        "        {%- else %}",
        "            {%- for c in message.content %}",
        "                {%- if c.type == 'image' or 'image' in c or 'image_url' in c %}",
        "                    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}",
        "                {%- elif c.type == 'video' or 'video' in c %}",
        "                    {{- '<|vision_start|><|video_pad|><|vision_end|>' }}",
        "                {%- elif 'text' in c %}",
        "                    {{- c.text }}",
        "                {%- endif %}",
        "            {%- endfor %}",
        "        {%- endif %}",
        "        {{- '<|im_end|>\\n' }}",
        "    {%- endif %}",
        "{%- endfor %}",
        "{%- if add_generation_prompt %}",
        "    {{- '<|im_start|>assistant\\n' }}",
        "{%- endif %}",
    ]
)


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load and merge all safetensors shards from an HF model directory."""
    state: dict[str, torch.Tensor] = {}
    shards = sorted(glob.glob(f"{path}/*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors found in {path}")
    for shard in shards:
        state.update(load_file(shard, device="cpu"))
    return {k: v.to(torch.bfloat16) for k, v in state.items()}


def assemble(qwen3_path: str, moonvit_path: str, output_path: str) -> None:
    """Assemble a Qwen3EuroVL HF checkpoint from Qwen3 + MoonViT-SO-400M into ``output_path``."""
    print("🔄 Assembling Qwen3EuroVL HF checkpoint")
    print(f"   LLM:    {qwen3_path}")
    print(f"   Vision: {moonvit_path}")
    print(f"   Output: {output_path}")

    # 1. Build EuroVLConfig from the Qwen3 + MoonViT configs. No vocab extension —
    #    the vision tokens live inside Qwen3's base vocab. Token ids = Qwen3-VL's.
    print("\n⚙️  Building EuroVLConfig (qwen3 text backbone) ...")
    text_config = Qwen3Config.from_pretrained(qwen3_path)
    vision_config = MoonViTConfig.from_pretrained(moonvit_path)
    config = EuroVLConfig(
        text_config=text_config,
        vision_config=vision_config,
        projector_output_dim=text_config.hidden_size,  # 2048 for Qwen3-1.7B
        image_token_id=_VISION_TOKENS["<|image_pad|>"],
        vision_start_token_id=_VISION_TOKENS["<|vision_start|>"],
        vision_end_token_id=_VISION_TOKENS["<|vision_end|>"],
        vision_pad_token_id=_VISION_TOKENS["<|vision_pad|>"],
        video_token_id=_VISION_TOKENS["<|video_pad|>"],
        tie_word_embeddings=text_config.tie_word_embeddings,
    )

    # 2. Instantiate the model (projector randomly initialised; AutoModelForCausalLM
    #    resolves text_config.model_type == "qwen3" -> Qwen3ForCausalLM, incl. QK-norm).
    print("🏗️  Instantiating EuroVLForConditionalGeneration (Qwen3 backbone) ...")
    model = EuroVLForConditionalGeneration(config).to(torch.bfloat16)

    # 3. Load Qwen3 weights into language_model.* as-is (no embedding extension).
    #    Tied embeddings: the checkpoint has no lm_head.weight — that missing key is
    #    expected (lm_head shares storage with embed_tokens after tying).
    print("⬇️  Loading Qwen3 weights into language_model.* ...")
    llm_state = _load_state_dict(qwen3_path)
    missing, unexpected = model.language_model.load_state_dict(llm_state, strict=False)
    assert not unexpected, f"Unexpected Qwen3 keys: {unexpected[:5]}"
    allowed_missing = {"lm_head.weight"} if text_config.tie_word_embeddings else set()
    assert set(missing) <= allowed_missing, f"Missing Qwen3 keys: {missing[:5]}"
    model.language_model.tie_weights()

    # 4. Load MoonViT pretrained weights into vision_tower.*.
    print("⬇️  Loading MoonViT weights into vision_tower.* ...")
    mv_state = _load_state_dict(moonvit_path)
    model.vision_tower.load_state_dict(mv_state, strict=True)

    # 5. Projector stays random (trained in stage PA). Save the model.
    print(f"\n💾 Saving HF model to {output_path} ...")
    model.save_pretrained(output_path)

    # 6. Tokenizer: assert the vision tokens already exist at the expected Qwen ids
    #    (nothing is added), set the vision-aware ChatML template, save.
    print("🔤 Saving tokenizer (built-in vision tokens, chat template) ...")
    tokenizer = AutoTokenizer.from_pretrained(qwen3_path)
    for token, expected_id in _VISION_TOKENS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        assert actual_id == expected_id, f"{token!r} got ID {actual_id}, expected {expected_id}"
    tokenizer.chat_template = _CHAT_TEMPLATE
    tokenizer.save_pretrained(output_path)

    # 7. Save the MoonViT image-processor config so Qwen3EuroVLProcessor can load
    #    everything from a single directory.
    print("🖼️  Saving MoonViT image-processor config ...")
    MoonViTImageProcessor.from_pretrained(moonvit_path).save_pretrained(output_path)

    print("\n✅ Qwen3EuroVL HF checkpoint assembled successfully.")


def main() -> None:
    """CLI entry point: parse args and assemble the Qwen3EuroVL HF checkpoint."""
    parser = argparse.ArgumentParser(description="Assemble Qwen3EuroVL HF checkpoint from Qwen3 + MoonViT-SO-400M.")
    parser.add_argument("--qwen3-path", required=True, help="Path to Qwen3 HF model (e.g. Qwen3-1.7B).")
    parser.add_argument("--moonvit-path", required=True, help="Path to MoonViT-SO-400M HF model.")
    parser.add_argument("--output-path", required=True, help="Directory to write the Qwen3EuroVL HF checkpoint.")
    args = parser.parse_args()

    assemble(args.qwen3_path, args.moonvit_path, args.output_path)


if __name__ == "__main__":
    main()
    # uv run --no-sync python qwen3_euro_vl_bridge.py \
    #     --qwen3-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/Qwen3-1.7B' \
    #     --moonvit-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/moonshotai-MoonViT-SO-400M' \
    #     --output-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/qwen3_euro_vl_2b_hf'
