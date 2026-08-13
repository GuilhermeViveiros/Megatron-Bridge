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

"""Assemble an EuroVL **HF** checkpoint from EuroLLM-1.7B + MoonViT-SO-400M.

EuroVL is trained from scratch — there is no pre-existing HF VLM checkpoint. This
script bootstraps the initial HF checkpoint so that ``AutoBridge`` can later
convert it to Megatron (clean separation: assemble once -> HF checkpoint -> bridge).

It builds an ``EuroVLForConditionalGeneration`` and:
  1. Loads EuroLLM-1.7B-Instruct weights into ``language_model.*`` (Llama decoder),
     extending the 128000-row embedding/lm_head to 128005 with mean-initialised
     rows for the 5 vision special tokens (LLaVA-NeXT convention).
  2. Loads MoonViT-SO-400M pretrained weights into ``vision_tower.*``.
  3. Leaves ``multi_modal_projector.*`` at random init (trained in stage PA).
  4. Saves the model, the extended tokenizer (vision tokens + chat template), and
     the MoonViT image-processor config to the output dir via ``save_pretrained``.

Usage (in container)::

    uv run --no-sync python eurollm_bridge.py \\
        --eurollm-path /path/to/EuroLLM-1.7B-Instruct \\
        --moonvit-path /path/to/moonshotai-MoonViT-SO-400M \\
        --output-path /path/to/euro_vl_2b_hf
"""

import argparse
import glob

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig

from megatron.bridge.models.euro_vl.configuration_euro_vl import EuroVLConfig
from megatron.bridge.models.euro_vl.modeling_euro_vl_hf import EuroVLForConditionalGeneration
from megatron.bridge.models.euro_vl.moonvit import MoonViTConfig, MoonViTImageProcessor


_NUM_VISION_TOKENS = 5
_VISION_TOKENS = {
    "<image>": 128000,
    "<|vision_start|>": 128001,
    "<|vision_end|>": 128002,
    "<|vision_pad|>": 128003,
    "<video>": 128004,
}

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
        "                    {{- '<|vision_start|><image><|vision_end|>' }}",
        "                {%- elif c.type == 'video' or 'video' in c %}",
        "                    {{- '<|vision_start|><video><|vision_end|>' }}",
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


def assemble(eurollm_path: str, moonvit_path: str, output_path: str) -> None:
    """Assemble an EuroVL HF checkpoint from EuroLLM-1.7B + MoonViT-SO-400M into ``output_path``."""
    print("🔄 Assembling EuroVL HF checkpoint")
    print(f"   LLM:    {eurollm_path}")
    print(f"   Vision: {moonvit_path}")
    print(f"   Output: {output_path}")

    # 1. Build EuroVLConfig from the EuroLLM (Llama) + MoonViT configs.
    print("\n⚙️  Building EuroVLConfig ...")
    text_config = LlamaConfig.from_pretrained(eurollm_path)
    base_vocab = text_config.vocab_size  # 128000
    text_config.vocab_size = base_vocab + _NUM_VISION_TOKENS  # 128005
    vision_config = MoonViTConfig.from_pretrained(moonvit_path)
    config = EuroVLConfig(text_config=text_config, vision_config=vision_config)

    # 2. Instantiate the model (projector randomly initialised).
    print("🏗️  Instantiating EuroVLForConditionalGeneration ...")
    model = EuroVLForConditionalGeneration(config).to(torch.bfloat16)

    # 3. Load EuroLLM weights into language_model.*, extending embed/lm_head vocab.
    print("⬇️  Loading EuroLLM weights into language_model.* ...")
    llm_state = _load_state_dict(eurollm_path)
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        w = llm_state[key]  # [128000, hidden]
        mean_row = w.mean(dim=0, keepdim=True)
        llm_state[key] = torch.cat([w, mean_row.expand(_NUM_VISION_TOKENS, -1).clone()], dim=0)
    missing, unexpected = model.language_model.load_state_dict(llm_state, strict=False)
    assert not unexpected, f"Unexpected EuroLLM keys: {unexpected[:5]}"
    assert not missing, f"Missing EuroLLM keys: {missing[:5]}"

    # 4. Load MoonViT pretrained weights into vision_tower.*.
    print("⬇️  Loading MoonViT weights into vision_tower.* ...")
    mv_state = _load_state_dict(moonvit_path)
    model.vision_tower.load_state_dict(mv_state, strict=True)

    # 5. Projector stays random (trained in stage PA). Save the model.
    print(f"\n💾 Saving HF model to {output_path} ...")
    model.save_pretrained(output_path)

    # 6. Extend the tokenizer with the 5 vision special tokens + chat template, save.
    print("🔤 Saving extended tokenizer (+ vision tokens, chat template) ...")
    tokenizer = AutoTokenizer.from_pretrained(eurollm_path)
    tokenizer.add_special_tokens({"additional_special_tokens": list(_VISION_TOKENS)})
    for token, expected_id in _VISION_TOKENS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        assert actual_id == expected_id, f"{token!r} got ID {actual_id}, expected {expected_id}"
    tokenizer.chat_template = _CHAT_TEMPLATE
    tokenizer.save_pretrained(output_path)

    # 7. Save the MoonViT image-processor config so EuroVLProcessor can load
    #    everything from a single directory. The video processor shares this config
    #    (it only adds num_frames at runtime), so one preprocessor_config.json suffices.
    print("🖼️  Saving MoonViT image-processor config ...")
    MoonViTImageProcessor.from_pretrained(moonvit_path).save_pretrained(output_path)

    print("\n✅ EuroVL HF checkpoint assembled successfully.")


def main() -> None:
    """CLI entry point: parse args and assemble the EuroVL HF checkpoint."""
    parser = argparse.ArgumentParser(description="Assemble EuroVL HF checkpoint from EuroLLM-1.7B + MoonViT-SO-400M.")
    parser.add_argument("--eurollm-path", required=True, help="Path to EuroLLM-1.7B-Instruct HF model.")
    parser.add_argument("--moonvit-path", required=True, help="Path to MoonViT-SO-400M HF model.")
    parser.add_argument("--output-path", required=True, help="Directory to write the EuroVL HF checkpoint.")
    args = parser.parse_args()

    assemble(args.eurollm_path, args.moonvit_path, args.output_path)


if __name__ == "__main__":
    main()
    # uv run --no-sync python eurollm_bridge.py \
    #     --eurollm-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/EuroLLM-1.7B-Instruct' \
    #     --moonvit-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/moonshotai-MoonViT-SO-400M' \
    #     --output-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/euro_vl_2b_hf'
