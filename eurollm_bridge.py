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

"""Assemble an EuroVL Megatron checkpoint from EuroLLM-1.7B + MoonViT-SO-400M.

EuroVL is trained from scratch — there is no pre-existing HF VLM checkpoint.
This script bootstraps the initial Megatron checkpoint by:
  1. Loading EuroLLM-1.7B-Instruct HF weights into the language_model.* component.
  2. Loading MoonViT-SO-400M pretrained weights into the vision_tower.* component.
  3. Leaving the projector at random initialisation (trained in Stage PA).

Usage::

    uv run python eurollm_bridge.py \\
        --eurollm-path /path/to/EuroLLM-1.7B-Instruct \\
        --moonvit-path /path/to/moonshotai-MoonViT-SO-400M \\
        --output-path /path/to/megatron_checkpoint
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

from megatron.bridge.models.euro_vl.euro_vl_bridge import EuroVLBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.training.model_load_save import save_megatron_model


def assemble(eurollm_path: str, moonvit_path: str, output_path: str) -> None:
    print("🔄 Assembling EuroVL checkpoint")
    print(f"   LLM:    {eurollm_path}")
    print(f"   Vision: {moonvit_path}")
    print(f"   Output: {output_path}")

    # 1. Load EuroLLM HF model.
    print("\n📥 Loading EuroLLM weights ...")
    eurollm = PreTrainedCausalLM.from_pretrained(eurollm_path, torch_dtype=torch.bfloat16)

    # 2. Build EuroVLModelProvider from EuroLLM config + MoonViT vision config.
    print("⚙️  Building EuroVLModelProvider ...")
    bridge = EuroVLBridge()
    provider = bridge.provider_bridge(eurollm)
    provider.vision_config = AutoConfig.from_pretrained(moonvit_path, trust_remote_code=True)
    provider.finalize()

    # 3. Instantiate EuroVL Megatron model (projector randomly initialised).
    print("🏗️  Instantiating EuroVL model ...")
    model = provider.provide_distributed_model(wrap_with_ddp=False)

    # 4. Load EuroLLM weights → language_model.* via bridge mapping registry.
    print("⬇️  Loading EuroLLM weights into language_model.* ...")
    bridge.load_weights_hf_to_megatron(eurollm, model)

    # 5. Load MoonViT pretrained weights → vision_tower.*.
    print("⬇️  Loading MoonViT weights into vision_tower.* ...")
    moonvit = AutoModel.from_pretrained(moonvit_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model[0].vision_tower.load_state_dict(moonvit.state_dict())
    del moonvit

    # 6. Save Megatron checkpoint (weights only, no optimizer state).
    print(f"\n💾 Saving Megatron checkpoint to {output_path} ...")
    save_megatron_model(model, output_path, hf_tokenizer_path=eurollm_path)

    checkpoint_path = Path(output_path)
    if checkpoint_path.exists():
        print("📁 Checkpoint structure:")
        for item in sorted(checkpoint_path.iterdir()):
            print(f"   {'📂' if item.is_dir() else '📄'} {item.name}")

    print("\n✅ EuroVL checkpoint assembled successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble EuroVL Megatron checkpoint from EuroLLM-1.7B + MoonViT-SO-400M."
    )
    parser.add_argument("--eurollm-path", required=True, help="Path to EuroLLM-1.7B-Instruct HF model.")
    parser.add_argument("--moonvit-path", required=True, help="Path to MoonViT-SO-400M HF model.")
    parser.add_argument("--output-path", required=True, help="Directory to write the Megatron checkpoint.")
    args = parser.parse_args()

    assemble(args.eurollm_path, args.moonvit_path, args.output_path)


if __name__ == "__main__":
    main()
    # uv run python eurollm_bridge.py \
    #     --eurollm-path '/e/scratch/jureap126/gviveiros/hf_models/EuroLLM-1.7B-Instruct' \
    #     --moonvit-path '/e/scratch/jureap126/gviveiros/hf_models/moonshotai-MoonViT-SO-400M' \
    #     --output-path '/e/scratch/jureap126/gviveiros/eurovlm-data/models/megatron/euro_vl_2b'
