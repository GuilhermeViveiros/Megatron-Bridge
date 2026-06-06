#!/usr/bin/env bash
set -euo pipefail

SIF=/e/scratch/e-ext-2025e01-100/viveiros1/singularity/megatron-bridge.sif
SCRATCH=/e/scratch/e-ext-2025e01-100/viveiros1

# uv uses the container's /opt/venv (which has torch) and skips syncing.
# megatron.bridge is importable via PYTHONPATH from the mounted source tree.
# uv cache goes to scratch to avoid filling the container's /opt/uv_cache.
#
# Run eurollm_bridge.py:
#   uv run --no-sync python eurollm_bridge.py \
#       --eurollm-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/EuroLLM-1.7B-Instruct' \
#       --moonvit-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/moonshotai-MoonViT-SO-400M' \
#       --output-path '/e/scratch/e-ext-2025e01-100/viveiros1/eurovlm-data/models/megatron/euro_vl_2b'


apptainer shell \
    --nv \
    --writable-tmpfs \
    --bind "$HOME/Megatron-Bridge":/opt/Megatron-Bridge \
    --bind "$SCRATCH":/scratch \
    --bind "$SCRATCH/.cache/huggingface":/root/.cache/ \
    --env HF_HOME=/root/.cache/huggingface \
    --env UV_CACHE_DIR=/scratch/uv_cache \
    --pwd /opt/Megatron-Bridge \
    "$SIF"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 eurollm_bridge.py \
#     --eurollm-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/EuroLLM-1.7B-Instruct' \
#     --moonvit-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/moonshotai-MoonViT-SO-400M' \
#     --output-path '/e/scratch/e-ext-2025e01-100/viveiros1/eurovlm-data/models/megatron/euro_vl_2b'

# MASTER_ADDR=localhost MASTER_PORT=$((20000 + RANDOM % 20000)) uv run --no-sync scripts/training/run_recipe.py --recipe euro_vl_2b_sft_config --step_func vlm_step logger.log_interval=1 train.micro_batch_size=1
# uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py --recipe euro_vl_2b_sft_config --step_func vlm_step logger.log_interval=1 train.micro_batch_size=6