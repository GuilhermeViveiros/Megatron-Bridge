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


# Load local secrets (WANDB_API_KEY, …) from the gitignored .env if present.
if [ -f "$(dirname "$0")/.env" ]; then set -a; . "$(dirname "$0")/.env"; set +a; fi

# Container environment variables (passed through to the apptainer shell).
ENV_ARGS=(
    --env HF_HOME=/root/.cache/huggingface
    --env UV_CACHE_DIR=/scratch/uv_cache
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    --env PYTORCH_ALLOC_CONF=expandable_segments:True
    --env OMP_NUM_THREADS=1
    --env TORCH_NCCL_AVOID_RECORD_STREAMS=1
    --env NCCL_NVLS_ENABLE=0
    --env CUDA_DEVICE_MAX_CONNECTIONS=1
    --env NCCL_GRAPH_REGISTER=0
    # W&B: pass the key by REFERENCE from the shell (never hardcode it here).
    #   export WANDB_API_KEY=<your-key>   before running this script.
    # Add  --env WANDB_MODE=offline  if the compute node has no internet (then `wandb sync` later).
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}"
)

apptainer shell \
    --nv \
    --writable-tmpfs \
    --bind "$HOME/Megatron-Bridge":/opt/Megatron-Bridge \
    --bind "$SCRATCH":/scratch \
    --bind "$SCRATCH/.cache/huggingface":/root/.cache/ \
    "${ENV_ARGS[@]}" \
    --pwd /opt/Megatron-Bridge \
    "$SIF"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 eurollm_bridge.py \
#     --eurollm-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/EuroLLM-1.7B-Instruct' \
#     --moonvit-path '/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/moonshotai-MoonViT-SO-400M' \
#     --output-path '/e/scratch/e-ext-2025e01-100/viveiros1/eurovlm-data/models/megatron/euro_vl_2b'

# MASTER_ADDR=localhost MASTER_PORT=$((20000 + RANDOM % 20000)) 
# uv run --no-sync scripts/training/run_recipe.py --recipe euro_vl_2b_sft_config --step_func vlm_step logger.log_interval=1 train.micro_batch_size=1

# MASTER_ADDR=localhost MASTER_PORT=$((20000 + RANDOM % 20000)) uv run --no-sync python scripts/training/run_recipe.py --recipe euro_vl_2b_sft_energon_config --step_func vlm_step dataset.micro_batch_size=1 dataset.num_workers=4 checkpoint.pretrained_checkpoint=/e/scratch/e-ext-2025e01-100/viveiros1/hf_models/euro_vl_2b_hf logger.log_interval=1 dataset.root=/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data

# uv run --no-sync python -m torch.distributed.run --nproc_per_node=4 scripts/training/run_recipe.py --recipe euro_vl_2b_sft_energon_config --step_func vlm_step logger.log_interval=1 train.global_batch_size=128 train.micro_batch_size=1 dataset.num_workers=16 dataset.packing_buffer_size=256 model.cross_entropy_fusion_impl=te

# uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py --recipe qwen3_euro_vl_pa_sft_config --step_func vlm_step logger.log_interval=1 train.micro_batch_size=1 dataset.num_workers=4 train.global_batch_size=128

# qwen3_euro_vl_sft_energon_config
# dataset.packing_buffer_size=256 model.cross_entropy_fusion_impl=te