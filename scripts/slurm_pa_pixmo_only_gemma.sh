#!/usr/bin/env bash
# ==============================================================================
# GemmaEuroVL PA (projector-alignment) run: Tower-Plus-2B (Gemma2-2B) backbone +
# MoonViT, on pixmo-cap only.
#
# The LLM-backbone arm of the EuroVL-vs-TowerVision investigation. TowerVision's
# own PA stage beats our Qwen3 PA -- and even our full SFT -- on matched
# pixmo-cap-only data, and TowerVision's backbone is Tower-Plus-2B. This run puts
# that backbone in OUR framework, so the remaining gap can be attributed to the
# backbone or to the framework/data pipeline, but not to both at once.
#
# Deliberately identical to scripts/slurm_pa_pixmo_only.sh in every hyperparameter
# (GBS=384, 1797 iters = 1 epoch of pixmo-cap's 690,000 samples, lr=1e-3, warmup
# 90, PACK_BUF=60, same mixture file). Only the recipe and output paths differ.
#
# Prerequisites (one-off, already done -- rerun only if the HF dir is rebuilt):
#   uv run --no-sync python sanity_check/gemma_euro_vl_assembly.py \
#       --tower-path   $SCRATCH/hf_models/Tower-Plus-2B \
#       --moonvit-path $SCRATCH/hf_models/moonshotai-MoonViT-SO-400M \
#       --output-path  $SCRATCH/hf_models/gemma_euro_vl_hf
#   uv run --no-sync python examples/conversion/convert_checkpoints.py import \
#       --hf-model      $SCRATCH/hf_models/gemma_euro_vl_hf \
#       --megatron-path $SCRATCH/megatron_models/gemma_euro_vl \
#       --torch-dtype bfloat16
#
# Usage:
#   sbatch scripts/slurm_pa_pixmo_only_gemma.sh
# ==============================================================================

#SBATCH --job-name=gemma-eurovl-pa-pixmo-only
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/gemma_pa/slurm_pixmo_only_%j.out
#SBATCH --error=logs/gemma_pa/slurm_pixmo_only_%j.err

set -euo pipefail

RECIPE="gemma_euro_vl_pa_sft_config"
NUM_WORKERS=8
PACK_BUF=60
GBS=384
SAVE_INTERVAL=100
TRAIN_ITERS=1797
SAVE_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/euro_vl_runs/gemma_pa_pixmo_only"
RUN_NAME="gemma-pa-pixmo-only"
MIXTURE_FILE="/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data/mixture_pa_pixmo_only.yaml"

cd "$HOME/Megatron-Bridge"

mkdir -p logs/gemma_pa

echo "=== $(date) | job $SLURM_JOB_ID | nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) ==="
echo "expect world_size = 4 * $SLURM_JOB_NUM_NODES = $((4 * SLURM_JOB_NUM_NODES))"
echo "recipe=$RECIPE gbs=$GBS iters=$TRAIN_ITERS mixture=$MIXTURE_FILE"
echo "save_dir=$SAVE_DIR run_name=$RUN_NAME"

# srun-native: inherits the header allocation (nodes x ntasks-per-node), so this scales with
# --nodes alone. Bridge derives RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT from SLURM.
srun --gpu-bind=none ./apptainer.sh \
    uv run --no-sync python scripts/training/run_recipe.py \
        --recipe "$RECIPE" --step_func vlm_step \
        logger.log_interval=1 \
        train.micro_batch_size=1 \
        train.train_iters="$TRAIN_ITERS" \
        train.global_batch_size="$GBS" \
        dataset.num_workers="$NUM_WORKERS" \
        dataset.packing_buffer_size="$PACK_BUF" \
        dataset.mixture_file="$MIXTURE_FILE" \
        checkpoint.save_interval="$SAVE_INTERVAL" \
        checkpoint.save="$SAVE_DIR" \
        checkpoint.load="$SAVE_DIR" \
        logger.wandb_exp_name="$RUN_NAME" \
        logger.wandb_save_dir="$SAVE_DIR/wandb" \
        optimizer.lr=0.001 \
        optimizer.min_lr=0.0001 \
        scheduler.lr_warmup_iters=90 \
        scheduler.lr_decay_iters="$TRAIN_ITERS"

echo "=== $(date) | job $SLURM_JOB_ID done ==="
