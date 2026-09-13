#!/usr/bin/env bash
# ==============================================================================
# EuroVL PA (projector-alignment) run using ONLY pixmo-cap (no multiimage/video)
# with TowerVision-2B-stage1-projector-matching hyperparameters (lr=1e-3, ~1
# epoch), to isolate whether EuroVL's image-captioning quality gap vs
# TowerVision is explained by task dilution / a too-low learning rate, rather
# than a code bug (ruled out separately this session -- see the
# project_eurovl_vision_grounding_hallucination memory).
#
# pixmo-cap has 690,000 training samples (confirmed via megatron.energon's own
# dataset object, not estimated) -- at GBS=384, 1 epoch = ceil(690000/384) = 1797.
#
# Usage:
#   sbatch scripts/slurm_pa_pixmo_only.sh
# ==============================================================================

#SBATCH --job-name=qwen3-eurovl-pa-pixmo-only
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_pa/slurm_pixmo_only_%j.out
#SBATCH --error=logs/qwen3_pa/slurm_pixmo_only_%j.err

set -euo pipefail

RECIPE="qwen3_euro_vl_pa_sft_config"
NUM_WORKERS=8
PACK_BUF=60
GBS=384
SAVE_INTERVAL=100
TRAIN_ITERS=1797
SAVE_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/euro_vl_runs/qwen3_pa_pixmo_only"
RUN_NAME="qwen3-pa-pixmo-only"
MIXTURE_FILE="/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data/mixture_pa_pixmo_only.yaml"

cd "$HOME/Megatron-Bridge"

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
