#!/usr/bin/env bash
# ==============================================================================
# EuroVL PA (projector-alignment) run, v2: same pixmo-cap-only setup as
# scripts/slurm_pa_pixmo_only.sh, with two additional corrections found while
# investigating the EuroVL-vs-TowerVision quality gap:
#
#   1. sqrt_loss_weighting=False (code fix in qwen3_euro_vl_pa_sft_config): the
#      PA recipe pairs calculate_per_token_loss=False + average_in_collective=True
#      (local per-microbatch mean loss), but the base config's task_encoder was
#      built with sqrt_loss_weighting=True (per-sample 1/sqrt(N) loss_mask
#      scaling) -- a mismatch that only works with calculate_per_token_loss=True
#      + average_in_collective=False. Left mismatched, the loss normalization
#      becomes pack-composition-dependent instead of the intended global Σwℓ/Σw.
#      NOTE: v1 (pixmo_only) already had this bug -- this run isolates the fix.
#
#   2. weight_decay=0.0 (CLI override): TowerVision's real LLaVA-NeXT tape
#      (deep-spin/LLaVA-NeXT tapes/main.tape + vblocks.tconf, PretrainModel
#      task) uses --weight_decay 0. explicitly. Our optimizer helper defaults
#      to weight_decay=0.1, which was NOT matched in v1 -- a real, previously
#      uncontrolled difference for a small randomly-initialized 2-layer MLP
#      projector trained from scratch over only 1797 steps.
#
# LR (1e-3) and warmup (90 iters ~= 5% of 1797) were ALREADY matched to
# TowerVision in v1 (pixmo_only) -- kept unchanged here. min_lr moved from
# 1e-4 (a floor) to ~0, matching TowerVision's cosine-to-zero (no eta_min).
#
# pixmo-cap has 690,000 training samples; at GBS=384, 1 epoch = ceil(690000/384)
# = 1797 (unchanged from v1).
#
# Usage:
#   sbatch scripts/slurm_pa_pixmo_only_v2.sh
# ==============================================================================

#SBATCH --job-name=qwen3-eurovl-pa-pixmo-only-v2
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_pa/slurm_pixmo_only_v2_%j.out
#SBATCH --error=logs/qwen3_pa/slurm_pixmo_only_v2_%j.err

set -euo pipefail

RECIPE="qwen3_euro_vl_pa_sft_config"
NUM_WORKERS=8
PACK_BUF=60
GBS=384
SAVE_INTERVAL=100
TRAIN_ITERS=1797
SAVE_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/euro_vl_runs/qwen3_pa_pixmo_only_v2"
RUN_NAME="qwen3-pa-pixmo-only-v2"
MIXTURE_FILE="/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data/mixture_pa_pixmo_only.yaml"

cd "$HOME/Megatron-Bridge"

echo "=== $(date) | job $SLURM_JOB_ID | nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) ==="
echo "expect world_size = 4 * $SLURM_JOB_NUM_NODES = $((4 * SLURM_JOB_NUM_NODES))"
echo "recipe=$RECIPE gbs=$GBS iters=$TRAIN_ITERS mixture=$MIXTURE_FILE"
echo "save_dir=$SAVE_DIR run_name=$RUN_NAME"

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
        optimizer.min_lr=0.000001 \
        optimizer.weight_decay=0.0 \
        scheduler.lr_warmup_iters=90 \
        scheduler.lr_decay_iters="$TRAIN_ITERS"

echo "=== $(date) | job $SLURM_JOB_ID done ==="
