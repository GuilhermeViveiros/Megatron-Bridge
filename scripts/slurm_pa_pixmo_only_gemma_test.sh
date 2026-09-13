#!/usr/bin/env bash
# ==============================================================================
# GemmaEuroVL PA — 1-HOUR SMOKE TEST (not a real run).
#
# Purpose: prove the Gemma2/Tower-Plus-2B backbone arm actually trains end to end
# before committing an 8-node, 12-hour job -- i.e. that the Megatron checkpoint
# loads (loss must start well below ln(256000)=12.45, not at it), the energon
# pixmo-cap pipeline feeds it, packed forward/backward runs under TE attention
# with Gemma2's sliding-window config, and a checkpoint saves.
#
# Differences from scripts/slurm_pa_pixmo_only_gemma.sh (the real run):
#   - 2 nodes instead of 8, 1h instead of 12h (schedules far sooner via backfill)
#   - save_interval=50 so a checkpoint save happens inside the hour
#   - its own SAVE_DIR, so the real run still starts from a clean directory
# Everything else (GBS, iters, LR schedule, mixture, packing) is identical, so
# the loss curve is comparable -- but throughput is NOT (fewer nodes => more
# grad-accum micro-steps per iteration).
#
# Expect the job to end in TIMEOUT at the 1h wall clock. That is success here.
#
# Usage:
#   sbatch scripts/slurm_pa_pixmo_only_gemma_test.sh
# ==============================================================================

#SBATCH --job-name=gemma-eurovl-pa-test
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/gemma_pa/slurm_test_%j.out
#SBATCH --error=logs/gemma_pa/slurm_test_%j.err

set -euo pipefail

RECIPE="gemma_euro_vl_pa_sft_config"
NUM_WORKERS=8
PACK_BUF=60
GBS=384
SAVE_INTERVAL=50
TRAIN_ITERS=1797
SAVE_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/euro_vl_runs/gemma_pa_pixmo_only_test"
RUN_NAME="gemma-pa-pixmo-only-test"
MIXTURE_FILE="/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data/mixture_pa_pixmo_only.yaml"

cd "$HOME/Megatron-Bridge"

mkdir -p logs/gemma_pa

echo "=== $(date) | job $SLURM_JOB_ID | nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) ==="
echo "expect world_size = 4 * $SLURM_JOB_NUM_NODES = $((4 * SLURM_JOB_NUM_NODES))"
echo "recipe=$RECIPE gbs=$GBS iters=$TRAIN_ITERS mixture=$MIXTURE_FILE"
echo "save_dir=$SAVE_DIR run_name=$RUN_NAME"

# srun-native: inherits the header allocation (nodes x ntasks-per-node). Bridge derives
# RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT from SLURM.
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
