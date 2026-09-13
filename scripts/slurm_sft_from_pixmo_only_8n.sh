#!/usr/bin/env bash
# ==============================================================================
# Small full-SFT round starting from pixmo_only's PA-aligned checkpoint, to test
# whether unfreezing everything (vision tower + LLM + projector, matching
# TowerVision's actual finetune recipe: mm_vision_tower,mm_mlp_adapter,
# mm_language_model all trainable) closes the quality gap -- the leading
# hypothesis from this session's TowerVision+MoonViT/Qwen3VE investigation
# (MoonViT's full finetune in TowerVision's framework beat every PA-only
# checkpoint, including SigLIP2's).
#
# Single-variable change from pixmo_only: SAME pixmo-cap-only data, SAME
# starting weights (resumed as `pretrained_checkpoint`, not `checkpoint.load`,
# so this starts a FRESH optimizer/LR schedule/iteration count at 0, not a
# continuation of pixmo_only's already-fully-annealed schedule). Only the
# freeze flags change: nothing frozen (recipe default), vs pixmo_only's
# frozen-LLM+frozen-vision PA setup.
#
# LR: no per-submodule vision/LLM LR split exists in this codebase (checked),
# so a single conservative LR is used, in TowerVision's finetune LLM-LR
# ballpark (1e-5) rather than PA-stage-style 1e-3 -- full-model finetuning at
# PA-scale LR would risk catastrophic forgetting of the pretrained LLM/vision
# weights this run is specifically trying to ADAPT, not destroy.
#
# Usage:
#   sbatch scripts/slurm_sft_from_pixmo_only.sh
# ==============================================================================

#SBATCH --job-name=qwen3-eurovl-sft-from-pixmo-only-8n
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_sft/slurm_sft_from_pixmo_only_%j.out
#SBATCH --error=logs/qwen3_sft/slurm_sft_from_pixmo_only_%j.err

set -euo pipefail

RECIPE="qwen3_euro_vl_sft_energon_config"
NUM_WORKERS=8
PACK_BUF=256
GBS=384
SAVE_INTERVAL=100
TRAIN_ITERS=600
PRETRAINED_CKPT="/e/scratch/e-ext-2025e01-100/viveiros1/euro_vl_runs/qwen3_pa_pixmo_only"
SAVE_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/euro_vl_runs/qwen3_sft_from_pixmo_only_8n"
RUN_NAME="qwen3-sft-from-pixmo-only-8n"
MIXTURE_FILE="/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data/mixture_pa_pixmo_only.yaml"

cd "$HOME/Megatron-Bridge"

echo "=== $(date) | job $SLURM_JOB_ID | nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) ==="
echo "expect world_size = 4 * $SLURM_JOB_NUM_NODES = $((4 * SLURM_JOB_NUM_NODES))"
echo "recipe=$RECIPE gbs=$GBS iters=$TRAIN_ITERS mixture=$MIXTURE_FILE"
echo "pretrained_ckpt=$PRETRAINED_CKPT (fresh optimizer/schedule, NOT a resume)"
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
        dataset.global_batch_size="$GBS" \
        dataset.mixture_file="$MIXTURE_FILE" \
        checkpoint.pretrained_checkpoint="$PRETRAINED_CKPT" \
        checkpoint.save_interval="$SAVE_INTERVAL" \
        checkpoint.save="$SAVE_DIR" \
        checkpoint.load="$SAVE_DIR" \
        logger.wandb_exp_name="$RUN_NAME" \
        logger.wandb_save_dir="$SAVE_DIR/wandb" \
        optimizer.lr=0.00001 \
        optimizer.min_lr=0.000001 \
        scheduler.lr_warmup_iters=30 \
        scheduler.lr_decay_iters="$TRAIN_ITERS"

echo "=== $(date) | job $SLURM_JOB_ID done ==="
