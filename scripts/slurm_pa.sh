#!/usr/bin/env bash

#SBATCH --job-name=qwen3-eurovl-pa_vect
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_pa/slurm_%j.out
#SBATCH --error=logs/qwen3_pa/slurm_%j.err

set -euo pipefail

# ── Tunables (env-overridable) ───────────────────────────────────────────
RECIPE="${RECIPE:-qwen3_euro_vl_pa_sft_config}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PACK_BUF="${PACK_BUF:-60}"
GBS="${GBS:-192}"          # 128 for scaling comparison; recipe default is 512 for real throughput
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
TRAIN_ITERS="${TRAIN_ITERS:-4000}"
# Per-run output dir + W&B name — CHANGE THESE ACROSS RUNS (e.g. A/B: pil vs vectorized).
# SAVE_DIR overrides the recipe's checkpoint.save/load (each run gets its own empty dir so it
# starts from the pretrained checkpoint, not a stale resume). Leave empty to use the recipe default.
SAVE_DIR="${SAVE_DIR:-}"          # e.g. SAVE_DIR=$SCRATCH/euro_vl_runs/qwen3_pa_pil
RUN_NAME="${RUN_NAME:-qwen3-pa}"  # W&B experiment name

cd "$HOME/Megatron-Bridge"

# Build optional checkpoint/logging overrides (only when SAVE_DIR is set).
EXTRA_ARGS=(logger.wandb_exp_name="$RUN_NAME")
if [ -n "$SAVE_DIR" ]; then
    EXTRA_ARGS+=(
        checkpoint.save="$SAVE_DIR"
        checkpoint.load="$SAVE_DIR"
        logger.wandb_save_dir="$SAVE_DIR/wandb"
    )
fi

echo "=== $(date) | job $SLURM_JOB_ID | nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) ==="
echo "expect world_size = 4 * $SLURM_JOB_NUM_NODES = $((4 * SLURM_JOB_NUM_NODES))"
echo "recipe=$RECIPE workers=$NUM_WORKERS pack_buf=$PACK_BUF gbs=$GBS save_interval=$SAVE_INTERVAL iters=$TRAIN_ITERS"
echo "save_dir=${SAVE_DIR:-<recipe default>} run_name=$RUN_NAME"

# srun-native: no --ntasks / --nproc here — it inherits the header allocation (nodes ×
# ntasks-per-node) so this scales with --nodes alone. Bridge derives RANK/WORLD_SIZE/LOCAL_RANK/
# MASTER_ADDR/MASTER_PORT from the SLURM env. --gpu-bind=none lets every task see all 4 GPUs so
# Bridge selects by LOCAL_RANK (drop it only if the cluster already binds per task).
srun --gpu-bind=none ./apptainer.sh \
    uv run --no-sync python scripts/training/run_recipe.py \
        --recipe "$RECIPE" --step_func vlm_step \
        logger.log_interval=1 \
        train.micro_batch_size=1 \
        train.train_iters="$TRAIN_ITERS" \
        train.global_batch_size="$GBS" \
        dataset.num_workers="$NUM_WORKERS" \
        dataset.packing_buffer_size="$PACK_BUF" \
        checkpoint.save_interval="$SAVE_INTERVAL" \
        "${EXTRA_ARGS[@]}"

echo "=== $(date) | job $SLURM_JOB_ID done ==="
