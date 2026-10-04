#!/usr/bin/env bash
# ==============================================================================
# Qwen3EuroVL-2B full SFT — full mixture.yaml, 64 nodes (256 GPUs).
#
# The `normal` QoS caps a job at 12 h, so one epoch (~18,615 iterations) cannot
# run in a single job. This script CHAINS itself: each job trains for
# EXIT_MINS minutes, saves, exits cleanly, and resubmits the next link until
# latest_checkpointed_iteration.txt reaches TRAIN_ITERS. checkpoint.load ==
# checkpoint.save, so every link resumes from the previous one's checkpoint.
#
#   sbatch scripts/slurm_qwen3_sft_full_64n.sh
#   NNODES=32 sbatch ... scripts/slurm_qwen3_sft_full_64n.sh   # fewer nodes
#   CHAIN=0  sbatch scripts/slurm_qwen3_sft_full_64n.sh        # single link only
# ==============================================================================

#SBATCH --job-name=qwen3-sft-full
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen3_sft_full/slurm_%j.out
#SBATCH --error=logs/qwen3_sft_full/slurm_%j.err

set -euo pipefail
mkdir -p logs/qwen3_sft_full

RECIPE="${RECIPE:-qwen3_euro_vl_2b_sft_config}"
TRAIN_ITERS="${TRAIN_ITERS:-18615}"      # 1 epoch of the full mixture (156.2B tokens / 8.39M per iter)
GBS="${GBS:-1024}"
# ~3.7 h apart at the measured ~6-7 s/iter on 64 nodes -> 9 checkpoints, 0.26 TB,
# and 3 saves inside each 11 h chain link (the QoS wall is 12 h).
# Each link also saves unconditionally when exit_duration_in_mins fires
# (train.py: save_checkpoint_and_time before exit), so this only bounds loss
# from an UNEXPECTED crash inside a link, not the hand-off between links.
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PACK_BUF="${PACK_BUF:-256}"
# Stop and save this many minutes in, leaving head-room under the 12 h wall for
# the final async checkpoint to flush before Slurm kills the step.
EXIT_MINS="${EXIT_MINS:-660}"            # 11 h of a 12 h job
SAVE_DIR="${SAVE_DIR:-/e/scratch/e-ext-2025e01-100/viveiros1/euro_vl_runs/qwen3_sft_full_mix}"
RUN_NAME="${RUN_NAME:-qwen3-sft-full-mix}"
CHAIN="${CHAIN:-1}"
# Slurm runs a SPOOLED COPY of this file, so $0 is not a usable path for resubmission.
SELF="$HOME/Megatron-Bridge/scripts/slurm_qwen3_sft_full_64n.sh"

TRACKER="$SAVE_DIR/latest_checkpointed_iteration.txt"
START_ITER=$( [ -f "$TRACKER" ] && cat "$TRACKER" || echo 0 )

# 72 cores/task are now visible; cap OMP so torch does not spawn 72 threads
# per rank on top of the dataloader workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$HOME/Megatron-Bridge"
mkdir -p "$SAVE_DIR"

echo "=== $(date) | job ${SLURM_JOB_ID} | nodes=${SLURM_JOB_NUM_NODES} | world=$((4 * SLURM_JOB_NUM_NODES)) ==="
echo "recipe=$RECIPE  train_iters=$TRAIN_ITERS  resuming_from=$START_ITER"
echo "gbs=$GBS save_interval=$SAVE_INTERVAL eval=$EVAL_ITERS/$EVAL_INTERVAL exit_after=${EXIT_MINS}min"
echo "save_dir=$SAVE_DIR"
df -h /e/scratch/e-ext-2025e01-100 | tail -1

srun --gpu-bind=none ./apptainer.sh \
    uv run --no-sync python scripts/training/run_recipe.py \
        --recipe "$RECIPE" --step_func vlm_step \
        logger.log_interval=10 \
        logger.wandb_exp_name="$RUN_NAME" \
        logger.wandb_save_dir="$SAVE_DIR/wandb" \
        train.micro_batch_size=1 \
        train.global_batch_size="$GBS" \
        train.train_iters="$TRAIN_ITERS" \
        train.exit_duration_in_mins="$EXIT_MINS" \
        train.exit_signal_handler=true \
        dataset.num_workers="$NUM_WORKERS" \
        dataset.packing_buffer_size="$PACK_BUF" \
        model.tensor_model_parallel_size=1 \
        checkpoint.save="$SAVE_DIR" \
        checkpoint.load="$SAVE_DIR" \
        checkpoint.save_interval="$SAVE_INTERVAL" \
        validation.eval_interval="$EVAL_INTERVAL" \
        validation.eval_iters="$EVAL_ITERS"

RC=$?
END_ITER=$( [ -f "$TRACKER" ] && cat "$TRACKER" || echo 0 )
echo "=== $(date) | job ${SLURM_JOB_ID} finished rc=$RC | iteration $START_ITER -> $END_ITER ==="

# Chain the next link unless we are done, the run made no progress (a real
# failure -- do not spin), or chaining was disabled.
if [ "$CHAIN" = "1" ] && [ "$END_ITER" -lt "$TRAIN_ITERS" ]; then
    if [ "$END_ITER" -le "$START_ITER" ]; then
        echo "NO PROGRESS ($START_ITER -> $END_ITER): not resubmitting. Investigate before rerunning."
        exit 1
    fi
    NEXT=$(sbatch --parsable --nodes="${SLURM_JOB_NUM_NODES}" "$SELF")
    echo "chained next link: job $NEXT  ($END_ITER / $TRAIN_ITERS done)"
else
    echo "chain complete or disabled ($END_ITER / $TRAIN_ITERS)."
fi
