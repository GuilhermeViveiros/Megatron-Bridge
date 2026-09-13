#!/usr/bin/env bash
# ==============================================================================
# EuroVL captioning eval sweep — one Slurm array task per checkpoint (6 total:
# 10/50/100%-of-training for both the PIL-backend run (qwen3_pa_pyav) and
# the vectorized-backend run (qwen3_pa_pyav_vect)), each on its own node with
# its own local vLLM judge. Requires evals/scripts/prepare_manifests.sh to have
# already built evals/data/{image,multi_image,video}_eval.jsonl.
#
# Usage:
#   sbatch --dependency=afterok:<prep_job_id> evals/scripts/run_eval_array.sh
# ==============================================================================

#SBATCH --job-name=euro-vl-eval
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/euro_vl_eval/array_%A_%a.out
#SBATCH --error=logs/euro_vl_eval/array_%A_%a.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

# This cluster's compute nodes have no outbound network access (unlike the login node) —
# everything this job touches (checkpoints, tokenizer, MoonViT, the judge model) is already
# cached locally from evals/scripts/prepare_manifests.sh / interactive setup, so force
# huggingface_hub to skip the metadata check it'd otherwise make even for cached files.
export HF_HUB_OFFLINE=1

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-640}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer}"
MOONVIT_PATH="${MOONVIT_PATH:-/scratch/hf_models/moonshotai-MoonViT-SO-400M}"

# select_checkpoints.py runs directly on the compute node (plain python3, no container —
# it's pure filesystem globbing, no torch/network needed), so it needs real host paths.
# generate_captions.py, in contrast, runs inside the apptainer container where the host's
# $SCRATCH is bind-mounted to /scratch — HOST_SCRATCH here must match apptainer.sh's $SCRATCH.
HOST_SCRATCH="/e/scratch/e-ext-2025e01-100/viveiros1"
declare -A RUN_DIR=(
  [pil]="$HOST_SCRATCH/euro_vl_runs/qwen3_pa_pyav"
  [vect]="$HOST_SCRATCH/euro_vl_runs/qwen3_pa_pyav_vect"
)
# Task index -> (run, percentile). 0-2 = pil 10/50/100, 3-5 = vect 10/50/100.
COMBOS=(pil:10 pil:50 pil:100 vect:10 vect:50 vect:100)
combo="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
run="${combo%%:*}"
pct="${combo##*:}"
tag="${run}_${pct}"

echo "=== $(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURMD_NODENAME | $tag ==="

# Resolve this task's checkpoint from the run's own available checkpoint history.
ckpt_path=""
while IFS=$'\t' read -r p path; do
  if [ "$p" = "$pct" ]; then
    ckpt_path="$path"
  fi
done < <(python3 -m evals.scripts.select_checkpoints "${RUN_DIR[$run]}")
if [ -z "$ckpt_path" ]; then
  echo "FATAL: could not resolve checkpoint for $tag"
  exit 1
fi
# generate_captions.py runs inside the apptainer container, where apptainer.sh binds
# $SCRATCH (== $HOST_SCRATCH) to /scratch — translate the host path select_checkpoints.py
# returned into the container-internal path generate_captions.py's --checkpoint expects.
ckpt_container_path="/scratch${ckpt_path#"$HOST_SCRATCH"}"
echo "$tag -> $ckpt_path (container: $ckpt_container_path)"

# ── Judge server, managed within this task ──────────────────────────────
# NOTE: deliberately no `srun` wrapper here or below. This is a single-task,
# single-node job with no need for Slurm's task-launcher semantics — and by
# default srun steps do NOT share CPUs with other concurrent steps in the
# same job, so backgrounding this step under srun while also `srun`-ing
# generate_captions/judge below deadlocks forever ("Job step creation still
# disabled, retrying (Requested nodes are busy)") — hit live, all 8 tasks
# ran the full 12h and produced zero captions. Plain subprocess calls to
# ./apptainer.sh inherit the job's GPU/cgroup allocation fine without srun.
echo "--- starting local judge (vLLM) ---"
./apptainer.sh \
    uv run --no-sync vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --port 8000 --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.3 \
    > "logs/euro_vl_eval/vllm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "--- stopping vLLM (pid $VLLM_PID) $(date) ---"
  kill "$VLLM_PID" 2>/dev/null
  sleep 3
  kill -9 "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT

echo "--- waiting for judge to become healthy ---"
ready=0
for i in $(seq 1 60); do
  if curl -sS -m 3 http://localhost:8000/health >/dev/null 2>&1; then
    echo "judge healthy after $((i * 10))s"
    ready=1
    break
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  echo "judge never became healthy, aborting"
  exit 1
fi

# ── Generate + judge this task's checkpoint across all 3 modalities ─────
for modality in image multi_image video; do
  manifest="evals/data/${modality}_eval.jsonl"
  [ -s "$manifest" ] || { echo "skip $tag/$modality: no manifest"; continue; }
  echo "--- generate $tag ($modality) $(date) ---"
  ./apptainer.sh \
      uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 -m evals.generate_captions \
      --manifest "$manifest" \
      --checkpoint "$ckpt_container_path" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --moonvit_path "$MOONVIT_PATH" \
      --model_tag "$tag" --max_new_tokens "$MAX_NEW_TOKENS" --num_frames 6

  captions="evals/results/$tag/${modality}_captions.jsonl"
  [ -s "$captions" ] || { echo "skip judge $tag/$modality: no captions"; continue; }
  echo "--- judge $tag ($modality) $(date) ---"
  ./apptainer.sh uv run --no-sync python -m evals.judge \
      --manifest "$manifest" \
      --captions "$captions" \
      --model_tag "$tag" --judge_provider local --judge_model Qwen/Qwen2.5-32B-Instruct-AWQ
done

echo "=== $(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID done ($tag) ==="
