#!/usr/bin/env bash
# ==============================================================================
# Generate 200-sample video captions for vect_10/50/100 (the full 402-item run
# timed out mid-generation in job 1479887, losing all progress since captions
# are only written at the end of each modality), then judge BOTH the already-
# generated multi_image captions (593 items, complete, never judged after the
# job 1479887 timeout) and the fresh 200-item video captions in one judge
# session -- avoids paying vLLM's ~7-13min startup cost twice.
#
# Includes this session's fixes: attn_implementation="flash_attention_2"
# (evals/generate_captions.py), judge.py's IPv4-loopback fix + <think>-tag
# zero-score + --concurrency 16 (all baked into the code already).
#
# Usage:
#   sbatch evals/scripts/video200_and_judge_pending.sh
# ==============================================================================

#SBATCH --job-name=euro-vl-video200-judge
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/euro_vl_eval/video200_judge_%A_%a.out
#SBATCH --error=logs/euro_vl_eval/video200_judge_%A_%a.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

export HF_HUB_OFFLINE=1

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-640}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer}"
MOONVIT_PATH="${MOONVIT_PATH:-/scratch/hf_models/moonshotai-MoonViT-SO-400M}"

TAGS=(vect_10 vect_50 vect_100)
tag="${TAGS[$SLURM_ARRAY_TASK_ID]}"
hf_export="/scratch/hf_models/${tag}_export"

echo "=== $(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURMD_NODENAME | $tag ==="

VIDEO_MANIFEST="evals/data/video_eval_200.jsonl"
echo "--- generate $tag (video, 200 samples) $(date) ---"
./apptainer.sh \
    uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 -m evals.generate_captions \
    --manifest "$VIDEO_MANIFEST" \
    --model_backend hf \
    --hf_model_path "$hf_export" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --moonvit_path "$MOONVIT_PATH" \
    --model_tag "$tag" --max_new_tokens "$MAX_NEW_TOKENS" --num_frames 6 \
    --batch_size "$BATCH_SIZE"
echo "--- done $tag (video) $(date) ---"

# ── Judge, host eurovlm env (no container) ──────────────────────────────
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export FLASHINFER_WORKSPACE_BASE="/e/scratch/e-ext-2025e01-100/viveiros1/flashinfer_cache_v200_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

PY=/e/project1/e-ext-2025e01-100/viveiros1/envs/eurovlm/bin/python3
JUDGE_MODEL="Qwen/Qwen3.8-27B-FP8"

declare -A MANIFEST=(
  [multi_image]="evals/data/multi_image_eval_1000.jsonl"
  [video]="$VIDEO_MANIFEST"
)

MANIFEST_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/vjudge_manifests_v200_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MANIFEST_DIR"
for modality in multi_image video; do
  src="${MANIFEST[$modality]}"
  sed 's#/opt/Megatron-Bridge#'"$HOME"'/Megatron-Bridge#g' "$src" > "$MANIFEST_DIR/$(basename "$src")"
done

echo "--- starting vision judge (vLLM, host env, no container) ---"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85 --max-num-seqs 256 \
    > "logs/euro_vl_eval/vllm_video200_judge_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
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
for i in $(seq 1 90); do
  if curl -sS -m 3 http://localhost:8000/health >/dev/null 2>&1; then
    echo "judge healthy after $((i * 10))s"
    ready=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM died during startup"
    break
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  echo "judge never became healthy, aborting"
  exit 1
fi

for modality in multi_image video; do
  src="${MANIFEST[$modality]}"
  manifest="$MANIFEST_DIR/$(basename "$src")"
  captions="evals/results/${tag}/${modality}_captions.jsonl"
  [ -s "$manifest" ] && [ -s "$captions" ] || { echo "skip judge $tag/$modality: missing manifest or captions"; continue; }
  echo "--- judge $tag ($modality) $(date) ---"
  "$PY" -m evals.judge \
      --manifest "$manifest" \
      --captions "$captions" \
      --model_tag "$tag" --judge_provider local --judge_model "$JUDGE_MODEL" \
      --out_dir evals/results/judged_vision --concurrency 16
  echo "--- done judge $tag ($modality) $(date) ---"
done

echo "=== $(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID done ($tag) ==="
