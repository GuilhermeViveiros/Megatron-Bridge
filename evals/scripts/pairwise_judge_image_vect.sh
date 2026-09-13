#!/usr/bin/env bash
# ==============================================================================
# Pairwise-compare vect_10/50/100's image captions progressively (vect_10 vs
# vect_50, vect_50 vs vect_100), on the full 1000-item manifest. Per user preference:
# pairwise (winner A/B/tie) over absolute 1-5 rubric scoring for checkpoint
# comparisons -- see evals/judge.py's pairwise_judge_captions.
#
# Usage:
#   sbatch evals/scripts/pairwise_judge_image_vect.sh
# ==============================================================================

#SBATCH --job-name=euro-vl-pairwise-image-vect
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/euro_vl_eval/pairwise_image_vect_%j.out
#SBATCH --error=logs/euro_vl_eval/pairwise_image_vect_%j.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

export HF_HUB_OFFLINE=1

MANIFEST_SRC="evals/data/image_eval_1000.jsonl"

export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export FLASHINFER_WORKSPACE_BASE="/e/scratch/e-ext-2025e01-100/viveiros1/flashinfer_cache_pairwise_img"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

PY=/e/project1/e-ext-2025e01-100/viveiros1/envs/eurovlm/bin/python3
JUDGE_MODEL="Qwen/Qwen3.8-27B-FP8"

MANIFEST_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/vjudge_manifests_pairwise_img"
mkdir -p "$MANIFEST_DIR"
sed 's#/opt/Megatron-Bridge#'"$HOME"'/Megatron-Bridge#g' "$MANIFEST_SRC" > "$MANIFEST_DIR/image_eval_1000.jsonl"

echo "--- starting vision judge (vLLM, host env, no container) ---"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85 --max-num-seqs 256 \
    > "logs/euro_vl_eval/vllm_pairwise_image_vect_${SLURM_JOB_ID}.log" 2>&1 &
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

PAIRS=("vect_10 vect_50" "vect_50 vect_100")
for pair in "${PAIRS[@]}"; do
  read -r tag_a tag_b <<< "$pair"
  echo "--- pairwise $tag_a vs $tag_b (image) $(date) ---"
  "$PY" -m evals.judge \
      --manifest "$MANIFEST_DIR/image_eval_1000.jsonl" \
      --captions "evals/results/${tag_a}/image_captions.jsonl" \
      --captions_b "evals/results/${tag_b}/image_captions.jsonl" \
      --model_tag "$tag_a" --model_tag_b "$tag_b" \
      --judge_provider local --judge_model "$JUDGE_MODEL" \
      --out_dir evals/results/judged_pairwise --concurrency 16
  echo "--- done pairwise $tag_a vs $tag_b (image) $(date) ---"
done

echo "=== $(date) | job $SLURM_JOB_ID done ==="
