#!/usr/bin/env bash
# ==============================================================================
# Judge just the (already-complete) image modality for vect_10/50/100, without
# waiting for the still-running multi_image/video generation in the main sweep
# (evals/scripts/run_eval_array_hf.sh, job 1479887) to finish. Same judge
# pattern as that script's judge stage -- writes to the same
# evals/results/judged_vision/image_vect_{10,50,100}_scores.jsonl destination
# the main sweep would eventually produce for image anyway, now scored against
# the full 1000-item image_eval_1000.jsonl manifest (supersedes the older
# 250-item pass). Includes the <think>-tag -> score 0 fix in evals/judge.py.
#
# Usage:
#   sbatch evals/scripts/judge_image_vect_only.sh
# ==============================================================================

#SBATCH --job-name=euro-vl-judge-image-vect
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/euro_vl_eval/judge_image_vect_%j.out
#SBATCH --error=logs/euro_vl_eval/judge_image_vect_%j.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

# Compute nodes have no outbound network access -- everything vLLM needs (the judge model)
# is already cached locally, so force offline mode or huggingface_hub hangs/dies trying to
# reach the Hub for metadata (see run_eval_array_hf.sh for the same pattern).
export HF_HUB_OFFLINE=1

TAGS=(vect_10 vect_50 vect_100)
MANIFEST_SRC="evals/data/image_eval_1000.jsonl"

export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export FLASHINFER_WORKSPACE_BASE="/e/scratch/e-ext-2025e01-100/viveiros1/flashinfer_cache_judgeimg"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

PY=/e/project1/e-ext-2025e01-100/viveiros1/envs/eurovlm/bin/python3
JUDGE_MODEL="Qwen/Qwen3.8-27B-FP8"

# Manifest was built inside the apptainer container (container-internal media paths,
# /opt/Megatron-Bridge/...); the judge runs on the bare host, so rewrite to real host paths.
MANIFEST_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/vjudge_manifests_imgonly"
mkdir -p "$MANIFEST_DIR"
sed 's#/opt/Megatron-Bridge#'"$HOME"'/Megatron-Bridge#g' "$MANIFEST_SRC" > "$MANIFEST_DIR/image_eval_1000.jsonl"

echo "--- starting vision judge (vLLM, host env, no container) ---"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85 --max-num-seqs 256 \
    > "logs/euro_vl_eval/vllm_judge_image_vect_${SLURM_JOB_ID}.log" 2>&1 &
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

for tag in "${TAGS[@]}"; do
  captions="evals/results/${tag}/image_captions.jsonl"
  [ -s "$captions" ] || { echo "skip judge $tag: no captions"; continue; }
  echo "--- judge $tag (image) $(date) ---"
  "$PY" -m evals.judge \
      --manifest "$MANIFEST_DIR/image_eval_1000.jsonl" \
      --captions "$captions" \
      --model_tag "$tag" --judge_provider local --judge_model "$JUDGE_MODEL" \
      --out_dir evals/results/judged_vision --concurrency 16
  echo "--- done judge $tag (image) $(date) ---"
done

echo "=== $(date) | job $SLURM_JOB_ID done ==="
