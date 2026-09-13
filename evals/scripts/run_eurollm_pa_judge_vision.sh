#!/usr/bin/env bash
# ==============================================================================
# Vision-grounded pairwise judge for eurollm_pa checkpoints, using the same
# vision-capable local judge (Qwen/Qwen3.8-27B-FP8) as
# evals/scripts/pairwise_judge_array.sh -- the container's bundled CUDA
# toolkit can't JIT-compile this model's Gated DeltaNet FlashInfer kernel, so
# this runs a dedicated host venv directly, no apptainer.
#
# Reuses captions already generated -- no checkpoint loading, no generation.
# Runs every pair in PAIRS (tag_a:tag_b) sequentially against one warm judge
# server.
#
# Usage:
#   PAIRS="eurollm-pa-iter3720:eurollm-pa-iter1000 eurollm-pa-iter3720:eurollm-pa-iter2000 eurollm-pa-iter3720:qwen3_pa_broad eurollm-pa-iter3720:towervision_stage1" \
#     sbatch evals/scripts/run_eurollm_pa_judge_vision.sh
# ==============================================================================

#SBATCH --job-name=eurollm-pa-vjudge
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/euro_vl_eval/eurollm_pa_vjudge_%j.out
#SBATCH --error=logs/euro_vl_eval/eurollm_pa_vjudge_%j.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export FLASHINFER_WORKSPACE_BASE=/e/scratch/e-ext-2025e01-100/viveiros1/flashinfer_cache_eurollm_vjudge
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

PY=/e/project1/e-ext-2025e01-100/viveiros1/envs/eurovlm/bin/python3
JUDGE_MODEL="Qwen/Qwen3.8-27B-FP8"

# tag_a:tag_b pairs to compare, e.g. "eurollm-pa-iter3720:eurollm-pa-iter1000 ...".
: "${PAIRS:?set PAIRS to a space-separated list of tag_a:tag_b pairs}"

echo "=== $(date) | job $SLURM_JOB_ID | node $SLURMD_NODENAME | pairs: $PAIRS ==="

# generate_captions.py's manifests carry container-internal media paths
# (/opt/Megatron-Bridge/...) -- this judge runs outside the container against
# the host filesystem directly, so rewrite them to the host repo path.
MANIFEST_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/vjudge_manifests_eurollm_pa"
mkdir -p "$MANIFEST_DIR"
sed 's#/opt/Megatron-Bridge#'"$HOME"'/Megatron-Bridge#g' \
    "evals/data/image_eval_100.jsonl" > "$MANIFEST_DIR/image_eval_100.jsonl"

echo "--- starting vision judge (vLLM, host env, no container) ---"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85 --max-num-seqs 256 \
    > "logs/euro_vl_eval/vllm_eurollm_pa_vjudge_${SLURM_JOB_ID}.log" 2>&1 &
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

for pair in $PAIRS; do
  tag_a="${pair%%:*}"
  tag_b="${pair##*:}"
  captions_a="evals/results/${tag_a}/image_captions.jsonl"
  captions_b="evals/results/${tag_b}/image_captions.jsonl"
  if [ ! -s "$captions_a" ] || [ ! -s "$captions_b" ]; then
    echo "skip $tag_a vs $tag_b: missing captions ($captions_a / $captions_b)"
    continue
  fi
  echo "--- pairwise vision judge $tag_a vs $tag_b $(date) ---"
  "$PY" -m evals.judge \
      --manifest "$MANIFEST_DIR/image_eval_100.jsonl" \
      --captions "$captions_a" --captions_b "$captions_b" \
      --model_tag "$tag_a" --model_tag_b "$tag_b" \
      --judge_provider local --judge_model "$JUDGE_MODEL" \
      --out_dir evals/results/judged_vision_pairwise
done

echo "=== $(date) | job $SLURM_JOB_ID done ==="
