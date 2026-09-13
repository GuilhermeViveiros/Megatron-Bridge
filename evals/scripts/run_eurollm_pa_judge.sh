#!/usr/bin/env bash
# ==============================================================================
# Pairwise-judge eurollm_pa's iter_0001000 and iter_0002000 checkpoints against
# each other and against the existing full-manifest baselines (qwen3_pa_broad,
# towervision_stage1), using the same local vLLM judge (Qwen2.5-32B-Instruct-AWQ)
# as evals/scripts/run_eval_array.sh -- not Anthropic, so no ANTHROPIC_API_KEY
# or outbound network is needed (compute nodes have none).
#
# Runs over the 100-sample manifest (evals/data/image_eval_100.jsonl), not the
# full 1000 -- the Megatron eurollm decode path has no KV cache and a 1000-item
# generation run was observed live to need ~7-8h per 500-item shard, so
# generation was scaled down to 100 samples (see
# run_eurollm_pa_generate_100.sh). The full-manifest baselines' caption files
# are a superset that includes this same 100-id subset, so the comparison is
# still valid on the shared ids.
#
# Requires evals/scripts/run_eurollm_pa_generate_100.sh to have already
# produced evals/results/eurollm-pa-iter{1000,2000}/image_captions.jsonl
# (iter1000's was generated earlier, ad hoc, on this same 100-sample manifest).
#
# Pairwise, not absolute rubric scoring (per this repo's stated preference for
# checkpoint comparisons) -- winner "A"/"B"/"tie" + rationale per item.
#
# Usage:
#   sbatch evals/scripts/run_eurollm_pa_judge.sh
# ==============================================================================

#SBATCH --job-name=eurollm-pa-eval-judge
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/euro_vl_eval/eurollm_pa_judge_%j.out
#SBATCH --error=logs/euro_vl_eval/eurollm_pa_judge_%j.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

export HF_HUB_OFFLINE=1
MANIFEST="evals/data/image_eval_100.jsonl"
OUT_DIR="evals/results/judged_pairwise"
mkdir -p "$OUT_DIR"

echo "=== $(date) | job $SLURM_JOB_ID | node $SLURMD_NODENAME ==="

echo "--- starting local judge (vLLM) ---"
./apptainer.sh \
    uv run --no-sync vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --port 8000 --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.3 \
    > "logs/euro_vl_eval/vllm_judge_${SLURM_JOB_ID}.log" 2>&1 &
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

judge_pair() {
  local tag_a="$1" tag_b="$2"
  echo "--- judge $tag_a vs $tag_b $(date) ---"
  ./apptainer.sh uv run --no-sync python -m evals.judge \
      --manifest "$MANIFEST" \
      --captions "evals/results/$tag_a/image_captions.jsonl" \
      --model_tag "$tag_a" \
      --captions_b "evals/results/$tag_b/image_captions.jsonl" \
      --model_tag_b "$tag_b" \
      --judge_provider local --judge_model Qwen/Qwen2.5-32B-Instruct-AWQ --judge_text_only \
      --concurrency 16 --out_dir "$OUT_DIR"
}

judge_pair eurollm-pa-iter1000 eurollm-pa-iter2000
judge_pair eurollm-pa-iter1000 qwen3_pa_broad
judge_pair eurollm-pa-iter1000 towervision_stage1
judge_pair eurollm-pa-iter2000 qwen3_pa_broad
judge_pair eurollm-pa-iter2000 towervision_stage1

echo "=== $(date) | job $SLURM_JOB_ID done ==="
