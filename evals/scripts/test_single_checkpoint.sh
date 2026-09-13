#!/usr/bin/env bash
# ==============================================================================
# Minimal single-checkpoint validation run — NOT the full sweep. Confirms the
# pipeline (judge boot, checkpoint resolution, generation, judging) actually
# works end-to-end on one checkpoint with a handful of samples, before
# committing to the full 8-task/12h-each array (evals/scripts/run_eval_array.sh).
# Uses evals/data/*_test.jsonl (small slices of the already-built manifests —
# no new downloads) and a short time limit so a bug fails fast and cheap.
#
# Usage:
#   sbatch evals/scripts/test_single_checkpoint.sh
# ==============================================================================

#SBATCH --job-name=euro-vl-eval-test1ckpt
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/euro_vl_eval/test1ckpt_%j.out
#SBATCH --error=logs/euro_vl_eval/test1ckpt_%j.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

# This cluster's compute nodes have no outbound network access — the judge model is
# already cached locally, so force huggingface_hub/vllm to skip the metadata check
# they'd otherwise make even for cached files (that check is what crashes vLLM here
# without this: ConnectionError contacting huggingface.co).
export HF_HUB_OFFLINE=1

HOST_SCRATCH="/e/scratch/e-ext-2025e01-100/viveiros1"
TOKENIZER_PATH="/scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer"
MOONVIT_PATH="/scratch/hf_models/moonshotai-MoonViT-SO-400M"
TAG="pil_100_test"
CKPT_HOST="$HOST_SCRATCH/euro_vl_runs/qwen3_pa_pyav/iter_0004000"
CKPT_CONTAINER="/scratch${CKPT_HOST#"$HOST_SCRATCH"}"

echo "=== $(date) | job $SLURM_JOB_ID | node $SLURMD_NODENAME | $TAG (test, 1 checkpoint) ==="
echo "checkpoint: $CKPT_CONTAINER"

# No srun wrapper — see run_eval_array.sh's comment for why (srun step
# contention deadlocks the backgrounded judge against the generation step).
echo "--- starting local judge (vLLM) ---"
./apptainer.sh \
    uv run --no-sync vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --port 8000 --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.3 \
    > "logs/euro_vl_eval/vllm_test_${SLURM_JOB_ID}.log" 2>&1 &
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

for modality in image multi_image video; do
  manifest="evals/data/${modality}_eval_test.jsonl"
  [ -s "$manifest" ] || { echo "skip $modality: no test manifest"; continue; }
  echo "--- generate $TAG ($modality) $(date) ---"
  ./apptainer.sh \
      uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 -m evals.generate_captions \
      --manifest "$manifest" \
      --checkpoint "$CKPT_CONTAINER" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --moonvit_path "$MOONVIT_PATH" \
      --model_tag "$TAG" --max_new_tokens 640 --num_frames 6

  captions="evals/results/$TAG/${modality}_captions.jsonl"
  [ -s "$captions" ] || { echo "FAIL: no captions written for $modality"; continue; }
  echo "--- generated $(wc -l < "$captions") captions for $modality ---"

  echo "--- judge $TAG ($modality) $(date) ---"
  ./apptainer.sh uv run --no-sync python -m evals.judge \
      --manifest "$manifest" \
      --captions "$captions" \
      --model_tag "$TAG" --judge_provider local --judge_model Qwen/Qwen2.5-32B-Instruct-AWQ
done

echo "=== $(date) | job $SLURM_JOB_ID done ==="
find evals/results -name "*${TAG}*" -exec echo {} \; -exec cat {} \;
