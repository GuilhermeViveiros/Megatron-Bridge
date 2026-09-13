#!/usr/bin/env bash
# ==============================================================================
# EuroVL captioning eval sweep for the vect checkpoints (10/50/100%-of-training),
# one Slurm array task per checkpoint, using the fast HF-exported/KV-cached/
# batched generation path (--model_backend hf, hf_cached_decode_batch) instead
# of the original uncached Megatron greedy_decode -- see evals/generate_captions.py's
# module docstring for why model.generate() is deliberately not used, and this
# session's sanity_check/ scripts for the verified ~5.4x (KV cache) x ~3x
# (batching) speedup and its expected (batch-size-dependent GPU numerics)
# per-item caption drift vs single-item generation.
#
# Manifests: 1000-sample image/multi_image, 500-sample video (video is hard-capped
# at 548 vimeo-sourced rows in the whole Molmo2-CapEval dataset -- see
# evals/data/{image,multi_image}_eval_1000.jsonl and video_eval_500.jsonl,
# built via evals/datasets/build_*.
#
# JUDGE: vision-capable Qwen/Qwen3.8-27B-FP8, served via the host `eurovlm` venv
# (NOT the apptainer container -- its bundled nvcc/CUDA-toolkit headers can't
# JIT-compile the Gated DeltaNet FlashInfer kernel this architecture needs; see
# evals/scripts/rejudge_vision_array.sh, whose pattern this mirrors).
#
# Usage:
#   sbatch evals/scripts/run_eval_array_hf.sh
# ==============================================================================

#SBATCH --job-name=euro-vl-eval-hf
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/euro_vl_eval/array_hf_%A_%a.out
#SBATCH --error=logs/euro_vl_eval/array_hf_%A_%a.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

export HF_HUB_OFFLINE=1

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-640}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer}"
MOONVIT_PATH="${MOONVIT_PATH:-/scratch/hf_models/moonshotai-MoonViT-SO-400M}"

# Task index -> checkpoint tag. HF exports already built (this session):
#   vect_10  -> /scratch/hf_models/vect_10_export
#   vect_50  -> /scratch/hf_models/vect_50_export
#   vect_100 -> /scratch/hf_models/vect_100_export
TAGS=(vect_10 vect_50 vect_100)
tag="${TAGS[$SLURM_ARRAY_TASK_ID]}"
hf_export="/scratch/hf_models/${tag}_export"

echo "=== $(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURMD_NODENAME | $tag ==="

declare -A MANIFEST=(
  [image]="evals/data/image_eval_1000.jsonl"
  [multi_image]="evals/data/multi_image_eval_1000.jsonl"
  [video]="evals/data/video_eval_500.jsonl"
)

for modality in image multi_image video; do
  manifest="${MANIFEST[$modality]}"
  [ -s "$manifest" ] || { echo "skip $tag/$modality: no manifest ($manifest)"; continue; }
  echo "--- generate $tag ($modality) $(date) ---"
  ./apptainer.sh \
      uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 -m evals.generate_captions \
      --manifest "$manifest" \
      --model_backend hf \
      --hf_model_path "$hf_export" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --moonvit_path "$MOONVIT_PATH" \
      --model_tag "$tag" --max_new_tokens "$MAX_NEW_TOKENS" --num_frames 6 \
      --batch_size "$BATCH_SIZE"
  echo "--- done $tag ($modality) $(date) ---"
done

# ── Judge, host eurovlm env (no container) ──────────────────────────────
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export FLASHINFER_WORKSPACE_BASE="/e/scratch/e-ext-2025e01-100/viveiros1/flashinfer_cache_hf${SLURM_ARRAY_TASK_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

PY=/e/project1/e-ext-2025e01-100/viveiros1/envs/eurovlm/bin/python3
JUDGE_MODEL="Qwen/Qwen3.8-27B-FP8"

# Manifests were built inside the apptainer container (container-internal media
# paths, /opt/Megatron-Bridge/...); the judge runs on the bare host, so rewrite
# them to real host paths first, into a per-task scratch dir.
MANIFEST_DIR="/e/scratch/e-ext-2025e01-100/viveiros1/vjudge_manifests_hf${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MANIFEST_DIR"
for modality in image multi_image video; do
  src="${MANIFEST[$modality]}"
  [ -s "$src" ] || continue
  sed 's#/opt/Megatron-Bridge#'"$HOME"'/Megatron-Bridge#g' "$src" > "$MANIFEST_DIR/$(basename "$src")"
done

echo "--- starting vision judge (vLLM, host env, no container) ---"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85 --max-num-seqs 256 \
    > "logs/euro_vl_eval/vllm_hf_array_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
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

for modality in image multi_image video; do
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
done

echo "=== $(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID done ($tag) ==="
