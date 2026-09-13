#!/usr/bin/env bash
# ==============================================================================
# Generate EuroVL captions for the eurollm_pa run's iter_0001000 and
# iter_0002000 checkpoints on the full 1000-sample image manifest
# (evals/data/image_eval_1000.jsonl), so they can be pairwise-judged against
# the existing full-manifest baselines (qwen3_pa_broad, towervision_stage1, …
# — see evals/results/*/image_captions.jsonl, all generated on this same
# 1000-sample manifest).
#
# eurollm_pa is EuroLLM-1.7B-Instruct-2512 + MoonViT-SO-400M (M-RoPE) — the
# "eurollm" model_variant in evals/generate_captions.py, which is
# --model_backend megatron ONLY (its HF export path does not apply M-RoPE for
# this variant, see generate_captions.py's _VARIANTS comment).
# matches this run's assembled HF dir (preprocessor_config.json declares
# MoonViTImageProcessor, not the vectorized backend).
#
# This cluster does not allow partial-node GPU allocation (a 1-GPU request
# still bills/holds the whole 4-GPU node), so this script uses all 4 GPUs:
# the manifest is split in half (500 items each), and both checkpoints run
# their two halves concurrently (2 checkpoints x 2 shards = 4 processes, one
# per GPU), then shards are merged back into each checkpoint's single
# image_captions.jsonl.
#
# Overwrites evals/results/eurollm-pa-iter1000/image_captions.jsonl (previously
# only the 100-sample subset -- this run's ids are a strict superset) and
# creates evals/results/eurollm-pa-iter2000/image_captions.jsonl.
#
# Usage:
#   sbatch evals/scripts/run_eurollm_pa_generate.sh
# ==============================================================================

#SBATCH --job-name=eurollm-pa-eval-gen
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/euro_vl_eval/eurollm_pa_gen_%j.out
#SBATCH --error=logs/euro_vl_eval/eurollm_pa_gen_%j.err

set -euo pipefail
cd "$HOME/Megatron-Bridge"

# Compute nodes have no outbound network -- everything below (checkpoints, tokenizer,
# MoonViT config) is already local, so force offline mode rather than fail on the HF Hub
# metadata check huggingface_hub otherwise makes even for cached files.
export HF_HUB_OFFLINE=1

TOKENIZER_PATH="/scratch/hf_models/euro_vl_2b_2512_hf"
MOONVIT_PATH="/scratch/hf_models/moonshotai-MoonViT-SO-400M"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-640}"

HOST_SCRATCH="/e/scratch/e-ext-2025e01-100/viveiros1"
TMP_DIR="$HOST_SCRATCH/eurovl_eval_tmp"
mkdir -p "$TMP_DIR"

# Split the manifest into 2 halves (500 items each) on the host -- pure filesystem/JSON
# work, no container needed.
FULL_MANIFEST="evals/data/image_eval_1000.jsonl"
SHARD0="$TMP_DIR/image_eval_1000_shard0.jsonl"
SHARD1="$TMP_DIR/image_eval_1000_shard1.jsonl"
TOTAL_LINES=$(wc -l < "$FULL_MANIFEST")
HALF=$(((TOTAL_LINES + 1) / 2))
split -l "$HALF" -d --additional-suffix=.jsonl "$FULL_MANIFEST" "$TMP_DIR/image_eval_1000_shard"
mv "$TMP_DIR/image_eval_1000_shard00.jsonl" "$SHARD0"
mv "$TMP_DIR/image_eval_1000_shard01.jsonl" "$SHARD1"
echo "split $FULL_MANIFEST ($TOTAL_LINES lines) -> shard0/shard1 ($HALF each)"

declare -A CKPT=(
  [eurollm-pa-iter1000]="/scratch/euro_vl_runs/eurollm_pa/iter_0001000"
  [eurollm-pa-iter2000]="/scratch/euro_vl_runs/eurollm_pa/iter_0002000"
)

echo "=== $(date) | job $SLURM_JOB_ID | node $SLURMD_NODENAME ==="

# 4 concurrent single-GPU processes: (tag, shard) -> GPU. Each is a fully independent
# single-process job (world_size=1, no cross-process communication), isolated to its own
# GPU via CUDA_VISIBLE_DEVICES and its own rendezvous port to avoid collisions.
run_shard() {
  local tag="$1" shard_path="$2" gpu="$3" port="$4" out_tag="$5"
  CUDA_VISIBLE_DEVICES="$gpu" ./apptainer.sh \
      uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 --master_port="$port" \
      -m evals.generate_captions \
      --manifest "$shard_path" \
      --model_backend megatron \
      --model_variant eurollm \
      --checkpoint "${CKPT[$tag]}" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --moonvit_path "$MOONVIT_PATH" \
      --model_tag "$out_tag" --max_new_tokens "$MAX_NEW_TOKENS" \
      > "logs/euro_vl_eval/gen_${out_tag}_${SLURM_JOB_ID}.log" 2>&1
}

echo "--- launching 4 concurrent shards across 4 GPUs $(date) ---"
run_shard eurollm-pa-iter1000 "$SHARD0" 0 29500 eurollm-pa-iter1000_shard0 &
PID0=$!
run_shard eurollm-pa-iter1000 "$SHARD1" 1 29501 eurollm-pa-iter1000_shard1 &
PID1=$!
run_shard eurollm-pa-iter2000 "$SHARD0" 2 29502 eurollm-pa-iter2000_shard0 &
PID2=$!
run_shard eurollm-pa-iter2000 "$SHARD1" 3 29503 eurollm-pa-iter2000_shard1 &
PID3=$!

fail=0
for pid in $PID0 $PID1 $PID2 $PID3; do
  wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "FATAL: at least one shard failed, see logs/euro_vl_eval/gen_*_${SLURM_JOB_ID}.log"
  exit 1
fi
echo "--- all 4 shards done $(date) ---"

# Merge each checkpoint's two shard outputs back into its single image_captions.jsonl.
for tag in eurollm-pa-iter1000 eurollm-pa-iter2000; do
  out_dir="evals/results/$tag"
  mkdir -p "$out_dir"
  cat "evals/results/${tag}_shard0/image_captions.jsonl" "evals/results/${tag}_shard1/image_captions.jsonl" \
      > "$out_dir/image_captions.jsonl"
  rm -rf "evals/results/${tag}_shard0" "evals/results/${tag}_shard1"
  echo "merged -> $out_dir/image_captions.jsonl ($(wc -l < "$out_dir/image_captions.jsonl") lines)"
done

rm -rf "$TMP_DIR"
echo "=== $(date) | job $SLURM_JOB_ID done ==="
