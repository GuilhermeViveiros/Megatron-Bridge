#!/usr/bin/env bash
# ==============================================================================
# Generate EuroVL captions for one eurollm_pa checkpoint on the 100-sample
# image manifest (evals/data/image_eval_100.jsonl) -- the same manifest used
# for every other eurollm-pa-iter* tag, so results stay directly comparable.
#
# Scaled down from the full 1000-sample run: the Megatron backend's greedy
# decode has no KV cache, so at max_new_tokens=640 each item takes ~1 minute
# -- fine for 100 items, far too slow for 1000 (would need ~7-8h per 500-item
# shard, observed live). Judging is still valid against the full-manifest
# baselines (qwen3_pa_broad, towervision_stage1): their caption files are a
# superset that includes this same 100-id subset.
#
# This cluster does not allow partial-node GPU allocation (a 1-GPU request
# still bills/holds the whole 4-GPU node), so the 100 items are split into 4
# shards of 25 and run concurrently, one per GPU.
#
# Usage:
#   CKPT_ITER=iter_0003720 TAG=eurollm-pa-iter3720 sbatch evals/scripts/run_eurollm_pa_generate_100.sh
# ==============================================================================

#SBATCH --job-name=eurollm-pa-eval-gen-100
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/euro_vl_eval/eurollm_pa_gen100_%j.out
#SBATCH --error=logs/euro_vl_eval/eurollm_pa_gen100_%j.err

set -euo pipefail
cd "$HOME/Megatron-Bridge"

export HF_HUB_OFFLINE=1

TOKENIZER_PATH="/scratch/hf_models/euro_vl_2b_2512_hf"
MOONVIT_PATH="/scratch/hf_models/moonshotai-MoonViT-SO-400M"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-640}"
CKPT_ITER="${CKPT_ITER:-iter_0002000}"
TAG="${TAG:-eurollm-pa-iter2000}"
CHECKPOINT="/scratch/euro_vl_runs/eurollm_pa/$CKPT_ITER"

HOST_SCRATCH="/e/scratch/e-ext-2025e01-100/viveiros1"
TMP_DIR="$HOST_SCRATCH/eurovl_eval_tmp_100_${TAG}"
mkdir -p "$TMP_DIR"

FULL_MANIFEST="evals/data/image_eval_100.jsonl"
split -n l/4 -d --additional-suffix=.jsonl "$FULL_MANIFEST" "$TMP_DIR/shard"
echo "split $FULL_MANIFEST into 4 shards under $TMP_DIR"

echo "=== $(date) | job $SLURM_JOB_ID | node $SLURMD_NODENAME | tag=$TAG ckpt=$CHECKPOINT ==="

run_shard() {
  local shard_path="$1" gpu="$2" port="$3" out_tag="$4"
  CUDA_VISIBLE_DEVICES="$gpu" ./apptainer.sh \
      uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 --master_port="$port" \
      -m evals.generate_captions \
      --manifest "$shard_path" \
      --model_backend megatron \
      --model_variant eurollm \
      --checkpoint "$CHECKPOINT" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --moonvit_path "$MOONVIT_PATH" \
      --model_tag "$out_tag" --max_new_tokens "$MAX_NEW_TOKENS" \
      > "logs/euro_vl_eval/gen_${out_tag}_${SLURM_JOB_ID}.log" 2>&1
}

echo "--- launching 4 concurrent shards across 4 GPUs $(date) ---"
run_shard "$TMP_DIR/shard00.jsonl" 0 29500 "${TAG}_shard0" &
PID0=$!
run_shard "$TMP_DIR/shard01.jsonl" 1 29501 "${TAG}_shard1" &
PID1=$!
run_shard "$TMP_DIR/shard02.jsonl" 2 29502 "${TAG}_shard2" &
PID2=$!
run_shard "$TMP_DIR/shard03.jsonl" 3 29503 "${TAG}_shard3" &
PID3=$!

fail=0
for pid in $PID0 $PID1 $PID2 $PID3; do
  wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "FATAL: at least one shard failed, see logs/euro_vl_eval/gen_${TAG}_shard*_${SLURM_JOB_ID}.log"
  exit 1
fi
echo "--- all 4 shards done $(date) ---"

# Defensive: this cluster's parallel filesystem has shown a brief metadata-visibility lag
# right after a writer process exits (hit live -- 3 of 4 shard output files were briefly
# not statable immediately after `wait` returned, and got silently merged as empty before
# their data could ever be read). Poll each expected file until it's present and stops
# growing, and never delete a shard dir before every file has been verified.
SHARD_FILES=(
  "evals/results/${TAG}_shard0/image_captions.jsonl"
  "evals/results/${TAG}_shard1/image_captions.jsonl"
  "evals/results/${TAG}_shard2/image_captions.jsonl"
  "evals/results/${TAG}_shard3/image_captions.jsonl"
)
for f in "${SHARD_FILES[@]}"; do
  ok=0
  for i in $(seq 1 30); do
    if [ -s "$f" ]; then
      ok=1
      break
    fi
    echo "waiting for $f to become visible (attempt $i)"
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "FATAL: $f never became visible/non-empty"
    exit 1
  fi
done

out_dir="evals/results/${TAG}"
mkdir -p "$out_dir"
cat "${SHARD_FILES[@]}" > "$out_dir/image_captions.jsonl"
merged_lines=$(wc -l < "$out_dir/image_captions.jsonl")
expected_lines=$(cat "${SHARD_FILES[@]}" | wc -l)
if [ "$merged_lines" -ne "$expected_lines" ] || [ "$merged_lines" -lt 90 ]; then
  echo "FATAL: merged file has $merged_lines lines, expected ~100 -- NOT deleting shard dirs"
  exit 1
fi
rm -rf "evals/results/${TAG}_shard0" "evals/results/${TAG}_shard1" \
       "evals/results/${TAG}_shard2" "evals/results/${TAG}_shard3"
echo "merged -> $out_dir/image_captions.jsonl ($merged_lines lines)"

rm -rf "$TMP_DIR"
echo "=== $(date) | job $SLURM_JOB_ID done ==="
