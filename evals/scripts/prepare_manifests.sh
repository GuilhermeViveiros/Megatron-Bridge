#!/usr/bin/env bash
# ==============================================================================
# Build the (shared, read-only) eval manifests once, ahead of the per-checkpoint
# array job (evals/scripts/run_eval_array.sh) — avoids 8 parallel tasks racing
# to write the same manifest files.
#
# IMPORTANT: run this directly on the login node, NOT via sbatch. This
# cluster's compute nodes have no outbound network access, and every step
# here needs it (HF Hub downloads). Submitting it as a batch job fails
# silently: `set -uo pipefail` (no `-e`) means a failed download doesn't stop
# the script, so it exits 0 and the dependent array job starts anyway —
# happened once already, see evals/results git history / session notes.
#
# Usage:
#   bash evals/scripts/prepare_manifests.sh
# ==============================================================================

set -uo pipefail
cd "$HOME/Megatron-Bridge"

IMAGE_SAMPLES="${IMAGE_SAMPLES:-250}"
MULTI_IMAGE_SAMPLES="${MULTI_IMAGE_SAMPLES:-400}"   # oversample: some hotlinked images are dead
VIDEO_SAMPLES="${VIDEO_SAMPLES:-40}"                # oversample slightly: some rows are ego4d/bdd100k (not bundled)

echo "=== $(date) ==="

echo "--- image manifest ($IMAGE_SAMPLES samples, ShareGPT-4o) ---"
./apptainer.sh uv run --no-sync python -m evals.datasets.build_single_image \
    --num_samples "$IMAGE_SAMPLES" --seed 42 --out evals/data/image_eval.jsonl

echo "--- multi-image manifest ($MULTI_IMAGE_SAMPLES samples, Molmo2-MultiImageQA) ---"
./apptainer.sh uv run --no-sync python -m evals.datasets.build_multi_image \
    --num_samples "$MULTI_IMAGE_SAMPLES" --seed 42 --out evals/data/multi_image_eval.jsonl

echo "--- video manifest ($VIDEO_SAMPLES samples, Molmo2-CapEval, vimeo subset) ---"
./apptainer.sh uv run --no-sync python -m evals.datasets.build_video \
    --num_samples "$VIDEO_SAMPLES" --seed 42 --out evals/data/video_eval.jsonl

echo "=== $(date) done ==="
wc -l evals/data/image_eval.jsonl evals/data/multi_image_eval.jsonl evals/data/video_eval.jsonl
