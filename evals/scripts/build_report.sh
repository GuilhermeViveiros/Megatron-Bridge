#!/usr/bin/env bash
# ==============================================================================
# Aggregate all judged scores from the array sweep (evals/scripts/run_eval_array.sh)
# into evals/results/report.md. No GPU/network work — pure aggregation.
#
# Usage:
#   sbatch --dependency=afterok:<array_job_id> evals/scripts/build_report.sh
# ==============================================================================

#SBATCH --job-name=euro-vl-eval-report
#SBATCH --account=e-ext-2025e01-100
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/euro_vl_eval/report_%j.out
#SBATCH --error=logs/euro_vl_eval/report_%j.err

set -uo pipefail
cd "$HOME/Megatron-Bridge"

echo "=== $(date) | job $SLURM_JOB_ID | report ==="
./apptainer.sh uv run --no-sync python -m evals.report
cat evals/results/report.md
echo "=== $(date) | job $SLURM_JOB_ID done ==="
