#!/usr/bin/env bash
set -euo pipefail

SIF=/e/scratch/jureap126/gviveiros/singularity/megatron-bridge.sif
SCRATCH=/e/scratch/jureap126/gviveiros

# PYTHONPATH is pre-set to /opt/Megatron-Bridge/src so megatron.bridge is importable without any install step.
# Run eurollm_bridge.py directly with the container's python:
#   python eurollm_bridge.py \
#       --eurollm-path '/e/scratch/jureap126/gviveiros/hf_models/EuroLLM-1.7B-Instruct' \
#       --moonvit-path '/e/scratch/jureap126/gviveiros/hf_models/moonshotai-MoonViT-SO-400M' \
#       --output-path '/e/scratch/jureap126/gviveiros/eurovlm-data/models/megatron/euro_vl_2b'

apptainer shell \
    --nv \
    --env PYTHONPATH=/opt/Megatron-Bridge/src \
    --bind "$(pwd)":/opt/Megatron-Bridge \
    --bind "$HOME/.cache/huggingface":/root/.cache/huggingface \
    --pwd /opt/Megatron-Bridge \
    --writable-tmpfs \
    "$SIF"
