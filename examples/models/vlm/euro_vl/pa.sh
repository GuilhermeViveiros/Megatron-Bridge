#!/usr/bin/env bash

# Stage PA: projector alignment — vision encoder and LLM frozen, projector trained.
#
# Run after assembling the EuroVL checkpoint with eurollm_bridge.py.
# Before training, set WANDB_API_KEY or disable wandb:
# export WANDB_API_KEY=<your_wandb_api_key>
# export WANDB_MODE=disabled

WORKSPACE=${WORKSPACE:-/workspace}
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/euro_vl_2b

echo "Running PA stage: projector alignment (TP=1, PP=1)"
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe euro_vl_2b_sft_config \
    --step_func vlm_step \
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    checkpoint.save=${WORKSPACE}/results/euro_vl_2b_pa \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.seq_length=4096 \
    model.freeze_language_model=True \
    model.freeze_vision_model=True \
    model.freeze_vision_projection=False \
    train.train_iters=10000 \
    train.global_batch_size=32 \
    train.micro_batch_size=2 \
    validation.eval_interval=500 \
    validation.eval_iters=32 \
    optimizer.lr=1e-3 \
    optimizer.min_lr=1e-4 \
    scheduler.lr_warmup_iters=500 \
    dataset.seq_length=4096 \
    logger.log_interval=10
    #dataset.maker_name=make_euro_vl_dataset \
    #dataset.hf_processor_path=/path/to/euro_vl_processor
