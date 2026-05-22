# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EuroVL training recipe.

Single SFT config with all modules trainable by default.  For projector-alignment
(PA stage), override freeze flags at launch time::

    model.freeze_language_model=True model.freeze_vision_model=True

The recipe uses a placeholder dataset (make_cord_v2_dataset).  Before running,
override cfg.dataset.hf_processor_path and cfg.dataset.maker_name with the actual
EuroVL processor path and dataset maker.
"""

import torch
from transformers import AutoConfig

from megatron.bridge.models.euro_vl.euro_vl_provider import IMAGE_TOKEN_ID, EuroVLModelProvider
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


# Default paths — override via CLI or cfg.dataset.hf_processor_path at runtime.
_MOONVIT_PATH = "/e/scratch/jureap126/gviveiros/hf_models/moonshotai-MoonViT-SO-400M"


def _make_euro_vl_2b_provider() -> EuroVLModelProvider:
    """Build EuroVLModelProvider with EuroLLM-1.7B + MoonViT-SO-400M architecture."""
    moonvit_config = AutoConfig.from_pretrained(_MOONVIT_PATH, trust_remote_code=True)

    return EuroVLModelProvider(
        # EuroLLM-1.7B architecture
        num_layers=24,
        hidden_size=2048,
        ffn_hidden_size=5632,
        num_attention_heads=16,
        num_query_groups=8,
        # vocab_size extended: 128000 (EuroLLM) + 1 (<image> token)
        vocab_size=128001,
        make_vocab_size_divisible_by=128,
        seq_length=4096,
        max_position_embeddings=4096,
        # Llama / EuroLLM settings
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        rotary_base=10000.0,
        rotary_percent=1.0,
        gated_linear_unit=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        add_qkv_bias=False,
        share_embeddings_and_output_weights=False,
        bias_activation_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        apply_rope_fusion=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        # Vision
        vision_config=moonvit_config,
        projector_input_dim=4608,  # 4 * 1152 (MoonViT hidden * merge kernel area)
        projector_output_dim=2048,
        image_token_id=IMAGE_TOKEN_ID,
        use_bidirectional_image_attention=False,
    )


# =============================================================================
# EuroVL 2B SFT Configuration
# =============================================================================
def euro_vl_2b_sft_config() -> ConfigContainer:
    """SFT config for EuroVL-2B (EuroLLM-1.7B + MoonViT-SO-400M) — all modules trainable.

    Default: 1 node, 1 GPU (2B model fits on a single GPU at TP=1).

    For projector-alignment (PA) training, override freeze flags at launch::

        model.freeze_language_model=True model.freeze_vision_model=True
    """
    cfg = _sft_common_vlm()

    cfg.model = _make_euro_vl_2b_provider()
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    cfg.train.train_iters = 50000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=50000,
        max_lr=5e-5,
        min_lr=5e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.mixed_precision = "bf16_mixed"

    # TODO: replace with actual EuroVL dataset maker and processor path
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = _MOONVIT_PATH  # placeholder

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg
