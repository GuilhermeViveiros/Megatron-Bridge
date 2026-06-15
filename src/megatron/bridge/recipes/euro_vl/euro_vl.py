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

This module provides SFT and PEFT configurations for Euro-VL models (2B, 9B, 22B).
"""

import os

import torch

from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider
from megatron.bridge.models.euro_vl.euro_vl_processor import EuroVLProcessor
from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider
from megatron.bridge.models.euro_vl.moonvit import MoonViTConfig
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


_SCRATCH = os.environ["SCRATCH"]
# Assembled EuroVL HF checkpoint (from eurollm_bridge.py): holds the tokenizer
# (vision tokens + chat template) and the MoonViT image-processor config. Set
# cfg.checkpoint.pretrained_checkpoint to this to load weights via the bridge.
EUROVL_HF = f"{_SCRATCH}/hf_models/euro_vl_2b_hf"
# Root of the prepared Energon datasets (energon-data/). Static default; override per
# launch with dataset.root=... or the EUROVL_ENERGON_ROOT env var.
EUROVL_ENERGON_ROOT = os.environ.get("EUROVL_ENERGON_ROOT", "/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data")


def _make_euro_vl_2b_provider() -> EuroVLModelProvider:
    """Build EuroVLModelProvider with EuroLLM-1.7B + MoonViT-SO-400M architecture."""
    return EuroVLModelProvider(
        # EuroLLM-1.7B architecture
        num_layers=24,
        hidden_size=2048,
        ffn_hidden_size=5632,
        num_attention_heads=16,
        num_query_groups=8,
        # vocab_size extended: 128000 (EuroLLM) + 5 vision special tokens (128000-128004)
        vocab_size=128005,
        make_vocab_size_divisible_by=128,
        # Pad the (odd) 128005 vocab up to a TP-divisible size; required for TP>1
        # (VocabParallelEmbedding splits vocab across TP ranks). Harmless at TP=1.
        should_pad_vocab=True,
        seq_length=4096,
        # Llama / EuroLLM settings
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        position_embedding_type="rope",
        rotary_base=1000000,  # EuroLLM-1.7B rope_theta (NOT 10000); must match the base model
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
        # Vision — vendored MoonViT-SO-400M config (in-repo, serializes cleanly).
        vision_config=MoonViTConfig(),
        projector_input_dim=4608,  # 4 * 1152 (MoonViT hidden * merge kernel size)
        projector_output_dim=2048,
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

    # Model configuration
    cfg.model = _make_euro_vl_2b_provider()

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # cfg.train.train_iters = 50000
    # cfg.train.global_batch_size = 32
    # cfg.train.micro_batch_size = 1

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Training config
    cfg.train.train_iters = 500
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Validation config
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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Mock dataset using EuroVLProcessor for training tests. The processor loads the
    # tokenizer (vision tokens + chat template) and MoonViT image-processor config
    # from the single assembled HF checkpoint dir. Replace with
    # HFDatasetConversationProvider + real data for actual training.
    processor = EuroVLProcessor.from_pretrained(EUROVL_HF)
    cfg.dataset = MockVLMConversationProvider(
        seq_length=4096,
        hf_processor_path=EUROVL_HF,  # used only as a key; processor is pre-built
        image_size=(385, 356),
        num_images=1,
        pack_sequences_in_batch=False,
    )
    cfg.dataset._processor = processor  # inject pre-built EuroVLProcessor directly

    # To load the assembled EuroVL weights (instead of random init), point the bridge
    # at the HF checkpoint; it converts HF -> Megatron at setup time:
    # cfg.checkpoint.pretrained_checkpoint = EUROVL_HF

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg


# =============================================================================
# EuroVL 2B SFT — Megatron-Energon data path
# =============================================================================
def euro_vl_2b_sft_energon_config() -> ConfigContainer:
    """SFT config for EuroVL-2B over a Megatron-Energon (Crude) weighted blend.

    Reuses :func:`euro_vl_2b_sft_config` (model / parallelism / optimizer), then swaps
    the mock dataset for a ``EuroVLEnergonProvider`` driven by ``EuroVLTaskEncoder``
    (crude ``jpg`` + ``json`` ChatML -> ``EuroVLProcessor``).

    The data mix is built at runtime from a directory of prepared datasets (``root``) and
    ``mixture.yaml`` in that directory — InternVL-style **repeat factors** ``r`` per dataset
    (epochs per dataset; effective samples = ``r * size``; ``r=0`` excludes; ``r in [0,4]``).
    Edit ``energon-data/mixture.yaml`` to control the blend; a missing file = all ``r=1``::

        image/captioning:
          cc12m: 0.3
          sharegpt4o: 2.0

    The provider logs each dataset's r / size / step-share.

    Sequence packing: ``packing_buffer_size`` enables energon fill-to-``seq_length`` packing
    (InternVL / Nemotron-Nano-V2 style) — each training sequence packs ~``seq_length`` /
    avg-sample-length samples via THD/varlen attention, decoupled from ``micro_batch_size``.
    This requires ``train.micro_batch_size=1`` (THD/CP) and is mutually exclusive with
    ``pack_sequences_in_batch``; the provider asserts both. Launch example::

        train.micro_batch_size=1 train.global_batch_size=128 train.train_iters=10000

    ``train_iters`` is **not** auto-derived under packing — set it explicitly at launch
    (a sample-count → step estimate is ill-defined once packing collapses many samples
    per step). A packing-aware estimate is a tracked follow-up.

    Energon imports are local so the other EuroVL recipes do not require
    ``megatron-energon`` to be installed.
    """
    from megatron.bridge.data.energon.euro_vl_energon_provider import EuroVLEnergonProvider
    from megatron.bridge.data.energon.euro_vl_task_encoder import EuroVLTaskEncoder

    cfg = euro_vl_2b_sft_config()

    # Long context for the energon stage. Set before building the task encoder / provider
    # below so seq_length propagates to both (they read cfg.model.seq_length at build).
    cfg.model.seq_length = 8192
    # 12288
    # 8192
    # 16384

    # Square-root per-token loss (InternVL3.5 eq. 2): the task encoder bakes a 1/sqrt(N)
    # weight into each sample's loss_mask, and calculate_per_token_loss=True makes Megatron
    # normalize by the global sum of weights. Both must be set together.
    cfg.model.calculate_per_token_loss = True

    processor = EuroVLProcessor.from_pretrained(EUROVL_HF)
    task_encoder = EuroVLTaskEncoder(
        processor=processor,
        seq_length=cfg.model.seq_length,
        sqrt_loss_weighting=True,
    )
    cfg.dataset = EuroVLEnergonProvider(
        root=EUROVL_ENERGON_ROOT,
        mixture_file=os.path.join(EUROVL_ENERGON_ROOT, "mixture.yaml"),
        seq_length=cfg.model.seq_length,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        num_workers=8,
        task_encoder=task_encoder,
        # Energon fill-to-seq_length packing (requires train.micro_batch_size=1).
        # Mutually exclusive with pack_sequences_in_batch.
        packing_buffer_size=256,
        pack_sequences_in_batch=False,
    )
    return cfg
