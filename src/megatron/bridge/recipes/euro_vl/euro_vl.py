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
from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider, Qwen3EuroVLModelProvider
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
# Assembled Qwen3EuroVL oracle checkpoint (from qwen3_euro_vl_bridge.py): Qwen3-1.7B
# LLM + MoonViT + Qwen3 tokenizer (vision tokens built in). See current.md.
QWEN3_EUROVL_HF = f"{_SCRATCH}/hf_models/qwen3_euro_vl_2b_hf"
# Megatron-format conversion of the above (via convert_checkpoints.py import). The training
# loader only recognizes Megatron checkpoints for pretrained_checkpoint — pointing it at the
# raw HF dir silently loads nothing (checkpoint_exists() is False -> random init).
QWEN3_EUROVL_MCORE = f"{_SCRATCH}/megatron_models/qwen3_euro_vl_2b"
# Common scratch prefix for all EuroVL TRAINING outputs (checkpoints + tb logs), one per-recipe
# subdir. Kept OFF $HOME (the default nemo_experiments/ dir lands in the HOME-bind-mounted cwd
# and can blow HOME quota) and separate from megatron_models/ (which holds converted weights).
EUROVL_RUNS = f"{_SCRATCH}/euro_vl_runs"


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
        projector_input_dim=4608,  # 4 * 1152 (MoonViT hidden * merge kernel size (2,2))
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
    # Per-token loss requires the DP grad collective to SUM, not average: MCore sets
    # gradient_scaling_factor=1.0 and the single global division by the total weight-sum
    # (Sum_i sqrt(T_i)) happens once in finalize_model_grads — the Σwℓ/Σw of the sqrt
    # loss. MCore hard-asserts on average_in_collective=True + per-token loss at DDP init.
    cfg.ddp.average_in_collective = False

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


# =============================================================================
# Qwen3EuroVL oracle — Qwen3-1.7B backbone + MoonViT, interleaved M-RoPE
# =============================================================================
def _make_qwen3_euro_vl_provider() -> Qwen3EuroVLModelProvider:
    """Build Qwen3EuroVLModelProvider: Qwen3-1.7B backbone + MoonViT-SO-400M.

    The M-RoPE fields (``position_embedding_type="mrope"``, ``mrope_section=[24,20,20]``,
    ``apply_rope_fusion=False``) come from the :class:`Qwen3EuroVLModelProvider` defaults.
    """
    return Qwen3EuroVLModelProvider(
        # Qwen3-1.7B architecture
        num_layers=28,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=16,
        num_query_groups=8,
        kv_channels=128,  # Qwen3 head_dim (explicit in its HF config)
        # Vision tokens are built into Qwen3's vocab (151652-151656) — no extension.
        # 151936 is already divisible by 128, so no padding is required either.
        vocab_size=151936,
        make_vocab_size_divisible_by=128,
        should_pad_vocab=False,
        seq_length=4096,
        # Qwen3 settings (Llama-style dense GQA + QK-norm)
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        qk_layernorm=True,  # Qwen3 q_norm/k_norm per attention
        rotary_base=1000000,
        rotary_percent=1.0,
        gated_linear_unit=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        add_qkv_bias=False,
        share_embeddings_and_output_weights=True,  # Qwen3-1.7B ties embeddings
        bias_activation_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        # Vision — same vendored MoonViT-SO-400M as EuroVL.
        vision_config=MoonViTConfig(),
        projector_input_dim=4608,  # 4 * 1152 (MoonViT hidden * merge kernel area)
        projector_output_dim=2048,  # Qwen3-1.7B hidden size (same as EuroLLM's)
        use_bidirectional_image_attention=False,
        # Qwen3's built-in vision token ids (identical to Qwen3-VL's).
        image_token_id=151655,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        vision_pad_token_id=151654,
        video_token_id=151656,
    )


def qwen3_euro_vl_sft_energon_config() -> ConfigContainer:
    """Qwen3EuroVL **oracle** SFT config — Qwen3-1.7B + MoonViT over the energon caption blend.

    Validation scaffold (see ``current.md``): our vision path + recipe + data on Qwen3-VL's
    proven interleaved-M-RoPE stack, loading the assembled pretrained checkpoint
    (``qwen3_euro_vl_bridge.py`` -> Qwen3 LLM + MoonViT; projector random -> PA-trains).
    Mirrors :func:`euro_vl_2b_sft_energon_config` (packing, sqrt loss, native CE); differences:
    Qwen3 backbone/tokenizer (``Qwen3EuroVLProcessor``), a caption-only PA blend
    (``mixture_pa.yaml`` — captioning across image/multiimage/video), and
    ``checkpoint.pretrained_checkpoint`` set to the assembled HF dir.

    Projector-alignment (PA) launch — freeze the pretrained parts, train the projector::

        train.micro_batch_size=1 train.global_batch_size=128 train.train_iters=2000 \\
        model.freeze_language_model=true model.freeze_vision_model=true
    """
    from megatron.bridge.data.energon.euro_vl_energon_provider import EuroVLEnergonProvider
    from megatron.bridge.data.energon.euro_vl_task_encoder import EuroVLTaskEncoder
    from megatron.bridge.models.euro_vl.euro_vl_processor import Qwen3EuroVLProcessor

    cfg = _sft_common_vlm()

    # Model: Qwen3-1.7B backbone + MoonViT (mrope from provider defaults).
    cfg.model = _make_qwen3_euro_vl_provider()

    # Parallel settings (2B-scale: TP=1 is fastest; see throughput notes in current.md).
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # All modules trainable by default; freeze via CLI for PA (docstring above).
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # TE / kernels (mirror euro_vl_2b_sft_config).
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Training config.
    cfg.train.train_iters = 500
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Validation config.
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

    # DDP settings (mirror euro_vl_2b_sft_config).
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"

    # Long-context energon stage; set before building the task encoder / provider.
    cfg.model.seq_length = 8192

    # Square-root per-token loss (Qwen3VL)
    cfg.model.calculate_per_token_loss = True
    # Per-token loss requires the DP grad collective to SUM, not average: MCore sets
    # gradient_scaling_factor=1.0 and the single global division by the total weight-sum
    # (Sum_i sqrt(T_i)) happens once in finalize_model_grads — the Σwℓ/Σw of the sqrt
    # loss. MCore hard-asserts on average_in_collective=True + per-token loss at DDP init.
    cfg.ddp.average_in_collective = False

    # Frames sampled per video — set once here (the task encoder reads it back)
    processor = Qwen3EuroVLProcessor.from_pretrained(QWEN3_EUROVL_HF, num_frames=4)
    task_encoder = EuroVLTaskEncoder(
        processor=processor,
        seq_length=cfg.model.seq_length,
        sqrt_loss_weighting=True,
    )
    cfg.dataset = EuroVLEnergonProvider(
        root=EUROVL_ENERGON_ROOT,
        mixture_file=os.path.join(EUROVL_ENERGON_ROOT, "mixture_pa.yaml"),
        seq_length=cfg.model.seq_length,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        num_workers=8,
        task_encoder=task_encoder,
        packing_buffer_size=256,
        pack_sequences_in_batch=False,
        # Do NOT let energon eagerly decode media: with auto_decode=False every sample reaches
        # the task encoder as raw bytes, so _cook does bounded, native-resolution, on-demand
        # decode (images via PIL; video via _decode_video_bytes -> only num_frames frames)
        # instead of energon decoding whole clips into RAM (~2.6 GB/clip -> OOM/slow buffer fill).
        energon_dataset_kwargs={"auto_decode": False},
    )

    # Load the pretrained weights (Qwen3 LLM + MoonViT; projector random) from the
    # Megatron-converted checkpoint at setup time. This is the whole point of the oracle —
    # training starts from real weights, not random init. MUST be the Megatron dir, NOT the
    # HF dir (the loader silently skips a raw HF checkpoint -> random init -> loss ~ln(vocab)).
    cfg.checkpoint.pretrained_checkpoint = QWEN3_EUROVL_MCORE

    # Write training checkpoints/logs to scratch (per-recipe subdir), NOT the $HOME-bound cwd.
    cfg.checkpoint.save = f"{EUROVL_RUNS}/qwen3_sft"
    cfg.checkpoint.load = cfg.checkpoint.save  # resume from same dir (empty 1st run -> uses pretrained)
    cfg.logger.tensorboard_dir = None  # W&B only (TB is enabled by a non-None dir; disable it)
    # W&B (project only — the API key comes from the WANDB_API_KEY env var, never committed).
    cfg.logger.wandb_project = "EuroVL"
    cfg.logger.wandb_exp_name = "qwen3-sft"
    cfg.logger.wandb_save_dir = f"{EUROVL_RUNS}/qwen3_sft/wandb"

    return cfg


def qwen3_euro_vl_pa_sft_config() -> ConfigContainer:
    """Qwen3EuroVL **projector-alignment (PA / stage-1)** config.

    Freezes the pretrained LLM + vision tower and trains ONLY the (random-init) multimodal
    projector so it learns to map MoonViT features into the Qwen3 embedding space — the standard
    VLM stage-1 alignment. This mirrors Qwen3-VL's own Stage 0 (train only the MLP merger, vision
    encoder + LLM frozen; arxiv 2511.21631) and the in-repo ``qwen_vl/qwen3_vl.py`` default
    (freeze LM+vision, train projection). Built on :func:`qwen3_euro_vl_sft_energon_config` (same
    model, energon ``mixture_pa.yaml`` blend, packing, sqrt loss, bounded video decode); the only
    differences are the freezing and the projector-appropriate LR/batch.

    Hyperparameters follow **InternVL** alignment (``max_lr=2e-4``, ``global_batch_size=512``),
    which sits in the validated band (Nemotron 2e-4 → in-repo Qwen3-VL 3e-4 → LLaVA 1e-3). Short
    ~3% warmup, cosine decay over the run. Set ``train_iters`` to ~1 epoch of the blend (read the
    blue ``Suggested train_iters`` line the provider logs at startup) and sweep ``max_lr`` if
    tuning.

    Launch (freezing + LR/batch baked in)::

        uv run --no-sync python -m torch.distributed.run --nproc_per_node=4 \\
            scripts/training/run_recipe.py --recipe qwen3_euro_vl_pa_sft_config --step_func vlm_step \\
            train.train_iters=2000 dataset.num_workers=8
    """
    cfg = qwen3_euro_vl_sft_energon_config()

    # PA: freeze the pretrained LLM + vision tower; train only the projector.
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False

    # Full activation recompute (gradient checkpointing). PA keeps the WHOLE frozen-LLM forward
    # activation stack (needed to backprop to the upstream projector) — ~25-30 GB at seq 8192 and
    # the real OOM cause (not global_batch_size, which only sets grad-accum steps at mbs=1).
    # Freeing it also gives the vision-tower spike headroom on dense mixed-modality packs. The
    # frozen backbone makes the recompute compute cost negligible, so recompute every layer.
    # cfg.model.recompute_granularity = "full"
    # cfg.model.recompute_method = "uniform"
    # cfg.model.recompute_num_layers = 1

    # InternVL alignment batch. The energon provider captured global_batch_size at construction
    # (128 from the base config), so update BOTH the train config and the provider field — the
    # provider reads it at build time (setup), after this override.
    cfg.train.global_batch_size = 512
    cfg.dataset.global_batch_size = 512

    # Projector-only schedule: InternVL LR + short warmup + cosine decay over the run.
    cfg.train.train_iters = 2000  # ~1 epoch of the blend; size from the blue train_iters log
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=max(50, cfg.train.train_iters // 30),  # ~3% warmup
        lr_decay_iters=cfg.train.train_iters,  # cosine completes over the actual run
        max_lr=2e-4,
        min_lr=2e-5,
    )
    # Preserve the fp32 optimizer-state settings the sqrt per-token loss path relies on.
    opt_cfg.use_precision_aware_optimizer = False
    opt_cfg.main_grads_dtype = torch.float32
    opt_cfg.main_params_dtype = torch.float32
    opt_cfg.exp_avg_dtype = torch.float32
    opt_cfg.exp_avg_sq_dtype = torch.float32
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Write checkpoints/logs to scratch (PA subdir), NOT the $HOME-bound cwd. PA converges fast,
    # so save more often than the SFT default (every 100 iters).
    cfg.checkpoint.save = f"{EUROVL_RUNS}/qwen3_pa"
    cfg.checkpoint.load = cfg.checkpoint.save  # resume from same dir (empty 1st run -> uses pretrained)
    cfg.logger.tensorboard_dir = None  # W&B only (TB is enabled by a non-None dir; disable it)
    # W&B (project only — the API key comes from the WANDB_API_KEY env var, never committed).
    cfg.logger.wandb_project = "EuroVL"
    cfg.logger.wandb_exp_name = "qwen3-pa"
    cfg.logger.wandb_save_dir = f"{EUROVL_RUNS}/qwen3_pa/wandb"

    return cfg
