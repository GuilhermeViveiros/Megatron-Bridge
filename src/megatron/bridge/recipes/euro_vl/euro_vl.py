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

Two configs for EuroLLM-1.7B-Instruct-2512 + MoonViT-SO-400M (interleaved M-RoPE), over real
Megatron-Energon weighted blends: :func:`euro_vl_2b_pa_sft_config` (projector-alignment
stage-1: freeze LLM+vision, train only the projector, caption-only blend) and
:func:`euro_vl_2b_sft_config` (stage-2: all modules trainable over the complete multi-category
``mixture.yaml`` blend, REQUIRES a PA checkpoint to already exist -- it continues from the
projector PA trained, not a random-init one).

This module also provides the Qwen3EuroVL oracle and a standalone Gemma2 backbone
experiment (see their own sections below).
"""

import logging
import os

import torch

from megatron.bridge.models.euro_vl.euro_vl_processor import EuroVLProcessor
from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider, Qwen3EuroVLModelProvider
from megatron.bridge.models.euro_vl.moonvit import MoonViTConfig
from megatron.bridge.models.euro_vl.utils import EUROLLM_PADDED_VOCAB_SIZE
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)

_SCRATCH = os.environ["SCRATCH"]
# Assembled EuroVL HF checkpoint (from eurollm_bridge.py), built on EuroLLM-1.7B-Instruct-2512
# (rope_theta=1000000, max_position_embeddings=32768 -- NOT the stale 10000/4096 an older
# EuroLLM-1.7B-Instruct revision carried, which would silently corrupt the frozen LLM if used).
# Holds the tokenizer (vision tokens + chat template) and the MoonViT image-processor config.
EUROVL_HF = f"{_SCRATCH}/hf_models/euro_vl_2b_2512_hf"
# Megatron-format conversion of the above (via convert_checkpoints.py import). The training
# loader only recognizes Megatron checkpoints for pretrained_checkpoint — pointing it at the
# raw HF dir silently loads nothing (checkpoint_exists() is False -> random init).
# `_v128512`: vocab padded from 128005 to 128512 rows so the (odd) EuroLLM vocab can be split
# across TP ranks; all other weights are byte-identical to euro_vl_2b_2512 (see
# sanity_check/pad_vocab_checkpoint.py and euro_vl/utils.py).
EUROVL_MCORE = f"{_SCRATCH}/megatron_models/euro_vl_2b_2512_v128512"
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
    """Build EuroVLModelProvider: EuroLLM-1.7B-Instruct-2512 + MoonViT-SO-400M, M-RoPE.

    M-RoPE (interleaved t/h/w position ids) is :class:`EuroVLModelProvider`'s default as of
    2026-09 -- ``position_embedding_type``/``apply_rope_fusion``/``mrope_section``/
    ``qk_layernorm`` all come from the provider's own defaults and are not overridden here.
    """
    return EuroVLModelProvider(
        # EuroLLM-1.7B architecture
        num_layers=24,
        hidden_size=2048,
        ffn_hidden_size=5632,
        num_attention_heads=16,
        num_query_groups=8,
        kv_channels=128,  # EuroLLM head_dim; mrope_section=[24,20,20] sums to half of this
        # vocab_size extended: 128000 (EuroLLM) + 5 vision special tokens (128000-128004)
        vocab_size=128005,
        make_vocab_size_divisible_by=128,
        # Pad the (odd) 128005 vocab up to a TP-divisible size; required for TP>1
        # (VocabParallelEmbedding splits vocab across TP ranks). Harmless at TP=1.
        should_pad_vocab=True,
        # One pinned size instead of Megatron's per-TP rule, so a checkpoint saved at one TP
        # degree loads at TP=1/2/4 (see euro_vl/utils.py). Checkpoints written before this was
        # introduced hold 128005 rows and must be migrated (sanity_check/pad_vocab_checkpoint.py).
        padded_vocab_size=EUROLLM_PADDED_VOCAB_SIZE,
        seq_length=32768,  # EuroLLM-1.7B-Instruct-2512 max_position_embeddings -> overrided to 8192 for the instruct training stage
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        rotary_base=1000000,  # EuroLLM-1.7B-Instruct-2512's rope_theta; must match the base model
        rotary_percent=1.0,
        gated_linear_unit=True,
        # EuroLLM is SwiGLU (SiLU-gated MLP). The TransformerConfig default is GELU
        activation_func=torch.nn.functional.silu,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        add_qkv_bias=False,
        share_embeddings_and_output_weights=False,
        bias_activation_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
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
def _build_base_sft_config(seq_length: int = 8192) -> ConfigContainer:
    """Shared builder for :func:`euro_vl_2b_sft_config` and :func:`euro_vl_2b_pa_sft_config`.

    ``seq_length`` must be chosen here, not by a CLI override afterwards: the processor (whose
    video token budget is a fraction of it), the task encoder (pack length) and the provider are
    all built from it below. ``run_recipe.py --seq_length N`` passes it through.

    Model, parallel settings, kernels, real energon ``mixture.yaml`` blend, packing, and DDP --
    everything except which checkpoint to initialize from, since the two public configs disagree
    on that (:func:`euro_vl_2b_sft_config` requires a PA checkpoint; :func:`euro_vl_2b_pa_sft_config`
    is the stage that produces one, so it must not depend on one existing yet -- it cannot call
    :func:`euro_vl_2b_sft_config` as its base the way the Qwen3 PA config calls its own SFT
    config, since that would make PA require a PA checkpoint to run).
    """
    from megatron.bridge.data.energon.euro_vl_energon_provider import EuroVLEnergonProvider
    from megatron.bridge.data.energon.euro_vl_task_encoder import EuroVLTaskEncoder

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

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    # "te", not "native": benchmarked on one GH200 at EuroVL shapes (logits 8192x1x128005) the
    # TE fused kernel is 5.29 ms vs 17.35 ms fwd+bwd and allocates 0 MiB of extra peak vs ~9.8 GiB,
    # at identical numerics (both 1.91e-06 max abs err vs an fp32 reference). The native path
    # materializes an fp32 copy of the logits, which is what OOM'd 16k training (7.81 GiB alloc).
    cfg.model.cross_entropy_fusion_impl = "te"

    # Long context for the energon stage. Set before building the task encoder / provider
    # below so seq_length propagates to both (they read cfg.model.seq_length at build).
    cfg.model.seq_length = seq_length

    # Training config
    cfg.train.train_iters = 500
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # square-root normalized per-token loss (Qwen3-VL)
    cfg.model.calculate_per_token_loss = True

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

    # Real Megatron-Energon data path. The processor loads the tokenizer (vision tokens +
    # chat template) and MoonViT image-processor config from the single assembled HF
    # checkpoint dir.
    processor = EuroVLProcessor.from_pretrained(EUROVL_HF, seq_length=cfg.model.seq_length)
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
        # Do NOT let energon eagerly decode media: with auto_decode=False every sample reaches
        # the task encoder as raw bytes, so _cook does bounded, native-resolution, on-demand
        # decode (images via PIL; video via video_processor.decode_video_bytes -> only the
        # policy-planned frame count) instead of energon decoding whole clips into RAM
        # (~2.6 GB/clip -> OOM/slow buffer fill). Without it EVERY video sample raises
        # "auto-decoded video has no fps for timestamps" in _frames_from_video and is dropped
        # (measured: 62,788 failures / 0 iterations on a video-only mixture, job 1911843).
        energon_dataset_kwargs={"auto_decode": False},
    )

    # Load the assembled EuroVL weights (EuroLLM + MoonViT; projector random). MUST be the
    # Megatron-converted dir, NOT the HF dir (the loader silently skips a raw HF checkpoint
    # -> random init -> loss ~ ln(vocab)). Callers may override this.
    cfg.checkpoint.pretrained_checkpoint = EUROVL_MCORE

    # Same override the Qwen3/Gemma arms use (missing here previously -- confirmed the eurollm_pa
    # checkpoints' bundled iter_*/tokenizer dirs were all empty as a result): without
    # tokenizer_type="HuggingFaceTokenizer" this defaults to NullTokenizer, whose
    # save_tokenizer_assets branch writes nothing (the dir still gets created, just left empty).
    # Inference/eval can then load the tokenizer from the checkpoint itself.
    cfg.checkpoint.save_tokenizer_assets = True
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = EUROVL_HF

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    # Per-token loss requires the DP grad collective to SUM, not average: MCore sets
    # gradient_scaling_factor=1.0 and the single global division by the total weight-sum
    # (Sum_i sqrt(T_i)) happens once in finalize_model_grads — the Σwℓ/Σw of the sqrt
    # loss. MCore hard-asserts on average_in_collective=True + per-token loss at DDP init.
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"

    return cfg


def euro_vl_2b_sft_config(seq_length: int = 8192) -> ConfigContainer:
    """SFT config for EuroVL-2B (EuroLLM-1.7B-Instruct-2512 + MoonViT-SO-400M, M-RoPE) —
    all modules trainable, over a real Megatron-Energon (Crude) weighted blend.

    **Requires a PA checkpoint to already exist** (see :func:`euro_vl_2b_pa_sft_config`):
    initializes from the highest-iteration ``iter_*`` checkpoint under ``euro_vl_runs/eurollm_pa``,
    not the random-projector base checkpoint. Raises ``FileNotFoundError`` if no PA checkpoint
    exists yet -- this stage should never silently start from a random-init projector, which
    would make PA-vs-no-PA comparisons meaningless.

    Default: 1 node, 1 GPU (2B model fits on a single GPU at TP=1).

    The data mix is built at runtime from a directory of prepared datasets (``root``) and
    ``mixture.yaml`` in that directory — InternVL-style **repeat factors** ``r`` per dataset
    (epochs per dataset; effective samples = ``r * size``; ``r=0`` excludes; ``r in [0,4]``).
    Edit ``energon-data/mixture.yaml`` to control the blend; a missing file = all ``r=1``::

        image/captioning:
          cc12m: 0.3
          sharegpt4o: 2.0
          ...
        multimage/..
        video/..

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
    cfg = _build_base_sft_config(seq_length=seq_length)

    # Requires a PA checkpoint to already exist -- this stage continues from the projector PA
    # trained, not a random-init one. Raises rather than silently falling back to the
    # random-projector base checkpoint, which would make PA-vs-no-PA comparisons meaningless.
    # `_v128512` holds the vocab-padded copy of the PA run (sanity_check/pad_vocab_checkpoint.py):
    # the model is now built with a padded vocabulary, so the original 128005-row checkpoint no
    # longer matches. Weights are otherwise byte-identical to eurollm_pa_v2/iter_0003720.
    pa_run_dir = f"{EUROVL_RUNS}/eurollm_pa_v2_v128512"
    pa_checkpoint = f"{pa_run_dir}/iter_0003720/"
    if not os.path.exists(pa_checkpoint):
        raise FileNotFoundError(
            f"No PA checkpoint found under {pa_run_dir!r} (expected iter_* subdirs). "
            "Run euro_vl_2b_pa_sft_config first."
        )
    cfg.checkpoint.pretrained_checkpoint = pa_checkpoint
    logger.info("Continuing from PA checkpoint: %s", cfg.checkpoint.pretrained_checkpoint)

    # Full-SFT schedule: same shape as PA (10% linear warmup, cosine decay over the run), but a
    # lower peak LR since this stage fine-tunes the whole model (LLM+vision+projector) rather
    # than just aligning a random-init projector, and stronger weight decay (0.05 vs PA's 0.01
    # Nemotron Stage 0 value) since a full-parameter update needs more regularization than a
    # projector-only one to avoid overfitting the larger trainable surface.
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=round(0.1 * cfg.train.train_iters),
        lr_decay_iters=cfg.train.train_iters,
        max_lr=2e-5,
        min_lr=2e-6,
        weight_decay=0.05,
    )
    opt_cfg.use_precision_aware_optimizer = False
    opt_cfg.main_grads_dtype = torch.float32
    opt_cfg.main_params_dtype = torch.float32
    opt_cfg.exp_avg_dtype = torch.float32
    opt_cfg.exp_avg_sq_dtype = torch.float32
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    cfg.checkpoint.save = f"{EUROVL_RUNS}/eurollm_sft"
    cfg.checkpoint.load = cfg.checkpoint.save  # resume from same dir (empty 1st run -> uses pretrained)
    cfg.logger.tensorboard_dir = None  # W&B only (TB is enabled by a non-None dir; disable it)
    cfg.logger.wandb_project = "EuroVL"
    cfg.logger.wandb_exp_name = "eurollm-sft"
    cfg.logger.wandb_save_dir = f"{EUROVL_RUNS}/eurollm_sft/wandb"

    return cfg


def euro_vl_2b_pa_sft_config(seq_length: int = 8192) -> ConfigContainer:
    """EuroVL-2B (EuroLLM-1.7B-Instruct-2512 + MoonViT, M-RoPE) **projector-alignment (PA /
    stage-1)** config.

    Freezes the pretrained LLM + vision tower and trains ONLY the (random-init) multimodal
    projector so it learns to map MoonViT features into the EuroLLM embedding space — the
    standard VLM stage-1 alignment, mirroring :func:`qwen3_euro_vl_pa_sft_config`. Starts from
    the random-projector base checkpoint (NOT :func:`euro_vl_2b_sft_config`, which requires a
    PA checkpoint that this function is the one that produces).

    Hyperparameters match Nemotron's published Stage 0 (projector-alignment) config, same as
    the Qwen3 PA run: ``max_lr=2e-4``, ``global_batch_size=1024``, 10% linear warmup,
    ``weight_decay=0.01``, cosine decay over the run. ``train_iters=2158`` is the same ~1-epoch
    figure as the Qwen3 PA run since both consume the same ``mixture_pa.yaml`` blend at the same
    global_batch_size.

    Launch (freezing + LR/batch baked in)::

        uv run --no-sync python -m torch.distributed.run --nproc_per_node=4 \\
            scripts/training/run_recipe.py --recipe euro_vl_2b_pa_sft_config --step_func vlm_step \\
            train.train_iters=2158 dataset.num_workers=8
    """
    cfg = _build_base_sft_config(seq_length=seq_length)

    # PA: freeze the pretrained LLM + vision tower; train only the projector.
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False

    # Caption-only PA blend (image/multiimage/video captioning), not the full mixture.yaml.
    cfg.dataset.mixture_file = os.path.join(EUROVL_ENERGON_ROOT, "mixture_pa.yaml")

    # PA does NOT use the sqrt-weighted per-token loss: it needs calculate_per_token_loss=False
    # (local per-microbatch mean loss + averaged DP collective) for the freeze/small-LR projector
    # regime below, which is the OPPOSITE pairing from the sqrt scheme (that requires
    # calculate_per_token_loss=True + average_in_collective=False, see euro_vl_2b_sft_config
    # above). The base config's task_encoder was already constructed with
    # sqrt_loss_weighting=True; left unchanged here, the loss_mask would still carry per-sample
    # 1/sqrt(N) weights into a mean-loss computation that expects uniform weights, making the
    # per-microbatch normalization pack-composition-dependent instead of the intended global
    # Σwℓ/Σw. Turn sqrt weighting off for PA so the loss_mask matches the loss path actually used.
    cfg.dataset.task_encoder.sqrt_loss_weighting = False
    cfg.model.calculate_per_token_loss = False
    cfg.ddp.average_in_collective = True

    # global_batch_size=1024 already comes from the base config's default; no override needed.

    # Projector-only schedule: Nemotron-matched LR/warmup/weight_decay + cosine decay over the run.
    cfg.train.train_iters = 2158  # ~1 epoch of the blend; same figure as the Qwen3 PA run
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=round(0.1 * cfg.train.train_iters),  # Nemotron: 10% linear warmup
        lr_decay_iters=cfg.train.train_iters,  # cosine completes over the actual run
        max_lr=2e-4,
        min_lr=2e-5,
        weight_decay=0.01,  # Nemotron Stage 0
    )
    # Preserve the fp32 optimizer-state settings the base config relies on.
    opt_cfg.use_precision_aware_optimizer = False
    opt_cfg.main_grads_dtype = torch.float32
    opt_cfg.main_params_dtype = torch.float32
    opt_cfg.exp_avg_dtype = torch.float32
    opt_cfg.exp_avg_sq_dtype = torch.float32
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Write checkpoints/logs to scratch (PA subdir), NOT the $HOME-bound cwd. PA converges fast,
    # so save more often than the SFT default.
    cfg.checkpoint.save = f"{EUROVL_RUNS}/eurollm_pa"
    cfg.checkpoint.load = cfg.checkpoint.save  # resume from same dir (empty 1st run -> uses pretrained)
    cfg.logger.tensorboard_dir = None  # W&B only (TB is enabled by a non-None dir; disable it)
    # W&B (project only — the API key comes from the WANDB_API_KEY env var, never committed).
    cfg.logger.wandb_project = "EuroVL"
    cfg.logger.wandb_exp_name = "eurollm-pa"
    cfg.logger.wandb_save_dir = f"{EUROVL_RUNS}/eurollm_pa/wandb"

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
        # Qwen3 is SwiGLU (SiLU-gated MLP). The TransformerConfig default is GELU
        activation_func=torch.nn.functional.silu,
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
    Mirrors :func:`euro_vl_2b_sft_config` (packing, sqrt loss, native CE); differences:
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
    cfg.model.cross_entropy_fusion_impl = "te"  # see _build_base_sft_config for the benchmark

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

    # seq_length sets the video token budget (fps + total_pixels smart-resize policy).
    processor = Qwen3EuroVLProcessor.from_pretrained(QWEN3_EUROVL_HF, seq_length=cfg.model.seq_length)
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
        # decode (images via PIL; video via video_processor.decode_video_bytes -> only the
        # policy-planned frame count) instead of energon decoding whole clips into RAM
        # (~2.6 GB/clip -> OOM/slow buffer fill).
        energon_dataset_kwargs={"auto_decode": False},
    )

    # Load the pretrained weights (Qwen3 LLM + MoonViT; projector random) from the
    # Megatron-converted checkpoint at setup time. This is the whole point of the oracle —
    # training starts from real weights, not random init. MUST be the Megatron dir, NOT the
    # HF dir (the loader silently skips a raw HF checkpoint -> random init -> loss ~ln(vocab)).
    cfg.checkpoint.pretrained_checkpoint = QWEN3_EUROVL_MCORE

    # Self-contained checkpoints: train with the REAL assembled Qwen3EuroVL tokenizer (vision
    # special tokens + chat template) instead of the _sft_common_vlm NullTokenizer placeholder.
    # The energon task encoder still does the actual tokenization via its own processor; this
    # override only (a) sizes the vocab — harmless, since model.vocab_size=151936 is preset and
    # >= this tokenizer's vocab, so _validate_and_set_vocab_size keeps 151936 (resume-compatible)
    # — and (b) makes save_tokenizer_assets write real tokenizer files into every iter_*/tokenizer
    # (the NullTokenizer branch writes nothing, leaving an empty dir). Inference can then load the
    # tokenizer from the checkpoint itself instead of a separate hardcoded path.
    cfg.checkpoint.save_tokenizer_assets = True
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = QWEN3_EUROVL_HF

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

    Hyperparameters match **Nemotron**'s published Stage 0 (projector-alignment) config:
    ``max_lr=2e-4``, ``global_batch_size=1024``, 10% linear warmup, ``weight_decay=0.01``,
    cosine decay over the run. ``max_lr`` sits in the validated band (Nemotron 2e-4 → in-repo
    Qwen3-VL 3e-4 → LLaVA 1e-3). Set ``train_iters`` to ~1 epoch of the blend (read the blue
    ``Suggested train_iters`` line the provider logs at startup) and sweep ``max_lr`` if tuning.
    seq_length stays at 8192 (not matched to Nemotron's 16384) -- that's a real compute/memory
    change, deliberately deferred.

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

    # PA does NOT use the sqrt-weighted per-token loss: it needs calculate_per_token_loss=False
    # (local per-microbatch mean loss + averaged DP collective) for the freeze/small-LR projector
    # regime below, which is the OPPOSITE pairing from the sqrt scheme (that requires
    # calculate_per_token_loss=True + average_in_collective=False, see the base SFT config
    # above). The base config's task_encoder was already constructed with
    # sqrt_loss_weighting=True; left unchanged here, the loss_mask would still carry per-sample
    # 1/sqrt(N) weights into a mean-loss computation that expects uniform weights, making the
    # per-microbatch normalization pack-composition-dependent instead of the intended global
    # Σwℓ/Σw. Turn sqrt weighting off for PA so the loss_mask matches the loss path actually used.
    cfg.dataset.task_encoder.sqrt_loss_weighting = False
    cfg.model.calculate_per_token_loss = False
    cfg.ddp.average_in_collective = True

    cfg.train.global_batch_size = 1024
    cfg.dataset.global_batch_size = 1024

    # Projector-only schedule: Nemotron-matched LR/warmup/weight_decay + cosine decay over the run.
    cfg.train.train_iters = 2158  # ~1 epoch of the blend; size from the blue train_iters log
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=round(0.1 * cfg.train.train_iters),  # Nemotron: 10% linear warmup
        lr_decay_iters=cfg.train.train_iters,  # cosine completes over the actual run
        max_lr=2e-4,
        min_lr=2e-5,
        weight_decay=0.01,  # Nemotron Stage 0
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
    cfg.checkpoint.save = f"{EUROVL_RUNS}/qwen3_pa_pyav_vect"
    cfg.checkpoint.load = cfg.checkpoint.save  # resume from same dir (empty 1st run -> uses pretrained)
    cfg.logger.tensorboard_dir = None  # W&B only (TB is enabled by a non-None dir; disable it)
    # W&B (project only — the API key comes from the WANDB_API_KEY env var, never committed).
    cfg.logger.wandb_project = "EuroVL"
    cfg.logger.wandb_exp_name = "qwen3-pa"
    cfg.logger.wandb_save_dir = f"{EUROVL_RUNS}/qwen3_pa/wandb"

    return cfg


# ---------------------------------------------------------------------------
# Gemma2 (Tower-Plus-2B) backbone experiment — standalone, additive.
# Everything below is self-contained: no function above references it.
# ---------------------------------------------------------------------------

# Assembled by sanity_check/gemma_euro_vl_assembly.py, then imported to Megatron.
GEMMA_EUROVL_HF = f"{_SCRATCH}/hf_models/gemma_euro_vl_hf"
GEMMA_EUROVL_MCORE = f"{_SCRATCH}/megatron_models/gemma_euro_vl"


def _make_gemma_euro_vl_provider():
    """Build GemmaEuroVLModelProvider: Tower-Plus-2B (Gemma2-2B) backbone + MoonViT-SO-400M.

    Gemma2's distinctive settings (zero-centered RMSNorm, fast_gelu, the four-norm layer
    spec, sliding-window pattern, ``query_pre_attn_scalar`` -> ``softmax_scale``, final
    logit softcapping) come from the :class:`GemmaEuroVLModelProvider` defaults.
    """
    from megatron.bridge.models.euro_vl.gemma_euro_vl_provider import GemmaEuroVLModelProvider

    return GemmaEuroVLModelProvider(
        # Tower-Plus-2B architecture (Gemma2-2B).
        num_layers=26,
        hidden_size=2304,
        ffn_hidden_size=9216,
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=256,  # Gemma2 head_dim (256, wider than hidden/heads)
        # Vision tokens reuse Gemma2's <unused0..4> slots (ids 7-11) — no vocab extension.
        # 256000 is already divisible by 128, so no padding is required either.
        vocab_size=256000,
        make_vocab_size_divisible_by=128,
        should_pad_vocab=False,
        seq_length=8192,  # Gemma2's full max_position_embeddings
        bf16=True,
        params_dtype=torch.bfloat16,
        # Vision — same vendored MoonViT-SO-400M as every other EuroVL variant.
        vision_config=MoonViTConfig(),
        projector_input_dim=4608,  # 4 * 1152 (MoonViT hidden * merge kernel area)
        projector_output_dim=2304,  # Tower-Plus-2B hidden size
        use_bidirectional_image_attention=False,
        # Repurposed <unused*> ids; must match GemmaEuroVLConfig / GemmaEuroVLProcessor.
        image_token_id=7,
        vision_start_token_id=8,
        vision_end_token_id=9,
        vision_pad_token_id=10,
        video_token_id=11,
    )


def gemma_euro_vl_pa_sft_config() -> ConfigContainer:
    """GemmaEuroVL **projector-alignment (PA / stage-1)** config — Tower-Plus-2B + MoonViT.

    The LLM-backbone arm of the EuroVL-vs-TowerVision investigation. TowerVision's PA stage
    beats our Qwen3 PA (and even our full SFT) on matched pixmo-cap-only data, and its
    backbone is Tower-Plus-2B. This recipe puts *that* backbone in *our* framework so the
    remaining gap can be attributed to the backbone or to the pipeline, but not both.

    Deliberately matched to the ``qwen3_pa_pixmo_only`` baseline so the only intended
    difference is the LLM: same MoonViT tower, same pixmo-cap-only blend, same
    projector-only freezing, same GBS/iters/LR (TowerVision-stage1-matching: lr=1e-3,
    ~1 epoch = ceil(690000/384) = 1797 iters).

    Two unavoidable secondary differences from the Qwen3 arm, both inherent to Gemma2:
    it uses 1D RoPE rather than M-RoPE, and attention-logit softcapping is dropped (see
    ``gemma_euro_vl_provider``). The former actually brings this arm *closer* to
    TowerVision, which is also 1D.

    Launch::

        sbatch scripts/slurm_pa_pixmo_only_gemma.sh
    """
    from megatron.bridge.data.energon.euro_vl_energon_provider import EuroVLEnergonProvider
    from megatron.bridge.data.energon.euro_vl_task_encoder import EuroVLTaskEncoder
    from megatron.bridge.models.euro_vl.gemma_euro_vl_hf import GemmaEuroVLProcessor

    cfg = _sft_common_vlm()

    cfg.model = _make_gemma_euro_vl_provider()

    # 2B-scale: TP=1 is fastest (see throughput notes for the Qwen3 arm).
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # PA: freeze the pretrained LLM + vision tower; train only the random-init projector.
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"  # see _build_base_sft_config for the benchmark

    cfg.model.seq_length = 8192

    # PA uses the plain mean loss, NOT the sqrt-weighted per-token scheme: per-token loss
    # pairs with average_in_collective=False, which is the opposite pairing from this
    # freeze/projector regime. Keep loss_mask uniform to match the loss path actually used.
    cfg.model.calculate_per_token_loss = False
    cfg.ddp.average_in_collective = True

    # Matched to qwen3_pa_pixmo_only.
    cfg.train.train_iters = 1797  # ~1 epoch of pixmo-cap at GBS=384
    cfg.train.global_batch_size = 384
    cfg.train.micro_batch_size = 1

    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=90,
        lr_decay_iters=cfg.train.train_iters,
        max_lr=1e-3,  # TowerVision-2B stage-1 projector LR
        min_lr=1e-4,
    )
    opt_cfg.use_precision_aware_optimizer = False
    opt_cfg.main_grads_dtype = torch.float32
    opt_cfg.main_params_dtype = torch.float32
    opt_cfg.exp_avg_dtype = torch.float32
    opt_cfg.exp_avg_sq_dtype = torch.float32
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"

    processor = GemmaEuroVLProcessor.from_pretrained(GEMMA_EUROVL_HF, seq_length=cfg.model.seq_length)
    task_encoder = EuroVLTaskEncoder(
        processor=processor,
        seq_length=cfg.model.seq_length,
        sqrt_loss_weighting=False,  # see calculate_per_token_loss note above
    )
    cfg.dataset = EuroVLEnergonProvider(
        root=EUROVL_ENERGON_ROOT,
        mixture_file=os.path.join(EUROVL_ENERGON_ROOT, "mixture_pa_pixmo_only.yaml"),
        seq_length=cfg.model.seq_length,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        num_workers=8,
        task_encoder=task_encoder,
        packing_buffer_size=60,
        pack_sequences_in_batch=False,
        energon_dataset_kwargs={"auto_decode": False},
    )

    # MUST be the Megatron-converted dir, NOT the HF dir (the loader silently skips a raw
    # HF checkpoint -> random init -> loss ~ ln(vocab)).
    cfg.checkpoint.pretrained_checkpoint = GEMMA_EUROVL_MCORE

    cfg.checkpoint.save_tokenizer_assets = True
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = GEMMA_EUROVL_HF

    cfg.checkpoint.save = f"{EUROVL_RUNS}/gemma_pa_pixmo_only"
    cfg.checkpoint.load = cfg.checkpoint.save  # resume from same dir (empty 1st run -> pretrained)
    cfg.logger.tensorboard_dir = None  # W&B only
    cfg.logger.wandb_project = "EuroVL"
    cfg.logger.wandb_exp_name = "gemma-pa-pixmo-only"
    cfg.logger.wandb_save_dir = f"{EUROVL_RUNS}/gemma_pa_pixmo_only/wandb"

    return cfg
