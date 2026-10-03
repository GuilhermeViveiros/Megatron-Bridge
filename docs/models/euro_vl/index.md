# EuroVL

EuroVL is a European vision-language model built in this repository: a **MoonViT-SO-400M**
vision encoder joined to a decoder-only language backbone through a patch-merging MLP
connector, trained with interleaved multimodal RoPE over images, multi-image documents and
video.

Unlike most families documented here, EuroVL is not an upstream Hugging Face release that the
Bridge imports — it is assembled here from a vision tower and a text backbone, so this page
describes both the architecture and the training pipeline that produces it.

## Backbone variants

The vision tower and connector are identical across variants; only the language model changes.

| Variant | Language backbone | Recipe prefix | Notes |
|---|---|---|---|
| **EuroVL-2B** (primary) | EuroLLM-1.7B-Instruct-2512 | `euro_vl_2b_*` | Multilingual European backbone; untied embeddings |
| **Qwen3EuroVL-2B** | Qwen3-1.7B | `qwen3_euro_vl_*` | Oracle arm — validates the stack against a proven M-RoPE implementation |
| **GemmaEuroVL** | Tower-Plus-2B (Gemma2-2B) | `gemma_euro_vl_*` | Backbone-comparison arm |

## Architecture

### Vision encoder — MoonViT-SO-400M

| | |
|---|---|
| patch size | 14 |
| layers | 27 |
| hidden size | 1152 |
| intermediate size | 4304 |
| attention heads | 16 |
| patch merge kernel | 2 x 2 |

Images are encoded at native aspect ratio. **Video is 2D-only**: every frame is encoded
independently by the same 2D vision tower, matching how Kimi-VL works — there is no temporal
grouping in the encoder.

### Language backbones

| | EuroLLM-1.7B | Qwen3-1.7B | Tower-Plus-2B |
|---|---|---|---|
| layers | 24 | 28 | 26 |
| hidden size | 2048 | 2048 | 2304 |
| FFN hidden | 5632 | 6144 | 9216 |
| attention heads | 16 | 16 | 8 |
| query groups (GQA) | 8 | 8 | — |
| head dim (`kv_channels`) | 128 | 128 | — |
| normalization | RMSNorm | RMSNorm + QK-LayerNorm | zero-centered RMSNorm |
| activation | SwiGLU (SiLU) | SwiGLU | fast GELU |
| `rope_theta` | 1,000,000 | 1,000,000 | — |
| embeddings | **untied** | tied | tied |

Two settings are load-bearing and easy to get wrong: EuroLLM's embeddings are **untied**
(`share_embeddings_and_output_weights=False`), and the activation is **SwiGLU/SiLU** — a
provider defaulting to GELU silently corrupts a frozen LLM without any error.

### Multimodal RoPE

Position encoding is **interleaved M-RoPE** with `mrope_section=[24, 20, 20]` (temporal,
height, width), which must sum to `head_dim // 2 = 64`. `position_embedding_type="mrope"` and
`apply_rope_fusion=False` are provider defaults and must survive checkpoint import — a
checkpoint whose `run_config.yaml` records plain `rope` cannot be exported.

Context parallelism (`context_parallel_size > 1`) is **not supported** with M-RoPE; use
activation recomputation for longer sequences instead.

## Vocabulary

The most error-prone part of the model. Three numbers are in play for the EuroLLM variant:

| Number | Value | Meaning |
|---|---|---|
| base vocab | 128,000 | EuroLLM text tokens |
| + vision specials | 128,005 | `<image>`, `<\|vision_start\|>`, `<\|vision_end\|>`, `<\|vision_pad\|>`, `<video>` |
| + grounding markers | **128,015** | 10 markers for boxes, points, quads and temporal spans |
| Megatron padded vocab | **128,512** | `EUROLLM_PADDED_VOCAB_SIZE` — pinned, not derived |

The padded size is **pinned** because Megatron's usual rule
(`make_vocab_size_divisible_by * TP`) produces a different size per tensor-parallel degree,
which makes a checkpoint saved at one TP unloadable at another. Rows above the real vocab are
zero-padded on import and trimmed on export, so the Hugging Face side always shows the true
vocab size.

Qwen3 needs no resize: it already carries object-ref/box/quad markers natively and has spare
rows inside its declared 151,936.

## Training stages

1. **PA (projector alignment)** — vision encoder and language model frozen, only the connector
   trains. Recipes: `*_pa_sft_config`.
2. **Full SFT** — all parameters train over the full instruction mixture. Recipes:
   `euro_vl_2b_sft_config`, `qwen3_euro_vl_2b_sft_config`.

The LR schedule is expressed **relatively** (`lr_decay_iters=None`, `lr_warmup_fraction=0.1`)
so it follows whatever `train_iters` a launch sets. Baking absolute counts from a recipe
default silently ruins long runs — the cosine completes early and the rest of training sits
pinned at `min_lr`.

## Recipes

All exported from [`src/megatron/bridge/recipes/euro_vl/euro_vl.py`](../../../src/megatron/bridge/recipes/euro_vl/euro_vl.py):

| Recipe | Stage | Backbone |
|---|---|---|
| `euro_vl_2b_pa_sft_config` | projector alignment | EuroLLM-1.7B |
| `euro_vl_2b_sft_config` | full SFT | EuroLLM-1.7B |
| `qwen3_euro_vl_pa_sft_config` | projector alignment | Qwen3-1.7B |
| `qwen3_euro_vl_2b_sft_config` | full SFT | Qwen3-1.7B |
| `gemma_euro_vl_pa_sft_config` | projector alignment | Tower-Plus-2B |

## Data pipeline

Training data is served by **Megatron-Energon**. A repeat-factor blend is generated at runtime
from a `mixture.yaml` of per-dataset weights, discovered under a configurable data root
(`EuroVLEnergonProvider`). Each sample is a media file plus a `{key}.json` holding a ChatML
conversation.

Three behaviours worth knowing:

* **Fill-to-`seq_length` packing** — several samples are packed into one sequence with
  `cu_seqlens`, so `train_iters` is not auto-derived and must be set explicitly at launch.
* **Marker-derived loss mask** — assistant answers are located structurally, between
  `<\|im_start\|>assistant\n` and the next `<\|im_end\|>`, never by searching for the answer
  text. The closing `<\|im_end\|>` **is** supervised: it is the EOS the model must learn to
  emit.
* **Message trees** — branching conversations are packed once with per-token subsegment ids
  and branch-isolated attention. See [message-tree-packing.md](message-tree-packing.md).

Samples whose attached media and `<image>`/`<video>` markers disagree, whose video fails to
decode, or which contain no supervisable answer are skipped with a warning rather than
silently mistrained.

## Source layout

| Path | Role |
|---|---|
| `models/euro_vl/euro_vl_bridge.py` | HF <-> Megatron conversion |
| `models/euro_vl/euro_vl_provider.py` | Megatron model provider (M-RoPE defaults) |
| `models/euro_vl/euro_vl_processor.py` | Image/video + text processor |
| `models/euro_vl/modeling_euro_vl.py` | Combined vision + language model |
| `models/euro_vl/modeling_euro_vl_hf.py` | Hugging Face-side backbone (`EuroVLTextForCausalLM`) |
| `models/euro_vl/moonvit/` | Vision encoder, image and video processors |
| `models/euro_vl/rope.py`, `branch_attention.py` | M-RoPE and message-tree attention |
| `models/euro_vl/gemma_euro_vl_*.py` | Gemma2/Tower-Plus backbone arm |
| `data/energon/euro_vl_task_encoder.py` | Sample encoding, loss mask, packing |
| `data/energon/euro_vl_energon_provider.py` | Repeat-factor blend from `mixture.yaml` |
| `recipes/euro_vl/euro_vl.py` | All training recipes |

## Further reading

* [architecture-trace.md](architecture-trace.md) — a running record of measured facts and the
  traps that cost time: vocabulary layering, weight-vs-tokenizer provenance, supervision
  rules, scaling numbers and the invariants to assert against.
* [message-tree-packing.md](message-tree-packing.md) — branch-isolated attention for
  tree-structured conversations.
