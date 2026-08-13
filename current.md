# EuroVL — Current Status & Decisions

_Last updated: 2026-06-15. Branch: `feat/euro-vl` (pushed to `fork`, HEAD `9485716b`)._

EuroVL = EuroLLM-1.7B (24L, hidden 2048, ffn 5632, 16 heads / 8 KV, head_dim 128,
rope_theta **1e6**, 32K ctx, untied embeddings, vocab 128000+5 vision = 128005) +
MoonViT-SO-400M vision tower (per-frame 2D, 2×2 spatial merge). Hardware: 4×GH200 (Hopper).
Run in-container: `apptainer exec --nv … megatron-bridge.sif uv run --no-sync …`.

---

## ✅ Done recently

- **PA (projector-alignment) recipe (2026-08-03)** — `qwen3_euro_vl_pa_config` in `recipes/euro_vl/euro_vl.py`
  (exported in the family `__init__`). Reuses `qwen3_euro_vl_sft_energon_config` then: freeze LLM + vision,
  train **only** the projector; **InternVL** schedule `max_lr=2e-4`, `min_lr=2e-5`, **GBS=512**, ~3% warmup,
  cosine over the run, `train_iters=2000` default.
  - **LR grounding**: Qwen3-VL Stage 0 (arXiv 2511.21631) = train only the MLP merger, vision+LLM frozen,
    ~67B tokens, seq 8192; in-repo `qwen_vl/qwen3_vl.py` default is projector-only at `lr=3e-4`; Nemotron 2e-4;
    LLaVA 1e-3. So 2e-4 sits in the validated band. "Best" is empirical — sweep `max_lr` if tuning.
  - **GBS=512 gotcha**: the energon provider captures `global_batch_size` at construction, so the recipe sets
    it on **both** `cfg.train` and `cfg.dataset` (provider reads it at build/setup, after the override).
  - **PA freezes** LLM+vision → optimizer/grads only for the tiny projector → ~10–15 GB GPU (huge margin). The
    recipe defaults are all-trainable; PA freezing is what `qwen3_euro_vl_pa_config` turns on (was the cause of
    the earlier full-training 42 GB GPU OOM — recipe defaults `freeze_*=False`).
- **Video path end-to-end (2026-08)** — model + data now consume video (per-frame 2D MoonViT):
  - **Forward**: `EuroVLModel` / `Qwen3EuroVLModel` take `pixel_values_videos` + `video_grid_thw`; video
    vision block mirrors the image block (`repeat_interleave` → per-frame `(h,w)` → vision tower → projector →
    `masked_scatter` into `video_token_id`). `get_rope_index` already split per-frame; now fed real
    `video_grid_thw` (removed the P4 `None`).
  - **Data** (`euro_vl_task_encoder.py`): video `visual_keys`, `_cook` detects `mp4`/`vid{i}` (images now
    optional), `_decode_video_bytes` (PyAV, `video_num_frames=8`). Base `hf_encoder_task_encoder.py` now
    converts `<video>`→`{"type":"video"}` (was image-only — the real data's literal `<video>` would not have
    expanded on the Qwen3 backbone) + symmetric `_video_token_id`.
  - **Inference**: `sanity_check/euro_vl_test_inference.py` uses the real `videos=` path; unified
    `MoonViTVisionProcessor` (image+video). Demo verified working by user.
  - **Data layout confirmed**: `energon-data/video/captioning/molmo2_cap` = CrudeWebdataset `{key}.mp4`+`.json`,
    conversation uses literal `<video>`.
  - **PA blend**: `mixture_captions.yaml` → **`mixture_pa.yaml`** (caption-only across image/multiimage/video);
    recipe `qwen3_euro_vl_sft_energon_config` points at it. First smoke = **video-only** (`molmo2_cap: 1.0`).
  - **Token budget tool**: `sanity_check/euro_vl_token_budget.py` (MoonViT uncapped vs SigLIP2 tiling plots).
  - **Limitation**: base truncation repairs partial *image* blocks only; keep video samples ≤ `seq_length`
    until per-video truncation repair is wired (would use the new `_video_token_id`).
  - **Video-training debugging (2026-08-03)**, in order fixed: worker **OOM** (my full-clip PIL decode →
    memory-bounded `_decode_video_bytes`); **all-tokens-masked** (8 frames×1024 tok = full 8192 seq →
    truncated responses → `num_frames=4`, single source of truth = processor `num_frames`, encoder reads it
    via `_video_num_frames`); `make_batched_videos` **IndexError** + `VideoData` **TypeError** (energon
    auto-decodes `.mp4` → `VideoData.frames` `[T,C,H,W]`; `_frames_from_video` + layout-aware `_frame_to_pil`
    handle it); **iter-0 stall** (256-buffer × full-clip decode). Final fix: **`auto_decode=False`**
    (`energon_dataset_kwargs` passthrough added to base `EnergonProvider`) → media reach `_cook` as raw
    **bytes** → bounded native-res decode (images via PIL, video via `_decode_video_bytes`, 4 frames ~25 MB
    vs whole-clip 2.6 GB). Chosen over AVData (which forces a 224² resize that breaks MoonViT dynamic-res).
    Scoped to the qwen3 recipe only. See #28 for the frozen→unfrozen tensor/vectorized follow-up.
- **Sequence packing** (energon fill-to-`seq_length`, THD via `cu_seqlens`) — validated.
- **Sqrt per-token loss** (InternVL3.5 eq.2) — `1/√N` weight in `loss_mask`, `calculate_per_token_loss=True`,
  `num_tokens` kept float in `losses.py`. Verified by `sanity_check/_sqrt_loss_sanity.py` (math/packing-indep)
  + `_sqrt_loss_encode_check.py` (real `encode_sample`). Committed in `9485716b`.
- **Vocab/rope fixes**: `should_pad_vocab` honored at setup (TP>1), `rotary_base=1e6`, `max_pos=32768`.
- **Throughput investigation** — ~340 TFLOP/s is the genuine ceiling on 4×GH200 8K. **FP8 tested → ZERO gain**
  (vision tower + flash-attn dominate, not LLM GEMMs; FA2 has no FP8). All single-node knobs flat
  (workers, GBS, buffer, CE impl, DP overlap, FP8). Stay bf16.

---

## 🔄 In progress — M-RoPE (task #20)

**Goal:** replace 1D RoPE with Qwen3-VL **interleaved** M-RoPE (3D `(t,h,w)` positions).

### Decisions locked
- **Qwen3-VL INTERLEAVED** (not Qwen2.5-VL chunked). Chunked confines each axis to a contiguous freq band
  (imbalanced spectrum, poor long-video); interleaved (`apply_interleaved_mrope`, stride-3) spreads each
  axis across the whole spectrum → balanced. MCore native `position_embedding_type="mrope"` is chunked-only,
  so interleaved needs a **custom rotary embedding** (heavy wiring).
- **Option A (model-side):** compute 3D `position_ids` in `modeling_euro_vl.forward` (all PP stages), like
  Qwen `modelling_qwen3_vl/model.py:598`. **No `euro_vl_step` needed** — `vlm_step` already passes
  `input_ids`, `image_grid_thw`, `video_grid_thw`, and `cu_seqlens` to the model forward.
- **Vendor** the rotary embedding into `models/euro_vl/rope.py` (repo convention = self-contained model dirs).
- `mrope_section=[24,20,20]` (sums to 64 = head_dim/2). `apply_rope_fusion=False` (fused kernel can't do
  interleaved mrope → throughput cost).
- **Temporal lives in the LLM, not MoonViT.** MoonViT is per-frame 2D; the LLM M-RoPE carries video temporal.
  EuroVL uses the **Qwen3-VL timestamp video format**: per-frame blocks `<ts><vis_start>VID…<vis_end>`, and
  **`t_index` is always 0** — temporal ordering comes from the **timestamp TEXT advancing the base** (NOT a
  t-axis increment; that's the older Qwen2-VL behavior).

### 🧪 Validation scaffold — "MoonViT + Qwen3" oracle (NEW, user idea 2026-06-15)
De-risk before touching EuroLLM. Key structural fact: **EuroVL wraps a STOCK Megatron `GPTModel`**
(`euro_vl_provider.py:87`, `modeling_euro_vl.py:96`) which computes rotary internally (1D, no mrope hook)
— that's why P3 is hard. **Qwen3-VL has a proven custom mrope-aware GPT stack** (`Qwen3VLModel`,
config-parametrized).
- **Idea (elevated 2026-06-15): `QwenEuroVLModel`** — a sibling class in `modeling_euro_vl.py` that keeps
  our MoonViT + projector but swaps the backbone to a **Qwen3-config LLM on the mrope-aware stack**. Backbone
  becomes a swappable config; vision integration is shared with `EuroVLModel`.
- **Not just a scaffold — a first-class variant + oracle.** Strategy framing (user): the **recipe + data is
  the IP**; backbone is fungible. A Qwen3-backbone VLM (same LLM as Qwen3-VL) + our MoonViT + our recipe that
  reaches Qwen3-VL quality both validates the pipeline AND is a strong model. Also the natural FIRST target
  of P2/P3 (Qwen3 config is what Qwen3-VL's stack was validated on → diff against Qwen3-VL for correctness),
  then swap EuroLLM config.
- **Honest caveats:** (1) "match Qwen3-VL" is mostly a DATA-SCALE bar, not a recipe bar — frame the goal as
  "competent VLM + validated pipeline", matching is aspirational. (2) **EuroLLM is still the mission**
  (European languages; Qwen3 is EN/ZH-leaning) — Qwen3 variant is parallel, not a replacement. (3) Qwen3
  backbone does NOT give mrope for free — stock GPTModel is 1D; still needs P2/P3. (4) Weights = Qwen3 LLM
  (load, not random) + MoonViT + projector (train); PA→SFT recipe.
- **Realization options:** (A) fastest smoke = compose Qwen3-VL `Qwen3VLModel`, swap vision→MoonViT (couples
  to Qwen vision internals); (B) RECOMMENDED = `QwenEuroVLModel` in euro_vl, our vision + Qwen3-config LLM on
  the shared mrope stack (consistent; needs the same P2/P3).
- **Open decisions:** (a) reuse `modelling_qwen3_vl/*` directly vs vendor a shared mrope-GPT into euro_vl;
  (b) confirm the mrope stack accepts EuroLLM config later (GQA 16/8, head_dim 128, untied, vocab pad);
  (c) when building the class, follow the `adding-model-support` skill.

### Scaffold DONE (2026-06-15): `Qwen3EuroVLModel` + provider
- `models/euro_vl/modeling_euro_vl.py`: **`Qwen3EuroVLModel(EuroVLModel)`** — inherits the MoonViT+
  projector+masked_scatter vision path; overrides `forward` to compute 3D `[3,B,S]` positions via
  `get_rope_index` (per packed sub-seq) and delegate to `super().forward`. Deepstack off.
- `models/euro_vl/euro_vl_provider.py`: **`Qwen3EuroVLModelProvider(EuroVLModelProvider)`** —
  `mrope_section=[24,20,20]`, `position_embedding_type="mrope"`, `apply_rope_fusion=False`; `provide()`
  → `Qwen3EuroVLModel`; `provide_language_model()` builds Qwen3-VL's **`Qwen3VLGPTModel`** (interleaved
  mrope) reusing `get_transformer_block_with_experimental_attention_variant_spec` + `Qwen3VLSelfAttention`.
- Imports/instantiates/ruff clean. **Uncommitted.**
- **Progress (2026-07-22):**
  - ✅ **Qwen3-1.7B downloaded** (`hf_models/Qwen3-1.7B`): 28L, hidden 2048, ffn 6144, 16/8 GQA,
    head_dim 128, vocab 151936, rope_theta 1e6, **tied embeddings**, max_pos 40960, eps 1e-6.
  - ✅ **Tokenizer finding:** Qwen3 ships the vision tokens built-in — vision_start 151652, vision_end
    151653, vision_pad 151654, **image 151655**, **video 151656** (= Qwen3-VL's real ids). **No vocab
    extension needed** (151936 already 128-divisible). ChatML matches EuroLLM's template style.
  - ✅ **HF wrapper generalized:** `modeling_euro_vl_hf.py` builds the LM via `AutoModelForCausalLM
    .from_config(text_config)` (Llama for EuroLLM — unchanged; Qwen3ForCausalLM incl. QK-norm for oracle).
  - ✅ **`EuroVLConfig` composite fix:** text_config dicts resolved via `CONFIG_MAPPING[model_type]`
    (LLaVA convention; was hardcoded LlamaConfig → would break Qwen3 reload); `sub_configs` → AutoConfig.
    Round-trip verified both backbones.
  - ✅ **`Qwen3EuroVLProcessor`** (euro_vl_processor.py): subclass overriding just
    `image_token="<|image_pad|>"`, `video_token="<|video_pad|>"` (start/end already match).
  - ✅ **Assembly script `qwen3_euro_vl_bridge.py`** + assembled **`hf_models/qwen3_euro_vl_2b_hf`**:
    Qwen3 weights → language_model.* (tied lm_head handled), MoonViT → vision_tower.*, projector random,
    tokenizer + vision ChatML template + MoonViT processor config saved. Reload verified end-to-end
    (config class, processor token strings, 2×2-merge image expansion math).
  - ✅ **(3) Bridge extras (2026-07-22):** `EuroVLBridge` now dispatches on `text_config.model_type`
    (forced: both backbones share HF model_type `euro_vl` → one bridge). qwen3 → `Qwen3EuroVLModelProvider`
    (subclass mrope defaults survive; no rope/fusion clobber) + `qk_layernorm=True` + q_norm/k_norm weight
    mappings (mirrors `qwen/qwen3_bridge.py`); llama path byte-identical to before. Verified against BOTH
    real assembled checkpoints: qwen3 → mrope/[24,20,20]/qk_ln/tied/28L/151936; eurollm → unchanged.
  - ✅ **(4) Recipe (2026-07-23):** `qwen3_euro_vl_sft_energon_config` in `recipes/euro_vl/euro_vl.py`
    (+ exports): `_make_qwen3_euro_vl_provider()` (Qwen3-1.7B arch, qk_layernorm, tied, kv_ch 128,
    vocab 151936 no-pad; mrope from provider defaults), `Qwen3EuroVLProcessor`, energon packing + sqrt
    loss + native CE (mirrors euro_vl), **`checkpoint.pretrained_checkpoint=qwen3_euro_vl_2b_hf`**,
    caption-only **`mixture_captions.yaml`** (flickr30k=1.0; independent of mixture.yaml). Config builds ✓.
  - ✅ **(5) Build + weight-load smoke GREEN for BOTH backbones** (`sanity_check/_qwen3_oracle_build_check.py`,
    1 GPU): block spec + `Qwen3VLGPTModel` build fine; bridge conversion completes; weights real (std≈0.017).
    Surfaced + fixed **three LATENT bridge bugs** (the euro_vl pretrained path had NEVER been exercised —
    recipe ships `pretrained_checkpoint` commented; old runs started at loss≈ln(vocab)=random):
    (a) `AutoConfig.register("euro_vl")+AutoModel.register` missing → "Transformers does not recognize
    euro_vl" (repo pattern: stepfun/bailing); (b) vision_tower/projector `AutoMapping` → **`ReplicatedMapping`**
    (AutoMapping can't infer parallelism for plain nn.Linear/Conv2d); (c) backbone dispatch (earlier).
    EuroLLM `euro_vl_2b_hf` pretrained load now works for the FIRST time (verified).
- **READY: (6) launch caption training** — small fully-trainable end-to-end run (user to launch).

### TODO (deferred)
- **KV-cache generation with M-RoPE `rope_deltas`** (2026-08): the diagnostic greedy decode currently
  runs NO KV cache (re-feeds full sequence each step → must pass image every step; Option A). A proper
  cached generation path (prefill once, decode 1 token/step) needs the new token's M-RoPE position =
  `last_pos + 1` *after* the image "jump" — i.e. Qwen's `mrope_position_deltas`/`rope_deltas`, which our
  `get_rope_index` does not yet compute/return. Port that before enabling KV-cache inference. Not needed
  for PA/training (no KV cache there).
- **Log per-component `lm_head_grad_norm`** (user request 2026-07-31): global L2 of the LM-head
  weight's grad, gated behind a config flag (default off). NOTE: Qwen3EuroVL is **tied**
  (`share_embeddings_and_output_weights=True`) → the "head" is the shared vocab embedding, which
  MCore EXCLUDES from the global `grad norm` (`param_is_not_shared`), so this is additive info. Must
  sum shard-norms² across the DP group (distributed optimizer reduce-scatters `.main_grad`); add TP
  group if TP>1. Inject in `train.py train_step` right before `optimizer.step()`; thread into the log.
- **`pretrained_checkpoint` must be Megatron format, not HF** (fixed 2026-07-31): the training loader
  `checkpoint_exists()` only recognizes Megatron markers, so a raw HF dir silently loads nothing →
  random init → loss ≈ ln(vocab) ≈ 12, grad_norm ≈ 88. Fix = convert via `convert_checkpoints.py
  import` then point at the Megatron dir. Recipe now uses `QWEN3_EUROVL_MCORE`.

### Phases
- **P1 (task #27) ✅ DONE** — `models/euro_vl/rope.py::get_rope_index` ported, adapted for THD `[1,total]`
  packing (per-`cu_seqlens` segment, base resets to 0) + plain/padded batch via `attention_mask`. Returns
  `[3,B,S]`. Handles `vision_end` (plain text), `spatial_merge_size=2`, the "position jump", and the
  Qwen3-VL **video grid split** (`repeat_interleave` → per-frame `t=1` → `t_index` always 0; temporal via
  timestamp text). `sanity_check/_mrope_sanity.py` (9 scenarios) green, ruff clean. **The committed `rope.py`
  (`a54599af`) predates the video-split fix — the fix is uncommitted in the working tree.**
- **P2 (task #28)** — vendor `Qwen3VLMultimodalRotaryEmbedding` (interleaved) into `euro_vl/rope.py`, strip
  `Qwen3VLTransformerConfig`; numerical sanity (`_mrope_emb_sanity.py`): channel k driven by expected axis,
  text (t=h=w) reduces to ordinary RoPE.
- **P3 (task #29)** — THE HEAVY PART. Inject interleaved rotary into the decoder. MCore `GPTModel` computes
  rotary internally; native mrope is chunked-only → need custom injection (reuse Qwen3-VL mrope GPT blocks for
  EuroLLM, OR narrow `rotary_pos_emb` injection). Compute `position_ids` in `modeling_euro_vl.forward`; set
  config (`position_embedding_type`, `apply_rope_fusion=False`, `mrope_section`); keep bridge/HF-export consistent.
- **P4 (task #30)** — short packed training run: finite loss, `[3,1,S]` shapes, per-sub-seq reset live,
  throughput delta from losing rope fusion; compare loss curve vs 1D baseline.

### ✅ Video-split fix DONE (2026-06-15, user-caught)
Added Qwen3-VL's `repeat_interleave` split to `get_rope_index` (a `[t,h,w]` video row → `t` per-frame
`[1,h,w]` rows → every frame `t=1` → **`t_index` always 0**; temporal via timestamp text advancing the
base). Sanity `[5]` now asserts split-equivalence (`[[2,4,4]]` == explicit `[[1,4,4],[1,4,4]]`); `[8]`
asserts `t_index=0` + timestamp-base-advance. **Uncommitted in the working tree** (needs commit).

---

## ⏳ Pending (deprioritized after M-RoPE, per user)

- **#22 FA3** (Hopper) for MoonViT attention — `flash_attn_varlen_func` (FA2 now). Vision-attention only,
  partial win. Heavier install than FA4 (CuTeDSL already staged).
- **#23 FA4** (CuTeDSL, supports Hopper) — investigate; only flash-attn-addressable site is the VE
  (`modeling_moonvit.py:81`); LLM attention is TE-owned (FA install can't help it).
- **#24 torch.compile / CUDA graphs** the MoonViT tower — no deps, fuses whole tower (launch-overhead-bound);
  better ROI than FA for the VE.
- **#25 FP8 vision tower via TE** — lower priority; ViT activations outlier-heavy (accuracy risk), modest gain.
- **#26 Wire MoonViT FLOPs into reported TFLOP/s** — reported number is LLM-only today (undercounts MFU).
- **#12 Vectorize `MoonViTVideoProcessor`** — per-frame Python loop will bottleneck video dataloading.
- **#28 Fully-tensor / vectorized video path (unfrozen-vision only)** — the video pipeline has 3 stages:
  **(1) decode** (CPU dataloader), **(2) preprocess/patchify** (CPU dataloader), **(3) vision encode**
  (GPU forward). Stage 3 is *already* batched (all frames' patches in one `vision_tower` call). The
  remaining per-frame Python cost is stage 2 (PIL `preprocess_videos`). Today `_frames_from_video` emits
  **PIL** on purpose: frozen MoonViT was pretrained on PIL-preprocessed images, and `vectorized_preprocess`
  (tensor backend) drifts features ~5% — a frozen encoder can't adapt, so PIL is correct for **PA**. Once
  vision **unfreezes (SFT)**, switch: have `_decode_video_bytes` return the sampled frames as a `[N,C,H,W]`
  tensor batch and feed `MoonViTVideoProcessor.vectorized_preprocess` (ties into #12) — no PIL, all frames
  processed together. `auto_decode=False` already gives us that control (we decode N frames ourselves).

---

## Notes / constraints
- Storage: project moved `jureap126` → `e-ext-2025e01-100`; `~/.cache` symlink + scratch relocation pending
  (separate effort). Energon data at `/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data`.
- `sanity_check/` scripts are gitignored (local only).
- For M-RoPE: user wants **every change/scenario explained** before/as done (confirm-as-we-go).
