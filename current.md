# EuroVL — Current Status & Decisions

_Last updated: 2026-06-15. Branch: `feat/euro-vl` (pushed to `fork`, HEAD `9485716b`)._

EuroVL = EuroLLM-1.7B (24L, hidden 2048, ffn 5632, 16 heads / 8 KV, head_dim 128,
rope_theta **1e6**, 32K ctx, untied embeddings, vocab 128000+5 vision = 128005) +
MoonViT-SO-400M vision tower (per-frame 2D, 2×2 spatial merge). Hardware: 4×GH200 (Hopper).
Run in-container: `apptainer exec --nv … megatron-bridge.sif uv run --no-sync …`.

---

## ✅ Done recently

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

### Phases
- **P1 (task #27) ✅ DONE** — `models/euro_vl/rope.py::get_rope_index` ported, adapted for THD `[1,total]`
  packing (per-`cu_seqlens` segment, base resets to 0) + plain/padded batch via `attention_mask`. Returns
  `[3,B,S]`. Handles `vision_end` (plain text), `spatial_merge_size=2`, the "position jump", timestamp video.
  `sanity_check/_mrope_sanity.py` (9 scenarios) green, ruff clean. **`rope.py` is uncommitted (new tracked file).**
- **P2 (task #28)** — vendor `Qwen3VLMultimodalRotaryEmbedding` (interleaved) into `euro_vl/rope.py`, strip
  `Qwen3VLTransformerConfig`; numerical sanity (`_mrope_emb_sanity.py`): channel k driven by expected axis,
  text (t=h=w) reduces to ordinary RoPE.
- **P3 (task #29)** — THE HEAVY PART. Inject interleaved rotary into the decoder. MCore `GPTModel` computes
  rotary internally; native mrope is chunked-only → need custom injection (reuse Qwen3-VL mrope GPT blocks for
  EuroLLM, OR narrow `rotary_pos_emb` injection). Compute `position_ids` in `modeling_euro_vl.forward`; set
  config (`position_embedding_type`, `apply_rope_fusion=False`, `mrope_section`); keep bridge/HF-export consistent.
- **P4 (task #30)** — short packed training run: finite loss, `[3,1,S]` shapes, per-sub-seq reset live,
  throughput delta from losing rope fusion; compare loss curve vs 1D baseline.

### ⚠️ OPEN FIX before P2 (user-caught, agreed to do)
`get_rope_index` is **missing the video grid split** (Qwen3-VL `rope.py:196-198`:
`video_grid_thw = repeat_interleave(video_grid_thw, t); t=1`). Add it so EuroVL faithfully matches Qwen3-VL
(`t_index` always 0; data contract = one `[t,h,w]` row per video, split internally to per-frame `[1,h,w]`
rows aligning with per-frame token runs). Then fix sanity labels: **[5]** = Qwen2-VL-style t>1 (NOT EuroVL —
keep as code-path coverage or drop); **[8]** = feed per-video `[[2,4,4]]` to exercise the split, assert
`t_index=0` + temporal via timestamp-text base advance.

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

---

## Notes / constraints
- Storage: project moved `jureap126` → `e-ext-2025e01-100`; `~/.cache` symlink + scratch relocation pending
  (separate effort). Energon data at `/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data`.
- `sanity_check/` scripts are gitignored (local only).
- For M-RoPE: user wants **every change/scenario explained** before/as done (confirm-as-we-go).
