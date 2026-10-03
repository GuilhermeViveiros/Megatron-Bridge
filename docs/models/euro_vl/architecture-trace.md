# EuroVL architecture trace

A running record of the facts about EuroVL that are easy to get wrong and expensive to rediscover:
what the model is made of, where each piece's numbers come from, and which invariants must hold.
Every figure here was measured, not inferred; the command that produced it is given so anyone can
re-check after a change.

Newest entries at the bottom of each section.

## Composition

| Piece | What |
|---|---|
| LLM | EuroLLM-1.7B-Instruct-2512, untied embeddings (`share_embeddings_and_output_weights=False`) |
| Vision | MoonViT-SO-400M, per-frame 2D encoding (video is 2D-only; no temporal grouping) |
| Position | Interleaved M-RoPE, `mrope_section=[24,20,20]` summing to `head_dim/2 = 64` |
| Oracle arm | Qwen3EuroVL = Qwen3-1.7B + MoonViT, used to validate our stack against a proven M-RoPE |
| Chat template | `<\|im_start\|>{role}\n{content}<\|im_end\|>\n` |

`position_embedding_type="mrope"` and `apply_rope_fusion=False` are provider defaults. They must not
be overridden on import: a checkpoint whose `run_config.yaml` says `rope` cannot be exported
(`AssertionError: EuroVLModelProvider requires mrope_section`). Fixed in `1047276cc`.

## Vocabulary

The single most error-prone part of this model. Three different numbers are in play at once.

| Number | Value | Meaning |
|---|---|---|
| Base vocab | 128000 | EuroLLM text tokens |
| + vision specials | 128005 | `<image>` 128000, `<\|vision_start\|>` 128001, `<\|vision_end\|>` 128002, `<\|vision_pad\|>` 128003, `<video>` 128004 |
| + grounding markers | 128015 | 10 markers, ids 128005-128014 (below) |
| Megatron padded vocab | **128512** | `EUROLLM_PADDED_VOCAB_SIZE`; pinned so the checkpoint has ONE shape at TP=1/2/4 |

`128512 = 251 x 512`. It is pinned rather than derived because Megatron's usual rule
(`make_vocab_size_divisible_by * TP`) yields a *different* size per TP degree (128128 / 128256 /
128512), which makes a saved checkpoint unloadable at another TP. Rows above the real vocab are
zero-padded on import and trimmed back on export, so the HF side always shows the true vocab size.

Because 128015 <= 128512, adding the grounding markers **does not change any Megatron checkpoint
shape** — existing PA/SFT checkpoints stay loadable.

### Grounding markers (added 2026-10-02)

Before: each marker split into 7-9 text pieces whose split depended on context (`▁<` at a string
start, `<` after text). A labelled box cost 51 tokens. After: one id each.

| id | token | | id | token |
|---|---|---|---|---|
| 128005 | `<\|object_ref_start\|>` | | 128010 | `<\|point_end\|>` |
| 128006 | `<\|object_ref_end\|>` | | 128011 | `<\|quad_start\|>` |
| 128007 | `<\|box_start\|>` | | 128012 | `<\|quad_end\|>` |
| 128008 | `<\|box_end\|>` | | 128013 | `<\|temporal_start\|>` |
| 128009 | `<\|point_start\|>` | | 128014 | `<\|temporal_end\|>` |

The canonical dirs `hf_models/euro_vl_2b_2512_hf` and `hf_models/qwen3_euro_vl_2b_hf` now carry
the markers; the pre-marker tokenizers are kept as `*_pre_grounding_hf`. Qwen3 only needed 4 of
them (`point`/`temporal` start+end at 151669-151672) because it already had object_ref/box/quad
natively at 151646-151651, and its declared 151936 vocab had 267 spare rows, so it needed no resize
and no `vocab_size` change. Build + verify:

```bash
python sanity_check/build_grounding_tokenizer.py --force     # src/out default to the EuroLLM pair
python sanity_check/test_grounding_tokenizer.py --family eurollm   # 64 checks
python sanity_check/test_grounding_tokenizer.py --family qwen3     # 60 checks
```

Measured savings: labelled box 51 -> 26 tokens, point 24 -> 14, temporal 29 -> 19, quad 33 -> 22.
Ordinary text (EU languages, CJK, emoji, numbers, code, existing specials) tokenizes **identically**,
and the existing 128005 embedding rows are bit-identical.

Usage is widespread, so this is not a niche win — first 40 MB of one shard per dataset:
`bigdocs_pubtables_1m` ~20.6k object-refs / ~20.9k boxes, `textocr_grounded` 7.5k/5.4k/2.1k quads,
`nvidia_ocr_synth_enru` 6.1k, `bigdocs_cocotext` 3.7k, `seeclick_ground` 1.9k, `molmopoint_guisyn`
1.8k points, plus whole families (`*_count`, `*_ground`, `*_point`, GUI `*_actions`) and 27
video/multi-image datasets including every temporal-grounding set.

## Where weights and tokenizer come from (they are NOT the same place)

```python
cfg.tokenizer.tokenizer_model        = EUROVL_HF        # HF dir  -> tokenizer + processor
cfg.checkpoint.pretrained_checkpoint = pa_checkpoint    # Megatron -> ALL weights
```

Consequence, verified by reading both checkpoints: rows 128005-128014 of
`language_model.embedding.word_embeddings.weight` are **exactly zero** in
`eurollm_pa_v2_v128512/iter_0003720` and in `megatron_models/euro_vl_2b_2512_v128512/iter_0000000`
(they are the import pad rows). So an embedding initialisation written into the HF dir's
`model.safetensors` is **ignored** by any run that starts from a Megatron checkpoint. To give new
tokens a non-zero start in training, the Megatron checkpoint rows must be patched (same technique as
`sanity_check/pad_vocab_checkpoint.py`).

For reference, embedding row norms: real tokens median 1.61 (1% quantile 1.19), existing vision
specials 0.138, mean-of-sub-pieces init 0.78-0.82.

## Supervision (loss mask)

Derived from the chat-template markers, never by searching for the answer text:
`assistant_answer_spans` / `assistant_answer_mask` in `data/energon/euro_vl_task_encoder.py`.

* An answer is exactly the tokens between `<\|im_start\|>assistant\n` and the next `<\|im_end\|>`.
* `<\|im_end\|>` **is supervised** — it is EuroVL's EOS (`eos_token_id=4`). It is also in the shared
  `QWEN_TOKENS` pad list that `extract_skipped_token_ids` masks, so it has to be exempted explicitly;
  before 2026-10 no run ever trained the model to emit its own stop token. Only the *assistant*
  turn's `<\|im_end\|>` is supervised (46 of 92 in a sample pack), not the user turn's.
* Media must match markers 1:1. Attached media with no `<image>`/`<video>` marker is silently dropped
  by the processor, so the sample would train text-only on a question about an image; this now logs
  and raises `SkipSample`.

Commits: `1047276cc` (marker-based mask + EOS), `76eb96c8a` (media/marker check),
`6ae2663da` (video decode failure -> SkipSample).

## Scale and throughput

Measured on cc12m only (uniform samples; video decode is bimodal and would swamp the signal),
8K sequence, TP=1, 4 GPUs per node, fixed iteration count:

| GBS | 2 nodes | 4 nodes | 8 nodes |
|---|---|---|---|
| 64 | 1.00x (100%) | 1.71x (85%) | 2.67x (67%) |
| 256 | 1.00x (100%) | **2.03x (102%)** | **3.10x (77%)** |

Deeper gradient accumulation fixes the 4-node gap entirely and most of the 8-node one, so a small
global batch — not the code — caused the earlier apparent loss. TFLOP/s per GPU 351 / 287 / 319 at
GBS=256 (the counter is **LLM-only**; the vision tower is not in it). Beyond 8 nodes is unmeasured.

## Data loader: the energon worker-split bug (found 2026-10-03)

energon cuts a split into one slice per worker, where the worker count is
`world_size x dataset.num_workers` (4 GPUs/node x 8 workers = 32 slices/node). A split with
fewer samples than that is legal — the trailing workers simply get nothing — but **energon
6.0.1, the version in the container, crashes instead**:

```python
while True:                                        # flavors/webdataset/sharder.py:64
    if end_offset <= shard_cumsums[end_index + 1]: # reads one past the array
```

When a rank's whole slice starts at the end of the data, `np.searchsorted` returns the last
valid index and the next read is out of bounds:

```
IndexError: index 2 is out of bounds for axis 0 with size 2
```

Worked example — `mi_taco` val, 10 samples in 1 shard (`shard_cumsums = [0, 10]`, 2 entries):

| | rank 15's worker offsets | outcome |
|---|---|---|
| 1 node (32 slices) | no rank starts at 10 | fine |
| 4 nodes (128 slices) | `[10, 10, 10, 10, 10, 10, 10, 10, 10]` | asks for `shard_cumsums[2]` -> IndexError |

So it is **scale-dependent**: the same dataset loads on few ranks and fails once enough ranks
are added. One rank raises while the rest block in a collective until the NCCL timeout, which
buries the real error. Measured against the live mixture (`num_workers=8`):

| nodes | ranks | slices | val splits that break | train splits that break |
|---|---|---|---|---|
| 1 | 4 | 32 | 0 | 0 |
| 4 | 16 | 128 | 1 | 0 |
| 8 | 32 | 256 | 5 | 0 |
| 16 | 64 | 512 | 13 | 1 — `handwritten_math_expr` (40) |
| 32 | 128 | 1024 | 31 | 2 — + `mi_taco` (95) |
| 64 | 256 | 2048 | 65 | 2 |

Note the train column: **skipping validation does not fix this.** At >=16 nodes training itself
breaks. Having fewer samples than slices is necessary but not sufficient — the bit-reversal
permutation spares most such datasets, so the counts must be measured, not inferred (192 val
splits sit below 2048 slices at 64 nodes but only 65 actually break).

**Fix:** `data/energon/sharder_compat.py` installs upstream 7.x's bounded loop — the only
behavioural difference between 6.0.1 and 7.3.2 in the splitting path; everything else 7.x added
is the `subset` feature, a no-op when `subset is None`. It is gated on *feature detection* (run
the failing case), not a version string, so it becomes a no-op once the container ships
energon >= 7. Applied once at import of `base_energon_datamodule`, the single chokepoint every
energon provider goes through.

Verified, since a silent mis-split would be far worse than a crash — the per-worker ranges must
tile `[0, total)` with no gap, overlap or loss:

| configuration | cases | raised | partition violations |
|---|---|---|---|
| 6.0.1 unpatched | 2792 real + 504 synthetic | 82 / 201 | **0** |
| 6.0.1 + backport | same | 0 | **0** |
| 7.3.2 native | same | 0 | **0** |

Patched 6.0.1 is byte-identical to native 7.3.2 on all 3296 cases, and identical to unpatched
6.0.1 on every case the old code already handled (0 regressions; 283 crashes fixed). Unit tests:
`tests/unit_tests/data/energon/test_sharder_compat.py` (77 tests).

## Invariants worth asserting against

1. `EUROLLM_PADDED_VOCAB_SIZE` must stay >= the real vocab and divisible by every TP degree used.
2. A probe must never share a recipe's run dir: `euro_vl_2b_sft_config` sets
   `checkpoint.load = checkpoint.save`, so a short run with fewer `train_iters` than the stored
   checkpoint resumes, trains nothing and still exits 0.
3. The LR schedule must be expressed relatively (`lr_decay_iters=None` + `lr_warmup_fraction`).
   Baking absolute counts from the recipe's default `train_iters` means a CLI override does not
   recompute them — a 10000-iteration launch would finish its cosine by iteration 500 and spend 95%
   of the run pinned at `min_lr`.
4. Cross-TP *resume* (weights + optimizer) needs `checkpoint.dist_ckpt_optim_fully_reshardable=true`
   set on the saving run; the error message naming `fully_parallel_save` is wrong. Weights-only
   loads (`pretrained_checkpoint`) reshard across TP freely.
5. A data split smaller than `world_size * dataset.num_workers` must still load. energon < 7
   raises `IndexError` in `Sharder._split_shards` instead; `data/energon/sharder_compat.py`
   backports the upstream guard. Verify a loader change by checking the per-worker ranges tile
   `[0, total)` exactly — a wrong split is silent, a crash is not.
