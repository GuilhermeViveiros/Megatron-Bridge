# EuroVL: message-tree packing for multi-annotation video data

Design notes for the technical report. Records what we decided, what we measured, and which
alternatives we rejected and why. Status: data format decided and piloted; model/cooker
implementation in progress.

## Problem

Many EuroVL video datasets attach several independent annotations to one clip: ~7–8 QA pairs per
video on average for the `general_qa` sources, and one caption per language for the multilingual
captioning sources. Stored and trained the naive way, each annotation becomes its own sample:

- **Storage.** The repack step wrote one full copy of the video per `(clip, language)` pair. For
  the multilingual EuroVideoLM captioning set this reached 14 TB at 83% of source shards and hit
  the `/e/scratch` quota.
- **Compute.** Each copy is decoded and pushed through MoonViT separately, so a clip with N
  annotations pays N vision forward passes for identical frames.

Video dominates the sequence. MoonViT encodes each frame independently (no temporal merge), and
the video budget is sized to ~85% of `seq_length`. Measured on the pilot (640×360 clips, 64
frames): **5,824 `<video>` placeholder tokens + ~560 per-frame overhead tokens ≈ 6,390 tokens**
of an 8,192-token sequence, leaving ~1,800 tokens for text.

## Approach

Following Molmo2's message-tree encoding: the video is encoded once as a shared prefix, and each
annotation becomes an independent *branch* after it. A subsegment attention mask stops branches
from attending to each other; every branch sees the prefix and itself.

```
[ <|im_start|>user <video: ~6,390 tokens> ]    shared    -> visible to all branches
[ question 1 ... assistant answer 1       ]    branch 1  -> sees shared + branch 1
[ question 2 ... assistant answer 2       ]    branch 2  -> sees shared + branch 2
  ...
```

Loss is computed only on assistant answers. The remainder of the sequence is filled by the
existing first-fit-decreasing packer with unrelated samples.

### Example: one 640×360 clip with 20 QA pairs (~70 tokens each)

| | Flat (one sample per QA) | Message-tree |
|---|---|---|
| Video copies on disk | 20 | 1 per group (here 1) |
| Video decodes + MoonViT passes per epoch | 20 | 1 |
| Video tokens through the LLM per epoch | 20 × 6,390 = 127,800 | 6,390 |
| QA pairs trained per epoch | 20 | 20 |

20 × 70 = 1,400 text tokens fit in the ~1,800 left after the video, so the clip is one group.
The cost of the video depends on resolution and duration (the pilot's largest clip used 6,912
placeholders, ~7,470 tokens with overhead, leaving ~720), so the number of groups is decided per
clip.

## Data format

Two stages.

**1. `processed-data` (source of truth, one per clip):** one video file and one JSON holding every
annotation in an `annotations` list. Nothing model-specific (no chat template, no media tokens).

**2. Energon (training input, one per group):** the conversion step bin-packs a clip's annotations
into the minimum number of groups that fit `seq_length`, balanced across groups, and writes one
ordinary energon sample per group. The video is copied into each group.

```json
{
  "message_tree": true,
  "shared":   [{"role": "user", "content": "<video>"}],
  "branches": [
    [{"role": "user", "content": "…question / prompt…"}, {"role": "assistant", "content": "…"}],
    [{"role": "user", "content": "…"},                   {"role": "assistant", "content": "…"}]
  ],
  "meta": {"clip_id": "…", "group": 0, "n_groups": 1, "duration_s": 179.3,
           "video_tokens": 5824, "languages": ["fr"]}
}
```

- The media placeholder appears only in `shared`; each branch is its own ChatML list.
- Rendering (e.g. multiple-choice options into text) happens at conversion, so the cooker stays
  generic across QA, MC and captioning.
- The cooker dispatches on the presence of `branches`; datasets without it use the existing flat
  ChatML path unchanged.
- Files must be flat and share a base name (`KEY.mp4` + `KEY.json`, adjacent in the tar). Energon
  derives the sample key with `^((?:.*/|)[^.]+)[.]([^/]*)$`, which keeps directories, so
  `videos/KEY.mp4` + `annotations/KEY.mp4.json` would become two separate, broken samples.
- Group budgeting must use the real rendered length (placeholders + per-frame overhead + chat
  template), measured with the actual processor. Using only `video_tokens` underestimates by ~560
  tokens for a 64-frame clip; on the pilot this let 5 of 104 groups exceed 8,192 (max 8,545).

## Design decisions

**Group offline, not at cook time.** Energon's `Cooker` is strictly one-to-one
(`cook: Callable[[dict], Sample]`), exposes no epoch or visit counter to the cooker (the loader's
`_epoch_count` is internal), and `RepeatDataset` re-yields identical samples for every repeat. So
a cooker cannot deterministically rotate through a clip's annotations across visits; random
subsets per visit would give no coverage guarantee. Grouping during conversion gives exact
coverage: every annotation is in exactly one group, every group is seen once per epoch.

**Copy the video per group rather than reference one copy.** True single-copy storage is not
reachable inside energon's format: `prepare.py` skips non-regular tar members
(`if not member.isreg(): continue`), so hardlinks are dropped, and joined datasets match on an
exact, unique `sample_key`, so several groups cannot join one video. The only single-copy option
is reading the video by path inside the cooker, outside energon's format and tooling. With
~7–8 annotations per clip most clips are a single group, so copying per group costs little extra
storage over single-copy and keeps the data energon-native. Revisit if copy-per-group storage
reaches tens of TB.

**Fill the text budget; do not cap the number of branches.** The vision cost is paid once per
group regardless of branch count, so unused budget is wasted amortisation.

**Accept lower per-sequence visual diversity.** A message-tree sequence carries one video; the
flat layout would spread the same tokens over several videos. This is structural to the
technique. With video at ~10–15% of the mixture, ~100–150 of the 1,024 sequences in a global
batch are message-tree sequences, and their correlation is diluted by the rest. Raising the
global batch size does not change that ratio.

**Scope.** Applies to any dataset whose annotations are independent questions about the same
media, regardless of count. Datasets with dependent turns (real dialogue) stay flat. Datasets
whose branches are translations of the same caption are trained with message-tree only once the
branch mask exists; without it a branch can copy the preceding translation instead of describing
the video.

## Pilot (2026-09-19)

Two multilingual captioning shards (`bg_new_part_0000`, `fr_part_0097`), 263 clips.

- Schema correct on every sample; `<video>` only in `shared`; annotations fully covered.
- 168 samples with 1 branch, 95 with 2; every clip fits in one group (`n_groups = 1`), so group
  splitting is not exercised by this pilot.
- `meta.video_tokens` matches the processor's placeholder count exactly; the real video cost is
  a further 556–560 tokens.
- Issues found and sent to the data pipeline (fix pending re-check): directory-prefixed file
  names and `.mp4.json` extension (not loadable by energon); budgeting without per-frame overhead
  (5/104 over length on `bg`).

## Video token budget

- Frames: `fps=1.0`, bounded to `[2, 64]` (`EuroVLProcessor.from_pretrained` defaults). A ~3 min
  clip is capped at 64 frames, one every ~2.7 s.
- Pixels: `total_pixels = 0.85 × seq_length × 28²` is split across the sampled frames, so more
  frames means fewer pixels per frame. On 640×360 clips, 64 frames get ~85K px each → 91 tokens
  per frame → 5,824 placeholder tokens. MoonViT has no temporal merge, so temporal coverage and
  per-frame detail trade off directly.
- Timestamps: each sampled frame is preceded by its time in the clip as text (`<2.7 seconds>`,
  Qwen3-VL style) so the model can place frames in time. Recipes use the deterministic
  `"seconds"` format; the processor also supports `"hms"` and a random mix, which change the
  token count per frame.
- Per-frame markers (timestamp, vision start/end) add ~560 tokens for 64 frames; together with
  the chat template, the real overhead beyond the placeholders ranged 239–2,324 tokens on the pilot.
- Conversion budget: groups are measured as the full rendered sequence (shared turn followed by
  every branch's turns) and capped at **8,100** tokens rather than 8,192, leaving margin for
  tokenizer/template drift and for changing the timestamp format. On the regenerated pilot the
  conversion's `meta.rendered_tokens` equals the encoder's real length exactly (24/24 checked);
  the pilot predates the 8,100 cap, so a few samples reach 8,151.

## Cooker implementation (phase 1: data path, no model changes)

`_cook` is unchanged: it decodes `KEY.mp4` and passes the JSON through. `encode_sample`
dispatches on `"message_tree": true`; all other samples take the existing flat path.

1. **Turns.** Build `shared` followed by every branch's turns directly. The flat path's
   `cook_chatml_sample` treats any odd-length conversation as having a leading system prompt
   (without checking roles) and would relabel the video turn as `system`; a tree always has
   `1 + 2 × branches` turns.
2. **Encode once.** Apply the chat template and call the processor once: one video decode, one
   expansion of `<video>` into per-frame blocks.
3. **Branch boundaries** from the positions of `<|im_start|>`, which starts every turn and never
   appears inside the video expansion. The count must equal
   `len(shared) + Σ len(branch)`; otherwise raise an error (a data bug, not something to guess
   around).
4. **`subsegment_ids`:** shared prefix = `10000` (attend-all), branch *b* = *b* (0-based), as in
   Molmo2, whose mask is `causal AND ids[q] <= ids[k]`. Carried through packing for the phase-2
   mask; flat samples are attend-all everywhere, so they behave as before.
5. **Loss** on assistant spans only, with the same search function as the flat path. A branch
   whose answer is not found raises an error (the flat search only warns when *every* span fails,
   which would let a single lost branch go unnoticed).
6. **Square-root loss weighting per branch** (`1/√N_b`, when `sqrt_loss_weighting` is on), so
   each branch weighs what it would as a separate flat sample. Per-sample weighting would shrink a
   2-branch sample by ~30% (e.g. captions of 400 + 300 tokens: √400 + √300 = 37.3 vs
   700/√700 = 26.5).
7. **Optional divide by √B** (`root_subsegments`, **off by default**; B = kept branches). Off, a
   group weighs exactly what its branches would weigh as separate flat samples, so grouping is a
   pure compute optimization and cannot change training — the property we want while converting
   existing flat datasets. On, it is Molmo2's `root_subsegments`: combined with step 6 the group
   weighs `Σ_b √N_b / √B`, which for equal-length branches is exactly `√(Σ_b N_b)`, i.e. the group
   is weighted like *one* sample of the combined answer length. That assumes annotations of one
   video are partly redundant (between counting them once, 1/B, and counting them as independent
   samples, 1). It costs video datasets ~√B of weight against the flat baseline, so the mixture
   repeat factors must be revisited before enabling it.
8. **Over length (safety net):** cut at the last branch boundary that fits. Branches follow the
   video, so a cut never touches it and needs no re-encode. B in step 7 counts kept branches
   only. Skip only if no branch fits.

Worked example — clip with 8 QAs of 25 answer tokens, `sqrt_loss_weighting` on:

| | weight/token | per branch | clip total |
|---|---|---|---|
| the 8 QAs as flat samples (today) | 0.2 | 5 | 40 |
| tree, `root_subsegments=False` (default) | 0.2 | 5 | 40 (identical) |
| tree, `root_subsegments=True` | 0.0707 | 1.77 | 14.1 = √200 |

Both weightings are heuristics with no published ablation (Molmo2's paper does not mention √B at
all), so the plan is to measure per-dataset answer-token shares first and treat √B as a separate
experiment.

Position ids still run continuously through all branches. Molmo2 restarts them for every branch
after the shared prefix (`build_subsegment_pos_ids`), so each branch sees the positions it would
have alone; this goes with the phase-2 mask (EuroVL's M-RoPE ids are computed in the model, so the
reset has to be done there).

Until the phase-2 mask exists, branches in one sequence can attend to each other, i.e. they
train like an ordinary multi-turn conversation. That is acceptable for independent QA; datasets
whose branches are translations of one caption should wait for the mask.

## References

Molmo2 (`allenai/molmo2`, commit f3cb108):

- `olmo/preprocessing/text_preprocessor.py`: `tokenize_message_list` (branch layout, ÷√B at the
  end), `build_subsegment_pos_ids` (position reset), `ATTEND_ALL`.
- `olmo/models/molmo2/molmo2.py`: the subsegment attention mask.
- `olmo/data/dynamic_packer.py`, `olmo/preprocessing/multimodal_collator.py`: packing and
  collation of `subsegment_ids`.
- `launch_scripts/sft.py`: `"root_subsegments_root_tokens"` loss weighting.

## Status

- [x] Storage layout and message-tree JSON format.
- [x] Pilot conversion; issues above reported to the data pipeline.
- [x] Encoder support for `message_tree` samples (`EuroVLTaskEncoder._encode_message_tree`):
      decode the video once, render shared + branches, loss on assistant spans only, per-branch
      `subsegment_ids` carried through packing to the batch. Verified on 24 real pilot samples:
      video tokens appear once and lie in the shared prefix; every branch's supervised tokens
      decode to exactly its answer; lengths equal `meta.rendered_tokens`.
- [x] Over-length handling: drop branches rather than the whole sample; skip only if nothing fits.
- [ ] Subsegment attention mask and per-branch position reset. Open risk: Transformer Engine's `arbitrary` mask combined with
      THD `cu_seqlens` runs only on the unfused backend (O(T²)); verify on the pinned TE version.
- [ ] Measurements for the report: throughput and vision-tower time, flat vs message-tree.
