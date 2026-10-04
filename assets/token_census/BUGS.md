# EuroVL data & pipeline bugs (running list)

Found by the token census + a sampled data audit (5 random shards x 50 samples per dataset, real
training encoder; the audit script is not in the repo). Newest at the bottom of each section. Status: OPEN unless
marked. Full sample-level details: `_audit/<modality>/issues.jsonl` in the census work directory.

## Fix status — curator, 10-02 ~19:30 (re-check these instead of re-reporting)
Every fix below rewrote shards in place (kept samples byte-copied, metadata regenerated: `.tar.idx`,
index.sqlite, `.info.yaml`), then passed `scripts/energon_validate.py` 20/20 on train+val; the full
mixture resolves 361 datasets with the trainer's own `EuroVLEnergonProvider` resolution.
Scripts/logs: `data_audit/b_issues/` in the curator's work directory.
- B1 `molmo2_multiimageqa_translated` — FIXED (labelled leaks): orphan images removed (15,916); val rebuilt
  from 107 held-out source items, 0 train/val overlap; "A:/Q:" leaks dropped (7,842); translated-label leaks
  (Răspuns/Ответ/Antwort/Antwoord/Vastus/Svaret/...) dropped (2,255 = this file's 2,045 ∪ a wider scan).
  Now 651,687 samples. STILL OPEN: unlabelled merged answers (e.g. `molmo2_multiimageqa_0003843`) — needs the
  translation pipeline (compare each translated user turn with its English source).
- B2 duplicate keys — FIXED: cc12m 2,533 renamed `<key>_dupN` + 17 exact copies dropped (9,797,215);
  molmo2_chart_translated 129 renamed; molmo2_doc_translated 51 renamed. 0 in-shard repeats left.
- B4 — FIXED: caul_wikisql 140 junk turns; viquae 1; rendered_text 31; wkvvqa 18,181 "None"/"-" turns
  (19 all-junk samples dropped; the templated "X when Y is Z" questions are kept on purpose); pmc_vqa 29 bad
  samples (the flagged `train_2_0356889` was a parser artefact: `C:MRI` with no space — kept).
- B4 vqav2 "None." — NOT A BUG, KEPT: 12 random cases checked against the images, all correct false-premise
  answers ("What Disney character is in the picture?" with none present). 1,309 turns were removed and then
  restored from processed-data the same day; 84,751 samples.
- B5 — FIXED: spot_the_diff 4 junk turns; leopard_monkey 11; visualwebinstruct 455 question/answer language
  mismatches dropped (21,937; 44/45 audited flags genuine); molmo2_chart_translated 8,513 answers that only
  restate the question removed (379,258 samples). docmatix "None." (111) — NOT A BUG: checked on the page
  images, the minutes literally say "None" (e.g. "Matters Arising: None").
- B6 temporal segments past the video end — FIXED in all temporal-grounding datasets, measured against the
  VIDEO TRACK duration (not the container): hc_stvg 154 dropped (1,747), hacs 65 (11,641), ava_temporal 3
  branches (1,493 / 33,501), breakfast_actions 38 (1,207). Re-scan: 0 answers >0.5 s past the video end.
- B8 — FIXED: subtitleqa 928 clips and capqa 282 clips re-encoded (H.264, first frame = keyframe, same
  frames/fps/size); videopoint 230 re-encoded + 1 undecodable clip dropped (278,605); the 3 no-keyframe clips
  (youcook2_2 ×2, hacs ×1) had ZERO video packets and were dropped (also the same 2 videos in youcook2_1).
  NOTE: the capqa list `kf_1002/molmo2_capqa_packed.tsv` (13:56) predates the rewrite — e.g.
  `shard-000035.tar/-5dlMyg4xPo_g0` now has its first keyframe at 0.0 s.
- B10 — FIXED: llava_instruct 10 leaked "Question:" answers removed; wordart 58 "?" answers; mv_math
  `mv_math_0001958` dropped (2,008); mp_docvqa 7 NA/None turns.
- D val splits — FIXED (split.yaml only): videogpt_plus_qa / molmo2_capqa / molmo2_askmodelanything val is
  now 1 shard of ~1,000; llava_video_178k_qa kept at 604 (shards are either tiny or ≥1,512). ava/sav
  temporal_grounding folders moved to `video/temporal-grounding/`. spot_the_diff 42% val — OPEN (below).
- A3 (code) — patched, UNCOMMITTED: `video_processing_moonvit.py::decode_video_bytes` falls back to the
  decodable span when the keyframe seek fails (tested: 4 previously failing clips -> 64 frames; normal clips
  identical). A2 media/tag-count scan is running.

### Fix status — curator, 10-03 (re-check these instead of re-reporting)
- `eurovideolm_temporal_merged_packed` split.yaml listed only 471/6,855 shards (48K of 734K samples) — FIXED:
  train 6,834 shards / 732,628, val unchanged 21 / 2,126.
- `fv_unichart` 25,310 `"nan"` answer turns removed (0 samples dropped).
- Exact duplicates set to weight 0 in mixture.yaml (strict check: same question AND same full answer):
  `fv_mmc_instruct` (=code/mmc_instruct), `caul_ai2d` (⊂ ai2d_merged), `chartqa_llava` (⊂ ureader_qa),
  `textcaps` (96% same caption in finevision_ureader_cap). Mixture resolves 356 datasets.
- `bigdocs_cocotext` / `bigdocs_pubtables_1m`: pairs with an EMPTY `<|object_ref_start|><|object_ref_end|>`
  removed (186,668 / 204,898 pairs; 8,859 / 293 samples had nothing else) -> 21,364 / 345,696 samples.
  KNOWN, LEFT AS-IS by user decision: pubtables_1m images often are not tables (source issue).
- `molmo2_multiimageqa_translated`: 119,117 samples with leaked translation prompt / merged next Q&A / stray
  tags dropped (regex R3, ~99.6% precise; `data_audit/b_issues/verify_mmqa/`) -> 532,570. Unlabelled merges
  without any marker (~21K, length-heuristic only) were NOT dropped.
- Grounding markers are now single special tokens: `hf_models/euro_vl_2b_2512_grounding_hf`
  (see docs/models/euro_vl/architecture-trace.md).
- Leading `\n` in assistant answers (Qwen3 merges `assistant\n`+`\n` -> no answer span -> sample skipped):
  stripped (`content.lstrip("\n")`, assistant turns only) in caul_multihiertt 7,830 turns, fv_finqa 6,251,
  chartmimic 3,421, docreason25k_refined 15, text/chat/euroblocks 962. Counts unchanged.
  Re-verified with the CORRECT predicate `re.match(r"[ \t\f\v]*\n", content)` (tested against the real Qwen3
  tokenizer: `' \n'`, `'\t\n'` also break; `' '`, `'\t'`, `'\r\n'`, nbsp do not) -> found 1 (chartmimic) + 101
  (euroblocks chat) more, fixed with `re.sub(r"^\s*\n", "", content)`. Full re-scan of all 5: 0 left.
- Text fixes: euroblocks chat/stem literal `<|im_start|>`/`<|im_end|>` stripped (56/91 edited, 3/25 dropped);
  nemotron_ocr7 `&amp;lt;`->`&lt;` (150); llava_video_178k literal `\n\n`/`\uXXXX` decoded (46,839).
  molmo2_multiimageqa_translated: 21 more residual leaks dropped -> 532,549.
- Video "MPEG-TS" check: full scan of 62 video datasets found NO TS; FLV/WebM members decode in the
  training path (auto_decode=False), so left unchanged.
- KNOWN, NOT FIXED: cc12m holds two curate runs of raw tars 0000-0697 (2.65M duplicate keys, ~27% of rows).

## B. Data bugs

All reported data bugs were fixed or resolved by the dataset owner (verified 10-02/10-03). Remaining notes are informational.


### B7. Very long tracking answers — INFO
- `video/grounding/molmo2_videotrack_packed` (up to 57k chars per branch answer, e.g.
  `shard-000002.tar/molmo2_videotrack_sav_sav_002471`) and `molmopoint_trackany_packed_v2` (30-39k).
  Will be trimmed by the overflow logic at 8K seq_length; decide whether to split tracks.

## Checked and NOT bugs (audit false positives, for reference)
- `ava_*`/`sav_*` "declared vs detected language": Irish (ga) / Maltese (mt) are unknown to langdetect,
  Danish/Norwegian confusion — the declared labels are right.
- `perception_test_1` "MCQ answer not in options": option `(A)` is glued to the question mark
  (`...sign?(A) left`), parser artefact.
- Caption "timestamps past the end" in activitynet_new / molmo2_cap_clips / llava_video_178k: years
  ("early 1980s") or "17-second clip", not timestamps.
- OCR/code/named-entity language mismatches in image datasets; yes/no answer repetition in figureqa/vsr.
- Legit "empty-looking" answers: `cauldron_docvqa` `"Na."` (formula for sodium), `visual_genome`
  `"None."` to "How many people...?" (= zero), `dvqa` "none" (chart label), `molmo2_diagram_single`
  `"None"` (no option satisfies the constraints).
- Shards newer than `index.sqlite` in `videogpt_plus_qa_packed`, `nextsceneqa_genqa`,
  `llava_video_178k_qa_packed`, `zechen_clevrer_multilingual`: only the timestamps changed — every
  checked index row (~16k) still points at the right sample. Re-preparing is optional.

## D. Informational (not bugs, decide if wanted)
- `image/code/webmmu`: the long-context samples (originally 4,088 samples, 246.7M tokens, ~60k tokens/sample,
  87% Chinese) moved to `image/code/webmmu_long`, reserved for the LONG-CONTEXT phase (not in any mixture yet);
  the filtered `webmmu` is back in the mixture.
- Large val shares (never trained) in image: `rendered_text` 4,989 / 9,969 samples (50%), `wordart` 21%, `textocr_grounded` 20%.
- `multiimage/general_qa/spot_the_diff`: 3,543 / 8,514 samples (42%) sit in the val split (never trained), far above other datasets.
- Large val splits (never trained): `videogpt_plus_qa_packed` 269/733 shards, `molmo2_capqa_packed`
  168/427, `llava_video_178k_qa_packed` 153/330, `molmo2_askmodelanything_packed` 44/89.
- `sav_temporal_grounding`, `ava_temporal_grounding` live in `video/grounding/` but mixture.yaml lists
  them under `temporal_grounding` (training resolves by name, so they load).
