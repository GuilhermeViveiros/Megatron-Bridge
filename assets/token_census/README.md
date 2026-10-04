# EuroVL data mixture — exact token census

Exact (not sampled) text+vision token counts for **every dataset in `mixture.yaml`**. Two totals:
**r=1** counts every mixture entry once (repeat factors ignored, including `r=0` entries), and
**effective** = Σ r × tokens, i.e. what one pass over the mixture feeds the model (`r=0` entries
contribute nothing, `r=2` counts twice). Every sample of every shard is counted by
`assets/token_census/scripts/census/estimate_token_budget.py`; nothing is extrapolated.

**How the counts are made and checked**

- **Tokenizer:** `euro_vl_2b_2512_hf`, updated 10-03 with the 10 grounding markers
  (`<|object_ref_start|>`…`<|temporal_end|>`, incl. box/point/quad) as single special tokens — each
  grounding block (ref + box/point/span) is 26 tokens shorter than with the previous vocabulary. (An earlier census
  used `euro_vl_2b_hf`, whose `tokenizer_config.json` had lost the `legacy` flag and split role headers
  like `assistant` into `ass`+`istant`. Fixing it moves text by ~+1 token per turn: +0.06% to +0.43% of
  totals on a shard-for-shard spot-check; vision tokens are unaffected.)
- **Validated against the real training encoder:** 132 samples across every format (image,
  multi-image, flat video, message-tree video, temporal grounding, text), plus 100 more video samples
  spread over 5 shards each of `activitynet_new` (flat) and `eurovideolm_packed` (message tree) — the
  census count equals `EuroVLTaskEncoder` `input_ids` length exactly, total and vision, 0 mismatches.
  Re-run 10-03 with the updated tokenizer on 164 samples, adding grounding / pointing / temporal data
  (refcoco_ground, refcoco_point, mi_o365, sav_grounding, sav_tracking, ava_temporal_grounding): 0 mismatches. Message-tree
  totals also match the curation side's own `meta.rendered_tokens` exactly on every dataset that
  carries it (`ego4d_narrations`: curation estimate is 0.3% below the real count).
  Re-run 10-04 with answer (loss-masked) tokens added, on 254 samples of 34 datasets incl. code and math
  (`scripts/validation/check_against_encoder.py`): total, vision and answer tokens (= positions with
  `loss_mask > 0`) all equal, 0 mismatches.
- **Vision** is computed from image/video headers only (no pixel decode) with the processor's own
  resize math; **video** counts assume the default `seq_length=8192` video budget
  (`total_pixels = 0.85 × 8192 × 28²`; a 16K run gets a larger budget and more tokens per video).
- **Message-tree samples** (`{"message_tree": true, "shared", "branches"}`) are counted the way training
  renders them: shared turns + every branch in one sequence, at full length.
- **Pre-filter counts:** no `seq_length` truncation/overflow skipping and no `max_num_images` skip —
  this is what the data contains, not what survives the training-time filters.
- **Which shards:** exactly the ones energon reads — the `split.yaml` parts of each dataset (so
  e.g. `train-*.tar` names count too). **All splits are counted** (train + val); the val share is
  shown separately (1.72B tokens in total). A mixture entry is resolved by dataset
  name under any category folder, like training does.
- **Samples are grouped like webdataset/energon** (consecutive tar members sharing a key), so a key
  that repeats inside one shard counts as separate samples, as training reads them.
- **Text-only** samples have any literal `<image>`/`<video>` placeholder neutralized before tokenizing
  (it would otherwise tokenize as a real vision slot).

## Overall total

**360 datasets, 116,613,829 samples, 156.16B tokens (r=1); 156.16B effective (r×tokens).**

| modality | datasets | samples | tokens (r=1) | % | effective (r×) | % |
|---|---|---|---|---|---|---|
| image | 248 | 90,862,914 | 82.90B | 53.1% | 82.90B | 53.1% |
| multiimage | 46 | 9,725,929 | 29.87B | 19.1% | 29.87B | 19.1% |
| video | 62 | 5,376,790 | 32.92B | 21.1% | 32.92B | 21.1% |
| text | 4 | 10,648,196 | 10.46B | 6.7% | 10.46B | 6.7% |

| category | tokens (r=1) | effective (r×) |
|---|---|---|
| code | 30.02B | 30.02B |
| general_qa | 23.99B | 23.99B |
| captioning | 22.80B | 22.80B |
| ocr | 13.72B | 13.72B |
| chart | 11.71B | 11.71B |
| grounding | 9.60B | 9.60B |
| doc | 8.47B | 8.47B |
| temporal_grounding | 7.06B | 7.06B |
| pointing | 4.82B | 4.82B |
| knowledge | 4.65B | 4.65B |
| counting | 4.19B | 4.19B |
| chat | 3.68B | 3.68B |
| science | 2.82B | 2.82B |
| gui | 2.60B | 2.60B |
| math | 2.34B | 2.34B |
| stem | 1.53B | 1.53B |
| subtitles | 695.85M | 695.85M |
| spatial | 499.20M | 499.20M |
| medical | 357.76M | 357.76M |
| embodied_reasoning | 301.27M | 301.27M |
| 3d_grounding | 297.21M | 297.21M |
| **TOTAL** | **156.16B** | **156.16B** |

## Languages

**77.42% English, 22.49% other languages, 0.09% undetermined** (of 156.16B tokens). Non-English EU official languages alone: **28.16B (18.03%)**.

**Effective (r×tokens):** 77.42% English, 22.49% other languages, 0.09% undetermined (of 156.16B tokens); non-English EU official languages 28.16B (18.03%).

Samples with too little natural language for detection (labels + coordinates, GUI actions, table cells, LaTeX, code) come out undetermined; their questions are English, so in every dataset that is **not multilingual** they are counted as **English** (8.35B). Undetermined tokens left in multilingual datasets stay undetermined.

How each dataset's tokens are attributed to languages (first source that covers it):

1. **prorated** (28 datasets) — `smurf4eu-vision-curator/data_sync/language_counts.md` (curation-side per-language *sample* counts, from source metadata where present). `tokens(lang) ≈ total_tokens × samples(lang) / samples`: exact for samples, approximate for tokens (assumes equal tokens/sample across languages — tight where vision tokens dominate, weaker for caption-heavy sets like `wit`, `culturalground_*`).
2. **exact** (332 datasets: all video + datasets newer than the curation file) — the census attributes each sample's real token count (each *branch* for message trees; shared video tokens split by branch text share) to a language: the sample's own `meta.language` when declared, else langdetect restricted to confident calls (≥20 letters, ≥8 distinct words, p ≥ 0.90; assistant text, then user text, then both), else `und`. Stricter than the curation rule on purpose: answers that are coordinates, JSON, timestamps, option letters or 2-word labels were confidently misdetected (e.g. `tapos` 72% "da", MCQ letters "hu"). Residual known error is <1% of video tokens (`crosstask` step labels ~9% "fr"; English captions quoting Chinese signs → `zh`). Detection agrees with declared labels where both exist (`eurovideolm`: 51/49 detected vs 52/48 declared bg/en).
3. **assumed-en** (0 datasets) — not flagged as multilingual by curation.

`und` = no confident language (structured/numeric answers), not necessarily non-English.

| language | tokens | % of mixture | EU official | EuroLLM | source(s) |
|---|---|---|---|---|---|
| en | 120.90B | 77.421% | ✓ | ✓ | exact, prorated |
| fr | 7.20B | 4.609% | ✓ | ✓ | exact, prorated |
| zh | 3.23B | 2.065% |  | ✓ | exact, prorated |
| de | 3.15B | 2.018% | ✓ | ✓ | exact, prorated |
| es | 2.80B | 1.790% | ✓ | ✓ | exact, prorated |
| pt | 2.45B | 1.569% | ✓ | ✓ | exact, prorated |
| it | 1.94B | 1.242% | ✓ | ✓ | exact, prorated |
| nl | 1.68B | 1.076% | ✓ | ✓ | exact, prorated |
| pl | 1.17B | 0.751% | ✓ | ✓ | exact, prorated |
| ru | 940.18M | 0.602% |  | ✓ | exact, prorated |
| cs | 903.08M | 0.578% | ✓ | ✓ | exact, prorated |
| ro | 683.39M | 0.438% | ✓ | ✓ | exact, prorated |
| uk | 631.77M | 0.405% |  | ✓ | exact, prorated |
| sv | 627.51M | 0.402% | ✓ | ✓ | exact, prorated |
| bg | 594.48M | 0.381% | ✓ | ✓ | exact, prorated |
| hu | 587.21M | 0.376% | ✓ | ✓ | exact, prorated |
| fi | 509.37M | 0.326% | ✓ | ✓ | exact, prorated |
| ja | 499.45M | 0.320% |  | ✓ | exact, prorated |
| et | 496.72M | 0.318% | ✓ | ✓ | exact, prorated |
| el | 470.32M | 0.301% | ✓ | ✓ | exact, prorated |
| hr | 452.23M | 0.290% | ✓ | ✓ | exact, prorated |
| da | 444.09M | 0.284% | ✓ | ✓ | exact, prorated |
| sl | 443.14M | 0.284% | ✓ | ✓ | exact, prorated |
| lt | 434.06M | 0.278% | ✓ | ✓ | exact, prorated |
| sk | 415.65M | 0.266% | ✓ | ✓ | exact, prorated |
| lv | 412.61M | 0.264% | ✓ | ✓ | exact, prorated |
| mk | 324.49M | 0.208% |  | ⚠️ | exact, prorated |
| sq | 320.63M | 0.205% |  | ⚠️ | exact, prorated |
| cy | 253.29M | 0.162% |  | ⚠️ | exact, prorated |
| ko | 206.32M | 0.132% |  | ✓ | exact, prorated |
| ga | 205.60M | 0.132% | ✓ | ✓ | prorated |
| ca | 188.25M | 0.121% |  | ✓ | exact, prorated |
| und | 136.00M | 0.087% |  | ⚠️ | exact, prorated |
| no | 102.42M | 0.066% |  | ✓ | exact, prorated |
| mt | 94.15M | 0.060% | ✓ | ✓ | prorated |
| id | 88.93M | 0.057% |  | ⚠️ | exact, prorated |
| tr | 45.49M | 0.029% |  | ✓ | exact, prorated |
| gl | 21.89M | 0.014% |  | ✓ | prorated |
| vi | 17.27M | 0.011% |  | ⚠️ | exact, prorated |
| af | 11.41M | 0.007% |  | ⚠️ | exact, prorated |
| sw | 10.55M | 0.007% |  | ⚠️ | exact, prorated |
| pa | 9.60M | 0.006% |  | ⚠️ | exact, prorated |
| ta | 9.47M | 0.006% |  | ⚠️ | exact, prorated |
| tl | 8.82M | 0.006% |  | ⚠️ | exact, prorated |
| hi | 8.01M | 0.005% |  | ✓ | exact, prorated |
| te | 7.36M | 0.005% |  | ⚠️ | exact, prorated |
| fa | 6.99M | 0.004% |  | ⚠️ | exact, prorated |
| ar | 5.71M | 0.004% |  | ✓ | exact, prorated |
| so | 5.52M | 0.004% |  | ⚠️ | exact, prorated |
| gu | 3.44M | 0.002% |  | ⚠️ | exact, prorated |
| th | 2.72M | 0.002% |  | ⚠️ | exact, prorated |
| bn | 1.68M | 0.001% |  | ⚠️ | exact, prorated |
| ml | 1.50M | 0.001% |  | ⚠️ | exact, prorated |
| he | 849,045 | 0.001% |  | ⚠️ | exact, prorated |
| ne | 718,667 | 0.000% |  | ⚠️ | exact, prorated |
| ur | 635,498 | 0.000% |  | ⚠️ | exact, prorated |
| mr | 398,037 | 0.000% |  | ⚠️ | exact, prorated |
| sr | 359,405 | 0.000% |  | ⚠️ | prorated |
| kn | 285,310 | 0.000% |  | ⚠️ | exact, prorated |

Datasets with non-English content, by multilingual token volume:

| dataset | modality | tokens | source | en % | other % | und % | top languages |
|---|---|---|---|---|---|---|---|
| temporal_grounding/eurovideolm_temporal_merged_packed | video | 5.56B | exact | 41.4 | 58.6 | 0.0 | fr 32%, es 7%, de 7%, pt 6% |
| captioning/multi_synth_cc12m_cc3m_grit | multiimage | 3.12B | prorated | 0.0 | 100.0 | 0.0 | de 12%, fr 11%, es 9%, pt 8%, nl 8% |
| general_qa/multi_synth_cc12m_cc3m_grit | multiimage | 2.97B | prorated | 0.0 | 100.0 | 0.0 | de 12%, fr 10%, es 9%, pt 8%, nl 8% |
| general_qa/eurovideolm_qa_packed | video | 5.29B | exact | 44.3 | 55.7 | 0.0 | fr 32%, pt 8%, es 6%, de 5% |
| doc/molmo2_doc_translated | multiimage | 2.99B | prorated | 6.6 | 93.4 | 0.0 | de 12%, fr 11%, es 9%, it 7% |
| captioning/eurovideolm_packed | video | 4.84B | exact | 44.5 | 55.5 | 0.0 | fr 33%, pt 7%, es 6%, de 5% |
| science/leopard_arxiv_enriched_translated | image | 2.58B | prorated | 0.0 | 100.0 | 0.0 | hr 5%, sl 5%, et 5%, sq 5%, lv 5% |
| knowledge/wit | image | 2.68B | prorated | 18.6 | 81.4 | 0.0 | de 11%, fr 9%, es 7%, it 6% |
| chart/molmo2_chart_translated | multiimage | 2.18B | prorated | 2.1 | 97.9 | 0.0 | de 12%, fr 11%, es 9%, it 8%, nl 7% |
| code/webcode2m_new | image | 17.39B | exact | 91.4 | 8.6 | 0.0 | zh 5%, ja 2%, ko 1%, et 0% |
| general_qa/molmo2_multiimageqa_translated | multiimage | 1.15B | prorated | 1.7 | 97.8 | 0.5 | et 5%, es 5%, it 4%, pt 4%, sv 4% |
| knowledge/culturalground_oe | image | 1.04B | prorated | 23.7 | 76.3 | 0.0 | de 12%, fr 12%, nl 11%, es 8% |
| ocr/chartmoe_chart2code | image | 1.49B | exact | 48.1 | 51.9 | 0.0 | zh 52%, it 0%, pl 0%, fr 0% |
| chat/euroblocks | text | 3.68B | exact | 82.6 | 17.4 | 0.0 | zh 3%, fr 2%, es 2%, ru 2% |
| grounding/sav_tracking | video | 1.26B | exact | 50.3 | 48.9 | 0.9 | fr 4%, es 4%, it 4%, de 4% |
| knowledge/culturalground_mcq | image | 774.61M | prorated | 24.5 | 75.5 | 0.0 | fr 11%, nl 11%, de 10%, es 8% |
| ocr/chartmoe_chart2json | image | 1.15B | exact | 50.5 | 49.5 | 0.0 | zh 49%, ko 0%, ja 0% |
| ocr/nemotron_ocr | image | 835.25M | prorated | 46.2 | 53.8 | 0.0 | zh 51%, ca 0%, pl 0%, de 0% |
| grounding/sav_pointing | video | 810.53M | exact | 50.1 | 48.3 | 1.6 | es 4%, fr 4%, it 4%, de 4% |
| grounding/sav_grounding | video | 813.68M | exact | 49.9 | 48.0 | 2.1 | fr 4%, es 4%, it 4%, de 4% |
| ocr/chartmoe_chart2table | image | 854.20M | exact | 55.4 | 44.6 | 0.0 | zh 45%, ca 0%, ro 0%, de 0% |
| temporal_grounding/sav_temporal_grounding | video | 801.58M | exact | 49.4 | 40.8 | 9.7 | fr 4%, es 4%, de 4% |
| general_qa/pangea_laion_multi | image | 241.06M | prorated | 0.0 | 100.0 | 0.0 | pt 12%, fr 12%, it 12%, bg 11%, de 11% |
| ocr/synthdog_multilingual_eu | image | 250.98M | prorated | 4.9 | 95.1 | 0.0 | uk 9%, hr 5%, bg 5%, el 5% |
| spatial/zechen_clevrer_multilingual | video | 240.02M | exact | 0.5 | 96.8 | 2.7 | es 7%, fr 7%, pl 7%, pt 7%, it 7% |
| science/leopard_arxiv_enriched | image | 172.71M | prorated | 6.8 | 93.2 | 0.0 | hr 5%, sl 5%, et 5%, sq 5% |
| embodied_reasoning/alfred_multilingual | video | 144.59M | exact | 0.4 | 98.1 | 1.5 | es 8%, it 7%, nl 7%, pl 7%, pt 7% |
| captioning/llava_video_178k | video | 1.15B | exact | 89.8 | 10.2 | 0.0 | ko 4%, ja 4%, zh 3%, hr 0% |
| ocr/pangea_webui_ocr | image | 105.62M | prorated | 0.0 | 100.0 | 0.0 | es 33%, fr 33%, pt 33% |
| ocr/nvidia_ocr_synth_enru | image | 158.68M | prorated | 49.4 | 50.6 | 0.0 | ru 49%, bg 0%, mk 0%, fr 0% |
| code/chartnet_code | image | 5.40B | exact | 98.5 | 1.5 | 0.0 | id 1%, es 0%, ca 0%, pt 0% |
| ocr/chartnet_csv | image | 3.67B | exact | 98.1 | 1.9 | 0.0 | id 0%, ro 0%, es 0%, ca 0% |
| ocr/wkvvqa | image | 874.70M | exact | 93.2 | 6.8 | 0.0 | zh 3%, ko 2%, de 0%, id 0% |
| gui/seeclick_ground | image | 368.85M | exact | 85.6 | 14.4 | 0.0 | de 7%, nl 3%, fr 2%, pt 0% |
| general_qa/doclingmatix | multiimage | 6.01B | exact | 99.4 | 0.6 | 0.0 | zh 0%, ja 0%, ko 0%, ro 0% |
| ocr/olmocr_mix | image | 477.14M | prorated | 94.6 | 5.4 | 0.0 | de 1%, es 1%, fr 1%, id 0% |
| chart/fv_synthchartnet | image | 367.04M | exact | 94.6 | 5.4 | 0.0 | ca 5%, ro 0%, it 0%, fr 0% |
| subtitles/molmo2_subtitleqa_packed | video | 695.85M | exact | 97.3 | 2.7 | 0.0 | es 2%, pt 0%, fr 0%, de 0% |
| ocr/SynthFormulaNet | image | 83.28M | exact | 79.6 | 20.4 | 0.0 | ca 18%, sv 0%, cy 0%, ro 0% |
| general_qa/llava_video_178k_qa_packed | video | 2.66B | exact | 99.4 | 0.6 | 0.0 | ko 0%, ja 0%, zh 0%, ru 0% |
| grounding/ava_tracking | video | 29.66M | exact | 51.0 | 48.0 | 1.0 | nl 4%, pt 4%, pl 4%, es 3% |
| doc/molmo2_doc | multiimage | 413.35M | prorated | 96.8 | 3.2 | 0.0 | fr 1%, es 0%, de 0%, pt 0% |
| doc/bigdocs_pubtables_1m | image | 435.09M | exact | 97.4 | 2.6 | 0.0 | ca 1%, de 0%, it 0%, ro 0% |
| doc/pangea_table_vqa | image | 11.48M | prorated | 0.0 | 100.0 | 0.0 | fr 100% |
| doc/pangea_doc_vqa | image | 10.79M | prorated | 0.0 | 100.0 | 0.0 | fr 100% |
| ocr/nemotron_ocr7 | image | 21.21M | prorated | 49.7 | 50.3 | 0.0 | zh 50%, no 0%, id 0%, ca 0% |
| ocr/SynthCodeNet | image | 595.62M | exact | 98.5 | 1.5 | 0.0 | fr 0%, ca 0%, pt 0%, es 0% |
| ocr/latexformulas | image | 73.87M | exact | 90.6 | 9.4 | 0.0 | ca 7%, cy 0%, sv 0%, da 0% |
| ocr/doclaynet | image | 128.06M | exact | 94.9 | 5.1 | 0.0 | de 2%, zh 1%, ru 0%, fr 0% |
| grounding/ava_grounding | video | 13.48M | exact | 51.2 | 47.4 | 1.4 | nl 4%, pt 4%, es 4%, fr 3% |
| grounding/ava_pointing | video | 13.11M | exact | 51.3 | 47.4 | 1.3 | nl 4%, pt 4%, es 4%, fr 3% |
| temporal_grounding/ava_temporal_grounding | video | 12.75M | exact | 51.4 | 47.3 | 1.4 | nl 4%, pt 4%, es 4%, fr 3% |
| grounding/textocr_grounded | image | 49.52M | exact | 89.8 | 10.2 | 0.0 | de 4%, es 1%, pt 1%, fr 1% |
| code/datik | image | 159.32M | exact | 97.2 | 2.8 | 0.0 | cy 1%, ca 0%, ro 0%, fr 0% |
| chart/pangea_chartqa | image | 4.45M | prorated | 0.0 | 100.0 | 0.0 | ro 17%, pt 16%, tr 16%, ru 16%, fr 14% |
| captioning/sharegpt4o | image | 30.24M | exact | 86.4 | 13.6 | 0.0 | zh 13%, ja 0%, ko 0% |
| ocr/pangea_mtvqa | image | 3.55M | prorated | 0.0 | 100.0 | 0.0 | de 32%, fr 26%, ru 21%, it 21% |
| code/webmmu | image | 4.59M | exact | 47.1 | 52.9 | 0.0 | es 18%, de 17%, fr 16%, zh 2% |
| ocr/nemotron_ocr6 | image | 83.35M | prorated | 97.4 | 2.6 | 0.0 | zh 1%, es 1%, de 0%, ro 0% |
| science/kaleidoscope | image | 1.94M | prorated | 0.0 | 100.0 | 0.0 | uk 19%, sr 19%, ru 16%, hu 10%, nl 9% |
| ocr/vcr_wiki_en_hard | image | 278.32M | prorated | 99.4 | 0.6 | 0.0 | fr 0%, de 0%, ro 0%, it 0% |
| temporal_grounding/crosstask_v2_packed | video | 15.64M | exact | 90.2 | 9.8 | 0.0 | fr 9%, cy 0%, et 0%, it 0% |
| chart/molmo2_table | multiimage | 286.06M | exact | 99.5 | 0.5 | 0.0 | zh 0%, es 0%, pt 0%, ja 0% |
| ocr/cc_ocr_multi_lan | image | 1.49M | exact | 38.5 | 61.5 | 0.0 | vi 12%, ko 9%, ar 9%, it 5% |
| doc/bigdocs_wikitq | image | 28.49M | exact | 96.9 | 3.1 | 0.0 | zh 1%, ja 0%, es 0%, id 0% |
| general_qa/worldvqa | image | 1.98M | exact | 60.8 | 39.2 | 0.0 | zh 39%, es 0%, it 0%, ja 0% |
| pointing/molmopoint_guisyn | image | 51.39M | exact | 98.5 | 1.5 | 0.0 | zh 1%, ja 0%, ko 0%, de 0% |
| ocr/vcr_wiki_en_easy | image | 109.69M | prorated | 99.4 | 0.6 | 0.0 | fr 0%, de 0%, ro 0%, it 0% |
| chart/caul_sqa | image | 7.47M | exact | 91.3 | 8.7 | 0.0 | de 2%, ja 1%, id 1%, af 1% |
| chart/leopard_chartgemma | multiimage | 92.91M | exact | 99.4 | 0.6 | 0.0 | fr 0%, ca 0%, it 0%, es 0% |
| ocr/textocr | image | 24.17M | exact | 97.9 | 2.1 | 0.0 | zh 1%, ja 1%, ko 0% |
| general_qa/gqa | image | 47.12M | exact | 98.9 | 1.1 | 0.0 | it 0%, da 0%, cy 0%, tl 0% |
| doc/bigdocs_tabfact | image | 19.57M | exact | 97.7 | 2.3 | 0.0 | zh 1%, it 0%, es 0%, fr 0% |
| code/mmcode | image | 6.89M | exact | 93.9 | 6.1 | 0.0 | ro 1%, no 1%, ja 1%, ca 1% |
| math/visualwebinstruct | multiimage | 36.19M | exact | 99.0 | 1.0 | 0.0 | pt 0%, es 0%, zh 0%, vi 0% |
| chart/caul_wtq | image | 33.89M | exact | 98.9 | 1.1 | 0.0 | zh 0%, ja 0%, af 0%, it 0% |
| general_qa/molmo2_multiimageqa | multiimage | 61.28M | prorated | 99.5 | 0.5 | 0.0 | tl 0%, de 0%, fr 0%, vi 0% |
| science/molmo2_syn_chemical | multiimage | 8.94M | exact | 97.3 | 2.7 | 0.0 | ro 2%, fr 0%, nl 0%, ca 0% |
| gui/waveui_ground | image | 19.41M | exact | 98.8 | 1.2 | 0.0 | zh 1%, ja 0%, ko 0%, de 0% |
| temporal_grounding/breakfast_actions_packed | video | 6.61M | exact | 96.8 | 3.2 | 0.0 | fr 3% |
| captioning/gemini_textcaps_vqa | image | 24.74M | exact | 99.2 | 0.8 | 0.0 | zh 0%, ja 0%, ko 0%, de 0% |
| doc/bigdocs_cocotext | image | 15.26M | exact | 98.8 | 1.2 | 0.0 | de 1%, it 0%, ca 0%, pt 0% |
| ocr/hw_squad | image | 20.78M | exact | 99.5 | 0.5 | 0.0 | fr 0%, id 0%, es 0%, de 0% |
| gui/waveui_point | image | 8.37M | exact | 98.7 | 1.3 | 0.0 | zh 1%, ja 0% |
| science/molmo2_syn_circuit | multiimage | 12.45M | exact | 99.1 | 0.9 | 0.0 | ro 0%, id 0%, zh 0%, pt 0% |
| doc/cauldron_docvqa | image | 12.28M | exact | 99.4 | 0.6 | 0.0 | de 0%, fr 0%, id 0%, ca 0% |
| code/chartmimic | image | 10.66M | exact | 99.3 | 0.7 | 0.0 | tl 0%, ca 0%, zh 0%, id 0% |
| chart/molmo2_diagram_single | image | 3.09M | exact | 98.6 | 1.4 | 0.0 | fr 0%, es 0%, pt 0%, zh 0% |
| ocr/coco-text | image | 4.30M | exact | 99.1 | 0.9 | 0.0 | zh 0%, ja 0%, fr 0%, ko 0% |
| general_qa/infographic_vqa | image | 4.89M | exact | 99.3 | 0.7 | 0.0 | de 0%, id 0%, nl 0%, ca 0% |
| doc/pixmo_docs | image | 5.01M | exact | 99.4 | 0.6 | 0.0 | de 0%, zh 0%, es 0%, af 0% |
| doc/bigdocs_cord_v2 | image | 2.26M | exact | 99.2 | 0.8 | 0.0 | de 1%, ro 0% |
| chart/molmo2_graphic_single | image | 993,234 | exact | 99.0 | 1.0 | 0.0 | es 0%, fr 0%, de 0%, so 0% |
| code/plot2code | image | 419,824 | exact | 97.8 | 2.2 | 0.0 | ca 1%, hu 0%, cy 0%, so 0% |

`language_counts.md` sections with no dataset in the mixture: `image/cg_mcq_smoke`.

## Code and math

Datasets in the mixture's `code` and `math` categories (formula/code OCR sets under `ocr` are not included). **Answer tokens** are the supervised positions of the training loss mask (each assistant answer plus its closing `<|im_end|>`, read off the template markers by the encoder's own `assistant_answer_mask`; checked against the real encoder, 0 mismatches): the code/math the model learns to write. **Sample tokens** are everything in those samples (images, instructions, chat template). Shares are of all tokens (r=1).

### code: 20.43B answer tokens (13.08%), 30.02B sample tokens (19.23%)

| modality | datasets | samples | answer tokens | % of all | sample tokens | % of all |
|---|---|---|---|---|---|---|
| image | 10 | 8,178,235 | 18.11B | 11.60% | 26.58B | 17.02% |
| text | 1 | 2,071,395 | 2.31B | 1.48% | 3.44B | 2.20% |

Datasets (answer / sample tokens): `image/webcode2m_new` (13.90B / 17.39B), `image/chartnet_code` (2.88B / 5.40B), `text/euroblocks` (2.31B / 3.44B), `image/websight_new` (660.93M / 2.11B), `image/web2code_new` (430.66M / 1.33B), `image/datikz_union` (132.04M / 168.64M), `image/datik` (104.22M / 159.32M), `image/chartmimic` (4.67M / 10.66M), `image/mmcode` (2.42M / 6.89M), `image/webmmu` (644,414 / 4.59M), `image/plot2code` (101,364 / 419,824)

### math: 1.74B answer tokens (1.12%), 2.34B sample tokens (1.50%)

| modality | datasets | samples | answer tokens | % of all | sample tokens | % of all |
|---|---|---|---|---|---|---|
| image | 21 | 767,235 | 95.29M | 0.06% | 483.86M | 0.31% |
| multiimage | 2 | 23,945 | 14.23M | 0.01% | 37.83M | 0.02% |
| text | 1 | 2,283,874 | 1.64B | 1.05% | 1.82B | 1.17% |

Datasets (answer / sample tokens): `text/euroblocks` (1.64B / 1.82B), `image/visualwebinstruct_onevision` (32.25M / 116.96M), `image/finevision_cosyn_400k_math` (29.87M / 75.79M), `multiimage/visualwebinstruct` (13.63M / 36.19M), `image/finevision_mavis_math_rule_geo` (12.83M / 127.98M), `image/r1_vision_stratos_17k` (6.90M / 19.99M), `image/finevision_mavis_math_metagen` (5.74M / 42.48M), `image/finevision_geomverse` (2.50M / 13.05M), `image/finevision_geo170k_align` (1.95M / 3.80M), `image/finevision_geo170k_qa` (1.27M / 3.23M), `image/finevision_clevr_math` (1.14M / 31.38M), `multiimage/mv_math` (604,756 / 1.64M), `image/mathvision` (199,406 / 1.91M), `image/finevision_geoqa_plus_mathv360k` (134,894 / 2.24M), `image/finevision_raven` (105,081 / 30.41M), `image/finevision_unigeo_mathv360k` (93,568 / 1.53M), `image/finevision_geometry3k_mathv360k` (78,521 / 2.72M), `image/mathverse_refined_final` (71,615 / 1.42M), `image/finevision_super_clevr_mathv360k` (53,304 / 3.90M), `image/finevision_mapqa_mathv360k` (49,858 / 2.65M), `image/finevision_clevr_math_mathv360k` (32,886 / 1.42M), `image/finevision_intergps` (8,800 / 359,553), `image/finevision_geo3k` (4,182 / 562,884), `image/finevision_geos_mathv360k` (4,005 / 69,732)

## Data issues found by the census

Everything else found by the sampled data audit (answer-format, language, timestamp, index and training-code problems, with sample keys): [`BUGS.md`](BUGS.md).

## Regenerating

Inside the container, with `EUROVL_DATA_ROOT` (energon data root holding `mixture.yaml`), `EUROVL_HF` (the model's HF export) and `EUROVL_CENSUS_WORK_DIR` (per-shard results, not in git) set. Use about 16 workers: 48 workers importing the model code at once ran out of file descriptors on the container filesystem.

1. `scripts/census/estimate_token_budget.py` → `results/<modality>_summary.json` (`--stale` lists datasets whose shards changed since they were counted, `--refresh` recounts them)
2. `scripts/report/compute_language_split.py --language-counts <curator language_counts.md>` → `results/language_split.json`
3. `scripts/report/render_readme.py` → this file

Checks: `scripts/validation/check_against_encoder.py` (census vs. the real training encoder, sample by sample) and `scripts/validation/validate_token_estimates.py` (vision-token math vs. the real processor). Standalone single-sample estimator for sharing: `scripts/standalone_estimate_tokens.py`.

# image

248 dataset(s), 90,862,914 samples, **82.90B tokens (r=1)**, **82.90B effective (r×tokens)**.

## 3d_grounding

1 dataset(s), 129.82M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| hypersim_3d_ground | 1 | 100,003 | 1036 | 262 | 103.60M | 26.22M | 129.82M | 1024 | 768 | 1.33 |

## captioning

12 dataset(s), 12.61B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| cc12m | 1 | 9,797,215 | 446 | 110 | 4.37B | 1.07B | 5.45B | 624 | 536 | 1.21 |
| grit | 1 | 8,507,802 | 503 | 117 | 4.28B | 993.27M | 5.28B | 728 | 584 | 1.34 |
| cc3m | 1 | 1,548,826 | 424 | 95 | 656.89M | 147.62M | 804.51M | 619 | 517 | 1.24 |
| pixmo-cap | 1 | 702,205 | 850 | 259 | 596.80M | 181.61M | 778.41M | 1435 | 1374 | 1.20 |
| sbu-captions | 1 | 499,580 | 100 | 93 | 49.96M | 46.56M | 96.52M | 256 | 256 | 1.00 |
| finevision_ureader_cap | 1 | 87,762 | 971 | 42 | 85.18M | 3.67M | 88.86M | 949 | 815 | 1.23 |
| sharegpt4o | 1 | 42,636 | 572 | 138 | 24.38M | 5.86M | 30.24M | 857 | 775 | 1.27 |
| gemini_textcaps_vqa | 1 | 21,946 | 971 | 156 | 21.31M | 3.43M | 24.74M | 949 | 817 | 1.23 |
| textcaps | 1 | 21,942 | 971 | 40 | 21.31M | 874,404 | 22.18M | 949 | 817 | 1.23 |
| coco-caption | 1 | 39,621 | 369 | 95 | 14.62M | 3.75M | 18.38M | 576 | 485 | 1.25 |
| mminstruct_caption_en | 1 | 17,512 | 606 | 245 | 10.60M | 4.28M | 14.89M | 907 | 723 | 1.41 |
| flickr30k | 1 | 30,000 | 235 | 97 | 7.04M | 2.92M | 9.96M | 460 | 395 | 1.23 |

## chart

40 dataset(s), 8.61B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| chartnet_summary | 1 | 2,513,168 | 979 | 487 | 2.46B | 1.23B | 3.69B | 1744 | 1185 | 1.49 |
| caul_plotqa | 1 | 1,089,485 | 956 | 1038 | 1.04B | 1.13B | 2.17B | 1106 | 683 | 1.62 |
| fv_unichart | 1 | 727,728 | 667 | 548 | 485.61M | 398.92M | 884.53M | 894 | 552 | 1.66 |
| fv_synthchartnet | 1 | 500,000 | 573 | 161 | 286.66M | 80.37M | 367.04M | 772 | 591 | 1.37 |
| fv_mmc_instruct | 1 | 168,178 | 773 | 838 | 129.94M | 140.88M | 270.82M | 622 | 1277 | 0.62 |
| mmtab | 1 | 232,879 | 798 | 298 | 185.88M | 69.44M | 255.32M | 1903 | 853 | 2.86 |
| fv_cosyn_chart | 1 | 116,813 | 1053 | 794 | 122.98M | 92.81M | 215.79M | 1769 | 1243 | 1.45 |
| fv_arxivqa | 1 | 100,000 | 1030 | 150 | 103.04M | 14.97M | 118.01M | 1800 | 1275 | 1.71 |
| chartmoe_chartgemma | 1 | 163,238 | 526 | 113 | 85.80M | 18.38M | 104.18M | 701 | 559 | 1.29 |
| fv_cosyn_table | 1 | 46,518 | 979 | 777 | 45.53M | 36.13M | 81.66M | 1389 | 917 | 1.76 |
| fv_figureqa | 1 | 100,000 | 322 | 392 | 32.20M | 39.22M | 71.42M | 589 | 400 | 1.47 |
| fv_cosyn_diagram | 1 | 34,963 | 1048 | 584 | 36.63M | 20.43M | 57.07M | 1470 | 1153 | 1.70 |
| caul_wikisql | 1 | 74,868 | 508 | 168 | 38.06M | 12.56M | 50.62M | 698 | 595 | 1.90 |
| chartnet_realworldchart | 1 | 30,000 | 550 | 984 | 16.51M | 29.51M | 46.02M | 770 | 608 | 1.37 |
| caul_wtq | 1 | 38,246 | 678 | 208 | 25.94M | 7.95M | 33.89M | 977 | 644 | 2.38 |
| nemo_chartqa_nothink | 1 | 23,571 | 600 | 274 | 14.15M | 6.46M | 20.61M | 773 | 593 | 1.34 |
| fv_ureader_ie | 1 | 17,320 | 1059 | 47 | 18.34M | 809,894 | 19.15M | 1515 | 2012 | 0.76 |
| fv_chart2text | 1 | 26,961 | 565 | 132 | 15.22M | 3.56M | 18.79M | 716 | 602 | 1.25 |
| gemini_chartqa_filtered | 1 | 25,055 | 605 | 112 | 15.16M | 2.80M | 17.96M | 771 | 601 | 1.32 |
| leopard_figureqa | 1 | 17,945 | 322 | 668 | 5.78M | 11.99M | 17.77M | 589 | 400 | 1.47 |
| nemo_fintabnet_nothink | 1 | 8,352 | 1004 | 721 | 8.39M | 6.02M | 14.41M | 771 | 999 | 0.77 |
| chartqa_llava | 1 | 18,260 | 613 | 67 | 11.20M | 1.23M | 12.43M | 779 | 606 | 1.33 |
| leopard_chartgemma | 1 | 16,323 | 528 | 104 | 8.61M | 1.70M | 10.31M | 702 | 561 | 1.29 |
| caul_vistext | 1 | 9,969 | 696 | 147 | 6.94M | 1.46M | 8.40M | 779 | 690 | 1.15 |
| caul_sqa | 1 | 8,644 | 558 | 306 | 4.83M | 2.65M | 7.47M | 976 | 480 | 2.90 |
| fv_finqa | 1 | 5,276 | 173 | 1145 | 914,233 | 6.04M | 6.95M | 696 | 181 | 4.79 |
| fv_figureqa_mathv360k | 1 | 17,587 | 301 | 63 | 5.30M | 1.11M | 6.40M | 548 | 400 | 1.37 |
| molmo2_chart_single | 1 | 3,701 | 975 | 537 | 3.61M | 1.99M | 5.60M | 1420 | 1011 | 1.51 |
| ov_ai2d_internvl | 1 | 12,403 | 393 | 53 | 4.88M | 653,256 | 5.53M | 605 | 473 | 1.41 |
| pangea_chartqa | 1 | 6,844 | 605 | 45 | 4.14M | 307,038 | 4.45M | 772 | 600 | 1.33 |
| chartmoe_chartqa | 1 | 6,266 | 622 | 47 | 3.90M | 294,064 | 4.19M | 767 | 629 | 1.28 |
| molmo2_diagram_single | 1 | 2,171 | 1013 | 408 | 2.20M | 886,425 | 3.09M | 1733 | 1343 | 1.92 |
| caul_multihiertt | 1 | 7,619 | 301 | 78 | 2.29M | 596,139 | 2.89M | 697 | 330 | 3.43 |
| caul_hitab | 1 | 2,516 | 637 | 255 | 1.60M | 642,348 | 2.25M | 978 | 565 | 2.78 |
| caul_tat_qa | 1 | 2,199 | 253 | 667 | 555,628 | 1.47M | 2.02M | 696 | 271 | 3.37 |
| chartqapro_refined_235b | 1 | 1,948 | 766 | 128 | 1.49M | 249,885 | 1.74M | 1194 | 986 | 1.37 |
| fv_lrv_chart | 1 | 1,776 | 700 | 172 | 1.24M | 306,192 | 1.55M | 780 | 692 | 1.15 |
| caul_ai2d | 1 | 2,439 | 402 | 204 | 979,651 | 497,293 | 1.48M | 613 | 481 | 1.40 |
| molmo2_table_single | 1 | 780 | 1012 | 570 | 789,202 | 444,930 | 1.23M | 1530 | 1253 | 1.56 |
| molmo2_graphic_single | 1 | 859 | 799 | 357 | 686,542 | 306,692 | 993,234 | 1254 | 1010 | 1.40 |

## code

10 dataset(s), 26.58B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| webcode2m_new | 1 | 3,167,358 | 1064 | 4426 | 3.37B | 14.02B | 17.39B | 1295 | 1617 | 0.90 |
| chartnet_code | 1 | 2,513,168 | 979 | 1170 | 2.46B | 2.94B | 5.40B | 1744 | 1185 | 1.49 |
| websight_new | 1 | 1,319,321 | 1071 | 529 | 1.41B | 697.61M | 2.11B | 2561 | 1656 | 1.62 |
| web2code_new | 1 | 806,900 | 1069 | 579 | 862.60M | 467.29M | 1.33B | 1322 | 882 | 1.60 |
| datikz_union | 1 | 138,395 | 221 | 997 | 30.62M | 138.02M | 168.64M | 416 | 416 | 1.00 |
| datik | 1 | 220,183 | 225 | 499 | 49.54M | 109.78M | 159.32M | 420 | 420 | 1.00 |
| chartmimic | 1 | 4,800 | 942 | 1280 | 4.52M | 6.14M | 10.66M | 1124 | 782 | 1.53 |
| mmcode | 1 | 4,427 | 194 | 1363 | 859,251 | 6.03M | 6.89M | 445 | 265 | 2.73 |
| webmmu | 1 | 3,315 | 1083 | 302 | 3.59M | 1.00M | 4.59M | 1493 | 1538 | 0.96 |
| plot2code | 1 | 368 | 595 | 546 | 219,068 | 200,756 | 419,824 | 903 | 664 | 1.40 |

## counting

5 dataset(s), 2.20B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| objects365_count | 1 | 2,676,042 | 547 | 55 | 1.46B | 146.12M | 1.61B | 753 | 612 | 1.27 |
| openimages_count | 1 | 503,045 | 967 | 71 | 486.44M | 35.51M | 521.95M | 965 | 793 | 1.28 |
| tallyqa | 1 | 98,680 | 329 | 24 | 32.51M | 2.38M | 34.90M | 546 | 453 | 1.27 |
| pixmo_count_derived | 1 | 33,428 | 943 | 35 | 31.51M | 1.18M | 32.69M | 1299 | 1033 | 1.30 |
| taco_count | 1 | 646 | 1058 | 55 | 683,474 | 35,269 | 718,743 | 2899 | 3060 | 1.00 |

## doc

19 dataset(s), 1.73B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| docmatix | 1 | 566,267 | 1069 | 441 | 605.46M | 249.97M | 855.44M | 1506 | 1782 | 0.86 |
| bigdocs_pubtables_1m | 1 | 345,696 | 396 | 862 | 136.99M | 298.11M | 435.09M | 614 | 526 | 1.63 |
| bigdocs_arxiv_ocr | 1 | 110,855 | 1068 | 444 | 118.36M | 49.24M | 167.60M | 1680 | 2226 | 0.76 |
| docreason51k | 1 | 51,726 | 919 | 125 | 47.53M | 6.47M | 54.00M | 1603 | 1973 | 1.48 |
| bigdocs_arxiv_table_cap | 1 | 72,524 | 413 | 53 | 29.97M | 3.83M | 33.80M | 924 | 362 | 3.82 |
| bigdocs_wikitq | 1 | 22,007 | 613 | 682 | 13.48M | 15.01M | 28.49M | 918 | 646 | 1.94 |
| leopard_mplugdocreason | 1 | 25,863 | 919 | 121 | 23.76M | 3.12M | 26.89M | 1603 | 1973 | 1.48 |
| bigdocs_tabfact | 1 | 16,572 | 463 | 718 | 7.67M | 11.90M | 19.57M | 835 | 437 | 2.48 |
| leopard_monkey | 1 | 31,156 | 490 | 62 | 15.27M | 1.94M | 17.21M | 651 | 754 | 1.04 |
| bigdocs_cocotext | 1 | 21,364 | 374 | 340 | 7.99M | 7.27M | 15.26M | 585 | 484 | 1.27 |
| leopard_dude | 1 | 12,108 | 1065 | 61 | 12.90M | 733,067 | 13.63M | 2013 | 2490 | 0.85 |
| cauldron_docvqa | 1 | 10,177 | 1055 | 152 | 10.74M | 1.55M | 12.28M | 1739 | 2089 | 0.88 |
| gemini_docvqa_filtered | 1 | 9,664 | 1056 | 178 | 10.20M | 1.72M | 11.93M | 1741 | 2092 | 0.88 |
| pangea_table_vqa | 1 | 16,408 | 423 | 277 | 6.93M | 4.55M | 11.48M | 930 | 374 | 3.76 |
| pangea_doc_vqa | 1 | 9,665 | 908 | 208 | 8.78M | 2.01M | 10.79M | 1649 | 1628 | 1.39 |
| docreason25k_refined | 1 | 8,726 | 869 | 317 | 7.59M | 2.76M | 10.35M | 1400 | 1734 | 1.33 |
| pixmo_docs | 1 | 3,634 | 1053 | 327 | 3.83M | 1.19M | 5.01M | 1848 | 1289 | 1.45 |
| bigdocs_cord_v2 | 1 | 997 | 936 | 1334 | 933,321 | 1.33M | 2.26M | 1001 | 1578 | 0.65 |
| tat_dqa | 1 | 1,969 | 64 | 770 | 126,016 | 1.52M | 1.64M | 224 | 224 | 1.00 |

## general_qa

23 dataset(s), 1.28B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| pangea_laion_multi | 1 | 324,232 | 455 | 288 | 147.65M | 93.42M | 241.06M | 671 | 516 | 1.34 |
| pixmo_cap_qa | 1 | 147,118 | 828 | 319 | 121.85M | 47.00M | 168.85M | 1374 | 1245 | 1.23 |
| mminstruct_qa | 1 | 146,205 | 581 | 549 | 85.01M | 80.30M | 165.30M | 862 | 705 | 1.34 |
| mmevol | 1 | 160,215 | 450 | 494 | 72.03M | 79.07M | 151.11M | 738 | 607 | 1.28 |
| dvqa | 1 | 200,000 | 256 | 424 | 51.20M | 84.71M | 135.91M | 448 | 448 | 1.00 |
| pixmo_ask_model_anything | 1 | 70,590 | 732 | 277 | 51.64M | 19.53M | 71.17M | 985 | 889 | 1.20 |
| llava_instruct | 1 | 81,467 | 369 | 490 | 30.08M | 39.95M | 70.03M | 578 | 483 | 1.26 |
| rsvqa_hr | 1 | 74,542 | 361 | 498 | 26.91M | 37.10M | 64.01M | 512 | 512 | 1.00 |
| gqa | 1 | 87,931 | 278 | 258 | 24.41M | 22.71M | 47.12M | 498 | 411 | 1.27 |
| vqav2 | 1 | 84,751 | 369 | 153 | 31.30M | 12.99M | 44.29M | 578 | 484 | 1.26 |
| clevr | 1 | 69,995 | 216 | 358 | 15.12M | 25.08M | 40.20M | 480 | 320 | 1.50 |
| alfworldgpt | 1 | 43,716 | 121 | 416 | 5.29M | 18.20M | 23.49M | 300 | 300 | 1.00 |
| cocoqa | 1 | 46,238 | 370 | 55 | 17.10M | 2.56M | 19.66M | 580 | 483 | 1.26 |
| lrv_normal | 1 | 12,745 | 243 | 590 | 3.09M | 7.52M | 10.61M | 473 | 392 | 1.27 |
| visual7w | 1 | 14,412 | 273 | 316 | 3.93M | 4.55M | 8.48M | 492 | 409 | 1.27 |
| infographic_vqa | 1 | 4,214 | 978 | 182 | 4.12M | 765,681 | 4.89M | 880 | 1563 | 0.69 |
| gemini_infographic_vqa_filtered | 1 | 2,049 | 917 | 122 | 1.88M | 250,669 | 2.13M | 907 | 1100 | 0.94 |
| worldvqa | 1 | 2,990 | 631 | 32 | 1.89M | 96,132 | 1.98M | 895 | 673 | 1.46 |
| spark | 1 | 3,703 | 458 | 59 | 1.70M | 218,767 | 1.92M | 641 | 485 | 1.39 |
| yesbut | 1 | 1,084 | 1056 | 243 | 1.14M | 263,422 | 1.41M | 1432 | 1053 | 1.35 |
| vsr | 1 | 2,157 | 387 | 50 | 835,124 | 108,154 | 943,278 | 580 | 507 | 1.20 |
| vizwiz | 1 | 737 | 916 | 274 | 675,097 | 201,649 | 876,746 | 1026 | 1324 | 0.78 |
| newyorker_caption_contest | 1 | 1,632 | 415 | 119 | 676,500 | 194,578 | 871,078 | 606 | 474 | 1.32 |

## grounding

6 dataset(s), 3.40B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| objects365_ground | 1 | 4,469,839 | 554 | 83 | 2.48B | 370.10M | 2.85B | 761 | 616 | 1.28 |
| openimages_box | 1 | 344,300 | 966 | 118 | 332.59M | 40.72M | 373.31M | 968 | 788 | 1.29 |
| refcoco_ground | 1 | 267,242 | 379 | 57 | 101.21M | 15.17M | 116.37M | 592 | 483 | 1.28 |
| textocr_grounded | 1 | 24,863 | 971 | 1020 | 24.15M | 25.37M | 49.52M | 949 | 817 | 1.23 |
| groundui_ground | 1 | 12,518 | 1067 | 53 | 13.36M | 662,738 | 14.02M | 1404 | 1105 | 1.46 |
| taco_ground | 1 | 3,123 | 1057 | 64 | 3.30M | 198,471 | 3.50M | 2947 | 3078 | 1.02 |

## gui

23 dataset(s), 1.70B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| aguvis_stage1 | 1 | 504,911 | 1029 | 315 | 519.46M | 159.05M | 678.51M | 1742 | 1133 | 1.60 |
| seeclick_ground | 1 | 263,098 | 1032 | 370 | 271.52M | 97.33M | 368.85M | 1920 | 1080 | 1.78 |
| seeclick_qa | 1 | 129,704 | 1032 | 116 | 133.85M | 14.99M | 148.84M | 1920 | 1080 | 1.78 |
| odyssey_actions | 1 | 111,050 | 1065 | 61 | 118.25M | 6.75M | 124.99M | 1534 | 2211 | 0.77 |
| androidcontrol_actions | 1 | 83,819 | 1056 | 65 | 88.51M | 5.47M | 93.99M | 1090 | 2417 | 0.45 |
| amex_actions | 1 | 38,704 | 1071 | 77 | 41.44M | 2.97M | 44.41M | 1429 | 3034 | 0.47 |
| widget_ground | 1 | 33,890 | 960 | 49 | 32.53M | 1.64M | 34.17M | 963 | 1711 | 0.56 |
| aitw_actions | 1 | 43,129 | 662 | 51 | 28.55M | 2.20M | 30.74M | 493 | 1011 | 0.49 |
| screenqa | 1 | 30,054 | 954 | 38 | 28.66M | 1.14M | 29.80M | 952 | 1693 | 0.56 |
| ricosca_refer | 1 | 27,865 | 954 | 42 | 26.58M | 1.17M | 27.75M | 953 | 1694 | 0.56 |
| waveui_ground | 1 | 17,411 | 1057 | 57 | 18.41M | 999,613 | 19.41M | 1199 | 822 | 1.53 |
| ricosca_point | 1 | 17,405 | 954 | 38 | 16.61M | 660,245 | 17.27M | 954 | 1695 | 0.56 |
| widget_point | 1 | 14,435 | 960 | 40 | 13.86M | 577,271 | 14.44M | 963 | 1712 | 0.56 |
| uibert_ground | 1 | 11,681 | 953 | 53 | 11.13M | 619,955 | 11.75M | 952 | 1692 | 0.56 |
| screen2words | 1 | 11,629 | 949 | 60 | 11.03M | 692,670 | 11.73M | 945 | 1679 | 0.56 |
| waveui_point | 1 | 7,567 | 1057 | 49 | 8.00M | 369,946 | 8.37M | 1200 | 827 | 1.52 |
| mind2web_actions | 1 | 7,362 | 1066 | 68 | 7.85M | 499,397 | 8.35M | 1287 | 4883 | 0.40 |
| omniact_actions | 1 | 6,713 | 1054 | 57 | 7.07M | 379,580 | 7.45M | 2240 | 1354 | 1.67 |
| leopard_rico | 1 | 6,290 | 960 | 37 | 6.04M | 231,478 | 6.27M | 963 | 1710 | 0.56 |
| uibert_point | 1 | 4,979 | 953 | 44 | 4.74M | 220,976 | 4.96M | 951 | 1690 | 0.56 |
| leopard_mind2web | 1 | 1,765 | 1060 | 211 | 1.87M | 372,162 | 2.24M | 1289 | 1760 | 0.80 |
| ricosca_ground | 1 | 1,602 | 934 | 46 | 1.50M | 74,046 | 1.57M | 920 | 1636 | 0.56 |
| leopard_omniact | 1 | 1,038 | 1066 | 86 | 1.11M | 89,751 | 1.20M | 1440 | 900 | 1.60 |

## knowledge

12 dataset(s), 4.64B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| wit | 1 | 12,916,297 | 116 | 91 | 1.50B | 1.18B | 2.68B | 300 | 278 | 1.25 |
| culturalground_oe | 1 | 5,044,547 | 148 | 58 | 744.88M | 290.22M | 1.04B | 356 | 312 | 1.21 |
| culturalground_mcq | 1 | 3,498,781 | 148 | 73 | 519.18M | 255.43M | 774.61M | 359 | 311 | 1.23 |
| visual_genome | 1 | 132,760 | 251 | 251 | 33.26M | 33.35M | 66.61M | 478 | 397 | 1.27 |
| localized_narratives | 1 | 118,272 | 369 | 80 | 43.68M | 9.50M | 53.18M | 578 | 484 | 1.25 |
| cosyn_music | 1 | 11,969 | 1000 | 434 | 11.96M | 5.19M | 17.15M | 820 | 1008 | 0.88 |
| a-okvqa | 1 | 17,315 | 374 | 38 | 6.47M | 655,739 | 7.12M | 587 | 482 | 1.28 |
| gemini_aokvqa_filtered | 1 | 11,853 | 374 | 120 | 4.43M | 1.42M | 5.85M | 587 | 482 | 1.28 |
| okvqa | 1 | 8,998 | 372 | 37 | 3.35M | 334,370 | 3.68M | 618 | 448 | 1.40 |
| viquae | 1 | 2,384 | 373 | 48 | 889,216 | 115,225 | 1.00M | 512 | 536 | 1.12 |
| web_landmark | 1 | 500 | 860 | 201 | 429,949 | 100,717 | 530,666 | 1356 | 943 | 1.50 |
| web_celebrity | 1 | 495 | 778 | 159 | 385,056 | 78,902 | 463,958 | 1196 | 738 | 1.67 |

## math

21 dataset(s), 483.86M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| finevision_mavis_math_rule_geo | 1 | 99,986 | 1055 | 225 | 105.51M | 22.47M | 127.98M | 1647 | 1645 | 1.11 |
| visualwebinstruct_onevision | 1 | 263,578 | 277 | 166 | 73.11M | 43.85M | 116.96M | 524 | 351 | 1.76 |
| finevision_cosyn_400k_math | 1 | 66,714 | 667 | 469 | 44.47M | 31.31M | 75.79M | 1017 | 537 | 2.30 |
| finevision_mavis_math_metagen | 1 | 87,348 | 328 | 158 | 28.65M | 13.84M | 42.48M | 592 | 388 | 1.76 |
| finevision_clevr_math | 1 | 70,000 | 216 | 232 | 15.12M | 16.26M | 31.38M | 480 | 320 | 1.50 |
| finevision_raven | 1 | 42,000 | 693 | 32 | 29.09M | 1.32M | 30.41M | 664 | 804 | 0.88 |
| r1_vision_stratos_17k | 1 | 12,585 | 802 | 786 | 10.10M | 9.90M | 19.99M | 1375 | 994 | 2.74 |
| finevision_geomverse | 1 | 9,303 | 1047 | 356 | 9.74M | 3.31M | 13.05M | 1395 | 1690 | 0.91 |
| finevision_super_clevr_mathv360k | 1 | 8,642 | 414 | 37 | 3.58M | 320,920 | 3.90M | 640 | 480 | 1.33 |
| finevision_geo170k_align | 1 | 35,297 | 29 | 79 | 1.01M | 2.79M | 3.80M | 152 | 118 | 1.39 |
| finevision_geo170k_qa | 1 | 12,101 | 42 | 225 | 504,710 | 2.73M | 3.23M | 191 | 129 | 1.48 |
| finevision_geometry3k_mathv360k | 1 | 9,724 | 201 | 79 | 1.95M | 769,292 | 2.72M | 436 | 310 | 1.44 |
| finevision_mapqa_mathv360k | 1 | 5,225 | 450 | 57 | 2.35M | 297,589 | 2.65M | 700 | 500 | 1.40 |
| finevision_geoqa_plus_mathv360k | 1 | 17,162 | 27 | 104 | 459,142 | 1.78M | 2.24M | 148 | 114 | 1.40 |
| mathvision | 1 | 3,344 | 410 | 160 | 1.37M | 534,586 | 1.91M | 770 | 500 | 1.79 |
| finevision_unigeo_mathv360k | 1 | 11,949 | 24 | 103 | 289,706 | 1.24M | 1.53M | 140 | 110 | 1.42 |
| finevision_clevr_math_mathv360k | 1 | 5,280 | 216 | 53 | 1.14M | 281,357 | 1.42M | 480 | 320 | 1.50 |
| mathverse_refined_final | 1 | 3,128 | 386 | 68 | 1.21M | 212,647 | 1.42M | 618 | 525 | 1.29 |
| finevision_geo3k | 1 | 2,091 | 188 | 81 | 392,828 | 170,056 | 562,884 | 421 | 300 | 1.44 |
| finevision_intergps | 1 | 1,280 | 178 | 103 | 227,416 | 132,137 | 359,553 | 405 | 294 | 1.42 |
| finevision_geos_mathv360k | 1 | 498 | 52 | 88 | 25,693 | 44,039 | 69,732 | 225 | 154 | 1.48 |

## medical

5 dataset(s), 357.76M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| pmc_vqa | 1 | 414,950 | 348 | 64 | 144.40M | 26.73M | 171.12M | 514 | 448 | 1.34 |
| EuroVL-Medical-Cap | 1 | 335,523 | 332 | 150 | 111.43M | 50.40M | 161.83M | 502 | 430 | 1.33 |
| path_vqa | 1 | 32,632 | 530 | 54 | 17.30M | 1.77M | 19.07M | 766 | 518 | 1.49 |
| slake | 1 | 7,033 | 519 | 49 | 3.65M | 346,227 | 4.00M | 607 | 607 | 1.00 |
| vqa_rad | 1 | 2,244 | 720 | 53 | 1.62M | 119,995 | 1.74M | 770 | 776 | 1.02 |

## ocr

49 dataset(s), 13.72B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| chartnet_csv | 1 | 2,513,168 | 979 | 481 | 2.46B | 1.21B | 3.67B | 1744 | 1185 | 1.49 |
| chartmoe_chart2code | 1 | 898,609 | 823 | 838 | 739.40M | 753.24M | 1.49B | 1064 | 578 | 1.82 |
| chartmoe_chart2json | 1 | 898,609 | 823 | 452 | 739.40M | 405.72M | 1.15B | 1064 | 578 | 1.82 |
| wkvvqa | 1 | 522,944 | 1060 | 613 | 554.25M | 320.45M | 874.70M | 1289 | 1720 | 0.75 |
| chartmoe_chart2table | 1 | 898,609 | 823 | 128 | 739.40M | 114.80M | 854.20M | 1064 | 578 | 1.82 |
| nemotron_ocr | 1 | 439,192 | 1020 | 882 | 447.91M | 387.34M | 835.25M | 1034 | 988 | 1.11 |
| synthtabnet | 1 | 600,369 | 228 | 805 | 136.96M | 483.29M | 620.25M | 481 | 354 | 1.74 |
| SynthCodeNet | 1 | 499,914 | 621 | 571 | 310.28M | 285.34M | 595.62M | 638 | 886 | 1.27 |
| synthdog | 1 | 500,000 | 1047 | 124 | 523.58M | 61.91M | 585.49M | 1090 | 1090 | 1.10 |
| nemotron_ocr9 | 1 | 224,170 | 989 | 1231 | 221.76M | 276.04M | 497.80M | 761 | 998 | 0.76 |
| olmocr_mix | 1 | 261,983 | 1063 | 758 | 278.59M | 198.55M | 477.14M | 1260 | 1611 | 0.80 |
| vcr_wiki_en_hard | 1 | 1,268,328 | 154 | 65 | 195.71M | 82.61M | 278.32M | 300 | 375 | 0.86 |
| synthdog_multilingual_eu | 1 | 205,000 | 1047 | 177 | 214.65M | 36.33M | 250.98M | 1091 | 1089 | 1.10 |
| ureader_qa | 1 | 252,953 | 911 | 48 | 230.42M | 12.12M | 242.55M | 1200 | 1094 | 1.65 |
| mathwriting-google | 1 | 300,000 | 712 | 42 | 213.57M | 12.58M | 226.16M | 1511 | 432 | 4.74 |
| nvidia_ocr_synth_enru | 1 | 100,000 | 867 | 719 | 86.74M | 71.94M | 158.68M | 864 | 864 | 1.03 |
| doclaynet | 1 | 68,700 | 1089 | 775 | 74.81M | 53.25M | 128.06M | 1025 | 1025 | 1.00 |
| vcr_wiki_en_easy | 1 | 500,000 | 154 | 65 | 77.13M | 32.57M | 109.69M | 300 | 375 | 0.86 |
| pangea_webui_ocr | 1 | 90,000 | 1064 | 110 | 95.72M | 9.90M | 105.62M | 1280 | 1304 | 1.08 |
| nemotron_ocr6 | 1 | 48,309 | 1089 | 636 | 52.61M | 30.75M | 83.35M | 1025 | 1025 | 1.00 |
| SynthFormulaNet | 1 | 499,997 | 66 | 100 | 33.06M | 50.23M | 83.28M | 342 | 79 | 4.26 |
| ocr-vqa | 1 | 288,797 | 231 | 41 | 66.71M | 11.80M | 78.51M | 355 | 489 | 0.73 |
| latexformulas | 1 | 552,340 | 36 | 98 | 19.66M | 54.21M | 73.87M | 326 | 65 | 5.69 |
| textvqa | 1 | 31,728 | 971 | 37 | 30.80M | 1.17M | 31.97M | 947 | 818 | 1.22 |
| latex_handwritten | 1 | 39,583 | 663 | 77 | 26.25M | 3.03M | 29.28M | 1381 | 368 | 3.96 |
| infovqa | 1 | 23,946 | 1006 | 39 | 24.10M | 943,376 | 25.04M | 1196 | 2627 | 0.67 |
| textocr | 1 | 21,571 | 971 | 149 | 20.95M | 3.21M | 24.17M | 948 | 818 | 1.22 |
| nemotron_ocr7 | 1 | 25,281 | 476 | 363 | 12.04M | 9.17M | 21.21M | 699 | 550 | 1.44 |
| hw_squad | 1 | 20,464 | 889 | 126 | 18.19M | 2.59M | 20.78M | 918 | 997 | 1.12 |
| gemini_textvqa_filtered | 1 | 15,690 | 972 | 80 | 15.24M | 1.25M | 16.49M | 949 | 817 | 1.23 |
| sujet_finance | 1 | 9,801 | 999 | 488 | 9.79M | 4.78M | 14.57M | 813 | 957 | 0.87 |
| rendered_text | 1 | 9,969 | 1089 | 49 | 10.86M | 493,297 | 11.35M | 1024 | 1024 | 1.00 |
| llavar | 1 | 33,629 | 262 | 64 | 8.80M | 2.14M | 10.94M | 428 | 467 | 0.96 |
| pdfvqa | 1 | 9,279 | 634 | 372 | 5.88M | 3.45M | 9.33M | 595 | 793 | 0.75 |
| hme100k | 1 | 74,492 | 35 | 55 | 2.64M | 4.13M | 6.77M | 272 | 66 | 4.41 |
| st_vqa | 1 | 17,247 | 291 | 46 | 5.03M | 794,689 | 5.82M | 484 | 414 | 1.19 |
| captcha | 1 | 113,062 | 12 | 32 | 1.36M | 3.62M | 4.98M | 150 | 40 | 3.75 |
| coco-text | 1 | 9,982 | 373 | 58 | 3.72M | 578,913 | 4.30M | 586 | 482 | 1.27 |
| visualmrc | 1 | 3,035 | 1061 | 149 | 3.22M | 452,133 | 3.67M | 911 | 1963 | 0.48 |
| pangea_mtvqa | 1 | 3,035 | 1054 | 116 | 3.20M | 350,854 | 3.55M | 2365 | 2354 | 1.10 |
| wordart | 1 | 19,008 | 160 | 25 | 3.04M | 475,435 | 3.52M | 418 | 214 | 2.21 |
| bentham | 1 | 10,843 | 269 | 37 | 2.92M | 402,840 | 3.32M | 1476 | 124 | 12.27 |
| sroie | 1 | 33,616 | 16 | 41 | 549,824 | 1.39M | 1.94M | 207 | 36 | 5.71 |
| cc_ocr_multi_lan | 1 | 1,498 | 864 | 132 | 1.29M | 197,139 | 1.49M | 1702 | 1671 | 1.03 |
| ctw1500 | 1 | 8,060 | 95 | 28 | 763,951 | 221,997 | 985,948 | 359 | 128 | 4.60 |
| chrome_writting | 1 | 8,825 | 53 | 51 | 466,712 | 453,874 | 920,586 | 314 | 104 | 3.07 |
| poie | 1 | 482 | 396 | 107 | 190,754 | 51,455 | 242,209 | 544 | 632 | 1.03 |
| iiit5k | 1 | 1,990 | 14 | 36 | 28,284 | 72,178 | 100,462 | 111 | 44 | 2.62 |
| handwritten_math_expr | 1 | 59 | 121 | 38 | 7,118 | 2,268 | 9,386 | 514 | 150 | 3.58 |

## pointing

8 dataset(s), 2.63B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| objects365_point | 1 | 2,675,114 | 547 | 48 | 1.46B | 129.43M | 1.59B | 753 | 612 | 1.27 |
| openimages_point | 1 | 870,779 | 968 | 47 | 842.63M | 40.68M | 883.31M | 957 | 800 | 1.26 |
| refcoco_point | 1 | 136,810 | 379 | 46 | 51.79M | 6.34M | 58.14M | 592 | 483 | 1.28 |
| molmopoint_guisyn | 1 | 36,960 | 919 | 472 | 33.96M | 17.44M | 51.39M | 1580 | 1072 | 1.44 |
| pixmo_count | 1 | 33,428 | 943 | 74 | 31.51M | 2.47M | 33.98M | 1299 | 1033 | 1.30 |
| groundui_point | 1 | 5,480 | 1067 | 44 | 5.85M | 241,919 | 6.09M | 1412 | 1097 | 1.47 |
| taco_point | 1 | 3,123 | 1057 | 49 | 3.30M | 152,533 | 3.45M | 2947 | 3078 | 1.02 |
| pointarena | 1 | 951 | 852 | 43 | 810,293 | 41,089 | 851,382 | 1569 | 1211 | 1.38 |

## science

11 dataset(s), 2.80B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| leopard_arxiv_enriched_translated | 1 | 2,068,959 | 1032 | 217 | 2.13B | 448.69M | 2.58B | 1803 | 1273 | 1.72 |
| leopard_arxiv_enriched | 1 | 144,567 | 1032 | 163 | 149.18M | 23.53M | 172.71M | 1802 | 1273 | 1.71 |
| verisciqa | 1 | 10,000 | 1015 | 100 | 10.15M | 997,981 | 11.15M | 1788 | 1302 | 1.61 |
| cosyn_circuit | 1 | 10,470 | 572 | 423 | 5.99M | 4.43M | 10.42M | 881 | 490 | 1.86 |
| cosyn_chemical | 1 | 8,942 | 517 | 442 | 4.62M | 3.96M | 8.58M | 760 | 537 | 1.47 |
| ai2d_merged | 1 | 4,866 | 401 | 489 | 1.95M | 2.38M | 4.33M | 611 | 483 | 1.40 |
| r1_vision_ai2d | 1 | 7,791 | 388 | 75 | 3.02M | 583,000 | 3.61M | 600 | 470 | 1.42 |
| pathvqa | 1 | 4,301 | 500 | 252 | 2.15M | 1.09M | 3.23M | 733 | 502 | 1.46 |
| scienceqa_nona_context | 1 | 5,078 | 259 | 138 | 1.31M | 702,490 | 2.02M | 513 | 359 | 1.94 |
| kaleidoscope | 1 | 5,395 | 239 | 120 | 1.29M | 649,646 | 1.94M | 469 | 297 | 1.99 |
| r1_vision_scienceqa | 1 | 758 | 223 | 86 | 168,898 | 64,948 | 233,846 | 483 | 301 | 1.84 |

## spatial

3 dataset(s), 13.66M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| omnispatial | 1 | 6,693 | 709 | 99 | 4.75M | 665,090 | 5.41M | 1386 | 994 | 1.58 |
| tetris_analogy | 1 | 7,935 | 528 | 91 | 4.19M | 719,560 | 4.91M | 616 | 668 | 0.92 |
| dise_singleimage | 1 | 4,479 | 693 | 52 | 3.10M | 232,041 | 3.34M | 920 | 568 | 1.62 |

# multiimage

46 dataset(s), 9,725,929 samples, **29.87B tokens (r=1)**, **29.87B effective (r×tokens)**.

## 3d_grounding

1 dataset(s), 167.39M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| hypersim_3d_ground_mv | 1 | 50,083 | 3108 | 234 | 155.65M | 11.74M | 167.39M | 1024 | 768 | 1.33 |

## captioning

2 dataset(s), 3.27B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| multi_synth_cc12m_cc3m_grit | 1 | 1,906,820 | 1435 | 199 | 2.74B | 378.77M | 3.12B | 675 | 562 | 1.27 |
| internvl_multi_en | 1 | 77,704 | 1543 | 419 | 119.92M | 32.53M | 152.45M | 722 | 557 | 1.35 |

## chart

9 dataset(s), 3.10B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| molmo2_chart_translated | 1 | 379,258 | 4843 | 918 | 1.84B | 348.26M | 2.18B | 1455 | 1032 | 1.48 |
| molmo2_table | 1 | 33,980 | 7659 | 760 | 260.24M | 25.82M | 286.06M | 1575 | 1030 | 1.71 |
| molmo2_chart | 1 | 46,309 | 4847 | 716 | 224.48M | 33.14M | 257.62M | 1454 | 1031 | 1.48 |
| molmo2_diagram | 1 | 21,047 | 4979 | 760 | 104.79M | 15.99M | 120.77M | 1449 | 1302 | 1.54 |
| leopard_chartgemma | 1 | 48,972 | 1576 | 321 | 77.19M | 15.72M | 92.91M | 701 | 559 | 1.29 |
| leopard_figureqa | 1 | 35,990 | 644 | 752 | 23.19M | 27.07M | 50.26M | 589 | 400 | 1.47 |
| leopard_arxiv | 1 | 15,806 | 2754 | 387 | 43.52M | 6.11M | 49.63M | 1798 | 1296 | 1.68 |
| molmo2_graphic | 1 | 11,564 | 2343 | 707 | 27.10M | 8.17M | 35.27M | 718 | 532 | 1.45 |
| leopard_multihiertt | 1 | 14,710 | 1392 | 198 | 20.48M | 2.91M | 23.39M | 697 | 380 | 2.95 |

## counting

1 dataset(s), 1.99B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| mi_oiv7 | 1 | 500,000 | 3850 | 124 | 1.93B | 62.19M | 1.99B | 961 | 802 | 1.26 |

## doc

6 dataset(s), 6.74B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| molmo2_doc_translated | 1 | 391,246 | 7027 | 606 | 2.75B | 236.95M | 2.99B | 1601 | 1793 | 0.99 |
| docmatix | 1 | 716,889 | 2890 | 564 | 2.07B | 404.04M | 2.48B | 1344 | 1653 | 0.83 |
| leopard_mpdocvqa | 1 | 43,778 | 11314 | 144 | 495.31M | 6.32M | 501.63M | 1811 | 2148 | 0.88 |
| molmo2_doc | 1 | 54,717 | 7053 | 501 | 385.92M | 27.43M | 413.35M | 1600 | 1791 | 0.99 |
| leopard_dude | 1 | 34,500 | 7943 | 75 | 274.04M | 2.59M | 276.63M | 1727 | 2168 | 0.81 |
| leopard_monkey | 1 | 62,074 | 1233 | 180 | 76.54M | 11.16M | 87.69M | 656 | 764 | 1.05 |

## general_qa

10 dataset(s), 10.31B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| doclingmatix | 1 | 1,224,738 | 2090 | 2813 | 2.56B | 3.45B | 6.01B | 1359 | 1675 | 0.83 |
| multi_synth_cc12m_cc3m_grit | 1 | 1,496,835 | 1436 | 548 | 2.15B | 819.70M | 2.97B | 675 | 562 | 1.27 |
| molmo2_multiimageqa_translated | 1 | 532,549 | 1861 | 305 | 990.89M | 162.57M | 1.15B | 982 | 883 | 1.21 |
| nlvr2 | 1 | 50,426 | 1097 | 126 | 55.33M | 6.35M | 61.68M | 771 | 627 | 1.28 |
| molmo2_multiimageqa | 1 | 27,846 | 1884 | 316 | 52.47M | 8.80M | 61.28M | 982 | 884 | 1.20 |
| img_diff | 1 | 18,461 | 1723 | 97 | 31.80M | 1.79M | 33.59M | 864 | 864 | 1.00 |
| mimic_cgd | 1 | 70,939 | 128 | 118 | 9.08M | 8.39M | 17.47M | 224 | 224 | 1.00 |
| mp_docvqa | 1 | 943 | 5633 | 171 | 5.31M | 160,784 | 5.47M | 1807 | 2109 | 0.90 |
| mirb | 1 | 1,323 | 1760 | 71 | 2.33M | 94,244 | 2.42M | 973 | 815 | 1.36 |
| spot_the_diff | 1 | 8,514 | 128 | 54 | 1.09M | 460,765 | 1.55M | 224 | 224 | 1.00 |

## grounding

1 dataset(s), 1.12B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| mi_grounding | 1 | 500,000 | 1970 | 278 | 984.97M | 139.18M | 1.12B | 693 | 556 | 1.29 |

## gui

5 dataset(s), 903.40M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| mi_gui_next_action | 1 | 136,860 | 3329 | 76 | 455.60M | 10.35M | 465.95M | 1017 | 1923 | 0.54 |
| mi_gui_before_after | 1 | 153,833 | 1874 | 50 | 288.33M | 7.68M | 296.01M | 1001 | 1938 | 0.53 |
| mi_gui_step_order | 1 | 20,967 | 2803 | 60 | 58.78M | 1.26M | 60.04M | 965 | 1973 | 0.50 |
| leopard_rico | 1 | 18,743 | 2864 | 123 | 53.68M | 2.31M | 55.99M | 958 | 1702 | 0.56 |
| leopard_mind2web | 1 | 5,597 | 4352 | 189 | 24.36M | 1.06M | 25.42M | 1287 | 1434 | 0.90 |

## knowledge

1 dataset(s), 9.89M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| molmo2_syn_music | 1 | 4,785 | 1596 | 471 | 7.63M | 2.26M | 9.89M | 825 | 569 | 1.70 |

## math

2 dataset(s), 37.83M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| visualwebinstruct | 1 | 21,937 | 947 | 703 | 20.78M | 15.42M | 36.19M | 587 | 391 | 1.94 |
| mv_math | 1 | 2,008 | 372 | 444 | 747,297 | 890,954 | 1.64M | 314 | 263 | 1.27 |

## pointing

5 dataset(s), 2.19B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| mi_o365 | 1 | 500,000 | 2097 | 123 | 1.05B | 61.71M | 1.11B | 726 | 585 | 1.29 |
| molmo2_multiimagepoint | 1 | 312,183 | 2271 | 375 | 708.85M | 116.96M | 825.81M | 1011 | 874 | 1.27 |
| mi_refcoco | 1 | 150,000 | 1516 | 70 | 227.47M | 10.45M | 237.92M | 593 | 483 | 1.28 |
| mi_pixmo | 1 | 3,602 | 3489 | 216 | 12.57M | 777,522 | 13.35M | 1286 | 1055 | 1.27 |
| mi_taco | 1 | 105 | 4060 | 129 | 426,283 | 13,554 | 439,837 | 2747 | 2892 | 1.00 |

## science

2 dataset(s), 21.38M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| molmo2_syn_circuit | 1 | 3,416 | 3075 | 569 | 10.50M | 1.94M | 12.45M | 1187 | 911 | 1.52 |
| molmo2_syn_chemical | 1 | 4,872 | 1326 | 508 | 6.46M | 2.48M | 8.94M | 433 | 346 | 1.26 |

## spatial

1 dataset(s), 12.96M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| dise_multiimage | 1 | 9,000 | 1382 | 58 | 12.44M | 518,400 | 12.96M | 448 | 448 | 1.00 |

# video

62 dataset(s), 5,376,790 samples, **32.92B tokens (r=1)**, **32.92B effective (r×tokens)**.

## captioning

13 dataset(s), 6.92B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| eurovideolm_packed (696,403 message trees) | 1 | 696,403 | 5886 | 1066 | 4.10B | 742.62M | 4.84B | - | - | - |
| llava_video_178k | 1 | 176,584 | 5507 | 989 | 972.39M | 174.69M | 1.15B | - | - | - |
| molmo2_cap_clips | 1 | 96,395 | 5691 | 1234 | 548.54M | 118.98M | 667.52M | - | - | - |
| vatex | 1 | 34,245 | 3655 | 235 | 125.16M | 8.04M | 133.20M | - | - | - |
| vln | 1 | 8,345 | 4461 | 306 | 37.23M | 2.55M | 39.78M | - | - | - |
| videogpt_plus_caption | 1 | 6,884 | 3335 | 328 | 22.96M | 2.26M | 25.21M | - | - | - |
| medvidqa_recap_packed | 1 | 2,885 | 6198 | 476 | 17.88M | 1.37M | 19.25M | - | - | - |
| didemo | 1 | 2,167 | 6114 | 608 | 13.25M | 1.32M | 14.57M | - | - | - |
| activitynet_new | 1 | 1,444 | 5213 | 745 | 7.53M | 1.08M | 8.60M | - | - | - |
| vdc | 1 | 1,027 | 6550 | 953 | 6.73M | 978,353 | 7.71M | - | - | - |
| nextsceneqa_train | 1 | 918 | 6562 | 728 | 6.02M | 667,857 | 6.69M | - | - | - |
| dream1k | 1 | 1,000 | 6010 | 305 | 6.01M | 305,039 | 6.31M | - | - | - |
| video_detail_caption | 1 | 457 | 5958 | 669 | 2.72M | 305,657 | 3.03M | - | - | - |

## embodied_reasoning

3 dataset(s), 301.27M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| alfred_multilingual | 1 | 22,632 | 5733 | 656 | 129.74M | 14.85M | 144.59M | - | - | - |
| alfred | 1 | 22,665 | 5734 | 618 | 129.95M | 14.01M | 143.97M | - | - | - |
| perception_test_cot | 1 | 1,794 | 6524 | 563 | 11.70M | 1.01M | 12.71M | - | - | - |

## general_qa

13 dataset(s), 12.40B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| eurovideolm_qa_packed (696,336 message trees) | 1 | 696,336 | 5886 | 1704 | 4.10B | 1.19B | 5.29B | - | - | - |
| llava_video_178k_qa_packed (406,431 message trees) | 1 | 406,431 | 5557 | 999 | 2.26B | 405.88M | 2.66B | - | - | - |
| kinetics_k710_packed | 1 | 503,270 | 3259 | 111 | 1.64B | 56.01M | 1.70B | - | - | - |
| molmo2_capqa_packed (223,239 message trees) | 1 | 223,239 | 6034 | 695 | 1.35B | 155.23M | 1.50B | - | - | - |
| videogpt_plus_qa_packed (237,893 message trees) | 1 | 237,893 | 2991 | 396 | 711.65M | 94.23M | 805.88M | - | - | - |
| molmo2_askmodelanything_packed (43,268 message trees) | 1 | 43,268 | 6134 | 705 | 265.40M | 30.51M | 295.91M | - | - | - |
| seeker_llava_qa_packed (11,365 message trees) | 1 | 11,365 | 5838 | 1388 | 66.35M | 15.78M | 82.12M | - | - | - |
| videovista_packed (3,670 message trees) | 1 | 3,670 | 5931 | 1176 | 21.77M | 4.31M | 26.08M | - | - | - |
| perception_test_1 (1,955 message trees) | 1 | 1,955 | 6525 | 472 | 12.76M | 923,251 | 13.68M | - | - | - |
| clevrer_packed (9,930 message trees) | 1 | 9,930 | 935 | 285 | 9.28M | 2.83M | 12.11M | - | - | - |
| nextsceneqa_genqa (1,088 message trees) | 1 | 1,088 | 6671 | 1108 | 7.26M | 1.21M | 8.46M | - | - | - |
| seeker_nextgqa_packed (568 message trees) | 1 | 568 | 6023 | 682 | 3.42M | 387,614 | 3.81M | - | - | - |
| egoschema | 1 | 500 | 6726 | 824 | 3.36M | 411,751 | 3.77M | - | - | - |

## grounding

10 dataset(s), 5.07B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| molmo2_videopoint_packed (278,605 message trees) | 1 | 278,605 | 5899 | 647 | 1.64B | 180.38M | 1.82B | - | - | - |
| sav_tracking (117,070 message trees) | 1 | 117,070 | 6582 | 4196 | 770.55M | 491.25M | 1.26B | - | - | - |
| sav_grounding (117,849 message trees) | 1 | 117,849 | 6582 | 322 | 775.70M | 37.98M | 813.68M | - | - | - |
| sav_pointing (117,849 message trees) | 1 | 117,849 | 6582 | 296 | 775.70M | 34.83M | 810.53M | - | - | - |
| molmopoint_trackany_packed_v2 (11,387 message trees) | 1 | 11,387 | 6305 | 8567 | 71.80M | 97.56M | 169.36M | - | - | - |
| molmo2_videotrack_packed (1,683 message trees) | 1 | 1,683 | 5472 | 56513 | 9.21M | 95.11M | 104.32M | - | - | - |
| vln_pointing_uvo | 1 | 6,567 | 4407 | 131 | 28.94M | 861,113 | 29.80M | - | - | - |
| ava_tracking (1,494 message trees) | 1 | 1,494 | 6284 | 13566 | 9.39M | 20.27M | 29.66M | - | - | - |
| ava_grounding (1,494 message trees) | 1 | 1,494 | 6284 | 2739 | 9.39M | 4.09M | 13.48M | - | - | - |
| ava_pointing (1,494 message trees) | 1 | 1,494 | 6284 | 2490 | 9.39M | 3.72M | 13.11M | - | - | - |

## spatial

2 dataset(s), 472.58M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| zechen_clevrer_multilingual | 1 | 228,939 | 935 | 113 | 214.06M | 25.96M | 240.02M | - | - | - |
| zechen_clevrer_en | 1 | 224,490 | 935 | 101 | 209.90M | 22.67M | 232.56M | - | - | - |

## subtitles

1 dataset(s), 695.85M tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| molmo2_subtitleqa_packed (102,937 message trees) | 1 | 102,937 | 5975 | 785 | 615.08M | 80.76M | 695.85M | - | - | - |

## temporal_grounding

20 dataset(s), 7.06B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| eurovideolm_temporal_merged_packed (734,754 message trees) | 1 | 734,754 | 5911 | 1653 | 4.34B | 1.21B | 5.56B | - | - | - |
| sav_temporal_grounding (117,070 message trees) | 1 | 117,070 | 6582 | 265 | 770.55M | 31.03M | 801.58M | - | - | - |
| ego4d_fho_sta_packed (28,449 message trees) | 1 | 28,449 | 6863 | 1082 | 195.25M | 30.78M | 226.02M | - | - | - |
| ego4d_narrations (14,282 message trees) | 1 | 14,282 | 6735 | 1252 | 96.19M | 17.88M | 114.07M | - | - | - |
| hacs_packed | 1 | 11,641 | 5818 | 1563 | 67.73M | 18.20M | 85.93M | - | - | - |
| vln_temporal_grounding_uvo | 1 | 12,538 | 4414 | 159 | 55.34M | 1.99M | 57.33M | - | - | - |
| activity_net_2 | 1 | 5,909 | 6003 | 694 | 35.47M | 4.10M | 39.58M | - | - | - |
| activity_net_1 | 1 | 4,500 | 5788 | 1500 | 26.05M | 6.75M | 32.80M | - | - | - |
| tapos_packed (3,657 message trees) | 1 | 3,657 | 6129 | 1153 | 22.42M | 4.22M | 26.63M | - | - | - |
| ego4d_mq_packed (2,628 message trees) | 1 | 2,628 | 6528 | 987 | 17.16M | 2.59M | 19.75M | - | - | - |
| ego4d_fho_lta_packed (2,374 message trees) | 1 | 2,374 | 6827 | 998 | 16.21M | 2.37M | 18.58M | - | - | - |
| crosstask_v2_packed | 1 | 2,301 | 5929 | 869 | 13.64M | 2.00M | 15.64M | - | - | - |
| perception_test_2 | 1 | 1,989 | 6523 | 406 | 12.97M | 808,468 | 13.78M | - | - | - |
| ava_temporal_grounding (1,493 message trees) | 1 | 1,493 | 6287 | 2255 | 9.39M | 3.37M | 12.75M | - | - | - |
| hc_stvg_packed | 1 | 1,747 | 5966 | 256 | 10.42M | 447,848 | 10.87M | - | - | - |
| youcook2_2_fixed (944 message trees) | 1 | 944 | 6041 | 1163 | 5.70M | 1.10M | 6.80M | - | - | - |
| breakfast_actions_packed | 1 | 1,207 | 4869 | 608 | 5.88M | 734,231 | 6.61M | - | - | - |
| youcook2_1 | 1 | 844 | 5940 | 916 | 5.01M | 772,946 | 5.79M | - | - | - |
| hirest_1_packed | 1 | 755 | 5995 | 669 | 4.53M | 504,897 | 5.03M | - | - | - |
| hirest_2_packed | 1 | 492 | 5959 | 849 | 2.93M | 417,717 | 3.35M | - | - | - |

# text

4 dataset(s), 10,648,196 samples, **10.46B tokens (r=1)**, **10.46B effective (r×tokens)**.

## chat

1 dataset(s), 3.68B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| euroblocks | 1 | 3,937,952 | 0 | 933 | 0 | 3.68B | 3.68B | - | - | - |

## code

1 dataset(s), 3.44B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| euroblocks | 1 | 2,071,395 | 0 | 1661 | 0 | 3.44B | 3.44B | - | - | - |

## math

1 dataset(s), 1.82B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| euroblocks | 1 | 2,283,874 | 0 | 797 | 0 | 1.82B | 1.82B | - | - | - |

## stem

1 dataset(s), 1.53B tokens.

| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| euroblocks | 1 | 2,354,975 | 0 | 649 | 0 | 1.53B | 1.53B | - | - | - |
