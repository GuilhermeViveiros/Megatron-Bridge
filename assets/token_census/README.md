# PA data mixture — exact token census

Exact (not sampled) text+vision token counts per energon dataset, computed by
`assets/token_census/scripts/estimate_token_budget.py`. Every sample of every shard is counted — no
extrapolation. Vision tokens are computed analytically (image/video headers only, no pixel
decode) using the same math the real MoonViT processors use; validated at 0/60 mismatches
against the real end-to-end `EuroVLProcessor` output (image, multiimage, video, including the
real per-frame timestamp text) before running at scale — see
`assets/token_census/scripts/validate_token_estimates.py`.

Regenerate: `./apptainer.sh uv run --no-sync python assets/token_census/scripts/estimate_token_budget.py --categories <cats> --workers 16`
(use 16, not 64 — 64 concurrent worker imports exhausted file descriptors on this filesystem).

Under `training/`: a target PA token budget of 40-67B (Nemotron ~40B, Qwen3-VL ~67B) needs
`train_iters * global_batch_size * seq_length` tokens under energon fill-to-`seq_length` packing;
repeat factors (`r` in `mixture_pa.yaml`) scale each dataset's contribution to that total.

## Overall total (all 275 datasets, r=1) — as of 2026-09-01

| category | tokens (r=1) |
|---|---|
| remaining (chart/code/counting/general_qa/grounding/gui/math/medical/pointing/spatial/science/3d_grounding) | 66.84B |
| captioning + knowledge | 18.11B |
| ocr | 13.81B |
| doc | 3.91B |
| **TOTAL** | **102.68B** |

Already well above the 40-67B target at r=1 (no repeats needed) once every category and every
known drift/bug is accounted for — the opposite problem from where this census started (13.5B,
needing r=3-5 on a narrow blend). The decision now is which subset/repeat-factors to actually put
in `mixture_pa.yaml`, not whether there's enough data. `code` and `chart` alone are 40.15B (39% of
the total) — worth deciding deliberately whether to include them at full weight, since they're a
very different data type (code screenshots / chart images) from a typical VLM alignment blend.

## captioning + knowledge (all modalities) — run 2026-08-17, re-run 2026-08-31 (drift + image-size stats)

25 datasets on disk, all 25 successfully censused. The 2026-08-31 re-check caught real drift
against the 2026-08-17 run (`discover_datasets` vs. what was censused): **3 large new datasets**
had been added to the `knowledge` category mid-session — `wit` (12.9M samples),
`culturalground_oe` (5.0M), `culturalground_mcq` (3.5M) — contributing **4.55B tokens combined**,
none of which were in the original 21-dataset/13.545B table. A 25th, `molmo2_syn_music`
(multiimage, 4,785 samples), initially failed 100% of samples — its JSON is doubly-nested
(`[[{...turns...}]]`) unlike every other dataset's flat `[{...}, {...}]`, which
`_split_conversation` didn't handle, caught silently by `process_shard`'s broad exception handler.
Same root cause found in ~15 datasets in the remaining-categories census (see below) — fixed
2026-09-01 in `process_shard` (unwrap a single-element list whose one element is itself a list)
and re-run; now 4,785/4,785 succeed, contributing 9.85M tokens.

This run also adds exhaustive (not sampled) average image width/height/aspect-ratio per dataset —
tracked directly in `process_shard` (`sum_w`/`sum_h`/`sum_ratio`/`n_images`), image + multiimage
modalities only (video frames aren't a single "image size" in the same sense).

| dataset | modality | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|---|
| cc12m | image | 9,794,682 | 446 | 109 | 4.372B | 1.065B | 624 | 536 | 1.21 |
| grit | image | 8,507,801 | 503 | 116 | 4.283B | 0.986B | 728 | 584 | 1.34 |
| wit | image | 12,916,713 | 116 | 97 | 1.503B | 1.248B | 300 | 278 | 1.25 |
| culturalground_oe | image | 5,044,547 | 148 | 57 | 0.745B | 0.286B | 356 | 312 | 1.21 |
| cc3m | image | 1,548,826 | 424 | 94 | 0.657B | 0.146B | 619 | 517 | 1.24 |
| pixmo-cap | image | 702,205 | 850 | 258 | 0.597B | 0.181B | 1435 | 1374 | 1.20 |
| culturalground_mcq | image | 3,498,781 | 148 | 72 | 0.519B | 0.253B | 359 | 311 | 1.23 |
| molmo2_cap | video | 91,162 | 6077 | 1356 | 0.554B | 0.124B | - | - | - |
| internvl_multi_en | multiimage | 77,704 | 1543 | 418 | 0.120B | 0.032B | 722 | 557 | 1.35 |
| sbu-captions | image | 499,580 | 100 | 92 | 0.050B | 0.046B | 256 | 256 | 1.00 |
| finevision_ureader_cap | image | 87,762 | 971 | 41 | 0.085B | 0.004B | 949 | 815 | 1.23 |
| visual_genome | image | 132,760 | 251 | 240 | 0.033B | 0.032B | 478 | 397 | 1.27 |
| localized_narratives | image | 118,272 | 369 | 79 | 0.044B | 0.009B | 578 | 484 | 1.25 |
| sharegpt4o | image | 42,636 | 572 | 137 | 0.024B | 0.006B | 857 | 775 | 1.27 |
| textcaps | image | 21,942 | 971 | 39 | 0.021B | 0.001B | 949 | 817 | 1.23 |
| coco-caption | image | 39,621 | 369 | 94 | 0.015B | 0.004B | 576 | 485 | 1.25 |
| cosyn_music | image | 11,969 | 1000 | 427 | 0.012B | 0.005B | 820 | 1008 | 0.88 |
| mminstruct_caption_en | image | 17,512 | 606 | 244 | 0.011B | 0.004B | 907 | 723 | 1.41 |
| flickr30k | image | 30,000 | 235 | 96 | 0.007B | 0.003B | 460 | 395 | 1.23 |
| a-okvqa | image | 17,315 | 374 | 37 | 0.006B | 0.001B | 587 | 482 | 1.28 |
| okvqa | image | 8,998 | 372 | 36 | 0.003B | 0.000B | 618 | 448 | 1.40 |
| viquae | image | 2,385 | 373 | 47 | 0.001B | 0.000B | 512 | 536 | 1.12 |
| web_landmark | image | 500 | 860 | 200 | 0.000B | 0.000B | 1356 | 943 | 1.50 |
| web_celebrity | image | 495 | 778 | 158 | 0.000B | 0.000B | 1196 | 738 | 1.67 |
| molmo2_syn_music | multiimage | 4,785 | 1596 | 462 | 0.008B | 0.002B | 825 | 569 | 1.70 |

**Grand total (r=1, one epoch each): 18,109,224,806 tokens** — up from 13,545,154,674 before the
drift fix (+34%), now much closer to the 40-67B target without needing large repeat factors.
`cc12m` + `grit` + `wit` are now ~78% of the pool at r=1.

Raw per-shard data: `assets/token_census/results/captioning_knowledge/<modality>__<category>__<name>/*.json`,
combined: `assets/token_census/results/captioning_knowledge/_combined_summary.json`.

## ocr (all modalities) — run 2026-08-17, re-run 2026-08-31 (image-size stats added, no drift: 48/48 on-disk datasets match)

48 datasets, 12,987,566 samples, 0 failures.

Top 15 by total tokens (full 48-row table in `assets/token_census/results/ocr/_combined_summary.json`).
`avg w`/`avg h`/`ratio` are exhaustive (every image, not sampled) — see the captioning+knowledge
section above for how they're computed:

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| chartnet_csv | 2,513,168 | 979 | 481 | 2.46B | 1.21B | 1744 | 1185 | 1.49 |
| chartmoe_chart2code | 898,609 | 823 | 838 | 0.74B | 0.75B | 1064 | 578 | 1.82 |
| chartmoe_chart2json | 898,609 | 823 | 451 | 0.74B | 0.40B | 1064 | 578 | 1.82 |
| wkvvqa | 522,963 | 1060 | 596 | 0.55B | 0.31B | 1289 | 1720 | 0.75 |
| chartmoe_chart2table | 898,609 | 823 | 127 | 0.74B | 0.11B | 1064 | 578 | 1.82 |
| nemotron_ocr | 438,847 | 1020 | 881 | 0.45B | 0.39B | 1034 | 988 | 1.11 |
| synthtabnet | 600,369 | 228 | 804 | 0.14B | 0.48B | 481 | 354 | 1.74 |
| SynthCodeNet | 499,908 | 621 | 570 | 0.31B | 0.28B | 638 | 886 | 1.27 |
| synthdog | 500,000 | 1047 | 123 | 0.52B | 0.06B | 1090 | 1090 | 1.10 |
| nemotron_ocr9 | 224,170 | 989 | 1231 | 0.22B | 0.28B | 761 | 998 | 0.76 |
| olmocr_mix | 261,982 | 1063 | 757 | 0.28B | 0.20B | 1260 | 1611 | 0.80 |
| vcr_wiki_en_hard | 1,268,328 | 154 | 65 | 0.20B | 0.08B | 300 | 375 | 0.86 |
| synthdog_multilingual_eu | 205,000 | 1047 | 176 | 0.21B | 0.04B | 1091 | 1089 | 1.10 |
| ureader_qa | 252,953 | 911 | 47 | 0.23B | 0.01B | 1200 | 1094 | 1.65 |
| mathwriting-google | 300,000 | 712 | 41 | 0.21B | 0.01B | 1511 | 432 | 4.74 |

`mathwriting-google` stands out with a 4.74:1 aspect ratio (wide handwritten-math strips).

**Grand total (r=1): 13,813,427,521 tokens** (matches the 2026-08-17 run closely; tiny sample-count
delta is normal shard-processing variance, not drift — dataset list itself is unchanged).

Notably higher avg text-token share than captioning+knowledge (36.8% vs 19.6%) — chart-to-code/table/json
transcription tasks (`chartmoe_*`, `synthtabnet`) produce long structured-text targets, not short captions.
Avg tokens/sample (~1063) is ~1.7x captioning+knowledge's (~622), not the ~26x gap Qwen3/Nemotron's stated
sample-count-vs-token-count numbers implied — consistent with their "36B/67B tokens" figures being total
tokens *consumed over training* (multiple epochs via repeat factors), not raw single-pass corpus size,
same accounting this file uses (`r=1` totals here, scaled by repeat factors to hit the target).

Raw per-shard data: `assets/token_census/results/ocr/<modality>__<category>__<name>/*.json`,
combined: `assets/token_census/results/ocr/_combined_summary.json`.

## spatial (all modalities) — run 2026-08-17

4 datasets, 28,107 samples, 0 failures.

| dataset | modality | samples | avg tokens | total tokens |
|---|---|---|---|---|
| dise_multiimage | multiimage | 9,000 | 1439 | 12,951,000 |
| omnispatial | image | 6,693 | 808 | 5,409,541 |
| tetris_analogy | image | 7,935 | 618 | 4,901,305 |
| dise_singleimage | image | 4,479 | 744 | 3,331,509 |

**Grand total (r=1): 26,593,355 tokens.**

Raw: `assets/token_census/results/spatial/`.

## science (all modalities) — run 2026-08-17

7 datasets, 49,052 samples, 0 failures.

| dataset | samples | avg tokens | total tokens |
|---|---|---|---|
| verisciqa | 10,000 | 1114 | 11,138,116 |
| cosyn_circuit | 10,470 | 989 | 10,350,963 |
| cosyn_chemical | 8,942 | 953 | 8,522,104 |
| ai2d_merged | 4,866 | 889 | 4,324,218 |
| pathvqa | 4,301 | 745 | 3,205,092 |
| scienceqa_nona_context | 5,078 | 395 | 2,007,863 |
| kaleidoscope | 5,395 | 359 | 1,934,753 |

**Grand total (r=1): 41,483,109 tokens.**

Raw: `assets/token_census/results/science/`.

## 3d_grounding (all modalities) — run 2026-08-17

2 datasets, 150,086 samples, 0 failures.

| dataset | modality | samples | avg tokens | total tokens |
|---|---|---|---|---|
| hypersim_3d_ground_mv | multiimage | 50,083 | 3341 | 167,345,388 |
| hypersim_3d_ground | image | 100,003 | 1297 | 129,753,420 |

**Grand total (r=1): 297,098,808 tokens.**

Raw: `assets/token_census/results/3d_grounding/`.

## doc (all modalities) — run 2026-08-17, re-run 2026-09-01 (new dataset + image-size stats)

18 datasets, 1,166,371 samples. The 2026-08-17 run's 17-dataset/932.5M-token total was stale — a
new dataset, `molmo2_doc_translated` (391,195 samples, multiimage), had been added and wasn't
picked up until this re-check. It alone contributes 2.98B tokens, dwarfing everything else in this
category, and pushes the category total 4.2x higher (932.5M -> 3.91B). The first attempt at this
re-run (2026-08-31) hit a genuine kernel-level I/O hang (one worker stuck in uninterruptible
D-state, survived SIGKILL) partway through processing this same large dataset's very large images
(>100 megapixel `DecompressionBombWarning`s observed) -- a filesystem hiccup, not a code bug; a
clean retry completed without issue.

| dataset | samples | total tokens | avg w | avg h | ratio |
|---|---|---|---|---|---|
| molmo2_doc_translated | 391,195 | 2.98B | 1601 | 1793 | 0.99 |
| bigdocs_pubtables_1m | 345,989 | 475.3M | 614 | 526 | 1.63 |
| bigdocs_arxiv_ocr | 110,854 | 173.1M | 1680 | 2226 | 0.76 |
| docreason51k | 51,726 | 53.9M | 1603 | 1973 | 1.48 |
| bigdocs_arxiv_table_cap | 72,524 | 33.8M | 924 | 362 | 3.82 |
| bigdocs_wikitq | 22,007 | 28.5M | 918 | 646 | 1.94 |
| bigdocs_cocotext | 30,223 | 28.5M | 585 | 482 | 1.27 |
| leopard_mplugdocreason | 25,863 | 26.9M | 1603 | 1973 | 1.48 |
| bigdocs_tabfact | 16,572 | 19.5M | 835 | 437 | 2.48 |
| leopard_monkey | 31,156 | 17.2M | 651 | 754 | 1.04 |
| leopard_dude | 12,108 | 13.6M | 2013 | 2490 | 0.85 |
| sujet_finance | 9,212 | 13.6M | 813 | 957 | 0.87 |
| cauldron_docvqa | 10,177 | 12.2M | 1739 | 2089 | 0.88 |
| pangea_table_vqa | 16,408 | 11.4M | 930 | 374 | 3.76 |
| pangea_doc_vqa | 9,665 | 10.7M | 1649 | 1628 | 1.39 |
| docreason25k_refined | 8,726 | 10.3M | 1400 | 1734 | 1.33 |
| bigdocs_cord_v2 | 997 | 2.3M | 1001 | 1578 | 0.65 |
| tat_dqa | 1,969 | 1.6M | 224 | 224 | 1.00 |

`bigdocs_arxiv_table_cap` and `pangea_table_vqa` stand out with wide (~3.8:1) aspect ratios
(table images, naturally wider than tall).

**Grand total (r=1): 3,914,677,242 tokens.**

Raw: `assets/token_census/results/doc/`.

## Remaining categories — chart, code, counting, general_qa, grounding, gui, math, medical, pointing, spatial, science, 3d_grounding

184 datasets, run 2026-09-01 (includes spatial, science, and 3d_grounding, previously censused
2026-08-17 without image-size stats or the doubly-nested-JSON fix -- redone from scratch alongside
the 9 categories censused for the first time). A first pass found 60.56B tokens with ~15 datasets
(~1.66M samples) failing 100% -- same doubly-nested-JSON root cause as `molmo2_syn_music` (see
captioning+knowledge section). Fixed in `process_shard` (unwrap a single-element list whose one
element is itself a list), verified on real `pixmo_cap_qa` samples (0/20 -> 20/20 succeed), then
re-run at scale: 11/15 datasets recovered (+6.28B tokens: `doclingmatix` alone contributes 5.84B).

**4 datasets still fail 100%, each for a different, dataset-specific reason (not a script bug, not
worth bespoke fixes for ~148K samples out of tens of millions)**:
- `rsvqa_hr` (43,241 samples): image files are literally 0 bytes on disk — real data corruption,
  not fixable in this script.
- `finevision_clevr_math_mathv360k` (5,280) / `finevision_mavis_math_rule_geo` (99,986): text has
  2 `<image>` tags but only 1 image file provided — likely a templated question that references
  the image twice.
- `mirb` (150): 1 `<image>` tag but 5-6 image files per sample — a genuine one-tag-many-images
  format this study's tag-count safety check doesn't support.

By category (r=1):

| category | samples | total tokens |
|---|---|---|
| code | 9,153,290 | 28.446B |
| chart | 6,722,629 | 11.706B |
| general_qa | 3,563,473 | 8.754B |
| pointing | 4,695,107 | 5.052B |
| grounding | 5,597,022 | 4.793B |
| counting | 3,811,841 | 4.314B |
| gui | 1,712,098 | 2.657B |
| math | 709,817 | 0.380B |
| medical | 792,411 | 0.357B |
| 3d_grounding | 150,086 | 0.297B |
| science | 57,340 | 0.063B |
| spatial | 28,107 | 0.027B |

Top 25 datasets by total tokens (full 184-row table in
`assets/token_census/results/remaining_categories/_combined_summary.json`):

| dataset | modality | samples | total tokens | avg w | avg h | ratio |
|---|---|---|---|---|---|---|
| code/webcode2m_new | image | 3,166,069 | 17.376B | 1295 | 1617 | 0.90 |
| general_qa/doclingmatix | multiimage | 1,199,362 | 5.840B | 1357 | 1674 | 0.83 |
| code/chartnet_code | image | 2,513,168 | 5.399B | 1744 | 1185 | 1.49 |
| chart/chartnet_summary | image | 2,513,168 | 3.684B | 1744 | 1185 | 1.49 |
| grounding/objects365_ground | image | 4,469,839 | 3.043B | 761 | 616 | 1.28 |
| chart/molmo2_chart_translated | multiimage | 379,129 | 2.181B | 1455 | 1032 | 1.48 |
| chart/caul_plotqa | image | 1,089,485 | 2.152B | 1106 | 683 | 1.62 |
| code/websight_new | image | 1,319,321 | 2.109B | 2561 | 1656 | 1.62 |
| counting/mi_oiv7 | multiimage | 500,000 | 2.020B | 961 | 802 | 1.26 |
| pointing/objects365_point | image | 2,675,114 | 1.690B | 753 | 612 | 1.27 |
| counting/objects365_count | image | 2,676,042 | 1.676B | 753 | 612 | 1.27 |
| general_qa/molmo2_multiimageqa_translated | multiimage | 677,700 | 1.537B | 982 | 884 | 1.20 |
| code/chartmoe_chart2code | image | 898,609 | 1.492B | 1064 | 578 | 1.82 |
| code/web2code_new | image | 806,710 | 1.329B | 1322 | 882 | 1.60 |
| grounding/mi_grounding | multiimage | 500,000 | 1.200B | 693 | 556 | 1.29 |
| pointing/mi_o365 | multiimage | 500,000 | 1.142B | 726 | 585 | 1.29 |
| pointing/openimages_point | image | 870,779 | 0.921B | 957 | 800 | 1.26 |
| pointing/molmo2_multiimagepoint | multiimage | 312,183 | 0.916B | 1011 | 874 | 1.27 |
| chart/fv_unichart | image | 727,728 | 0.879B | 894 | 552 | 1.66 |
| gui/aguvis_stage1 | image | 504,911 | 0.675B | 1742 | 1133 | 1.60 |
| counting/openimages_count | image | 503,045 | 0.545B | 965 | 793 | 1.28 |
| gui/mi_gui_next_action | multiimage | 136,860 | 0.467B | 1017 | 1923 | 0.54 |
| gui/seeclick_ground | image | 263,098 | 0.421B | 1920 | 1080 | 1.78 |
| grounding/openimages_box | image | 344,300 | 0.410B | 968 | 788 | 1.29 |
| chart/fv_synthchartnet | image | 500,000 | 0.367B | 772 | 591 | 1.37 |

**Grand total (r=1): 66,844,960,351 tokens.**

Raw: `assets/token_census/results/remaining_categories/`.

`multilingual` (mentioned in the original 2026-08-17 category list) no longer exists on disk under
`DATA_ROOT` — confirmed via a fresh `discover_datasets(None)` scan (275 datasets total, 16
categories, `multilingual` absent from the list). Likely renamed or merged into another category
at some point; not investigated further since it isn't blocking the census of what's actually
there now.
