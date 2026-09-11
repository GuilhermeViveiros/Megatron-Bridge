# EuroVL data mixture — exact token census

Exact (not sampled) text+vision token counts per energon dataset, computed by
`assets/token_census/scripts/estimate_token_budget.py`. Every sample of every shard is counted —
no extrapolation. Vision tokens are computed analytically (image/video headers only, no pixel
decode) using the same math the real MoonViT processors use; validated at 0/60 mismatches
against the real end-to-end `EuroVLProcessor` output (image, multiimage, video, including the
real per-frame timestamp text) before running at scale — see
`assets/token_census/scripts/validate_token_estimates.py`.

Organized by real dataset category (matching `mixture.yaml`'s taxonomy), and within each
category by modality (image / multiimage / video / text). There are currently no text-only
(zero-vision) datasets in this census — the "text" subsection is always empty, kept explicit
rather than dropped so a future text-only addition has an obvious home.

Regenerate the raw census: `./apptainer.sh uv run --no-sync python assets/token_census/scripts/estimate_token_budget.py --categories <cats> --workers 16`
(use 16, not 64 — 64 concurrent worker imports exhausted file descriptors on this filesystem).
Regenerate this file from existing raw results: `uv run --no-sync python assets/token_census/scripts/render_readme.py`.

## Overall total

**276 datasets censused, 99,461,938 samples, 103.89B tokens (r=1, one epoch each).**

Note: as of this generation, 9 real datasets discovered on disk are NOT yet in this census
(`leopard_arxiv_enriched_translated`, `doc750k`, `docmatix`, `leopard_dude`, `leopard_monkey`,
`leopard_mpdocvqa`, `molmo2_doc` — some appear under both `image` and `multiimage`) — either
missing `.nv-meta` energon indexing or added/renamed after the last census run. A handful of
`_buggy_backup`/`_preshuffle_backup` directories also exist on disk and are intentionally excluded
(not real additional data).

| category | tokens (r=1) |
|---|---|
| code | 28.45B |
| ocr | 13.81B |
| captioning | 13.40B |
| chart | 11.71B |
| general_qa | 8.81B |
| knowledge | 5.74B |
| pointing | 5.05B |
| grounding | 4.79B |
| counting | 4.31B |
| doc | 3.91B |
| gui | 2.66B |
| math | 509.39M |
| medical | 357.10M |
| 3d_grounding | 297.10M |
| science | 62.79M |
| spatial | 26.59M |
| **TOTAL** | **103.89B** |

# image

236 dataset(s), 93,840,310 samples, **82.74B tokens (r=1)**.

## 3d_grounding

1 dataset(s), 129.75M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| hypersim_3d_ground | 100,003 | 1036 | 261 | 103.60M | 26.15M | 1024 | 768 | 1.33 |

Raw per-shard data: `assets/token_census/results/*/image__3d_grounding__<name>/*.json`

## captioning

11 dataset(s), 12.57B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| cc12m | 9,794,682 | 446 | 109 | 4.37B | 1.07B | 624 | 536 | 1.21 |
| grit | 8,507,801 | 503 | 116 | 4.28B | 985.82M | 728 | 584 | 1.34 |
| cc3m | 1,548,826 | 424 | 94 | 656.89M | 146.26M | 619 | 517 | 1.24 |
| pixmo-cap | 702,205 | 850 | 258 | 596.80M | 180.99M | 1435 | 1374 | 1.20 |
| sbu-captions | 499,580 | 100 | 92 | 49.96M | 46.12M | 256 | 256 | 1.00 |
| finevision_ureader_cap | 87,762 | 971 | 41 | 85.18M | 3.60M | 949 | 815 | 1.23 |
| sharegpt4o | 42,636 | 572 | 137 | 24.38M | 5.83M | 857 | 775 | 1.27 |
| textcaps | 21,942 | 971 | 39 | 21.31M | 855,205 | 949 | 817 | 1.23 |
| coco-caption | 39,621 | 369 | 94 | 14.62M | 3.72M | 576 | 485 | 1.25 |
| mminstruct_caption_en | 17,512 | 606 | 244 | 10.60M | 4.27M | 907 | 723 | 1.41 |
| flickr30k | 30,000 | 235 | 96 | 7.04M | 2.89M | 460 | 395 | 1.23 |

Raw per-shard data: `assets/token_census/results/*/image__captioning__<name>/*.json`

## chart

35 dataset(s), 8.61B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| chartnet_summary | 2,513,168 | 979 | 487 | 2.46B | 1.22B | 1744 | 1185 | 1.49 |
| caul_plotqa | 1,089,485 | 956 | 1019 | 1.04B | 1.11B | 1106 | 683 | 1.62 |
| fv_unichart | 727,728 | 667 | 540 | 485.61M | 392.90M | 894 | 552 | 1.66 |
| fv_synthchartnet | 500,000 | 573 | 160 | 286.66M | 79.87M | 772 | 591 | 1.37 |
| fv_mmc_instruct | 168,173 | 773 | 837 | 129.93M | 140.71M | 622 | 1277 | 0.62 |
| mmtab | 232,879 | 798 | 297 | 185.88M | 69.20M | 1903 | 853 | 2.86 |
| fv_cosyn_chart | 116,813 | 1053 | 785 | 122.98M | 91.74M | 1769 | 1243 | 1.45 |
| leopard_arxiv_enriched | 144,567 | 1032 | 162 | 149.18M | 23.44M | 1802 | 1273 | 1.71 |
| fv_arxivqa | 100,000 | 1030 | 149 | 103.04M | 14.91M | 1800 | 1275 | 1.71 |
| fv_cosyn_table | 46,518 | 979 | 768 | 45.53M | 35.71M | 1389 | 917 | 1.76 |
| fv_figureqa | 100,000 | 322 | 379 | 32.20M | 37.93M | 589 | 400 | 1.47 |
| fv_cosyn_diagram | 34,963 | 1048 | 576 | 36.63M | 20.13M | 1470 | 1153 | 1.70 |
| caul_wikisql | 74,989 | 509 | 166 | 38.13M | 12.43M | 698 | 596 | 1.90 |
| chartnet_realworldchart | 30,000 | 550 | 983 | 16.51M | 29.49M | 770 | 608 | 1.37 |
| caul_wtq | 38,246 | 678 | 207 | 25.94M | 7.92M | 977 | 644 | 2.38 |
| nemo_chartqa_nothink | 23,571 | 600 | 273 | 14.15M | 6.44M | 773 | 593 | 1.34 |
| fv_ureader_ie | 17,320 | 1059 | 46 | 18.34M | 792,574 | 1515 | 2012 | 0.76 |
| fv_chart2text | 26,961 | 565 | 131 | 15.22M | 3.54M | 716 | 602 | 1.25 |
| leopard_figureqa | 17,945 | 322 | 655 | 5.78M | 11.76M | 589 | 400 | 1.47 |
| nemo_fintabnet_nothink | 8,352 | 1004 | 720 | 8.39M | 6.01M | 771 | 999 | 0.77 |
| leopard_chartgemma | 16,323 | 528 | 103 | 8.61M | 1.69M | 702 | 561 | 1.29 |
| caul_vistext | 9,969 | 696 | 146 | 6.94M | 1.45M | 779 | 690 | 1.15 |
| caul_sqa | 8,644 | 558 | 302 | 4.83M | 2.61M | 976 | 480 | 2.90 |
| fv_finqa | 5,276 | 173 | 1145 | 914,233 | 6.04M | 696 | 181 | 4.79 |
| fv_figureqa_mathv360k | 17,587 | 301 | 62 | 5.30M | 1.09M | 548 | 400 | 1.37 |
| ov_ai2d_internvl | 12,403 | 393 | 52 | 4.88M | 644,677 | 605 | 473 | 1.41 |
| pangea_chartqa | 6,844 | 605 | 44 | 4.14M | 301,858 | 772 | 600 | 1.33 |
| fv_ai2d_merged | 4,866 | 401 | 487 | 1.95M | 2.37M | 611 | 483 | 1.40 |
| caul_visualmrc | 3,035 | 1056 | 145 | 3.21M | 440,888 | 1517 | 3614 | 0.48 |
| caul_multihiertt | 7,619 | 301 | 78 | 2.29M | 596,809 | 697 | 330 | 3.43 |
| caul_hitab | 2,516 | 637 | 252 | 1.60M | 635,207 | 978 | 565 | 2.78 |
| caul_tat_qa | 2,199 | 253 | 661 | 555,628 | 1.45M | 696 | 271 | 3.37 |
| chartqapro_refined_235b | 1,948 | 766 | 127 | 1.49M | 247,658 | 1194 | 986 | 1.37 |
| fv_lrv_chart | 1,776 | 700 | 170 | 1.24M | 301,197 | 780 | 692 | 1.15 |
| caul_ai2d | 2,439 | 402 | 202 | 979,651 | 492,270 | 613 | 481 | 1.40 |

Raw per-shard data: `assets/token_census/results/*/image__chart__<name>/*.json`

## code

12 dataset(s), 28.45B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| webcode2m_new | 3,166,069 | 1064 | 4424 | 3.37B | 14.01B | 1295 | 1617 | 0.90 |
| chartnet_code | 2,513,168 | 979 | 1169 | 2.46B | 2.94B | 1744 | 1185 | 1.49 |
| websight_new | 1,319,321 | 1071 | 528 | 1.41B | 696.58M | 2561 | 1656 | 1.62 |
| chartmoe_chart2code | 898,609 | 823 | 838 | 739.40M | 752.68M | 1064 | 578 | 1.82 |
| web2code_new | 806,710 | 1069 | 578 | 862.40M | 466.65M | 1322 | 882 | 1.60 |
| mmc_instruct | 168,106 | 784 | 837 | 131.87M | 140.66M | 632 | 1326 | 0.62 |
| webmmu | 4,088 | 1071 | 59280 | 4.38M | 242.34M | 1490 | 1554 | 0.95 |
| datik | 220,183 | 225 | 498 | 49.54M | 109.56M | 420 | 420 | 1.00 |
| datikz | 47,441 | 144 | 801 | 6.83M | 37.99M | 336 | 336 | 1.00 |
| chartmimic | 4,800 | 942 | 1280 | 4.52M | 6.14M | 1124 | 782 | 1.53 |
| mmcode | 4,427 | 194 | 1362 | 859,251 | 6.03M | 445 | 265 | 2.73 |
| plot2code | 368 | 595 | 545 | 219,068 | 200,382 | 903 | 664 | 1.40 |

Raw per-shard data: `assets/token_census/results/*/image__code__<name>/*.json`

## counting

5 dataset(s), 2.29B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| objects365_count | 2,676,042 | 547 | 80 | 1.46B | 213.20M | 753 | 612 | 1.27 |
| openimages_count | 503,045 | 967 | 116 | 486.44M | 58.39M | 965 | 793 | 1.28 |
| pixmo_count | 33,428 | 943 | 163 | 31.51M | 5.47M | 1299 | 1033 | 1.30 |
| tallyqa | 98,680 | 329 | 23 | 32.51M | 2.29M | 546 | 453 | 1.27 |
| taco_count | 646 | 1058 | 75 | 683,474 | 48,215 | 2899 | 3060 | 1.00 |

Raw per-shard data: `assets/token_census/results/*/image__counting__<name>/*.json`

## doc

17 dataset(s), 932.54M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| bigdocs_pubtables_1m | 345,989 | 396 | 978 | 137.10M | 338.21M | 614 | 526 | 1.63 |
| bigdocs_arxiv_ocr | 110,854 | 1068 | 494 | 118.36M | 54.73M | 1680 | 2226 | 0.76 |
| docreason51k | 51,726 | 919 | 124 | 47.53M | 6.42M | 1603 | 1973 | 1.48 |
| bigdocs_arxiv_table_cap | 72,524 | 413 | 52 | 29.97M | 3.80M | 924 | 362 | 3.82 |
| bigdocs_wikitq | 22,007 | 613 | 681 | 13.48M | 14.99M | 918 | 646 | 1.94 |
| bigdocs_cocotext | 30,223 | 372 | 569 | 11.25M | 17.21M | 585 | 482 | 1.27 |
| leopard_mplugdocreason | 25,863 | 919 | 120 | 23.76M | 3.10M | 1603 | 1973 | 1.48 |
| bigdocs_tabfact | 16,572 | 463 | 716 | 7.67M | 11.86M | 835 | 437 | 2.48 |
| leopard_monkey | 31,156 | 490 | 61 | 15.27M | 1.89M | 651 | 754 | 1.04 |
| leopard_dude | 12,108 | 1065 | 60 | 12.90M | 721,746 | 2013 | 2490 | 0.85 |
| sujet_finance | 9,212 | 999 | 477 | 9.20M | 4.39M | 813 | 957 | 0.87 |
| cauldron_docvqa | 10,177 | 1055 | 148 | 10.74M | 1.51M | 1739 | 2089 | 0.88 |
| pangea_table_vqa | 16,408 | 423 | 273 | 6.93M | 4.48M | 930 | 374 | 3.76 |
| pangea_doc_vqa | 9,665 | 908 | 204 | 8.78M | 1.97M | 1649 | 1628 | 1.39 |
| docreason25k_refined | 8,726 | 869 | 314 | 7.59M | 2.74M | 1400 | 1734 | 1.33 |
| bigdocs_cord_v2 | 997 | 936 | 1418 | 933,321 | 1.41M | 1001 | 1578 | 0.65 |
| tat_dqa | 1,969 | 64 | 765 | 126,016 | 1.51M | 224 | 224 | 1.00 |

Raw per-shard data: `assets/token_census/results/*/image__doc__<name>/*.json`

## general_qa

24 dataset(s), 1.25B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| pangea_laion_multi | 324,231 | 455 | 287 | 147.64M | 93.18M | 671 | 516 | 1.34 |
| pixmo_cap_qa | 147,104 | 828 | 318 | 121.85M | 46.75M | 1374 | 1245 | 1.23 |
| mminstruct_qa | 146,401 | 581 | 557 | 85.12M | 81.59M | 862 | 705 | 1.34 |
| mmevol | 160,211 | 450 | 490 | 72.03M | 78.49M | 738 | 607 | 1.28 |
| dvqa | 200,000 | 256 | 348 | 51.20M | 69.61M | 448 | 448 | 1.00 |
| pixmo_ask_model_anything | 70,517 | 732 | 273 | 51.59M | 19.27M | 985 | 889 | 1.20 |
| llava_instruct | 81,467 | 369 | 486 | 30.08M | 39.59M | 578 | 483 | 1.26 |
| rsvqa_hr | 74,542 | 361 | 379 | 26.91M | 28.22M | 512 | 512 | 1.00 |
| gqa | 87,931 | 278 | 248 | 24.41M | 21.78M | 498 | 411 | 1.27 |
| vqav2 | 84,751 | 369 | 148 | 31.30M | 12.55M | 578 | 484 | 1.26 |
| clevr | 69,995 | 216 | 348 | 15.12M | 24.38M | 480 | 320 | 1.50 |
| alfworldgpt | 43,716 | 121 | 415 | 5.29M | 18.16M | 300 | 300 | 1.00 |
| cocoqa | 46,238 | 370 | 54 | 17.10M | 2.48M | 580 | 483 | 1.26 |
| lrv_normal | 12,745 | 243 | 578 | 3.09M | 7.36M | 473 | 392 | 1.27 |
| visual7w | 14,412 | 273 | 311 | 3.93M | 4.48M | 492 | 409 | 1.27 |
| infographic_vqa | 4,214 | 978 | 176 | 4.12M | 743,704 | 880 | 1563 | 0.69 |
| molmo2_diagram_single | 2,171 | 1013 | 399 | 2.20M | 865,315 | 1733 | 1343 | 1.92 |
| worldvqa | 2,990 | 631 | 31 | 1.89M | 92,278 | 895 | 673 | 1.46 |
| spark | 3,703 | 458 | 58 | 1.70M | 213,502 | 641 | 485 | 1.39 |
| yesbut | 1,084 | 1056 | 239 | 1.14M | 259,524 | 1432 | 1053 | 1.35 |
| molmo2_graphic_single | 859 | 799 | 347 | 686,542 | 298,331 | 1254 | 1010 | 1.40 |
| vsr | 2,157 | 387 | 49 | 835,124 | 104,801 | 580 | 507 | 1.20 |
| newyorker_caption_contest | 1,632 | 415 | 119 | 676,500 | 194,578 | 606 | 474 | 1.32 |
| vizwiz | 737 | 916 | 265 | 675,097 | 195,108 | 1026 | 1324 | 0.78 |

Raw per-shard data: `assets/token_census/results/*/image__general_qa__<name>/*.json`

## grounding

5 dataset(s), 3.59B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| objects365_ground | 4,469,839 | 554 | 127 | 2.48B | 567.38M | 761 | 616 | 1.28 |
| openimages_box | 344,300 | 966 | 224 | 332.59M | 77.10M | 968 | 788 | 1.29 |
| refcoco_ground | 267,242 | 379 | 82 | 101.21M | 21.84M | 592 | 483 | 1.28 |
| groundui_ground | 12,518 | 1067 | 79 | 13.36M | 990,463 | 1404 | 1105 | 1.46 |
| taco_ground | 3,123 | 1057 | 103 | 3.30M | 320,729 | 2947 | 3078 | 1.02 |

Raw per-shard data: `assets/token_census/results/*/image__grounding__<name>/*.json`

## gui

23 dataset(s), 1.75B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| aguvis_stage1 | 504,911 | 1029 | 308 | 519.46M | 155.51M | 1742 | 1133 | 1.60 |
| seeclick_ground | 263,098 | 1032 | 570 | 271.52M | 149.93M | 1920 | 1080 | 1.78 |
| seeclick_qa | 129,704 | 1032 | 115 | 133.85M | 14.89M | 1920 | 1080 | 1.78 |
| odyssey_actions | 111,050 | 1065 | 69 | 118.25M | 7.62M | 1534 | 2211 | 0.77 |
| androidcontrol_actions | 83,819 | 1056 | 71 | 88.51M | 5.95M | 1090 | 2417 | 0.45 |
| amex_actions | 38,704 | 1071 | 87 | 41.44M | 3.35M | 1429 | 3034 | 0.47 |
| widget_ground | 33,890 | 960 | 74 | 32.53M | 2.50M | 963 | 1711 | 0.56 |
| aitw_actions | 43,129 | 662 | 60 | 28.55M | 2.60M | 493 | 1011 | 0.49 |
| screenqa | 30,053 | 954 | 37 | 28.66M | 1.11M | 952 | 1693 | 0.56 |
| ricosca_refer | 27,865 | 954 | 51 | 26.58M | 1.42M | 953 | 1694 | 0.56 |
| waveui_ground | 17,411 | 1057 | 84 | 18.41M | 1.46M | 1199 | 822 | 1.53 |
| ricosca_point | 17,405 | 954 | 64 | 16.61M | 1.11M | 954 | 1695 | 0.56 |
| widget_point | 14,435 | 960 | 65 | 13.86M | 941,872 | 963 | 1712 | 0.56 |
| uibert_ground | 11,681 | 953 | 79 | 11.13M | 924,972 | 952 | 1692 | 0.56 |
| screen2words | 11,628 | 949 | 59 | 11.03M | 681,519 | 944 | 1679 | 0.56 |
| waveui_point | 7,567 | 1057 | 75 | 8.00M | 569,255 | 1200 | 827 | 1.52 |
| mind2web_actions | 7,362 | 1066 | 77 | 7.85M | 568,906 | 1287 | 4883 | 0.40 |
| omniact_actions | 6,713 | 1054 | 67 | 7.07M | 451,187 | 2240 | 1354 | 1.67 |
| leopard_rico | 6,290 | 960 | 36 | 6.04M | 226,058 | 963 | 1710 | 0.56 |
| uibert_point | 4,979 | 953 | 71 | 4.74M | 351,099 | 951 | 1690 | 0.56 |
| leopard_mind2web | 1,765 | 1060 | 210 | 1.87M | 370,397 | 1289 | 1760 | 0.80 |
| ricosca_ground | 1,602 | 934 | 72 | 1.50M | 115,550 | 920 | 1636 | 0.56 |
| leopard_omniact | 1,038 | 1066 | 85 | 1.11M | 88,713 | 1440 | 900 | 1.60 |

Raw per-shard data: `assets/token_census/results/*/image__gui__<name>/*.json`

## knowledge

12 dataset(s), 5.73B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| wit | 12,916,713 | 116 | 97 | 1.50B | 1.25B | 300 | 278 | 1.25 |
| culturalground_oe | 5,044,547 | 148 | 57 | 744.88M | 285.85M | 356 | 312 | 1.21 |
| wit_en | 4,913,730 | 115 | 93 | 564.95M | 458.27M | 300 | 274 | 1.26 |
| culturalground_mcq | 3,498,781 | 148 | 72 | 519.18M | 253.08M | 359 | 311 | 1.23 |
| visual_genome | 132,760 | 251 | 240 | 33.26M | 31.91M | 478 | 397 | 1.27 |
| localized_narratives | 118,272 | 369 | 79 | 43.68M | 9.36M | 578 | 484 | 1.25 |
| cosyn_music | 11,969 | 1000 | 427 | 11.96M | 5.11M | 820 | 1008 | 0.88 |
| a-okvqa | 17,315 | 374 | 37 | 6.47M | 639,341 | 587 | 482 | 1.28 |
| okvqa | 8,998 | 372 | 36 | 3.35M | 325,802 | 618 | 448 | 1.40 |
| viquae | 2,385 | 373 | 47 | 889,653 | 111,981 | 512 | 536 | 1.12 |
| web_landmark | 500 | 860 | 200 | 429,949 | 100,217 | 1356 | 943 | 1.50 |
| web_celebrity | 495 | 778 | 158 | 385,056 | 78,402 | 1196 | 738 | 1.67 |

Raw per-shard data: `assets/token_census/results/*/image__knowledge__<name>/*.json`

## math

21 dataset(s), 470.85M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| finevision_mavis_math_rule_geo | 99,986 | 1055 | 225 | 105.51M | 22.47M | 1647 | 1645 | 1.11 |
| visualwebinstruct_onevision | 263,578 | 277 | 165 | 73.11M | 43.61M | 524 | 351 | 1.76 |
| finevision_cosyn_400k_math | 66,714 | 667 | 469 | 44.47M | 31.27M | 1017 | 537 | 2.30 |
| finevision_mavis_math_metagen | 87,348 | 328 | 158 | 28.65M | 13.84M | 592 | 388 | 1.76 |
| finevision_clevr_math | 70,000 | 216 | 224 | 15.12M | 15.71M | 480 | 320 | 1.50 |
| finevision_raven | 42,000 | 693 | 31 | 29.09M | 1.30M | 664 | 804 | 0.88 |
| finevision_geomverse | 9,303 | 1047 | 355 | 9.74M | 3.30M | 1395 | 1690 | 0.91 |
| finevision_pmc_vqa_mathv360k | 35,948 | 146 | 75 | 5.23M | 2.70M | 321 | 290 | 1.19 |
| finevision_super_clevr_mathv360k | 8,642 | 414 | 56 | 3.58M | 485,118 | 640 | 480 | 1.33 |
| finevision_geo170k_align | 35,297 | 29 | 78 | 1.01M | 2.75M | 152 | 118 | 1.39 |
| finevision_geo170k_qa | 12,101 | 42 | 215 | 504,710 | 2.60M | 191 | 129 | 1.48 |
| finevision_geometry3k_mathv360k | 9,724 | 201 | 78 | 1.95M | 759,568 | 436 | 310 | 1.44 |
| finevision_mapqa_mathv360k | 5,225 | 450 | 56 | 2.35M | 292,364 | 700 | 500 | 1.40 |
| finevision_geoqa_plus_mathv360k | 17,162 | 27 | 103 | 459,142 | 1.77M | 148 | 114 | 1.40 |
| mathvision | 3,344 | 410 | 159 | 1.37M | 531,840 | 770 | 500 | 1.79 |
| finevision_unigeo_mathv360k | 11,949 | 24 | 101 | 289,706 | 1.20M | 140 | 110 | 1.42 |
| mathverse_refined_final | 3,128 | 386 | 67 | 1.21M | 209,731 | 618 | 525 | 1.29 |
| finevision_clevr_math_mathv360k | 5,280 | 216 | 52 | 1.14M | 276,077 | 480 | 320 | 1.50 |
| finevision_geo3k | 2,091 | 188 | 80 | 392,828 | 167,965 | 421 | 300 | 1.44 |
| finevision_intergps | 1,280 | 178 | 103 | 227,416 | 131,657 | 405 | 294 | 1.42 |
| finevision_geos_mathv360k | 498 | 52 | 87 | 25,693 | 43,541 | 225 | 154 | 1.48 |

Raw per-shard data: `assets/token_census/results/*/image__math__<name>/*.json`

## medical

5 dataset(s), 357.10M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| pmc_vqa | 414,979 | 348 | 64 | 144.41M | 26.37M | 514 | 448 | 1.34 |
| EuroVL-Medical-Cap | 335,523 | 332 | 149 | 111.43M | 50.11M | 502 | 430 | 1.33 |
| path_vqa | 32,632 | 530 | 54 | 17.30M | 1.75M | 766 | 518 | 1.49 |
| slake | 7,033 | 519 | 49 | 3.65M | 341,621 | 607 | 607 | 1.00 |
| vqa_rad | 2,244 | 720 | 53 | 1.62M | 118,846 | 770 | 776 | 1.02 |

Raw per-shard data: `assets/token_census/results/*/image__medical__<name>/*.json`

## ocr

48 dataset(s), 13.81B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| chartnet_csv | 2,513,168 | 979 | 481 | 2.46B | 1.21B | 1744 | 1185 | 1.49 |
| chartmoe_chart2code | 898,609 | 823 | 838 | 739.40M | 752.68M | 1064 | 578 | 1.82 |
| chartmoe_chart2json | 898,609 | 823 | 451 | 739.40M | 404.95M | 1064 | 578 | 1.82 |
| wkvvqa | 522,963 | 1060 | 596 | 554.27M | 311.43M | 1289 | 1720 | 0.75 |
| chartmoe_chart2table | 898,609 | 823 | 127 | 739.40M | 114.25M | 1064 | 578 | 1.82 |
| nemotron_ocr | 438,847 | 1020 | 881 | 447.56M | 386.80M | 1034 | 988 | 1.11 |
| synthtabnet | 600,369 | 228 | 804 | 136.96M | 482.79M | 481 | 354 | 1.74 |
| SynthCodeNet | 499,908 | 621 | 570 | 310.28M | 285.00M | 638 | 886 | 1.27 |
| synthdog | 500,000 | 1047 | 123 | 523.58M | 61.54M | 1090 | 1090 | 1.10 |
| nemotron_ocr9 | 224,170 | 989 | 1231 | 221.76M | 275.86M | 761 | 998 | 0.76 |
| olmocr_mix | 261,982 | 1063 | 757 | 278.59M | 198.36M | 1260 | 1611 | 0.80 |
| vcr_wiki_en_hard | 1,268,328 | 154 | 65 | 195.71M | 81.85M | 300 | 375 | 0.86 |
| synthdog_multilingual_eu | 205,000 | 1047 | 176 | 214.65M | 36.13M | 1091 | 1089 | 1.10 |
| ureader_qa | 252,953 | 911 | 47 | 230.42M | 11.88M | 1200 | 1094 | 1.65 |
| mathwriting-google | 300,000 | 712 | 41 | 213.57M | 12.34M | 1511 | 432 | 4.74 |
| nvidia_ocr_synth_enru | 99,999 | 867 | 1276 | 86.74M | 127.59M | 864 | 864 | 1.03 |
| doclaynet | 68,698 | 1089 | 774 | 74.81M | 53.20M | 1025 | 1025 | 1.00 |
| vcr_wiki_en_easy | 500,000 | 154 | 65 | 77.13M | 32.27M | 300 | 375 | 0.86 |
| pangea_webui_ocr | 90,000 | 1064 | 109 | 95.72M | 9.85M | 1280 | 1304 | 1.08 |
| nemotron_ocr6 | 48,307 | 1089 | 636 | 52.61M | 30.71M | 1025 | 1025 | 1.00 |
| SynthFormulaNet | 499,997 | 66 | 100 | 33.06M | 49.89M | 342 | 79 | 4.26 |
| ocr-vqa | 288,797 | 231 | 40 | 66.71M | 11.51M | 355 | 489 | 0.73 |
| latexformulas | 552,340 | 36 | 97 | 19.66M | 53.77M | 326 | 65 | 5.69 |
| textocr_grounded | 24,863 | 971 | 1847 | 24.15M | 45.92M | 949 | 817 | 1.23 |
| textvqa | 31,728 | 971 | 36 | 30.80M | 1.14M | 947 | 818 | 1.22 |
| latex_handwritten | 39,583 | 663 | 76 | 26.25M | 3.00M | 1381 | 368 | 3.96 |
| infovqa | 23,946 | 1006 | 39 | 24.10M | 924,077 | 1196 | 2627 | 0.67 |
| textocr | 21,571 | 971 | 148 | 20.95M | 3.19M | 948 | 818 | 1.22 |
| nemotron_ocr7 | 25,281 | 476 | 362 | 12.04M | 9.15M | 699 | 550 | 1.44 |
| hw_squad | 20,464 | 889 | 122 | 18.19M | 2.50M | 918 | 997 | 1.12 |
| sujet_finance | 9,801 | 999 | 477 | 9.79M | 4.67M | 813 | 957 | 0.87 |
| rendered_text | 10,000 | 1089 | 49 | 10.89M | 486,707 | 1024 | 1024 | 1.00 |
| llavar | 33,629 | 262 | 63 | 8.80M | 2.11M | 428 | 467 | 0.96 |
| pdfvqa | 9,279 | 634 | 362 | 5.88M | 3.36M | 595 | 793 | 0.75 |
| hme100k | 74,492 | 35 | 55 | 2.64M | 4.13M | 272 | 66 | 4.41 |
| st_vqa | 17,247 | 291 | 45 | 5.03M | 772,028 | 484 | 414 | 1.19 |
| captcha | 113,062 | 12 | 31 | 1.36M | 3.55M | 150 | 40 | 3.75 |
| coco-text | 9,982 | 373 | 57 | 3.72M | 568,931 | 586 | 482 | 1.27 |
| visualmrc | 3,035 | 1061 | 145 | 3.22M | 440,362 | 911 | 1963 | 0.48 |
| pangea_mtvqa | 3,035 | 1054 | 112 | 3.20M | 340,904 | 2365 | 2354 | 1.10 |
| wordart | 19,066 | 160 | 24 | 3.04M | 462,515 | 417 | 214 | 2.21 |
| bentham | 10,843 | 269 | 37 | 2.92M | 396,063 | 1476 | 124 | 12.27 |
| sroie | 33,616 | 16 | 40 | 549,824 | 1.36M | 207 | 36 | 5.71 |
| ctw1500 | 8,060 | 95 | 27 | 763,951 | 215,966 | 359 | 128 | 4.60 |
| chrome_writting | 8,825 | 53 | 51 | 466,712 | 453,874 | 314 | 104 | 3.07 |
| poie | 482 | 396 | 106 | 190,754 | 51,095 | 544 | 632 | 1.03 |
| orand_car_a | 1,999 | 15 | 42 | 30,279 | 82,998 | 167 | 54 | 3.15 |
| iiit5k | 1,990 | 14 | 35 | 28,284 | 70,188 | 111 | 44 | 2.62 |

Raw per-shard data: `assets/token_census/results/*/image__ocr__<name>/*.json`

## pointing

7 dataset(s), 2.74B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| objects365_point | 2,675,114 | 547 | 85 | 1.46B | 226.18M | 753 | 612 | 1.27 |
| openimages_point | 870,779 | 968 | 90 | 842.63M | 77.99M | 957 | 800 | 1.26 |
| refcoco_point | 136,810 | 379 | 71 | 51.79M | 9.76M | 592 | 483 | 1.28 |
| molmopoint_guisyn | 36,960 | 919 | 616 | 33.96M | 22.76M | 1580 | 1072 | 1.44 |
| groundui_point | 5,480 | 1067 | 70 | 5.85M | 385,406 | 1412 | 1097 | 1.47 |
| taco_point | 3,123 | 1057 | 88 | 3.30M | 274,010 | 2947 | 3078 | 1.02 |
| pointarena | 951 | 852 | 60 | 810,293 | 56,909 | 1569 | 1211 | 1.38 |

Raw per-shard data: `assets/token_census/results/*/image__pointing__<name>/*.json`

## science

7 dataset(s), 41.48M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| verisciqa | 10,000 | 1015 | 99 | 10.15M | 989,415 | 1788 | 1302 | 1.61 |
| cosyn_circuit | 10,470 | 572 | 416 | 5.99M | 4.36M | 881 | 490 | 1.86 |
| cosyn_chemical | 8,942 | 517 | 436 | 4.62M | 3.90M | 760 | 537 | 1.47 |
| ai2d_merged | 4,866 | 401 | 487 | 1.95M | 2.37M | 611 | 483 | 1.40 |
| pathvqa | 4,301 | 500 | 246 | 2.15M | 1.06M | 733 | 502 | 1.46 |
| scienceqa_nona_context | 5,078 | 259 | 137 | 1.31M | 695,044 | 513 | 359 | 1.94 |
| kaleidoscope | 5,395 | 239 | 120 | 1.29M | 645,407 | 469 | 297 | 1.99 |

Raw per-shard data: `assets/token_census/results/*/image__science__<name>/*.json`

## spatial

3 dataset(s), 13.64M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| omnispatial | 6,693 | 709 | 99 | 4.75M | 661,114 | 1386 | 994 | 1.58 |
| tetris_analogy | 7,935 | 528 | 90 | 4.19M | 711,625 | 616 | 668 | 0.92 |
| dise_singleimage | 4,479 | 693 | 51 | 3.10M | 227,562 | 920 | 568 | 1.62 |

Raw per-shard data: `assets/token_census/results/*/image__spatial__<name>/*.json`

# multiimage

39 dataset(s), 5,530,466 samples, **20.48B tokens (r=1)**.

## 3d_grounding

1 dataset(s), 167.35M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| hypersim_3d_ground_mv | 50,083 | 3108 | 234 | 155.65M | 11.70M | 1024 | 768 | 1.33 |

Raw per-shard data: `assets/token_census/results/*/multiimage__3d_grounding__<name>/*.json`

## captioning

1 dataset(s), 152.37M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| internvl_multi_en | 77,704 | 1543 | 418 | 119.92M | 32.45M | 722 | 557 | 1.35 |

Raw per-shard data: `assets/token_census/results/*/multiimage__captioning__<name>/*.json`

## chart

9 dataset(s), 3.10B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| molmo2_chart_translated | 379,129 | 4843 | 911 | 1.84B | 345.21M | 1455 | 1032 | 1.48 |
| molmo2_table | 33,980 | 7659 | 750 | 260.24M | 25.50M | 1575 | 1030 | 1.71 |
| molmo2_chart | 46,309 | 4847 | 706 | 224.48M | 32.70M | 1454 | 1031 | 1.48 |
| molmo2_diagram | 21,047 | 4979 | 750 | 104.79M | 15.78M | 1449 | 1302 | 1.54 |
| leopard_chartgemma | 48,972 | 1576 | 318 | 77.19M | 15.57M | 701 | 559 | 1.29 |
| leopard_figureqa | 35,990 | 644 | 739 | 23.19M | 26.59M | 589 | 400 | 1.47 |
| leopard_arxiv | 15,806 | 2754 | 384 | 43.52M | 6.07M | 1798 | 1296 | 1.68 |
| molmo2_graphic | 11,564 | 2343 | 697 | 27.10M | 8.06M | 718 | 532 | 1.45 |
| leopard_multihiertt | 14,710 | 1392 | 197 | 20.48M | 2.90M | 697 | 380 | 2.95 |

Raw per-shard data: `assets/token_census/results/*/multiimage__chart__<name>/*.json`

## counting

1 dataset(s), 2.02B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| mi_oiv7 | 500,000 | 3850 | 190 | 1.93B | 95.13M | 961 | 802 | 1.26 |

Raw per-shard data: `assets/token_census/results/*/multiimage__counting__<name>/*.json`

## doc

1 dataset(s), 2.98B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| molmo2_doc_translated | 391,195 | 7027 | 596 | 2.75B | 233.13M | 1601 | 1793 | 0.99 |

Raw per-shard data: `assets/token_census/results/*/multiimage__doc__<name>/*.json`

## general_qa

9 dataset(s), 7.56B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| doclingmatix | 1,199,362 | 2072 | 2797 | 2.49B | 3.35B | 1357 | 1674 | 0.83 |
| molmo2_multiimageqa_translated | 677,700 | 1874 | 394 | 1.27B | 267.33M | 982 | 884 | 1.20 |
| nlvr2 | 50,426 | 1097 | 124 | 55.33M | 6.26M | 771 | 627 | 1.28 |
| molmo2_multiimageqa | 27,842 | 1884 | 315 | 52.46M | 8.76M | 982 | 884 | 1.20 |
| img_diff | 18,461 | 1723 | 97 | 31.80M | 1.79M | 864 | 864 | 1.00 |
| mimic_cgd | 70,939 | 128 | 116 | 9.08M | 8.25M | 224 | 224 | 1.00 |
| mp_docvqa | 911 | 5579 | 155 | 5.08M | 141,189 | 1813 | 2117 | 0.90 |
| mirb | 1,323 | 1760 | 70 | 2.33M | 93,128 | 973 | 815 | 1.36 |
| spot_the_diff | 8,566 | 128 | 53 | 1.10M | 455,028 | 224 | 224 | 1.00 |

Raw per-shard data: `assets/token_census/results/*/multiimage__general_qa__<name>/*.json`

## grounding

1 dataset(s), 1.20B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| mi_grounding | 500,000 | 1970 | 430 | 984.97M | 214.80M | 693 | 556 | 1.29 |

Raw per-shard data: `assets/token_census/results/*/multiimage__grounding__<name>/*.json`

## gui

5 dataset(s), 905.43M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| mi_gui_next_action | 136,860 | 3329 | 83 | 455.60M | 11.37M | 1017 | 1923 | 0.54 |
| mi_gui_before_after | 153,833 | 1874 | 57 | 288.33M | 8.77M | 1001 | 1938 | 0.53 |
| mi_gui_step_order | 20,967 | 2803 | 59 | 58.78M | 1.24M | 965 | 1973 | 0.50 |
| leopard_rico | 18,742 | 2864 | 120 | 53.68M | 2.25M | 958 | 1702 | 0.56 |
| leopard_mind2web | 5,597 | 4352 | 188 | 24.36M | 1.05M | 1287 | 1434 | 0.90 |

Raw per-shard data: `assets/token_census/results/*/multiimage__gui__<name>/*.json`

## knowledge

1 dataset(s), 9.85M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| molmo2_syn_music | 4,785 | 1596 | 462 | 7.63M | 2.21M | 825 | 569 | 1.70 |

Raw per-shard data: `assets/token_census/results/*/multiimage__knowledge__<name>/*.json`

## math

2 dataset(s), 38.54M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| visualwebinstruct | 22,477 | 940 | 702 | 21.12M | 15.79M | 584 | 389 | 1.95 |
| mv_math | 2,008 | 372 | 443 | 747,297 | 888,941 | 314 | 263 | 1.27 |

Raw per-shard data: `assets/token_census/results/*/multiimage__math__<name>/*.json`

## pointing

5 dataset(s), 2.31B tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| mi_o365 | 500,000 | 2097 | 188 | 1.05B | 93.85M | 726 | 585 | 1.29 |
| molmo2_multiimagepoint | 312,183 | 2271 | 663 | 708.85M | 207.13M | 1011 | 874 | 1.27 |
| mi_refcoco | 150,000 | 1516 | 79 | 227.47M | 11.84M | 593 | 483 | 1.28 |
| mi_pixmo | 3,602 | 3489 | 379 | 12.57M | 1.37M | 1286 | 1055 | 1.27 |
| mi_taco | 105 | 4060 | 194 | 426,283 | 20,329 | 2747 | 2892 | 1.00 |

Raw per-shard data: `assets/token_census/results/*/multiimage__pointing__<name>/*.json`

## science

2 dataset(s), 21.31M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| molmo2_syn_circuit | 3,416 | 3075 | 559 | 10.50M | 1.91M | 1187 | 911 | 1.52 |
| molmo2_syn_chemical | 4,872 | 1326 | 499 | 6.46M | 2.43M | 433 | 346 | 1.26 |

Raw per-shard data: `assets/token_census/results/*/multiimage__science__<name>/*.json`

## spatial

1 dataset(s), 12.95M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| dise_multiimage | 9,000 | 1382 | 57 | 12.44M | 509,400 | 448 | 448 | 1.00 |

Raw per-shard data: `assets/token_census/results/*/multiimage__spatial__<name>/*.json`

# video

1 dataset(s), 91,162 samples, **677.62M tokens (r=1)**.

## captioning

1 dataset(s), 677.62M tokens.

| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |
|---|---|---|---|---|---|---|---|---|
| molmo2_cap | 91,162 | 6077 | 1356 | 553.99M | 123.63M | - | - | - |

Raw per-shard data: `assets/token_census/results/*/video__captioning__<name>/*.json`

# text

No text-only (zero-vision) datasets in this census yet.
