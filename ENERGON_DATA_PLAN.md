# Plan: Convert EuroVL curated data → Megatron-Energon (category-organized, weighted-blend ready)

> **Audience:** an engineering agent starting cold. This document is self-contained.
> **Goal:** convert NeMo-Curator-produced VLM datasets into Megatron-Energon format under a new
> `energon-data/` tree, organized by `(modality, task)` category, so EuroVL training can consume a
> **weighted multi-source blend**. Start with **flickr30k** as the end-to-end pilot, then generalize.

---

## 0. Environment & conventions (read first)

- **Repo:** `/e/home/jusers/viveiros1/jupiter/Megatron-Bridge` (package `megatron.bridge`, Python 3.12).
- **All Python/training runs go inside the apptainer container**, via `uv run --no-sync`.
  Launcher: `apptainer.sh` (`apptainer shell ... "$SIF"`, then run commands inside).
  **Do NOT use bare `python3`/`uv` outside the container** — it lacks `megatron.core` and friends.
- Conversion scripts that only need `pyarrow`/`Pillow`/`tarfile` can run with `uv run --no-sync python ...`
  inside the shell.
- **Never modify `3rdparty/Megatron-LM/`.** Add NVIDIA Apache headers to new `.py` files. Sign off commits (`git commit -s`).
- Remote: push to `fork` (= `GuilhermeViveiros/Megatron-Bridge`), **not** `origin` (= NVIDIA upstream).

---

## 1. Source data — verified facts

Location: `/e/scratch/e-ext-2025e01-100/EuroVL-Data/processed-data/<dataset>/`

Each `<dataset>` dir holds **paired files**: `images-<hash>-000000.tar` + `images-<hash>-000000.parquet` (101 pairs for sharegpt4o, 62 for flickr30k, etc.).

- **`.tar`** = images only: members named `<image_id>.jpg` (e.g. `flickr30k-...jpg`). No text in the tar.
- **`.parquet`** = one row per image, columns:
  | column | meaning |
  |---|---|
  | `image_id` | sample key, e.g. `sharegpt4o-00003_003072` |
  | `tar_file` | **STALE absolute path** (`/e/scratch/jureap126/gviveiros/...`) — DO NOT USE |
  | `member_name` | jpg name inside the **local** sibling tar, e.g. `sharegpt4o-00003_003072.jpg` |
  | `original_path` | stale raw path — ignore |
  | `metadata` | **Python-`repr` dict string** (single quotes!) holding `caption` (and possibly QA fields) |

### ⚠️ Gotchas (must handle)
1. **`metadata` is NOT JSON** — it's `{'caption': '...'}` with single quotes. Parse with
   `ast.literal_eval(metadata)`, **not** `json.loads`.
2. **`tar_file` / `original_path` are stale** (point to an old `jureap126` location). Always resolve images
   from the **parquet file's own directory** + the matching local `.tar` + `member_name`.
3. **Captions are separate from images** (parquet vs tar) → energon needs them together, so conversion
   must emit new sample files keyed to the image (see §4).
4. Sources are **heterogeneous** — caption / VQA / OCR / grounding / video. The `metadata` schema differs
   per source; **inspect each before assigning a category** (see §3).

### Datasets present (assign categories in Phase A; initial guess)
| dataset | guessed modality/task | notes |
|---|---|---|
| flickr30k, cc3m, cc12m, coco-caption, sbu-captions, textcaps, pixmo-cap, molmo2_cap, wit | image / captioning | `metadata.caption` |
| sharegpt4o | image / detailed-caption (or conversation) | verify schema |
| a-okvqa | image / vqa (reasoning) | verify Q/A fields |
| ocr-vqa, textvqa, textocr, coco-text, llavar | image / ocr(-vqa) | text-in-image |
| grit | image / grounding | referring expr / grounded caption — verify |
| vatex | **video** / captioning | video samples — different handling, defer |

**flickr30k confirmed:** `metadata = {'caption': '...'}` → **image / captioning**. 62 tar+parquet pairs, ~380 imgs/shard.

---

## 2. What is Megatron-Energon (background)

NVIDIA's large-scale multimodal data loader. You **prepare** each dataset once into a sharded
**WebDataset** (tars where one sample shares a key across extensions, e.g. `key.jpg` + `key.json`); prepare
creates a `.nv-meta/` dir describing sample type, field mapping, and train/val split. At train time energon
gives: resumable/deterministic sampling, distributed loading across DP ranks, **weighted blending** of many
datasets via a **metadataset** YAML, and a **TaskEncoder** that converts raw samples → model batches.
Bridge's energon integration lives in `src/megatron/bridge/data/energon/`.

---

## 3. Target layout (`energon-data/`)

Create: `/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data/`

```
energon-data/
  image/
    captioning/   flickr30k/  cc3m/  coco-caption/ ...
    vqa/          a-okvqa/ ...
    ocr/          ocr-vqa/  textvqa/ ...
    grounding/    grit/
  video/
    captioning/   vatex/
  datasets.yaml   # manifest (machine-readable, drives blends)
```
- Each leaf `<dataset>/` is an **energon-prepared** dataset (shards + `.nv-meta/`).
- `datasets.yaml` manifest entry per dataset:
  ```yaml
  flickr30k:
    path: image/captioning/flickr30k
    modality: image
    task: captioning
    n_samples: <verified count>
    default_weight: 1.0
  ```
- Blends are built later as energon **metadataset** YAMLs that reference these paths with weights, generated
  from the manifest (e.g. "image/captioning @ 0.7 + image/vqa @ 0.3").

---

## 4. Conversion approach

**Pilot uses Option A (re-tar).** Note Option B for the huge sources.

- **Option A — re-tar (simple, recommended for pilot & small/medium sets):**
  For each parquet row: read jpg bytes from the local sibling tar by `member_name`; `ast.literal_eval` the
  `metadata`; write a new WebDataset shard containing `{key}.jpg` + `{key}.json` (the json holds
  `{"caption": ...}` or a conversation). Then `energon prepare`.
  - **Cost:** temporarily duplicates image bytes (need disk headroom; reclaimed when `processed-data/<ds>`
    is deleted in Phase F).
- **Option B — caption-tars + energon join (for cc12m-scale, later):** keep existing image tars; write
  parallel `{key}.json` caption tars; use energon's join-by-key so images aren't copied. More complex;
  out of scope for the pilot.

---

## 5. Step-by-step plan

### Phase A — Prerequisites (blocking)
- [ ] **Verify energon is importable in the container:** `uv run --no-sync python -c "import megatron.energon; print(megatron.energon.__version__)"`.
  `megatron-energon` is **not** in `pyproject.toml`, so it may be missing. If missing, resolve install
  (optional extra / separate dependency PR per repo policy) **before** Phase D. Conversion (Phases B–C) does
  NOT need energon and can proceed regardless.
- [ ] Confirm disk headroom on the scratch FS for the re-tar duplication (flickr30k tiny; fine).
- [ ] Create `energon-data/` and the `image/captioning/` subtree.

### Phase B — Inspect per-source metadata (category assignment)
- [ ] For each dataset, read 1–2 parquet rows and `ast.literal_eval(metadata)`; record the keys.
- [ ] Assign `(modality, task)` and the text-construction rule:
  - captioning → `metadata.caption`
  - vqa → question + answer fields (verify names)
  - ocr → caption/answer (verify)
  - video (vatex) → defer (needs frame handling)
- [ ] (flickr30k already done: image/captioning, `caption`.)

### Phase C — Write the converter `convert_to_energon.py`
Location suggestion: `scripts/data/convert_to_energon.py` (new; NVIDIA header).
Spec (one dataset per invocation):
```
args: --src  /e/scratch/.../processed-data/flickr30k
      --dst  /e/scratch/.../energon-data/image/captioning/flickr30k
      --task captioning            # controls JSON/sample construction
      --shard-size 1000            # samples per output tar
```
Logic:
1. Glob `--src/*.parquet`. For each parquet + its sibling `.tar` (same basename):
   - Open the local tar (NOT `tar_file` column).
   - For each row: `meta = ast.literal_eval(row["metadata"])`; get caption/QA; read
     `tar.extractfile(row["member_name"]).read()` → jpg bytes.
   - Build sample: key = `row["image_id"]`, write `{key}.jpg` (raw bytes) + `{key}.json`
     (`{"caption": ...}` for captioning; conversation list for QA).
2. Write output as WebDataset shards `shard-%06d.tar` (use `webdataset.ShardWriter`, or `tarfile` manually
   with deterministic ordering) under `--dst`.
3. Print a summary: rows read, samples written, any decode/extract failures (skip + count, don't crash).
4. Be **idempotent / resumable** (skip shards already written).

### Phase D — `energon prepare`
- [ ] `cd energon-data/image/captioning/flickr30k && uv run --no-sync energon prepare .`
  - Choose sample type: `CaptioningSample` (image + caption). For QA sources later: `VQASample` /
    custom. Set the train/val split ratio (e.g. 95/5).
  - This creates `.nv-meta/` (dataset.yaml field_map, split config).
- [ ] Capture the exact prepare answers in the plan/README so other datasets are consistent.

### Phase E — Manifest
- [ ] Append the flickr30k entry to `energon-data/datasets.yaml` with verified `n_samples`.
- [ ] (Later) a small helper to build an energon **metadataset** blend YAML from `datasets.yaml` given a
  category filter + weights.

### Phase F — Verify, then (gated) delete source ⚠️
- [ ] **Verify** the prepared dataset loads: iterate N samples via energon `get_train_dataset`/loader; assert
  image decodes (PIL) and caption present; assert sample count ≈ sum of parquet rows.
- [ ] **Only after** energon-load verification **and** a successful EuroVL training step reading it (Phase G):
  delete `processed-data/flickr30k`. **Deletion must be a separate, explicit, manual step** — never inside
  the converter. Consider keeping a small checksum/manifest of what was deleted.

### Phase G — Bridge integration (separate workstream; read code first)
Wire energon into EuroVL training. **Read these before coding:**
- `src/megatron/bridge/data/energon/energon_provider.py`
- `src/megatron/bridge/data/energon/base_energon_datamodule.py`
- `src/megatron/bridge/data/energon/hf_encoder_task_encoder.py` (+ qwen/nemotron task encoders as references)
- Existing EuroVL consumers to reuse: `src/megatron/bridge/models/euro_vl/euro_vl_processor.py`
  (`EuroVLProcessor`) and `src/megatron/bridge/data/vlm_datasets/collate.py::euro_vl_collate_fn`
  (sample schema = `{"conversation": [...], "image": PIL}`; container = `EuroVLVisualInputs`).
Tasks:
- [ ] Implement a **EuroVL energon TaskEncoder** that maps an energon sample (image + caption/conversation)
  → EuroVL model inputs via `EuroVLProcessor` → `EuroVLVisualInputs` (mirror existing task encoders).
- [ ] Add an energon-backed recipe variant of `euro_vl_2b_sft_config` that replaces
  `MockVLMConversationProvider` with the energon datamodule pointing at a metadataset blend.
- [ ] For PA stage: build the caption blend (image/captioning sources) with weights.

---

## 6. Acceptance criteria (pilot = flickr30k)
1. `energon-data/image/captioning/flickr30k/` exists with shards + `.nv-meta/`.
2. `datasets.yaml` has a correct flickr30k entry (modality=image, task=captioning, verified n_samples).
3. Energon load of N samples succeeds: images decode, captions present, count matches parquet.
4. A short EuroVL training run (Phase G) reads the energon blend and loss decreases.
5. Source `processed-data/flickr30k` deleted **only** after 1–4 pass, as an explicit manual step.

## 7. Generalize after pilot
- Run Phase B–F per remaining **image/captioning** source (cc3m, coco-caption, sbu, textcaps, pixmo, molmo2, wit, sharegpt4o).
- Then image/vqa (a-okvqa), image/ocr (ocr-vqa, textvqa, textocr, coco-text, llavar), image/grounding (grit).
- **vatex (video)** last — needs frame extraction + a video sample type / per-frame handling (EuroVL video
  path uses per-frame 2D MoonViT; see `euro_vl_processor.py` video branch).
- For cc12m-scale, evaluate **Option B (join)** to avoid copying image bytes.

## 8. Open questions to resolve with the owner
- Task taxonomy granularity (e.g. split `captioning` vs `detailed-caption` vs `conversation`?).
- Train/val split ratio for `energon prepare`.
- Blend weights per category for the PA stage.
- energon install path if not present (optional extra vs separate PR).
