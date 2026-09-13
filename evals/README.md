# EuroVL Captioning Eval Harness

Small LLM-as-judge harness for comparing caption quality between two trained
EuroVL checkpoints, across three modalities: single image, multi-image, and
video.

This is a standalone tool, separate from `tests/` and from
`examples/evaluation/` (which deploys a checkpoint via Ray and runs
lm-eval-harness-style text benchmarks). It is not part of the
`megatron.bridge` package and has no dedicated test coverage, consistent
with other one-off scripts under `examples/`.

Run everything as a module from the repo root so the `evals` package's
imports resolve (`uv run python -m evals.<script> ...`, not
`uv run python evals/<script>.py`). `evals.generate_captions` imports
`megatron.core`/`megatron.bridge`, so it must run **inside the apptainer
container** (`./apptainer.sh uv run --no-sync python -m evals.generate_captions ...`,
or `./apptainer.sh uv run --no-sync python -m evals.run_eval ...`). The
dataset builders (`evals.datasets.build_*`) and `evals.judge`/`evals.report`
are plain HF-datasets/HTTP code and don't need the container.

## Known EuroVL caveats

- **Video is 2D-only.** EuroVL encodes video frames independently through
  MoonViT (per-frame 2D features), with no temporal grouping across frames
  the way Qwen-family VLMs do. Weak video captions may reflect this
  architectural limitation rather than a training regression.
- **Multi-image has a known, previously-investigated issue** where the
  second image in a multi-image prompt can be effectively ignored. If
  multi-image results look off, re-check this before assuming a harness bug.

## 1. Build eval manifests

```bash
HF_TOKEN=... uv run python -m evals.datasets.build_single_image --num_samples 20   # ShareGPT-4o (gated)
uv run python -m evals.datasets.build_multi_image --num_samples 20                 # Molmo2-MultiImageQA
uv run python -m evals.datasets.build_video --num_samples 20                       # Molmo2-CapEval (vimeo subset)
```

Each writes `evals/data/<modality>_eval.jsonl` plus downloaded media under
`evals/data/media/`. Schemas were verified live against each dataset (run
`--inspect_only` again if a dataset's schema changes upstream):

- **`build_single_image` (ShareGPT-4o) is gated and works end-to-end** — an
  authenticated, license-accepted `HF_TOKEN` is required (`load_dataset`
  raises `DatasetNotFoundError` without one). Run for real: the config is
  `image_caption`, split `images` (not `train`), schema is `{"image":
  "<filename>.jpg", "width", "height", "conversations": [{"from":
  "human"|"gpt", "value": ...}]}` — the row has no image bytes, only a
  filename. Images live separately in a single ~6.5GB `images.zip` at the
  repo root (internal prefix
  `mnt/petrelfs/wangwenhai/workspace_cef/4o/image/`); the script downloads
  it once via `hf_hub_download` (cached under `$HF_HOME/hub`) and extracts
  only the sampled files.
- **`build_multi_image` (Molmo2-MultiImageQA) works end-to-end** and was
  run for real: schema is `{"image_urls": [url, ...], "image_sha256s":
  [...], "qa_pairs": {"question": [...], "answer": [...]}}`, one row has
  multiple QA pairs over the same image set (only the first is used).
  Images are hotlinked third-party URLs — expect a nontrivial dead-link
  rate (~20-40% in a live 5-sample run) and hotlink-protected CDNs that
  need a browser-like `User-Agent` (already handled in
  `evals/datasets/_common.py::save_image_from_url`).
- **`build_video` (Molmo2-CapEval) works end-to-end for the `vimeo`-sourced
  rows.** Verified columns: `video_id`, `source` (`vimeo`/`ego4d`/`bdd100k`),
  `video_start`/`video_end`/`duration`, `aggregated_caption`,
  `atomic_statements`. The row itself has no video bytes, but — like
  ShareGPT-4o's `images.zip` — the dataset's own repo bundles an official
  `vimeo_videos.zip` (~11.4GB, path `vimeo/<category>/vimeo_<video_id>.mp4`)
  that this script auto-downloads once (`hf_hub_download`, cached) and
  extracts from directly; no scraping involved. `ego4d`/`bdd100k` rows are
  **not** bundled (their README requires downloading from the original
  providers) — those rows are skipped unless you pass `--video_dir` pointing
  at clips you've sourced yourself, named `<video_id>.mp4`.

## 2. Generate captions from each checkpoint

Run once per checkpoint per modality manifest, inside the container. Mirrors
the verified inference path in `sanity_check/euro_vl_test_inference.py`
(`load_megatron_model` + a `Qwen3EuroVLProcessor` built from the checkpoint's
tokenizer + a separate MoonViT HF directory — EuroVL checkpoints are not
`AutoProcessor`-loadable as a single HF repo, so this does not reuse
`examples/conversion/hf_to_megatron_generate_vlm.py`'s generic path):

```bash
./apptainer.sh uv run --no-sync python -m evals.generate_captions \
  --manifest evals/data/image_eval.jsonl \
  --checkpoint /scratch/euro_vl_runs/qwen3_pa_pyav/iter_0004000 \
  --tokenizer_path /scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer \
  --moonvit_path /scratch/hf_models/moonshotai-MoonViT-SO-400M \
  --model_tag pil

./apptainer.sh uv run --no-sync python -m evals.generate_captions \
  --manifest evals/data/image_eval.jsonl \
  --checkpoint /scratch/euro_vl_runs/qwen3_pa_pyav_vect/iter_0004000 \
  --tokenizer_path /scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer \
  --moonvit_path /scratch/hf_models/moonshotai-MoonViT-SO-400M \
  --model_tag vect
```

There is no image-backend choice: `MoonViTImageProcessor` resizes with torchvision (the
former `VectorizedMoonViTImageProcessor`, folded into it 2026-09), so checkpoints trained on
the older PIL backend can no longer be preprocessed exactly as trained. Output:
`evals/results/<model_tag>/<modality>_captions.jsonl`.

## 3. Judge and score

Requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the environment
(vision-capable judge model recommended, since the judge is shown the
source media alongside the caption):

```bash
uv run python -m evals.judge \
  --manifest evals/data/image_eval.jsonl \
  --captions evals/results/v1/image_captions.jsonl \
  --model_tag v1 --judge_provider anthropic --judge_model claude-sonnet-5
```

Scores each caption 1-5 on accuracy, detail, fluency, and hallucination
(absolute rubric, not pairwise), plus an overall score and a short
rationale. Output: `evals/results/judged/<modality>_<model_tag>_scores.jsonl`.

### Local judge (no API key, e.g. for harness testing)

`--judge_provider local` points at any OpenAI-compatible server (default
`http://localhost:8000/v1`, override with `--judge_base_url`) instead of
Anthropic/OpenAI. This was validated against a locally-hosted
`vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ` inside the container. Since that
model is text-only, the judge in this mode does **not** see the source
media — it grades the response against the prompt and the weak reference
caption only (see `_TEXT_ONLY_RUBRIC_INSTRUCTIONS` in `evals/judge.py`).
Treat this as a harness-testing fallback, not a substitute for a
vision-grounded judge when scoring real eval runs.

`--gpu-memory-utilization` defaults to fraction-of-total and vLLM reserves it
eagerly at startup — `0.85` on a 97GB GPU leaves almost nothing for
`evals.generate_captions`'s model load if run concurrently (hit live: a
0.85 server left ~12GB free, and the EuroVL checkpoint load OOM'd). Use a
low value like `0.3` (~29GB, plenty for this 20GB AWQ model at low
concurrency) if you'll run generation and the local judge at the same time
on one GPU:

```bash
./apptainer.sh uv run --no-sync vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --port 8000 --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.3 &

./apptainer.sh uv run --no-sync python -m evals.judge \
  --manifest evals/data/image_eval.jsonl \
  --captions evals/results/v1/image_captions.jsonl \
  --model_tag v1 --judge_provider local --judge_model Qwen/Qwen2.5-32B-Instruct-AWQ
```

## 4. Report

```bash
uv run python -m evals.report
```

Aggregates every judged file under `evals/results/judged/` into
`evals/results/report.md` — a per-modality table of mean scores per model.

## All-in-one

`evals/run_eval.py` runs steps 2-4 for a single manifest and both
checkpoints in one command (must run inside the container, since it shells
out to `evals.generate_captions`):

```bash
ANTHROPIC_API_KEY=... ./apptainer.sh uv run --no-sync python -m evals.run_eval \
  --manifest evals/data/image_eval.jsonl \
  --tokenizer_path /scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer \
  --moonvit_path /scratch/hf_models/moonshotai-MoonViT-SO-400M \
  --checkpoint_a /scratch/euro_vl_runs/qwen3_pa_pyav/iter_0004000 --model_tag_a pil \
  --checkpoint_b /scratch/euro_vl_runs/qwen3_pa_pyav_vect/iter_0004000 --model_tag_b vect
```

Run it once per modality manifest (image / multi_image / video).

## Slurm sweep across training checkpoints

`evals/scripts/` runs the full harness as proper Slurm jobs (not an ad-hoc
background process — see `scripts/slurm_pa.sh` for the repo's usual sbatch
conventions, which these follow) across **25/50/75/100%-of-training**
checkpoints for both runs, one node per checkpoint:

```bash
bash evals/scripts/prepare_manifests.sh          # build manifests — MUST run on the login node, not sbatch
sbatch evals/scripts/run_eval_array.sh            # 8-task array, 1 node/checkpoint
sbatch --dependency=afterok:<array_job_id> evals/scripts/build_report.sh   # aggregate when the array finishes
```

- **`prepare_manifests.sh` is a plain script, not an sbatch job — run it with
  `bash`, on the login node.** This cluster's compute (`booster`) nodes have
  no outbound network access; every manifest builder needs it (HF Hub
  downloads). Submitting this as `sbatch` fails silently: `set -uo pipefail`
  (no `-e`) means a failed download doesn't stop the script, so it exits 0
  and looks like it worked — the array job would then start using stale
  manifests from whatever was on disk before. Everything the array job itself
  touches (checkpoints, tokenizer, MoonViT, the already-cached judge model)
  is local, so it doesn't need network — it sets `HF_HUB_OFFLINE=1` to make
  sure `huggingface_hub` never tries anyway.
- `evals/scripts/select_checkpoints.py` picks checkpoints at ~25/50/75/100%
  of each run's **available** checkpoint history (early checkpoints get
  pruned, so "25% of the final iter number" can point at a directory that no
  longer exists) — run it directly to preview: `python3 -m
  evals.scripts.select_checkpoints /scratch/euro_vl_runs/qwen3_pa_pyav`.
- `evals/scripts/run_eval_array.sh` is a `--array=0-7` job: tasks 0-3 are
  `pil_{25,50,75,100}`, tasks 4-7 are `vect_{25,50,75,100}`. Each task starts
  and manages its own local vLLM judge (killed via an `EXIT` trap), so there's
  no shared judge server to keep alive across the whole sweep.
- Logs land under `logs/euro_vl_eval/`; results under `evals/results/<tag>/`
  and `evals/results/judged/`, same layout as the manual harness above.
