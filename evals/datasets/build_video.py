# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a small video captioning eval manifest from Molmo2-CapEval.

Verified live against the dataset: annotation rows are ``video_id``, ``source``
(``"vimeo"``, ``"ego4d"``, or ``"bdd100k"``), ``video_start``/``video_end``/
``duration``, ``aggregated_caption``, ``atomic_statements``. The dataset's own
README documents where the actual clips come from — Vimeo clips are bundled
in this same HF repo as ``vimeo_videos.zip`` (~11.4GB, official redistribution,
auto-downloaded here via ``hf_hub_download`` the same way
``build_single_image.py`` handles ShareGPT-4o's ``images.zip``) at internal
path ``vimeo/<category>/vimeo_<video_id>.mp4``. Ego4D and BDD100K clips are
NOT bundled — their README requires manually downloading from the original
providers (ego4d-data.org / bdd100k.com) — so this script only samples
``source == "vimeo"`` rows; use ``--video_dir`` to additionally supply
manually-downloaded Ego4D/BDD100K clips (named ``<video_id>.<video_ext>``).

Note (see project memory on EuroVL): EuroVL encodes video as independent 2D
per-frame MoonViT features with no temporal grouping, unlike Qwen-family
temporal merging. Weak video captioning results may reflect this modeling
limitation rather than a bug in this harness.

Run as a module (so the ``evals`` package imports resolve) from the repo root:

Example:
  uv run python -m evals.datasets.build_video --num_samples 20

  # Inspect the raw dataset schema before building (columns vary by config):
  uv run python -m evals.datasets.build_video --inspect_only
"""

import argparse
import logging
import re
import shutil
import zipfile
from pathlib import Path

from evals.common import MEDIA_DIR, EvalItem, write_jsonl
from evals.datasets._common import iter_dataset_rows, reservoir_sample


logger = logging.getLogger(__name__)

HF_DATASET_ID = "allenai/Molmo2-CapEval"
DEFAULT_PROMPT = "Describe this video in detail."
VIMEO_ZIP_FILENAME = "vimeo_videos.zip"
VIMEO_NAME_RE = re.compile(r"vimeo_(\d+)\.mp4$")


def _build_vimeo_index() -> dict[str, str]:
    """Download (or reuse the cached) ``vimeo_videos.zip`` and index video_id -> zip member name."""
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(HF_DATASET_ID, filename=VIMEO_ZIP_FILENAME, repo_type="dataset")
    zf = zipfile.ZipFile(zip_path)
    index = {}
    for name in zf.namelist():
        match = VIMEO_NAME_RE.search(name)
        if match:
            index[match.group(1)] = name
    logger.info("Indexed %d vimeo clips from %s", len(index), VIMEO_ZIP_FILENAME)
    return zf, index


def build(
    num_samples: int, seed: int, out_path: str, video_dir: str | None, video_ext: str, inspect_only: bool
) -> None:
    """Sample ``num_samples`` vimeo-sourced rows from Molmo2-CapEval and write a video manifest.

    Rows from ``ego4d``/``bdd100k`` are only included when a local clip is found
    under ``video_dir`` (as ``<video_id>.<video_ext>``) — those sources aren't
    bundled in the HF repo, see the module docstring.
    """
    rows = iter_dataset_rows(HF_DATASET_ID, split="test")

    if inspect_only:
        first = next(rows)
        logger.info("Columns available: %s", sorted(first.keys()))
        logger.info("Sample row (truncated): %s", {k: str(v)[:200] for k, v in first.items()})
        return

    sampled = reservoir_sample(rows, num_samples, seed)
    vimeo_zip, vimeo_index = _build_vimeo_index()

    items = []
    media_dir = MEDIA_DIR / "video"
    for row in sampled:
        video_id = row["video_id"]
        source = row.get("source")
        dst = media_dir / f"{video_id}.mp4"
        media_dir.mkdir(parents=True, exist_ok=True)

        if source == "vimeo" and video_id in vimeo_index:
            with vimeo_zip.open(vimeo_index[video_id]) as src, open(dst, "wb") as out_f:
                shutil.copyfileobj(src, out_f)
        elif video_dir:
            local = Path(video_dir) / f"{video_id}.{video_ext}"
            if not local.exists():
                logger.warning("Skipping video_id=%s (source=%s): no local clip at %s", video_id, source, local)
                continue
            shutil.copy2(local, dst)
        else:
            logger.warning(
                "Skipping video_id=%s: source=%s not bundled (need --video_dir); see module docstring",
                video_id,
                source,
            )
            continue

        items.append(
            EvalItem(
                id=f"molmo2_cap_{video_id}",
                modality="video",
                media=[str(dst)],
                prompt=DEFAULT_PROMPT,
                reference=row.get("aggregated_caption"),
                source=f"{HF_DATASET_ID} ({source})",
            )
        )

    write_jsonl(items, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build the video captioning eval manifest.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of videos to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--out", type=str, default=str(MEDIA_DIR.parent / "video_eval.jsonl"), help="Output JSONL.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Optional dir of manually-downloaded ego4d/bdd100k clips named '<video_id>.<video_ext>'.",
    )
    parser.add_argument("--video_ext", type=str, default="mp4", help="Local clip file extension.")
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Print the first row's schema and exit, without building the manifest.",
    )
    args = parser.parse_args()
    build(args.num_samples, args.seed, args.out, args.video_dir, args.video_ext, args.inspect_only)
