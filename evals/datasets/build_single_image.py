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

"""Build a small single-image captioning eval manifest from ShareGPT-4o.

``OpenGVLab/ShareGPT-4o`` is a gated dataset on the Hub — loading it requires
an authenticated ``HF_TOKEN`` with the dataset's license accepted (log in at
https://huggingface.co/datasets/OpenGVLab/ShareGPT-4o first).

Schema verified live against the dataset's ``image_caption`` config, split
``images`` (not ``train``): columns are ``{"image": "<filename>.jpg",
"width", "height", "conversations": [{"from": "human"|"gpt", "value": ...}]}``
— the row does not embed image bytes, only a filename. The actual images
ship separately as a single ~6.5GB ``images.zip`` at the repo root (internal
path prefix ``mnt/petrelfs/wangwenhai/workspace_cef/4o/image/``). This
script downloads that zip once via ``hf_hub_download`` (cached under
``$HF_HOME/hub`` for subsequent runs) and extracts only the sampled images.

Run as a module (so the ``evals`` package imports resolve) from the repo root:

Example:
  HF_TOKEN=... uv run python -m evals.datasets.build_single_image --num_samples 20

  # Inspect the raw dataset schema before building (columns vary by config):
  HF_TOKEN=... uv run python -m evals.datasets.build_single_image --inspect_only
"""

import argparse
import io
import logging
import zipfile

from PIL import Image

from evals.common import MEDIA_DIR, EvalItem, write_jsonl
from evals.datasets._common import iter_dataset_rows, reservoir_sample, save_image


logger = logging.getLogger(__name__)

HF_DATASET_ID = "OpenGVLab/ShareGPT-4o"
HF_CONFIG_NAME = "image_caption"
HF_SPLIT = "images"
IMAGES_ZIP_PREFIX = "mnt/petrelfs/wangwenhai/workspace_cef/4o/image/"
DEFAULT_PROMPT = "Describe this image in detail."


def _extract_caption(conversations: list[dict]) -> str | None:
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == "gpt" and isinstance(turn.get("value"), str):
            return turn["value"].strip()
    return None


def _open_images_zip() -> zipfile.ZipFile:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_DATASET_ID, filename="images.zip", repo_type="dataset")
    return zipfile.ZipFile(path)


def build(num_samples: int, seed: int, out_path: str, inspect_only: bool) -> None:
    """Sample ``num_samples`` rows from ShareGPT-4o and write an image eval manifest."""
    rows = iter_dataset_rows(HF_DATASET_ID, split=HF_SPLIT, name=HF_CONFIG_NAME)

    if inspect_only:
        first = next(rows)
        logger.info("Columns available: %s", sorted(first.keys()))
        logger.info("Sample row (truncated): %s", {k: str(v)[:200] for k, v in first.items()})
        return

    sampled = reservoir_sample(rows, num_samples, seed)
    images_zip = _open_images_zip()

    items = []
    media_dir = MEDIA_DIR / "image"
    for i, row in enumerate(sampled):
        filename = row.get("image")
        if not filename:
            logger.warning("Skipping row %d: no image filename", i)
            continue
        try:
            data = images_zip.read(IMAGES_ZIP_PREFIX + filename)
        except KeyError:
            logger.warning("Skipping row %d: %s not found in images.zip", i, filename)
            continue
        image = Image.open(io.BytesIO(data))
        image_path = save_image(image, media_dir / f"sharegpt4o_{i:04d}.jpg")
        items.append(
            EvalItem(
                id=f"sharegpt4o_{i:04d}",
                modality="image",
                media=[str(image_path)],
                prompt=DEFAULT_PROMPT,
                reference=_extract_caption(row.get("conversations") or []),
                source=f"{HF_DATASET_ID}/{HF_CONFIG_NAME}",
            )
        )

    write_jsonl(items, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build the single-image captioning eval manifest.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--out", type=str, default=str(MEDIA_DIR.parent / "image_eval.jsonl"), help="Output JSONL.")
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Print the first row's schema and exit, without building the manifest.",
    )
    args = parser.parse_args()
    build(args.num_samples, args.seed, args.out, args.inspect_only)
