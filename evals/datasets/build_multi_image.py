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

"""Build a small multi-image eval manifest from Molmo2-MultiImageQA.

There is no dedicated multi-image *captioning* benchmark, so this reuses a
multi-image QA dataset: each item's ``prompt`` is one of the dataset's
questions rather than a generic "describe" instruction, and ``reference`` is
the corresponding answer (weak hint for the judge, not gold truth for
captioning). Schema verified live against the dataset:
``{"image_urls": [url, ...], "image_sha256s": [...], "qa_pairs": {"question":
[...], "answer": [...]}}`` — each row has multiple QA pairs over the same
image set; this uses only the first.

Images are hotlinked third-party URLs and commonly dead; rows with any
unreachable image are skipped.

Run as a module (so the ``evals`` package imports resolve) from the repo root:

Example:
  uv run python -m evals.datasets.build_multi_image --num_samples 20

  # Inspect the raw dataset schema before building (columns vary by config):
  uv run python -m evals.datasets.build_multi_image --inspect_only
"""

import argparse
import logging

from evals.common import MEDIA_DIR, EvalItem, write_jsonl
from evals.datasets._common import iter_dataset_rows, reservoir_sample, save_image_from_url


logger = logging.getLogger(__name__)

HF_DATASET_ID = "allenai/Molmo2-MultiImageQA"


def build(num_samples: int, seed: int, out_path: str, inspect_only: bool) -> None:
    """Sample ``num_samples`` rows from Molmo2-MultiImageQA and write a multi_image manifest."""
    rows = iter_dataset_rows(HF_DATASET_ID, split="train")

    if inspect_only:
        first = next(rows)
        logger.info("Columns available: %s", sorted(first.keys()))
        logger.info("Sample row (truncated): %s", {k: str(v)[:200] for k, v in first.items()})
        return

    sampled = reservoir_sample(rows, num_samples, seed)

    items = []
    media_dir = MEDIA_DIR / "multi_image"
    for i, row in enumerate(sampled):
        urls = row.get("image_urls") or []
        qa_pairs = row.get("qa_pairs") or {}
        questions = qa_pairs.get("question") or []
        answers = qa_pairs.get("answer") or []
        if len(urls) < 2 or not questions:
            logger.warning("Skipping row %d: needs >=2 images and >=1 QA pair", i)
            continue

        media_paths = []
        try:
            for j, url in enumerate(urls):
                media_paths.append(str(save_image_from_url(url, media_dir / f"molmo2_mi_{i:04d}_{j:02d}.jpg")))
        except Exception as e:
            logger.warning("Skipping row %d: failed to download an image (%s)", i, e)
            continue

        items.append(
            EvalItem(
                id=f"molmo2_mi_{i:04d}",
                modality="multi_image",
                media=media_paths,
                prompt=questions[0],
                reference=answers[0] if answers else None,
                source=HF_DATASET_ID,
            )
        )

    write_jsonl(items, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build the multi-image eval manifest.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--out", type=str, default=str(MEDIA_DIR.parent / "multi_image_eval.jsonl"), help="Output JSONL."
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Print the first row's schema and exit, without building the manifest.",
    )
    args = parser.parse_args()
    build(args.num_samples, args.seed, args.out, args.inspect_only)
