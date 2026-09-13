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

"""Shared helpers for the eval-set builder scripts under evals/datasets/."""

import logging
import random
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from PIL import Image


logger = logging.getLogger(__name__)

T = TypeVar("T")


def reservoir_sample(rows: Iterable[T], num_samples: int, seed: int) -> list[T]:
    """Reservoir-sample ``num_samples`` rows from a (possibly streaming) iterable.

    Used instead of materializing the full dataset before sampling, since the
    HF datasets here can be loaded with ``streaming=True``.
    """
    rng = random.Random(seed)
    reservoir: list[T] = []
    for i, row in enumerate(rows):
        if i < num_samples:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < num_samples:
                reservoir[j] = row
    return reservoir


def save_image(image: Image.Image, out_path: Path) -> Path:
    """Save a PIL image as JPEG, creating parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, format="JPEG", quality=95)
    return out_path


def save_image_from_url(url: str, out_path: Path) -> Path:
    """Download a hotlinked image URL and save it as JPEG.

    Raises on network/decode failure — callers should catch and skip the row,
    since third-party hotlinked URLs in scraped datasets are commonly dead or
    hotlink-protected. Sends a browser-like User-Agent (many CDNs 403 the
    default urllib UA); this needs a header-carrying request, which
    ``megatron.bridge.utils.safe_url.safe_url_open`` doesn't accept, so the
    same validate-then-redirect-revalidate logic is duplicated here.
    """
    import io
    import urllib.error
    import urllib.request

    from megatron.bridge.utils.safe_url import is_safe_public_http_url

    is_safe, reason = is_safe_public_http_url(url)
    if not is_safe:
        raise ValueError(f"Refusing to fetch image URL ({reason}): {url}")

    class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            is_safe, reason = is_safe_public_http_url(newurl)
            if not is_safe:
                raise urllib.error.URLError(f"redirect blocked ({reason}): {newurl}")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_SafeRedirectHandler())
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Megatron-Bridge-eval/1.0)"})
    with opener.open(request) as resp:  # noqa: S310  -- URL validated above + redirect handler
        image = Image.open(io.BytesIO(resp.read()))
    return save_image(image, out_path)


def iter_dataset_rows(hf_id: str, *, split: str, name: str | None = None) -> Iterator[dict]:
    """Stream rows from a HF dataset, deferring the ``datasets`` import to call time."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, name=name, split=split, streaming=True)
    yield from ds
