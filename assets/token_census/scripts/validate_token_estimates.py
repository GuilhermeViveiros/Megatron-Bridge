#!/usr/bin/env python3
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

"""Validate header-only analytic vision-token math before trusting it at scale.

Sizing the PA data mixture to a token budget (40-67B) requires tokens/sample per candidate
dataset, and vision tokens dominate over caption text by 10-50x. Decoding + resizing every
sampled image/video just to count tokens is wasteful: this checks that the SAME token math the
real processors use can be computed from headers alone (image ``.size``, video container
duration + ``codec_context.width/height``) with zero pixel decode.

  image: ``compute_moonvit_visual_tokens`` (megatron.bridge.models.euro_vl.utils) vs.
         ``MoonViTImageProcessor.preprocess`` (real rescale + patchify).
  video: ``plan_num_frames`` + ``per_frame_cap_pixels`` + ``_smart_resize`` (all already on
         ``MoonViTVideoProcessor`` / its module) vs. ``decode_video_bytes`` + ``vectorized_preprocess``
         (real seek-decode + resize).

Run inside the container:
    uv run --no-sync python assets/token_census/scripts/validate_token_estimates.py --n 30
"""

import argparse
import io
import os
import tarfile
from pathlib import Path

import av
from PIL import Image

from megatron.bridge.models.euro_vl.euro_vl_processor import EuroVLProcessor
from megatron.bridge.models.euro_vl.moonvit.video_processing_moonvit import _smart_resize
from megatron.bridge.models.euro_vl.utils import compute_moonvit_visual_tokens


SCRATCH = os.environ["SCRATCH"]
EUROVL_HF = f"{SCRATCH}/hf_models/euro_vl_2b_hf"
# Shared dataset location — independent of the per-user SCRATCH env var (matches
# sanity_check/debug_multiimage_truncation.py's DATASET_DIR convention).
DATA_ROOT = Path("/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data")
IMAGE_SHARD = DATA_ROOT / "image" / "captioning" / "pixmo-cap" / "shard-000000.tar"
VIDEO_SHARD = DATA_ROOT / "video" / "captioning" / "molmo2_cap" / "shard-000000.tar"


def check_images(image_processor, n: int) -> bool:
    """Compare header-only analytic image tokens against the real processor, on real pixmo-cap samples."""
    mismatches = []
    checked = 0
    with tarfile.open(IMAGE_SHARD) as tf:
        names = [m.name for m in tf.getmembers() if m.name.endswith(".jpg")][:n]
        for name in names:
            raw = tf.extractfile(name).read()
            img = Image.open(io.BytesIO(raw))
            w, h = img.size  # header-only; PIL doesn't decode pixels until .load()/.convert()
            analytic = compute_moonvit_visual_tokens((w, h), image_processor)

            out = image_processor.preprocess([img])
            gh, gw = out["image_grid_hws"][0]
            mh, mw = image_processor.merge_kernel_size
            real = (gh // mh) * (gw // mw)

            checked += 1
            if analytic != real:
                mismatches.append((name, (w, h), analytic, real))

    print(f"[image] {checked} samples checked ({IMAGE_SHARD.name}), {len(mismatches)} mismatches")
    for name, size, analytic, real in mismatches[:10]:
        print(f"  {name}: size={size} analytic={analytic} real={real}")
    return not mismatches


def check_videos(video_processor, n: int) -> bool:
    """Compare header-only analytic video tokens against the real processor, on real molmo2_cap samples."""
    mismatches = []
    checked = 0
    with tarfile.open(VIDEO_SHARD) as tf:
        names = [m.name for m in tf.getmembers() if m.name.endswith(".mp4")][:n]
        for name in names:
            raw = tf.extractfile(name).read()

            with av.open(io.BytesIO(raw)) as container:
                stream = container.streams.video[0]
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
                elif container.duration is not None:
                    duration = float(container.duration) / 1_000_000.0
                else:
                    continue
                native_w, native_h = stream.codec_context.width, stream.codec_context.height
                total = max(1, int(duration * float(stream.average_rate))) if stream.average_rate else 10**9

            n_frames = video_processor.plan_num_frames(duration, total)
            cap = video_processor.per_frame_cap_pixels(n_frames)
            factor = video_processor._merge_factor
            new_h, new_w = _smart_resize(
                native_h, native_w, factor=factor, min_pixels=video_processor.min_pixels, max_pixels=cap
            )
            analytic = n_frames * (new_h // factor) * (new_w // factor)

            frames, _ = video_processor.decode_video_bytes(raw)
            out = video_processor.vectorized_preprocess([frames], return_tensors=None)
            t, gh, gw = out["video_grid_thw"][0]
            mh, mw = video_processor.merge_kernel_size
            real = int(t) * (int(gh) // mh) * (int(gw) // mw)

            checked += 1
            if analytic != real:
                mismatches.append((name, (native_w, native_h), duration, analytic, real))

    print(f"[video] {checked} samples checked ({VIDEO_SHARD.name}), {len(mismatches)} mismatches")
    for name, size, duration, analytic, real in mismatches[:10]:
        print(f"  {name}: native={size} duration={duration:.1f}s analytic={analytic} real={real}")
    return not mismatches


def main() -> None:
    """Assert analytic (header-only) vision-token math matches the real MoonViT processors exactly."""
    parser = argparse.ArgumentParser(description="Validate header-only analytic vision-token math")
    parser.add_argument("--n", type=int, default=30, help="Samples to check per modality")
    args = parser.parse_args()

    processor = EuroVLProcessor.from_pretrained(EUROVL_HF, seq_length=8192)
    ok_image = check_images(processor.image_processor, args.n)
    ok_video = check_videos(processor.video_processor, args.n)

    if ok_image and ok_video:
        print("\nPASSED — analytic header-only math matches the real processor exactly.")
    else:
        raise SystemExit("\nFAILED — analytic math diverges from the real processor; do not trust it at scale.")


if __name__ == "__main__":
    main()
