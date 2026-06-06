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

"""EuroVL utilities."""

import math
import random


def format_timestamp(seconds: float, fmt: str = "seconds") -> str:
    """Format a frame timestamp as the inner text of a ``<...>`` marker (Qwen3-VL style).

    EuroVL prefixes each video frame with a textual timestamp so the model can
    perceive temporal information (video grounding, dense captioning). Following
    Qwen3-VL, training mixes two formats so the model learns diverse timecodes.

    Args:
        seconds: Timestamp of the frame in seconds.
        fmt: One of ``"seconds"`` (e.g. ``"3.0 seconds"``), ``"hms"`` (e.g.
            ``"00:00:03"``), or ``"random"`` (pick one per call).

    Returns:
        The inner timestamp string; the caller wraps it as ``f"<{...}>"``.
    """
    if fmt == "random":
        fmt = random.choice(("seconds", "hms"))
    if fmt == "seconds":
        return f"{seconds:.1f} seconds"
    if fmt == "hms":
        total = int(round(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    raise ValueError(f"Unknown timestamp format {fmt!r}; expected 'seconds', 'hms', or 'random'.")


def compute_moonvit_visual_tokens(image_size: tuple[int, int], processor) -> int:
    """Compute the number of projected visual tokens MoonViT will produce for an image.

    Exactly replicates the processor's rescale → patchify → spatial-merge pipeline
    so callers know N before running the vision tower, e.g. to build the right
    number of ``<image>`` placeholder tokens in a prompt.

    Args:
        image_size: ``(width, height)`` of the input image in pixels.
        processor:  ``MoonViTImageProcessor`` instance, providing ``patch_size``,
                    ``in_token_limit``, ``merge_kernel_size``, and ``pad_input``.

    Returns:
        Number of tokens the vision tower will output:
        ``(grid_h // merge_h) * (grid_w // merge_w)``.
    """
    w, h = image_size
    patch_size: int = processor.patch_size
    merge_h, merge_w = processor.merge_kernel_size
    in_token_limit: int = processor.in_token_limit
    pad_input: bool = processor.pad_input

    # Mirror rescale() — scale down if raw patch count exceeds the token limit.
    if (w // patch_size) * (h // patch_size) > in_token_limit:
        scale = math.sqrt(in_token_limit / ((w // patch_size) * (h // patch_size)))
        w, h = int(w * scale), int(h * scale)

    # Mirror rescale() — pad or crop to ensure dimensions are patch-aligned.
    if pad_input:
        pad_size_h = merge_h * patch_size
        pad_size_w = merge_w * patch_size
        h = h + (pad_size_h - h % pad_size_h) % pad_size_h
        w = w + (pad_size_w - w % pad_size_w) % pad_size_w
    else:
        h = h - h % patch_size
        w = w - w % patch_size

    # Mirror patchify() — compute grid in patch units.
    grid_h = h // patch_size
    grid_w = w // patch_size

    return (grid_h // merge_h) * (grid_w // merge_w)
