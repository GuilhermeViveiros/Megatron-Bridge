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

"""Standalone EuroVL-2B token estimator — text and/or a single image.

Self-contained: this file has NO dependency on megatron.bridge / megatron.core, so anyone can
copy just this one file and run it with a plain `pip install transformers pillow`. It's a
single-file extract of the same math the real token census uses:

  - Vision tokens: an exact copy of ``compute_moonvit_visual_tokens()``
    (``src/megatron/bridge/models/euro_vl/utils.py``) -- mirrors the real ``MoonViTImageProcessor``'s
    rescale -> patchify -> spatial-merge pipeline, using EuroVL-2B's actual config constants
    (patch_size, merge_kernel_size, in_token_limit, pad_input) baked in below instead of loaded
    from a live processor object, so no custom model code needs to be importable. Validated at
    0/60 mismatches against the real processor -- see
    ``assets/token_census/scripts/validate_token_estimates.py``.
  - Text tokens: the real tokenizer (needs a local path or HF repo id for the EuroVL tokenizer).

What this does NOT do (see assets/token_census/scripts/estimate_token_budget.py for the full,
exact, at-scale version used for the actual data mixture census):
  - No chat-template wrapping (system/user/assistant role tokens, special tokens beyond BOS/EOS)
  - No multi-image / video token math
  - No <image> placeholder expansion inside a larger prompt string -- this just reports the
    image's own vision-token count and the sentence's own text-token count side by side

Usage:
    python standalone_estimate_tokens.py --image path/to/photo.jpg
    python standalone_estimate_tokens.py --image-size 1024x768
    python standalone_estimate_tokens.py --text "Describe this chart."
    python standalone_estimate_tokens.py --text "What is shown?" --image path/to/photo.jpg
    python standalone_estimate_tokens.py --text "..." --tokenizer-path /path/to/euro_vl_2b_hf

Dependencies: `pip install transformers pillow` (only `pillow` needed for --image/--image-size;
`transformers` + a real tokenizer path needed for --text).

--text on this cluster: run it inside the apptainer container, not with the bare login-node
python3 -- loading the tokenizer from the shared filesystem is unreliably slow outside the
container (timed out after 90s+ in testing; ~1s inside it):
    ./apptainer.sh uv run --no-sync python assets/token_census/scripts/standalone_estimate_tokens.py --text "..."
--image / --image-size have no such issue and run fine standalone anywhere.
"""

import argparse
import math
import os


# Set before any transformers import: this checkpoint path is always local, so never let a Hub
# connectivity check block/slow a run (compute nodes here have no internet; even where there is
# internet, checking for updates on a fixed local checkpoint is pointless and can hang).
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---- MoonViT vision-token config, from EuroVL-2B's real preprocessor_config.json ----
# (confirmed live against the deployed checkpoint 2026-09-13; update these four constants if the
# checkpoint's vision config ever changes)
PATCH_SIZE = 14
MERGE_KERNEL_SIZE = (2, 2)  # (merge_h, merge_w)
IN_TOKEN_LIMIT = 4096
PAD_INPUT = True

# Must match the training recipe's EUROVL_HF. The older euro_vl_2b_hf export lacks the tokenizer's
# `legacy` flag and mis-tokenizes chat role headers (e.g. "assistant" -> "ass" + "istant").
DEFAULT_TOKENIZER_PATH = os.environ.get("EUROVL_HF")  # local path or HF repo id of the EuroVL tokenizer


def compute_vision_tokens(width: int, height: int) -> int:
    """Number of vision tokens MoonViT will produce for an image of this pixel size.

    Exact replica of ``compute_moonvit_visual_tokens()`` in
    ``src/megatron/bridge/models/euro_vl/utils.py``.
    """
    w, h = width, height
    merge_h, merge_w = MERGE_KERNEL_SIZE

    # Mirror rescale(): scale down if raw patch count exceeds the token limit.
    if (w // PATCH_SIZE) * (h // PATCH_SIZE) > IN_TOKEN_LIMIT:
        scale = math.sqrt(IN_TOKEN_LIMIT / ((w // PATCH_SIZE) * (h // PATCH_SIZE)))
        w, h = int(w * scale), int(h * scale)

    # Mirror rescale(): pad or crop to ensure dimensions are patch-aligned.
    if PAD_INPUT:
        pad_size_h = merge_h * PATCH_SIZE
        pad_size_w = merge_w * PATCH_SIZE
        h = h + (pad_size_h - h % pad_size_h) % pad_size_h
        w = w + (pad_size_w - w % pad_size_w) % pad_size_w
    else:
        h = h - h % PATCH_SIZE
        w = w - w % PATCH_SIZE

    # Mirror patchify(): grid in patch units, then spatial merge.
    grid_h = h // PATCH_SIZE
    grid_w = w // PATCH_SIZE
    return (grid_h // merge_h) * (grid_w // merge_w)


def compute_text_tokens(text: str, tokenizer_path: str) -> int:
    """Token count of plain text with the EuroVL tokenizer (no chat template)."""
    # local_files_only avoids a Hub connectivity check -- compute nodes here have no internet,
    # and even on nodes that do, checking for updates on every run is unnecessary and slow for a
    # tokenizer path that's already a local, fixed checkpoint.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return len(tokenizer.encode(text, add_special_tokens=True))


def parse_image_size(spec: str) -> tuple[int, int]:
    """(width, height) from a WIDTHxHEIGHT string."""
    try:
        w_str, h_str = spec.lower().split("x")
        return int(w_str), int(h_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError("--image-size must look like WIDTHxHEIGHT, e.g. 1024x768") from e


def main() -> None:
    """Print text and/or image token estimates for the command-line inputs."""
    parser = argparse.ArgumentParser(
        description="Estimate EuroVL-2B tokens for a sentence and/or a single image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text", type=str, default=None, help="Sentence/paragraph to tokenize.")
    parser.add_argument("--image", type=str, default=None, help="Path to an image file.")
    parser.add_argument(
        "--image-size", type=str, default=None, help="WIDTHxHEIGHT (e.g. 1024x768), if you don't have the file."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help="Path or HF repo id for the EuroVL tokenizer (only needed with --text). Default: $EUROVL_HF.",
    )
    args = parser.parse_args()

    if not args.text and not args.image and not args.image_size:
        parser.error("Provide at least one of --text, --image, --image-size.")
    if args.image and args.image_size:
        parser.error("Pass only one of --image or --image-size, not both.")
    if args.text and not args.tokenizer_path:
        parser.error("--text needs --tokenizer-path (or $EUROVL_HF).")

    text_tokens = 0
    if args.text:
        text_tokens = compute_text_tokens(args.text, args.tokenizer_path)
        print(f"Text tokens:   {text_tokens}")

    vision_tokens = 0
    width = height = None
    if args.image:
        from PIL import Image

        with Image.open(args.image) as img:
            width, height = img.size  # header-only; PIL doesn't decode pixels for .size
    elif args.image_size:
        width, height = parse_image_size(args.image_size)

    if width is not None:
        vision_tokens = compute_vision_tokens(width, height)
        print(f"Image size:    {width}x{height}")
        print(f"Vision tokens: {vision_tokens}")

    if args.text and width is not None:
        print(f"Total:         {text_tokens + vision_tokens}  (sentence + image, no chat-template overhead)")


if __name__ == "__main__":
    main()
