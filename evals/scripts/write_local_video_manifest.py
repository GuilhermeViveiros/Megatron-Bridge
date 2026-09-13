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

"""Write a 1-item video eval manifest from the local sanity_check test asset.

Molmo2-CapEval ships no video files (see evals/datasets/build_video.py's
docstring) — this is a stand-in single example, not a benchmark sample.

Example:
  uv run python -m evals.scripts.write_local_video_manifest
"""

import argparse

from evals.common import EvalItem, write_jsonl


DEFAULT_VIDEO = "sanity_check/assets/grok-imagine.mp4"
DEFAULT_PROMPT = "Describe what happens in this video in detail."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO, help="Path to a local video file.")
    parser.add_argument("--out", type=str, default="evals/data/video_eval.jsonl", help="Output JSONL.")
    args = parser.parse_args()

    write_jsonl(
        [
            EvalItem(
                id="video_demo_grok",
                modality="video",
                media=[args.video],
                prompt=DEFAULT_PROMPT,
                reference=None,
                source=f"local asset: {args.video}",
            )
        ],
        args.out,
    )
