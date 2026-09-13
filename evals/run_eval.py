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

"""End-to-end orchestrator: generate captions from two checkpoints, judge
them, and produce a comparison report.

Caption generation is run in a subprocess per checkpoint (it initializes
CUDA/torch.distributed and loads a full model, so each run gets a clean
process rather than reusing state across two sequential model loads).
Judging and reporting run in-process since they are pure Python/HTTP.

Run inside the apptainer container (see evals/README.md).

Example:
  ANTHROPIC_API_KEY=... uv run --no-sync python -m evals.run_eval \\
    --manifest evals/data/image_eval.jsonl \\
    --tokenizer_path /scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer \\
    --moonvit_path /scratch/hf_models/moonshotai-MoonViT-SO-400M \\
    --checkpoint_a /scratch/euro_vl_runs/qwen3_pa_pyav/iter_0004000 --model_tag_a pyav \\
    --checkpoint_b /scratch/euro_vl_runs/qwen3_pa_pyav_vect/iter_0004000 --model_tag_b vect
"""

import argparse
import logging
import subprocess
import sys

from evals.common import RESULTS_DIR, read_jsonl, write_json_records
from evals.judge import LLMJudge, judge_captions
from evals.report import aggregate, render_markdown


logger = logging.getLogger(__name__)


def generate(manifest: str, checkpoint: str, model_tag: str, args) -> None:
    """Run evals.generate_captions in a subprocess for one checkpoint."""
    cmd = [
        sys.executable,
        "-m",
        "evals.generate_captions",
        "--manifest",
        manifest,
        "--checkpoint",
        checkpoint,
        "--tokenizer_path",
        args.tokenizer_path,
        "--moonvit_path",
        args.moonvit_path,
        "--model_tag",
        model_tag,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--num_frames",
        str(args.num_frames),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def judge_and_score(manifest_path: str, model_tag: str, judge: LLMJudge) -> None:
    """Judge a model's generated captions and write per-item scores."""
    manifest = read_jsonl(manifest_path)
    modality = manifest[0]["modality"] if manifest else "unknown"
    captions_path = RESULTS_DIR / model_tag / f"{modality}_captions.jsonl"
    captions = read_jsonl(captions_path)

    scored = judge_captions(manifest, captions, judge, model_tag)
    out_path = RESULTS_DIR / "judged" / f"{modality}_{model_tag}_scores.jsonl"
    write_json_records(scored, out_path)


def main(args) -> None:
    """Run the full generate -> judge -> report pipeline for both checkpoints."""
    generate(args.manifest, args.checkpoint_a, args.model_tag_a, args)
    generate(args.manifest, args.checkpoint_b, args.model_tag_b, args)

    judge = LLMJudge(args.judge_provider, args.judge_model, args.judge_api_key, base_url=args.judge_base_url)
    judge_and_score(args.manifest, args.model_tag_a, judge)
    judge_and_score(args.manifest, args.model_tag_b, judge)

    aggregated = aggregate(RESULTS_DIR / "judged")
    report = render_markdown(aggregated)
    out_path = RESULTS_DIR / "report.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote report to %s", out_path)
    logger.info("\n%s", report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="End-to-end caption generation, judging, and reporting.")
    parser.add_argument("--manifest", type=str, required=True, help="Eval manifest JSONL to run.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the checkpoint's tokenizer dir.")
    parser.add_argument("--moonvit_path", type=str, required=True, help="Path to the MoonViT HF config directory.")
    parser.add_argument("--checkpoint_a", type=str, required=True, help="First checkpoint to evaluate.")
    parser.add_argument("--model_tag_a", type=str, required=True, help="Label for the first checkpoint.")
    parser.add_argument("--checkpoint_b", type=str, required=True, help="Second checkpoint to evaluate.")
    parser.add_argument("--model_tag_b", type=str, required=True, help="Label for the second checkpoint.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens for generation.")
    parser.add_argument("--num_frames", type=int, default=4, help="Frames sampled per video.")
    parser.add_argument("--judge_provider", type=str, choices=["anthropic", "openai", "local"], default="anthropic")
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5", help="Judge model id.")
    parser.add_argument("--judge_api_key", type=str, default=None, help="Overrides ANTHROPIC_API_KEY/OPENAI_API_KEY.")
    parser.add_argument(
        "--judge_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for --judge_provider local (default http://localhost:8000/v1).",
    )
    args = parser.parse_args()
    main(args)
