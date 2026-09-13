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

"""Aggregate judged caption scores into a Markdown comparison report.

Pure aggregation, no network calls — can be re-run cheaply after judging
without hitting the LLM judge API again.

Example:
  uv run python -m evals.report --judged_dir evals/results/judged --out evals/results/report.md
"""

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evals.common import RESULTS_DIR, read_jsonl
from evals.judge import RUBRIC_AXES


logger = logging.getLogger(__name__)


def aggregate(judged_dir: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Return {modality: {model_tag: {axis: mean_score}}} from judged JSONL files."""
    scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path in sorted(Path(judged_dir).glob("*.jsonl")):
        for record in read_jsonl(path):
            modality = record["modality"]
            model_tag = record["model_tag"]
            counts[modality][model_tag] += 1
            for axis in RUBRIC_AXES:
                value = record.get(axis)
                if isinstance(value, (int, float)):
                    scores[modality][model_tag][axis].append(value)

    result: dict[str, dict[str, dict[str, float]]] = {}
    for modality, by_model in scores.items():
        result[modality] = {}
        for model_tag, by_axis in by_model.items():
            result[modality][model_tag] = {
                axis: round(mean(values), 3) if values else float("nan") for axis, values in by_axis.items()
            }
            result[modality][model_tag]["n"] = counts[modality][model_tag]
    return result


def _tag_sort_key(tag: str) -> tuple[str, int]:
    """Sort ``<run>_<pct>`` tags (e.g. ``pil_25``) numerically by percentile, not lexically."""
    match = re.match(r"^(.*)_(\d+)$", tag)
    if match:
        return match.group(1), int(match.group(2))
    return tag, -1


def render_markdown(aggregated: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render the aggregated scores as a Markdown report."""
    lines = ["# EuroVL Captioning Eval Report", ""]
    if not aggregated:
        lines.append("No judged results found.")
        return "\n".join(lines)

    for modality in sorted(aggregated):
        lines.append(f"## {modality}")
        lines.append("")
        model_tags = sorted(aggregated[modality], key=_tag_sort_key)
        header = ["model_tag", "n", *RUBRIC_AXES]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for model_tag in model_tags:
            row = aggregated[modality][model_tag]
            cells = [model_tag, str(row.get("n", 0))] + [str(row.get(axis, "n/a")) for axis in RUBRIC_AXES]
            lines.append("| " + " | ".join(cells) + " |")
        if len(model_tags) >= 2:
            winner = max(model_tags, key=lambda t: aggregated[modality][t].get("overall", float("-inf")))
            lines.append("")
            lines.append(f"**Higher overall score:** `{winner}`")
        lines.append("")

    return "\n".join(lines)


def main(args) -> None:
    """Aggregate judged scores under --judged_dir and write a Markdown report."""
    aggregated = aggregate(args.judged_dir)
    report = render_markdown(aggregated)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote report to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Aggregate judged caption scores into a Markdown report.")
    parser.add_argument("--judged_dir", type=str, default=str(RESULTS_DIR / "judged"), help="Dir of judged JSONL.")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "report.md"), help="Output Markdown path.")
    args = parser.parse_args()
    main(args)
