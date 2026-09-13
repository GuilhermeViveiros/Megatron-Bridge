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

"""Regenerate assets/token_census/README.md from the raw _combined_summary.json files,
grouped by real dataset category (captioning, chart, code, ... 16 total) and, within each
category, by modality (image / multiimage / video / text).

Run: uv run --no-sync python assets/token_census/scripts/render_readme.py
"""

import json
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results"
README_PATH = Path(__file__).parent.parent / "README.md"

SOURCE_FILES = [
    RESULTS_DIR / "captioning_knowledge" / "_combined_summary.json",
    RESULTS_DIR / "ocr" / "_combined_summary.json",
    RESULTS_DIR / "doc" / "_combined_summary.json",
    RESULTS_DIR / "remaining_categories" / "_combined_summary.json",
    RESULTS_DIR / "text" / "_combined_summary.json",
]

MODALITY_ORDER = ["image", "multiimage", "video", "text"]


def fmt_tokens(n: int) -> str:
    """Format a token count as a human-readable B/M/comma-grouped string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,}"


# Censused (real, exact) but intentionally excluded from the rendered README because they're
# not in mixture.yaml's blend -- keeps this file describing the mixture actually being trained
# on, not just "everything that happens to be indexed on disk".
#
# doc750k (both image and multiimage variants) -- dropped 2026-09-11: a 60-sample audit found
# 0% of follow-up questions reference anything visual (the full paper text is given in the
# prompt alongside the page images, and every question is answerable from that text alone).
#
# webmmu -- dropped 2026-09-12: avg 59,280 text tokens/sample (both the full original HTML/CSS/JS
# file and the full rewritten file are given verbatim; one sample alone is ~938k characters). Too
# long-context for this training phase; also only ~28% of instructions describe a visual defect
# the model couldn't diagnose from the code text alone (a 60-sample audit), so the image is
# decorative more often than not on top of the token cost. Revisit for a long-context phase.
EXCLUDED_FROM_MIXTURE = {
    ("image", "doc", "doc750k"),
    ("multiimage", "doc", "doc750k"),
    ("image", "code", "webmmu"),
}


def load_rows() -> list[dict]:
    """Load every censused dataset row, excluding ones dropped from the real mixture."""
    rows = []
    for f in SOURCE_FILES:
        d = json.loads(f.read_text())
        rows.extend(d["datasets"])
    return [r for r in rows if (r["modality"], r["category"], r["name"]) not in EXCLUDED_FROM_MIXTURE]


def render_dataset_table(rows: list[dict]) -> str:
    """Render one markdown table of per-dataset token/image stats, sorted by total tokens."""
    rows = sorted(rows, key=lambda r: r["total_tokens"], reverse=True)
    lines = [
        "| dataset | samples | avg vision | avg text | total vision | total text | avg w | avg h | ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        w = f"{r['avg_image_width']:.0f}" if r.get("avg_image_width") else "-"
        h = f"{r['avg_image_height']:.0f}" if r.get("avg_image_height") else "-"
        ratio = f"{r['avg_image_aspect_ratio']:.2f}" if r.get("avg_image_aspect_ratio") else "-"
        lines.append(
            f"| {r['name']} | {r['n_ok']:,} | {r['avg_vision_tokens']:.0f} | {r['avg_text_tokens']:.0f} | "
            f"{fmt_tokens(r['total_vision_tokens'])} | {fmt_tokens(r['total_text_tokens'])} | {w} | {h} | {ratio} |"
        )
    return "\n".join(lines)


def render_modality_section(modality: str, rows: list[dict], categories: list[str]) -> str:
    """Render one modality's `# heading` section, broken down into per-category `## ` tables."""
    total_tokens = sum(r["total_tokens"] for r in rows)
    total_samples = sum(r["n_ok"] for r in rows)
    parts = [f"# {modality}", ""]

    if not rows:
        note = (
            "No text-only (zero-vision) datasets in this census yet."
            if modality == "text"
            else f"No {modality} datasets censused."
        )
        parts.append(note)
        parts.append("")
        return "\n".join(parts)

    parts.append(f"{len(rows)} dataset(s), {total_samples:,} samples, **{fmt_tokens(total_tokens)} tokens (r=1)**.")
    parts.append("")

    by_category: dict[str, list[dict]] = {c: [] for c in categories}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r)

    for category in categories:
        crows = by_category.get(category, [])
        if not crows:
            continue
        ctotal = sum(r["total_tokens"] for r in crows)
        parts.append(f"## {category}")
        parts.append("")
        parts.append(f"{len(crows)} dataset(s), {fmt_tokens(ctotal)} tokens.")
        parts.append("")
        parts.append(render_dataset_table(crows))
        parts.append("")
        parts.append(f"Raw per-shard data: `assets/token_census/results/*/{modality}__{category}__<name>/*.json`")
        parts.append("")

    return "\n".join(parts)


def main() -> None:
    """Regenerate README.md from the combined per-category census summaries."""
    rows = load_rows()
    categories = sorted({r["category"] for r in rows})
    grand_total = sum(r["total_tokens"] for r in rows)
    grand_samples = sum(r["n_ok"] for r in rows)

    header = f"""# EuroVL data mixture — exact token census

Exact (not sampled) text+vision token counts per energon dataset, computed by
`assets/token_census/scripts/estimate_token_budget.py`. Every sample of every shard is counted —
no extrapolation. Vision tokens are computed analytically (image/video headers only, no pixel
decode) using the same math the real MoonViT processors use; validated at 0/60 mismatches
against the real end-to-end `EuroVLProcessor` output (image, multiimage, video, including the
real per-frame timestamp text) before running at scale — see
`assets/token_census/scripts/validate_token_estimates.py`.

Organized by real dataset category (matching `mixture.yaml`'s taxonomy), and within each
category by modality (image / multiimage / video / text). Text-only (zero-vision) samples are
tokenized the same way minus the image/video expansion step — no pixel decode is needed there
either, since there's nothing to decode.

Text-only samples also get a placeholder-sanitization pass (`_sanitize_text_only_placeholders`
in `estimate_token_budget.py`) before tokenizing: some text corpora (found 2026-09-12 in
`code/euroblocks`, a Codeforces-style competitive-programming source) retain a literal `<image>`
placeholder from their original source even though this text-only dataset carries no actual
image file. Left as-is, that tokenizes to the same id as a real vision slot (confirmed against
the live tokenizer) — a real training-time hazard the actual text-SFT encoder path needs its own
fix for, independent of this census script's sanitization.

Regenerate the raw census: `./apptainer.sh uv run --no-sync python assets/token_census/scripts/estimate_token_budget.py --categories <cats> --workers 16`
(use 16, not 64 — 64 concurrent worker imports exhausted file descriptors on this filesystem).
Regenerate this file from existing raw results: `uv run --no-sync python assets/token_census/scripts/render_readme.py`.

## Overall total

**{len(rows)} datasets censused, {grand_samples:,} samples, {fmt_tokens(grand_total)} tokens (r=1, one epoch each).**

Note: `doc750k` (both `image` and `multiimage` variants) and `webmmu` are censused but
deliberately excluded from this file and from `mixture.yaml`. `doc750k`: a 60-sample audit found
0% of its follow-up questions reference anything visual (the full paper text is given in the
prompt alongside the page images, so every question is answerable from that text alone).
`webmmu`: avg 59,280 text tokens/sample (full original + full rewritten HTML/CSS/JS file, one
sample alone ~938k characters) is too long-context for this training phase, and only ~28% of its
edit instructions describe a visual defect the code text alone doesn't reveal. A handful of
`_buggy_backup`/`_preshuffle_backup` directories also exist on disk and are intentionally excluded
(not real additional data).

| category | tokens (r=1) |
|---|---|
"""
    for cat in sorted(categories, key=lambda c: -sum(r["total_tokens"] for r in rows if r["category"] == c)):
        cat_total = sum(r["total_tokens"] for r in rows if r["category"] == cat)
        header += f"| {cat} | {fmt_tokens(cat_total)} |\n"
    header += f"| **TOTAL** | **{fmt_tokens(grand_total)}** |\n"

    sections = []
    for modality in MODALITY_ORDER:  # image, multiimage, video, text (text last -- future data)
        mrows = [r for r in rows if r["modality"] == modality]
        sections.append(render_modality_section(modality, mrows, categories))

    README_PATH.write_text(header + "\n" + "\n".join(sections))
    print(f"Wrote {README_PATH} ({len(rows)} datasets, {len(categories)} categories)")


if __name__ == "__main__":
    main()
