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
category, by modality (image / multiimage / video / text). There are currently no text-only
(zero-vision) datasets in the census -- the "text" bucket is always empty, kept as an explicit
heading rather than silently dropped so a future text-only addition has an obvious home.

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
]

MODALITY_ORDER = ["image", "multiimage", "video", "text"]


def fmt_tokens(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,}"


def load_rows() -> list[dict]:
    rows = []
    for f in SOURCE_FILES:
        d = json.loads(f.read_text())
        rows.extend(d["datasets"])
    return rows


def render_dataset_table(rows: list[dict]) -> str:
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
category by modality (image / multiimage / video / text). There are currently no text-only
(zero-vision) datasets in this census — the "text" subsection is always empty, kept explicit
rather than dropped so a future text-only addition has an obvious home.

Regenerate the raw census: `./apptainer.sh uv run --no-sync python assets/token_census/scripts/estimate_token_budget.py --categories <cats> --workers 16`
(use 16, not 64 — 64 concurrent worker imports exhausted file descriptors on this filesystem).
Regenerate this file from existing raw results: `uv run --no-sync python assets/token_census/scripts/render_readme.py`.

## Overall total

**{len(rows)} datasets censused, {grand_samples:,} samples, {fmt_tokens(grand_total)} tokens (r=1, one epoch each).**

Note: as of this generation, 9 real datasets discovered on disk are NOT yet in this census
(`leopard_arxiv_enriched_translated`, `doc750k`, `docmatix`, `leopard_dude`, `leopard_monkey`,
`leopard_mpdocvqa`, `molmo2_doc` — some appear under both `image` and `multiimage`) — either
missing `.nv-meta` energon indexing or added/renamed after the last census run. A handful of
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
