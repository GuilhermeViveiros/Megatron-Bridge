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

"""Regenerate assets/token_census/README.md from the census summaries in assets/token_census/results/.

Run: python assets/token_census/scripts/report/render_readme.py [--mixture PATH]

``--mixture`` (or ``$EUROVL_MIXTURE``, else ``$EUROVL_DATA_ROOT/mixture.yaml``) is the live mixture:
repeat factors are refreshed from it and datasets no longer in it are dropped. Without it the
factors recorded at census time are used.
"""

import argparse
import json
import os
from pathlib import Path

import yaml


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
README_PATH = Path(__file__).resolve().parents[2] / "README.md"
MODALITY_ORDER = ["image", "multiimage", "video", "text"]
SOURCE_FILES = [RESULTS_DIR / f"{m}_summary.json" for m in MODALITY_ORDER]

# The census only covers mixture.yaml entries, so these are no longer censused at all; kept so
# a stray older result can never re-enter the totals. Why they were dropped: see git history of
# this file (doc750k: 0% image-grounded follow-up questions). webmmu is back in the mixture after its
# long-context samples were filtered out (moved to image/code/webmmu_long, which is not in the mixture).
EXCLUDED_FROM_MIXTURE = {
    ("image", "doc", "doc750k"),
    ("multiimage", "doc", "doc750k"),
}


def eff(r: dict) -> float:
    """Tokens training sees per epoch of the mixture: the mixture repeat factor r times the dataset's tokens."""
    return float(r.get("mixture_r", 1.0)) * r["total_tokens"]


def fmt_tokens(n: float) -> str:
    """Format a token count as a human-readable B/M/comma-grouped string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{int(n):,}"


def default_mixture() -> Path | None:
    """The live mixture.yaml from ``$EUROVL_MIXTURE`` or ``$EUROVL_DATA_ROOT``, if either is set."""
    if os.environ.get("EUROVL_MIXTURE"):
        return Path(os.environ["EUROVL_MIXTURE"])
    if os.environ.get("EUROVL_DATA_ROOT"):
        return Path(os.environ["EUROVL_DATA_ROOT"]) / "mixture.yaml"
    return None


def live_mixture_weights(mixture: Path | None) -> dict[tuple[str, str, str], float]:
    """{(modality, category, name): r} from the CURRENT mixture.yaml, so weights changed after the census
    still show up. A key containing '/' is a full relative path (name present in two categories)."""
    out = {}
    if mixture is None or not mixture.exists():
        return out
    for modality, cats in (yaml.safe_load(mixture.read_text()) or {}).items():
        for category, entries in (cats or {}).items():
            for name, r in (entries or {}).items():
                if "/" in name:
                    _, category, name = name.rsplit("/", 2)
                out[(modality, category, name)] = float(r)
    return out


def load_rows(mixture: Path | None) -> list[dict]:
    """Every censused dataset row across the four modalities (missing modalities are skipped).

    `mixture_r` is refreshed from the live mixture.yaml; rows whose dataset left the mixture are dropped.
    """
    rows = []
    for f in SOURCE_FILES:
        if f.exists():
            rows.extend(json.loads(f.read_text())["datasets"])
    rows = [r for r in rows if (r["modality"], r["category"], r["name"]) not in EXCLUDED_FROM_MIXTURE]
    weights = live_mixture_weights(mixture)
    if not weights:
        return rows
    by_name = {}  # mixture category can differ from the on-disk one (e.g. temporal_grounding vs temporal-grounding)
    for (m, c, n), r in weights.items():
        by_name.setdefault((m, n), []).append(r)
    kept = []
    for r in rows:
        key = (r["modality"], r["category"], r["name"])
        if key in weights:
            r["mixture_r"] = weights[key]
        elif len(by_name.get((r["modality"], r["name"]), [])) == 1:
            r["mixture_r"] = by_name[(r["modality"], r["name"])][0]
        else:
            continue  # not in the current mixture
        kept.append(r)
    return kept


def load_skipped() -> list[str]:
    """Mixture entries the census could not resolve or count, from every modality summary."""
    out = []
    for f in SOURCE_FILES:
        if f.exists():
            out += json.loads(f.read_text()).get("skipped", [])
    return out


def render_language_section() -> str:
    """Per-language token counts and the English / multilingual split (compute_language_split.py)."""
    path = RESULTS_DIR / "language_split.json"
    if not path.exists():
        return ""
    d = json.loads(path.read_text())
    g = d["grand_total_tokens"]
    en, ml, und, eu = d["en_tokens"], d["multilingual_tokens"], d["und_tokens"], d["eu_non_en_tokens"]
    sc = d["sources_count"]
    lines = [
        "## Languages",
        "",
        f"**{100 * en / g:.2f}% English, {100 * ml / g:.2f}% other languages, {100 * und / g:.2f}% undetermined** "
        f"(of {fmt_tokens(g)} tokens). Non-English EU official languages alone: "
        f"**{fmt_tokens(eu)} ({100 * eu / g:.2f}%)**.",
        "",
    ]
    e = d.get("effective")
    if e:
        ge = e["grand_total_tokens"] or 1
        lines += [
            f"**Effective (r×tokens):** {100 * e['en_tokens'] / ge:.2f}% English, "
            f"{100 * e['multilingual_tokens'] / ge:.2f}% other languages, {100 * e['und_tokens'] / ge:.2f}% undetermined "
            f"(of {fmt_tokens(e['grand_total_tokens'])} tokens); non-English EU official languages "
            f"{fmt_tokens(e['eu_non_en_tokens'])} ({100 * e['eu_non_en_tokens'] / ge:.2f}%).",
            "",
        ]
    moved = sum(r.get("und_counted_as_en", 0.0) for r in d.get("per_dataset", []))
    if moved:
        lines += [
            f"Samples with too little natural language for detection (labels + coordinates, GUI actions, table "
            f"cells, LaTeX, code) come out undetermined; their questions are English, so in every dataset that is "
            f"**not multilingual** they are counted as **English** ({fmt_tokens(moved)}). Undetermined tokens left "
            f"in multilingual datasets stay undetermined.",
            "",
        ]
    lines += [
        "How each dataset's tokens are attributed to languages (first source that covers it):",
        "",
        f"1. **prorated** ({sc['prorated']} datasets) — `smurf4eu-vision-curator/data_sync/language_counts.md` "
        "(curation-side per-language *sample* counts, from source metadata where present). "
        "`tokens(lang) ≈ total_tokens × samples(lang) / samples`: exact for samples, approximate for tokens "
        "(assumes equal tokens/sample across languages — tight where vision tokens dominate, weaker for "
        "caption-heavy sets like `wit`, `culturalground_*`).",
        f"2. **exact** ({sc['exact']} datasets: all video + datasets newer than the curation file) — the census "
        "attributes each sample's real token count (each *branch* for message trees; shared video tokens split "
        "by branch text share) to a language: the sample's own `meta.language` when declared, else langdetect "
        "restricted to confident calls (≥20 letters, ≥8 distinct words, p ≥ 0.90; assistant text, then user "
        "text, then both), else `und`. Stricter than the curation rule on purpose: answers that are "
        "coordinates, JSON, timestamps, option letters or 2-word labels were confidently misdetected "
        '(e.g. `tapos` 72% "da", MCQ letters "hu"). Residual known error is <1% of video tokens '
        '(`crosstask` step labels ~9% "fr"; English captions quoting Chinese signs → `zh`). Detection '
        "agrees with declared labels where both exist (`eurovideolm`: 51/49 detected vs 52/48 declared bg/en).",
        f"3. **assumed-en** ({sc['assumed-en']} datasets) — not flagged as multilingual by curation.",
        "",
        "`und` = no confident language (structured/numeric answers), not necessarily non-English.",
        "",
        "| language | tokens | % of mixture | EU official | EuroLLM | source(s) |",
        "|---|---|---|---|---|---|",
    ]
    for L in d["languages"]:
        lines.append(
            f"| {L['lang']} | {fmt_tokens(L['tokens'])} | {L['pct']:.3f}% | {'✓' if L['eu_official'] else ''} | "
            f"{'✓' if L['eurollm'] else '⚠️'} | {', '.join(L['sources'])} |"
        )
    lines += [
        "",
        "Datasets with non-English content, by multilingual token volume:",
        "",
        "| dataset | modality | tokens | source | en % | other % | und % | top languages |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in d["per_dataset"]:
        if r["multilingual_pct"] < 0.5:
            continue
        top = ", ".join(
            f"{k} {100 * v / r['tokens']:.0f}%" for k, v in r["top_languages"].items() if k not in ("en", "und")
        )
        lines.append(
            f"| {r['category']}/{r['name']} | {r['modality']} | {fmt_tokens(r['tokens'])} | {r['source']} | "
            f"{r['en_pct']:.1f} | {r['multilingual_pct']:.1f} | {r['und_pct']:.1f} | {top} |"
        )
    if d["doc_unmatched"]:
        lines += [
            "",
            "`language_counts.md` sections with no dataset in the mixture: "
            + ", ".join(f"`{n}`" for n in d["doc_unmatched"])
            + ".",
        ]
    lines.append("")
    return "\n".join(lines)


def render_dataset_table(rows: list[dict]) -> str:
    """Markdown table with one line per dataset."""
    rows = sorted(rows, key=lambda r: r["total_tokens"], reverse=True)
    lines = [
        "| dataset | r | samples | avg vision | avg text | total vision | total text | effective (r×) | avg w | avg h | ratio |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        w = f"{r['avg_image_width']:.0f}" if r.get("avg_image_width") else "-"
        h = f"{r['avg_image_height']:.0f}" if r.get("avg_image_height") else "-"
        ratio = f"{r['avg_image_aspect_ratio']:.2f}" if r.get("avg_image_aspect_ratio") else "-"
        trees = f" ({r['n_message_trees']:,} message trees)" if r.get("n_message_trees") else ""
        lines.append(
            f"| {r['name']}{trees} | {float(r.get('mixture_r', 1.0)):g} | {r['n_ok']:,} | {r.get('avg_vision_tokens', 0):.0f} | "
            f"{r.get('avg_text_tokens', 0):.0f} | {fmt_tokens(r['total_vision_tokens'])} | {fmt_tokens(r['total_text_tokens'])} | "
            f"{fmt_tokens(eff(r))} | {w} | {h} | {ratio} |"
        )
    return "\n".join(lines)


def render_modality_section(modality: str, rows: list[dict]) -> str:
    """One modality's section: totals, then a dataset table per category."""
    if not rows:
        return f"# {modality}\n\nNo {modality} datasets censused.\n"
    parts = [
        f"# {modality}",
        "",
        f"{len(rows)} dataset(s), {sum(r['n_ok'] for r in rows):,} samples, "
        f"**{fmt_tokens(sum(r['total_tokens'] for r in rows))} tokens (r=1)**, "
        f"**{fmt_tokens(sum(eff(r) for r in rows))} effective (r×tokens)**.",
        "",
    ]
    for category in sorted({r["category"] for r in rows}):
        crows = [r for r in rows if r["category"] == category]
        parts += [
            f"## {category}",
            "",
            f"{len(crows)} dataset(s), {fmt_tokens(sum(r['total_tokens'] for r in crows))} tokens.",
            "",
            render_dataset_table(crows),
            "",
        ]
    return "\n".join(parts)


def render_domain_section(rows: list[dict]) -> str:
    """Code and math token totals, by mixture category (`code`, `math`) and modality.

    Two measures: answer tokens (the supervised positions of the training loss mask, i.e. the
    code/math the model learns to write) and sample tokens (everything in those samples, including
    images, instructions and the chat template).
    """
    grand = sum(r["total_tokens"] for r in rows) or 1
    lines = [
        "## Code and math",
        "",
        "Datasets in the mixture's `code` and `math` categories (formula/code OCR sets under `ocr` are not "
        "included). **Answer tokens** are the supervised positions of the training loss mask (each assistant "
        "answer plus its closing `<|im_end|>`, read off the template markers by the encoder's own "
        "`assistant_answer_mask`; checked against the real encoder, 0 mismatches): the code/math "
        "the model learns to write. **Sample tokens** are everything in those samples (images, instructions, "
        "chat template). Shares are of all tokens (r=1).",
        "",
    ]
    for cat in ("code", "math"):
        crow = [r for r in rows if r["category"] == cat]
        t = sum(r["total_tokens"] for r in crow)
        a = sum(r.get("total_answer_tokens", 0) for r in crow)
        lines += [
            f"### {cat}: {fmt_tokens(a)} answer tokens ({100 * a / grand:.2f}%), "
            f"{fmt_tokens(t)} sample tokens ({100 * t / grand:.2f}%)",
            "",
            "| modality | datasets | samples | answer tokens | % of all | sample tokens | % of all |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in MODALITY_ORDER:
            mr = [r for r in crow if r["modality"] == m]
            if mr:
                mt = sum(r["total_tokens"] for r in mr)
                ma = sum(r.get("total_answer_tokens", 0) for r in mr)
                lines.append(
                    f"| {m} | {len(mr)} | {sum(r['n_ok'] for r in mr):,} | {fmt_tokens(ma)} | "
                    f"{100 * ma / grand:.2f}% | {fmt_tokens(mt)} | {100 * mt / grand:.2f}% |"
                )
        lines += [
            "",
            "Datasets (answer / sample tokens): "
            + ", ".join(
                f"`{r['modality']}/{r['name']}` ({fmt_tokens(r.get('total_answer_tokens', 0))} / {fmt_tokens(r['total_tokens'])})"
                for r in sorted(crow, key=lambda r: -r.get("total_answer_tokens", 0))
            ),
            "",
        ]
    return "\n".join(lines)


def render_data_issues(rows: list[dict], skipped: list[str]) -> str:
    """Samples the census could not count and mixture entries it could not resolve."""
    lines = ["## Data issues found by the census", ""]
    bad = [r for r in rows if r["n_failed"]]
    if bad:
        lines += [
            "Samples the census could not count (the training encoder would also reject or mis-handle "
            "them). Reasons are the census's per-sample errors:",
            "",
            "| dataset | failed | of | top reason |",
            "|---|---|---|---|",
        ]
        for r in sorted(bad, key=lambda r: -r["n_failed"]):
            reason = next(iter(r.get("fail_reasons") or {"?": 0}))
            lines.append(
                f"| {r['modality']}/{r['category']}/{r['name']} | {r['n_failed']:,} | "
                f"{r['n_ok'] + r['n_failed']:,} | `{reason}` |"
            )
        lines.append("")
    if skipped:
        lines += ["In `mixture.yaml` but not censusable:", ""] + [f"- `{s}`" for s in skipped] + [""]
    lines += [
        "Everything else found by the sampled data audit (answer-format, language, timestamp, "
        "index and training-code problems, with sample keys): [`BUGS.md`](BUGS.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write README.md from the summaries in results/."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mixture", type=Path, default=default_mixture())
    args = parser.parse_args()
    rows = load_rows(args.mixture)
    skipped = load_skipped()
    grand = sum(r["total_tokens"] for r in rows)
    grand_eff = sum(eff(r) for r in rows)
    val_tokens = sum(r.get("non_train_tokens", 0) for r in rows)
    samples = sum(r["n_ok"] for r in rows)
    categories = sorted({r["category"] for r in rows})

    header = f"""# EuroVL data mixture — exact token census

Exact (not sampled) text+vision token counts for **every dataset in `mixture.yaml`**. Two totals:
**r=1** counts every mixture entry once (repeat factors ignored, including `r=0` entries), and
**effective** = Σ r × tokens, i.e. what one pass over the mixture feeds the model (`r=0` entries
contribute nothing, `r=2` counts twice). Every sample of every shard is counted by
`assets/token_census/scripts/census/estimate_token_budget.py`; nothing is extrapolated.

**How the counts are made and checked**

- **Tokenizer:** `euro_vl_2b_2512_hf`, updated 10-03 with the 10 grounding markers
  (`<|object_ref_start|>`…`<|temporal_end|>`, incl. box/point/quad) as single special tokens — each
  grounding block (ref + box/point/span) is 26 tokens shorter than with the previous vocabulary. (An earlier census
  used `euro_vl_2b_hf`, whose `tokenizer_config.json` had lost the `legacy` flag and split role headers
  like `assistant` into `ass`+`istant`. Fixing it moves text by ~+1 token per turn: +0.06% to +0.43% of
  totals on a shard-for-shard spot-check; vision tokens are unaffected.)
- **Validated against the real training encoder:** 132 samples across every format (image,
  multi-image, flat video, message-tree video, temporal grounding, text), plus 100 more video samples
  spread over 5 shards each of `activitynet_new` (flat) and `eurovideolm_packed` (message tree) — the
  census count equals `EuroVLTaskEncoder` `input_ids` length exactly, total and vision, 0 mismatches.
  Re-run 10-03 with the updated tokenizer on 164 samples, adding grounding / pointing / temporal data
  (refcoco_ground, refcoco_point, mi_o365, sav_grounding, sav_tracking, ava_temporal_grounding): 0 mismatches. Message-tree
  totals also match the curation side's own `meta.rendered_tokens` exactly on every dataset that
  carries it (`ego4d_narrations`: curation estimate is 0.3% below the real count).
  Re-run 10-04 with answer (loss-masked) tokens added, on 254 samples of 34 datasets incl. code and math
  (`scripts/validation/check_against_encoder.py`): total, vision and answer tokens (= positions with
  `loss_mask > 0`) all equal, 0 mismatches.
- **Vision** is computed from image/video headers only (no pixel decode) with the processor's own
  resize math; **video** counts assume the default `seq_length=8192` video budget
  (`total_pixels = 0.85 × 8192 × 28²`; a 16K run gets a larger budget and more tokens per video).
- **Message-tree samples** (`{{"message_tree": true, "shared", "branches"}}`) are counted the way training
  renders them: shared turns + every branch in one sequence, at full length.
- **Pre-filter counts:** no `seq_length` truncation/overflow skipping and no `max_num_images` skip —
  this is what the data contains, not what survives the training-time filters.
- **Which shards:** exactly the ones energon reads — the `split.yaml` parts of each dataset (so
  e.g. `train-*.tar` names count too). **All splits are counted** (train + val); the val share is
  shown separately ({fmt_tokens(val_tokens)} tokens in total). A mixture entry is resolved by dataset
  name under any category folder, like training does.
- **Samples are grouped like webdataset/energon** (consecutive tar members sharing a key), so a key
  that repeats inside one shard counts as separate samples, as training reads them.
- **Text-only** samples have any literal `<image>`/`<video>` placeholder neutralized before tokenizing
  (it would otherwise tokenize as a real vision slot).

## Overall total

**{len(rows)} datasets, {samples:,} samples, {fmt_tokens(grand)} tokens (r=1); {fmt_tokens(grand_eff)} effective (r×tokens).**

| modality | datasets | samples | tokens (r=1) | % | effective (r×) | % |
|---|---|---|---|---|---|---|
"""
    for m in MODALITY_ORDER:
        mr = [r for r in rows if r["modality"] == m]
        t = sum(r["total_tokens"] for r in mr)
        e = sum(eff(r) for r in mr)
        header += (
            f"| {m} | {len(mr)} | {sum(r['n_ok'] for r in mr):,} | {fmt_tokens(t)} | {100 * t / grand:.1f}% | "
            f"{fmt_tokens(e)} | {100 * e / max(grand_eff, 1):.1f}% |\n"
        )
    header += "\n| category | tokens (r=1) | effective (r×) |\n|---|---|---|\n"
    for cat in sorted(categories, key=lambda c: -sum(r["total_tokens"] for r in rows if r["category"] == c)):
        crow = [r for r in rows if r["category"] == cat]
        header += f"| {cat} | {fmt_tokens(sum(r['total_tokens'] for r in crow))} | {fmt_tokens(sum(eff(r) for r in crow))} |\n"
    header += f"| **TOTAL** | **{fmt_tokens(grand)}** | **{fmt_tokens(grand_eff)}** |\n\n"
    header += (
        render_language_section()
        + "\n"
        + render_domain_section(rows)
        + "\n"
        + render_data_issues(rows, skipped)
        + "\n"
    )
    header += (
        "## Regenerating\n\n"
        "Inside the container, with `EUROVL_DATA_ROOT` (energon data root holding `mixture.yaml`), `EUROVL_HF` "
        "(the model's HF export) and `EUROVL_CENSUS_WORK_DIR` (per-shard results, not in git) set. Use about 16 "
        "workers: 48 workers importing the model code at once ran out of file descriptors on the container "
        "filesystem.\n\n"
        "1. `scripts/census/estimate_token_budget.py` → `results/<modality>_summary.json` "
        "(`--stale` lists datasets whose shards changed since they were counted, `--refresh` recounts them)\n"
        "2. `scripts/report/compute_language_split.py --language-counts <curator language_counts.md>` → "
        "`results/language_split.json`\n"
        "3. `scripts/report/render_readme.py` → this file\n\n"
        "Checks: `scripts/validation/check_against_encoder.py` (census vs. the real training encoder, sample by "
        "sample) and `scripts/validation/validate_token_estimates.py` (vision-token math vs. the real processor). "
        "Standalone single-sample estimator for sharing: `scripts/standalone_estimate_tokens.py`.\n\n"
    )

    sections = [render_modality_section(m, [r for r in rows if r["modality"] == m]) for m in MODALITY_ORDER]
    README_PATH.write_text(header + "\n".join(sections))
    print(f"Wrote {README_PATH} ({len(rows)} datasets, {len(categories)} categories)")


if __name__ == "__main__":
    main()
