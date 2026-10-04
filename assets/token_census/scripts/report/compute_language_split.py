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

"""Per-language token counts for the EuroVL mixture (and the English vs. multilingual split).

Each dataset's tokens are attributed to languages by the first source that covers it:
  1. ``language_counts.md`` (smurf4eu-vision-curator, ``to_energon.py --detect-lang``): per-language
     SAMPLE counts, from source metadata where present. Tokens are PRORATED:
     ``tokens(lang) ~= total_tokens(dataset) * samples(lang) / samples(dataset)`` -- exact for the
     sample math, approximate for tokens (assumes uniform tokens/sample across languages).
  2. The census's own per-sample attribution (``lang_tokens``, estimate_token_budget.py): EXACT
     tokens per sample (per branch for message trees), language from the sample's ``meta.language``
     when declared, else confidence-gated langdetect. Used for all of video and for datasets new
     since the curation file was generated.
  3. Otherwise assumed 100% English (not flagged as multilingual by the curation pipeline).

Run after the census:
    python assets/token_census/scripts/report/compute_language_split.py \
        --language-counts <curator repo>/data_sync/language_counts.md [--mixture PATH]
Then: python assets/token_census/scripts/report/render_readme.py
"""

import argparse
import json
import os
import re
from pathlib import Path

from render_readme import RESULTS_DIR, default_mixture, load_rows  # same dataset set as the README


OUTPUT_PATH = RESULTS_DIR / "language_split.json"

SECTION_MODALITY = {
    "# Image datasets": "image",
    "# MultiImage datasets": "multiimage",
    "# Video datasets": "video",
    "# Text datasets": "text",
}

# language_counts.md names that don't map 1:1 to one (modality, category, name) census row.
SPECIAL_MAP = {
    ("multi_synth_cc12m_cc3m_grit_caption", "multiimage"): [
        ("multiimage", "captioning", "multi_synth_cc12m_cc3m_grit")
    ],
    ("multi_synth_cc12m_cc3m_grit_general_qa", "multiimage"): [
        ("multiimage", "general_qa", "multi_synth_cc12m_cc3m_grit")
    ],
    ("euroblocks", "text"): [
        ("text", "chat", "euroblocks"),
        ("text", "code", "euroblocks"),
        ("text", "math", "euroblocks"),
        ("text", "stem", "euroblocks"),
    ],
}

# euroblocks' source metadata uses English names / locale pairs instead of ISO codes.
LABEL_NORMALIZE = {
    "Greek": "el",
    "Afrikaans": "af",
    "Estonian": "et",
    "Bulgarian": "bg",
    "Slovenian": "sl",
    "Latvian": "lv",
    "Indonesian": "id",
    "Chinese (Traditional)": "zh",
    "Danish": "da",
    "Croatian": "hr",
    "Tagalog": "tl",
    "Welsh": "cy",
    "Somali": "so",
    "Persian": "fa",
    "Thai": "th",
    "Swahili": "sw",
    "Arabic": "ar",
    "Lithuanian": "lt",
    "Hebrew": "he",
    "Macedonian": "mk",
    "Albanian": "sq",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Telugu": "te",
    "Nepali": "ne",
    "Punjabi": "pa",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "Unknown": "und",
    "unknown": "und",
    # Bilingual translation pairs / explicit mixes carry no single language.
    "mixed": "und",
    "en_es-latam": "und",
    "en-xx": "und",
    "pt-pt_en-GB": "und",
    "pt-pt_en-US": "und",
}

EU_OFFICIAL = {
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "ga",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
}
# EuroLLM's 35 languages: the 24 EU official + these 11.
EUROLLM = EU_OFFICIAL | {"ar", "ca", "zh", "gl", "hi", "ja", "ko", "no", "ru", "tr", "uk"}

# Samples come out "und" when there is too little natural language for langdetect: labels + coordinates
# (grounding / pointing / counting), GUI actions, table cells, LaTeX, code. Their questions are English, so
# (user decision 10-03) undetermined tokens of every dataset that is NOT multilingual count as English.
MULTILINGUAL_NAME = re.compile(r"eurovideo|^ava_|^sav_|multilingual|translated|multi_synth|pangea")

DS_PATTERN = re.compile(r"^## (.+?)\s+\(total: (\d+)\)\s*$", re.M)
ROW_PATTERN = re.compile(r"^\|\s*([^\|]+?)\s*\|\s*(\d+)\s*\|", re.M)
SECTION_PATTERN = re.compile(r"^# (.+)$", re.M)


def norm(lang: str) -> str:
    """ISO code for a curation-file language label."""
    return LABEL_NORMALIZE.get(lang, lang)


def parse_language_counts_md(path: Path) -> dict[tuple[str, str, str], dict[str, int]]:
    """{census key: {lang: samples}} for every dataset section in the curation file."""
    text = path.read_text()
    marks = list(SECTION_PATTERN.finditer(text))
    out: dict[tuple[str, str, str], dict[str, int]] = {}
    unmatched = []
    for i, m in enumerate(marks):
        modality = SECTION_MODALITY.get(f"# {m.group(1).strip()}")
        if modality is None:
            continue
        body = text[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        ds_marks = list(DS_PATTERN.finditer(body))
        for j, dm in enumerate(ds_marks):
            block = body[dm.end() : ds_marks[j + 1].start() if j + 1 < len(ds_marks) else len(body)]
            langs: dict[str, int] = {}
            for rm in ROW_PATTERN.finditer(block):
                if rm.group(1).strip() != "language":
                    lang = norm(rm.group(1).strip())
                    langs[lang] = langs.get(lang, 0) + int(rm.group(2))
            name = dm.group(1).strip()
            keys = SPECIAL_MAP.get((name, modality))
            if keys is None:
                keys = [("?", "?", name)]  # resolved against census rows by (modality, name) below
                unmatched.append((modality, name))
            for k in keys:
                out[(modality, k[1], k[2]) if k[0] != "?" else (modality, "?", name)] = langs
    return out


def main() -> None:
    """Write results/language_split.json."""
    parser = argparse.ArgumentParser(description="Per-language token split of the census")
    parser.add_argument(
        "--language-counts",
        type=Path,
        default=Path(os.environ["EUROVL_LANGUAGE_COUNTS"]) if os.environ.get("EUROVL_LANGUAGE_COUNTS") else None,
        help="curator's data_sync/language_counts.md ($EUROVL_LANGUAGE_COUNTS)",
    )
    parser.add_argument("--mixture", type=Path, default=default_mixture())
    args = parser.parse_args()
    if args.language_counts is None:
        raise SystemExit("--language-counts (or $EUROVL_LANGUAGE_COUNTS) is required")
    rows = load_rows(args.mixture)
    doc = parse_language_counts_md(args.language_counts)
    by_mod_name: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for r in rows:
        by_mod_name.setdefault((r["modality"], r["name"]), []).append((r["modality"], r["category"], r["name"]))

    # Resolve doc entries keyed by (modality, "?", name) to real census keys.
    doc_resolved: dict[tuple[str, str, str], dict[str, int]] = {}
    doc_unmatched = []
    for (mod, cat, name), langs in doc.items():
        if cat != "?":
            doc_resolved[(mod, cat, name)] = langs
            continue
        keys = by_mod_name.get((mod, name), [])
        if not keys:
            doc_unmatched.append(f"{mod}/{name}")
        for k in keys:
            doc_resolved[k] = langs

    lang_totals: dict[str, float] = {}
    lang_eff: dict[str, float] = {}
    lang_sources: dict[str, set] = {}
    per_dataset = []
    grand = 0
    for r in rows:
        key = (r["modality"], r["category"], r["name"])
        tokens = r["total_tokens"]
        grand += tokens
        # A curation section shared by several census rows (e.g. one "euroblocks" section for the
        # chat/code/math/stem text categories) averages their languages; when the census has exact
        # per-sample languages for the row, prefer those so each category keeps its own split.
        shared_doc = key in doc_resolved and sum(v is doc_resolved[key] for v in doc_resolved.values()) > 1
        if shared_doc and r.get("lang_tokens"):
            s = sum(r["lang_tokens"].values()) or 1
            split = {}
            for lang, v in r["lang_tokens"].items():
                split[norm(lang)] = split.get(norm(lang), 0.0) + tokens * v / s
            source = "exact"
        elif key in doc_resolved:
            samples = doc_resolved[key]
            n = sum(samples.values()) or 1
            split = {lang: tokens * c / n for lang, c in samples.items()}
            source = "prorated"
        elif r.get("lang_tokens"):
            # Census attribution is exact per sample; rescale by the tiny float drift so it sums to total.
            s = sum(r["lang_tokens"].values()) or 1
            split = {}
            for lang, v in r["lang_tokens"].items():
                split[norm(lang)] = split.get(norm(lang), 0.0) + tokens * v / s
            source = "exact"
        else:
            split = {"en": float(tokens)}
            source = "assumed-en"
        und_as_en = 0.0
        if not MULTILINGUAL_NAME.search(r["name"]) and split.get("und"):
            und_as_en = split.pop("und")
            split["en"] = split.get("en", 0.0) + und_as_en
        r_w = float(r.get("mixture_r", 1.0))  # mixture repeat factor: effective = r x tokens
        for lang, v in split.items():
            lang_totals[lang] = lang_totals.get(lang, 0.0) + v
            lang_eff[lang] = lang_eff.get(lang, 0.0) + r_w * v
            lang_sources.setdefault(lang, set()).add(source)
        en = split.get("en", 0.0)
        und = split.get("und", 0.0)
        per_dataset.append(
            {
                "modality": r["modality"],
                "category": r["category"],
                "name": r["name"],
                "tokens": tokens,
                "source": source,
                "en_pct": 100 * en / tokens if tokens else 0.0,
                "und_pct": 100 * und / tokens if tokens else 0.0,
                "multilingual_pct": 100 * (tokens - en - und) / tokens if tokens else 0.0,
                "top_languages": dict(sorted(((k, v) for k, v in split.items()), key=lambda kv: -kv[1])[:5]),
                "languages": dict(sorted(split.items(), key=lambda kv: -kv[1])),
                "und_counted_as_en": und_as_en,
            }
        )

    languages = [
        {
            "lang": lang,
            "tokens": v,
            "pct": 100 * v / grand,
            "eu_official": lang in EU_OFFICIAL,
            "eurollm": lang in EUROLLM,
            "sources": sorted(lang_sources[lang]),
        }
        for lang, v in sorted(lang_totals.items(), key=lambda kv: -kv[1])
    ]
    en = lang_totals.get("en", 0.0)
    und = lang_totals.get("und", 0.0)
    g_eff = sum(lang_eff.values())
    result = {
        "effective": {  # weighted by each dataset's mixture repeat factor r
            "grand_total_tokens": g_eff,
            "en_tokens": lang_eff.get("en", 0.0),
            "und_tokens": lang_eff.get("und", 0.0),
            "multilingual_tokens": g_eff - lang_eff.get("en", 0.0) - lang_eff.get("und", 0.0),
            "eu_non_en_tokens": sum(v for k, v in lang_eff.items() if k in EU_OFFICIAL and k != "en"),
            "languages": {k: v for k, v in sorted(lang_eff.items(), key=lambda kv: -kv[1])},
        },
        "grand_total_tokens": grand,
        "en_tokens": en,
        "und_tokens": und,
        "multilingual_tokens": grand - en - und,
        "eu_non_en_tokens": sum(v for k, v in lang_totals.items() if k in EU_OFFICIAL and k != "en"),
        "languages": languages,
        "per_dataset": sorted(per_dataset, key=lambda d: -d["tokens"] * d["multilingual_pct"]),
        "doc_unmatched": doc_unmatched,
        "sources_count": {
            s: sum(1 for d in per_dataset if d["source"] == s) for s in ("prorated", "exact", "assumed-en")
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Wrote {OUTPUT_PATH}")
    print(
        f"EN {100 * en / grand:.2f}%  multilingual {100 * (grand - en - und) / grand:.2f}%  und {100 * und / grand:.2f}%"
    )
    print(f"sources: {result['sources_count']}  doc sections not in mixture: {doc_unmatched}")


if __name__ == "__main__":
    main()
