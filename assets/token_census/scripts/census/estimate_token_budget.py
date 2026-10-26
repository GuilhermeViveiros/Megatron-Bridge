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

"""Exact token census of the EuroVL training mixture: every sample of every dataset in ``mixture.yaml``.

Per sample it counts text, vision and answer (loss-masked) tokens exactly as training produces them:
the real chat template and tokenizer, ``EuroVLProcessor.__call__``'s placeholder expansion with
analytically computed vision-token counts (no pixel decode), and the training encoder's own
``assistant_answer_mask``. ``validation/check_against_encoder.py`` checks this against the real
``EuroVLTaskEncoder`` sample by sample.

- Datasets come from the live ``mixture.yaml``; shards from each dataset's ``.nv-meta/split.yaml``
  (the list energon reads), and every split is counted (val/test is also reported on its own).
- Samples are grouped the way webdataset/energon group a tar: consecutive members sharing a key.
- Per-sample language (``meta.language`` / per-branch languages, else confidence-gated langdetect)
  is recorded for every dataset.
- Work is split per (dataset, shard). Per-shard results go to ``--work-dir`` (large, not in git) so
  a run can resume and changed datasets can be recounted alone; the per-modality summaries go to
  ``assets/token_census/results/<modality>_summary.json``.

Paths come from flags or environment variables (no defaults, nothing cluster-specific in the repo):
``--data-root``/``EUROVL_DATA_ROOT`` (energon data root holding ``mixture.yaml``), ``--tokenizer``/
``EUROVL_HF`` (HF export of the model, must match the recipe), ``--work-dir``/``EUROVL_CENSUS_WORK_DIR``.

Run inside the container:
    python assets/token_census/scripts/census/estimate_token_budget.py --modality image,video
    python .../estimate_token_budget.py --modality text --only euroblocks   # recount some datasets
    python .../estimate_token_budget.py --stale                            # list changed datasets
    python .../estimate_token_budget.py --refresh                          # recount changed datasets
    python .../estimate_token_budget.py --combine-only                     # rebuild the summaries
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import re
import tarfile
from collections import Counter
from pathlib import Path

import av
import yaml
from PIL import Image


MODALITIES = ("image", "multiimage", "video", "text")
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

_processor = None  # per-worker global, set once by _worker_init


def _worker_init(tokenizer_path: str) -> None:
    global _processor
    from megatron.bridge.models.euro_vl.euro_vl_processor import EuroVLProcessor

    _processor = EuroVLProcessor.from_pretrained(tokenizer_path, seq_length=8192)


def split_of_shards(path: Path) -> dict[str, str]:
    """{shard file name: split} from ``.nv-meta/split.yaml``, the shard list energon actually reads.

    Falls back to every ``shard*.tar`` as "train" when a dataset has no split.yaml. Excluded shards and
    listed-but-missing files are left out.
    """
    split_file = path / ".nv-meta" / "split.yaml"
    if not split_file.exists():
        return {p.name: "train" for p in path.glob("shard[-_]*.tar")}
    cfg = yaml.safe_load(split_file.read_text()) or {}
    excluded = set(cfg.get("exclude") or [])
    out = {}
    for split, parts in (cfg.get("split_parts") or {}).items():
        for part in parts or []:
            if part not in excluded and (path / part).exists():
                out[part] = split
    return out


def build_datasets(data_root: Path, mixture: Path, modality: str) -> tuple[list[dict], list[str]]:
    """Every ``mixture.yaml`` entry of one modality, resolved on disk the way training resolves it.

    Category folders may be spelled with '-' on disk where the mixture uses '_'; a key containing '/'
    is a full relative path; otherwise a unique leaf name under any category folder is accepted
    (as in ``EuroVLEnergonProvider._resolve_repeat_factors``). Returns (datasets, skipped entries).
    """
    cfg = yaml.safe_load(mixture.read_text()) or {}
    datasets, skipped = [], []
    for category, entries in (cfg.get(modality) or {}).items():
        for name, r in (entries or {}).items():
            if "/" in name:
                candidates = [data_root / name]
            else:
                candidates = [
                    data_root / modality / category / name,
                    data_root / modality / category.replace("_", "-") / name,
                ]
            path = next((p for p in candidates if (p / ".nv-meta" / "index.sqlite").exists()), None)
            if path is None:
                hits = [p.parent.parent for p in (data_root / modality).glob(f"*/{name}/.nv-meta/index.sqlite")]
                path = hits[0] if len(hits) == 1 else None
            if path is None:
                skipped.append(f"{modality}/{category}/{name}: not found on disk")
                continue
            if "/" in name:  # report full-path keys under their real leaf name + on-disk category
                category, name = path.parent.name, path.name
            shard_split = split_of_shards(path)
            if not shard_split:
                skipped.append(f"{modality}/{category}/{name}: indexed but 0 shards")
                continue
            datasets.append(
                {
                    "modality": modality,
                    "category": category,
                    "name": name,
                    "path": str(path),
                    "mixture_r": r,
                    "shard_split": shard_split,
                    "track_language": True,
                }
            )
    return datasets, skipped


def stale_datasets(work_dir: Path, datasets: list[dict]) -> list[tuple[str, str]]:
    """(name, reason) for datasets whose shard set changed, or with a shard newer than its result."""
    out = []
    for ds in datasets:
        d = work_dir / _dataset_key(ds)
        done = {f.name[: -len(".json")]: f.stat().st_mtime for f in d.glob("*.json")} if d.exists() else {}
        expected = set(ds["shard_split"])
        if set(done) != expected:
            out.append((ds["name"], f"{len(expected)} shards listed vs {len(done)} counted"))
            continue
        newer = [s for s in expected if (Path(ds["path"]) / s).stat().st_mtime > done[s]]
        if newer:
            out.append((ds["name"], f"{len(newer)}/{len(expected)} shards changed since counted"))
    return out


def _dataset_key(ds: dict) -> str:
    return f"{ds['modality']}__{ds['category'].replace('/', '_')}__{ds['name']}"


def _consecutive_samples(tf: tarfile.TarFile):
    """Yield (key, {ext: member}) the way webdataset/energon group a tar: consecutive members sharing
    a key (basename up to the first '.') form one sample.

    Grouping by key over the whole tar is NOT equivalent: some datasets reuse one key for several
    samples in the same shard (e.g. molmo2_chart_translated writes every translation of a chart under
    the source key), and a dict-by-key would merge them into one sample with mixed media.
    """
    key, parts = None, {}
    for m in tf.getmembers():
        if not m.isfile():
            continue
        k, _, ext = os.path.basename(m.name).partition(".")
        if k != key or ext in parts:  # a repeated ext under the same key starts a new sample
            if parts:
                yield key, parts
            key, parts = k, {}
        parts[ext] = m
    if parts:
        yield key, parts


def _iter_image_samples(tf: tarfile.TarFile):
    for _, parts in _consecutive_samples(tf):
        media = [m for ext, m in parts.items() if ext != "json"]
        if "json" in parts and media:
            yield [media[0]], parts["json"]


def _iter_multiimage_samples(tf: tarfile.TarFile):
    for _, parts in _consecutive_samples(tf):
        imgs = {}
        for ext, m in parts.items():
            idx = ext.split(".")[0][len("img") :] if ext.startswith("img") else ""
            if idx.isdigit():
                imgs[int(idx)] = m
        if "json" in parts and imgs:
            yield [imgs[i] for i in sorted(imgs)], parts["json"]


def _sanitize_text_only_placeholders(conv: list) -> list:
    """Neutralize stray ``<image>``/``<video>`` placeholders in text-only (zero-media) samples.

    Some text corpora (found 2026-09-12: ``code/euroblocks``, a Codeforces-style competitive-
    programming corpus) retain a literal ``<image>`` placeholder from their source (the original
    problem page had an embedded diagram) even though this text-only dataset carries no actual
    image file. Since ``EuroVLProcessor``'s ``image_token``/``video_token`` are real special
    tokens shared with the multimodal path, left as-is these tokenize to the same id as a genuine
    vision slot -- confirmed via the real tokenizer (``<image>`` embedded in prose -> the same
    id as ``image_token_id``). For a text-modality sample specifically, ANY such tag is always a
    stray placeholder (there is never a real image/video to pair it with), so it's always
    correct to neutralize it here -- not a heuristic guess.
    """
    for turn in conv:
        content = turn.get("content")
        if isinstance(content, str):
            turn["content"] = content.replace("<image>", "[image omitted]").replace("<video>", "[video omitted]")
    return conv


# ---- Per-sample language detection -------------------------------------------------------------
# A faithful copy of smurf4eu-vision-curator's ``to_energon.py::_get_language`` (the function that
# produced language_counts.md), so languages detected here are comparable with that file:
# detect on the ASSISTANT text (the user prompt is often a fixed English template regardless of
# the source language), strip grounding refs / markdown-LaTeX noise, resolve zh/ko/ja by Unicode
# script, else langdetect (seed 0). Only used when the sample itself doesn't declare a language.
_OBJECT_REF_RE = re.compile(r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>", re.DOTALL)
_MD_NOISE_RE = re.compile(r"\\begin\{.*?\}|\\end\{.*?\}|<sup>.*?</sup>|<sub>.*?</sub>|\[\d+\]|#{1,6}\s*|\*{1,3}|\\")
_SCRIPT_RANGES = (
    ("zh", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))),
    ("ko", ((0xAC00, 0xD7A3), (0x1100, 0x11FF))),
    ("ja", ((0x3040, 0x30FF),)),
)


def _script_language(text: str) -> str | None:
    counts: Counter = Counter()
    total = 0
    for ch in text:
        cp = ord(ch)
        for lang, ranges in _SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[lang] += 1
                total += 1
                break
    if total < 8:
        return None
    lang, n = counts.most_common(1)[0]
    return lang if n / total > 0.5 else None


def _detect_text_language(raw: str) -> str:
    from langdetect import DetectorFactory, LangDetectException
    from langdetect import detect as _langdetect

    DetectorFactory.seed = 0
    refs = _OBJECT_REF_RE.findall(raw)
    text = " ".join(refs) if refs else raw
    text = _MD_NOISE_RE.sub(" ", text).strip()
    if len(text) < 8:
        return "und"
    script_lang = _script_language(text)
    if script_lang:
        return script_lang
    try:
        return _langdetect(text[:500])
    except LangDetectException:
        return "und"


_SPECIAL_MARKUP_RE = re.compile(r"<\|[a-z_]+\|>")
_MIN_LETTERS = 20
_MIN_DISTINCT_WORDS = 8
_MIN_PROB = 0.90


def _confident_language(raw: str) -> tuple[str, float]:
    """(lang, probability) with the curation stripping rules; ("und", 0) if too little real text."""
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
    stripped = raw.lstrip()
    if stripped.startswith(("[", "{")):  # structured answers (JSON points/boxes) carry no language
        return "und", 0.0
    refs = _OBJECT_REF_RE.findall(raw)
    text = " ".join(refs) if refs else raw
    text = _MD_NOISE_RE.sub(" ", _SPECIAL_MARKUP_RE.sub(" ", text)).strip()
    script_lang = _script_language(text)
    if script_lang:
        return script_lang, 1.0
    if sum(ch.isalpha() for ch in text) < _MIN_LETTERS:
        return "und", 0.0
    # langdetect is confidently wrong on low-variety text ("Answer: D Answer: B ..." -> de,
    # "Helicopter Helicopter ..." -> nl, "molecule 2, molecule 4" -> ro; verified 2026-09-28),
    # so require real lexical variety before trusting it at all.
    if len({w.lower() for w in re.findall(r"[^\W\d_]{2,}", text)}) < _MIN_DISTINCT_WORDS:
        return "und", 0.0
    try:
        best = detect_langs(text[:500])[0]
        return best.lang, best.prob
    except LangDetectException:
        return "und", 0.0


def _turns_language(turns: list) -> str:
    """Language of a list of turns.

    Starts from the curation rule (detect on the assistant text), but the census also covers
    grounding/temporal/MCQ data whose answers are coordinates, JSON, timestamps, option letters or
    2-word labels -- langdetect confidently mislabels those (verified 2026-09-28: tapos 72% "da",
    molmopoint_trackany 57% "fr", seeker MCQ letters "hu", kinetics labels "tl", all English).
    So: accept a language only with >=20 letters of real text and langdetect prob >= 0.90, trying
    assistant text, then user text (in the target language for translated datasets), then both;
    else "und".
    """

    def _text(role: str | None) -> str:
        return (
            " ".join(
                t["content"]
                for t in turns
                if (role is None or t.get("role") == role) and isinstance(t.get("content"), str)
            )
            .replace("<image>", " ")
            .replace("<video>", " ")
        )

    for text in (_text("assistant"), _text("user"), _text(None)):
        lang, prob = _confident_language(text)
        if lang != "und" and prob >= _MIN_PROB:
            return lang
    return "und"


def _flatten_message_tree(tree: dict) -> tuple[list, list[list]]:
    """Turns exactly as training renders a message tree (euro_vl_task_encoder._encode_message_tree):
    ``shared`` turns followed by every branch's turns, templated once as one sequence."""
    shared, branches = tree["shared"], tree["branches"]
    return [dict(t) for t in shared] + [dict(t) for b in branches for t in b], branches


def _iter_text_samples(tf: tarfile.TarFile):
    """Text-only samples: one bare ``{key}.json`` conversation per sample, no media file.

    Yields ``([], json_member)`` -- the empty media list makes ``process_shard``'s shared
    image/multiimage branch (``for m in media: ...``) a no-op, so no separate code path is
    needed there; ``_exact_sample_tokens(conv, [], [])`` naturally handles zero images/videos.
    """
    for m in tf.getmembers():
        if m.name.endswith(".json"):
            yield [], m


def _iter_video_samples(tf: tarfile.TarFile):
    for _, parts in _consecutive_samples(tf):
        video = [m for ext, m in parts.items() if "." + ext.rsplit(".", 1)[-1] in VIDEO_EXTS]
        if "json" in parts and video:
            yield video[0], parts["json"]


def _split_conversation(conv: list) -> list:
    """Mirror hf_encoder_task_encoder.encode_sample's <image>/<video> -> structured-content split."""
    out = []
    for turn in conv:
        content = turn.get("content")
        if not isinstance(content, str) or not (("<image>" in content) or ("<video>" in content)):
            out.append(turn)
            continue
        parts = re.split(r"(<image>|<video>)", content)
        content_parts = []
        for part in parts:
            if part == "<image>":
                content_parts.append({"type": "image"})
            elif part == "<video>":
                content_parts.append({"type": "video"})
            elif part.strip():
                content_parts.append({"type": "text", "text": part.strip()})
        out.append({**turn, "content": content_parts})
    return out


def _video_duration_and_size(raw: bytes) -> tuple | None:
    """(duration, native_w, native_h, total_frames_estimate) via container header only."""
    with av.open(io.BytesIO(raw)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration) / 1_000_000.0
        else:
            return None
        w, h = stream.codec_context.width, stream.codec_context.height
        total = max(1, int(duration * float(stream.average_rate))) if stream.average_rate else 10**9
    return duration, w, h, total


_SKIPPED_TOKEN_IDS = None


def _skipped_token_ids():
    """Pad/media ids the training encoder force-masks (computed once per worker)."""
    global _SKIPPED_TOKEN_IDS
    if _SKIPPED_TOKEN_IDS is None:
        from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids

        _SKIPPED_TOKEN_IDS = extract_skipped_token_ids(_processor)
    return _SKIPPED_TOKEN_IDS


def _exact_sample_tokens(conv: list, image_sizes: list, video_infos: list) -> tuple[int, int, int]:
    """(text_tokens, vision_tokens, answer_tokens) matching EuroVLProcessor.__call__'s exact output.

    ``answer_tokens`` counts the supervised positions: the training encoder's own
    ``assistant_answer_mask`` (answer tokens plus each answer's closing ``<|im_end|>``) on the full,
    untruncated sequence.

    Builds the literal expanded text the same way __call__ does (image: bare ``<image>`` -> N
    copies; video: ``<|vision_start|><video><|vision_end|>`` -> per-frame timestamped blocks,
    using the SAME timestamps ``decode_video_bytes``/the task encoder actually produces — matching
    what real training passes via ``video_metadata``, not the fps-fallback default), then tokenizes
    the single final string. Avoids isolated-substring token-count arithmetic, which doesn't account
    for BPE merge effects across concatenation boundaries.
    """
    from megatron.bridge.models.euro_vl.moonvit.video_processing_moonvit import _smart_resize
    from megatron.bridge.models.euro_vl.utils import compute_moonvit_visual_tokens, format_timestamp

    structured = _split_conversation(conv)
    text = _processor.apply_chat_template(structured, tokenize=False)

    vision_tokens = 0
    image_token, video_token = _processor.image_token, _processor.video_token
    vstart, vend = _processor.vision_start_token, _processor.vision_end_token

    # Safety check: this dataset's conversation format must match what we assume (one bare
    # <image>/<video> tag per media file, in order) — datasets with a different convention
    # (structured content, mismatched image/tag counts) would otherwise silently produce a
    # wrong count instead of an error. Fail loudly (caught by the caller as n_failed) instead.
    n_image_tags, n_video_tags = text.count(image_token), text.count(video_token)
    if n_image_tags != len(image_sizes) or n_video_tags != len(video_infos):
        raise ValueError(
            f"tag/media count mismatch: {n_image_tags} <image> tags vs {len(image_sizes)} images, "
            f"{n_video_tags} <video> tags vs {len(video_infos)} videos"
        )

    for w, h in image_sizes:
        n = compute_moonvit_visual_tokens((w, h), _processor.image_processor)
        vision_tokens += n
        text = text.replace(image_token, "<|placeholder|>" * n, 1)
    text = text.replace("<|placeholder|>", image_token)

    vp = _processor.video_processor
    whole_block = f"{vstart}{video_token}{vend}"
    for duration, native_w, native_h, total in video_infos:
        n_frames = vp.plan_num_frames(duration, total)
        cap = vp.per_frame_cap_pixels(n_frames)
        factor = vp._merge_factor
        new_h, new_w = _smart_resize(native_h, native_w, factor=factor, min_pixels=vp.min_pixels, max_pixels=cap)
        frame_tokens = (new_h // factor) * (new_w // factor)
        vision_tokens += n_frames * frame_tokens
        # Same timestamps decode_video_bytes actually produces (evenly spread over duration) —
        # matches what the task encoder passes as video_metadata in real training.
        times = [duration / 2.0] if n_frames == 1 else [i * duration / (n_frames - 1) for i in range(n_frames)]
        per_frame = "".join(
            f"<{format_timestamp(t, _processor.timestamp_format)}>" + vstart + "<|placeholder|>" * frame_tokens + vend
            for t in times
        )
        if whole_block in text:
            text = text.replace(whole_block, per_frame, 1)
        else:
            text = text.replace(video_token, per_frame, 1)
    text = text.replace("<|placeholder|>", video_token)

    from megatron.bridge.data.energon.euro_vl_task_encoder import assistant_answer_mask

    ids = _processor.tokenizer.encode(text, add_special_tokens=True)
    answer_tokens = int(
        assistant_answer_mask(ids, _processor.tokenizer, _skipped_token_ids(), require_answer=False).sum()
    )
    return len(ids) - vision_tokens, vision_tokens, answer_tokens


def count_sample(modality: str, conv, media_raws: list[bytes]) -> dict:
    """Exact (text, vision) tokens for ONE sample, given its parsed JSON and raw media bytes.

    The single per-sample code path used by ``process_shard`` -- factored out so tests can check
    it sample-by-sample against the real training task encoder. Raises on anything malformed
    (the caller counts it as a failed sample).
    """
    meta, branches = {}, None
    if isinstance(conv, dict) and conv.get("message_tree") is True:
        meta = conv.get("meta") or {}
        conv, branches = _flatten_message_tree(conv)
    if not isinstance(conv, list):
        raise ValueError(f"conversation is {type(conv).__name__}, not a list or message tree")
    # Some datasets (found 2026-08-31: molmo2_syn_music, pixmo_cap_qa, doclingmatix, and ~12
    # others) wrap the conversation in an extra list layer -- [[{...turns...}]]. Unwrap it.
    if len(conv) == 1 and isinstance(conv[0], list):
        conv = conv[0]
    if modality == "text":
        conv = _sanitize_text_only_placeholders(conv)
    image_sizes: list[tuple[int, int]] = []
    if modality == "video":
        info = _video_duration_and_size(media_raws[0])
        if info is None:
            raise ValueError("video container has no duration")
        tt, vt, at = _exact_sample_tokens(conv, [], [info])
    else:
        image_sizes = [Image.open(io.BytesIO(raw)).size for raw in media_raws]
        tt, vt, at = _exact_sample_tokens(conv, image_sizes, [])
    return {
        "text": tt,
        "vision": vt,
        "answer": at,
        "conv": conv,
        "meta": meta,
        "branches": branches,
        "image_sizes": image_sizes,
    }


def process_shard(dataset_key: str, ds: dict, shard_path: str) -> dict:
    """Exact text+vision token sums over every sample in one shard.

    Also accumulates raw image width/height/pixel-count sums (image + multiimage modalities only
    -- video frames aren't a single "image size" in the same sense, so left out of this metric) so
    ``combine()`` can report avg image size and aspect ratio per dataset, exhaustively over every
    image, not a spot-check sample.
    """
    n_ok = n_failed = 0
    sum_text = sum_vision = sum_answer = 0
    sum_w = sum_h = n_images = 0
    sum_ratio = 0.0  # sum of per-image w/h, NOT (sum_w/sum_h) -- ratio-of-averages skews on
    # datasets with mixed portrait/landscape images; average-of-ratios is the correct per-image mean.
    n_trees = 0
    # Per-language token attribution, only when the driver asks for it (ds["track_language"]):
    # langdetect on every sample of the whole mixture would cost tens of CPU-hours.
    track_language = bool(ds.get("track_language"))
    lang_tokens: dict[str, float] = {}
    lang_units: dict[str, int] = {}  # samples (flat) or branches (message tree) per language
    lang_from_meta = lang_detected = 0
    # Cross-check vs. the curation side's own per-sample count, when a message tree carries one.
    sum_meta_rendered = sum_census_for_meta = n_meta_rendered = 0
    fail_reasons: dict[str, int] = {}  # so silent per-sample failures are visible, not just counted
    try:
        with tarfile.open(shard_path) as tf:
            if ds["modality"] == "image":
                sample_iter = _iter_image_samples(tf)
            elif ds["modality"] == "multiimage":
                sample_iter = _iter_multiimage_samples(tf)
            elif ds["modality"] == "text":
                sample_iter = _iter_text_samples(tf)
            else:
                sample_iter = _iter_video_samples(tf)

            for media, json_member in sample_iter:
                try:
                    if ds["modality"] == "video":
                        media_raws = [tf.extractfile(media).read()]
                    else:
                        media_raws = [tf.extractfile(m).read() for m in media]
                    s = count_sample(ds["modality"], json.loads(tf.extractfile(json_member).read()), media_raws)
                    tt, vt, conv, meta, branches = s["text"], s["vision"], s["conv"], s["meta"], s["branches"]
                    n_trees += branches is not None
                    for w, h in s["image_sizes"]:
                        sum_w += w
                        sum_h += h
                        sum_ratio += w / h
                        n_images += 1
                    sum_text += tt
                    sum_vision += vt
                    sum_answer += s["answer"]
                    n_ok += 1
                    if meta.get("rendered_tokens"):
                        sum_meta_rendered += int(meta["rendered_tokens"])
                        sum_census_for_meta += tt + vt
                        n_meta_rendered += 1
                    if track_language:
                        declared = meta.get("language") or meta.get("lang")
                        # Units that carry a language: each branch of a message tree (branches of
                        # one packed sample can be in different languages), else the whole sample.
                        units = branches if branches is not None else [conv]
                        weights = [
                            sum(len(t.get("content", "")) for t in u if isinstance(t.get("content"), str)) or 1
                            for u in units
                        ]
                        wsum = sum(weights)
                        for unit, w in zip(units, weights):
                            if declared:
                                lang = str(declared)
                                lang_from_meta += 1
                            else:
                                lang = _turns_language(unit)
                                lang_detected += 1
                            # Shared media/prompt tokens are split across branches by text share.
                            lang_tokens[lang] = lang_tokens.get(lang, 0.0) + (tt + vt) * w / wsum
                            lang_units[lang] = lang_units.get(lang, 0) + 1
                except Exception as e:  # noqa: BLE001 — one bad sample shouldn't kill the shard
                    n_failed += 1
                    reason = f"{type(e).__name__}: {str(e)[:120]}"
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
    except Exception as e:  # noqa: BLE001 — one bad shard shouldn't kill the dataset
        return {
            "dataset_key": dataset_key,
            "shard": os.path.basename(shard_path),
            "n_ok": 0,
            "n_failed": 0,
            "sum_text": 0,
            "sum_vision": 0,
            "sum_w": 0,
            "sum_h": 0,
            "sum_ratio": 0.0,
            "n_images": 0,
            "shard_error": f"{type(e).__name__}: {e}",
        }
    return {
        "dataset_key": dataset_key,
        "shard": os.path.basename(shard_path),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "sum_text": sum_text,
        "sum_vision": sum_vision,
        "sum_answer": sum_answer,
        "sum_w": sum_w,
        "sum_h": sum_h,
        "sum_ratio": sum_ratio,
        "n_images": n_images,
        "n_trees": n_trees,
        "lang_tokens": lang_tokens,
        "lang_units": lang_units,
        "lang_from_meta": lang_from_meta,
        "lang_detected": lang_detected,
        "sum_meta_rendered": sum_meta_rendered,
        "sum_census_for_meta": sum_census_for_meta,
        "n_meta_rendered": n_meta_rendered,
        "fail_reasons": fail_reasons,
        "shard_error": None,
    }


def _worker(args) -> dict:
    dataset_key, ds, shard_path, outdir = args
    result = process_shard(dataset_key, ds, shard_path)
    ds_dir = outdir / dataset_key
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / f"{result['shard']}.json").write_text(json.dumps(result))
    return result


def combine(work_dir: Path, datasets: list[dict], summary_path: Path, skipped: list[str], data_root: Path) -> None:
    """Merge one modality's per-shard results into ``summary_path`` (one row per dataset).

    Dataset paths are stored relative to ``data_root`` so the summary holds no machine-specific paths.
    """
    by_key = {_dataset_key(ds): ds for ds in datasets}
    rows = []
    for ds_dir in sorted(work_dir.iterdir()):
        # Only datasets in this run's list: leftover dirs (e.g. dropped from the mixture) are ignored.
        if not ds_dir.is_dir() or ds_dir.name not in by_key:
            continue
        key = ds_dir.name
        ds = by_key[key]
        # Energon split of each shard (from .nv-meta/split.yaml). Totals cover every split; the
        # val/test share is also reported on its own (non_train_*).
        shard_split = ds.get("shard_split", {})
        val_samples = val_tokens = 0
        n_ok = n_failed = sum_text = sum_vision = 0
        sum_answer = 0
        has_answer = True  # False if any shard predates answer counting -> no partial total
        sum_w = sum_h = n_images = 0
        sum_ratio = 0.0
        shard_errors = []
        n_trees = lang_from_meta = lang_detected = 0
        sum_meta_rendered = sum_census_for_meta = n_meta_rendered = 0
        lang_tokens: dict[str, float] = {}
        lang_units: dict[str, int] = {}
        fail_reasons: dict[str, int] = {}
        for f in ds_dir.glob("*.json"):
            r = json.loads(f.read_text())
            if shard_split.get(f.name[: -len(".json")], "train") != "train":
                val_samples += r["n_ok"]
                val_tokens += r["sum_text"] + r["sum_vision"]
            n_trees += r.get("n_trees", 0)
            lang_from_meta += r.get("lang_from_meta", 0)
            lang_detected += r.get("lang_detected", 0)
            sum_meta_rendered += r.get("sum_meta_rendered", 0)
            sum_census_for_meta += r.get("sum_census_for_meta", 0)
            n_meta_rendered += r.get("n_meta_rendered", 0)
            for lang, v in r.get("lang_tokens", {}).items():
                lang_tokens[lang] = lang_tokens.get(lang, 0.0) + v
            for lang, v in r.get("lang_units", {}).items():
                lang_units[lang] = lang_units.get(lang, 0) + v
            for reason, v in r.get("fail_reasons", {}).items():
                fail_reasons[reason] = fail_reasons.get(reason, 0) + v
            n_ok += r["n_ok"]
            n_failed += r["n_failed"]
            sum_text += r["sum_text"]
            sum_vision += r["sum_vision"]
            if "sum_answer" in r:
                sum_answer += r["sum_answer"]
            elif r["n_ok"]:
                has_answer = False
            # .get() with a default: older per-shard files (pre image-size tracking) lack these
            # keys -- treated as contributing 0 rather than crashing combine() on mixed old+new data.
            sum_w += r.get("sum_w", 0)
            sum_h += r.get("sum_h", 0)
            sum_ratio += r.get("sum_ratio", 0.0)
            n_images += r.get("n_images", 0)
            if r.get("shard_error"):
                shard_errors.append((r["shard"], r["shard_error"]))
        row = {
            **ds,
            "path": os.path.relpath(ds["path"], data_root),
            "dataset_key": key,
            "n_ok": n_ok,
            "n_failed": n_failed,
            "total_text_tokens": sum_text,
            "total_vision_tokens": sum_vision,
            "total_tokens": sum_text + sum_vision,
            "shard_errors": shard_errors,
            "n_message_trees": n_trees,
            "fail_reasons": dict(sorted(fail_reasons.items(), key=lambda kv: -kv[1])[:10]),
            "non_train_samples": val_samples,
            "non_train_tokens": val_tokens,
        }
        if has_answer:
            row["total_answer_tokens"] = sum_answer
        if lang_tokens:
            row["lang_tokens"] = dict(sorted(lang_tokens.items(), key=lambda kv: -kv[1]))
            row["lang_units"] = lang_units
            row["lang_from_meta"] = lang_from_meta
            row["lang_detected"] = lang_detected
        if n_meta_rendered:
            row["meta_rendered_tokens"] = sum_meta_rendered
            row["census_tokens_for_meta_samples"] = sum_census_for_meta
            row["n_meta_rendered"] = n_meta_rendered
        if n_ok:
            row["avg_text_tokens"] = sum_text / n_ok
            row["avg_vision_tokens"] = sum_vision / n_ok
            row["avg_total_tokens"] = (sum_text + sum_vision) / n_ok
        if n_images:
            row["avg_image_width"] = sum_w / n_images
            row["avg_image_height"] = sum_h / n_images
            row["avg_image_aspect_ratio"] = sum_ratio / n_images
            row["n_images"] = n_images
        rows.append(row)

    rows.sort(key=lambda r: -r["total_tokens"])
    print(f"\n{'dataset':50s} {'modality':10s} {'n_ok':>10s} {'n_fail':>8s} {'avg_tot':>8s} {'total_tokens':>16s}")
    grand_total = 0
    for r in rows:
        grand_total += r["total_tokens"]
        avg = r.get("avg_total_tokens", 0.0)
        print(
            f"{r['category']}/{r['name']:<38.38s} {r['modality']:10s} {r['n_ok']:>10,d} {r['n_failed']:>8,d} "
            f"{avg:>8.0f} {r['total_tokens']:>16,d}"
        )
    print(f"\n{len(rows)} datasets, grand total EXACT tokens (all splits, all samples): {grand_total:,}")
    print(f"of which val/test splits: {sum(r['non_train_tokens'] for r in rows):,} tokens")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"grand_total_tokens": grand_total, "datasets": rows, "skipped": skipped}, indent=1) + "\n"
    )
    print(f"Wrote {summary_path}")


def _env_path(value: Path | None, env: str, flag: str) -> Path:
    """A path from its flag, else its environment variable; fail with a clear message otherwise."""
    if value is None and os.environ.get(env):
        value = Path(os.environ[env])
    if value is None:
        raise SystemExit(f"{flag} (or ${env}) is required")
    return value


def main() -> None:
    """Count, recount or summarize the census for the requested modalities."""
    parser = argparse.ArgumentParser(description="Exact token census of the EuroVL training mixture")
    parser.add_argument("--modality", default=",".join(MODALITIES), help="comma-separated, default all four")
    parser.add_argument("--data-root", type=Path, default=None, help="energon data root ($EUROVL_DATA_ROOT)")
    parser.add_argument("--mixture", type=Path, default=None, help="default: <data-root>/mixture.yaml")
    parser.add_argument("--tokenizer", type=str, default=None, help="HF export of the model ($EUROVL_HF)")
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="per-shard results, not in git ($EUROVL_CENSUS_WORK_DIR)"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="where <modality>_summary.json go")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--only", default="", help="comma-separated dataset names to recount; others keep results")
    parser.add_argument("--stale", action="store_true", help="only list datasets whose shards changed since counted")
    parser.add_argument("--refresh", action="store_true", help="recount the datasets --stale would list")
    parser.add_argument("--combine-only", action="store_true", help="no counting, rebuild the summaries")
    parser.add_argument("--smoke", action="store_true", help="first shard of each dataset, into <work-dir>/_smoke")
    args = parser.parse_args()

    data_root = _env_path(args.data_root, "EUROVL_DATA_ROOT", "--data-root")
    work_root = _env_path(args.work_dir, "EUROVL_CENSUS_WORK_DIR", "--work-dir")
    mixture = args.mixture or data_root / "mixture.yaml"
    counting = not (args.stale or args.combine_only)
    if counting and not (args.tokenizer or os.environ.get("EUROVL_HF")):
        raise SystemExit("--tokenizer (or $EUROVL_HF) is required to count")
    tokenizer = args.tokenizer or os.environ.get("EUROVL_HF")

    for modality in args.modality.split(","):
        assert modality in MODALITIES, modality
        datasets, skipped = build_datasets(data_root, mixture, modality)
        work_dir = work_root / ("_smoke" if args.smoke else "") / modality
        work_dir.mkdir(parents=True, exist_ok=True)
        only = {n for n in args.only.split(",") if n}
        if args.stale or args.refresh:
            stale = stale_datasets(work_dir, datasets)
            print(f"{modality}: {len(stale)} stale dataset(s)", flush=True)
            for name, why in stale:
                print(f"  {name}: {why}", flush=True)
            if args.stale:
                continue
            only = {name for name, _ in stale}
            if not only:
                continue
        missing = only - {ds["name"] for ds in datasets}
        assert not missing, f"--only names not in the mixture: {sorted(missing)}"

        tasks = []
        if counting:
            for ds in datasets:
                if only and ds["name"] not in only:
                    continue
                if only:  # drop old per-shard results so the summary only sees the recount
                    for old in (work_dir / _dataset_key(ds)).glob("*.json"):
                        old.unlink()
                shards = sorted(Path(ds["path"]) / name for name in ds["shard_split"])
                for shard in shards[:1] if args.smoke else shards:
                    tasks.append((_dataset_key(ds), ds, str(shard), work_dir))
        print(
            f"{modality}: {len(datasets)} datasets, {len(tasks)} shard tasks, {len(skipped)} skipped, "
            f"workers={args.workers}",
            flush=True,
        )
        for s in skipped:
            print(f"  SKIPPED {s}", flush=True)
        if tasks:
            done = 0
            with mp.Pool(processes=args.workers, initializer=_worker_init, initargs=(tokenizer,)) as pool:
                for _ in pool.imap_unordered(_worker, tasks, chunksize=1):
                    done += 1
                    if done % 100 == 0 or done == len(tasks):
                        print(f"[{done}/{len(tasks)}] shards done", flush=True)
        if args.smoke:
            combine(work_dir, datasets, work_dir / "summary.json", skipped, data_root)
        else:
            combine(work_dir, datasets, args.results_dir / f"{modality}_summary.json", skipped, data_root)


if __name__ == "__main__":
    main()
