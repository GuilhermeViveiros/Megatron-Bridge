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

"""Exact per-sample token census (text + vision) over EVERY sample of EVERY prepared dataset.

Sizing the PA data mixture to a token budget (40-67B) needs real total tokens/dataset — this
computes it exactly, not by sampling: every sample in every shard of every dataset under
energon-data/{image,multiimage,video} is counted, giving the true sum (not an extrapolation).

No pixel decode is used (validated header-only-vs-real-processor match in
``estimate_pa_tokens_validate.py``, 0 mismatches). Instead this reproduces
``EuroVLProcessor.__call__``'s exact placeholder-expansion arithmetic (image: bare ``<image>``
becomes N copies; video: the ``<|vision_start|><video><|vision_end|>`` block becomes
per-frame ``<timestamp><|vision_start|><video>*frame_tokens<|vision_end|>`` blocks) on top of the
real chat-templated text, using analytically-computed vision-token counts.

Task granularity is (dataset, shard) — 13k+ shards total — for good load balance across workers
even though dataset sizes range from ~500 to ~13M samples.

Run inside the container (this is a long, exhaustive run — expect hours; safe to background):
    ./apptainer.sh uv run --no-sync python assets/token_census/scripts/estimate_token_budget.py --workers 64
    uv run --no-sync python assets/token_census/scripts/estimate_token_budget.py --combine-only   # re-summarize
"""

import argparse
import io
import json
import os
import re
import sqlite3
import tarfile
from pathlib import Path

import av
from PIL import Image


SCRATCH = os.environ["SCRATCH"]
EUROVL_HF = f"{SCRATCH}/hf_models/euro_vl_2b_hf"
DATA_ROOT = Path("/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data")
OUTDIR_DEFAULT = Path(__file__).parent.parent / "results"
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

_processor = None  # per-worker global, set once by _worker_init


def _worker_init() -> None:
    global _processor
    from megatron.bridge.models.euro_vl.euro_vl_processor import EuroVLProcessor

    _processor = EuroVLProcessor.from_pretrained(EUROVL_HF, seq_length=8192)


def discover_datasets(categories: set[str] | None = None) -> list[dict]:
    """Every dataset with a prepared ``.nv-meta/index.sqlite`` under image/multiimage/video.

    ``categories``, if given, restricts to datasets whose category (e.g. "captioning",
    "knowledge") is in the set — across all three modalities.
    """
    datasets = []
    for modality in ("image", "multiimage", "video"):
        root = DATA_ROOT / modality
        if not root.is_dir():
            continue
        for meta in root.rglob(".nv-meta"):
            ds_dir = meta.parent
            if not (meta / "index.sqlite").exists():
                continue
            rel = ds_dir.relative_to(root)
            category = str(rel.parent) if rel.parent != Path(".") else "uncategorized"
            if categories is not None and category not in categories:
                continue
            datasets.append({"modality": modality, "category": category, "name": ds_dir.name, "path": str(ds_dir)})
    return datasets


def dataset_size(path: str) -> int:
    """Train-split sample count, mirroring EuroVLEnergonProvider._dataset_size."""
    idx = os.path.join(path, ".nv-meta", "index.sqlite")
    con = sqlite3.connect(f"file:{idx}?mode=ro", uri=True)
    try:
        total = int(con.execute("select count(*) from samples").fetchone()[0])
        split_path = os.path.join(path, ".nv-meta", "split.yaml")
        if not os.path.exists(split_path):
            return total
        import yaml

        def _count(shards) -> int:
            ids = [int(m.group(1)) for s in (shards or []) if (m := re.search(r"(\d+)", os.path.basename(s)))]
            if not ids:
                return 0
            q = "select count(*) from samples where tar_file_id in (%s)" % ",".join("?" * len(ids))
            return int(con.execute(q, ids).fetchone()[0])

        parts = (yaml.safe_load(open(split_path)) or {}).get("split_parts", {}) or {}
        train, val, test = _count(parts.get("train")), _count(parts.get("val")), _count(parts.get("test"))
        if train <= 0 or train + val + test != total:
            return total
        return train
    finally:
        con.close()


def _dataset_key(ds: dict) -> str:
    return f"{ds['modality']}__{ds['category'].replace('/', '_')}__{ds['name']}"


def _iter_image_samples(tf: tarfile.TarFile):
    by_stem: dict[str, dict] = {}
    for m in tf.getmembers():
        stem, ext = os.path.splitext(m.name)
        ext = ext.lstrip(".")
        entry = by_stem.setdefault(stem, {})
        if ext == "json":
            entry["json"] = m
        else:
            entry["img"] = m
    for stem in sorted(by_stem):
        e = by_stem[stem]
        if "json" in e and "img" in e:
            yield [e["img"]], e["json"]


def _iter_multiimage_samples(tf: tarfile.TarFile):
    by_stem: dict[str, dict] = {}
    for m in tf.getmembers():
        if m.name.endswith(".json"):
            stem = m.name[: -len(".json")]
            by_stem.setdefault(stem, {})["json"] = m
        elif ".img" in m.name:
            stem = m.name.split(".img")[0]
            idx_str = m.name.split(".img")[1].split(".")[0]
            if not idx_str.isdigit():
                continue
            by_stem.setdefault(stem, {}).setdefault("imgs", {})[int(idx_str)] = m
    for stem in sorted(by_stem):
        e = by_stem[stem]
        if "json" in e and "imgs" in e:
            yield [e["imgs"][i] for i in sorted(e["imgs"])], e["json"]


def _iter_video_samples(tf: tarfile.TarFile):
    by_stem: dict[str, dict] = {}
    for m in tf.getmembers():
        stem, ext = os.path.splitext(m.name)
        entry = by_stem.setdefault(stem, {})
        if ext == ".json":
            entry["json"] = m
        elif ext in VIDEO_EXTS:
            entry["video"] = m
    for stem in sorted(by_stem):
        e = by_stem[stem]
        if "json" in e and "video" in e:
            yield e["video"], e["json"]


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


def _exact_sample_tokens(conv: list, image_sizes: list, video_infos: list) -> tuple[int, int]:
    """(text_tokens, vision_tokens) matching EuroVLProcessor.__call__'s exact output length.

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
            f"<{format_timestamp(t, _processor.timestamp_format)}>"
            + vstart
            + "<|placeholder|>" * frame_tokens
            + vend
            for t in times
        )
        if whole_block in text:
            text = text.replace(whole_block, per_frame, 1)
        else:
            text = text.replace(video_token, per_frame, 1)
    text = text.replace("<|placeholder|>", video_token)

    total_tokens = len(_processor.tokenizer.encode(text, add_special_tokens=True))
    return total_tokens - vision_tokens, vision_tokens


def process_shard(dataset_key: str, ds: dict, shard_path: str) -> dict:
    """Exact text+vision token sums over every sample in one shard.

    Also accumulates raw image width/height/pixel-count sums (image + multiimage modalities only
    -- video frames aren't a single "image size" in the same sense, so left out of this metric) so
    ``combine()`` can report avg image size and aspect ratio per dataset, exhaustively over every
    image, not a spot-check sample.
    """
    n_ok = n_failed = 0
    sum_text = sum_vision = 0
    sum_w = sum_h = n_images = 0
    sum_ratio = 0.0  # sum of per-image w/h, NOT (sum_w/sum_h) -- ratio-of-averages skews on
    # datasets with mixed portrait/landscape images; average-of-ratios is the correct per-image mean.
    try:
        with tarfile.open(shard_path) as tf:
            if ds["modality"] == "image":
                sample_iter = _iter_image_samples(tf)
            elif ds["modality"] == "multiimage":
                sample_iter = _iter_multiimage_samples(tf)
            else:
                sample_iter = _iter_video_samples(tf)

            for media, json_member in sample_iter:
                try:
                    conv = json.loads(tf.extractfile(json_member).read())
                    if not isinstance(conv, list):
                        n_failed += 1
                        continue
                    # Some datasets (found 2026-08-31: molmo2_syn_music, pixmo_cap_qa,
                    # doclingmatix, and ~12 others) wrap the conversation in an extra list layer
                    # -- [[{...turns...}]] instead of every other dataset's flat [{...}, {...}].
                    # _split_conversation expects a flat list of role-dicts; unwrap the doubly-
                    # nested form here (single-element list whose one element is itself a list)
                    # rather than letting it fail silently as an n_failed sample.
                    if len(conv) == 1 and isinstance(conv[0], list):
                        conv = conv[0]
                    if ds["modality"] == "video":
                        raw = tf.extractfile(media).read()
                        info = _video_duration_and_size(raw)
                        if info is None:
                            n_failed += 1
                            continue
                        tt, vt = _exact_sample_tokens(conv, [], [info])
                    else:
                        sizes = []
                        for m in media:
                            raw = tf.extractfile(m).read()
                            sizes.append(Image.open(io.BytesIO(raw)).size)
                        tt, vt = _exact_sample_tokens(conv, sizes, [])
                        for w, h in sizes:
                            sum_w += w
                            sum_h += h
                            sum_ratio += w / h
                            n_images += 1
                    sum_text += tt
                    sum_vision += vt
                    n_ok += 1
                except Exception:  # noqa: BLE001 — one bad sample shouldn't kill the shard
                    n_failed += 1
    except Exception as e:  # noqa: BLE001 — one bad shard shouldn't kill the dataset
        return {
            "dataset_key": dataset_key, "shard": os.path.basename(shard_path),
            "n_ok": 0, "n_failed": 0, "sum_text": 0, "sum_vision": 0,
            "sum_w": 0, "sum_h": 0, "sum_ratio": 0.0, "n_images": 0, "shard_error": f"{type(e).__name__}: {e}",
        }
    return {
        "dataset_key": dataset_key, "shard": os.path.basename(shard_path),
        "n_ok": n_ok, "n_failed": n_failed, "sum_text": sum_text, "sum_vision": sum_vision,
        "sum_w": sum_w, "sum_h": sum_h, "sum_ratio": sum_ratio, "n_images": n_images, "shard_error": None,
    }


def _worker(args) -> dict:
    dataset_key, ds, shard_path, outdir = args
    result = process_shard(dataset_key, ds, shard_path)
    ds_dir = outdir / dataset_key
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / f"{result['shard']}.json").write_text(json.dumps(result))
    return result


def combine(outdir: Path, datasets: list[dict]) -> None:
    by_key = {_dataset_key(ds): ds for ds in datasets}
    rows = []
    for ds_dir in sorted(outdir.iterdir()):
        if not ds_dir.is_dir():
            continue
        key = ds_dir.name
        ds = by_key.get(key, {"modality": "?", "category": "?", "name": key, "path": ""})
        n_ok = n_failed = sum_text = sum_vision = 0
        sum_w = sum_h = n_images = 0
        sum_ratio = 0.0
        shard_errors = []
        for f in ds_dir.glob("*.json"):
            r = json.loads(f.read_text())
            n_ok += r["n_ok"]
            n_failed += r["n_failed"]
            sum_text += r["sum_text"]
            sum_vision += r["sum_vision"]
            # .get() with a default: older per-shard files (pre image-size tracking) lack these
            # keys -- treated as contributing 0 rather than crashing combine() on mixed old+new data.
            sum_w += r.get("sum_w", 0)
            sum_h += r.get("sum_h", 0)
            sum_ratio += r.get("sum_ratio", 0.0)
            n_images += r.get("n_images", 0)
            if r.get("shard_error"):
                shard_errors.append((r["shard"], r["shard_error"]))
        row = {
            **ds, "dataset_key": key, "n_ok": n_ok, "n_failed": n_failed,
            "total_text_tokens": sum_text, "total_vision_tokens": sum_vision,
            "total_tokens": sum_text + sum_vision, "shard_errors": shard_errors,
        }
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
    print(f"\n{len(rows)} datasets, grand total EXACT tokens (all samples, all shards): {grand_total:,}")

    (outdir / "_combined_summary.json").write_text(json.dumps({"grand_total_tokens": grand_total, "datasets": rows}, indent=2))


def main() -> None:
    """Exact tokens/dataset (text+vision) over every sample of every prepared energon dataset."""
    parser = argparse.ArgumentParser(description="Exact token census across all energon datasets")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    parser.add_argument("--combine-only", action="store_true", help="Skip processing, just re-summarize --outdir")
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category filter across all modalities, e.g. 'captioning,knowledge'. Default: all.",
    )
    args = parser.parse_args()

    categories = set(args.categories.split(",")) if args.categories else None
    datasets = discover_datasets(categories)

    if not args.combine_only:
        import multiprocessing as mp

        tasks = []
        for ds in datasets:
            key = _dataset_key(ds)
            for shard in sorted(Path(ds["path"]).glob("shard-*.tar")):
                tasks.append((key, ds, str(shard), args.outdir))
        print(f"Discovered {len(datasets)} datasets, {len(tasks)} shard tasks. Workers={args.workers}")

        done = 0
        with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
            for r in pool.imap_unordered(_worker, tasks, chunksize=4):
                done += 1
                if done % 200 == 0 or done == len(tasks):
                    print(f"[{done}/{len(tasks)}] shards done", flush=True)

    combine(args.outdir, datasets)


if __name__ == "__main__":
    main()
