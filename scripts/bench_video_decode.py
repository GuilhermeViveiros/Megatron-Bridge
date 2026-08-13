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

"""Benchmark strategies for sampling N frames from a video (the EuroVL dataloader hot path).

The training bottleneck is decode-time stragglers: the current path (``baseline``) decodes the
WHOLE clip sequentially to extract a handful of uniformly-spaced frames — and decodes it a second
time just to count frames when metadata is missing (common for the odd H.264 clips that throw
``libav`` warnings). At 16-24 DP ranks the slowest of N dataloaders gates every step, so the decode
*tail* (p95/max), not the median, is what stalls training.

This compares the current approach against seek-based ones (which skip to each target frame instead
of decoding everything) on REAL molmo2_cap clips, reporting the time distribution per strategy.

Run (in container, no GPU needed)::

    uv run --no-sync python scripts/bench_video_decode.py \
        --shards /e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data/video/captioning/molmo2_cap \
        --num-clips 80 --num-frames 4
"""

import argparse
import io
import statistics
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------------
# Clip loading (pull raw .mp4 bytes straight out of the energon webdataset tar shards)
# --------------------------------------------------------------------------------------------------
def load_clips(shards_dir: str, num_clips: int) -> list[tuple[str, bytes]]:
    """Return up to ``num_clips`` ``(name, mp4_bytes)`` pulled from the shard tars."""
    clips: list[tuple[str, bytes]] = []
    for tar_path in sorted(Path(shards_dir).glob("*.tar")):
        with tarfile.open(tar_path) as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith((".mp4", ".mkv", ".webm", ".avi")):
                    data = tf.extractfile(m).read()
                    clips.append((m.name, data))
                    if len(clips) >= num_clips:
                        return clips
    return clips


def _uniform_indices(total: int, n: int) -> list[int]:
    """``n`` uniformly-spaced frame indices in ``[0, total)`` (matches the task encoder's sampler)."""
    if n >= total:
        return list(range(total))
    return sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))


# --------------------------------------------------------------------------------------------------
# Strategies: each returns a list of frames (RGB uint8 HxWx3 ndarrays) for the sampled positions.
# --------------------------------------------------------------------------------------------------
def s_baseline(b: bytes, n: int) -> list[np.ndarray]:
    """Current task-encoder logic: count frames (decode-and-discard if no metadata), then a single
    sequential decode keeping only the targets (but iterates up to the LAST target ~= end of clip)."""
    import av

    with av.open(io.BytesIO(b)) as c:
        s = c.streams.video[0]
        total = int(s.frames or 0)
        if total <= 0 and s.duration is not None and s.average_rate:
            total = int(float(s.duration * s.time_base) * float(s.average_rate))
    if total <= 0:  # decode-and-discard counting pass (whole clip, O(1) mem)
        with av.open(io.BytesIO(b)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            total = sum(1 for _ in c.decode(s))
    targets = set(_uniform_indices(total, n))
    out: list[np.ndarray] = []
    with av.open(io.BytesIO(b)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i in targets:
                out.append(frame.to_ndarray(format="rgb24"))
                if len(out) >= len(targets):
                    break
    return out


def s_pyav_seek(b: bytes, n: int) -> list[np.ndarray]:
    """Seek to each target TIME (keyframe <= t) and decode forward to the first frame >= t.
    Uses duration only (no frame count), so it never does the count-pass."""
    import av

    out: list[np.ndarray] = []
    with av.open(io.BytesIO(b)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        dur = float(s.duration * s.time_base) if s.duration else (float(c.duration) / 1e6 if c.duration else 0.0)
        if dur <= 0:  # last resort: fall back to baseline behavior for this clip
            return s_baseline(b, n)
        times = [dur / 2] if n == 1 else [i * dur / (n - 1) for i in range(n)]
        for t in times:
            c.seek(int(t / s.time_base), stream=s, backward=True, any_frame=False)
            for frame in c.decode(s):
                if frame.time is not None and frame.time >= t - 1e-3:
                    out.append(frame.to_ndarray(format="rgb24"))
                    break
            else:  # decoded past end without a hit -> take the last decoded frame if any
                pass
    return out


def s_cv2_seek(b: bytes, n: int) -> list[np.ndarray]:
    """OpenCV VideoCapture with POS_FRAMES seeking (metadata frame count, no decode to count)."""
    import cv2

    out: list[np.ndarray] = []
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(b)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return s_baseline(b, n)
        for idx in _uniform_indices(total, n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if ok:
                out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cap.release()
    return out


def s_torchvision(b: bytes, n: int) -> list[np.ndarray]:
    """torchvision VideoReader with time seek (guarded — backend may be unavailable)."""
    from torchvision.io import VideoReader

    out: list[np.ndarray] = []
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(b)
        f.flush()
        vr = VideoReader(f.name, "video")
        dur = float(vr.get_metadata()["video"]["duration"][0])
        times = [dur / 2] if n == 1 else [i * dur / (n - 1) for i in range(n)]
        for t in times:
            vr.seek(t)
            frame = next(vr)
            out.append(frame["data"].permute(1, 2, 0).contiguous().numpy())
    return out


STRATEGIES = {
    "baseline": s_baseline,
    "pyav_seek": s_pyav_seek,
    "cv2_seek": s_cv2_seek,
    "torchvision": s_torchvision,
}


def main() -> None:
    """Load real clips, run every strategy, and print the per-strategy time distribution."""
    ap = argparse.ArgumentParser(description="Benchmark video frame-sampling strategies on real clips")
    ap.add_argument(
        "--shards",
        default="/e/scratch/e-ext-2025e01-100/EuroVL-Data/energon-data/video/captioning/molmo2_cap",
    )
    ap.add_argument("--num-clips", type=int, default=80)
    ap.add_argument("--num-frames", type=int, default=4)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    args = ap.parse_args()

    print(f"Loading up to {args.num_clips} clips from {args.shards} ...")
    clips = load_clips(args.shards, args.num_clips)
    sizes_mb = [len(b) / 1e6 for _, b in clips]
    print(f"  loaded {len(clips)} clips; size MB: median={statistics.median(sizes_mb):.1f} "
          f"max={max(sizes_mb):.1f}\n")

    print(f"{'strategy':<12} {'ok':>4} {'fail':>4} {'median_ms':>10} {'p95_ms':>9} {'max_ms':>9} {'total_s':>8}")
    print("-" * 62)
    for name in args.strategies:
        fn = STRATEGIES[name]
        times_ms, fails = [], 0
        t_all = time.perf_counter()
        for _, b in clips:
            t0 = time.perf_counter()
            try:
                frames = fn(b, args.num_frames)
                dt = (time.perf_counter() - t0) * 1e3
                if len(frames) == 0:
                    fails += 1
                else:
                    times_ms.append(dt)
            except Exception:
                fails += 1
        total_s = time.perf_counter() - t_all
        if times_ms:
            p95 = sorted(times_ms)[min(len(times_ms) - 1, int(0.95 * len(times_ms)))]
            print(f"{name:<12} {len(times_ms):>4} {fails:>4} {statistics.median(times_ms):>10.1f} "
                  f"{p95:>9.1f} {max(times_ms):>9.1f} {total_s:>8.1f}")
        else:
            print(f"{name:<12} {0:>4} {fails:>4} {'-':>10} {'-':>9} {'-':>9} {total_s:>8.1f}")


if __name__ == "__main__":
    main()
