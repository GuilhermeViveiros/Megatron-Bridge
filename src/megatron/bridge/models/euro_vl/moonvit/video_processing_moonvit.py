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

"""Video processor for MoonViT (per-frame 2D encoding, no temporal merge).

Each frame is encoded like an image; ``video_grid_thw``'s ``t`` is just the sampled frame count.
Per-video tokens = ``t * (h // merge_h) * (w // merge_w)``.

Smart-resize policy (Qwen-inspired, see ``qwen_vl_utils.vision_process.fetch_video`` and
``sanity_check/euro_vl_video_notes.md``): frame count comes from clip duration via ``fps``
(clamped to ``[min_frames, max_frames]``); a ``total_pixels`` budget is then spent across those
frames (shrinking resolution before ever dropping frames). No Qwen ``FRAME_FACTOR`` correction —
MoonViT has no temporal merge. See ``plan_num_frames`` / ``per_frame_cap_pixels``.

Preprocessing (``vectorized_preprocess``): batched torchvision resize, adopted after a PA
loss-curve A/B (see notes doc) — the only backend, no PIL fallback. Raw-bytes decode
(``decode_video_bytes``) lives here too, not in the energon task encoder, so the processor is
self-contained for HF export (only this class travels with an exported checkpoint). Requires
reliable container duration — no sequential-decode fallback; a clip missing it should be
re-muxed at the data-prep stage.
"""

import io
import logging
import math
from typing import Optional, Union

import av
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from transformers.image_processing_utils import BatchFeature
from transformers.utils import TensorType

from megatron.bridge.models.euro_vl.moonvit.image_processing_moonvit import MoonViTImageProcessor


logger = logging.getLogger(__name__)

# Many real-world H.264 clips are lightly corrupt (missing reference frames); libav logs a benign
# per-frame ERROR for each ("co located POCs unavailable", "Missing reference picture", ...) and
# keyframe seeking amplifies the volume. These are non-fatal — decoding continues and the frames
# are usable — so silence libav's own chatter (routed through the "libav" Python logger). Genuine,
# unrecoverable decode failures still raise ``av.error`` exceptions, which are unaffected.
logging.getLogger("libav").setLevel(logging.CRITICAL)


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Aspect-preserving resize target, snapped to a ``factor``-aligned grid.

    Own local implementation (not a vendored import) of the same sqrt-scale-to-fit algorithm
    MoonViT's own image ``rescale`` uses, generalized to: (a) round (not floor-then-crop) to the
    nearest ``factor`` multiple when no scaling is needed — never produces an unsafe/odd merge
    grid; (b) downscale when native pixels exceed ``max_pixels``; (c) upscale when native pixels
    are below the absolute ``min_pixels`` floor (rare — protects against tiny native frames, not
    a target every frame is forced to). Matches ``qwen_vl_utils``' ``smart_resize`` structure.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be < 200, got {max(height, width) / min(height, width):.1f}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class MoonViTVideoProcessor(MoonViTImageProcessor):
    """Process videos as sequences of 2D frames through MoonViT's image pipeline."""

    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        fps: float = 1.0,
        min_frames: int = 2,
        max_frames: int = 64,
        min_pixels: int = 40_000,
        max_pixels: int = 802_816,
        total_pixels: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            fps: Target frames sampled per second of clip duration.
            min_frames / max_frames: Bounds on the fps-derived frame count.
            min_pixels: Absolute floor on a single frame's resized pixel count (never shrink
                below this, matches ``qwen_vl_utils``' ``VIDEO_MIN_TOKEN_NUM`` role).
            max_pixels: Absolute ceiling on a single frame's resized pixel count.
            total_pixels: Total pixel budget for one clip, divided across its sampled frames.
                If ``None``, defaults to a value equivalent to 0.85 * 8192 seq_length (5_457_038
                px) — callers that know the real ``seq_length`` (``EuroVLProcessor.from_pretrained``)
                should always pass it explicitly.
        """
        super().__init__(**kwargs)
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels if total_pixels is not None else 5_457_038

    @property
    def _merge_factor(self) -> int:
        """Pixels per LLM token side (patch_size * merge_kernel_size), e.g. 14*2 = 28."""
        return int(self.patch_size) * int(self.merge_kernel_size[0])

    def plan_num_frames(self, duration: float, total_frames: int) -> int:
        """fps-driven frame count for a clip, clamped to bounds and the pixel budget.

        Args:
            duration: Clip duration in seconds.
            total_frames: Total decodable frames in the clip (can't sample more than this).
        """
        if total_frames <= 0:
            raise ValueError("total_frames must be positive")
        n = round(duration * self.fps) if duration > 0 else self.min_frames
        n = max(self.min_frames, min(n, self.max_frames))
        # Budget ceiling: sampling more frames than total_pixels/min_pixels affords is pointless
        # — per_frame_cap_pixels would just clamp back down to min_pixels anyway.
        budget_ceiling = max(1, self.total_pixels // self.min_pixels)
        n = min(n, budget_ceiling, total_frames)
        return max(1, n)

    def per_frame_cap_pixels(self, n: int) -> int:
        """Per-frame pixel cap that spends ``total_pixels`` evenly across ``n`` frames.

        No ``FRAME_FACTOR`` correction (unlike ``qwen_vl_utils``): MoonViT has no temporal
        merge, so every sampled frame costs its own full token count.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        target = self.total_pixels // n
        return max(self.min_pixels, min(self.max_pixels, target))

    def sample_frame_indices(self, duration: float, total: int) -> list[int]:
        """``plan_num_frames``-many uniformly-spaced frame indices in ``[0, total)`` (sorted, deduped).

        Used for video items that arrive already decoded (e.g. energon auto-decode) — the caller
        has all ``total`` frames in hand and just needs to know which ones to keep.
        """
        n = self.plan_num_frames(duration, total)
        if n >= total:
            return list(range(total))
        return sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))

    def decode_video_bytes(self, video_bytes: bytes) -> tuple[list[Image.Image], list[float]]:
        """Decode ``plan_num_frames``-many PIL frames + their timestamps (seconds) via keyframe seeking.

        Seeks to the nearest keyframe at each target timestamp and decodes forward, instead of
        walking the whole clip. Requires stream/container duration; raises if absent
        """
        frames: list[Optional[Image.Image]] = []

        with av.open(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration) / 1_000_000.0  # AV_TIME_BASE (microseconds)
            else:
                duration = 0.0
            if duration <= 0:
                raise ValueError(
                    "clip exposes no container/stream duration; cannot seek-decode. Re-mux the "
                    "source clip (e.g. `ffmpeg -i in.mp4 -c copy out.mp4`) to add duration metadata."
                )

            # total_frames is only a safety cap in plan_num_frames (don't request more frames
            # than exist); duration * average_rate is the standard, sufficiently reliable
            # estimate for that — no need for an exact count here.
            if stream.average_rate:
                total = max(1, int(duration * float(stream.average_rate)))
            else:
                logger.warning("stream.average_rate not found; using default total_frames=10**9")
                total = 10**9
            n = self.plan_num_frames(duration, total)

            # Uniformly-spaced target times across the clip (frame-accurate: decode fwd to >= t).
            times = [duration / 2.0] if n == 1 else [i * duration / (n - 1) for i in range(n)]
            try:
                frames = self._decode_at_times(container, stream, times)
            except av.error.PermissionError:
                # Stream-copy cut clips may not start on a keyframe: everything before the first
                # keyframe is undecodable, and a backward seek to a target there fails with EPERM.
                # Re-space the same n targets over the decodable span [first frame, duration] and
                # decode forward from the start in one pass (no seeks), so every frame is still the
                # first one at/after its reported time. Seeking near the first keyframe is not
                # reliable on these clips (it can land on the NEXT keyframe), hence no seeks here.
                container.seek(0, stream=stream, backward=True, any_frame=False)
                first = next(container.decode(stream), None)
                if first is None or first.time is None or first.time >= duration:
                    raise
                t0 = float(first.time)
                times = [(t0 + duration) / 2.0] if n == 1 else [t0 + i * (duration - t0) / (n - 1) for i in range(n)]
                frames = self._decode_sequential(container, stream, times)

        if not frames or any(f is None for f in frames):
            raise ValueError("no frames decoded from video bytes")
        return frames, times  # type: ignore[return-value]

    @staticmethod
    def _decode_at_times(container, stream, times: list[float]) -> list[Optional[Image.Image]]:
        """For each target time, keyframe-seek and decode forward to the first frame at/after it."""
        frames: list[Optional[Image.Image]] = []
        last: Optional[Image.Image] = None
        for t in times:
            container.seek(int(t / stream.time_base), stream=stream, backward=True, any_frame=False)
            picked: Optional[Image.Image] = None
            for frame in container.decode(stream):
                last = frame.to_image().convert("RGB")
                if frame.time is not None and frame.time >= t - 1e-3:
                    picked = last
                    break
            # Guarantee exactly n frames: if the seek overshot the end, reuse the last frame.
            frames.append(picked if picked is not None else last)
        return frames

    @staticmethod
    def _decode_sequential(container, stream, times: list[float]) -> list[Optional[Image.Image]]:
        """Same picking rule as ``_decode_at_times`` (first frame at/after each sorted target), in
        one forward pass from the start of the stream instead of per-target seeks."""
        container.seek(0, stream=stream, backward=True, any_frame=False)
        frames: list[Optional[Image.Image]] = []
        last = None
        for frame in container.decode(stream):
            last = frame
            while len(frames) < len(times) and frame.time is not None and frame.time >= times[len(frames)] - 1e-3:
                frames.append(frame.to_image().convert("RGB"))
            if len(frames) == len(times):
                break
        # Guarantee exactly n frames: targets past the last decoded frame reuse it.
        tail = last.to_image().convert("RGB") if last is not None else None
        return frames + [tail] * (len(times) - len(frames))

    def _patchify_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Batched mirror of ``MoonViTImageProcessor.patchify`` over ``[T, C, H, W]`` frames that
        already share one resolution (guaranteed here, post-resize) — one reshape/permute instead
        of ``T`` Python-loop calls. Produces the identical patch order (frame-major) a per-frame
        loop + concat would, i.e. ``[T*grid_h*grid_w, C, p, p]``.
        """
        patch_size = self.patch_size
        T, C, H, W = images.shape
        patches = images.reshape(T, C, H // patch_size, patch_size, W // patch_size, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # [T, gh, gw, C, p, p]
        return patches.contiguous().view(-1, C, patch_size, patch_size)  # [T*gh*gw, C, p, p]

    def _pack(self, pixel_values: list, video_grid_thw: list, return_tensors) -> BatchFeature:
        data = {
            "pixel_values_videos": torch.concat(pixel_values, dim=0),
            "video_grid_thw": np.array(video_grid_thw),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def vectorized_preprocess(
        self,
        videos: list,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """Tensor-native video preprocessing (torchvision bicubic) — the default backend.

        One batched ``TF.resize`` per clip (all its frames share a target size), instead of a
        per-frame Python loop — required to scale to the frame counts the smart-resize policy
        can produce. See the module docstring for the PIL-vs-vectorized validation.
        """
        pixel_values, video_grid_thw = [], []
        for video in videos:
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            if isinstance(video, np.ndarray):
                # transformers' make_batched_videos already stacked same-size PIL frames into
                # [T, H, W, C] before this call (the EuroVLProcessor.__call__ path).
                arr = video
                t, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
            else:
                # Raw list of PIL frames (e.g. direct decode_video_bytes callers).
                frames = list(video)
                sizes = {f.size for f in frames}  # PIL (w, h)
                if len(sizes) != 1:
                    raise ValueError(f"Video frames have inconsistent native sizes {sizes}; expected one resolution.")
                w, h = sizes.pop()
                arr = np.stack(frames)  # [T, H, W, C] uint8
                t = arr.shape[0]

            max_pixels_cap = self.per_frame_cap_pixels(t)
            new_h, new_w = _smart_resize(
                h,
                w,
                factor=self._merge_factor,
                min_pixels=self.min_pixels,
                max_pixels=max_pixels_cap,
            )
            batch = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)  # [T, C, H, W] in [0, 1]
            if (new_h, new_w) != (h, w):
                batch = TF.resize(batch, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            batch = TF.normalize(batch, self.image_mean, self.image_std)
            pixel_values.append(self._patchify_batch(batch))
            video_grid_thw.append((t, new_h // self.patch_size, new_w // self.patch_size))
        return self._pack(pixel_values, video_grid_thw, return_tensors)
