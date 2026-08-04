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

"""Energon task encoder for EuroVL (MoonViT + EuroLLM).

EuroVL data is curated as a **CrudeWebdataset** with a single, category-agnostic
sample structure: each sample is a raw ``{key}.jpg`` plus a ``{key}.json`` that
already holds the full ChatML conversation
(``[{"role": "user", "content": "<image>\\n..."}, {"role": "assistant", ...}]``).
The same layout is used for captioning, VQA, OCR, etc. — only the conversation
content differs — so a single cooker handles every source.

"Crude" means energon does not decode the sample; this encoder registers a
:class:`~megatron.energon.Cooker` that decodes the image and passes the conversation
through into a :class:`ChatMLSample`, then reuses the generic
:class:`HFEncoderVLMTaskEncoder` machinery (which drives ``EuroVLProcessor`` for
joint tokenization + MoonViT preprocessing and emits ``GenericVisualInputs`` with
``pixel_values`` + ``image_grid_thw``).
"""

import dataclasses
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
from megatron.energon import Cooker, basic_sample_keys
from megatron.energon.task_encoder.base import stateless
from PIL import Image

from megatron.bridge.data.energon.hf_encoder_task_encoder import (
    HFEncoderTaskBatch,
    HFEncoderTaskSample,
    HFEncoderVLMTaskEncoder,
)
from megatron.bridge.data.energon.task_encoder_utils import IGNORE_INDEX, ChatMLSample, cook_chatml_sample
from megatron.bridge.data.vlm_datasets.collate import create_multiturn_loss_mask_by_search
from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


# Crude-sample extensions to look for, in priority order.
_IMAGE_EXTS = ("jpg", "jpeg", "png", "image")
_VIDEO_EXTS = ("mp4", "webm", "mkv", "mov", "avi", "video")
_CONVERSATION_EXTS = ("json", "conversation", "txt")

# Multi-image samples (e.g. mi_grounding) store N images per key as separate WebDataset
# parts named `img{i}.<ext>` (i = 0-based image order, matching the conversation's leading
# `<image>` tokens positionally) instead of the single-image `jpg` part. See
# to_energon.py's `--from-multi-image-raw` writer for the producing side.
_MULTI_IMAGE_RE = re.compile(r"^img(\d+)\.(?:jpg|jpeg|png)$")

# Multi-video samples store N videos per key as separate WebDataset parts named
# `vid{i}.<ext>` (0-based video order, matching the conversation's leading `<video>` tokens),
# mirroring the multi-image `img{i}` layout. A single-video sample uses a bare `mp4` part.
_MULTI_VIDEO_RE = re.compile(r"^vid(\d+)\.(?:mp4|webm|mkv|mov|avi)$")


@dataclass
class EuroVLPackedSample(HFEncoderTaskSample):
    """An ``HFEncoderTaskSample`` that already concatenates several samples into one
    fill-to-``seq_length`` sequence, carrying the per-sub-sequence boundaries so the
    batch step can emit THD ``cu_seqlens`` for varlen attention.
    """

    cu_seqlens: Optional[torch.Tensor] = None  # [num_subseq + 1] real (unpadded) boundaries
    seqlens: List[int] = field(default_factory=list)  # per-sub-sequence token lengths


@dataclass
class EuroVLPackedBatch(HFEncoderTaskBatch):
    """A packed ``[1, seq_length]`` batch carrying THD metadata for varlen attention."""

    cu_seqlens: Optional[torch.Tensor] = None
    cu_seqlens_unpadded: Optional[torch.Tensor] = None
    cu_seqlens_argmin: Optional[torch.Tensor] = None
    max_seqlen: Optional[torch.Tensor] = None


def _crude_get(sample: dict, keys: tuple[str, ...]) -> Optional[Any]:
    """Return the first present key from a crude energon sample dict."""
    for k in keys:
        if k in sample:
            return sample[k]
    return None


class EuroVLTaskEncoder(HFEncoderVLMTaskEncoder):
    """Crude-sample task encoder for EuroVL energon datasets.

    Category-agnostic: every source shares the same crude structure (``jpg`` +
    ``json`` conversation), so one cooker serves captioning, VQA, OCR, etc.

    Args:
        processor: An ``EuroVLProcessor`` (supports ``apply_chat_template`` and
            ``__call__(text=, images=)`` returning ``pixel_values`` + ``image_grid_thw``).
        seq_length: Maximum sequence length (tokens truncated to this).
    """

    def __init__(
        self,
        processor,
        seq_length: int = 4096,
        sqrt_loss_weighting: bool = False,
    ) -> None:
        # EuroVLProcessor returns pixel_values + image_grid_thw for images and
        # pixel_values_videos + video_grid_thw for videos; capture all four so
        # GenericVisualInputs forwards them to EuroVLModel and the FLOP counter sees the grids.
        super().__init__(
            processor=processor,
            seq_length=seq_length,
            visual_keys=("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"),
        )
        # Square-root per-token loss reweighting (InternVL3.5 eq. 2). When True, each
        # supervised token's loss_mask weight is 1/sqrt(N) (N = supervised tokens in the
        # sample) instead of 1, so a sample's gradient scales with sqrt(N) rather than N.
        # Baked in per-sample here => packing-independent. Must be paired with
        # model.calculate_per_token_loss=True (Megatron then divides by the global sum of
        # weights, reproducing eq. 2 exactly).
        self.sqrt_loss_weighting = sqrt_loss_weighting
        # Register the cooker that decodes a crude sample into a ChatMLSample. A bound
        # method is picklable (the encoder itself is sent to dataloader workers).
        self.cookers = [Cooker(cook=self._cook)]

    @property
    def _video_num_frames(self) -> int:
        """Frames to sample per video — the single source of truth is the processor's video
        processor (``num_frames``), which the processor itself re-samples to; reading it here
        keeps decode and processor in lockstep. Raises if the processor is not configured for
        video (no ``video_processor.num_frames``), since a video sample then cannot be encoded.
        """
        vp = getattr(self.processor, "video_processor", None)
        n = getattr(vp, "num_frames", None) if vp is not None else None
        if n is None:
            raise ValueError(
                "A video sample was encountered but the processor is not configured for video "
                "(no video_processor.num_frames). Build the processor with num_frames set."
            )
        return int(n)

    def _sample_frame_indices(self, total: int) -> list[int]:
        """``_video_num_frames`` uniformly-spaced frame indices in ``[0, total)`` (sorted, deduped).

        Same ``np.linspace`` idiom as ``MoonViTVideoProcessor._sample_frames`` — feeding the
        processor exactly ``n`` frames is then an identity re-sample.
        """
        n = self._video_num_frames
        if n >= total:
            return list(range(total))
        return sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))

    @staticmethod
    def _frame_to_pil(f) -> Image.Image:
        """Coerce a single decoded frame (PIL, or ``[C,H,W]``/``[H,W,C]`` uint8 tensor/array) to PIL RGB."""
        if isinstance(f, Image.Image):
            return f.convert("RGB")
        if isinstance(f, torch.Tensor):
            # energon VideoData frames are [C,H,W]; torchvision are [H,W,C]. fromarray wants HWC.
            if f.ndim == 3 and f.shape[0] in (1, 3) and f.shape[-1] not in (1, 3):
                f = f.permute(1, 2, 0)
            f = f.cpu().numpy()
        else:
            f = np.asarray(f)
        return Image.fromarray(f).convert("RGB")

    def _frames_from_video(self, v) -> list[Image.Image]:
        """Sample ``_video_num_frames`` PIL frames from whatever form the crude sample carries.

        Energon auto-decodes ``.mp4`` into ``megatron.energon.av.video_data.VideoData`` whose
        ``.frames`` is a ``[T, C, H, W]`` uint8 tensor, so most video items arrive already
        decoded; raw bytes (auto-decode off) go through the memory-bounded PyAV path instead.
        Also handles a torchvision ``(vframes[T,H,W,C], …)`` tuple, a bare frame tensor, or a
        list of per-frame PIL/tensor images.
        """
        if isinstance(v, (bytes, bytearray)):
            return self._decode_video_bytes(v)
        # energon VideoData -> .frames [T,C,H,W]; torchvision -> .vframes / tuple[0] [T,H,W,C].
        frames = getattr(v, "frames", None)
        if frames is None:
            frames = getattr(v, "vframes", None)
        if frames is None:
            frames = v[0] if isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0], torch.Tensor) else v
        if isinstance(frames, torch.Tensor):
            total = int(frames.shape[0])
            if total == 0:
                raise ValueError("decoded video has 0 frames")
            return [self._frame_to_pil(frames[i]) for i in self._sample_frame_indices(total)]
        if isinstance(frames, (list, tuple)):
            total = len(frames)
            if total == 0:
                raise ValueError("decoded video has 0 frames")
            return [self._frame_to_pil(frames[i]) for i in self._sample_frame_indices(total)]
        raise TypeError(f"unsupported video item type {type(v)} (frames {type(frames)})")

    def _decode_video_bytes(self, video_bytes: bytes) -> list[Image.Image]:
        """Decode ``_video_num_frames`` uniformly-sampled PIL frames (PyAV), memory-bounded.

        Only the sampled frames are ever materialized — the whole clip is NOT loaded into
        memory (a multi-minute clip is thousands of frames and would OOM the dataloader
        worker, ×num_workers). Used when video arrives as raw bytes (energon video auto-decode
        off); with auto-decode on, :meth:`_frames_from_video` handles the decoded tensor.
        """
        import av

        # 1. Total frame count, cheaply: container metadata, else estimate from duration × rate,
        #    else a decode-and-discard counting pass (O(1) memory — frames are not retained).
        with av.open(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            total = int(stream.frames or 0)
            if total <= 0 and stream.duration is not None and stream.average_rate:
                total = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
        if total <= 0:
            with av.open(io.BytesIO(video_bytes)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                total = sum(1 for _ in container.decode(stream))
        if total <= 0:
            raise ValueError("no frames decoded from video bytes")

        # 2. Uniformly-spaced target indices (shared sampler).
        targets = set(self._sample_frame_indices(total))

        # 3. Single decode pass, keeping ONLY the target frames (peak memory O(n)).
        frames: list[Image.Image] = []
        with av.open(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                if i in targets:
                    frames.append(frame.to_image().convert("RGB"))
                    if len(frames) >= len(targets):
                        break
        if not frames:
            raise ValueError("no frames decoded from video bytes")
        return frames

    def _cook(self, sample: dict) -> ChatMLSample:
        """Decode a crude sample (``jpg``/``mp4`` + ``json`` conversation) into a :class:`ChatMLSample`.

        Energon's webdataset decoder already turns ``jpg`` into a CHW tensor and ``json``
        into a parsed object, so we pass those through: ``HFEncoderVLMTaskEncoder``
        converts image/video tensors to PIL (``_images_to_pil`` / ``_videos_to_pil``) and
        ``cook_chatml_sample`` parses the conversation. Raw-bytes inputs are handled too, in
        case a different energon decode config is used. A sample may carry images, videos, or
        both (at least one is required); images/videos line up positionally with the
        conversation's leading ``<image>`` / ``<video>`` tokens.
        """
        # Multi-image sample: one or more `img{i}.<ext>` parts, sorted by index so they line
        # up positionally with the conversation's leading `<image>` tokens. Falls through to
        # the single-image `jpg` lookup below when none are present (every other dataset).
        multi_image_matches = sorted(
            ((int(m.group(1)), k) for k in sample if (m := _MULTI_IMAGE_RE.match(k))),
            key=lambda pair: pair[0],
        )
        imgs: Optional[list] = None
        if multi_image_matches:
            imgs = [sample[k] for _, k in multi_image_matches]
            imgs = [Image.open(io.BytesIO(im)).convert("RGB") if isinstance(im, (bytes, bytearray)) else im for im in imgs]
        else:
            raw_img = _crude_get(sample, _IMAGE_EXTS)
            if raw_img is not None:
                # tensor / PIL -> pass through (converted to PIL downstream); bytes -> decode here.
                if isinstance(raw_img, (bytes, bytearray)):
                    raw_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
                imgs = [raw_img]

        # Multi-video sample: `vid{i}.<ext>` parts (positional with `<video>` tokens); else a
        # single bare `mp4` part. Each is decoded to a list of frames; None when no video part.
        multi_video_matches = sorted(
            ((int(m.group(1)), k) for k in sample if (m := _MULTI_VIDEO_RE.match(k))),
            key=lambda pair: pair[0],
        )
        videos: Optional[list] = None
        raw_videos = [sample[k] for _, k in multi_video_matches] if multi_video_matches else None
        if raw_videos is None:
            single = _crude_get(sample, _VIDEO_EXTS)
            raw_videos = [single] if single is not None else None
        if raw_videos is not None:
            # Normalize each video to a list of sampled PIL frames, regardless of whether energon
            # auto-decoded it (torchvision tensor) or left it as raw bytes.
            videos = [self._frames_from_video(v) for v in raw_videos]

        if imgs is None and videos is None:
            raise KeyError(
                f"crude sample has no image or video (looked for {_IMAGE_EXTS} / {_VIDEO_EXTS}); "
                f"keys={list(sample.keys())}"
            )

        raw_conv = _crude_get(sample, _CONVERSATION_EXTS)
        if raw_conv is None:
            raise KeyError(
                f"crude sample has no conversation (looked for {_CONVERSATION_EXTS}); keys={list(sample.keys())}"
            )
        if isinstance(raw_conv, (bytes, bytearray)):
            raw_conv = raw_conv.decode("utf-8")
        # ChatMLSample.conversation is a JSON string; serialize if energon already parsed it.
        if not isinstance(raw_conv, str):
            raw_conv = json.dumps(raw_conv)

        return ChatMLSample(
            **basic_sample_keys(sample),
            conversation=raw_conv,
            imgs=imgs,
            videos=videos,
        )

        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # encode_sample
    # ------------------------------------------------------------------
    def encode_sample(self, sample: ChatMLSample) -> HFEncoderTaskSample:
        """Encode like the generic HF encoder, but build the loss mask with the
        repo-standard search helper.

        The base ``HFEncoderVLMTaskEncoder`` masks via a naive exact-token search of the
        standalone assistant text, which fails for EuroLLM's SentencePiece tokenizer (a
        response following a newline tokenizes without its leading ``▁``). We reuse
        ``create_multiturn_loss_mask_by_search`` — the same helper every VLM collate uses
        (qwen2_5, glm4v, ministral3, the EuroVL mock path) — which searches the *final*
        ``input_ids`` (robust to ``<image>`` expansion) with newline-context candidates.
        """
        encoded = super().encode_sample(sample)

        conversation = cook_chatml_sample(sample.conversation)
        skipped = extract_skipped_token_ids(self.processor)
        mask = create_multiturn_loss_mask_by_search(
            {"conversation": conversation}, encoded.input_ids, self.processor, skipped
        )
        loss_mask = torch.tensor(mask, dtype=torch.float32)

        # Shift to align the loss with next-token labels (same convention as the base encoder).
        shifted = torch.zeros_like(loss_mask)
        shifted[:-1] = loss_mask[1:]
        labels = encoded.input_ids.clone().to(torch.long)
        labels[:-1] = encoded.input_ids[1:].to(torch.long)
        labels[-1] = IGNORE_INDEX
        labels[shifted == 0] = IGNORE_INDEX

        # Square-root per-token loss reweighting (InternVL3.5 eq. 2): scale this sample's
        # supervised tokens by 1/sqrt(N), N = number of supervised (response) tokens. Count
        # N on the still-binary mask, then divide. Pairs with calculate_per_token_loss=True.
        if self.sqrt_loss_weighting:
            num_supervised = int((shifted > 0).sum())
            if num_supervised > 0:
                shifted = shifted / (num_supervised**0.5)

        encoded.loss_mask = shifted
        encoded.labels = labels
        return encoded

    # ------------------------------------------------------------------
    # Fill-to-seq_length packing (InternVL / Nemotron-Nano-V2 style)
    #
    # Energon enables this when ``get_train_dataset(..., packing_buffer_size=N)`` is set:
    # it buffers N encoded samples and calls ``select_samples_to_pack`` (Search) then
    # ``pack_selected_samples`` (Pack). The packed sample carries ``cu_seqlens`` so the
    # batch is emitted as a single ``[1, seq_length]`` THD sequence whose sub-sequences
    # attend independently. This is decoupled from ``micro_batch_size`` (must be 1) and is
    # mutually exclusive with ``pack_sequences_in_batch`` (the vlm_step in-batch packer).
    # ------------------------------------------------------------------

    def select_samples_to_pack(self, samples: List[HFEncoderTaskSample]) -> List[List[HFEncoderTaskSample]]:
        """Group encoded samples into bins that each fit within ``seq_length`` (first-fit-decreasing).

        Operates on ``input_ids`` token lengths only (cheap, combinatorial). A local FFD is
        used deliberately — the diffusion ``first_fit_decreasing`` helper expects ``list[int]``
        lengths, not sample objects.
        """
        capacity = self.seq_length
        order = sorted(range(len(samples)), key=lambda i: int(samples[i].input_ids.shape[0]), reverse=True)
        bins: List[List[HFEncoderTaskSample]] = []
        bin_lens: List[int] = []
        for i in order:
            length = min(int(samples[i].input_ids.shape[0]), capacity)  # encode_sample already truncates
            placed = False
            for b in range(len(bins)):
                if bin_lens[b] + length <= capacity:
                    bins[b].append(samples[i])
                    bin_lens[b] += length
                    placed = True
                    break
            if not placed:
                bins.append([samples[i]])
                bin_lens.append(length)
        return bins

    @stateless
    def pack_selected_samples(self, samples: List[HFEncoderTaskSample]) -> EuroVLPackedSample:
        """Concatenate a selected group into one packed sample (Pack phase).

        Concatenates ``input_ids``/``labels``/``loss_mask`` and the visual tensors
        (``pixel_values`` + ``image_grid_thw``) **in group order** so the flattened image
        tokens line up with ``pixel_values`` for the model's ``masked_scatter``. Records the
        per-sub-sequence boundaries (``cu_seqlens``/``seqlens``); padding to ``seq_length`` and
        the THD key build happen in :meth:`batch`.
        """
        seqlens = [int(s.input_ids.shape[0]) for s in samples]
        input_ids = torch.cat([s.input_ids for s in samples], dim=0)
        labels = torch.cat([s.labels for s in samples], dim=0)
        loss_mask = torch.cat([s.loss_mask for s in samples], dim=0)

        visual_tensors: dict[str, torch.Tensor] = {}
        for key in self.visual_keys:
            parts = [
                s.visual_tensors[key] for s in samples if key in s.visual_tensors and s.visual_tensors[key] is not None
            ]
            if parts:
                visual_tensors[key] = torch.cat(parts, dim=0)

        cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32).cumsum(0)

        return EuroVLPackedSample(
            __key__=samples[0].__key__,
            __subflavors__=samples[0].__subflavors__,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            visual_tensors=visual_tensors,
            cu_seqlens=cu_seqlens,
            seqlens=seqlens,
        )

    def batch(self, samples: List[HFEncoderTaskSample]) -> HFEncoderTaskBatch:
        """Collate packed samples into a ``[1, seq_length]`` THD batch; else defer to base.

        Pads the single packed sequence up to ``seq_length`` and absorbs the trailing pad into
        ``cu_seqlens`` as a final pad-only sub-sequence (``loss_mask=0``), so ``tokens.shape[1]
        == seq_length == cu_seqlens[-1]`` always (fixed-shape, 128-multiple). ``position_ids``
        restart per sub-sequence (required for THD RoPE).
        """
        if not samples or not isinstance(samples[0], EuroVLPackedSample):
            return super().batch(samples)

        assert len(samples) == 1, (
            f"EuroVL energon packing yields one packed sequence per batch; got {len(samples)}. "
            "Set train.micro_batch_size=1 (THD/CP requires it)."
        )
        s = samples[0]
        pad_id = self._pad_token_id
        target_len = self.seq_length
        total = int(s.input_ids.shape[0])
        assert total <= target_len, f"packed length {total} exceeds seq_length {target_len}"
        pad_len = target_len - total

        input_ids = torch.zeros(target_len, dtype=s.input_ids.dtype)
        input_ids[:total] = s.input_ids
        input_ids[input_ids == pad_id] = 0  # model input expects pad -> 0 (matches base encoder)

        labels = torch.full((target_len,), IGNORE_INDEX, dtype=torch.long)
        labels[:total] = s.labels.to(torch.long)

        loss_mask = torch.zeros(target_len, dtype=torch.float32)
        loss_mask[:total] = s.loss_mask.to(torch.float32)

        # cu_seqlens / position_ids cover the full padded length; trailing pad is its own sub-seq.
        seqlens_full = list(s.seqlens) + ([pad_len] if pad_len > 0 else [])
        position_ids = torch.cat([torch.arange(length, dtype=torch.long) for length in seqlens_full])

        cu_seqlens = torch.zeros(len(seqlens_full) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(seqlens_full, dtype=torch.int32).cumsum(0)
        max_seqlen = torch.tensor(max(seqlens_full), dtype=torch.int32)
        # No sentinel padding in cu_seqlens -> argmin = len keeps every entry (see get_packed_seq_params).
        cu_seqlens_argmin = torch.tensor(len(cu_seqlens), dtype=torch.int32)

        batch_kwargs: dict = dict(
            __keys__=[s.__key__],
            __subflavors__=[s.__subflavors__],
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            loss_mask=loss_mask.unsqueeze(0),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0),
            visual_tensors=dict(s.visual_tensors),
            cu_seqlens=cu_seqlens,
            cu_seqlens_unpadded=cu_seqlens.clone(),
            cu_seqlens_argmin=cu_seqlens_argmin,
            max_seqlen=max_seqlen,
        )
        # Energon's Batch base may expose __key__ / __restore_key__ as init fields (varies by
        # version); only pass them when they are settable (mirrors HFEncoderVLMTaskEncoder.batch).
        init_fields = {f.name for f in dataclasses.fields(EuroVLPackedBatch) if f.init}
        if "__key__" in init_fields:
            batch_kwargs["__key__"] = s.__key__
        if "__restore_key__" in init_fields:
            batch_kwargs["__restore_key__"] = ()
        return EuroVLPackedBatch(**batch_kwargs)

    def encode_batch(self, batch: HFEncoderTaskBatch) -> dict:
        """Emit THD batch keys for the packed path; else defer to the base encoder."""
        if not isinstance(batch, EuroVLPackedBatch):
            return super().encode_batch(batch)
        vt = batch.visual_tensors if batch.visual_tensors else {}
        return {
            "tokens": batch.input_ids,
            "labels": batch.labels,
            "loss_mask": batch.loss_mask,
            "attention_mask": batch.attention_mask,
            "position_ids": batch.position_ids,
            "cu_seqlens": batch.cu_seqlens,
            "cu_seqlens_unpadded": batch.cu_seqlens_unpadded,
            "cu_seqlens_argmin": batch.cu_seqlens_argmin,
            "max_seqlen": batch.max_seqlen,
            "visual_inputs": GenericVisualInputs(**{k: v for k, v in vt.items() if v is not None}),
        }
