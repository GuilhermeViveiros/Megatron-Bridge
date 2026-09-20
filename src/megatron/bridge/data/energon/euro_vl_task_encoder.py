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
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
from megatron.energon import Cooker, SkipSample, basic_sample_keys
from megatron.energon.task_encoder.base import stateless
from PIL import Image

from megatron.bridge.data.energon.hf_encoder_task_encoder import (
    HFEncoderTaskBatch,
    HFEncoderTaskSample,
    HFEncoderVLMTaskEncoder,
)
from megatron.bridge.data.energon.task_encoder_utils import (
    IGNORE_INDEX,
    ChatMLSample,
    _images_to_pil,
    _videos_to_pil,
    cook_chatml_sample,
)
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

# Message-tree samples: tokens of the shared prefix (the media turn) get this id and are visible to
# every branch; branch b gets id b. Same convention as Molmo2
# (olmo/preprocessing/text_preprocessor.py), whose mask is causal AND ids[q] <= ids[k].
ATTEND_ALL_SUBSEGMENT_ID = 10000


@dataclass
class EuroVLTaskSample(HFEncoderTaskSample):
    """An encoded sample that may carry per-token message-tree subsegment ids.

    ``subsegment_ids`` is ``None`` for ordinary (flat) samples.
    """

    subsegment_ids: Optional[torch.Tensor] = None  # [seq_len] int32


@dataclass
class EuroVLPackedSample(EuroVLTaskSample):
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
    # [1, seq_length] message-tree subsegment ids; None when no sample in the pack is a tree.
    subsegment_ids: Optional[torch.Tensor] = None


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
        seq_length: int = 8192,
        sqrt_loss_weighting: bool = False,
        root_subsegments: bool = False,
        max_num_images: int = 16,
    ) -> None:
        # EuroVLProcessor returns pixel_values + image_grid_thw for images and
        # pixel_values_videos + video_grid_thw for videos; capture all four so
        # GenericVisualInputs forwards them to EuroVLModel and the FLOP counter sees the grids.
        super().__init__(
            processor=processor,
            seq_length=seq_length,
            visual_keys=("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"),
            skip_on_truncation=True,
        )
        # Cheap pre-filter for multi-image samples: dropped before any image is decoded.
        # Doesn't guarantee a sample fits seq_length on its own (per-image token cost varies a
        # lot across datasets) -- that's what skip_on_truncation (above) enforces exactly, using
        # real computed lengths. This just rejects pathological image counts early. 16 was
        # picked from a census of every real multi-image dataset in mixture.yaml: the count-based
        # tail is concentrated exactly at/above this threshold on the ~10 highest-image-count
        # datasets (e.g. doc/leopard_mpdocvqa avg_n=11.3, max_n=40; doc/doc750k avg_n=13.9,
        # max_n=63), while every other dataset's images stay well under it.
        self.max_num_images = max_num_images
        # Square-root per-token loss reweighting (InternVL3.5 eq. 2). When True, each
        # supervised token's loss_mask weight is 1/sqrt(N) (N = supervised tokens in the
        # sample) instead of 1, so a sample's gradient scales with sqrt(N) rather than N.
        # Baked in per-sample here => packing-independent. Must be paired with
        # model.calculate_per_token_loss=True (Megatron then divides by the global sum of
        # weights, reproducing eq. 2 exactly).
        self.sqrt_loss_weighting = sqrt_loss_weighting
        # Divide a message-tree sample's weights by sqrt(number of kept branches), as Molmo2's
        # "root_subsegments" does. OFF by default so a grouped sample carries exactly the loss it
        # would as separate flat samples (branch b keeps 1/sqrt(N_b)): grouping is then a pure
        # compute optimization and cannot change training. Turning it on treats the whole group
        # like one sample of the combined answer length (sum_b sqrt(N_b)/sqrt(B) = sqrt(sum_b N_b)
        # for equal-length branches) -- i.e. it assumes annotations of one video are partly
        # redundant. It costs video datasets a factor ~sqrt(B) of weight against the flat
        # baseline, so revisit the mixture repeat factors before enabling it.
        self.root_subsegments = root_subsegments
        # Register the cooker that decodes a crude sample into a ChatMLSample. A bound
        # method is picklable (the encoder itself is sent to dataloader workers).
        self.cookers = [Cooker(cook=self._cook)]

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

    def _frames_from_video(self, v) -> tuple[list[Image.Image], Optional[list[float]]]:
        """Sample the policy-planned number of PIL frames from whatever form the crude sample
        carries. The actual decode + frame-count planning lives on ``video_processor``
        (``decode_video_bytes`` / ``sample_frame_indices``) so the processor stays self-contained
        for HF export; this method only handles ENERGON-specific frame-type dispatch.

        Energon auto-decodes ``.mp4`` into ``megatron.energon.av.video_data.VideoData`` whose
        ``.frames`` is a ``[T, C, H, W]`` uint8 tensor, so most video items arrive already
        decoded; raw bytes (auto-decode off) go through the video processor's memory-bounded
        PyAV path instead. Also handles a torchvision ``(vframes[T,H,W,C], …)`` tuple, a bare
        frame tensor, or a list of per-frame PIL/tensor images.
        """
        vp = getattr(self.processor, "video_processor", None)
        if vp is None:
            raise ValueError(
                "A video sample was encountered but the processor is not configured for video "
                "(no video_processor). Build the processor with video support."
            )
        if isinstance(v, (bytes, bytearray)):
            return vp.decode_video_bytes(v)
        # energon VideoData -> .frames [T,C,H,W]; torchvision -> .vframes / tuple[0] [T,H,W,C].
        frames = getattr(v, "frames", None)
        if frames is None:
            frames = getattr(v, "vframes", None)
        if frames is None:
            frames = v[0] if isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0], torch.Tensor) else v
        # Pre-decoded (auto_decode=True) path: timestamps from the clip's fps.
        if isinstance(frames, torch.Tensor):
            total = int(frames.shape[0])
        elif isinstance(frames, (list, tuple)):
            total = len(frames)
        else:
            raise TypeError(f"unsupported video item type {type(v)} (frames {type(frames)})")
        if total == 0:
            raise ValueError("decoded video has 0 frames")
        fps = getattr(v, "fps", None) or getattr(v, "frame_rate", None)
        if fps is None:
            raise ValueError("auto-decoded video has no fps for timestamps; use auto_decode=False")
        duration = total / float(fps)
        idxs = vp.sample_frame_indices(duration, total)
        return [self._frame_to_pil(frames[i]) for i in idxs], [i / float(fps) for i in idxs]

    def _cook(self, sample: dict) -> ChatMLSample:
        """Decode a crude sample (``jpg``/``mp4`` + ``json`` conversation) into a :class:`ChatMLSample`.

        Energon's webdataset decoder already turns ``jpg`` into a CHW tensor and ``json``
        into a parsed object, so we pass those through: ``HFEncoderVLMTaskEncoder``
        converts image/video tensors to PIL (``_images_to_pil`` / ``_videos_to_pil``) and
        ``cook_chatml_sample`` parses the conversation. Raw-bytes inputs are handled too, in
        case a different energon decode config is used. A sample may carry images, videos,
        both, or neither (text-only, e.g. ``text/*/euroblocks``); images/videos line up
        positionally with the conversation's leading ``<image>`` / ``<video>`` tokens.
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
            imgs = [
                Image.open(io.BytesIO(im)).convert("RGB") if isinstance(im, (bytes, bytearray)) else im for im in imgs
            ]
        else:
            raw_img = _crude_get(sample, _IMAGE_EXTS)
            if raw_img is not None:
                # tensor / PIL -> pass through (converted to PIL downstream); bytes -> decode here.
                if isinstance(raw_img, (bytes, bytearray)):
                    raw_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
                imgs = [raw_img]

        if imgs is not None and len(imgs) > self.max_num_images:
            logging.warning(
                "Skipping sample %s: %d images exceeds max_num_images=%d",
                sample.get("__key__", "<unknown>"),
                len(imgs),
                self.max_num_images,
            )
            raise SkipSample()

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
        video_metadata: Optional[list] = None
        if raw_videos is not None:
            # Each video -> (sampled PIL frames, per-frame timestamps in seconds).
            decoded = [self._frames_from_video(v) for v in raw_videos]
            videos = [frames for frames, _ in decoded]
            video_metadata = [{"timestamps": ts} for _, ts in decoded]

        # Text-only samples (e.g. text/*/euroblocks) carry no image/video part at all --
        # imgs/videos stay None and flow through to a text-only ChatMLSample.
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
            video_metadata=video_metadata,
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

        Samples whose JSON is a message tree (``{"message_tree": true, ...}``) are routed to
        :meth:`_encode_message_tree`; everything else takes the flat path below.
        """
        tree = self._parse_message_tree(sample.conversation)
        if tree is not None:
            return self._encode_message_tree(sample, tree)

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
    # Message-tree samples (docs/models/euro_vl/message_tree_packing.md)
    #
    # One media prefix shared by several independent branches (e.g. QA pairs about the same
    # video), encoded once. Branch boundaries are recorded as per-token subsegment ids for the
    # branch-isolation mask; until that mask exists, branches attend to each other like an
    # ordinary multi-turn conversation.
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_message_tree(conversation: Any) -> Optional[dict]:
        """Return the parsed tree if ``conversation`` is a message-tree sample, else ``None``."""
        if isinstance(conversation, (str, bytes, bytearray)):
            try:
                conversation = json.loads(conversation)
            except (TypeError, ValueError):
                return None
        if isinstance(conversation, dict) and conversation.get("message_tree") is True:
            return conversation
        return None

    def _encode_message_tree(self, sample: ChatMLSample, tree: dict) -> EuroVLTaskSample:
        """Encode ``shared`` + ``branches`` as one sequence with a single media encode.

        Turns are laid out as the shared turns followed by every branch's turns. This bypasses
        ``cook_chatml_sample``, which would relabel the media turn as ``system`` because a tree
        always has an odd number of turns.
        """
        shared, branches = tree["shared"], tree["branches"]
        key = sample.__key__
        if not branches:
            raise ValueError(f"Message-tree sample {key} has no branches")
        if len(branches) >= ATTEND_ALL_SUBSEGMENT_ID:
            raise ValueError(
                f"Message-tree sample {key} has {len(branches)} branches (limit {ATTEND_ALL_SUBSEGMENT_ID})"
            )
        for branch in branches:
            for turn in branch:
                if "<image>" in turn["content"] or "<video>" in turn["content"]:
                    raise ValueError(f"Message-tree sample {key}: media placeholders must be in 'shared' only")

        images_pil = _images_to_pil(sample.imgs) if sample.imgs else None
        videos_pil = _videos_to_pil(sample.videos) if sample.videos else None
        turns = [dict(t) for t in shared] + [dict(t) for branch in branches for t in branch]
        self._structure_media_placeholders(turns, images_pil is not None, videos_pil is not None)
        prompt_text = self.processor.apply_chat_template(turns, tokenize=False)
        proc_output = self._run_processor(prompt_text, images_pil, videos_pil, getattr(sample, "video_metadata", None))
        input_ids = proc_output["input_ids"]
        input_ids = (input_ids[0] if input_ids.dim() == 2 else input_ids).to(torch.long)
        seq_len = int(input_ids.shape[0])

        # Every turn starts with <|im_start|>, which never occurs inside a media expansion, so the
        # k-th occurrence starts the k-th turn. A mismatch means the data or template is not what
        # this layout assumes -- fail loudly rather than guess boundaries.
        im_start_id = self._tokenizer.convert_tokens_to_ids("<|im_start|>")
        turn_starts = (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        expected_turns = len(turns)
        if len(turn_starts) != expected_turns:
            raise ValueError(
                f"Message-tree sample {key}: found {len(turn_starts)} <|im_start|> tokens, "
                f"expected {expected_turns} ({len(shared)} shared + {expected_turns - len(shared)} branch turns)"
            )
        branch_starts, turn_idx = [], len(shared)
        for branch in branches:
            branch_starts.append(turn_starts[turn_idx])
            turn_idx += len(branch)
        branch_ends = branch_starts[1:] + [seq_len]

        # Loss on assistant spans only, same search as the flat path; then shift to label positions.
        skipped = extract_skipped_token_ids(self.processor)
        mask = torch.tensor(
            create_multiturn_loss_mask_by_search({"conversation": turns}, input_ids, self.processor, skipped),
            dtype=torch.float32,
        )
        for b, (start, end) in enumerate(zip(branch_starts, branch_ends)):
            if mask[start:end].sum() == 0:
                raise ValueError(f"Message-tree sample {key}: no supervised tokens found for branch {b}")
        loss_mask = torch.zeros_like(mask)
        loss_mask[:-1] = mask[1:]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = IGNORE_INDEX
        labels[loss_mask == 0] = IGNORE_INDEX

        # Over length: keep the longest prefix of whole branches that fits. Branches follow the
        # media, so this never cuts it and needs no re-encode. Skip only if no branch fits.
        n_keep = len(branches)
        if seq_len > self.seq_length:
            n_keep = sum(1 for end in branch_ends if end <= self.seq_length)
            if n_keep == 0:
                logging.warning(
                    "Skipping message-tree sample %s: first branch ends at %d > seq_length=%d",
                    key,
                    branch_ends[0],
                    self.seq_length,
                )
                raise SkipSample()
            logging.warning(
                "Message-tree sample %s: length %d > seq_length=%d; keeping %d of %d branches",
                key,
                seq_len,
                self.seq_length,
                n_keep,
                len(branches),
            )
            cut = branch_ends[n_keep - 1]
            input_ids, labels, loss_mask = input_ids[:cut], labels[:cut], loss_mask[:cut].clone()
            labels[-1] = IGNORE_INDEX  # its next token was dropped
            loss_mask[-1] = 0.0
            branch_starts, branch_ends = branch_starts[:n_keep], branch_ends[:n_keep]

        subsegment_ids = torch.full((int(input_ids.shape[0]),), ATTEND_ALL_SUBSEGMENT_ID, dtype=torch.int32)
        for b, (start, end) in enumerate(zip(branch_starts, branch_ends)):
            subsegment_ids[start:end] = b
            # Square-root weighting per branch (1/sqrt(N_b)), so each branch weighs what it would
            # as a separate flat sample. Label positions of branch b lie inside [start, end).
            if self.sqrt_loss_weighting:
                n_supervised = int((loss_mask[start:end] > 0).sum())
                if n_supervised > 0:
                    loss_mask[start:end] /= n_supervised**0.5
        # Optional: down-weight by sqrt(number of kept branches), as Molmo2's "root_subsegments"
        # does (olmo/preprocessing/text_preprocessor.py, tokenize_message_list). Counted after
        # trimming, like Molmo2. See __init__ for why this is off by default.
        if self.root_subsegments:
            loss_mask /= math.sqrt(n_keep)

        return EuroVLTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            visual_tensors=self._collect_visual_tensors(proc_output),
            subsegment_ids=subsegment_ids,
        )

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

        # Message-tree ids: flat samples are one attend-all subsegment. Omitted when the pack has no tree.
        subsegment_ids = None
        if any(getattr(s, "subsegment_ids", None) is not None for s in samples):
            subsegment_ids = torch.cat(
                [
                    s.subsegment_ids
                    if getattr(s, "subsegment_ids", None) is not None
                    else torch.full((int(s.input_ids.shape[0]),), ATTEND_ALL_SUBSEGMENT_ID, dtype=torch.int32)
                    for s in samples
                ],
                dim=0,
            )

        return EuroVLPackedSample(
            __key__=samples[0].__key__,
            __subflavors__=samples[0].__subflavors__,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            visual_tensors=visual_tensors,
            cu_seqlens=cu_seqlens,
            seqlens=seqlens,
            subsegment_ids=subsegment_ids,
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
        subsegment_ids = None
        if s.subsegment_ids is not None:
            subsegment_ids = torch.full((target_len,), ATTEND_ALL_SUBSEGMENT_ID, dtype=torch.int32)
            subsegment_ids[:total] = s.subsegment_ids
            subsegment_ids = subsegment_ids.unsqueeze(0)

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
            subsegment_ids=subsegment_ids,
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
            "subsegment_ids": batch.subsegment_ids,
            "visual_inputs": GenericVisualInputs(**{k: v for k, v in vt.items() if v is not None}),
        }
