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

"""Batch caption generation for the eval harness.

Loads a Megatron EuroVL checkpoint once and generates a caption/answer for
every item in an eval manifest. Mirrors the verified-working inference path
in ``sanity_check/euro_vl_test_inference.py`` (``load_megatron_model`` +
``Qwen3EuroVLProcessor`` built from its tokenizer/MoonViT components, greedy
decode with no KV cache) rather than the generic HF-family
``examples/conversion/hf_to_megatron_generate_vlm.py`` path, which assumes an
``AutoProcessor``-loadable checkpoint that EuroVL's split tokenizer/MoonViT
layout does not provide.

``--model_backend hf`` uses the exported HF ``EuroVLForConditionalGeneration``
instead, with a manual KV-cached decode loop (``hf_cached_decode``) --
~5.4x faster than the Megatron path's uncached ``greedy_decode`` (verified:
~9.85s/item vs ~53.5s/item for 200 new tokens). Deliberately does NOT use
``GenerationMixin.generate()``: that API showed unexplained run-to-run
non-determinism in testing (~11/14 runs picked a wildly different first
token -- e.g. spuriously opening a ``<think>`` block -- despite raw
``forward()`` always confidently favoring the correct token by a ~26-logit
margin, reproducible via manual replication of every kwarg ``.generate()``
passes). A manual loop that calls ``forward()`` directly each step, reusing
``past_key_values``, was 100% reliable across every test and keeps the same
KV-cache speed win.

Single GPU, no TP/PP: the EuroVL checkpoints evaluated here are ~2B params.

Run as a module (so the ``evals`` package imports resolve), inside the
apptainer container:

Example (Megatron backend):
  uv run --no-sync python -m evals.generate_captions \\
    --manifest evals/data/image_eval.jsonl \\
    --checkpoint /scratch/euro_vl_runs/qwen3_pa_pyav_vect/iter_0004000 \\
    --tokenizer_path /scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer \\
    --moonvit_path /scratch/hf_models/moonshotai-MoonViT-SO-400M \\
    --model_tag vect

Example (HF backend, faster -- requires a Megatron->HF export first via
examples/conversion/convert_checkpoints.py export):
  uv run --no-sync python -m evals.generate_captions \\
    --manifest evals/data/image_eval.jsonl \\
    --model_backend hf \\
    --hf_model_path /scratch/hf_models/vect_100_export \\
    --tokenizer_path /scratch/megatron_models/qwen3_euro_vl_2b/iter_0000000/tokenizer \\
    --moonvit_path /scratch/hf_models/moonshotai-MoonViT-SO-400M \\
    --model_tag vect
"""

import argparse
import logging

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from evals.common import RESULTS_DIR, read_jsonl, write_json_records
from megatron.bridge.models.euro_vl.euro_vl_processor import EuroVLProcessor, Qwen3EuroVLProcessor
from megatron.bridge.models.euro_vl.gemma_euro_vl_hf import (
    GemmaEuroVLForConditionalGeneration,
    GemmaEuroVLProcessor,
)
from megatron.bridge.models.euro_vl.modeling_euro_vl_hf import EuroVLForConditionalGeneration
from megatron.bridge.models.euro_vl.moonvit import MoonViTImageProcessor, MoonViTVideoProcessor
from megatron.bridge.training.model_load_save import load_megatron_model


logger = logging.getLogger(__name__)

# EuroVL backbone arms differ only in their vision-token *strings* and their HF class, so
# one flag selects both. This matters more than it looks: the vision tokens are not shared
# across arms (Qwen3 uses its built-in <|image_pad|>/<|vision_start|>; the Gemma arm reuses
# Gemma2's <unused0..4>). Pairing a checkpoint with the wrong processor does NOT raise --
# the foreign token strings just tokenize into ordinary subwords, silently producing
# fluent-but-ungrounded captions that look like a bad model rather than a bad harness.
# "qwen3" stays the default so every existing arm behaves exactly as before.
# "eurollm" (EuroVLProcessor, base EuroLLM/M-RoPE arm) is --model_backend megatron only: its
# HF class dispatches to plain LlamaForCausalLM (see modeling_euro_vl_hf.py), which does not
# apply M-RoPE position_ids, so --model_backend hf would silently run 1D RoPE on this
# checkpoint's M-RoPE-trained weights.
_VARIANTS = {
    "qwen3": (Qwen3EuroVLProcessor, EuroVLForConditionalGeneration),
    "gemma": (GemmaEuroVLProcessor, GemmaEuroVLForConditionalGeneration),
    "eurollm": (EuroVLProcessor, EuroVLForConditionalGeneration),
}


def load_processor(
    tokenizer_path: str, moonvit_path: str, num_frames: int, variant: str = "qwen3"
) -> Qwen3EuroVLProcessor:
    """Build the EuroVL processor from its tokenizer and MoonViT components.

    The Megatron checkpoint holds only the tokenizer; the MoonViT image/video
    processor config lives in a separate HF directory. ``variant`` must match the
    checkpoint's LLM backbone (see ``_VARIANTS``).

    There is no image-backend choice: ``MoonViTImageProcessor`` resizes with torchvision, the
    former ``VectorizedMoonViTImageProcessor`` having been folded into it (2026-09). Checkpoints
    trained on the older PIL backend can no longer be preprocessed exactly as they were trained.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_VARIANTS)}, got {variant!r}")
    processor_cls = _VARIANTS[variant][0]
    image_processor = MoonViTImageProcessor.from_pretrained(moonvit_path)
    video_processor = MoonViTVideoProcessor.from_pretrained(moonvit_path)
    video_processor.num_frames = num_frames
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor_cls(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None),
    )


def sample_video_frames(path: str, num_frames: int) -> list[Image.Image]:
    """Decode ``num_frames`` uniformly-sampled RGB frames from a video file (via PyAV)."""
    import av

    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))
            if len(frames) >= 5000:  # safety cap for long clips
                break
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    if num_frames == 1:
        idxs = [len(frames) // 2]
    else:
        idxs = [round(i * (len(frames) - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [frames[i] for i in idxs]


def build_multimodal_inputs(
    processor: Qwen3EuroVLProcessor,
    prompt: str,
    images: list[Image.Image] | None,
    videos: list[list[Image.Image]] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build input_ids + vision tensors via the processor for the given images/videos."""
    content: list[dict] = []
    content += [{"type": "image"} for _ in (images or [])]
    content += [{"type": "video"} for _ in (videos or [])]
    content += [{"text": prompt}]
    messages = [{"role": "user", "content": content}]
    formatted = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = processor(images=images, videos=videos, text=formatted, return_tensors="pt")
    input_ids = enc["input_ids"].cuda()

    vision: dict[str, torch.Tensor] = {}
    if "pixel_values" in enc:
        vision["pixel_values"] = enc["pixel_values"].to(torch.bfloat16).cuda()
        vision["image_grid_thw"] = enc["image_grid_thw"].cuda()
    if "pixel_values_videos" in enc:
        vision["pixel_values_videos"] = enc["pixel_values_videos"].to(torch.bfloat16).cuda()
        vision["video_grid_thw"] = enc["video_grid_thw"].cuda()
    return input_ids, vision


def greedy_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    vision: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Autoregressive greedy decode (no KV cache: the full sequence is re-fed each step)."""
    generated = input_ids.clone()
    vision = vision or {}
    for _step in range(max_new_tokens):
        with torch.no_grad():
            outputs, _ = model(input_ids=generated, **vision)
        if outputs.dim() == 3 and outputs.shape[0] == generated.shape[0]:
            next_logits = outputs[:, -1, :]
        else:
            next_logits = outputs[-1, :, :]

        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if (next_token == eos_token_id).all():
            break

    return generated


def hf_cached_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    vision: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Autoregressive greedy decode for the HF EuroVL model, WITH a KV cache.

    Prefills once (with vision), then feeds a single new token per step reusing
    ``past_key_values`` -- unlike ``greedy_decode``, which re-feeds the whole
    growing sequence (and recomputes vision) every step. Deliberately does not
    use ``model.generate()``; see the module docstring for why.
    """
    vision = vision or {}
    past_key_values = DynamicCache()
    generated = input_ids

    with torch.no_grad():
        out = model(input_ids=generated, **vision, use_cache=True, past_key_values=past_key_values)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    past_key_values = out.past_key_values

    for _ in range(max_new_tokens - 1):
        if (next_token == eos_token_id).all():
            break
        with torch.no_grad():
            out = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        past_key_values = out.past_key_values

    return generated


def _load_media(item: dict, num_frames: int) -> tuple[list[Image.Image] | None, list[list[Image.Image]] | None]:
    """Load images/videos for one manifest item, dispatching on its modality."""
    modality = item["modality"]
    if modality == "image":
        return [Image.open(item["media"][0]).convert("RGB")], None
    elif modality == "multi_image":
        return [Image.open(p).convert("RGB") for p in item["media"]], None
    elif modality == "video":
        return None, [sample_video_frames(item["media"][0], num_frames)]
    else:
        raise ValueError(f"Unknown modality: {modality!r}")


def generate_one(
    model,
    processor: Qwen3EuroVLProcessor,
    item: dict,
    max_new_tokens: int,
    num_frames: int,
    decode_fn=greedy_decode,
) -> str:
    """Generate a response for one manifest item, dispatching on its modality."""
    images, videos = _load_media(item, num_frames)
    input_ids, vision = build_multimodal_inputs(processor, item["prompt"], images, videos)
    generated = decode_fn(model, input_ids, max_new_tokens, processor.tokenizer.eos_token_id, vision)
    return processor.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def _pad_and_concat_batch(
    pad_token_id: int,
    batch_input_ids: list[torch.Tensor],
    batch_vision: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Left-pad a list of ``[1, S_i]`` input_ids into one ``[B, max_S]`` batch (+ attention_mask),
    and concatenate each item's vision tensors in the same row order.

    Left-padding (not right) keeps every row's next-token position at the same column index,
    so the decode loop can always read/append at column -1 uniformly across the batch.
    Concatenating vision tensors in row order works because ``masked_scatter`` (in ``forward``)
    fills ``image_token_id``/``video_token_id`` positions in row-major order -- row 0's images
    first, then row 1's -- matching this concatenation order exactly, regardless of how many
    images/frames each row has or how much left-padding it carries (padding tokens are never
    part of the mask, so they don't shift this correspondence).
    """
    max_len = max(ids.shape[1] for ids in batch_input_ids)
    batch_size = len(batch_input_ids)
    device = batch_input_ids[0].device
    dtype = batch_input_ids[0].dtype

    padded = torch.full((batch_size, max_len), pad_token_id, dtype=dtype, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(batch_input_ids):
        length = ids.shape[1]
        padded[i, max_len - length :] = ids[0]
        attention_mask[i, max_len - length :] = 1

    concatenated: dict[str, torch.Tensor] = {}
    for pixel_key, grid_key in [("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")]:
        pixels = [v[pixel_key] for v in batch_vision if pixel_key in v]
        grids = [v[grid_key] for v in batch_vision if grid_key in v]
        if pixels:
            concatenated[pixel_key] = torch.cat(pixels, dim=0)
            concatenated[grid_key] = torch.cat(grids, dim=0)

    return padded, attention_mask, concatenated


def hf_cached_decode_batch(
    model: torch.nn.Module,
    pad_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    batch_input_ids: list[torch.Tensor],
    batch_vision: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Batched version of ``hf_cached_decode``: left-pads and processes multiple items in one
    forward pass per step instead of one item at a time, for higher GPU utilization.

    Every row keeps generating every step (even ones that already hit EOS) -- simpler than
    slicing finished rows out of the batch, at the cost of a little wasted compute on rows
    that finish early. The loop still exits as soon as ALL rows are finished.
    """
    prompt_ids, attention_mask, vision = _pad_and_concat_batch(pad_token_id, batch_input_ids, batch_vision)
    prompt_len = prompt_ids.shape[1]
    batch_size = prompt_ids.shape[0]
    past_key_values = DynamicCache()
    generated = prompt_ids

    with torch.no_grad():
        out = model(
            input_ids=generated,
            attention_mask=attention_mask,
            **vision,
            use_cache=True,
            past_key_values=past_key_values,
        )
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1
    )
    past_key_values = out.past_key_values
    finished = next_token.squeeze(-1) == eos_token_id

    for _ in range(max_new_tokens - 1):
        if finished.all():
            break
        with torch.no_grad():
            out = model(
                input_ids=next_token, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values
            )
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        past_key_values = out.past_key_values
        finished = finished | (next_token.squeeze(-1) == eos_token_id)

    return generated[:, prompt_len:]


def _decode_batch_and_split(
    model,
    pad_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    batch_input_ids: list[torch.Tensor],
    batch_vision: list[dict[str, torch.Tensor]],
    processor: Qwen3EuroVLProcessor,
    max_batch_tokens: int,
) -> list[str]:
    """Decode a batch, recursively halving it first if left-padding it as one batch would
    blow up memory.

    Left-padding pads every row to the batch's longest prompt -- if one item's prompt is
    far longer than its batch-mates (e.g. a multi_image item with many more images), the
    whole batch pays for that outlier's length, and attention cost scales with padded
    length, not real content length. Splitting instead of padding through this keeps each
    sub-batch's total padded token budget (batch_size * max_len) under max_batch_tokens.
    """
    max_len = max(ids.shape[1] for ids in batch_input_ids)
    if len(batch_input_ids) > 1 and max_len * len(batch_input_ids) > max_batch_tokens:
        mid = len(batch_input_ids) // 2
        left = _decode_batch_and_split(
            model,
            pad_token_id,
            eos_token_id,
            max_new_tokens,
            batch_input_ids[:mid],
            batch_vision[:mid],
            processor,
            max_batch_tokens,
        )
        right = _decode_batch_and_split(
            model,
            pad_token_id,
            eos_token_id,
            max_new_tokens,
            batch_input_ids[mid:],
            batch_vision[mid:],
            processor,
            max_batch_tokens,
        )
        return left + right

    generated = hf_cached_decode_batch(
        model, pad_token_id, eos_token_id, max_new_tokens, batch_input_ids, batch_vision
    )
    captions = []
    for row in generated:
        eos_positions = (row == eos_token_id).nonzero(as_tuple=True)[0]
        row_trimmed = row[: eos_positions[0]] if eos_positions.numel() > 0 else row
        captions.append(processor.decode(row_trimmed, skip_special_tokens=True).strip())
    return captions


def generate_batch(
    model,
    processor: Qwen3EuroVLProcessor,
    items: list[dict],
    max_new_tokens: int,
    num_frames: int,
    max_batch_tokens: int = 32768,
) -> list[str]:
    """Generate captions for a batch of manifest items in one set of forward passes.

    ``max_batch_tokens`` was originally set to 16384 as a workaround for an OOM that turned
    out to be caused by MoonViT's vision-tower attention doing dense O(total_patches^2)
    attention over every image in the batch (see attn_implementation="flash_attention_2" in
    this file's model loader) -- not by the text side's padded token budget. That threshold
    was silently shrinking every requested batch_size down to ~8 regardless of what was
    asked for, well below what's actually needed now that the real OOM cause is fixed.
    Raised generously; still bounds truly extreme outliers rather than removing the guard
    entirely.
    """
    batch_input_ids = []
    batch_vision = []
    for item in items:
        images, videos = _load_media(item, num_frames)
        input_ids, vision = build_multimodal_inputs(processor, item["prompt"], images, videos)
        batch_input_ids.append(input_ids)
        batch_vision.append(vision)

    pad_token_id = processor.tokenizer.pad_token_id
    eos_token_id = processor.tokenizer.eos_token_id
    return _decode_batch_and_split(
        model, pad_token_id, eos_token_id, max_new_tokens, batch_input_ids, batch_vision, processor, max_batch_tokens
    )


def main(args) -> None:
    """Generate captions for every item in the eval manifest with a single loaded checkpoint."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    processor = load_processor(
        args.tokenizer_path, args.moonvit_path, args.num_frames, args.model_variant
    )

    if args.model_backend == "megatron":
        logger.info("Loading Megatron model from %s", args.checkpoint)
        models = load_megatron_model(args.checkpoint)
        model = models[0].cuda().eval()
        decode_fn = greedy_decode
    else:
        logger.info("Loading HF model from %s", args.hf_model_path)
        # Force flash_attention_2 (not the transformers-default "sdpa") for the vision tower.
        # MoonViT's sdpa/eager attention paths (modeling_moonvit.py) build a DENSE
        # [1, total_patches, total_patches] mask over every image packed into the batch and
        # call scaled_dot_product_attention over the whole thing -- masking out cross-image
        # attention after the fact instead of avoiding computing it, so memory is
        # O(total_patches^2) across the WHOLE batch's images combined, not O(total_patches).
        # Only flash_attention_2 (flash_attn_varlen_func + cu_seqlens) is genuinely
        # block-diagonal/linear. This was invisible at batch_size=1 (one item's own patch
        # count squared is small) and only exploded once --batch_size > 1 concatenated
        # multiple items' images into one packed vision sequence.
        model_cls = _VARIANTS[args.model_variant][1]
        model = (
            model_cls.from_pretrained(
                args.hf_model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            .cuda()
            .eval()
        )
        decode_fn = hf_cached_decode

    manifest = read_jsonl(args.manifest)
    logger.info("Loaded %d items from %s", len(manifest), args.manifest)

    records = []
    if args.model_backend == "hf" and args.batch_size > 1:
        for start in range(0, len(manifest), args.batch_size):
            chunk = manifest[start : start + args.batch_size]
            captions = generate_batch(model, processor, chunk, args.max_new_tokens, args.num_frames)
            for item, caption in zip(chunk, captions):
                logger.info("[%s] %s", item["id"], caption[:120])
                records.append(
                    {"id": item["id"], "modality": item["modality"], "prompt": item["prompt"], "caption": caption}
                )
            # Chunk shapes vary a lot (item image/frame counts differ), so the CUDA caching
            # allocator can't always reuse the previous chunk's blocks for the next chunk's
            # different-shaped tensors -- left unchecked this fragments and grows until OOM
            # over a long manifest. Release cached-but-unused blocks after every chunk.
            torch.cuda.empty_cache()
    else:
        for item in manifest:
            caption = generate_one(model, processor, item, args.max_new_tokens, args.num_frames, decode_fn=decode_fn)
            logger.info("[%s] %s", item["id"], caption[:120])
            records.append(
                {"id": item["id"], "modality": item["modality"], "prompt": item["prompt"], "caption": caption}
            )

    modality = manifest[0]["modality"] if manifest else "unknown"
    out_path = RESULTS_DIR / args.model_tag / f"{modality}_captions.jsonl"
    write_json_records(records, out_path)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Batch EuroVL caption generation for the eval harness.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to an eval manifest JSONL.")
    parser.add_argument(
        "--model_backend",
        type=str,
        default="megatron",
        choices=["megatron", "hf"],
        help="'megatron' loads --checkpoint and decodes with no KV cache (greedy_decode). 'hf' loads"
        " --hf_model_path (a Megatron->HF export) and decodes with a KV cache (hf_cached_decode) --"
        " ~5.4x faster, see the module docstring for why model.generate() is deliberately not used.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the Megatron checkpoint directory (--model_backend megatron).",
    )
    parser.add_argument(
        "--hf_model_path", type=str, default=None, help="Path to the HF export directory (--model_backend hf)."
    )
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the checkpoint's tokenizer dir.")
    parser.add_argument("--moonvit_path", type=str, required=True, help="Path to the MoonViT HF config directory.")
    parser.add_argument("--model_tag", type=str, required=True, help="Label for this checkpoint, e.g. 'v1'.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    parser.add_argument("--num_frames", type=int, default=4, help="Frames sampled per video.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Items per batched forward pass (--model_backend hf only; ignored for megatron). >1 uses"
        " generate_batch/hf_cached_decode_batch instead of the per-item loop.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="qwen3",
        choices=sorted(_VARIANTS),
        help="EuroVL backbone arm this checkpoint belongs to. Selects the vision-token strings"
        " and HF class. MUST match the checkpoint: a mismatch fails silently (see _VARIANTS).",
    )
    args = parser.parse_args()
    if args.model_backend == "megatron" and args.checkpoint is None:
        parser.error("--checkpoint is required when --model_backend megatron")
    if args.model_backend == "hf" and args.hf_model_path is None:
        parser.error("--hf_model_path is required when --model_backend hf")
    main(args)
