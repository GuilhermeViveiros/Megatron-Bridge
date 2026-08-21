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
from megatron.bridge.models.euro_vl.euro_vl_processor import Qwen3EuroVLProcessor
from megatron.bridge.models.euro_vl.modeling_euro_vl_hf import EuroVLForConditionalGeneration
from megatron.bridge.models.euro_vl.moonvit import MoonViTImageProcessor, MoonViTVideoProcessor
from megatron.bridge.models.euro_vl.moonvit.image_processing_moonvit import VectorizedMoonViTImageProcessor
from megatron.bridge.training.model_load_save import load_megatron_model


logger = logging.getLogger(__name__)


def load_processor(tokenizer_path: str, moonvit_path: str, num_frames: int, backend: str) -> Qwen3EuroVLProcessor:
    """Build the Qwen3EuroVL processor from its tokenizer and MoonViT components.

    The Megatron checkpoint holds only the tokenizer; the MoonViT image/video
    processor config lives in a separate HF directory. ``backend`` must match
    the image-preprocessing backend the checkpoint was trained with.
    """
    if backend not in ("pil", "vectorized"):
        raise ValueError(f"backend must be 'pil' or 'vectorized', got {backend!r}")
    image_cls = MoonViTImageProcessor if backend == "pil" else VectorizedMoonViTImageProcessor
    image_processor = image_cls.from_pretrained(moonvit_path)
    video_processor = MoonViTVideoProcessor.from_pretrained(moonvit_path)
    video_processor.num_frames = num_frames
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return Qwen3EuroVLProcessor(
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


def generate_one(
    model,
    processor: Qwen3EuroVLProcessor,
    item: dict,
    max_new_tokens: int,
    num_frames: int,
    decode_fn=greedy_decode,
) -> str:
    """Generate a response for one manifest item, dispatching on its modality."""
    prompt = item["prompt"]
    modality = item["modality"]

    images = None
    videos = None
    if modality == "image":
        images = [Image.open(item["media"][0]).convert("RGB")]
    elif modality == "multi_image":
        images = [Image.open(p).convert("RGB") for p in item["media"]]
    elif modality == "video":
        videos = [sample_video_frames(item["media"][0], num_frames)]
    else:
        raise ValueError(f"Unknown modality: {modality!r}")

    input_ids, vision = build_multimodal_inputs(processor, prompt, images, videos)
    generated = decode_fn(model, input_ids, max_new_tokens, processor.tokenizer.eos_token_id, vision)
    return processor.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def main(args) -> None:
    """Generate captions for every item in the eval manifest with a single loaded checkpoint."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    processor = load_processor(args.tokenizer_path, args.moonvit_path, args.num_frames, args.backend)

    if args.model_backend == "megatron":
        logger.info("Loading Megatron model from %s", args.checkpoint)
        models = load_megatron_model(args.checkpoint)
        model = models[0].cuda().eval()
        decode_fn = greedy_decode
    else:
        logger.info("Loading HF model from %s", args.hf_model_path)
        model = (
            EuroVLForConditionalGeneration.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
        decode_fn = hf_cached_decode

    manifest = read_jsonl(args.manifest)
    logger.info("Loaded %d items from %s", len(manifest), args.manifest)

    records = []
    for item in manifest:
        caption = generate_one(model, processor, item, args.max_new_tokens, args.num_frames, decode_fn=decode_fn)
        logger.info("[%s] %s", item["id"], caption[:120])
        records.append({"id": item["id"], "modality": item["modality"], "prompt": item["prompt"], "caption": caption})

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
        "--backend",
        type=str,
        default="vectorized",
        choices=["pil", "vectorized"],
        help="Image preprocessing backend — must match the backend the checkpoint was trained with.",
    )
    args = parser.parse_args()
    if args.model_backend == "megatron" and args.checkpoint is None:
        parser.error("--checkpoint is required when --model_backend megatron")
    if args.model_backend == "hf" and args.hf_model_path is None:
        parser.error("--hf_model_path is required when --model_backend hf")
    main(args)
